"""Filesystem naming conventions for Autolamella and legacy datasets.

The single source of truth for what directories and image files are called. An
image directory is ``<family>[_<variant>]`` unless a project's
:class:`FamilyCorrections` say otherwise; a family's conditions share
``<family>_labels`` (:func:`resolve_labels_directory`). Loose images carry their
series and beam in the filename (:func:`classify_loose_image`). Nothing here reads
workflow state, but ``deselected/`` and ``corrected/`` stay reserved. Only the
image-set builders and the helpers at the end touch the filesystem, through
:func:`directory_cache`'s listings.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

logger = logging.getLogger(__name__)

#: The imaging modalities this package indexes, in display order.
MODALITIES: Final[tuple[str, ...]] = ("SEM", "FIB")

#: The subdirectory names a pre-declaration dataset kept its state in. Nothing
#: in the package reads state from them -- ``scripts/migrate_state.py`` does
#: that, once -- but they stay reserved so an unmigrated dataset reads as
#: *missing* those images rather than as holding selected ones.
STATE_FOLDER_NAMES: Final[frozenset[str]] = frozenset({"deselected", "corrected"})

#: Where a project keeps the labels an operation replaced. **Reserved, and
#: skipped by every traversal**: indexed, a superseded version would be read as
#: a current label and reach a training CSV, which is a corrupted training set
#: rather than merely a slow scan.
ARCHIVE_DIRECTORY: Final[str] = "AM_archive"

LABELS_SUFFIX: Final[str] = "_labels"
PREDICTIONS_SUFFIX: Final[str] = "_predictions"
PLOTS_DIRECTORY: Final[str] = "plots"

#: Subdirectories of a predictions directory holding different prediction sets.
#: All are indexed; not all may become a label. See
#: :func:`is_convertible_prediction` and :func:`prediction_origin`.
PREDICTION_KINDS: Final[tuple[str, ...]] = ("am_seg", "raw", "clean")

#: The prediction subdirectory this package writes, and the only one it may
#: overwrite. ``raw`` and ``clean`` are the segmentation pipeline's own output
#: and are never written to here: an operation may replace what it produced,
#: not what something else did.
OWN_PREDICTION_KIND: Final[str] = "am_seg"

#: Records which model produced each prediction in an :data:`OWN_PREDICTION_KIND`
#: directory. A predictions directory with no account of what filled it is a
#: problem that worsens with time.
PREDICTION_PROVENANCE_FILENAME: Final[str] = "am_seg_predictions.yaml"

#: The prediction subdirectory whose values do not survive post-processing, and
#: which therefore may never be converted into a label.
POST_PROCESSED_KIND: Final[str] = "clean"

#: Pre-convention label directory names, lowercased, to the modality they hold.
#: Read indefinitely: renaming one is ``scripts/rename_legacy_label_dirs.py``'s
#: to do, never the package's, which renames no directory of its own accord.
LEGACY_LABEL_DIRECTORIES: Final[dict[str, str]] = {
    "labels": "SEM",
    "labels_fib": "FIB",
}

#: Pre-convention prediction directory names, lowercased.
LEGACY_PREDICTION_DIRECTORIES: Final[dict[str, str]] = {
    "predictions": "SEM",
    "predictions_fib": "FIB",
}

#: Variant names that would collide with another directory role. An image
#: directory may not use these, because ``sem_labels`` must unambiguously mean
#: "labels for sem" rather than "images of variant 'labels'".
RESERVED_VARIANTS: Final[frozenset[str]] = frozenset(
    {"labels", "predictions", "plots", "deselected", "corrected"}
)

#: What separates a family from its variant in an image directory's name.
VARIANT_SEPARATOR: Final[str] = "_"

#: The modality an image directory's name starts with. What follows it must not
#: be a letter, so ``fibsem_x`` and ``semi-thin`` are not read as image
#: directories (``test_a_modality_followed_by_a_letter_is_not_an_image_directory``).
_MODALITY_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^(?P<modality>sem|fib)(?=$|[-._0-9])", re.IGNORECASE
)

#: A family name a correction may give: the modality, then only letters,
#: digits, ``-`` and ``.``, every one in the BioImage Archive's allowed set.
_FAMILY_NAME: Final[re.Pattern[str]] = re.compile(
    r"^(?P<modality>sem|fib)(?=$|[-.0-9])[A-Za-z0-9.-]*$", re.IGNORECASE
)

#: File extensions treated as images or labels.
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".tif", ".tiff"})

#: What a directory inside a stage is for.
DirectoryRole = Literal[
    "image", "labels", "predictions", "plots", "state", "archive", "unknown"
]

#: What an image was taken for. Orthogonal to modality and to the
#: imaging-condition variant: one directory holds several series, so a series is
#: a property of an image set rather than of the directory it sits in.
#:
#: ``adaptive`` images usually fill a ``<modality>`` directory of their own,
#: recognised by :func:`classify_image_directory`; the script-based prototype
#: wrote them loose in the lamella directory instead. Reference and alignment
#: images are always loose, sharing a directory with whatever else Autolamella
#: wrote there. :func:`classify_loose_image` recognises any of the three.
#:
#: ``other`` is every remaining image file, and is not a claim about what it is
#: for: it is where an image goes when no naming rule reaches it, so that a
#: picture nobody anticipated can be opened and looked at instead of merely
#: counted. It carries no labels and no predictions, and the write paths refuse
#: it.
Series = Literal["adaptive", "reference", "alignment", "other"]

#: The series, in display order. ``other`` last: it is the residue, and a
#: reader scanning a row wants the curated series first.
SERIES: Final[tuple[Series, ...]] = ("adaptive", "reference", "alignment", "other")

#: The series whose images can sit loose in a directory rather than filling
#: one, and so need a name invented for them before their labels have somewhere
#: to go. All three curated ones: an Adaptive Polishing image usually has a
#: directory to take its name from, but not where the prototype wrote it beside
#: the references. ``other`` is excluded because it has no labels to name a
#: directory for -- :func:`other_image_set_name` names those sets instead.
LOOSE_SERIES: Final[tuple[Series, ...]] = ("adaptive", "reference", "alignment")

#: The series an operation may be pointed at, and so the ``--series`` vocabulary
#: of every command that writes, measures or exports label data. Everything but
#: ``other``, whose images have no label directory and no predictions directory
#: for such a command to act on. Derived rather than written out, so that adding
#: a series has to decide which side of this it falls on.
LABELLED_SERIES: Final[tuple[Series, ...]] = tuple(
    series for series in SERIES if series != "other"
)

#: The series whose images are expected to be labelled, so that an unlabelled
#: one is worth reporting. Everything outside this is indexed and countable but
#: nobody is told it has no labels yet -- there are 5,237 such images in the
#: reference collection, and saying so once per set drowns the findings that
#: need acting on. A set that *does* have a label directory is reported on
#: normally whatever its series, so starting to label one needs no change here.
LABELS_EXPECTED: Final[frozenset[Series]] = frozenset({"adaptive"})

#: What to call each series where there is room to say it properly.
SERIES_LABELS: Final[dict[str, str]] = {
    "adaptive": "Adaptive Polishing",
    "reference": "reference",
    "alignment": "alignment",
    "other": "other",
}

#: Filename suffix to the modality it names. Autolamella writes one on nearly
#: every image it takes outside an Adaptive Polishing stage. A name carrying
#: neither leaves the modality unstated, which is the header's to answer -- see
#: :func:`beam_modality`.
BEAM_SUFFIXES: Final[dict[str, str]] = {"_eb": "SEM", "_ib": "FIB"}

#: The directory Autolamella collects its alignment runs in, one subdirectory
#: per run. Present below a lamella and below each milling task.
ALIGNMENT_DIRECTORY: Final[str] = "Alignment"

#: Plot files that summarise a whole stage rather than one acquisition, tried
#: in order. Unlike the per-acquisition plots in ``plots/``, these sit at the
#: stage root and there is one of them: an alignment run's ``figure.png``
#: summarises the several images it took to converge, and the Adaptive
#: Polishing plugin's ``AP_summary_plots.png`` summarises a whole milling run.
#:
#: ``centring.png`` sits beside them and is deliberately absent: it records
#: where the plugin placed its field of view, which is a diagnostic of the run.
STAGE_SUMMARY_PLOTS: Final[tuple[str, ...]] = ("figure.png", "AP_summary_plots.png")


@dataclass(frozen=True)
class ImageDirectory:
    """The modality, family and variant encoded in an image directory's name.

    Attributes:
        modality: ``"SEM"`` or ``"FIB"``.
        variant: What follows the first ``_``, or ``None`` for the family's
            base condition. Preserved with the case used on disk.
        family: What precedes the first ``_``, as spelled on disk. Two
            spellings differing only in case are one family.
    """

    modality: str
    variant: str | None
    family: str

    @property
    def display_variant(self) -> str:
        """Return the variant for display, naming the unsuffixed case."""
        return self.variant if self.variant is not None else "base"


@dataclass(frozen=True)
class FamilyCorrections:
    """A project's corrections to the families its condition names give.

    Plain data, read from ``AM_families.yaml`` by :mod:`.families` and passed
    into resolution, so nothing here reads a project file. Names compare
    case-insensitively, as a stage's directories do. Constructing one validates
    it, so an instance never holds a correction resolution could misread.

    Attributes:
        families: Condition name to the family it belongs to instead of the one
            its name gives.
        primary: Family name to the condition standing for it.

    Raises:
        ValueError: Listing every entry that is not valid.
    """

    families: Mapping[str, str] = field(default_factory=dict)
    primary: Mapping[str, str] = field(default_factory=dict)
    _families: dict[str, str] = field(init=False, repr=False, compare=False)
    _primary: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        problems = correction_problems(self.families, self.primary)
        if problems:
            raise ValueError("; ".join(problems))
        object.__setattr__(
            self, "_families", {key.lower(): value for key, value in self.families.items()}
        )
        object.__setattr__(
            self, "_primary", {key.lower(): value for key, value in self.primary.items()}
        )

    def family_of(self, condition: str) -> str | None:
        """Return the family *condition* is corrected into, or None if it is not."""
        return self._families.get(condition.lower())

    def primary_of(self, family: str) -> str | None:
        """Return the condition named as *family*'s primary, or None."""
        return self._primary.get(family.lower())


def is_family_name(name: str) -> bool:
    """Return whether *name* may be given to a family by a correction."""
    return _FAMILY_NAME.match(name) is not None


def correction_problems(
    families: Mapping[str, str], primary: Mapping[str, str]
) -> list[str]:
    """Return what is wrong with a set of family corrections, one line each.

    A condition may move only to a family of its own modality, and a primary
    must belong to the family it stands for once the corrections apply.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    lowered = {key.lower(): value for key, value in families.items() if isinstance(key, str)}
    for condition, family in families.items():
        parsed = classify_image_directory(condition) if isinstance(condition, str) else None
        if parsed is None:
            problems.append(f"{condition!r} is not the name of an image directory")
            continue
        if condition.lower() in seen:
            problems.append(f"{condition} is listed twice, as {seen[condition.lower()]} too")
            continue
        seen[condition.lower()] = condition
        if not isinstance(family, str) or not is_family_name(family):
            problems.append(
                f"{condition}: {family!r} is not a family name -- it starts with sem "
                f"or fib, and holds only letters, digits, '-' and '.'"
            )
        elif family[:3].upper() != parsed.modality:
            problems.append(
                f"{condition} is a {parsed.modality} condition and cannot join the "
                f"{family[:3].upper()} family {family}"
            )

    seen = {}
    for family, condition in primary.items():
        if not isinstance(family, str) or not is_family_name(family):
            problems.append(f"primary: {family!r} is not a family name")
            continue
        if family.lower() in seen:
            problems.append(f"primary: {family} is listed twice, as {seen[family.lower()]} too")
            continue
        seen[family.lower()] = family
        parsed = classify_image_directory(condition) if isinstance(condition, str) else None
        if parsed is None:
            problems.append(
                f"primary: {family}: {condition!r} is not the name of an image directory"
            )
            continue
        belongs_to = lowered.get(condition.lower(), parsed.family)
        if not isinstance(belongs_to, str) or belongs_to.lower() != family.lower():
            problems.append(
                f"primary: {condition} belongs to family {belongs_to}, not {family}"
            )
    return problems


#: No corrections: every condition in the family its name gives.
NO_CORRECTIONS: Final[FamilyCorrections] = FamilyCorrections()


@dataclass(frozen=True)
class FamilyEntries:
    """One lamella's section of ``AM_families.yaml``, as written.

    Not validated alone: an entry may mean something only over its project's,
    as a primary named for a family a project-wide entry moves a condition into.

    Attributes:
        families: Condition name to family. Naming the family the condition's
            name gives is an entry, undoing a project-wide move here.
        primary: Family name to the condition standing for it.
    """

    families: Mapping[str, str] = field(default_factory=dict)
    primary: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectFamilies:
    """A project's family corrections: its own, and each lamella's over them.

    Plain data, read by :mod:`.families`; resolution takes only what :meth:`at`
    returns. **A lamella's entries replace the project's entry by entry**, in
    every stage beneath it. Constructing one validates each lamella's entries
    combined with the project's (``test_a_lamella_entry_replaces_the_projects_there_only``).

    Attributes:
        project: The project-wide corrections.
        lamellae: A lamella's path relative to the project root, POSIX, to its
            entries.

    Raises:
        ValueError: Naming the lamella whose combined entries are not valid, or
            a key that is not a path below the project root.
    """

    project: FamilyCorrections = field(default_factory=FamilyCorrections)
    lamellae: Mapping[str, FamilyEntries] = field(default_factory=dict)
    _combined: dict[str, FamilyCorrections] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        combined: dict[str, FamilyCorrections] = {}
        for key, entries in self.lamellae.items():
            parts = key.split("/")
            if not key or key.startswith("/") or any(part in ("", ".", "..") for part in parts):
                raise ValueError(f"lamellae: {key!r} is not a path below the project root")
            try:
                combined[key] = FamilyCorrections(
                    families=_over(self.project.families, entries.families),
                    primary=_over(self.project.primary, entries.primary),
                )
            except ValueError as error:
                raise ValueError(f"lamellae: {key}: {error}") from error
        object.__setattr__(self, "_combined", combined)

    @property
    def has_lamellae(self) -> bool:
        """Return whether any lamella has entries of its own."""
        return any(entries.families or entries.primary for entries in self.lamellae.values())

    def at(self, relative: str) -> FamilyCorrections:
        """Return the corrections in force at *relative*, a POSIX path below the project root.

        The lamella's own over the project's where *relative* lies in a lamella
        with a section, the project's alone anywhere else.
        """
        within = [
            key for key in self._combined
            if relative == key or relative.startswith(f"{key}/")
        ]
        return self._combined[max(within, key=len)] if within else self.project


def _over(below: Mapping[str, str], above: Mapping[str, str]) -> dict[str, str]:
    """Return *below* with each of *above*'s entries in place of its own, keys compared whatever their case."""
    replaced = {key.lower() for key in above}
    return {
        **{key: value for key, value in below.items() if key.lower() not in replaced},
        **above,
    }


#: No corrections anywhere in a project.
NO_PROJECT_FAMILIES: Final[ProjectFamilies] = ProjectFamilies(project=NO_CORRECTIONS)


@dataclass(frozen=True)
class ImageSet:
    """Images of one series, modality and imaging condition, in one directory.

    Usually a whole image directory: an Adaptive Polishing set fills ``sem/``
    or ``sem_8_fi_1/`` and takes its name from it. A loose set instead holds
    named files picked out of a directory it shares -- a lamella directory
    holds reference and alignment images at once -- so *image_names* says which
    of them are this set's.

    Attributes:
        directory: The directory the images sit in.
        modality: ``"SEM"`` or ``"FIB"``, or ``None`` for an ``other`` set
            whose beam neither its filenames nor their headers state. Known for
            every curated series: an image of one whose beam nothing states is
            reported rather than put in a set, since nothing downstream could
            say which beam it is.
        variant: The imaging-condition suffix, or ``None`` for the base
            directory. Always ``None`` for a loose set, which has no image
            directory to carry one.
        family: The family whose label the set shares, as its directory spells
            it, or ``None`` for a loose set.
        series: What the images were taken for.
        image_names: The filenames belonging to this set, or ``None`` where the
            set is the whole directory. Read it through
            :func:`image_files_of` rather than testing it.
        labels_directory: Where this set's labels live, whether or not that
            directory exists yet. ``None`` for an ``other`` set, which has no
            labels and no place one could be written: a caller reaching for a
            label there has lost track of what it is holding.
        labels_exist: Whether *labels_directory* is present on disk.
        labels_are_shared: Whether another image set in the same stage resolves
            to the same label directory.
        labels_are_legacy: Whether the label directory was matched by a
            pre-convention name, making it a migration candidate.
    """

    directory: Path
    modality: str | None
    variant: str | None
    labels_directory: Path | None
    labels_exist: bool
    labels_are_shared: bool
    labels_are_legacy: bool
    series: Series = "adaptive"
    image_names: tuple[str, ...] | None = None
    family: str | None = None

    @property
    def is_loose(self) -> bool:
        """Return whether this set is a subset of a directory it shares."""
        return self.image_names is not None

    @property
    def name(self) -> str:
        """Return what identifies this set within its directory.

        The image directory's name, or the invented one a loose set's labels
        are named after -- which is the same string
        :func:`labels_directory_name` was given, either way. An ``other`` set
        has no labels, so :func:`other_image_set_name` names it instead.
        """
        if self.series == "other":
            return other_image_set_name(self.modality)
        assert self.modality is not None, "only an other set has no modality"
        if self.is_loose:
            return loose_image_set_name(self.series, self.modality)
        return self.directory.name

    @property
    def display_name(self) -> str:
        """Return a short label such as ``'SEM'``, ``'SEM [8_fi_1]'``,
        ``'sem-5kV [fi4]'`` or ``'SEM (reference)'``."""
        if self.modality is None:
            return f"({SERIES_LABELS[self.series]})"
        if self.is_loose:
            return f"{self.modality} ({SERIES_LABELS[self.series]})"
        family = self.family or self.modality
        prefix = family if family.lower() != self.modality.lower() else self.modality
        if self.variant is None:
            return prefix
        return f"{prefix} [{self.variant}]"


# ---------------------------------------------------------------------------
# Directory name classification
# ---------------------------------------------------------------------------

def is_labels_directory(name: str) -> bool:
    """Return whether *name* is a label directory, current or legacy."""
    lowered = name.lower()
    return lowered.endswith(LABELS_SUFFIX) or lowered in LEGACY_LABEL_DIRECTORIES


def is_predictions_directory(name: str) -> bool:
    """Return whether *name* is a predictions directory, current or legacy."""
    lowered = name.lower()
    return lowered.endswith(PREDICTIONS_SUFFIX) or lowered in LEGACY_PREDICTION_DIRECTORIES


#: The filename forms carrying an acquisition number, tried in order; the
#: number is the group named ``number``. In ``Image_1_013`` the middle digit is
#: the beam -- 1 electron, 2 ion -- which the image directory and the header
#: both state, so only the trailing number is read. That form is anchored and
#: takes a single beam digit, so a timestamp such as ``Image_2023_0811`` does
#: not read as an acquisition number.
_IMAGE_NUMBER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"AP_img_(?P<number>\d+)", re.IGNORECASE),
    re.compile(r"img_(?P<number>\d+)", re.IGNORECASE),
    re.compile(r"^image_\d_(?P<number>\d+)", re.IGNORECASE),
)


def extract_image_number(image_name: str) -> int | None:
    """Return the acquisition number an Autolamella filename carries.

    Recognises ``AP_img_013``, ``img_013`` and ``Image_1_013``. Here with the
    rest of the naming conventions rather than beside the filters that first
    needed it: it is what matches an SEM image to the FIB image of the same
    acquisition, which
    :func:`~am_segmentation_manager.frames.group_by_acquisition` relies on, and
    that has to sit below the operations layer.

    Returns:
        The number, or None where the name carries none.
    """
    for pattern in _IMAGE_NUMBER_PATTERNS:
        match = pattern.search(image_name)
        if match is not None:
            return int(match.group("number"))
    return None


#: The timestamp Autolamella puts in a stage folder name, in both the forms it
#: writes: 24-hour (``..._2026-08-07-15-30-00``) and the 12-hour one the
#: adaptive milling plugin uses (``adaptive_polish_2026-07-28-02-43-32PM``).
#: Here with the other naming conventions because two things read it and must
#: read it the same way -- :class:`~am_segmentation_manager.scanner.Scanner`,
#: to order stages by when they ran, and :func:`stage_base_name`.
STAGE_TIMESTAMP: Final[re.Pattern[str]] = re.compile(
    r"(\d{4})[-_]?([01]\d)[-_]?([0-3]\d)[T _-]?([0-2]\d)[-_:]?([0-5]\d)[-_:]?([0-5]\d)"
    r"\s*([AaPp][Mm])?"
)

#: What a stage whose name is nothing but a timestamp is grouped under. Rare,
#: and a row named after one run of one lamella would be a row of one.
UNNAMED_STAGE: Final[str] = "(timestamped)"


def stage_base_name(stage_name: str) -> str:
    """Return a stage folder's name with the timestamp taken out of it.

    What is left is the part that repeats across lamellae and projects --
    ``adaptive_polish``, or the name of the milling task a directory of
    reference images belongs to -- so that counting or filtering by it gives a
    handful of rows rather than one per folder on disk.

    Returns:
        The name without its timestamp and any separators left stranded
        either side of it, or :data:`UNNAMED_STAGE` where nothing else remains.
    """
    stripped = re.sub(r"[-_. ]{2,}", "_", STAGE_TIMESTAMP.sub(" ", stage_name))
    return stripped.strip(" -_.") or UNNAMED_STAGE


def classify_image_directory(name: str) -> ImageDirectory | None:
    """Return the modality, family and variant *name* encodes, or None if it is
    not an image directory.

    ``SEM-5kV_fi1`` is family ``SEM-5kV``, variant ``fi1``; ``SEM_8_fi_1`` is
    family ``SEM``, variant ``8_fi_1``, which is what every name from before
    families meant. Label, prediction, plot and state directories are rejected
    first, so that ``sem_labels`` and ``sem_fi8_labels`` are never mistaken for
    images of a variant called ``labels`` or ``fi8_labels``.
    """
    lowered = name.lower()
    if (
        is_labels_directory(name)
        or is_predictions_directory(name)
        or lowered == PLOTS_DIRECTORY
        or lowered in STATE_FOLDER_NAMES
    ):
        return None

    match = _MODALITY_PREFIX.match(name)
    if match is None:
        return None
    family, separator, variant = name.partition(VARIANT_SEPARATOR)
    if separator and (not variant or variant.lower() in RESERVED_VARIANTS):
        logger.debug("Ignoring image directory with reserved variant: %s", name)
        return None
    return ImageDirectory(
        modality=match["modality"].upper(), variant=variant or None, family=family
    )


def condition_of(name: str, corrections: FamilyCorrections) -> ImageDirectory | None:
    """Return the family and variant *name* has once *corrections* apply.

    :func:`classify_image_directory` says what a name gives; this says what the
    project makes of it, and is what anything asking which label a condition
    shares must use. **A condition is its family's base only when its name is
    the family's**: one corrected into another family is a variant there, named
    by its suffix, or by its whole name where it has none.
    """
    family = corrections.family_of(name)
    if family is None:
        return classify_image_directory(name)
    return condition_in(name, family)


def condition_in(name: str, family: str) -> ImageDirectory | None:
    """Return what the condition *name* is as a member of *family*, corrected there.

    Its family's base where *family* is its whole name, otherwise a variant
    named by its suffix, or by its whole name where it has none.
    """
    parsed = classify_image_directory(name)
    if parsed is None:
        return None
    if family_key(family) == family_key(name):
        variant = None
    else:
        variant = parsed.variant if parsed.variant is not None else name
    return ImageDirectory(modality=parsed.modality, variant=variant, family=family)


def directory_role(name: str) -> DirectoryRole:
    """Return what a directory inside a stage is for.

    Used by the survey to report directories the conventions do not recognise,
    which is how an unanticipated layout makes itself visible.
    """
    if classify_image_directory(name) is not None:
        return "image"
    if is_labels_directory(name):
        return "labels"
    if is_predictions_directory(name):
        return "predictions"
    if name.lower() == PLOTS_DIRECTORY:
        return "plots"
    if name.lower() in STATE_FOLDER_NAMES:
        return "state"
    if name == ARCHIVE_DIRECTORY:
        return "archive"
    return "unknown"


# ---------------------------------------------------------------------------
# Image file name classification
# ---------------------------------------------------------------------------

#: Markers making a loose image an alignment image whatever else its name says.
#: Tested before the reference rules, because Autolamella prefixes an alignment
#: image with ``ref_`` as readily as it does a reference image.
_ALIGNMENT_MARKERS: Final[tuple[str, ...]] = (
    "_initial_alignment_",
    "_alignment_reference_",
)

#: Prefixes naming an alignment image outright, whatever follows them.
#: ``alignment_target`` is what a milling task aligned against, written three
#: to a task just before it starts; the milling task's name follows.
_ALIGNMENT_PREFIXES: Final[tuple[str, ...]] = (
    "beam_shift_alignment",
    "alignment_target",
)

#: Whole names, with the beam suffix taken off, that are alignment images.
#: ``ref_alignment`` is matched entire rather than as a prefix because
#: everything after ``ref_`` is a *task* name somebody chose: as a prefix it
#: read a task called ``Alignment Check`` as an alignment image, silently
#: moving that task's reference images into another series. Autolamella writes
#: this one bare -- 76 of them here, none with anything after it.
_ALIGNMENT_STEMS: Final[frozenset[str]] = frozenset({"ref_alignment"})

#: What a run calls the image it aligned *against*, as opposed to the ones it
#: took while converging. Autolamella copies that image into every run using
#: it, so this is the name a duplicate wears; see
#: :func:`is_alignment_run_reference`.
ALIGNMENT_RUN_REFERENCE: Final[str] = "reference_image"

#: The fixed names Autolamella writes inside an alignment run directory,
#: alongside ``new_image_<n>``. They carry no marker of their own -- a bare
#: ``final_image.tif`` could be anything -- so they are recognised only where
#: the directory says what they are. See :func:`classify_loose_image`.
_ALIGNMENT_RUN_STEMS: Final[frozenset[str]] = frozenset(
    {ALIGNMENT_RUN_REFERENCE, "final_image"}
)
_ALIGNMENT_RUN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^new_image_\d+$")

#: The Adaptive Polishing images the script-based prototype wrote loose in the
#: lamella directory rather than into a ``<modality>`` directory. The beam is
#: spelled out after the acquisition number, which :func:`extract_image_number`
#: already reads. Runs up to 2024-06 repeated that number after the beam and
#: later ones stopped, the two always agreeing where both are there, so the
#: repeat is optional and nothing reads it.
_LOOSE_ADAPTIVE: Final[re.Pattern[str]] = re.compile(
    r"^adapt_mill_img_\d+_(?P<beam>electron|ion)(?:_\d+)?$", re.IGNORECASE
)

#: The beam a loose Adaptive Polishing name spells out, to the modality it
#: means. Separate from :data:`BEAM_SUFFIXES`, which is about a name's *ending*
#: and is what :func:`beam_modality` answers for.
_SPELLED_BEAMS: Final[dict[str, str]] = {"electron": "SEM", "ion": "FIB"}

#: Markers naming a reference image taken around a milling task: one before it
#: starts, and one after each of its milling stages finishes.
_REFERENCE_MARKERS: Final[tuple[str, ...]] = ("_pre-task_", "_finished_")

#: Everything else Autolamella prefixes ``ref_`` is a reference image, once the
#: alignment rules above have had their say.
_REFERENCE_PREFIX: Final[str] = "ref_"

#: Images deliberately not indexed. The centring image records where the
#: Adaptive Polishing plugin placed its field of view, which is a diagnostic of
#: the run rather than data anybody segments. Distinguished from a name no rule
#: recognises, which is reported so an unanticipated layout is visible.
#:
#: Anchored at the end rather than matched anywhere in the name, because the
#: plugin writes ``<lamella>_centring_<modality>`` and nothing else: as a bare
#: substring it swallowed every image of a milling task called ``Centring``,
#: and dropping an image without even reporting it unrecognised is the worst
#: way for this to be wrong.
_IGNORED_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"_centring_[a-z0-9]+$"),
)


@dataclass(frozen=True)
class ImageFile:
    """What a loose image's filename says it is.

    Attributes:
        series: ``"reference"`` or ``"alignment"``. Never ``"adaptive"``:
            Adaptive Polishing images fill a directory of their own, which
            :func:`classify_image_directory` recognises instead.
        modality: ``"SEM"`` or ``"FIB"``, or ``None`` where the name carries no
            beam suffix and only the image header can say.
    """

    series: Series
    modality: str | None


def beam_modality(image_name: str) -> str | None:
    """Return the modality the filename's beam suffix names.

    Returns:
        ``"SEM"``, ``"FIB"``, or None where the name ends in neither suffix. A
        None is not a defect to work around: roughly one image in twenty says
        nothing about its beam, and the header states it.
    """
    stem = Path(image_name).stem.lower()
    for suffix, modality in BEAM_SUFFIXES.items():
        if stem.endswith(suffix):
            return modality
    return None


def acquisition_stem(image_name: str) -> str:
    """Return the filename stem with any beam suffix removed.

    What matches the two beams' images of one moment: outside an Adaptive
    Polishing stage the pair carries no acquisition number, and differs only in
    the suffix -- ``ref_Polishing_start_eb`` and ``ref_Polishing_start_ib``.
    Beside :func:`extract_image_number` because both answer the same question
    about different eras of filename.
    """
    stem = Path(image_name).stem
    for suffix in BEAM_SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_alignment_directory(name: str) -> bool:
    """Return whether *name* is the directory collecting alignment runs."""
    return name.lower() == ALIGNMENT_DIRECTORY.lower()


def is_alignment_run_reference(image_name: str) -> bool:
    """Return whether *image_name* is what a run calls the image it aligned to.

    Only meaningful inside an :data:`ALIGNMENT_DIRECTORY` run: the name is
    generic enough to mean anything anywhere else, which is the same reason
    :func:`classify_loose_image` reads it only where the directory vouches
    for it.
    """
    return Path(image_name).stem.lower() == ALIGNMENT_RUN_REFERENCE


def is_ignored_image_name(image_name: str) -> bool:
    """Return whether an image is deliberately not indexed.

    Separate from "no rule recognises this", which the survey reports.
    """
    stem = Path(image_name).stem.lower()
    return any(pattern.search(stem) for pattern in _IGNORED_PATTERNS)


def classify_loose_image(image_name: str, in_alignment_run: bool = False) -> ImageFile | None:
    """Return what a loose image file's name says it is, or None if nothing does.

    Loose meaning a file sitting directly in a lamella, milling task or
    alignment run directory, rather than filling a ``<modality>`` directory of
    its own. The extension is not examined; callers have already selected image
    files with :func:`image_files`.

    Args:
        image_name: The filename, with or without its extension.
        in_alignment_run: Whether the file sits in a subdirectory of an
            :data:`ALIGNMENT_DIRECTORY`. The three fixed names an alignment run
            holds are generic enough to mean anything, so they are read as
            alignment images only where the directory vouches for them.

    Returns:
        The series and modality the name states, or None where no rule matches
        and where the name is one :func:`is_ignored_image_name` covers. A None
        is reported rather than guessed at: an Autolamella version naming
        something new should be visible on the next scan.
    """
    if is_ignored_image_name(image_name):
        return None

    stem = Path(image_name).stem
    lowered = stem.lower()
    modality = beam_modality(stem)

    # First, and anchored: this is the one loose name that states its own beam
    # rather than suffixing it, and the one that is not Autolamella's.
    spelled = _LOOSE_ADAPTIVE.match(stem)
    if spelled is not None:
        return ImageFile(
            series="adaptive", modality=_SPELLED_BEAMS[spelled["beam"].lower()]
        )

    if (
        lowered.startswith(_ALIGNMENT_PREFIXES)
        or acquisition_stem(stem).lower() in _ALIGNMENT_STEMS
        or any(marker in lowered for marker in _ALIGNMENT_MARKERS)
    ):
        return ImageFile(series="alignment", modality=modality)

    if in_alignment_run and (
        lowered in _ALIGNMENT_RUN_STEMS or _ALIGNMENT_RUN_PATTERN.match(lowered)
    ):
        return ImageFile(series="alignment", modality=modality)

    if lowered.startswith(_REFERENCE_PREFIX) or any(
        marker in lowered for marker in _REFERENCE_MARKERS
    ):
        return ImageFile(series="reference", modality=modality)

    return None


#: Where a name that is not part of a run's fixed sequence is placed: after all
#: of them, leaving the filename to order it.
UNSEQUENCED: Final[int] = 1 << 20


def alignment_run_order(image_name: str) -> int:
    """Return where a filename sits in an alignment run's fixed sequence.

    A run holds the reference image it aligns against, the images taken as it
    iterates, and the final one. A header records whole seconds, so several of
    them land on the same second and need something to break the tie, or a run
    reads ``final_image`` before its last ``new_image``. The sequence is fixed
    by the names Autolamella writes, which is what makes this a rule rather
    than a guess.

    Returns:
        A rank, or :data:`UNSEQUENCED` for a name outside the sequence.
    """
    stem = Path(image_name).stem.lower()
    if stem == "reference_image":
        return 0
    iteration = _ALIGNMENT_RUN_PATTERN.match(stem)
    if iteration is not None:
        return 1 + int(stem.rsplit("_", 1)[1])
    if stem == "final_image":
        return UNSEQUENCED - 1
    return UNSEQUENCED


def loose_image_set_name(series: Series, modality: str) -> str:
    """Return the name identifying one series and modality of loose images.

    Loose images have no image directory to take a name from, but a label
    directory still has to be called something. Feeding this to
    :func:`labels_directory_name` gives ``reference_sem_labels/``, keeping one
    rule for what a label directory is called.

    A modality is required: a set whose beam nothing has established gets no
    label directory rather than one named for a guess.

    Raises:
        ValueError: If *series* is not one of :data:`LOOSE_SERIES`, which is
            every series that has labels -- so this fires for ``other``, or for
            a value that is not a series at all, and means the caller has lost
            track of what it is holding.
    """
    if series not in LOOSE_SERIES:
        raise ValueError(
            f"Not a loose series: {series!r}; expected one of {list(LOOSE_SERIES)}"
        )
    return f"{series}_{modality.lower()}"


def other_image_set_name(modality: str | None) -> str:
    """Return the name identifying one modality of ``other`` images.

    The counterpart of :func:`loose_image_set_name` for the series that has no
    labels, and deliberately not that function: feeding this to
    :func:`labels_directory_name` would name a directory nothing may write to.
    The beam is part of the name where one is known, so that a directory
    holding both keeps them apart; a set whose beam nothing states is simply
    ``other``, there being at most one of those per directory.
    """
    return f"other_{modality.lower()}" if modality else "other"


# ---------------------------------------------------------------------------
# Milling task names
# ---------------------------------------------------------------------------

#: The directory a lamella collects its milling tasks in, one subdirectory per
#: task, in the 2026-07 layout and later. Beside :data:`ALIGNMENT_DIRECTORY`.
MILLING_DIRECTORY: Final[str] = "Milling"


def normalise_task_name(value: str) -> str:
    """Return *value* reduced to what makes two spellings of a task equal.

    Case and every separator go, so ``Rough 1``, ``Rough-1`` and ``rough_1``
    are one name, and the ``mill_polishing`` of a protocol matches the
    ``MillPolishing`` the same experiment writes into a filename. Task names
    are whatever whoever wrote the protocol chose, so nothing here may assume
    a spelling.
    """
    return re.sub(r"[^a-z0-9]", "", value.lower())


def task_name_words(value: str) -> list[str]:
    """Return *value* split into the words a task name could be built from."""
    return [word for word in re.split(r"[^A-Za-z0-9]+", value) if word]


def task_name_runs(value: str, max_words: int) -> list[str]:
    """Return the runs of consecutive words in *value*, most specific first.

    A directory or filename says which task it belongs to by carrying the
    task's name among the words describing the acquisition --
    ``Rough-Milling-Rough-2_finished_20-21-21``. Offering every run of words to
    a vocabulary is therefore one rule covering every filename shape, where a
    pattern per shape would need extending each time Autolamella names
    something new. It is also what splits a composite: ``Rough Milling`` and
    ``Rough 2`` are both in the vocabulary and both name the same task.

    Longest first, then leftmost, so a caller taking the first run the
    vocabulary knows takes the most specific one. *max_words* is the longest
    name the vocabulary holds, past which no run could match.
    """
    words = task_name_words(value)
    return [
        normalise_task_name("".join(words[start:start + length]))
        for length in range(min(max_words, len(words)), 0, -1)
        for start in range(len(words) - length + 1)
    ]


# ---------------------------------------------------------------------------
# Label and prediction directory resolution
# ---------------------------------------------------------------------------

def labels_directory_name(image_directory_name: str) -> str:
    """Return the label directory name belonging to an image directory."""
    return f"{image_directory_name}{LABELS_SUFFIX}"


def shared_labels_directory_name(modality: str) -> str:
    """Return the shared label directory name for a modality."""
    return f"{modality.lower()}{LABELS_SUFFIX}"


def _family_members(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    corrections: FamilyCorrections,
) -> list[tuple[str, ImageDirectory]]:
    """Return the conditions of *image_directory*'s family in its stage, base first.

    Each is a ``(directory name, condition)`` pair. *image_directory* is always
    among them, whether or not it is in *siblings*; an unrecognised name is read
    as a family of its own, so resolution still has an answer for it.
    """
    own = condition_of(image_directory.name, corrections) or ImageDirectory(
        modality=modality, variant=None, family=image_directory.name
    )
    members = {image_directory.name.lower(): (image_directory.name, own)}
    for key, sibling in siblings.items():
        condition = condition_of(sibling.name, corrections)
        if (
            condition is not None
            and condition.modality == own.modality
            and condition.family.lower() == own.family.lower()
        ):
            members.setdefault(key, (sibling.name, condition))
    return sorted(
        members.values(), key=lambda item: (item[1].variant is not None, item[0].lower())
    )


def _family_stem(family: str, modality: str) -> str:
    """Return what a family's own directories are named after: ``sem`` for the modality's."""
    return modality.lower() if family.lower() == modality.lower() else family


def families_claiming_one_directory(
    image_directories: Iterable[Path], *, corrections: FamilyCorrections
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return each directory name two families of one stage could both read.

    A family reads ``<family>_labels`` and its conditions' private
    ``<condition>_labels``, so a base corrected into another family keeps a
    name that is still its old family's own; whichever of them exists, both
    families read it, and two segmentations become one. **Judged by name,
    whether or not the directory exists**: one not yet written would join them
    the moment it is (``test_two_families_never_read_one_label_directory``).

    Args:
        image_directories: The image directories of one stage.

    Returns:
        ``(stem, families)`` per name claimed twice, the stem being what
        :data:`LABELS_SUFFIX` and :data:`PREDICTIONS_SUFFIX` are appended to.
    """
    claims: dict[str, tuple[str, dict[tuple[str, str], str]]] = {}
    for directory in image_directories:
        condition = condition_of(directory.name, corrections)
        if condition is None:
            continue
        family = (condition.modality, condition.family.lower())
        for stem in (_family_stem(condition.family, condition.modality), directory.name):
            spelled, families = claims.setdefault(stem.lower(), (stem, {}))
            families.setdefault(family, condition.family)
    return tuple(
        (spelled, tuple(sorted(families.values(), key=str.lower)))
        for spelled, families in (claims[key] for key in sorted(claims))
        if len(families) > 1
    )


#: How a stage's directories fail to give a family one label: two families
#: reading one directory, or a family whose labels or predictions are split.
FamilyProblemKind = Literal["shared", "labels", "predictions"]


@dataclass(frozen=True)
class FamilyDirectoryProblem:
    """A stage's families failing to read one directory each, as :func:`family_directory_problems` finds it.

    Attributes:
        kind: Two families reading the directories named after *stem*, or
            one family's ``labels`` or ``predictions`` split between
            *directories*.
        families: The families involved, as spelled.
        stem: What both families' directories are named after, for ``shared``.
        directories: The competing directories, for a split.
        condition: A condition of the split family, for naming it.
    """

    kind: FamilyProblemKind
    families: tuple[str, ...]
    stem: str = ""
    directories: tuple[Path, ...] = ()
    condition: str = ""

    @property
    def key(self) -> tuple[str, frozenset[str]]:
        """Return what identifies the problem across two readings of one stage."""
        if self.kind == "shared":
            return self.kind, frozenset({self.stem.lower()})
        return self.kind, frozenset(path.name.lower() for path in self.directories)


def family_directory_problems(
    image_directories: Iterable[Path],
    siblings: Mapping[str, Path],
    *,
    corrections: FamilyCorrections,
) -> tuple[FamilyDirectoryProblem, ...]:
    """Return each way *corrections* leave a stage's families not reading one directory each.

    What ``inspect-layout`` reports and ``edit-families`` refuses to make, so
    the two agree about what a conflict is: two families claiming one name,
    as :func:`families_claiming_one_directory` judges it, and a family whose
    labels or predictions lie in several existing directories.

    Args:
        image_directories: The stage's directories; any that is no image
            directory is passed over.
        siblings: The stage's subdirectories, as :func:`subdirectories_by_name`
            returns them.
    """
    directories = sorted(
        (path for path in image_directories if classify_image_directory(path.name) is not None),
        key=lambda path: path.name,
    )
    problems = [
        FamilyDirectoryProblem("shared", families, stem=stem)
        for stem, families in families_claiming_one_directory(directories, corrections=corrections)
    ]
    for kind, suffix, legacy in (
        ("labels", LABELS_SUFFIX, LEGACY_LABEL_DIRECTORIES),
        ("predictions", PREDICTIONS_SUFFIX, LEGACY_PREDICTION_DIRECTORIES),
    ):
        seen: set[tuple[str, str]] = set()
        for directory in directories:
            condition = condition_of(directory.name, corrections)
            assert condition is not None, "only image directories are kept"
            family = (condition.modality, family_key(condition.family))
            if family in seen:
                continue
            existing, _, _ = _family_directories(
                directory, condition.modality, siblings, suffix, legacy, corrections
            )
            if len(existing) > 1:
                seen.add(family)
                problems.append(
                    FamilyDirectoryProblem(
                        kind, (condition.family,), directories=tuple(existing),
                        condition=directory.name,
                    )
                )
    return tuple(problems)


def families_of_stage(
    image_directories: Iterable[Path], *, corrections: FamilyCorrections
) -> list[list[Path]]:
    """Return a stage's image directories grouped into their families, each in name order."""
    families: dict[tuple[str, str], list[Path]] = {}
    for directory in sorted(image_directories, key=lambda path: path.name):
        condition = condition_of(directory.name, corrections)
        if condition is not None:
            key = (condition.modality, family_key(condition.family))
            families.setdefault(key, []).append(directory)
    return list(families.values())


def _family_directories(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    suffix: str,
    legacy: Mapping[str, str],
    corrections: FamilyCorrections,
) -> tuple[list[Path], Path, list[Path]]:
    """Return a family's existing directories of one kind, and where a new one goes.

    Returns:
        ``(existing, new, legacy_existing)``: the family's own directory and its
        conditions' private ones that exist, in that order; the name a new one
        takes; and the pre-convention directories present for the modality,
        which only the family named after the modality reads.
    """
    members = _family_members(image_directory, modality, siblings, corrections)
    family = members[0][1].family
    named_after_modality = family.lower() == modality.lower()
    new_name = f"{_family_stem(family, modality)}{suffix}"

    candidates = [new_name, *(f"{name}{suffix}" for name, _ in members)]
    existing: list[Path] = []
    for name in dict.fromkeys(candidate.lower() for candidate in candidates):
        if name in siblings:
            existing.append(siblings[name])

    legacy_existing = [
        siblings[name]
        for name, legacy_modality in legacy.items()
        if named_after_modality and legacy_modality == modality and name in siblings
    ]
    return existing, image_directory.parent / new_name, legacy_existing


def _resolve_family_directory(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    suffix: str,
    legacy: Mapping[str, str],
    corrections: FamilyCorrections,
) -> tuple[Path, bool, bool]:
    """Resolve one kind of family directory; see :func:`resolve_labels_directory`."""
    existing, new, legacy_existing = _family_directories(
        image_directory, modality, siblings, suffix, legacy, corrections
    )
    if len(existing) > 1:
        own = siblings.get(f"{image_directory.name}{suffix}".lower())
        return (own or existing[0]), True, False
    if existing:
        return existing[0], True, False
    if legacy_existing:
        return legacy_existing[0], True, True
    return new, False, False


def resolve_labels_directory(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    *,
    corrections: FamilyCorrections,
) -> tuple[Path, bool, bool]:
    """Return where an image directory's labels live.

    A family's labels are ``<family>_labels`` -- ``sem_labels`` for the family
    named after its modality. **One condition's private directory stands in for
    it** where it is the only one (``SEM_8_labels`` beside ``SEM_8_fi_1``), and
    the pre-convention ``labels`` or ``labels_FIB`` where neither exists. **Where
    several exist, the family is in conflict** (:func:`conflicting_label_directories`):
    each condition keeps reading its own until they are merged, and one without
    its own reads the family's. With none, the answer is where a first label
    would be written, the same answer it will have once written.

    Args:
        image_directory: The image directory whose labels are wanted.
        modality: ``"SEM"`` or ``"FIB"``.
        siblings: Existing sibling directories, keyed by lowercased name.
            Matching through this mapping rather than :meth:`Path.exists` keeps
            resolution case-insensitive on every platform, which matters for
            the legacy ``labels_FIB``.
        corrections: The project's family corrections. Required, so that no
            caller resolves by name alone by forgetting them.

    Returns:
        A tuple of ``(directory, exists, is_legacy)``.
    """
    return _resolve_family_directory(
        image_directory, modality, siblings, LABELS_SUFFIX, LEGACY_LABEL_DIRECTORIES,
        corrections,
    )


def conflicting_label_directories(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    *,
    corrections: FamilyCorrections,
) -> tuple[Path, ...]:
    """Return the label directories competing for *image_directory*'s family.

    Empty unless more than one exists, which leaves the family's conditions
    each reading their own.
    """
    existing, _, _ = _family_directories(
        image_directory, modality, siblings, LABELS_SUFFIX, LEGACY_LABEL_DIRECTORIES,
        corrections,
    )
    return tuple(existing) if len(existing) > 1 else ()


def predictions_directory_name(image_directory_name: str) -> str:
    """Return the predictions directory name belonging to an image directory."""
    return f"{image_directory_name}{PREDICTIONS_SUFFIX}"


def is_convertible_prediction(prediction_path: Path) -> bool:
    """Return whether a prediction file may be converted into a label.

    Everything except the post-processing pipeline's ``clean/`` output, whose
    intensity values no longer correspond to the segmentation classes. Stated as
    an exclusion so that a prediction sitting directly in a ``predictions/``
    directory, with no ``raw``/``clean`` split, stays convertible.

    Reads ``parts`` rather than ``parents`` -- the latter builds a path object
    per level, and this is called once per image behind a line that recounts
    every time the selection changes. The final part is the filename, which
    names no directory and is excluded.
    """
    return not any(part.lower() == POST_PROCESSED_KIND for part in prediction_path.parts[:-1])


#: Where a convertible prediction came from: this package's own model, or the
#: microscope's segmentation pipeline -- its ``raw/`` output, or a predictions
#: directory with no ``raw``/``clean`` split.
PredictionOrigin = Literal["am-seg", "microscope"]

PREDICTION_ORIGINS: Final[tuple[PredictionOrigin, ...]] = ("am-seg", "microscope")

#: The source a conversion takes when an image has both. What the window opens
#: on and what the command line uses unless told otherwise.
DEFAULT_PREDICTION_ORIGIN: Final[PredictionOrigin] = "am-seg"


def prediction_origin(prediction_path: Path) -> PredictionOrigin:
    """Return which of :data:`PREDICTION_ORIGINS` wrote *prediction_path*."""
    kinds = {part.lower() for part in prediction_path.parts[:-1]}
    return "am-seg" if OWN_PREDICTION_KIND in kinds else "microscope"


def other_prediction_origin(source: PredictionOrigin) -> PredictionOrigin:
    """Return the source that is not *source*."""
    return "microscope" if source == "am-seg" else "am-seg"


def prediction_precedence(
    prediction_path: Path, prefer: PredictionOrigin = DEFAULT_PREDICTION_ORIGIN
) -> int:
    """Return how strongly a prediction claims to be the one converted.

    Lower wins: the *prefer* source, then the other, then post-processed output,
    which is never converted. Stated as an ordering rather than left to the order
    directories are listed in, since which file becomes a label is not a thing
    to decide by accident.
    """
    if not is_convertible_prediction(prediction_path):
        return 2
    return 0 if prediction_origin(prediction_path) == prefer else 1


def resolve_predictions_directory(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    *,
    corrections: FamilyCorrections,
) -> tuple[Path, bool, bool]:
    """Return where an image directory's predictions live.

    Resolved exactly as :func:`resolve_labels_directory` resolves labels, with
    ``predictions`` and ``predictions_fib`` as the pre-convention names, and
    shared by reading and writing. One directory serves a family because only
    its primary image is ever predicted -- see :mod:`..operations.predict` -- so
    two conditions cannot claim one filename.

    Args:
        image_directory: The image directory whose predictions are wanted.
        modality: ``"SEM"`` or ``"FIB"``.
        siblings: Existing sibling directories, keyed by lowercased name, as
            :func:`subdirectories_by_name` returns them.
        corrections: The project's family corrections.

    Returns:
        A tuple of ``(directory, exists, is_legacy)``.
    """
    return _resolve_family_directory(
        image_directory, modality, siblings, PREDICTIONS_SUFFIX, LEGACY_PREDICTION_DIRECTORIES,
        corrections,
    )


def conflicting_predictions_directories(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
    *,
    corrections: FamilyCorrections,
) -> tuple[Path, ...]:
    """Return the predictions directories competing for *image_directory*'s family."""
    existing, _, _ = _family_directories(
        image_directory, modality, siblings, PREDICTIONS_SUFFIX, LEGACY_PREDICTION_DIRECTORIES,
        corrections,
    )
    return tuple(existing) if len(existing) > 1 else ()


def resolve_loose_labels_directory(
    directory: Path,
    series: Series,
    modality: str,
    siblings: Mapping[str, Path],
) -> tuple[Path, bool]:
    """Return where a loose image set's labels live.

    **There is no fallback chain here**, unlike :func:`resolve_labels_directory`:
    the name is exact. Falling back to the shared ``<modality>_labels`` would
    let a reference set in a dataset that collapses the stage level claim that
    lamella's Adaptive Polishing labels, pairing every reference image with a
    label of a different scene.

    Args:
        directory: The directory the loose images sit in.
        series: Which series the set holds.
        modality: ``"SEM"`` or ``"FIB"``.
        siblings: Existing subdirectories of *directory*, keyed by lowercased
            name, as :func:`subdirectories_by_name` returns them.

    Returns:
        A tuple of ``(directory, exists)``. When the directory is absent the
        name is still returned: that is where a label would be written.
    """
    name = labels_directory_name(loose_image_set_name(series, modality))
    existing = siblings.get(name.lower())
    if existing is not None:
        return existing, True
    return directory / name, False


def resolve_loose_predictions_directory(
    directory: Path,
    series: Series,
    modality: str,
    siblings: Mapping[str, Path],
) -> tuple[Path, bool]:
    """Return where a loose image set's predictions live.

    **The name is exact, with no fallback chain**, for the same reason
    :func:`resolve_loose_labels_directory`'s is: falling back to the shared
    ``<modality>_predictions`` would let a reference set in a dataset that
    collapses the stage level claim the Adaptive Polishing pipeline's own
    output, pairing every reference image with a prediction of a different
    scene. Reading and writing share this, so a prediction is looked for where
    one would be written.

    Args:
        directory: The directory the loose images sit in.
        series: Which series the set holds.
        modality: ``"SEM"`` or ``"FIB"``.
        siblings: Existing subdirectories of *directory*, keyed by lowercased
            name, as :func:`subdirectories_by_name` returns them.

    Returns:
        A tuple of ``(directory, exists)``. When the directory is absent the
        name is still returned: that is where a prediction would be written.
    """
    name = predictions_directory_name(loose_image_set_name(series, modality))
    existing = siblings.get(name.lower())
    if existing is not None:
        return existing, True
    return directory / name, False


# ---------------------------------------------------------------------------
# Workflow state locations
# ---------------------------------------------------------------------------

def state_directories(directory: Path) -> list[Path]:
    """Return *directory* and the state subdirectories it may still have.

    Only migration and the survey look inside those: an unmigrated dataset is
    something to report, not something to read.
    """
    return [directory, *(directory / name for name in sorted(STATE_FOLDER_NAMES))]


# ---------------------------------------------------------------------------
# Image set assembly
# ---------------------------------------------------------------------------

def build_image_sets(
    stage_directory: Path, *, corrections: FamilyCorrections
) -> list[ImageSet]:
    """Return the image sets in a stage directory, ordered for display.

    Adaptive Polishing sets only -- the images loose in the same directory are
    :func:`build_loose_image_sets`'s. Keeping them apart is what stops
    discovery mistaking a lamella full of reference images for a stage; see
    :func:`~am_segmentation_manager.discovery.find_stage_directories`.

    One of the few functions in this module that lists a directory. Assembling
    an image set requires knowing which sibling directories exist, and doing it
    here keeps the convention in one place rather than duplicating it in the
    scanner and the survey.

    Ordering is by modality (SEM before FIB), then family, then the base
    condition first and variants by name, so output is stable across runs.
    Families and variants are those *corrections* give, as
    :func:`condition_of` reads them.
    """
    siblings = subdirectories_by_name(stage_directory)

    classified: list[tuple[Path, ImageDirectory, Path, bool, bool]] = []
    for directory in siblings.values():
        image_directory = condition_of(directory.name, corrections)
        if image_directory is None:
            continue
        labels_directory, labels_exist, labels_are_legacy = resolve_labels_directory(
            directory, image_directory.modality, siblings, corrections=corrections
        )
        classified.append(
            (directory, image_directory, labels_directory, labels_exist, labels_are_legacy)
        )

    shared_counts = Counter(str(entry[2]) for entry in classified)

    image_sets = [
        ImageSet(
            directory=directory,
            modality=image_directory.modality,
            variant=image_directory.variant,
            family=image_directory.family,
            labels_directory=labels_directory,
            labels_exist=labels_exist,
            labels_are_shared=shared_counts[str(labels_directory)] > 1,
            labels_are_legacy=labels_are_legacy,
        )
        for directory, image_directory, labels_directory, labels_exist, labels_are_legacy
        in classified
    ]
    image_sets.sort(key=_image_set_sort_key)
    return image_sets


def _image_set_sort_key(image_set: ImageSet) -> tuple[int, tuple[str, bool, str]]:
    """Order image sets by modality, then as :func:`condition_order` orders them."""
    modality_order = MODALITIES.index(image_set.modality) if image_set.modality in MODALITIES else len(MODALITIES)
    return modality_order, condition_order(image_set.family, image_set.variant)


def condition_order(family: str | None, variant: str | None) -> tuple[str, bool, str]:
    """Return where a condition sorts among its beam's: by family, then as :func:`primary_order`."""
    return (family_key(family), *primary_order(variant))


def family_key(family: str | None) -> str:
    """Return what identifies a family in comparisons: its name in lower case.

    Family names compare case-insensitively, as a stage's directories do; ``""``
    for no family.
    """
    return (family or "").lower()


def primary_order(variant: str | None) -> tuple[bool, str]:
    """Return where a condition sorts within its family: the base, then variants by name.

    **The default primary is the first in this order**, which is what the oldest
    layouts need, having no base: their SEM sets are ``SEM_8``, ``SEM_8_fi_1``
    and so on. A primary named in ``AM_families.yaml`` comes before it.
    """
    return variant is not None, (variant or "").lower()


@dataclass(frozen=True)
class LooseImageSets:
    """The loose image sets in one directory, and the files that did not fit.

    The leftovers are returned rather than logged, because each needs a
    different answer and only the survey can give one: an unrecognised name may
    be an Autolamella version naming something new, while an unresolved beam is
    an image whose header would not read.

    Attributes:
        sets: The sets found, ordered by series then modality.
        unrecognised: Filenames no classification rule matched. Deliberate
            omissions are not among them -- see :func:`is_ignored_image_name`.
            These are *also* in the ``other`` set, which is not a contradiction:
            the set is where the file is indexed and this is why it landed
            there, and the survey reports the reason so that a name Autolamella
            has newly invented stays visible rather than being absorbed.
        unresolved: Filenames classified into a series, but whose beam neither
            the name nor *resolve_modality* could establish. A curated series'
            set is keyed by beam and there is nothing to key these on, so they
            join the beamless ``other`` set -- indexed, and reported here so
            that the series their name does state is not quietly lost.
    """

    sets: tuple[ImageSet, ...] = ()
    unrecognised: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


def build_loose_image_sets(
    directory: Path,
    in_alignment_run: bool = False,
    resolve_modality: Callable[[Path], str | None] | None = None,
) -> LooseImageSets:
    """Return the reference and alignment sets among a directory's loose files.

    Loose images share their directory with whatever else Autolamella wrote
    there, so a set is a group of filenames rather than a directory: the files
    are classified with :func:`classify_loose_image` and grouped by series and
    modality. A directory holding no loose images yields nothing, which is the
    usual answer.

    Args:
        directory: The directory to look in. Not descended into.
        in_alignment_run: Whether *directory* is one alignment run, so the
            generic names a run holds can be read as alignment images.
        resolve_modality: Called for an image whose *name* does not state its
            beam, to get it from somewhere that can -- the image header. Left
            unset, such an image is reported as unresolved instead. The seam is
            here so that reading a header stays the caller's decision: scanning
            is a filesystem walk, and this is the one narrow exception to it.

    Returns:
        A :class:`LooseImageSets` holding the sets and, for each file that
        reached the ``other`` set or no set at all, why.
    """
    siblings = subdirectories_by_name(directory)
    grouped: dict[tuple[Series, str], list[str]] = {}
    other: dict[str | None, list[str]] = {}
    unrecognised: list[str] = []
    unresolved: list[str] = []

    for path in image_files(directory):
        if is_ignored_image_name(path.name):
            continue
        classified = classify_loose_image(path.name, in_alignment_run=in_alignment_run)
        # `beam_modality` for the unrecognised, because `classify_loose_image`
        # never got as far as reading the suffix off a name it rejected -- and
        # a suffix the filename states beats a header nobody has opened.
        modality = (
            classified.modality if classified is not None else beam_modality(path.name)
        )
        if modality is None and resolve_modality is not None:
            modality = resolve_modality(path)
        if classified is None:
            # No rule reached it, so it is indexed as `other` and reported as
            # unrecognised. Both: the file becomes something a reviewer can
            # open, and the finding that makes an unanticipated layout visible
            # keeps its data.
            unrecognised.append(path.name)
            other.setdefault(modality, []).append(path.name)
            continue
        if modality is None:
            # Its name says which series, but a set of that series is keyed by
            # beam. Indexing it as `other` keeps it openable; the finding says
            # what was known about it, which is what `other` cannot carry.
            unresolved.append(path.name)
            other.setdefault(None, []).append(path.name)
            continue
        grouped.setdefault((classified.series, modality), []).append(path.name)

    sets: list[ImageSet] = []
    for series, modality in sorted(grouped, key=_loose_group_sort_key):
        labels_directory, labels_exist = resolve_loose_labels_directory(
            directory, series, modality, siblings
        )
        sets.append(
            ImageSet(
                directory=directory,
                modality=modality,
                variant=None,
                labels_directory=labels_directory,
                labels_exist=labels_exist,
                # A loose label directory is named for its series and beam, so
                # no two sets in one directory can resolve to the same one.
                labels_are_shared=False,
                labels_are_legacy=False,
                series=series,
                image_names=tuple(sorted(grouped[(series, modality)])),
            )
        )
    for modality in sorted(other, key=_other_group_sort_key):
        sets.append(
            ImageSet(
                directory=directory,
                modality=modality,
                variant=None,
                labels_directory=None,
                labels_exist=False,
                labels_are_shared=False,
                labels_are_legacy=False,
                series="other",
                image_names=tuple(sorted(other[modality])),
            )
        )

    if unrecognised:
        logger.debug("%d unrecognised image name(s) in %s", len(unrecognised), directory)
    return LooseImageSets(
        sets=tuple(sets),
        unrecognised=tuple(sorted(unrecognised)),
        unresolved=tuple(sorted(unresolved)),
    )


def _loose_group_sort_key(group: tuple[Series, str]) -> tuple[int, int]:
    """Order loose sets by series, then by modality, both in declared order."""
    series, modality = group
    return (
        SERIES.index(series),
        MODALITIES.index(modality) if modality in MODALITIES else len(MODALITIES),
    )


def _other_group_sort_key(modality: str | None) -> tuple[int, int]:
    """Order ``other`` sets by modality, the beamless one last.

    Last because it is the least informative of them, and because sorting
    ``None`` against strings raises rather than choosing.
    """
    if modality is None:
        return (1, 0)
    return (0, MODALITIES.index(modality) if modality in MODALITIES else len(MODALITIES))


def image_files_of(image_set: ImageSet) -> dict[str, Path]:
    """Return the image files belonging to *image_set*, keyed by filename.

    **The one statement of which files are a set's**, so that a caller never
    has to know whether a set is a whole directory or a group of names picked
    out of one. A loose set's files are not re-listed or re-classified: they
    were named when the set was built.
    """
    if image_set.image_names is None:
        return image_files_by_name(image_set.directory)
    return {name: image_set.directory / name for name in image_set.image_names}


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    """One directory entry, and what the listing already knew it was.

    ``os.scandir`` reports the kind of an entry from the directory read itself,
    where asking a :class:`~pathlib.Path` costs a ``stat`` apiece -- 73,564 of
    them in one scan of the reference collection. Both flags follow symlinks,
    as ``Path.is_dir`` and ``Path.is_file`` do, and both are False for an entry
    that could not be inspected.
    """

    path: Path
    is_dir: bool
    is_file: bool


#: The listings held for the length of one :func:`directory_cache` block, or
#: None outside one. A :class:`~contextvars.ContextVar` rather than a module
#: global so that a scan running on a worker thread gets its own.
_listings: ContextVar[dict[Path, list[_Entry]] | None] = ContextVar(
    "_listings", default=None
)


@contextmanager
def directory_cache() -> Iterator[None]:
    """List each directory at most once for the length of this block.

    A scan reads the same directory several times -- once to ask whether it
    marks a project, once for its image sets, once for its loose images, once
    for the children to descend into -- and each listing is a syscall per entry
    on top of the read. Held for one scan the answers cannot go stale, because
    a scan is a snapshot of the filesystem either way; held any longer they
    could, which is why this is a block rather than a permanent memo. **Nothing
    that writes files may run inside one.** Nesting reuses the outer block.
    """
    if _listings.get() is not None:
        yield
        return
    token = _listings.set({})
    try:
        yield
    finally:
        _listings.reset(token)


def _list_directory(directory: Path) -> list[_Entry]:
    """Return the entries in *directory*, treating unreadable ones as empty.

    A directory that does not exist is silently empty: callers routinely ask
    about optional locations such as state subdirectories. Anything else --
    a permissions problem, or a cloud-storage placeholder that has not synced --
    is warned about but still does not raise, so one bad directory cannot abort
    a survey of an entire collection.

    Ordered by :func:`os.path.normcase` of the name, which is how sorting the
    paths themselves ordered them: case-insensitively where the platform's
    paths are.
    """
    cache = _listings.get()
    if cache is not None:
        cached = cache.get(directory)
        if cached is not None:
            return cached
    entries = _read_directory(directory)
    if cache is not None:
        cache[directory] = entries
    return entries


def _read_directory(directory: Path) -> list[_Entry]:
    """Return *directory*'s entries, unconditionally reading it."""
    try:
        with os.scandir(directory) as scan:
            entries = [
                _Entry(Path(entry.path), _kind(entry, "is_dir"), _kind(entry, "is_file"))
                for entry in scan
            ]
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        logger.warning("Could not read directory %s: %s", directory, exc)
        return []
    entries.sort(key=lambda entry: os.path.normcase(entry.path.name))
    return entries


def _kind(entry: os.DirEntry, question: str) -> bool:
    """Return what *entry* answers to ``is_dir`` or ``is_file``, errors being no.

    ``os.DirEntry`` swallows the errors that mean "gone" and raises the rest,
    which here is the same warning a whole unreadable directory gets.
    """
    try:
        return bool(getattr(entry, question)())
    except OSError as exc:  # pragma: no cover - transient filesystem error
        logger.warning("Could not inspect %s: %s", entry.path, exc)
        return False


def subdirectories_by_name(directory: Path) -> dict[str, Path]:
    """Return *directory*'s subdirectories keyed by lowercased name.

    Hidden directories are skipped, as is :data:`ARCHIVE_DIRECTORY`. This is
    the one place every traversal goes through, so excluding the archive here
    keeps it out of discovery, the scanner and the survey at once.
    """
    return {
        entry.path.name.lower(): entry.path
        for entry in _list_directory(directory)
        if entry.is_dir
        and not entry.path.name.startswith(".")
        and entry.path.name != ARCHIVE_DIRECTORY
    }


def image_files(directory: Path) -> list[Path]:
    """Return the image files directly inside *directory*, sorted by name."""
    return [
        entry.path for entry in _list_directory(directory)
        if entry.is_file and entry.path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def image_files_by_name(directory: Path) -> dict[str, Path]:
    """Return the image files directly in *directory*, keyed by filename.

    The filename is the identity an image and its label share. State
    subdirectories are **not** looked in: a file still sitting in one belongs
    to a dataset that has not been migrated, and reading it here would report
    it as selected. ``inspect-layout`` says so instead.
    """
    return {path.name: path for path in image_files(directory)}

