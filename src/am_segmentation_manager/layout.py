"""Filesystem naming conventions for Autolamella and legacy datasets.

The single source of truth for what directories and image files are called;
resolve names here rather than embedding string literals. Image directories are
``<modality>`` or ``<modality>_<variant>``, the variant opaque and never parsed.
Label directories are ``<image-directory>_labels`` beside them, resolved
most-specific-first, then the shared ``<modality>_labels``, then legacy names.
Images Autolamella leaves loose in a lamella or task directory instead carry
their series and their beam in the filename, which :func:`classify_loose_image`
reads. Workflow state is declared in :mod:`~am_segmentation_manager.state` and
nothing here reads it; the ``deselected/`` and ``corrected/`` names stay
reserved so an unmigrated dataset is never read as selected.

Except for the two image-set builders and the helpers at the end of the module,
nothing here touches the filesystem.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

logger = logging.getLogger(__name__)

#: The imaging modalities this package indexes, in display order.
MODALITIES: Final[tuple[str, ...]] = ("SEM", "FIB")

#: The subdirectory names a pre-declaration dataset kept its state in. Nothing
#: reads state from them any more -- :mod:`~..operations.migrate_state` does
#: that, once -- but they stay reserved so an unmigrated dataset reads as
#: *missing* those images rather than as holding selected ones.
STATE_FOLDER_NAMES: Final[frozenset[str]] = frozenset({"deselected", "corrected"})

LABELS_SUFFIX: Final[str] = "_labels"
PREDICTIONS_SUFFIX: Final[str] = "_predictions"
PLOTS_DIRECTORY: Final[str] = "plots"

#: Subdirectories of a predictions directory holding different prediction sets.
#: All are indexed; not all may become a label. See
#: :func:`is_convertible_prediction` and :func:`prediction_precedence`.
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

#: File extensions treated as images or labels.
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".tif", ".tiff"})

#: What a directory inside a stage is for.
DirectoryRole = Literal["image", "labels", "predictions", "plots", "state", "unknown"]

#: What an image was taken for. Orthogonal to modality and to the
#: imaging-condition variant: one directory holds several series, so a series is
#: a property of an image set rather than of the directory it sits in.
#:
#: ``adaptive`` images fill a ``<modality>`` directory of their own and are
#: recognised by :func:`classify_image_directory`. The other two are loose files
#: sharing a directory with whatever else Autolamella wrote there, and are
#: recognised by :func:`classify_loose_image`.
Series = Literal["adaptive", "reference", "alignment"]

#: The series, in display order.
SERIES: Final[tuple[Series, ...]] = ("adaptive", "reference", "alignment")

#: The series whose images sit loose in a directory rather than filling one.
#: These are the two :func:`classify_loose_image` can return, and the two that
#: need a name invented for them before their labels have somewhere to go.
LOOSE_SERIES: Final[tuple[Series, ...]] = ("reference", "alignment")

#: The series whose images are expected to be labelled, so that an unlabelled
#: one is worth reporting. Everything outside this is indexed and countable but
#: nobody is told it has no labels yet -- there are 5,237 such images in the
#: reference collection, and saying so once per set drowns the findings that
#: need acting on. A set that *does* have a label directory is reported on
#: normally whatever its series, so starting to label one needs no change here.
LABELS_EXPECTED: Final[frozenset[Series]] = frozenset({"adaptive"})

#: What to call each series where there is room to say it properly.
SERIES_LABELS: Final[dict[Series, str]] = {
    "adaptive": "Adaptive Polishing",
    "reference": "reference",
    "alignment": "alignment",
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
    """The modality and variant encoded in an image directory's name.

    Attributes:
        modality: ``"SEM"`` or ``"FIB"``.
        variant: The imaging-condition suffix, or ``None`` for the base
            directory. Preserved with the case used on disk, since it is an
            opaque display and grouping key.
    """

    modality: str
    variant: str | None

    @property
    def display_variant(self) -> str:
        """Return the variant for display, naming the unsuffixed case."""
        return self.variant if self.variant is not None else "base"


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
        modality: ``"SEM"`` or ``"FIB"``. Always known: an image whose beam
            neither its name nor its header states is reported rather than put
            in a set, since nothing downstream could say which beam it is.
        variant: The imaging-condition suffix, or ``None`` for the base
            directory. Always ``None`` for a loose set, which has no image
            directory to carry one.
        series: What the images were taken for.
        image_names: The filenames belonging to this set, or ``None`` where the
            set is the whole directory. Read it through
            :func:`image_files_of` rather than testing it.
        labels_directory: Where this set's labels live, whether or not that
            directory exists yet.
        labels_exist: Whether *labels_directory* is present on disk.
        labels_are_shared: Whether another image set in the same stage resolves
            to the same label directory.
        labels_are_legacy: Whether the label directory was matched by a
            pre-convention name, making it a migration candidate.
    """

    directory: Path
    modality: str
    variant: str | None
    labels_directory: Path
    labels_exist: bool
    labels_are_shared: bool
    labels_are_legacy: bool
    series: Series = "adaptive"
    image_names: tuple[str, ...] | None = None

    @property
    def is_loose(self) -> bool:
        """Return whether this set is a subset of a directory it shares."""
        return self.image_names is not None

    @property
    def name(self) -> str:
        """Return what identifies this set within its directory.

        The image directory's name, or the invented one a loose set's labels
        are named after -- which is the same string
        :func:`labels_directory_name` was given, either way.
        """
        if self.is_loose:
            return loose_image_set_name(self.series, self.modality)
        return self.directory.name

    @property
    def display_name(self) -> str:
        """Return a short label such as ``'SEM'``, ``'SEM [8_fi_1]'`` or
        ``'SEM (reference)'``."""
        if self.is_loose:
            return f"{self.modality} ({SERIES_LABELS[self.series]})"
        if self.variant is None:
            return self.modality
        return f"{self.modality} [{self.variant}]"


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


def extract_image_number(image_name: str) -> int | None:
    """Return the acquisition number an Autolamella filename carries.

    Recognises ``AP_img_013`` and ``img_013``. Here with the rest of the naming
    conventions rather than beside the filters that first needed it: it is what
    matches an SEM image to the FIB image of the same acquisition, which
    :func:`~am_segmentation_manager.frames.group_by_acquisition` relies on, and
    that has to sit below the operations layer.

    Returns:
        The number, or None where the name carries none.
    """
    for pattern in (r"AP_img_(\d+)", r"img_(\d+)"):
        match = re.search(pattern, image_name, re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
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
    """Return the modality and variant *name* encodes, or None if it is not an
    image directory.

    Label, prediction, plot and state directories are rejected first, so that
    ``sem_labels`` and ``sem_fi8_labels`` are never mistaken for images of a
    variant called ``labels`` or ``fi8_labels``.
    """
    lowered = name.lower()
    if (
        is_labels_directory(name)
        or is_predictions_directory(name)
        or lowered == PLOTS_DIRECTORY
        or lowered in STATE_FOLDER_NAMES
    ):
        return None

    for modality in MODALITIES:
        prefix = modality.lower()
        if lowered == prefix:
            return ImageDirectory(modality=modality, variant=None)
        if lowered.startswith(f"{prefix}_"):
            variant = name[len(prefix) + 1:]
            if not variant or variant.lower() in RESERVED_VARIANTS:
                logger.debug("Ignoring image directory with reserved variant: %s", name)
                return None
            return ImageDirectory(modality=modality, variant=variant)
    return None


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
_ALIGNMENT_PREFIXES: Final[tuple[str, ...]] = ("beam_shift_alignment",)

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
    them land on the same second and need something to break the tie -- in the
    reference collection 11 of 121 runs would otherwise read ``final_image``
    before the last ``new_image``. The sequence is fixed by the names
    Autolamella writes, which is what makes this a rule rather than a guess.

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
        ValueError: If *series* is not one of :data:`LOOSE_SERIES`. Adaptive
            Polishing images fill a directory that already has a name, so
            asking for a loose one means the caller has lost track of which
            kind of set it holds.
    """
    if series not in LOOSE_SERIES:
        raise ValueError(
            f"Not a loose series: {series!r}; expected one of {list(LOOSE_SERIES)}"
        )
    return f"{series}_{modality.lower()}"


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


def candidate_labels_directory_names(image_directory_name: str, modality: str) -> list[str]:
    """Return the label directory names to try, most specific first.

    Order: the private directory for this image directory, the shared directory
    for the modality, then legacy names. For a base image directory the first
    two coincide, which is intended -- ``sem/`` uses ``sem_labels/`` like every
    other ``sem*`` directory that has no private labels of its own.
    """
    names = [labels_directory_name(image_directory_name), shared_labels_directory_name(modality)]
    names.extend(
        legacy for legacy, legacy_modality in LEGACY_LABEL_DIRECTORIES.items()
        if legacy_modality == modality
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


def resolve_labels_directory(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
) -> tuple[Path, bool, bool]:
    """Return where an image directory's labels live.

    Args:
        image_directory: The image directory whose labels are wanted.
        modality: ``"SEM"`` or ``"FIB"``.
        siblings: Existing sibling directories, keyed by lowercased name.
            Matching through this mapping rather than :meth:`Path.exists` keeps
            resolution case-insensitive on every platform, which matters for
            the legacy ``labels_FIB``.

    Returns:
        A tuple of ``(directory, exists, is_legacy)``. When no candidate
        exists, the private directory name is returned with ``exists=False``:
        that is where a label would be written.
    """
    for name in candidate_labels_directory_names(image_directory.name, modality):
        existing = siblings.get(name.lower())
        if existing is not None:
            return existing, True, name.lower() in LEGACY_LABEL_DIRECTORIES
    return image_directory.parent / labels_directory_name(image_directory.name), False, False


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


def prediction_precedence(prediction_path: Path) -> int:
    """Return how strongly a prediction claims to be the one converted.

    Lower wins. A prediction this package made outranks the segmentation
    pipeline's, because it was asked for deliberately and is therefore the more
    current of the two; the pipeline's ``raw`` output is what the milling run
    happened to leave behind. Stated as an ordering rather than left to the
    order directories are listed in, since which file becomes a label is not a
    thing to decide by accident.
    """
    kinds = {part.lower() for part in prediction_path.parts[:-1]}
    if OWN_PREDICTION_KIND in kinds:
        return 0
    return 2 if POST_PROCESSED_KIND in kinds else 1


def candidate_predictions_directory_names(image_directory_name: str, modality: str) -> list[str]:
    """Return the prediction directory names to try, most specific first.

    Mirrors :func:`candidate_labels_directory_names`. Predictions are written by
    the external segmentation pipeline, so the legacy ``predictions`` and
    ``predictions_fib`` names stay first-class rather than being treated as
    migration candidates.
    """
    names = [
        predictions_directory_name(image_directory_name),
        f"{modality.lower()}{PREDICTIONS_SUFFIX}",
    ]
    names.extend(
        legacy for legacy, legacy_modality in LEGACY_PREDICTION_DIRECTORIES.items()
        if legacy_modality == modality
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


def resolve_predictions_directory(
    image_directory: Path,
    modality: str,
    siblings: Mapping[str, Path],
) -> tuple[Path, bool, bool]:
    """Return where an image directory's predictions live.

    The counterpart of :func:`resolve_labels_directory`, and shared by reading
    and writing. A directory shared with another image set is accepted, because
    only one image of a label group is ever predicted -- see
    :mod:`..operations.predict` -- so two conditions cannot claim one filename.

    Args:
        image_directory: The image directory whose predictions are wanted.
        modality: ``"SEM"`` or ``"FIB"``.
        siblings: Existing sibling directories, keyed by lowercased name, as
            :func:`subdirectories_by_name` returns them. Matching through this
            rather than :meth:`Path.exists` keeps resolution case-insensitive,
            which the legacy ``predictions_FIB`` needs.

    Returns:
        A tuple of ``(directory, exists, is_legacy)``. When no candidate
        exists, the private directory name is returned with ``exists=False``:
        that is where a prediction would be written.
    """
    for name in candidate_predictions_directory_names(image_directory.name, modality):
        existing = siblings.get(name.lower())
        if existing is not None:
            return existing, True, name.lower() in LEGACY_PREDICTION_DIRECTORIES
    private = predictions_directory_name(image_directory.name)
    return image_directory.parent / private, False, False


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

def build_image_sets(stage_directory: Path) -> list[ImageSet]:
    """Return the image sets in a stage directory, ordered for display.

    Adaptive Polishing sets only -- the images loose in the same directory are
    :func:`build_loose_image_sets`'s. Keeping them apart is what stops
    discovery mistaking a lamella full of reference images for a stage; see
    :func:`~am_segmentation_manager.discovery.find_stage_directories`.

    One of the few functions in this module that lists a directory. Assembling
    an image set requires knowing which sibling directories exist, and doing it
    here keeps the convention in one place rather than duplicating it in the
    scanner and the survey.

    Ordering is by modality (SEM before FIB), then base variant first, then
    variant name, so output is stable across runs.
    """
    siblings = subdirectories_by_name(stage_directory)

    classified: list[tuple[Path, ImageDirectory, Path, bool, bool]] = []
    for directory in siblings.values():
        image_directory = classify_image_directory(directory.name)
        if image_directory is None:
            continue
        labels_directory, labels_exist, labels_are_legacy = resolve_labels_directory(
            directory, image_directory.modality, siblings
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


def _image_set_sort_key(image_set: ImageSet) -> tuple[int, int, str]:
    """Order image sets by modality, then base variant, then variant name."""
    modality_order = MODALITIES.index(image_set.modality) if image_set.modality in MODALITIES else len(MODALITIES)
    return (modality_order, image_set.variant is not None, (image_set.variant or "").lower())


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
        unresolved: Filenames classified into a series, but whose beam neither
            the name nor *resolve_modality* could establish.
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
        A :class:`LooseImageSets` holding the sets and everything that did not
        fit into one.
    """
    siblings = subdirectories_by_name(directory)
    grouped: dict[tuple[Series, str], list[str]] = {}
    unrecognised: list[str] = []
    unresolved: list[str] = []

    for path in image_files(directory):
        if is_ignored_image_name(path.name):
            continue
        classified = classify_loose_image(path.name, in_alignment_run=in_alignment_run)
        if classified is None:
            unrecognised.append(path.name)
            continue
        modality = classified.modality
        if modality is None and resolve_modality is not None:
            modality = resolve_modality(path)
        if modality is None:
            unresolved.append(path.name)
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

def _list_directory(directory: Path) -> list[Path]:
    """Return the entries in *directory*, treating unreadable ones as empty.

    A directory that does not exist is silently empty: callers routinely ask
    about optional locations such as state subdirectories. Anything else --
    a permissions problem, or a cloud-storage placeholder that has not synced --
    is warned about but still does not raise, so one bad directory cannot abort
    a survey of an entire collection.
    """
    try:
        return sorted(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        logger.warning("Could not read directory %s: %s", directory, exc)
        return []


def subdirectories_by_name(directory: Path) -> dict[str, Path]:
    """Return *directory*'s subdirectories keyed by lowercased name.

    Hidden directories are skipped.
    """
    result: dict[str, Path] = {}
    for entry in _list_directory(directory):
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError as exc:  # pragma: no cover - transient filesystem error
            logger.warning("Could not inspect %s: %s", entry, exc)
            continue
        result[entry.name.lower()] = entry
    return result


def image_files(directory: Path) -> list[Path]:
    """Return the image files directly inside *directory*, sorted by name."""
    return [
        entry for entry in _list_directory(directory)
        if entry.suffix.lower() in IMAGE_EXTENSIONS and _is_file(entry)
    ]


def image_files_by_name(directory: Path) -> dict[str, Path]:
    """Return the image files directly in *directory*, keyed by filename.

    The filename is the identity an image and its label share. State
    subdirectories are **not** looked in: a file still sitting in one belongs
    to a dataset that has not been migrated, and reading it here would report
    it as selected. ``inspect-layout`` says so instead.
    """
    return {path.name: path for path in image_files(directory)}


def _is_file(path: Path) -> bool:
    """Return whether *path* is a regular file, treating errors as 'no'."""
    try:
        return path.is_file()
    except OSError as exc:  # pragma: no cover - transient filesystem error
        logger.warning("Could not inspect %s: %s", path, exc)
        return False
