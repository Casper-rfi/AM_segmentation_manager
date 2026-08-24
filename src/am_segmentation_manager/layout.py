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

Except for :func:`build_image_sets`, nothing here touches the filesystem.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
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
    """A directory of images acquired under a single imaging condition.

    Attributes:
        directory: The image directory itself.
        modality: ``"SEM"`` or ``"FIB"``.
        variant: The imaging-condition suffix, or ``None`` for the base
            directory.
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

    @property
    def name(self) -> str:
        """Return the image directory's name."""
        return self.directory.name

    @property
    def display_name(self) -> str:
        """Return a short label such as ``'SEM'`` or ``'SEM [8_fi_1]'``."""
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

#: Prefixes naming an alignment image outright.
_ALIGNMENT_PREFIXES: Final[tuple[str, ...]] = ("beam_shift_alignment", "ref_alignment")

#: The fixed names Autolamella writes inside an alignment run directory,
#: alongside ``new_image_<n>``. They carry no marker of their own -- a bare
#: ``final_image.tif`` could be anything -- so they are recognised only where
#: the directory says what they are. See :func:`classify_loose_image`.
_ALIGNMENT_RUN_STEMS: Final[frozenset[str]] = frozenset(
    {"reference_image", "final_image"}
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
_IGNORED_MARKERS: Final[tuple[str, ...]] = ("_centring_",)


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


def is_ignored_image_name(image_name: str) -> bool:
    """Return whether an image is deliberately not indexed.

    Separate from "no rule recognises this", which the survey reports.
    """
    return any(marker in Path(image_name).stem.lower() for marker in _IGNORED_MARKERS)


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

    if lowered.startswith(_ALIGNMENT_PREFIXES) or any(
        marker in lowered for marker in _ALIGNMENT_MARKERS
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

    This is the only function in this module that lists a directory. Assembling
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
