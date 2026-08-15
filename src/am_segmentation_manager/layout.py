"""Filesystem naming conventions for Autolamella and legacy datasets.

This module is the single source of truth for *what directories are called* and
*how they relate to each other*. Every other module should resolve directory
names through the functions here rather than embedding string literals, so that
supporting a new dataset layout is a change in one place.

The conventions are:

* An image directory is named ``<modality>`` or ``<modality>_<variant>``, where
  modality is ``sem`` or ``fib`` (matched case-insensitively, so the legacy
  ``SEM_8_fi_1`` works). The variant is an **opaque** string -- it is never
  parsed, so ``sem_5kv``, ``sem_defocus`` and ``sem_lowmag`` need no code
  changes. Authoritative imaging parameters come from the image headers.
* A label directory is named ``<image-directory-name>_labels`` and sits beside
  the image directory. ``<modality>_labels`` therefore acts as the *shared*
  label directory for a modality, and ``<modality>_<variant>_labels`` is
  *private* to one variant. Sharing is a consequence of how many image
  directories resolve to the same label directory, and is computed rather than
  declared. Resolution is most-specific-first: ``<image-dir>_labels`` if it
  exists, else the shared ``<modality>_labels``, else a legacy name, else
  ``<image-dir>_labels`` created on first write.
* Workflow state is physical location: a file in a ``deselected/`` or
  ``corrected/`` subdirectory is in that state, and anywhere else is selected.

Legacy names (``labels``, ``labels_fib``, ``labels_FIB``, ``predictions_fib``)
remain readable indefinitely; only newly written directories use the current
convention.

Beyond :func:`build_image_sets`, which necessarily lists one directory, the
functions here perform no filesystem access, which keeps the conventions cheap
to test in isolation.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

logger = logging.getLogger(__name__)

#: The imaging modalities this package indexes, in display order.
MODALITIES: Final[tuple[str, ...]] = ("SEM", "FIB")

#: A file's workflow state, which is determined by the directory it sits in.
RecordState = Literal["selected", "deselected", "corrected"]

#: State to the subdirectory name that holds it. ``None`` means the base
#: directory itself, so the selected state adds no nesting.
STATE_DIRECTORIES: Final[dict[str, str | None]] = {
    "selected": None,
    "deselected": "deselected",
    "corrected": "corrected",
}

#: The subdirectory names that carry a non-default state.
STATE_FOLDER_NAMES: Final[frozenset[str]] = frozenset(
    name for name in STATE_DIRECTORIES.values() if name is not None
)

#: The inverse of :data:`STATE_DIRECTORIES`: subdirectory name to the state it
#: represents. A directory not listed here means the selected state.
FOLDER_STATES: Final[dict[str, str]] = {
    folder: state for state, folder in STATE_DIRECTORIES.items() if folder is not None
}

LABELS_SUFFIX: Final[str] = "_labels"
PREDICTIONS_SUFFIX: Final[str] = "_predictions"
PLOTS_DIRECTORY: Final[str] = "plots"

#: Subdirectories of a predictions directory holding different prediction sets.
PREDICTION_KINDS: Final[tuple[str, ...]] = ("raw", "clean")

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


# ---------------------------------------------------------------------------
# Workflow state locations
# ---------------------------------------------------------------------------

def state_of_path(path: Path) -> RecordState:
    """Return the workflow state implied by the directory *path* sits in.

    The state folder a file sits in *is* its state; there is no separate store
    to consult.
    """
    return FOLDER_STATES.get(path.parent.name, "selected")  # type: ignore[return-value]


def base_path(path: Path) -> Path:
    """Return *path* with any state subdirectory stripped.

    ``sem/deselected/img.tif`` and ``sem/img.tif`` both resolve to
    ``sem/img.tif``: the shared identity used to find a file's other possible
    state locations.
    """
    if path.parent.name in STATE_FOLDER_NAMES:
        return path.parent.parent / path.name
    return path


def state_path(path: Path, state: str) -> Path:
    """Return where *path* would live in *state*."""
    if state not in STATE_DIRECTORIES:
        raise ValueError(f"Unknown workflow state: {state}")
    base = base_path(path)
    folder = STATE_DIRECTORIES[state]
    return base if folder is None else base.parent / folder / base.name


def state_directories(directory: Path) -> list[Path]:
    """Return *directory* and its state subdirectories, selected state first."""
    return [directory, *(directory / name for name in sorted(STATE_FOLDER_NAMES))]


def state_variants(path: Path) -> list[Path]:
    """Return every location *path* could occupy, one per workflow state.

    The selected location comes first. Used to find where a file actually is
    when the recorded path may be stale.
    """
    base = base_path(path)
    return [base, *(base.parent / folder / base.name for folder in sorted(STATE_FOLDER_NAMES))]


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
    """Return image files in *directory* and its state subdirectories.

    Keyed by filename, which is the identity an image and its label share. A
    file present in more than one state directory is a fault the caller should
    report; the first found wins here so that counting cannot raise.
    """
    found: dict[str, Path] = {}
    for candidate in state_directories(directory):
        for path in image_files(candidate):
            found.setdefault(path.name, path)
    return found


def _is_file(path: Path) -> bool:
    """Return whether *path* is a regular file, treating errors as 'no'."""
    try:
        return path.is_file()
    except OSError as exc:  # pragma: no cover - transient filesystem error
        logger.warning("Could not inspect %s: %s", path, exc)
        return False
