"""Shared helpers for reading and writing label images on disk.

Centralises the optional-dependency check for the image-processing stack
(tifffile/numpy/cv2), the compact label-writing format used by both
prediction-to-label conversion (:mod:`predictions`) and existing-label
recompression (:mod:`label_compression`), and the question every caller that
treats a label as finished work has to ask: is there anything in it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..display import is_empty_label
from ..history import pixel_digest
from ..io.image import has_image_stack_support, label_plane, read_array, read_tiff, to_2d
from ..parallel import map_in_processes
from .label_groups import find_matching_label

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    import numpy as np
    from numpy.typing import DTypeLike

    from ..models import ImageRecord

logger = logging.getLogger(__name__)

# OpenCV TIFF compression codes (see opencv2/imgcodecs.hpp).
_TIFF_COMPRESSION_NONE = 1
_TIFF_COMPRESSION_LZW = 5

#: What an image's label turned out to be. One vocabulary, because three
#: callers report it -- the per-image CSV writes it as a column, the Export
#: tab filters and counts by it -- and a fourth spelling of "there is no label"
#: would be a fourth thing to keep in step.
ANNOTATED = "annotated"
EMPTY = "empty"
MISSING = "missing"
UNREADABLE = "unreadable"

# Re-exported so existing callers keep importing the dependency check from here.
__all__ = [
    "ANNOTATED",
    "EMPTY",
    "MISSING",
    "UNREADABLE",
    "LabelReadiness",
    "LabelsRead",
    "changed_fraction",
    "classify_label",
    "classify_labels",
    "has_image_stack_support",
    "has_compression_support",
    "is_label_file_empty",
    "label_destination",
    "label_digest",
    "label_fields",
    "label_state",
    "read_emptiness",
    "read_label",
    "read_labels_of",
    "replacement_fields",
    "save_label_image",
]


def label_destination(record: ImageRecord) -> Path:
    """Return where *record*'s label is, or where one would be written.

    An existing label is answered with its own path, so nothing computes a
    fresh destination that would name a second label beside the one really
    there. Otherwise the answer is **read off the record**, which carries the
    directory its image set resolved to.

    Not re-derived from the image's path: how far a label directory sits from
    an image depends on the shape of its set, and the derivation only ever
    worked for an Adaptive Polishing image sitting one level below its stage in
    a directory naming a modality. A reference image sits directly in a
    directory naming a lamella, so deriving put its label in a
    ``<lamella>_labels`` outside the lamella -- and a proofread label is the
    one thing in this package nothing may misplace.

    Asked by everything that writes a label: prediction conversion, and
    proofreading a label that does not exist yet.

    Raises:
        ValueError: If the record does not say where its label lives, which
            means it was built by something that has not been taught to.
    """
    if record.labels:
        return record.labels[0]
    if record.labels_directory is None:
        raise ValueError(
            f"{record.image_path} does not record where its label lives"
        )
    return record.labels_directory / record.image_path.name


def is_label_file_empty(path: Path) -> bool | None:
    """Return whether a label file contains nothing but background.

    The external segmentation software writes an empty label for every image
    in its input folder, so the *existence* of a label file is not evidence
    that an image has been annotated. Anything that treats a label as finished
    work -- a training export, a progress display -- has to look inside.

    Returns:
        True if the file holds only background, False if it holds any
        annotation, and None when the answer cannot be determined: the image
        stack is not installed, the file is missing, or it could not be read.
        None is deliberately distinct from True so that an unreadable label is
        never mistaken for an empty one.
    """
    if not has_image_stack_support():
        logger.debug("Cannot inspect %s: tifffile/numpy/cv2 not installed", path)
        return None



    try:
        array = read_tiff(path)
    except Exception as exc:  # noqa: BLE001 - an unreadable label must not raise here
        logger.warning("Could not read label %s: %s", path, exc)
        return None
    # tifffile answers some damaged files with no pixels rather than an error.
    if array.size == 0:
        logger.warning("Could not read label %s: it holds no pixels", path)
        return None
    try:
        return is_empty_label(label_plane(array))
    except ValueError as error:
        logger.warning("Could not read label %s: %s", path, error)
        return None


#: Fewer labels than this are read in this process, reading them costing less
#: than starting the worker processes would.
LABEL_POOL_FILES = 16


def read_emptiness(paths: Iterable[Path]) -> dict[Path, bool | None]:
    """Return :func:`is_label_file_empty`'s answer for each of *paths*, each read once.

    Read in worker processes where there are enough of them.
    """
    paths = list(dict.fromkeys(paths))
    if not has_image_stack_support():
        logger.debug("Cannot inspect %d label(s): tifffile/numpy/cv2 not installed", len(paths))
        return dict.fromkeys(paths)
    answers = map_in_processes(
        is_label_file_empty, paths, chunk_size=1, min_items=LABEL_POOL_FILES
    )
    return dict(zip(paths, answers))


@dataclass(frozen=True)
class LabelsRead:
    """What a set of labels was found to hold, so a batch is read once for every question.

    A label it was not given is read when asked about.
    """

    #: Each label's :func:`is_label_file_empty` answer.
    emptiness: Mapping[Path, bool | None]

    def is_empty(self, label: Path) -> bool | None:
        """Return :func:`is_label_file_empty`'s answer for *label*."""
        return self.emptiness[label] if label in self.emptiness else is_label_file_empty(label)

    def state(self, record: ImageRecord) -> str:
        """Return :func:`label_state`'s answer for *record*."""
        label = find_matching_label(record)
        return classify_label(label, self.is_empty(label) if label is not None else None)


def read_labels_of(records: Iterable[ImageRecord]) -> LabelsRead:
    """Return what the labels of *records* hold, each read once and together."""
    return LabelsRead(read_emptiness(
        label for record in records if (label := find_matching_label(record)) is not None
    ))


def label_state(record: ImageRecord) -> str:
    """Return what *record*'s label turned out to be.

    A label existing is not evidence of annotation: the segmentation software
    writes an empty one for every image it is pointed at. Anything treating a
    label as finished work has to look inside, which is why this is the unit
    the Export tab filters and counts on rather than "has a label file".

    Returns:
        One of :data:`ANNOTATED`, :data:`EMPTY`, :data:`MISSING` or
        :data:`UNREADABLE`. Unreadable is kept apart from empty so a file this
        package could not open is never counted as a blank one.
    """
    label = find_matching_label(record)
    return classify_label(label, is_label_file_empty(label) if label else None)


def label_fields(array: np.ndarray) -> dict[str, object]:
    """Return what a history entry records about a label's contents.

    Gathered in one place so every entry describing a label describes it the
    same way: what the pixels are, how they are stored, how big they are, and
    whether there is anything in them at all.
    """

    plane = label_plane(array)
    return {
        "digest": pixel_digest(array),
        "dtype": str(plane.dtype),
        "shape": [int(size) for size in plane.shape],
        "empty": bool(is_empty_label(plane)),
    }


def changed_fraction(before: np.ndarray, after: np.ndarray) -> float | None:
    """Return the share of pixels differing between two versions of a label.

    Computed while both are in memory, because it cannot be had cheaply
    afterwards, and it is what makes a list of versions readable at a glance.

    Returns:
        The fraction, rounded, or None where the two do not have the same
        shape -- which is every conversion, predictions being written at model
        resolution. A percentage across a resize would be arithmetic on two
        different pictures, so the resize is reported instead.
    """

    first, second = to_2d(before), to_2d(after)
    if first.shape != second.shape:
        return None
    if not first.size:
        return 0.0
    return round(float((first != second).mean()), 6)


def read_label(path: Path) -> np.ndarray | None:
    """Return the pixels of the label at *path*, or None where it will not open.

    Logged rather than raised: one damaged label costs an entry its description,
    never the batch it is part of.
    """
    try:
        return read_array(path)
    except Exception as error:  # noqa: BLE001 - one unreadable label must not stop a batch
        logger.warning("Could not read label %s: %s", path, error)
        return None


def label_digest(path: Path | None) -> str | None:
    """Return the pixel digest of the label at *path*, or None where it has none to give."""
    if path is None or not path.is_file():
        return None
    pixels = read_label(path)
    if pixels is None:
        return None
    try:
        return str(label_fields(pixels)["digest"])
    except ValueError as error:
        logger.warning("Could not describe label %s: %s", path, error)
        return None


def replacement_fields(
    destination: Path,
    previous: np.ndarray | None,
    written: np.ndarray,
    archived: str | None,
) -> dict[str, object]:
    """Return what a history entry records about the version a write replaced.

    Args:
        destination: The label written, named in any warning.
        previous: The pixels replaced, or None where nothing was.
        written: The pixels that replaced them.
        archived: Where the replaced version was kept, or None.

    Returns:
        ``parent_digest``, ``changed_fraction`` and ``archived_as``, each only
        where it can be said; empty for a label written where none was.
    """
    fields: dict[str, object] = {}
    if previous is not None:
        try:
            fields["parent_digest"] = label_fields(previous)["digest"]
            fields["changed_fraction"] = changed_fraction(previous, written)
        except ValueError as error:
            logger.warning("Could not describe what %s replaced: %s", destination, error)
    if archived is not None:
        fields["archived_as"] = archived
    return fields


def classify_label(label: Path | None, empty: bool | None) -> str:
    """Return the state a label path and its emptiness amount to.

    Split from :func:`label_state` so a caller that has already opened the
    label -- reading a whole collection once, rather than a file at a time --
    reaches the same vocabulary without a second read and without a second
    copy of the rules.

    Args:
        label: The label file, or None where there is none.
        empty: What :func:`is_label_file_empty` said of it, None meaning it
            could not be read.
    """
    if label is None:
        return MISSING
    if empty is None:
        return UNREADABLE
    return EMPTY if empty else ANNOTATED


@dataclass(frozen=True)
class LabelReadiness:
    """The records of a set whose label is not an annotation.

    What "corrected" claims is that somebody proofread a label. These three
    groups are the ways that claim can be false, kept apart because they need
    different actions: annotate it, find where the label went, or work out why
    the file will not open.

    Attributes:
        missing: Records with no label file at all.
        empty: Records whose label holds nothing but background.
        unreadable: Records whose label could not be opened, so nothing is
            known about what is in it. Never folded into *empty*: reporting a
            file as blank that nobody managed to read is a guess presented as
            a fact.
    """

    missing: tuple[ImageRecord, ...] = ()
    empty: tuple[ImageRecord, ...] = ()
    unreadable: tuple[ImageRecord, ...] = ()

    def __bool__(self) -> bool:
        """Return whether anything at all was found."""
        return bool(self.missing or self.empty or self.unreadable)

    @property
    def records(self) -> tuple[ImageRecord, ...]:
        """Return every record found, missing first."""
        return (*self.missing, *self.empty, *self.unreadable)

    def summary(self) -> str:
        """Return a one-line count of what was found, or an empty string."""
        parts = [
            (len(self.missing), "with no label"),
            (len(self.empty), "with an empty label"),
            (len(self.unreadable), "with an unreadable label"),
        ]
        found = [f"{count} {text}" for count, text in parts if count]
        return ", ".join(found)


def classify_labels(
    records: Iterable[ImageRecord],
    state_of: Callable[[ImageRecord], str] | None = None,
    *,
    states: Sequence[str] = (MISSING, EMPTY, UNREADABLE),
) -> LabelReadiness:
    """Return the records of *records* whose label is in one of *states*.

    The check behind marking a set corrected, and behind exporting one. Both
    ask the same question -- is there actually an annotation here -- so they
    ask it through one function rather than each walking the labels itself.

    Args:
        state_of: How to read one record's label state. None opens every
            label of *records* once, together; a caller holding a cache passes
            its own lookup, which is what keeps this off the disk on every
            refresh.
        states: Which states to collect. Narrow it to leave out a finding a
            particular caller has no action for.

    Returns:
        The records found, grouped by what was wrong. Order within each group
        follows *records*.
    """
    records = list(records)
    if state_of is None:
        state_of = read_labels_of(records).state
    found: dict[str, list[ImageRecord]] = {MISSING: [], EMPTY: [], UNREADABLE: []}
    for record in records:
        state = state_of(record)
        if state in states and state in found:
            found[state].append(record)
    return LabelReadiness(
        missing=tuple(found[MISSING]),
        empty=tuple(found[EMPTY]),
        unreadable=tuple(found[UNREADABLE]),
    )


def has_compression_support() -> bool:
    """Return whether imagecodecs is installed.

    LZW-compressed TIFFs written by :func:`save_label_image` can only be
    decoded by tifffile (and therefore by the rest of this package, and by
    downstream training code) if imagecodecs is available. Without it,
    labels are written uncompressed instead of silently producing files
    that can't be read back.
    """
    try:
        import imagecodecs  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - broken counts as absent: neither decodes LZW
        return False


def save_label_image(mask: np.ndarray, output_path: Path, original_dtype: DTypeLike) -> None:
    """Write a discrete-label mask to *output_path* in a compact encoding.

    Label masks hold a small number of discrete values (e.g. 0-4), which
    LZW compression shrinks dramatically compared to the uncompressed TIFFs
    proofreading software typically writes. This preserves *original_dtype*
    (``uint16`` or ``uint8``) instead of upcasting, and writes via OpenCV
    with LZW compression when imagecodecs is installed (required to decode
    LZW TIFFs via tifffile); otherwise it falls back to an uncompressed
    write so the output always stays readable with whatever stack is
    present.

    **The file holds exactly the pixels asked for, or this raises**: it is read
    back before returning, so a caller writing to a temporary never renames a
    wrong or empty file over a label.

    Args:
        mask: 2D array of discrete label values.
        output_path: Destination path; parent directories are created if
            needed.
        original_dtype: The bit-depth to preserve -- ``numpy.uint16`` keeps
            16-bit output, anything else is written as 8-bit.

    Raises:
        ValueError: If *mask* is not a single 2D plane, or holds a value the
            written bit-depth cannot represent exactly.
        OSError: If OpenCV could not write the file, or what it wrote does not
            read back as *mask*.
    """
    import cv2
    import numpy as np

    if mask.ndim != 2:
        raise ValueError(
            f"{output_path.name}: a label is one 2D plane, not an array of shape {mask.shape}"
        )
    # A value the cast cannot represent is what the comparison below catches.
    with np.errstate(invalid="ignore"):
        stored: np.ndarray = mask.astype(np.uint16 if original_dtype == np.uint16 else np.uint8)
    if not np.array_equal(stored, mask):
        raise ValueError(
            f"{output_path.name}: holds values a {stored.dtype} label cannot represent "
            f"exactly (a {mask.dtype} array)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if has_compression_support():
        compression = _TIFF_COMPRESSION_LZW
    else:
        logger.warning(
            "imagecodecs not installed; writing %s uncompressed. Install the "
            "'gui' extra for smaller label files.",
            output_path,
        )
        compression = _TIFF_COMPRESSION_NONE
    # OpenCV reports a failed write by returning False rather than raising.
    if not cv2.imwrite(str(output_path), stored, [cv2.IMWRITE_TIFF_COMPRESSION, compression]):
        raise OSError(f"OpenCV could not write {output_path}")

    try:
        written = read_tiff(output_path)
    except Exception as error:  # any failure to decode is the same answer
        raise OSError(f"{output_path} did not read back after writing: {error}") from error
    if written.shape != stored.shape or not np.array_equal(written, stored):
        raise OSError(f"{output_path} did not read back as the label written")
