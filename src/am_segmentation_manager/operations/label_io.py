"""Shared helpers for reading and writing label images on disk.

Centralises the optional-dependency check for the image-processing stack
(tifffile/numpy/cv2), the compact label-writing format used by both
prediction-to-label conversion (:mod:`predictions`) and existing-label
recompression (:mod:`label_compression`), and the question every caller that
treats a label as finished work has to ask: is there anything in it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..display import is_empty_label
from ..history import pixel_digest
from ..io.image import has_image_stack_support, to_2d
from ..layout import (
    resolve_labels_directory,
    subdirectories_by_name,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
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
    "changed_fraction",
    "classify_label",
    "classify_labels",
    "has_image_stack_support",
    "has_compression_support",
    "is_label_file_empty",
    "label_destination",
    "label_fields",
    "label_state",
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

    import tifffile


    try:
        array = tifffile.imread(str(path))
    except Exception as exc:  # noqa: BLE001 - an unreadable label must not raise here
        logger.warning("Could not read label %s: %s", path, exc)
        return None
    return is_empty_label(to_2d(array))


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
    # Imported here rather than at module level: operations.csv imports this
    # module, and label_measure resolves the same cycle the same way.
    from .csv import find_matching_label

    label = find_matching_label(record)
    return classify_label(label, is_label_file_empty(label) if label else None)


def label_fields(array) -> dict[str, object]:
    """Return what a history entry records about a label's contents.

    Gathered in one place so every entry describing a label describes it the
    same way: what the pixels are, how they are stored, how big they are, and
    whether there is anything in them at all.
    """

    plane = to_2d(array)
    return {
        "digest": pixel_digest(array),
        "dtype": str(plane.dtype),
        "shape": [int(size) for size in plane.shape],
        "empty": bool(is_empty_label(plane)),
    }


def changed_fraction(before, after) -> float | None:
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
    state_of: Callable[[ImageRecord], str] = label_state,
    *,
    states: Sequence[str] = (MISSING, EMPTY, UNREADABLE),
) -> LabelReadiness:
    """Return the records of *records* whose label is in one of *states*.

    The check behind marking a set corrected, and behind exporting one. Both
    ask the same question -- is there actually an annotation here -- so they
    ask it through one function rather than each walking the labels itself.

    Args:
        state_of: How to read one record's label state. Defaults to opening
            the file; a caller holding a cache passes its own lookup, which is
            what keeps this off the disk on every refresh.
        states: Which states to collect. Narrow it to leave out a finding a
            particular caller has no action for.

    Returns:
        The records found, grouped by what was wrong. Order within each group
        follows *records*.
    """
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
    except Exception:
        return False


def save_label_image(mask, output_path: Path, original_dtype) -> None:
    """Write a discrete-label mask to *output_path* in a compact encoding.

    Label masks hold a small number of discrete values (e.g. 0-4), which
    LZW compression shrinks dramatically compared to the uncompressed TIFFs
    proofreading software typically writes. This preserves *original_dtype*
    (``uint16`` or ``uint8``) instead of upcasting, and writes via OpenCV
    with LZW compression when imagecodecs is installed (required to decode
    LZW TIFFs via tifffile); otherwise it falls back to an uncompressed
    write so the output always stays readable with whatever stack is
    present.

    Args:
        mask: 2D array of discrete label values.
        output_path: Destination path; parent directories are created if
            needed.
        original_dtype: The bit-depth to preserve -- ``numpy.uint16`` keeps
            16-bit output, anything else is written as 8-bit.
    """
    import cv2
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = mask.astype(np.uint16 if original_dtype == np.uint16 else np.uint8)
    if has_compression_support():
        cv2.imwrite(str(output_path), mask, [cv2.IMWRITE_TIFF_COMPRESSION, _TIFF_COMPRESSION_LZW])
    else:
        logger.warning(
            "imagecodecs not installed; writing %s uncompressed. Install the "
            "'all' extra for smaller label files.",
            output_path,
        )
        cv2.imwrite(str(output_path), mask, [cv2.IMWRITE_TIFF_COMPRESSION, _TIFF_COMPRESSION_NONE])
