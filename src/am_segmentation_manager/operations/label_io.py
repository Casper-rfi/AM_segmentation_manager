"""Shared helpers for reading and writing label images on disk.

Centralises the optional-dependency check for the image-processing stack
(tifffile/numpy/cv2), the compact label-writing format used by both
prediction-to-label conversion (:mod:`predictions`) and existing-label
recompression (:mod:`label_compression`), and the question every caller that
treats a label as finished work has to ask: is there anything in it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.image import has_image_stack_support

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
    "has_image_stack_support",
    "has_compression_support",
    "is_label_file_empty",
    "label_state",
    "save_label_image",
]


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

    from ..display import is_empty_label
    from ..io.image import to_2d

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
    if label is None:
        return MISSING
    empty = is_label_file_empty(label)
    if empty is None:
        return UNREADABLE
    return EMPTY if empty else ANNOTATED


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
