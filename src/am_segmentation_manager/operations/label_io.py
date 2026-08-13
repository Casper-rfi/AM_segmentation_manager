"""Shared helpers for writing label images to disk.

Centralises the optional-dependency check for the image-processing stack
(tifffile/numpy/cv2) and the compact label-writing format used by both
prediction-to-label conversion (:mod:`predictions`) and existing-label
recompression (:mod:`label_compression`).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenCV TIFF compression codes (see opencv2/imgcodecs.hpp).
_TIFF_COMPRESSION_NONE = 1
_TIFF_COMPRESSION_LZW = 5


def has_image_stack_support() -> bool:
    """Return whether the optional tifffile/numpy/cv2 stack is installed."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import tifffile  # noqa: F401
        return True
    except Exception:
        return False


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
