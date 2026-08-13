"""Reading image files into arrays.

Centralises the optional image-processing dependency check and the mechanics of
turning a file on disk into a numpy array, so that prediction conversion, label
compression and display preparation all read images the same way.

The heavy dependencies (numpy, tifffile, cv2) are optional extras, so they are
imported inside functions rather than at module level: importing this module
must stay safe in a minimal installation.

This module deliberately depends on nothing else in the package. Higher-level
concerns -- contrast, downscaling, overlays -- belong in
:mod:`am_segmentation_manager.display`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

TIFF_SUFFIXES = {".tif", ".tiff"}

INSTALL_HINT = (
    "Install the optional image dependencies with: "
    "pip install 'am_segmentation_manager[all]'"
)


def has_image_stack_support() -> bool:
    """Return whether the optional tifffile/numpy/cv2 stack is installed."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import tifffile  # noqa: F401
        return True
    except Exception:
        return False


def require_image_stack(operation: str) -> None:
    """Raise a helpful error if the optional image stack is unavailable.

    Args:
        operation: What the caller was trying to do, used in the message.

    Raises:
        RuntimeError: If numpy, tifffile or cv2 is not installed.
    """
    if not has_image_stack_support():
        raise RuntimeError(
            f"{operation} requires numpy, tifffile and opencv-python. {INSTALL_HINT}"
        )


def to_2d(array: "np.ndarray") -> "np.ndarray":
    """Reduce an image array to two dimensions.

    Multi-channel and multi-page files are reduced to a single plane, so that
    downstream code can assume a 2D array. Channel-first ``(C, H, W)`` and
    channel-last ``(H, W, C)`` layouts are distinguished by the size of the
    leading axis, since channel counts are small.

    Args:
        array: An image array of any dimensionality.

    Returns:
        A 2D view or slice of *array*.
    """
    import numpy as np

    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] <= 4 and array.shape[0] != array.shape[1]:
            return array[0]
        return array[..., 0]
    return np.squeeze(array)


def read_array(path: Path) -> "np.ndarray":
    """Read an image file into an array, preserving its stored dtype.

    TIFFs are read with tifffile; other formats (for example the PNG summary
    plots) are read with OpenCV. OpenCV returns channels in BGR order, so
    colour images are converted to RGB here and callers can always assume RGB.

    Args:
        path: The image file to read.

    Returns:
        The image as a numpy array: 2D for greyscale, ``(H, W, 3)`` for colour,
        ``(H, W, 4)`` when an alpha channel is present.

    Raises:
        RuntimeError: If the optional image stack is not installed.
        FileNotFoundError: If *path* does not exist.
        OSError: If the file exists but cannot be decoded.
    """
    require_image_stack(f"Reading {path.name}")
    import cv2
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if path.suffix.lower() in TIFF_SUFFIXES:
        import tifffile

        return tifffile.imread(str(path))

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError(f"Could not decode image: {path}")
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
    return np.asarray(array)
