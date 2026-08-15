"""Reading image files into arrays.

Centralises the optional dependency check and the mechanics of turning a file
into a numpy array, so prediction conversion, label compression and display
preparation all read images the same way.

numpy, tifffile and cv2 are optional extras, imported inside functions rather
than at module level: importing this module must stay safe in a minimal
installation. It depends on nothing else in the package.
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
    "In a conda environment install them with conda "
    "(numpy tifffile imagecodecs opencv), and do not also pip install "
    "'am_segmentation_manager[all]' -- conda's opencv and pip's opencv-python "
    "both provide cv2, and installing both breaks the import. Outside conda: "
    "pip install 'am_segmentation_manager[all]'"
)

#: The modules that must import for any image to be read, and the distribution
#: each comes from. cv2 is listed by import name because that is what fails.
IMAGE_STACK_MODULES = (("numpy", "numpy"), ("tifffile", "tifffile"), ("cv2", "opencv"))


def image_stack_problems() -> list[tuple[str, str]]:
    """Return ``(module, reason)`` for each dependency that will not import.

    Reporting *why* each one failed matters more than it looks. An earlier
    version caught every exception and said "not installed", which is wrong and
    misleading whenever a package is installed but broken -- the usual cause
    being two installations of cv2, one from conda's ``opencv`` and one from
    pip's ``opencv-python``. That reads as a missing dependency and sends you
    to install it again, which makes it worse.
    """
    problems: list[tuple[str, str]] = []
    for module, distribution in IMAGE_STACK_MODULES:
        try:
            __import__(module)
        except ImportError as error:
            problems.append((distribution, f"not installed ({error})"))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            problems.append(
                (distribution, f"installed but failed to import: {error!r}")
            )
    return problems


def has_image_stack_support() -> bool:
    """Return whether the optional tifffile/numpy/cv2 stack is usable."""
    return not image_stack_problems()


def require_image_stack(operation: str) -> None:
    """Raise a helpful error if the optional image stack is unavailable.

    Args:
        operation: What the caller was trying to do, used in the message.

    Raises:
        RuntimeError: If numpy, tifffile or cv2 cannot be imported, naming
            which ones and why.
    """
    problems = image_stack_problems()
    if not problems:
        return
    detail = "; ".join(f"{name}: {reason}" for name, reason in problems)
    logger.warning("%s unavailable -- %s", operation, detail)
    raise RuntimeError(f"{operation} needs {detail}. {INSTALL_HINT}")


def to_2d(array: np.ndarray) -> np.ndarray:
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


def read_array(path: Path) -> np.ndarray:
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


def read_color_array(path: Path) -> np.ndarray:
    """Read an image as 3-channel RGB, discarding any alpha channel on decode.

    Asking the decoder for three channels is materially faster than decoding
    four and compositing the alpha away afterwards, which for a large plot costs
    more than the decode itself.

    TIFFs are delegated to :func:`read_array`, since tifffile has no equivalent
    option; the caller is expected to normalise the result.

    Args:
        path: The image file to read.

    Returns:
        An ``(H, W, 3)`` RGB array for formats OpenCV handles, or whatever
        :func:`read_array` returns for a TIFF.

    Raises:
        RuntimeError: If the optional image stack is not installed.
        FileNotFoundError: If *path* does not exist.
        OSError: If the file exists but cannot be decoded.
    """
    require_image_stack(f"Reading {path.name}")
    import cv2

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if path.suffix.lower() in TIFF_SUFFIXES:
        return read_array(path)

    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if array is None:
        raise OSError(f"Could not decode image: {path}")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
