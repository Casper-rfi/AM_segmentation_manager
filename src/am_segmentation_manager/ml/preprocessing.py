"""Generation 1's preprocessing: a greyscale image as the batched tensor its model takes.

The steps are Adaptive Milling's, and ``tests/test_ml_generation.py`` holds them to
its own output bit for bit. Every prediction on disk was made through them, so a
change that fails that test changes what a generation-1 model sees.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional


def preprocess(
    image: np.ndarray, image_size: int, rgb: bool, device: str = "cpu"
) -> torch.Tensor:
    """Return *image* normalised and resized so that its longest edge is *image_size*.

    Args:
        image: A 2D greyscale array of any numeric type; it is not modified.
        image_size: Longest edge of the result, in pixels. The aspect ratio is kept.
        rgb: Whether to repeat the single channel into three.
        device: The torch device the work is done on.

    Returns:
        A ``float32`` tensor shaped ``(1, channels, height, width)``.
    """
    batch = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    batch = batch.to(device)[None, None]
    tensor = _resize(_normalise(batch), _resized_shape(batch.shape[-2:], image_size))
    return tensor.repeat(1, 3, 1, 1) if rgb else tensor


def _normalise(image: torch.Tensor) -> torch.Tensor:
    """Centre *image* on its mean, scale three standard deviations to one, and clip to [-1, 1]."""
    # correction=1 is the sample standard deviation, dividing by n - 1.
    return ((image - image.mean()) / (3 * image.std(correction=1))).clamp(-1, 1)


def _resized_shape(shape: Sequence[int], longest_edge: int) -> tuple[int, int]:
    """Return *shape* scaled so that its longer edge is *longest_edge*, rounding half to even."""
    height, width = shape
    scale = min(longest_edge / height, longest_edge / width)
    return round(height * scale), round(width * scale)


def _resize(image: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Return *image* resampled to *shape*, bicubic and antialiased; unchanged if already that shape."""
    if tuple(image.shape[-2:]) == shape:
        return image
    return functional.interpolate(
        image, size=shape, mode="bicubic", align_corners=False, antialias=True
    )
