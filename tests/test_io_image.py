"""Tests for reading image files into arrays.

A file that exists and will not decode is an OSError whatever is wrong with it,
because that is what the callers showing a picture catch.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile

    from am_segmentation_manager.display import prepare_display_image
    from am_segmentation_manager.io.image import read_array, read_color_array


def _valid_tiff(path: Path) -> bytes:
    tifffile.imwrite(str(path), np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    return path.read_bytes()


def _damaged(tmp_path: Path, damage: str) -> Path:
    """Write a .tif that is not one, in the way *damage* names."""
    whole = _valid_tiff(tmp_path / "whole.tif")
    contents = {
        "garbage": b"not a tiff",
        "empty": b"",
        "truncated": whole[: len(whole) // 2],
        "header only": whole[:8],
    }[damage]
    path = tmp_path / "damaged.tif"
    path.write_bytes(contents)
    return path


DAMAGE = ["garbage", "empty", "truncated", "header only"]


def test_a_tiff_is_read_with_its_stored_dtype(tmp_path):
    path = tmp_path / "whole.tif"
    _valid_tiff(path)

    array = read_array(path)

    assert array.shape == (64, 64)
    assert array.dtype == np.uint16


@pytest.mark.parametrize("damage", DAMAGE)
def test_a_tiff_that_will_not_decode_is_an_oserror(tmp_path, damage):
    path = _damaged(tmp_path, damage)

    with pytest.raises(OSError, match="Could not decode image") as raised:
        read_array(path)

    assert str(path) in str(raised.value)


def test_a_tiff_that_will_not_decode_keeps_what_tifffile_said(tmp_path):
    with pytest.raises(OSError) as raised:
        read_array(_damaged(tmp_path, "garbage"))

    assert isinstance(raised.value.__cause__, tifffile.TiffFileError)


def test_a_missing_tiff_is_still_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_array(tmp_path / "absent.tif")


def test_reading_in_colour_refuses_a_damaged_tiff_the_same_way(tmp_path):
    with pytest.raises(OSError, match="Could not decode image"):
        read_color_array(_damaged(tmp_path, "garbage"))


def test_a_png_that_will_not_decode_is_an_oserror(tmp_path):
    path = tmp_path / "plot.png"
    path.write_bytes(b"not a png")

    with pytest.raises(OSError, match="Could not decode image"):
        read_array(path)


@pytest.mark.parametrize("damage", DAMAGE)
def test_a_damaged_image_cannot_be_prepared_for_display(tmp_path, damage):
    with pytest.raises(OSError):
        prepare_display_image(_damaged(tmp_path, damage))


@pytest.mark.parametrize("damage", DAMAGE)
def test_a_damaged_label_cannot_be_prepared_for_display(tmp_path, damage):
    label = _damaged(tmp_path, damage)
    image = tmp_path / "whole.tif"

    with pytest.raises(OSError):
        prepare_display_image(image, label)
