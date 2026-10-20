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
    from am_segmentation_manager.io.image import (
        planar_shape,
        read_array,
        read_color_array,
        read_tiff,
        read_tiff_shape,
        to_2d,
    )


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


@pytest.mark.parametrize(
    "written",
    ["absent.tif", "a-file.tif/absent.tif", "absent.png"],
    ids=["missing", "under-a-file", "missing-png"],
)
def test_a_missing_image_is_still_file_not_found(tmp_path, written):
    """The window shows this message, so how the absence is found must not change it."""
    (tmp_path / "a-file.tif").write_bytes(b"")
    path = tmp_path / written

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        read_array(path)


def test_a_tiff_is_opened_without_asking_first_whether_it_is_there(tmp_path, monkeypatch):
    """Opening says what asking says; asking first is a second round trip on a
    network filesystem, for every label a training run checks."""
    import os

    path = tmp_path / "label.tif"
    tifffile.imwrite(str(path), np.ones((4, 4), dtype=np.uint8))
    asked: list[str] = []
    real = os.stat

    def watched(name, *args, **kwargs):
        asked.append(os.fspath(name))
        return real(name, *args, **kwargs)

    monkeypatch.setattr(os, "stat", watched)
    read_array(path)

    assert str(path) not in asked


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


@pytest.mark.parametrize("reader", ["read_array", "is_label_file_empty"])
def test_a_tiff_is_read_without_resolving_its_path(tmp_path, monkeypatch, reader):
    """Given a name, tifffile resolves it first, asking about every directory on
    the way: on a network filesystem a round trip each, for every label read."""
    import os

    from am_segmentation_manager.operations.label_io import is_label_file_empty

    path = tmp_path / "label.tif"
    tifffile.imwrite(str(path), np.ones((4, 4), dtype=np.uint8))
    resolved: list[str] = []
    real = os.path.realpath

    def watched(name, *args, **kwargs):
        resolved.append(os.fspath(name))
        return real(name, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", watched)
    {"read_array": read_array, "is_label_file_empty": is_label_file_empty}[reader](path)

    assert resolved == []


def test_a_whole_tiff_is_read_only_through_read_tiff():
    """The one reader that opens the file before tifffile does; found, not listed."""
    import ast

    package = Path(__file__).resolve().parent.parent / "src" / "am_segmentation_manager"
    readers = []
    for module in package.rglob("*.py"):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "imread"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tifffile"
            ):
                readers.append(module.relative_to(package).as_posix())

    assert readers == ["io/image.py"]


@pytest.mark.parametrize(
    "shape",
    [(5, 6), (3, 5, 6), (5, 6, 3), (4, 4, 3), (4, 4, 4), (1, 5, 6), (6, 5, 6),
     (1, 1, 5, 6), (2, 5, 6, 3), (7,)],
)
def test_the_planar_shape_is_the_shape_to_2d_gives(shape):
    """A conversion sizes a label by this without reading the image it is sized to."""
    assert planar_shape(shape) == to_2d(np.zeros(shape, dtype=np.uint8)).shape


@pytest.mark.parametrize(
    "shape, options",
    [((8, 6), {}), ((8, 6, 3), {"photometric": "rgb"}), ((3, 8, 6), {"photometric": "minisblack"}),
     ((3, 8, 6), {"photometric": "rgb", "planarconfig": "separate"}),
     ((2, 8, 6, 3), {"photometric": "rgb"})],
    ids=["plane", "rgb", "pages", "planar-rgb", "rgb-pages"],
)
def test_a_tiffs_shape_reads_from_its_header_as_from_its_pixels(tmp_path, shape, options):
    path = tmp_path / "image.tif"
    tifffile.imwrite(str(path), np.zeros(shape, dtype=np.uint8), **options)

    assert read_tiff_shape(path) == read_tiff(path).shape
