"""Tests for prediction_to_label conversion including resizing.

These tests are skipped if optional dependencies (tifffile, numpy, cv2) are not
installed so they won't break minimal installations.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Ensure src/ is on sys.path when running tests from the repository root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.operations.predictions import prediction_to_label


@pytest.mark.skipif(
    not (
        __import__("importlib").util.find_spec("tifffile") is not None
        and __import__("importlib").util.find_spec("numpy") is not None
        and __import__("importlib").util.find_spec("cv2") is not None
    ),
    reason="Optional image libs not installed",
)
def test_prediction_resizing(tmp_path: Path):
    import tifffile
    import numpy as np

    stage = tmp_path / "stage"
    stage.mkdir()

    pred = stage / "pred.tif"
    target = stage / "target.tif"
    out = stage / "label.tif"

    # create a small prediction (32x32) and a target (64x48)
    tifffile.imwrite(str(pred), (np.ones((32, 32), dtype=np.uint8) * 5))
    tifffile.imwrite(str(target), (np.zeros((48, 64), dtype=np.uint8)))

    prediction_to_label(pred, out, target_image_path=target)

    # verify output image exists and has target shape
    arr = tifffile.imread(str(out))
    assert arr.shape[0] == 48
    assert arr.shape[1] == 64
    assert arr.dtype == np.uint8


@pytest.mark.skipif(
    not (
        __import__("importlib").util.find_spec("tifffile") is not None
        and __import__("importlib").util.find_spec("numpy") is not None
        and __import__("importlib").util.find_spec("cv2") is not None
    ),
    reason="Optional image libs not installed",
)
def test_prediction_to_label_falls_back_when_tifffile_cannot_read(tmp_path: Path, monkeypatch):
    import tifffile

    pred = tmp_path / "pred.tif"
    target = tmp_path / "target.tif"
    out = tmp_path / "label.tif"
    pred.write_bytes(b"hello")
    target.write_bytes(b"hello")

    def fail_imread(path):
        raise OSError("imagecodecs unavailable")

    monkeypatch.setattr(tifffile, "imread", fail_imread)

    prediction_to_label(pred, out, target_image_path=target)

    assert out.exists()
    assert out.read_bytes() == pred.read_bytes()
