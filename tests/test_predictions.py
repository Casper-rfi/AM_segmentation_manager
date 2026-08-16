"""Tests for prediction_to_label conversion including resizing.

These tests are skipped if optional dependencies (tifffile, numpy, cv2) are not
installed so they won't break minimal installations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
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
    import numpy as np
    import tifffile

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


def test_conditions_sharing_a_label_directory_convert_once(tmp_path: Path) -> None:
    """Two imaging conditions of one scene resolve to a single label file.

    Planning the whole batch up front is what makes a preview possible, and it
    is also where a second record claiming the same destination stops being
    caught by the first one's file appearing on disk. Converting both would
    overwrite a label that had just been written, and report two.
    """
    from am_segmentation_manager.operations.predictions import (
        convert_record_predictions,
        plan_conversions,
    )

    stage = tmp_path / "project" / "01" / "stage"
    (stage / "sem_labels").mkdir(parents=True)  # shared: no sem_5kv_labels beside it
    records = []
    for folder, predictions in (("sem", "predictions"), ("sem_5kv", "predictions_5kv")):
        image = stage / folder / "img.tif"
        prediction = stage / predictions / "clean" / "img.tif"
        for path in (image, prediction):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"content")
        records.append(
            ImageRecord(image_path=image, modality="SEM", predictions=[prediction])
        )

    planned = plan_conversions(records, "clean")
    destinations = [item.destination for item in planned]

    assert len(destinations) == len(set(destinations))
    assert convert_record_predictions(records, "clean") == 1
    assert len(list((stage / "sem_labels").glob("*.tif"))) == 1
