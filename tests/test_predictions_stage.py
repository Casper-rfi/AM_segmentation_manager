"""Tests for the high-level convert_stage_predictions function."""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.operations.predictions import convert_stage_predictions


def _write_file(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_convert_stage_predictions_simple(tmp_path: Path):
    stage = tmp_path / "stage"
    sem = stage / "sem"
    preds = stage / "predictions" / "clean"

    # If tifffile is available, write minimal valid TIFFs so conversion uses image I/O.
    try:
        import numpy as np
        import tifffile
        sem.mkdir(parents=True, exist_ok=True)
        preds.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(sem / "img_sem.tif"), np.zeros((10, 10), dtype=np.uint8))
        tifffile.imwrite(str(preds / "img_sem.tif"), np.ones((10, 10), dtype=np.uint8))
    except Exception:
        # fallback: write simple bytes and let prediction_to_label fall back to copy
        _write_file(sem / "img_sem.tif")
        _write_file(preds / "img_sem.tif")

    converted = convert_stage_predictions(stage, modality="SEM", prediction_kind="clean")
    assert converted == 1
    assert (stage / "labels" / "img_sem.tif").exists()


@pytest.mark.skipif(not (lambda: __import__('importlib').util.find_spec('tifffile') and __import__('importlib').util.find_spec('numpy') and __import__('importlib').util.find_spec('cv2'))(), reason="Optional image libs not installed")
def test_convert_stage_predictions_resizes(tmp_path: Path):
    import numpy as np
    import tifffile

    stage = tmp_path / "stage2"
    stage.mkdir(parents=True)
    sem = stage / "sem"
    preds = stage / "predictions"
    sem.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    _write_file(sem / "img_sem.tif")

    # create shapes: pred 16x16, target 40x30
    # name prediction to match the target image so the function can find it
    tifffile.imwrite(str(preds / "img_sem.tif"), (np.ones((16, 16), dtype=np.uint8) * 7))
    tifffile.imwrite(str(sem / "img_sem.tif"), (np.zeros((30, 40), dtype=np.uint8)))

    converted = convert_stage_predictions(stage, modality="SEM", prediction_kind="")
    # ensure at least 1 converted
    assert converted >= 1
    # verify the label exists and matches target shape
    import tifffile as tf
    arr = tf.imread(str(stage / "labels" / "img_sem.tif"))
    assert arr.shape[0] == 30
    assert arr.shape[1] == 40
