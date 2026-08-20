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
        prediction = stage / predictions / "raw" / "img.tif"
        for path in (image, prediction):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"content")
        records.append(
            ImageRecord(image_path=image, modality="SEM", predictions=[prediction])
        )

    planned = plan_conversions(records).conversions
    destinations = [item.destination for item in planned]

    assert len(destinations) == len(set(destinations))
    assert convert_record_predictions(records) == 1
    assert len(list((stage / "sem_labels").glob("*.tif"))) == 1


def _record_with_prediction(tmp_path: Path, kind: str | None) -> ImageRecord:
    """Return a record whose single prediction sits in *kind*, or directly in
    the predictions directory when *kind* is None."""
    stage = tmp_path / "project" / "01" / "stage"
    image = stage / "sem" / "img.tif"
    directory = stage / "predictions"
    prediction = (directory / kind if kind else directory) / "img.tif"
    for path in (image, prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"content")
    return ImageRecord(image_path=image, modality="SEM", predictions=[prediction])


def test_post_processed_predictions_are_never_converted(tmp_path: Path) -> None:
    """Adaptive Polishing's clean/ output cannot become a label.

    Post-processing rewrites the intensity values, so they no longer index the
    segmentation classes. Converting one would produce a label that looks valid
    and teaches a model the wrong classes -- which is why this is a property of
    the layout rather than a choice offered to the caller.
    """
    from am_segmentation_manager.operations.predictions import plan_conversions

    plan = plan_conversions([_record_with_prediction(tmp_path, "clean")])

    assert plan.conversions == ()
    assert plan.count_skipped("post-processed") == 1
    # Not reported as a missing prediction: the pipeline ran, and the fix is to
    # re-run it, not to wait for it.
    assert plan.count_skipped("no-prediction") == 0


def test_an_undivided_predictions_directory_still_converts(tmp_path: Path) -> None:
    """Not every project splits predictions into raw/ and clean/.

    The rule excludes post-processed output rather than requiring ``raw/``, so a
    flat predictions directory keeps working.
    """
    from am_segmentation_manager.operations.predictions import plan_conversions

    plan = plan_conversions([_record_with_prediction(tmp_path, None)])

    assert len(plan.conversions) == 1


def test_raw_predictions_convert(tmp_path: Path) -> None:
    from am_segmentation_manager.operations.predictions import plan_conversions

    plan = plan_conversions([_record_with_prediction(tmp_path, "raw")])

    assert len(plan.conversions) == 1
    assert plan.skipped == ()


def test_a_label_is_replaced_where_it_lies_not_where_it_would_be_put(tmp_path: Path) -> None:
    """A record's existing label wins over a freshly resolved directory.

    A stage can hold both a legacy `labels/` and a current `sem_labels/`, and
    resolution prefers the latter. Computing the destination that way would
    write a *second* label beside the one really there while the real one
    stayed put -- worse than either overwriting or refusing.
    """
    from am_segmentation_manager.operations.predictions import plan_conversions

    record = _record_with_prediction(tmp_path, "raw")
    stage = record.image_path.parent.parent
    (stage / "sem_labels").mkdir(parents=True)  # preferred by resolution
    label = stage / "labels" / "img.tif"        # but the label is really here
    label.parent.mkdir(parents=True)
    label.write_bytes(b"label")
    record.labels = [label]

    plan = plan_conversions([record], overwrite=True)

    assert [item.destination for item in plan.conversions] == [label]


def test_a_corrected_label_is_refused_even_with_overwrite(tmp_path: Path) -> None:
    """There is no argument to `plan_conversions` that reaches proofread work."""
    from am_segmentation_manager.operations.predictions import plan_conversions

    record = _record_with_prediction(tmp_path, "raw")
    label = record.image_path.parent.parent / "sem_labels" / "img.tif"
    label.parent.mkdir(parents=True)
    label.write_bytes(b"proofread")
    record.labels = [label]
    record.state = "corrected"

    plan = plan_conversions([record], overwrite=True)

    assert plan.conversions == ()
    assert [item.destination for item in plan.refused] == [label]
    assert label.read_bytes() == b"proofread"


def test_an_interrupted_conversion_leaves_the_original_label_intact(
    tmp_path: Path, monkeypatch
) -> None:
    """The label appears whole or not at all.

    Now that conversion can overwrite, a write that fails part way through would
    otherwise destroy a readable label and leave a truncated file in its place.
    """
    from am_segmentation_manager.operations import predictions

    prediction = tmp_path / "pred.tif"
    label = tmp_path / "label.tif"
    prediction.write_bytes(b"prediction")
    label.write_bytes(b"the original label")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(predictions, "_write_label", explode)

    with pytest.raises(OSError):
        predictions.prediction_to_label(prediction, label)

    assert label.read_bytes() == b"the original label"
    # And no temporary file is left behind next to it.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["label.tif", "pred.tif"]
