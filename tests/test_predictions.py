"""Tests for prediction_to_label conversion including resizing.

These tests are skipped if optional dependencies (tifffile, numpy, cv2) are not
installed so they won't break minimal installations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.layout import (
    NO_CORRECTIONS,
    resolve_labels_directory,
    subdirectories_by_name,
)
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.pseudo_labels import as_converted, prediction_to_label
from conftest import write_prediction


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


needs_image_stack = pytest.mark.skipif(
    not all(
        __import__("importlib").util.find_spec(name) is not None
        for name in ("tifffile", "numpy", "cv2")
    ),
    reason="Optional image libs not installed",
)


@needs_image_stack
def test_a_conversion_writes_what_a_comparison_shows(tmp_path: Path):
    """The version strip draws a prediction through as_converted; a conversion
    that wrote anything else would make that picture a lie."""
    import numpy as np
    import tifffile

    prediction = tmp_path / "pred.tif"
    target = tmp_path / "target.tif"
    out = tmp_path / "label.tif"
    classes = np.zeros((32, 32), dtype=np.uint8)
    classes[3:9, 5:20] = 2
    classes[20:31, 0:7] = 4
    tifffile.imwrite(prediction, classes)
    tifffile.imwrite(target, np.zeros((61, 64), dtype=np.uint8))

    prediction_to_label(prediction, out, target_image_path=target)

    assert np.array_equal(
        tifffile.imread(out), as_converted(classes, (61, 64), prediction)
    )


def _convert(tmp_path: Path, prediction) -> Path:
    import numpy as np
    import tifffile

    pred, target, out = tmp_path / "pred.tif", tmp_path / "target.tif", tmp_path / "label.tif"
    tifffile.imwrite(str(pred), prediction)
    tifffile.imwrite(str(target), np.zeros(prediction.shape, dtype=np.uint8))
    prediction_to_label(pred, out, target_image_path=target)
    return out


@needs_image_stack
@pytest.mark.parametrize("kind", ["class past 255", "probabilities"])
def test_a_prediction_the_conversion_would_change_is_refused(tmp_path: Path, kind: str):
    """The cast to 8 bits wrapped 256 round to background, and turned a map of
    probabilities into an empty label written as if it were a prediction."""
    import numpy as np

    if kind == "class past 255":
        prediction = np.full((8, 8), 256, dtype=np.uint16)
    else:
        prediction = np.full((8, 8), 0.8, dtype=np.float32)

    with pytest.raises(ValueError, match="not class indices"):
        _convert(tmp_path, prediction)

    assert not (tmp_path / "label.tif").exists()


@needs_image_stack
@pytest.mark.parametrize("kind", ["probabilities", "unreadable"])
def test_one_prediction_that_will_not_convert_refuses_the_batch_before_any_write(
    tmp_path: Path, kind: str
):
    """The batch stopped after writing the labels before a probability map, and an
    unreadable prediction was copied in as a label."""
    import numpy as np
    import tifffile

    from am_segmentation_manager.operations.pseudo_labels import (
        UnconvertiblePredictions,
        convert_record_predictions,
    )

    def record(name: str) -> ImageRecord:
        image = tmp_path / "sem" / f"{name}.tif"
        image.parent.mkdir(exist_ok=True)
        tifffile.imwrite(image, np.zeros((8, 8), dtype=np.uint8))
        prediction = tmp_path / "sem_predictions" / f"{name}.tif"
        prediction.parent.mkdir(exist_ok=True)
        tifffile.imwrite(prediction, np.ones((8, 8), dtype=np.uint8))
        return ImageRecord(
            image_path=image, predictions=[prediction], modality="SEM",
            labels_directory=tmp_path / "sem_labels",
        )

    records = [record("a"), record("b")]
    bad = records[1].predictions[0]
    if kind == "probabilities":
        tifffile.imwrite(bad, np.full((8, 8), 0.8, dtype=np.float32))
    else:
        bad.write_bytes(b"not a tiff")

    with pytest.raises(UnconvertiblePredictions, match="nothing was converted"):
        convert_record_predictions(records)

    assert not list((tmp_path / "sem_labels").glob("*.tif"))


@needs_image_stack
def test_whole_classes_stored_as_floats_still_convert(tmp_path: Path):
    import numpy as np
    import tifffile

    prediction = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    out = _convert(tmp_path, prediction)

    assert tifffile.imread(str(out)).tolist() == [[0, 1], [2, 3]]


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
    from am_segmentation_manager.operations.pseudo_labels import (
        convert_record_predictions,
        plan_conversions,
    )

    stage = tmp_path / "project" / "01" / "stage"
    (stage / "sem_labels").mkdir(parents=True)  # shared: no sem_5kv_labels beside it
    records = []
    for folder, predictions in (("sem", "predictions"), ("sem_5kv", "predictions_5kv")):
        image = stage / folder / "img.tif"
        prediction = stage / predictions / "raw" / "img.tif"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"content")
        write_prediction(prediction)
        # Resolved the way the scanner resolves it, so the shared-directory
        # rule is exercised here rather than asserted by construction.
        labels_directory, _, _ = resolve_labels_directory(
            image.parent, "SEM", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
        )
        records.append(
            ImageRecord(
                image_path=image,
                modality="SEM",
                predictions=[prediction],
                labels_directory=labels_directory,
            )
        )

    planned = plan_conversions(records).conversions
    destinations = [item.destination for item in planned]

    assert len(destinations) == len(set(destinations))
    assert len(convert_record_predictions(records)) == 1
    assert len(list((stage / "sem_labels").glob("*.tif"))) == 1


def _record_with_prediction(tmp_path: Path, kind: str | None) -> ImageRecord:
    """Return a record whose single prediction sits in *kind*, or directly in
    the predictions directory when *kind* is None."""
    stage = tmp_path / "project" / "01" / "stage"
    image = stage / "sem" / "img.tif"
    directory = stage / "predictions"
    prediction = (directory / kind if kind else directory) / "img.tif"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"content")
    write_prediction(prediction)
    return ImageRecord(
        image_path=image,
        modality="SEM",
        predictions=[prediction],
        labels_directory=stage / "sem_labels",
    )


def test_converting_reports_each_label_written(tmp_path: Path) -> None:
    """Counted from nothing before the first label, so a bar an earlier phase
    filled does not sit full through this one."""
    from am_segmentation_manager.operations.pseudo_labels import convert_record_predictions

    records = [_record_with_prediction(tmp_path / str(index), "raw") for index in range(3)]
    seen: list[tuple[int, int]] = []

    convert_record_predictions(records, on_progress=lambda done, total: seen.append((done, total)))

    assert seen == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_post_processed_predictions_are_never_converted(tmp_path: Path) -> None:
    """Adaptive Polishing's clean/ output cannot become a label.

    Post-processing rewrites the intensity values, so they no longer index the
    segmentation classes. Converting one would produce a label that looks valid
    and teaches a model the wrong classes -- which is why this is a property of
    the layout rather than a choice offered to the caller.
    """
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

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
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

    plan = plan_conversions([_record_with_prediction(tmp_path, None)])

    assert len(plan.conversions) == 1


def test_raw_predictions_convert(tmp_path: Path) -> None:
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

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
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

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
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

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
    from am_segmentation_manager.operations import pseudo_labels

    prediction = tmp_path / "pred.tif"
    label = tmp_path / "label.tif"
    prediction.write_bytes(b"prediction")
    label.write_bytes(b"the original label")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pseudo_labels, "_write_label", explode)

    with pytest.raises(OSError):
        pseudo_labels.prediction_to_label(prediction, label)

    assert label.read_bytes() == b"the original label"
    # And no temporary file is left behind next to it.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["label.tif", "pred.tif"]


def _record_with_both(tmp_path: Path) -> ImageRecord:
    """Return a record with an AM-seg prediction and a microscope one."""
    record = _record_with_prediction(tmp_path, "raw")
    am_seg = record.predictions[0].parent.parent / "am_seg" / "img.tif"
    am_seg.parent.mkdir(parents=True, exist_ok=True)
    am_seg.write_bytes(b"content")
    record.predictions.append(am_seg)
    return record


def test_an_am_seg_prediction_is_converted_by_default(tmp_path: Path) -> None:
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

    (conversion,) = plan_conversions([_record_with_both(tmp_path)]).conversions

    assert conversion.origin == "am-seg"
    assert conversion.fallback is False


def test_the_microscope_prediction_is_converted_when_preferred(tmp_path: Path) -> None:
    """Sometimes the pipeline's guess is the better one, and the caller says so."""
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

    plan = plan_conversions([_record_with_both(tmp_path)], prefer="microscope")
    (conversion,) = plan.conversions

    assert conversion.prediction.parent.name == "raw"
    assert conversion.fallback is False
    assert plan.count_fallbacks() == 0


def test_an_image_without_the_preferred_source_uses_the_other_and_says_so(
    tmp_path: Path,
) -> None:
    """Falling back is allowed; doing it without a word is not."""
    from am_segmentation_manager.operations.pseudo_labels import plan_conversions

    plan = plan_conversions([_record_with_prediction(tmp_path, "raw")])
    (conversion,) = plan.conversions

    assert conversion.fallback is True
    assert plan.count_fallbacks() == 1
    assert conversion.describe() == "write, from microscope"


def test_the_counts_name_images_that_would_use_the_other_source(tmp_path: Path) -> None:
    from am_segmentation_manager.operations.pseudo_labels import count_convertible

    both = _record_with_both(tmp_path / "a")
    microscope_only = _record_with_prediction(tmp_path / "b", "raw")

    assert count_convertible([both, microscope_only]).fallback == 1
    assert count_convertible([both, microscope_only], prefer="microscope").fallback == 0
