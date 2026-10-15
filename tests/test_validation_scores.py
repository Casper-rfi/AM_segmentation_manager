"""Tests for per-image scores: any part of a list scored from them, as AM-model-training would."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from am_segmentation_manager.training.scores import (
    ImageScores,
    metrics_for,
    read_scores,
    scores_filename,
    write_scores,
)

#: Two classes; in the first image class 1 is found whole, in the second half of it is missed.
SCORES = ImageScores(
    epoch=7,
    images=("a.tif", "b.tif"),
    confusion=(((6, 0), (0, 4)), ((6, 0), (2, 2))),
    losses=(0.5, 1.5),
    weights=(1.0, 3.0),
)


def test_iou_and_dice_are_each_image_s_averaged():
    metrics = metrics_for(SCORES)

    # Class 1: 4/4 in the first image, 2/4 in the second.
    assert metrics.per_class["iou"][1] == pytest.approx((1.0 + 0.5) / 2)
    assert metrics.per_class["dice"][1] == pytest.approx((1.0 + 2 * 2 / (2 * 2 + 2)) / 2)
    assert (metrics.epoch, metrics.loss) == (7, 1.0)


def test_precision_recall_and_f1_are_of_the_images_added_up():
    metrics = metrics_for(SCORES)

    # Class 1 over both: 6 found, none wrongly, 2 missed.
    assert metrics.per_class["precision"][1] == 1.0
    assert metrics.per_class["recall"][1] == pytest.approx(6 / 8)
    assert metrics.per_class["accuracy"] == metrics.per_class["recall"]
    assert metrics.per_class["f1"][1] == pytest.approx(12 / 14)


def test_the_averages_over_classes_are_plain_and_weighted():
    metrics = metrics_for(SCORES)
    background, found = metrics.per_class["iou"]

    assert metrics.scalars["mean_iou"] == pytest.approx((background + found) / 2)
    assert metrics.scalars["weighted_average_iou"] == pytest.approx((background + 3 * found) / 4)


def test_a_part_of_the_list_is_scored_from_its_own_images():
    metrics = metrics_for(SCORES, {"b.tif", "never scored.tif"})

    assert metrics.per_class["iou"][1] == 0.5
    assert metrics.loss == 1.5


def test_a_part_with_no_image_in_it_has_no_metrics_rather_than_zeros():
    assert metrics_for(SCORES, {"never scored.tif"}) is None
    assert metrics_for(SCORES, set()) is None


def test_an_image_a_class_is_neither_in_nor_predicted_in_does_not_count_for_it():
    scores = ImageScores(1, ("a", "b"), (((5, 0), (0, 0)), ((3, 0), (1, 1))), (0.0, 0.0), (1.0, 1.0))

    assert metrics_for(scores).per_class["iou"][1] == 0.5
    assert math.isnan(metrics_for(scores, {"a"}).per_class["iou"][1])
    assert metrics_for(scores, {"a"}).per_class["f1"][1] == 0.0


def test_scores_read_back_as_written_and_are_never_written_over(tmp_path: Path):
    path = tmp_path / scores_filename(7)

    write_scores(path, SCORES)

    assert path.name == "scores_epoch007.json"
    assert read_scores(path) == SCORES
    with pytest.raises(FileExistsError):
        write_scores(path, SCORES)


@pytest.mark.parametrize("body", [
    "not json", "{}", '{"version": 9, "epoch": 1}',
    '{"version": 1, "epoch": 1, "weights": [1], "images": ["a"], "losses": [], "confusion": []}',
])
def test_a_file_that_is_not_scores_is_refused(tmp_path: Path, body: str):
    path = tmp_path / "scores_epoch001.json"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError):
        read_scores(path)


def test_per_image_scores_give_the_metrics_am_model_training_reports():
    """The same predictions through its own metrics and through a matrix per image."""
    torch = pytest.importorskip("torch")
    metrics_module = pytest.importorskip("am_model_training.metrics")

    classes, weights = 5, (1.0, 4.0, 3.0, 6.0, 2.0)
    generator = torch.Generator().manual_seed(0)
    theirs = metrics_module.Metrics(device=torch.device("cpu"), num_classes=classes)
    matrices = []
    for image in range(6):
        # Some classes absent from an image's label, some never predicted in it.
        label = torch.randint(0, classes if image % 2 else 3, (1, 1, 32, 32), generator=generator)
        predicted = torch.randint(0, classes if image < 4 else 2, (1, 1, 32, 32), generator=generator)
        theirs.update(y=label, y_pred=predicted)
        counted = torch.bincount(label.flatten() * classes + predicted.flatten(), minlength=classes**2)
        matrices.append(tuple(tuple(row) for row in counted.reshape(classes, classes).tolist()))
    reported = theirs.get_epoch_metrics([1.0], weights=weights, device=torch.device("cpu")).to_dict()

    ours = metrics_for(ImageScores(1, tuple("abcdef"), tuple(matrices), (1.0,) * 6, weights))

    for name in ("iou", "dice", "f1", "accuracy", "precision", "recall"):
        assert ours.per_class[name] == pytest.approx(reported[name], abs=1e-6), name
        for average in (f"mean_{name}", f"weighted_average_{name}"):
            assert ours.scalars[average] == pytest.approx(reported[average], abs=1e-6), average
