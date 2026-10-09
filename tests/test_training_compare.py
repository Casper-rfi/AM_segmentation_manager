"""Tests for setting training runs side by side: one value per run and class, None where there is none."""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from am_segmentation_manager.training_compare import (
    EVERY_SET,
    ComparedRun,
    ScoredValidation,
    compare_runs,
    dividing_columns,
    validation_sets,
)
from am_segmentation_manager.training_runs import EpochMetrics, TrainingRun, read_training_run
from am_segmentation_manager.validation_scores import ImageScores
from test_training_runs import FIXTURE


def _made(name: str, *scores: tuple[float, float], marked: int | None = None) -> ComparedRun:
    """Return a run of two classes whose validated epochs score *scores*, ranked by the first."""
    val = tuple(
        EpochMetrics(
            number, 1.0,
            {"weighted_average_iou": gis, "weighted_average_dice": gis},
            {"iou": (gis, crack)},
        )
        for number, (gis, crack) in enumerate(scores, 1)
    )
    run = TrainingRun(
        Path(f"training_{name}"), name, None, val=val, parameters={"label_names": ["gis", "crack"]}
    )
    return ComparedRun(name, run, marked)


def test_a_cell_is_the_metric_at_the_run_s_best_epoch():
    table = compare_runs([_made("a", (0.2, 0.9), (0.6, 0.1), (0.5, 0.5))])

    (row,) = table.rows

    assert table.columns == (("gis", "validation"), ("crack", "validation"))
    assert (row.name, row.epochs, row.values) == ("a", "2", (0.6, 0.1))


def test_the_real_run_reads_as_its_best_epoch_says():
    run = read_training_run(FIXTURE)

    table = compare_runs([ComparedRun("real", run)], classes=("lamella",))

    assert table.rows[0].values == (run.best().per_class["iou"][2],)


def test_the_epoch_marked_is_taken_and_the_run_s_own_where_none_is():
    runs = [_made("marked", (0.2, 0.9), (0.6, 0.1), marked=1), _made("not", (0.2, 0.9), (0.6, 0.1))]

    marked, not_marked = compare_runs(runs, value="marked").rows

    assert (marked.epochs, marked.values) == ("1 (marked)", (0.2, 0.9))
    assert (not_marked.epochs, not_marked.values) == ("2", (0.6, 0.1))


@pytest.mark.parametrize("value, summed", [("mean", statistics.fmean), ("median", statistics.median)])
def test_a_mean_or_median_is_over_the_epochs_in_the_range_only(value, summed):
    scores = [(0.1, 0.0), (0.2, 0.0), (0.9, 0.0), (0.4, 0.0), (0.0, 0.0)]

    (row,) = compare_runs([_made("a", *scores)], value=value, epoch_range=(2, 4)).rows

    assert row.epochs == "2-4 (3)"
    assert row.values[0] == pytest.approx(summed([0.2, 0.9, 0.4]))


def test_a_run_with_no_epoch_in_the_range_has_empty_cells_not_zeros():
    (row,) = compare_runs([_made("short", (0.5, 0.5))], value="mean", epoch_range=(60, 90)).rows

    assert (row.epochs, row.values) == ("60-90 (0)", (None, None))


def test_only_the_classes_asked_for_get_columns_in_the_order_asked():
    table = compare_runs([_made("a", (0.2, 0.9))], classes=("crack", "ice"))

    assert table.classes == ("crack", "ice")
    assert table.rows[0].values == (0.9, None)


def test_the_table_as_a_csv_leaves_a_missing_value_blank():
    table = compare_runs(
        [_made("a", (0.25, 0.5)), _made("b")], classes=("gis",),
    )

    assert table.header() == ("run", "epochs", "gis iou")
    assert table.cells() == [("a", "1", "0.2500"), ("b", "", "")]


def test_an_unknown_value_metric_or_a_range_backwards_is_refused():
    run = [_made("a", (0.2, 0.9))]

    with pytest.raises(ValueError, match="Unknown value"):
        compare_runs(run, value="last")
    with pytest.raises(ValueError, match="Unknown metric"):
        compare_runs(run, metric="sharpness")
    with pytest.raises(ValueError, match="not a range"):
        compare_runs(run, value="mean", epoch_range=(9, 3))


# -- validation sets beyond a run's own ---------------------------------------


def _scored(name: str = "held") -> ScoredValidation:
    """Two epochs over three images of two instruments; the Hydra image is found only in epoch 2."""
    perfect, missed = ((4, 0), (0, 2)), ((4, 0), (2, 0))
    images = ("a.tif", "b.tif", "c.tif")
    return ScoredValidation(
        name,
        scores=(
            ImageScores(1, images, (perfect, perfect, missed), (0.1,) * 3, (1.0, 1.0)),
            ImageScores(2, images, (perfect, perfect, perfect), (0.1,) * 3, (1.0, 1.0)),
        ),
        images=(
            {"image": "a.tif", "instrument": "Arctis", "lamella": "01"},
            {"image": "b.tif", "instrument": "Arctis", "lamella": "01"},
            {"image": "c.tif", "instrument": "Hydra", "lamella": "01"},
        ),
    )


def test_a_validation_is_one_set_or_a_set_per_kind_of_image():
    whole = validation_sets([_scored()], "held")
    parts = validation_sets([_scored()], "held", "instrument")

    assert list(whole) == ["held"] and [epoch.epoch for epoch in whole["held"]] == [1, 2]
    assert list(parts) == ["Arctis", "Hydra"]
    assert [epoch.per_class["iou"][1] for epoch in parts["Hydra"]] == [0.0, 1.0]
    assert [epoch.per_class["iou"][1] for epoch in parts["Arctis"]] == [1.0, 1.0]


def test_every_validation_can_be_asked_for_whole_and_none_gives_none():
    assert list(validation_sets([_scored("a"), _scored("b")], EVERY_SET)) == ["a", "b"]
    assert validation_sets([_scored()], "") == {}
    assert validation_sets([_scored()], "another") == {}


def test_only_a_column_holding_a_few_kinds_divides_a_set():
    assert dividing_columns([_scored()]) == ("instrument",)


def test_a_set_s_column_holds_the_run_s_chosen_epoch_scored_on_that_set():
    """The run's best is epoch 1 by its own validation, where Hydra was still missed."""
    run = _made("a", (0.9, 0.9), (0.5, 0.5))
    compared = ComparedRun("a", run.run, sets=validation_sets([_scored()], "held", "instrument"))

    table = compare_runs([compared], classes=("crack",), sets=("validation", "Arctis", "Hydra"))

    assert table.columns == (("crack", "validation"), ("crack", "Arctis"), ("crack", "Hydra"))
    assert table.rows[0].values == (0.9, 1.0, 0.0)
    assert table.header() == ("run", "epochs", "crack validation iou", "crack Arctis iou", "crack Hydra iou")


def test_a_set_the_chosen_epoch_was_not_scored_on_is_empty():
    run = _made("a", (0.1, 0.1), (0.2, 0.2), (0.9, 0.9))
    compared = ComparedRun("a", run.run, sets=validation_sets([_scored()], "held"))

    table = compare_runs([compared], classes=("crack",), sets=("held", "never validated"))

    assert table.rows[0].epochs == "3" and table.rows[0].values == (None, None)


def test_a_mean_over_a_set_is_over_the_epochs_scored_on_it():
    run = _made("a", (0.1, 0.1), (0.2, 0.2))
    compared = ComparedRun("a", run.run, sets=validation_sets([_scored()], "held", "instrument"))

    table = compare_runs(
        [compared], classes=("crack",), sets=("Hydra",), value="mean", epoch_range=(1, 2)
    )

    assert table.rows[0].values == (0.5,)
