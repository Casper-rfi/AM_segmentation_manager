"""Tests for setting training runs side by side: one value per run and class, None where there is none."""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from am_segmentation_manager.training_compare import ComparedRun, compare_runs
from am_segmentation_manager.training_runs import EpochMetrics, TrainingRun, read_training_run
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

    assert (row.epochs, row.values) == ("", (None, None))


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
