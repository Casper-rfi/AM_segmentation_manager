"""Tests for setting training runs side by side: one value per run and class, None where there is none."""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from am_segmentation_manager.training_compare import (
    EVERY_SET,
    NOT_DIVIDING,
    NOT_RECORDED,
    OWN_VALIDATION,
    ComparedRun,
    ScoredValidation,
    compare_runs,
    dividing_columns,
    offered_sets,
    set_sizes,
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


@pytest.mark.parametrize("column", sorted(NOT_DIVIDING))
def test_a_setting_version_or_identifier_does_not_divide_a_set(column):
    scored = _scored()
    images = [{**row, column: row["image"][0]} for row in scored.images]

    assert dividing_columns([ScoredValidation("held", scored.scores, images)]) == ("instrument",)


def test_a_run_s_own_list_is_divided_where_no_validation_is_named():
    own, held = _scored(OWN_VALIDATION), _scored("held")

    assert validation_sets([own, held], "") == {}
    assert list(validation_sets([own, held], "", "instrument")) == ["Arctis", "Hydra"]
    assert set_sizes([own, held], "", "instrument") == {"Arctis": 2, "Hydra": 1}
    assert validation_sets([held], "", "instrument") == {}


def test_every_set_leaves_out_the_own_list_s_scores_which_the_run_s_files_hold():
    assert list(validation_sets([_scored(OWN_VALIDATION), _scored("held")], EVERY_SET)) == ["held"]


def test_the_own_list_whole_is_offered_unless_it_or_one_validation_is_the_question():
    parts = [{"Arctis": (), "Hydra": ()}]

    assert offered_sets("", [{}]) == (OWN_VALIDATION,)
    assert offered_sets("", parts, "instrument") == ("Arctis", "Hydra")
    assert offered_sets("held", parts, "instrument") == ("Arctis", "Hydra")
    assert offered_sets(EVERY_SET, [{"held": ()}]) == (OWN_VALIDATION, "held")


def test_each_set_says_how_many_images_it_holds():
    assert set_sizes([_scored()], "held") == {"held": 3}
    assert set_sizes([_scored()], "held", "instrument") == {"Arctis": 2, "Hydra": 1}
    assert set_sizes([_scored("a"), _scored("b")], EVERY_SET) == {"a": 3, "b": 3}
    assert set_sizes([_scored()], "") == {}


def test_images_holding_no_value_are_a_part_of_their_own_and_the_last():
    scored = _scored()
    images = [dict(row) for row in scored.images]
    images[0]["instrument"] = ""
    scored = ScoredValidation(scored.name, scored.scores, images)

    assert list(validation_sets([scored], "held", "instrument")) == ["Arctis", "Hydra", NOT_RECORDED]
    assert set_sizes([scored], "held", "instrument") == {"Arctis": 1, "Hydra": 1, NOT_RECORDED: 1}
    assert sum(set_sizes([scored], "held", "instrument").values()) == len(images)


def test_a_column_some_images_hold_nothing_in_divides_a_set_by_that():
    scored = _scored()
    images = [{**row, "gas": "argon" if row["image"] == "a.tif" else ""} for row in scored.images]

    assert dividing_columns([ScoredValidation("held", scored.scores, images)]) == ("instrument", "gas")


def _counted(*sizes: dict[str, int]):
    """A table of a run per *sizes*, each holding the parts of the one validation."""
    sets = validation_sets([_scored()], "held", "instrument")
    runs = [
        ComparedRun(f"run{at}", _made("a", (0.9, 0.9)).run, sets=sets, sizes=held)
        for at, held in enumerate(sizes)
    ]
    return compare_runs(runs, classes=("crack",), sets=("validation", "Arctis", "Hydra"))


def test_the_table_carries_the_images_of_each_set_for_each_run():
    table = _counted({"Arctis": 2, "Hydra": 1})

    assert table.rows[0].counts == (None, 2, 1)
    assert [table.count_said(where) for where in table.sets] == ["", "n=2", "n=1"]


def test_runs_holding_different_numbers_of_a_set_say_from_how_few_to_how_many():
    table = _counted({"Arctis": 2, "Hydra": 1}, {"Arctis": 18, "Hydra": 1}, {})

    assert table.count_said("Arctis") == "n=2-18"
    assert table.count_said("Hydra") == "n=1"


def test_the_table_as_a_csv_gives_each_counted_set_a_column_of_images():
    table = _counted({"Arctis": 2, "Hydra": 1}, {"Arctis": 18})

    assert table.header()[:4] == ("run", "epochs", "Arctis images", "Hydra images")
    assert [row[2:4] for row in table.cells()] == [("2", "1"), ("18", "")]


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


# -- what the colours run between ---------------------------------------------


def _table(*values: float | None):
    from am_segmentation_manager.training_compare import Comparison, ComparisonRow

    return Comparison(
        "iou", "best", tuple((f"class{n}", "validation") for n in range(len(values))),
        (ComparisonRow("one", "3", tuple(values)),),
    )


def test_the_colours_fit_what_the_table_holds_rounded_outward():
    from am_segmentation_manager.training_compare import fitted_colour_range

    assert fitted_colour_range(_table(0.874, None, 0.951)) == (0.87, 0.96)
    assert fitted_colour_range(_table(0.29, 0.58)) == (0.29, 0.58)


def test_a_table_of_one_value_or_none_still_has_a_range():
    from am_segmentation_manager.training_compare import (
        DEFAULT_COLOUR_RANGE,
        fitted_colour_range,
    )

    assert fitted_colour_range(_table(None)) == DEFAULT_COLOUR_RANGE
    low, high = fitted_colour_range(_table(0.9, 0.9))
    assert low < high and low <= 0.9 <= high
    low, high = fitted_colour_range(_table(1.0))
    assert (low, high) == (0.99, 1.0)


def test_a_range_is_put_in_order_within_what_a_metric_can_be():
    from am_segmentation_manager.training_compare import colour_range

    assert colour_range(0.9, 0.2) == (0.2, 0.9)
    assert colour_range(-1, 2) == (0.0, 1.0)
    assert colour_range(1.0, 1.0) == (0.99, 1.0)

