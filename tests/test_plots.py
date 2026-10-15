"""Tests for the bar charts of aggregate breakdowns.

Charts are hard to assert on and easy to over-test. What is worth pinning is
the contract around them: that a file appears in the requested format, that a
missing optional dependency produces a clear error rather than a traceback,
and that a command still reports its counts when it cannot draw them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.plots import (
    FROZEN,
    bar_figure,
    comparison_figure,
    has_plot_support,
    heatmap_figure,
    nearest_curve_point,
    plot_comparison,
    plot_counts,
    plot_cross_tabulation,
    plot_training_run,
    require_plot_support,
    set_headings,
    training_figure,
)
from am_segmentation_manager.training_runs import EpochMetrics, TrainingRun, read_training_run

pytestmark = pytest.mark.skipif(not has_plot_support(), reason="matplotlib not installed")

COUNTS = {"Arctis": 2388, "Helios 5 Hydra CX": 182, "Aquilos": 100}
TABLE = {
    ("Arctis", "SEM"): 1261, ("Arctis", "FIB"): 1127,
    ("Aquilos", "SEM"): 50, ("Aquilos", "FIB"): 50,
}
SCOPE = ["--project", "all", "--lamella", "all", "--stage", "all", "--image", "all"]


def test_a_png_is_written(tmp_path: Path):
    output = plot_counts(COUNTS, "Images per microscope", tmp_path / "chart.png")

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")


def test_svg_is_available_for_figures(tmp_path: Path):
    """The suffix chooses the format, so a paper figure needs no extra option."""
    output = plot_counts(COUNTS, "Images per microscope", tmp_path / "chart.svg")

    assert output.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_a_cross_tabulation_is_drawn_as_a_grid(tmp_path: Path):
    """A grid rather than grouped bars: past about four columns the bars
    are unreadable, and it is the empty cells that carry the finding."""
    output = plot_cross_tabulation(TABLE, "microscope", "modality", tmp_path / "cross.png")

    assert output.exists()


def test_a_missing_directory_is_created(tmp_path: Path):
    output = plot_counts(COUNTS, "t", tmp_path / "nested" / "deep" / "chart.png")

    assert output.exists()


def test_plotting_nothing_is_an_error_rather_than_an_empty_chart(tmp_path: Path):
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_counts({}, "t", tmp_path / "chart.png")
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_cross_tabulation({}, "a", "b", tmp_path / "chart.png")


def test_drawing_needs_no_display(tmp_path: Path):
    """Charts are drawn from the command line, often over ssh. Figures are
    built directly rather than through pyplot, so none is ever handed to the
    global figure list -- which is both what would need a backend and what
    would keep every chart of a long session in memory."""
    import matplotlib.pyplot as plt

    plot_counts(COUNTS, "t", tmp_path / "chart.png")
    plot_cross_tabulation(TABLE, "a", "b", tmp_path / "cross.png")

    assert plt.get_fignums() == []


def test_cli_writes_a_chart(tmp_path: Path):
    from conftest import build_dataset

    build_dataset(tmp_path, acquisitions=2)
    chart = tmp_path / "out" / "modality.png"

    assert main(["stats", str(tmp_path), *SCOPE, "--by", "modality", "--plot", str(chart)]) == 0
    assert chart.exists()


def test_cli_still_prints_the_counts_when_it_cannot_draw(tmp_path: Path, capsys, monkeypatch):
    """A missing optional dependency must not lose the numbers that were
    already computed and printed."""
    import am_segmentation_manager.plots as plots
    from conftest import build_dataset

    build_dataset(tmp_path, acquisitions=2)
    monkeypatch.setattr(plots, "has_plot_support", lambda: False)

    code = main([
        "stats", str(tmp_path), *SCOPE, "--by", "modality",
        "--plot", str(tmp_path / "chart.png"),
    ])
    output = capsys.readouterr().out

    assert code == 2
    assert "Images per modality" in output
    assert not (tmp_path / "chart.png").exists()


# ---------------------------------------------------------------------------
# What each shape is for
# ---------------------------------------------------------------------------

def test_a_scale_keeps_its_order_rather_than_being_sorted():
    """Area bins reordered by population stop reading as a distribution."""
    bins = {"< 0.01": 2, "0.01 - 0.1": 17, "0.1 - 1": 32, "1 - 10": 9}
    figure = bar_figure(bins, "areas", ordered=True)

    drawn = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert drawn == list(bins)


def test_named_groups_carry_a_running_share():
    """How many groups reach 90% is the diversity question directly."""
    figure = bar_figure(COUNTS, "Images per microscope")

    assert len(figure.axes) == 2, "no cumulative axis was added"
    assert "cumulative" in figure.axes[1].get_xlabel()


def test_a_scale_carries_no_running_share():
    """Cumulative over bins is a different claim, and an unasked one."""
    figure = bar_figure({"a": 1, "b": 2, "c": 3}, "t", ordered=True)

    assert len(figure.axes) == 1


def test_two_bars_are_not_worth_a_share_line():
    figure = bar_figure({"a": 1, "b": 2}, "t")

    assert len(figure.axes) == 1


def test_a_grid_can_be_shaded_by_share_of_its_row():
    """One row dwarfing the rest hides the pattern inside the small ones."""
    figure = heatmap_figure(TABLE, "microscope", "modality", shading="row")

    texts = {text.get_text() for text in figure.axes[0].texts}
    assert "50%" in texts        # Aquilos is half SEM, half FIB
    assert "images" not in figure.axes[-1].get_ylabel()


def test_a_row_with_nothing_in_it_is_not_a_division_by_zero():
    """A cross-tabulation can carry a row the filters emptied."""
    figure = heatmap_figure({("a", "x"): 0, ("b", "x"): 3}, "r", "c", shading="row")

    assert {text.get_text() for text in figure.axes[0].texts} == {"100%"}


def test_a_combination_nothing_was_acquired_for_is_grey_and_unnumbered():
    """The empty cells are the finding, and against the palette's white floor
    a 1 and a 0 looked alike."""
    from matplotlib.colors import to_rgba

    from am_segmentation_manager.plots import ZERO_COLOUR

    figure = heatmap_figure({("a", "x"): 5, ("b", "y"): 1}, "r", "c")
    image = figure.axes[0].images[0]

    assert image.norm.vmin == 0
    assert list(image.get_array().mask.ravel()) == [False, True, True, False]
    assert tuple(image.cmap.get_bad()) == pytest.approx(to_rgba(ZERO_COLOUR))
    assert sorted(text.get_text() for text in figure.axes[0].texts) == ["1", "5"]


def test_a_share_too_small_to_round_up_is_not_read_as_nothing():
    """One image in a row of thousands is a finding; "0%" beside the grey
    cells that hold nothing would say it is not there."""
    figure = heatmap_figure({("a", "x"): 1, ("a", "y"): 999}, "r", "c", shading="row")

    assert {text.get_text() for text in figure.axes[0].texts} == {"<1%", "100%"}


def test_a_grid_can_be_shaded_by_share_of_its_column():
    figure = heatmap_figure(TABLE, "microscope", "modality", shading="column")

    texts = {text.get_text() for text in figure.axes[0].texts}
    assert {"96%", "4%"} <= texts        # Arctis took 1261 of the 1311 SEM images
    assert figure.axes[-1].get_ylabel() == "share of column (%)"


def test_an_unknown_shading_is_refused():
    with pytest.raises(ValueError, match="shading"):
        heatmap_figure(TABLE, "r", "c", shading="diagonal")


def test_the_missing_dependency_error_says_how_to_fix_it(monkeypatch):
    import am_segmentation_manager.plots as plots

    monkeypatch.setattr(plots, "has_plot_support", lambda: False)
    with pytest.raises(RuntimeError, match="pip install"):
        require_plot_support()


# ---------------------------------------------------------------------------
# An existing chart is somebody's
# ---------------------------------------------------------------------------

def test_an_existing_chart_is_refused_and_left_untouched(tmp_path: Path):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"a chart somebody kept")

    with pytest.raises(FileExistsError):
        plot_counts(COUNTS, "t", chart)
    with pytest.raises(FileExistsError):
        plot_cross_tabulation(TABLE, "a", "b", chart)

    assert chart.read_bytes() == b"a chart somebody kept"


def test_overwrite_replaces_an_existing_chart(tmp_path: Path):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"an old chart")

    plot_counts(COUNTS, "t", chart, overwrite=True)

    assert chart.read_bytes().startswith(b"\x89PNG")


def test_a_chart_appearing_while_it_is_written_is_not_replaced(tmp_path: Path, monkeypatch):
    """The up-front check and the rename are not one step, so the rename asks again."""
    from matplotlib.figure import Figure

    chart = tmp_path / "chart.png"
    draw = Figure.savefig

    def savefig(self, *args, **kwargs):
        chart.write_bytes(b"written meanwhile")
        return draw(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)

    with pytest.raises(FileExistsError, match="appeared"):
        plot_counts(COUNTS, "t", chart)

    assert chart.read_bytes() == b"written meanwhile"
    assert [path.name for path in tmp_path.iterdir()] == ["chart.png"], "no temporary left behind"


def test_cli_refuses_an_existing_chart_before_counting(tmp_path: Path, capsys):
    from conftest import build_dataset

    build_dataset(tmp_path, acquisitions=2)
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"a chart somebody kept")
    command = ["stats", str(tmp_path), *SCOPE, "--by", "modality", "--plot", str(chart)]

    assert main(command) == 2
    assert "Images per modality" not in capsys.readouterr().out
    assert chart.read_bytes() == b"a chart somebody kept"

    assert main([*command, "--overwrite"]) == 0
    assert chart.read_bytes().startswith(b"\x89PNG")


# -- a training run -------------------------------------------------------------

TRAINING_RUN = Path(__file__).parent / "data" / "amt_run" / "training_260927_093830"


def _legend_labels(axes) -> list[str]:
    return [text.get_text() for text in axes.get_legend().get_texts()]


def test_a_training_run_is_drawn_as_loss_beside_its_classes(tmp_path: Path):
    output = plot_training_run(read_training_run(TRAINING_RUN), tmp_path / "run.svg")

    assert output.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def _frozen_bands(axes) -> list[tuple[float, float]]:
    """Return where each band behind the frozen epochs starts and ends, in epochs."""
    return [
        (patch.get_x(), patch.get_x() + patch.get_width())
        for patch in axes.patches if patch.get_gid() == FROZEN
    ]


def test_the_best_epoch_is_marked():
    loss_axes, metric_axes = training_figure(read_training_run(TRAINING_RUN)).axes

    labels = _legend_labels(metric_axes)
    assert "best epoch (2)" in labels
    assert {"background", "gis", "lamella", "crack", "vacuum"} <= set(labels)
    assert {"training", "validation"} <= set(_legend_labels(loss_axes))


def _run_of(epochs: int, frozen: int | None) -> TrainingRun:
    scalars = {"weighted_average_iou": 0.4, "weighted_average_dice": 0.5}
    trained = tuple(
        EpochMetrics(epoch, 0.5, scalars, {"iou": (0.3, 0.5)})
        for epoch in range(1, epochs + 1)
    )
    parameters = {} if frozen is None else {"frozen_epochs": frozen}
    return TrainingRun(Path("run"), "run", None, train=trained, val=trained, parameters=parameters)


def test_the_epochs_the_encoder_was_frozen_for_are_shaded_on_both_panels():
    figure = training_figure(_run_of(epochs=6, frozen=3))

    for axes in figure.axes:
        assert _frozen_bands(axes) == [(0.5, 3.5)]
        assert "encoder frozen (1-3)" in _legend_labels(axes)
        assert "encoder unfrozen" not in _legend_labels(axes)


def test_the_shading_stops_at_the_last_epoch_trained():
    """A run two epochs into twenty-five frozen ones is not drawn twenty-five wide."""
    loss_axes, _ = training_figure(_run_of(epochs=2, frozen=25)).axes

    assert _frozen_bands(loss_axes) == [(0.5, 2.5)]
    assert "encoder frozen (1-25)" in _legend_labels(loss_axes)
    assert loss_axes.get_xlim()[1] < 3


def test_a_run_still_training_is_shaded_by_what_it_was_asked():
    """A run says how long it froze its encoder only once it has ended."""
    loss_axes, _ = training_figure(_run_of(epochs=6, frozen=None), frozen=3).axes

    assert _frozen_bands(loss_axes) == [(0.5, 3.5)]


def test_what_a_run_says_it_froze_is_believed_over_what_it_was_asked():
    loss_axes, _ = training_figure(_run_of(epochs=6, frozen=2), frozen=3).axes

    assert _frozen_bands(loss_axes) == [(0.5, 2.5)]


@pytest.mark.parametrize("frozen", [0, None])
def test_a_run_with_nothing_frozen_is_not_shaded(frozen):
    for axes in training_figure(_run_of(epochs=4, frozen=frozen)).axes:
        assert _frozen_bands(axes) == []


def test_a_run_not_yet_validated_draws_its_training_classes():
    epoch = EpochMetrics(1, 0.5, {"weighted_average_iou": 0.4}, {"iou": (0.3, 0.5)})
    run = TrainingRun(Path("training_261001_000000"), "set", None, train=(epoch,))

    _, metric_axes = training_figure(run).axes

    assert "from training" in metric_axes.get_ylabel()
    assert not any(label.startswith("best epoch") for label in _legend_labels(metric_axes))


def test_a_run_with_no_epoch_is_not_drawn():
    with pytest.raises(ValueError, match="no epoch"):
        training_figure(TrainingRun(Path("training_261001_000000"), None, None))


def test_an_unknown_training_metric_is_refused():
    with pytest.raises(ValueError, match="Unknown metric"):
        training_figure(read_training_run(TRAINING_RUN), "sharpness")


def test_an_existing_training_figure_is_refused(tmp_path: Path):
    output = tmp_path / "run.png"
    output.write_bytes(b"mine")

    with pytest.raises(FileExistsError):
        plot_training_run(read_training_run(TRAINING_RUN), output)
    assert output.read_bytes() == b"mine"


# -- which classes are drawn, and what the pointer is on ----------------------


def test_only_the_classes_asked_for_get_a_curve():
    _, metric_axes = training_figure(
        read_training_run(TRAINING_RUN), classes=("gis", "crack")
    ).axes

    labels = set(_legend_labels(metric_axes))

    assert {"gis", "crack"} <= labels
    assert not {"background", "lamella", "vacuum"} & labels
    assert "weighted average, validation" in labels


def test_a_class_the_run_does_not_have_is_refused():
    with pytest.raises(ValueError, match="Unknown class ice"):
        training_figure(read_training_run(TRAINING_RUN), classes=("ice",))


def _drawn(figure):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure).draw()
    return figure


def _pixel(axes, label: str, index: int) -> tuple[float, float]:
    line = next(line for line in axes.get_lines() if line.get_label() == label)
    return tuple(line.get_transform().transform(line.get_xydata()[index]))


def test_the_pointer_on_a_curve_reads_its_class_epoch_and_value():
    run = read_training_run(TRAINING_RUN)
    figure = _drawn(training_figure(run, classes=("lamella",)))
    _, metric_axes = figure.axes
    x, y = _pixel(metric_axes, "lamella", 1)

    point = nearest_curve_point(figure, x + 2, y - 2)

    assert point is not None and point.axes is metric_axes
    assert (point.curve, point.x) == ("lamella", 2.0)
    assert point.y == pytest.approx(run.val[1].per_class["iou"][2])
    assert point.describe() == f"lamella · epoch 2 · IoU {point.y:.3f}"


def test_the_loss_reads_on_its_logarithmic_axis():
    run = read_training_run(TRAINING_RUN)
    figure = _drawn(training_figure(run))
    loss_axes, _ = figure.axes

    point = nearest_curve_point(figure, *_pixel(loss_axes, "validation", 2))

    assert point is not None
    assert point.describe() == f"validation · epoch 3 · loss {run.val[2].loss:.3f}"


def test_away_from_every_curve_the_pointer_is_on_nothing():
    figure = _drawn(training_figure(read_training_run(TRAINING_RUN)))

    assert nearest_curve_point(figure, 1.0, 1.0) is None


def test_the_lines_marking_an_epoch_are_not_curves_to_point_at():
    figure = _drawn(training_figure(read_training_run(TRAINING_RUN), classes=()))
    _, metric_axes = figure.axes
    marker = next(line for line in metric_axes.get_lines() if line.get_label().startswith("best"))
    x, _ = marker.get_transform().transform((2, 0))
    _, y = metric_axes.transData.transform((2, 0.9))

    assert nearest_curve_point(figure, x, y) is None


# -- runs side by side --------------------------------------------------------


def _comparison(**asked):
    from am_segmentation_manager.training_compare import ComparedRun, compare_runs

    run = read_training_run(TRAINING_RUN)
    return compare_runs([ComparedRun("one", run), ComparedRun("two", run, 3)], **asked)


def test_runs_are_drawn_a_row_each_with_a_block_per_class():
    table = _comparison(classes=("gis", "lamella"))

    axes, _bar = comparison_figure(table).axes
    said = [text.get_text() for text in axes.texts]

    assert "gis" in said and "lamella" in said
    assert said.count(f"{table.rows[0].values[0]:.3f}") == 2
    assert [label.get_text() for label in axes.get_yticklabels()] == ["one   2", "two   2"]


def _split(parts: dict[str, int], classes=("gis", "lamella", "crack", "ice", "vacuum")):
    """A table of one run with a column per part under each of *classes*."""
    from am_segmentation_manager.training_compare import ComparedRun, compare_runs

    run = read_training_run(TRAINING_RUN)
    compared = ComparedRun("one", run, sets={name: run.val for name in parts}, sizes=parts)
    return compare_runs([compared], classes=classes, sets=tuple(parts))


def test_a_set_is_headed_by_its_name_and_its_images():
    axes, _bar = comparison_figure(_split({"Arctis": 439, "Hydra": 12}, classes=("gis",))).axes

    assert {"Arctis\nn=439", "Hydra\nn=12"} <= {text.get_text() for text in axes.texts}


def test_a_table_of_one_uncounted_set_has_no_row_of_set_headings():
    axes, _bar = comparison_figure(_comparison(classes=("gis",))).axes

    assert "validation" not in {text.get_text() for text in axes.texts}


def test_a_name_too_long_to_sit_across_its_cell_is_turned_upright_and_cut_short():
    long = "Casper Berger; Maria Lopes; Alan Turner; Grace Hopkinson"
    headings = set_headings([long, "Arctis"], ["n=39", "n=8"], columns=10)

    assert headings.upright
    assert headings.texts == ("Casper Berger; Maria Lopes;…  n=39", "Arctis  n=8")
    assert headings.height > 2


def test_short_names_in_wide_cells_sit_across_them_over_their_images():
    headings = set_headings(["Helios 5 Hydra CX", "Arctis"], ["n=12", ""], columns=6)

    assert not headings.upright and not headings.upright_values
    assert headings.texts == ("Helios 5 Hydra\nCX\nn=12", "Arctis")


def test_no_heading_of_a_table_of_many_parts_runs_over_the_next():
    """Twelve parts under five classes, drawn at the size a pane gives a figure."""
    parts = {f"Person {at}; Another Person {at}; A Third": at + 1 for at in range(12)}
    figure = comparison_figure(_split(parts))
    figure.set_size_inches(11, 4)
    axes, _bar = _drawn(figure).axes
    renderer = figure.canvas.get_renderer()
    boxes = sorted(
        (text.get_window_extent(renderer) for text in axes.texts if "Person" in text.get_text()),
        key=lambda box: box.x0,
    )

    assert len(boxes) == 60
    assert all(left.x1 <= right.x0 for left, right in zip(boxes, boxes[1:]))


def test_a_cell_with_no_value_is_drawn_empty():
    table = _comparison(classes=("gis",), value="mean", epoch_range=(60, 90))

    axes, _bar = comparison_figure(table).axes

    assert [text.get_text() for text in axes.texts] == ["gis"]


def test_a_table_with_nothing_in_it_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="Nothing to plot"):
        comparison_figure(_comparison(classes=()))


def test_the_table_is_written_and_never_over_a_file(tmp_path: Path):
    output = plot_comparison(_comparison(), tmp_path / "table.svg")

    assert output.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    with pytest.raises(FileExistsError):
        plot_comparison(_comparison(), output)


def _cell_colours(figure) -> list[tuple[float, ...]]:
    """Return the fill of each numbered cell of a comparison, in the order drawn."""
    axes = figure.axes[0]
    numbered = {
        (round(text.get_position()[0] - 0.5, 6), round(text.get_position()[1] - 0.5, 6))
        for text in axes.texts if text.get_text()[:1].isdigit()
    }
    return [
        patch.get_facecolor() for patch in axes.patches
        if (round(patch.get_x(), 6), round(patch.get_y(), 6)) in numbered
    ]


def test_the_colours_run_between_the_two_values_asked_for():
    """From 0 to 1, runs that differ in the second decimal are one green."""
    from matplotlib import colormaps

    from am_segmentation_manager.plots import COMPARISON_PALETTE
    from am_segmentation_manager.training_compare import fitted_colour_range

    table = _comparison(classes=("gis", "lamella"))
    palette = colormaps[COMPARISON_PALETTE]
    values = [value for row in table.rows for value in row.values]

    fitted = _cell_colours(comparison_figure(table, fitted_colour_range(table)))
    whole = _cell_colours(comparison_figure(table))

    assert len(set(values)) > 1
    assert len(set(fitted)) > 1
    assert palette(0.0) not in whole
    # Beyond either end is that end's colour, not an error.
    assert set(_cell_colours(comparison_figure(table, (0.0, 0.01)))) == {palette(1.0)}


def test_the_table_carries_a_bar_saying_what_its_colours_span():
    """A figure saved for a slide has nothing else to say it."""
    table = _comparison(classes=("gis",))

    _, bar = comparison_figure(table, (0.8, 0.95)).axes

    assert bar.get_ylim() == pytest.approx((0.8, 0.95))
    assert bar.get_ylabel() == "IoU"


def test_colours_that_run_nowhere_are_refused():
    with pytest.raises(ValueError, match="lower value to a higher"):
        comparison_figure(_comparison(), (0.5, 0.5))


def _panel(scale: str = "linear"):
    """Return a drawn figure of one panel from 1 to 100 each way, and that panel."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, 3))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_yscale(scale)
    axes.set_xlim(1, 100)
    axes.set_ylim(1, 100)
    figure.canvas.draw()
    return figure, axes


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_the_wheel_zooms_about_the_pointer_as_the_axis_looks(scale):
    """Measured on the screen: a logarithmic loss zooms about the value pointed at too."""
    from am_segmentation_manager.plots import zoom_about

    figure, axes = _panel(scale)
    x, y = axes.transData.transform((25.0, 10.0))

    assert zoom_about(figure, x, y, 2.0)

    assert axes.transData.transform((25.0, 10.0)) == pytest.approx((x, y))
    low, high = axes.get_xlim()
    assert high - low == pytest.approx(99 / 2)
    bottom, top = axes.get_ylim()
    assert 1 < bottom < 10 < top < 100

    zoom_about(figure, x, y, 0.5)
    assert axes.get_xlim() == pytest.approx((1, 100))
    assert axes.get_ylim() == pytest.approx((1, 100))


def test_the_wheel_beside_every_panel_changes_nothing():
    from am_segmentation_manager.plots import zoom_about

    figure, axes = _panel()

    assert not zoom_about(figure, 1.0, 1.0, 2.0)
    assert axes.get_xlim() == (1, 100)


def test_an_axis_that_reads_downwards_stays_that_way():
    """The table of runs counts its rows from the top."""
    from am_segmentation_manager.plots import zoom_about

    figure, axes = _panel()
    axes.set_ylim(100, 1)
    figure.canvas.draw()
    x, y = axes.transData.transform((50.0, 50.0))

    zoom_about(figure, x, y, 2.0)

    bottom, top = axes.get_ylim()
    assert bottom > top


def test_a_zoom_factor_must_be_above_nothing():
    from am_segmentation_manager.plots import zoom_about

    figure, _ = _panel()
    with pytest.raises(ValueError, match="above 0"):
        zoom_about(figure, 10, 10, 0)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_the_wheel_never_zooms_out_beyond_the_view_it_is_held_within(scale):
    """Zoomed in off-centre and out again, it comes to rest on the view Reset view gives."""
    from am_segmentation_manager.plots import view_limits, zoom_about

    figure, axes = _panel(scale)
    drawn_at = view_limits(figure)
    x, y = axes.transData.transform((80.0, 60.0))

    zoom_about(figure, x, y, 0.5, drawn_at)
    assert axes.get_xlim() == pytest.approx((1, 100))
    assert axes.get_ylim() == pytest.approx((1, 100))

    zoom_about(figure, x, y, 3.0, drawn_at)
    for _ in range(4):
        zoom_about(figure, axes.bbox.x0 + 5, axes.bbox.y1 - 5, 0.5, drawn_at)
    assert axes.get_xlim() == pytest.approx((1, 100))
    assert axes.get_ylim() == pytest.approx((1, 100))

