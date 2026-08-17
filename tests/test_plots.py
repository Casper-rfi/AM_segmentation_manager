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
    bar_figure,
    has_plot_support,
    heatmap_figure,
    plot_counts,
    plot_cross_tabulation,
    require_plot_support,
)

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
    figure = heatmap_figure(TABLE, "microscope", "modality", normalise=True)

    texts = {text.get_text() for text in figure.axes[0].texts}
    assert "50%" in texts        # Aquilos is half SEM, half FIB
    assert "images" not in figure.axes[-1].get_ylabel()


def test_a_row_with_nothing_in_it_is_not_a_division_by_zero():
    """A cross-tabulation can carry a row the filters emptied."""
    figure = heatmap_figure({("a", "x"): 0, ("b", "x"): 3}, "r", "c", normalise=True)

    assert {text.get_text() for text in figure.axes[0].texts} == {"0%", "100%"}


def test_a_combination_nothing_was_acquired_for_is_drawn_empty():
    """The blank cells are the finding, so zero has to be the palette's floor."""
    figure = heatmap_figure({("a", "x"): 5, ("b", "y"): 5}, "r", "c")
    image = figure.axes[0].images[0]

    assert image.norm.vmin == 0


def test_the_missing_dependency_error_says_how_to_fix_it(monkeypatch):
    import am_segmentation_manager.plots as plots

    monkeypatch.setattr(plots, "has_plot_support", lambda: False)
    with pytest.raises(RuntimeError, match="pip install"):
        require_plot_support()
