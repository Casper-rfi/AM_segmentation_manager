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
    has_plot_support,
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


def test_a_cross_tabulation_is_drawn_grouped(tmp_path: Path):
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
    """Charts are drawn from the command line, often over ssh. Choosing the Agg
    backend before pyplot is imported is what makes that work."""
    import matplotlib

    plot_counts(COUNTS, "t", tmp_path / "chart.png")

    assert matplotlib.get_backend().lower() == "agg"


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


def test_the_missing_dependency_error_says_how_to_fix_it(monkeypatch):
    import am_segmentation_manager.plots as plots

    monkeypatch.setattr(plots, "has_plot_support", lambda: False)
    with pytest.raises(RuntimeError, match="pip install"):
        require_plot_support()
