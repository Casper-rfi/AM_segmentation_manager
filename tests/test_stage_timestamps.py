"""Tests for stage timestamp parsing, ordering, and plot discovery."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from am_segmentation_manager.scanner import Scanner


def _write_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_project(base: Path) -> Path:
    """Create a minimal project and return the lamella folder."""
    project_root = base / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")
    return project_root / "01-test-lamella"


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "folder_name, expected",
    [
        # 24-hour names, unchanged behaviour
        ("Adaptive_polish_2026-08-07-15-30-00", datetime(2026, 8, 7, 15, 30, 0)),
        ("adaptive_polish_2026-07-28-09-05-01", datetime(2026, 7, 28, 9, 5, 1)),
        # 12-hour names written by Autolamella for adaptive milling stages
        ("adaptive_polish_2026-07-28-02-43-32PM", datetime(2026, 7, 28, 14, 43, 32)),
        ("adaptive_polish_2026-07-28-11-15-36AM", datetime(2026, 7, 28, 11, 15, 36)),
        # Noon and midnight are the cases a naive +12 gets wrong
        ("adaptive_polish_2026-07-28-12-05-03PM", datetime(2026, 7, 28, 12, 5, 3)),
        ("adaptive_polish_2026-07-28-12-05-03AM", datetime(2026, 7, 28, 0, 5, 3)),
        # Case insensitivity
        ("adaptive_polish_2026-07-28-02-43-32pm", datetime(2026, 7, 28, 14, 43, 32)),
    ],
)
def test_timestamp_parsing(folder_name, expected):
    assert Scanner._timestamp_from_filename(folder_name) == expected


def test_timestamp_without_a_timestamp_is_none():
    assert Scanner._timestamp_from_filename("adaptive_polish") is None


def test_contradictory_meridiem_is_ignored():
    """A 24-hour hour with a PM suffix must not become an invalid hour."""
    assert Scanner._timestamp_from_filename(
        "adaptive_polish_2026-07-28-13-00-00PM"
    ) == datetime(2026, 7, 28, 13, 0, 0)


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------

def test_afternoon_stage_sorts_after_morning_stage(tmp_path: Path):
    """The regression this fix targets: 02-43-32PM is 14:43, not 02:43."""
    base = tmp_path / "workspace"
    lamella = _make_project(base)
    afternoon = lamella / "adaptive_polish_2026-07-28-02-43-32PM"
    morning = lamella / "adaptive_polish_2026-07-28-11-15-36AM"
    _write_file(afternoon / "sem" / "img_000_SEM.tif")
    _write_file(morning / "sem" / "img_000_SEM.tif")

    stages = Scanner(base).build_index().all_projects()[0].lamellae[0].stages

    assert [stage.name for stage in stages] == [morning.name, afternoon.name]
    assert [stage.number for stage in stages] == ["01", "02"]


def test_stages_across_midday_boundary_order_correctly(tmp_path: Path):
    base = tmp_path / "workspace"
    lamella = _make_project(base)
    names = [
        "adaptive_polish_2026-07-28-11-15-36AM",
        "adaptive_polish_2026-07-28-12-05-03PM",
        "adaptive_polish_2026-07-28-01-30-00PM",
        "adaptive_polish_2026-07-28-12-30-00AM",
    ]
    for name in names:
        _write_file(lamella / name / "sem" / "img_000_SEM.tif")

    stages = Scanner(base).build_index().all_projects()[0].lamellae[0].stages

    assert [stage.name for stage in stages] == [
        "adaptive_polish_2026-07-28-12-30-00AM",  # 00:30
        "adaptive_polish_2026-07-28-11-15-36AM",  # 11:15
        "adaptive_polish_2026-07-28-12-05-03PM",  # 12:05
        "adaptive_polish_2026-07-28-01-30-00PM",  # 13:30
    ]


# ---------------------------------------------------------------------------
# Plot discovery
# ---------------------------------------------------------------------------

def test_scanner_discovers_stage_plots(tmp_path: Path):
    base = tmp_path / "workspace"
    lamella = _make_project(base)
    stage = lamella / "adaptive_polish_2026-07-28-11-15-36AM"
    _write_file(stage / "sem" / "lam_AP_img_000_SEM.tif")
    _write_file(stage / "plots" / "lam_AP_img_001_plot.png")
    _write_file(stage / "plots" / "lam_AP_img_000_plot.png")

    stages = Scanner(base).build_index().all_projects()[0].lamellae[0].stages

    assert [plot.name for plot in stages[0].plots] == [
        "lam_AP_img_000_plot.png",
        "lam_AP_img_001_plot.png",
    ]


def test_stage_without_plots_folder_has_no_plots(tmp_path: Path):
    base = tmp_path / "workspace"
    lamella = _make_project(base)
    stage = lamella / "adaptive_polish_1"
    _write_file(stage / "sem" / "img_000_SEM.tif")

    stages = Scanner(base).build_index().all_projects()[0].lamellae[0].stages

    assert stages[0].plots == []


def test_plot_discovery_ignores_non_png_files(tmp_path: Path):
    base = tmp_path / "workspace"
    lamella = _make_project(base)
    stage = lamella / "adaptive_polish_1"
    _write_file(stage / "sem" / "img_000_SEM.tif")
    _write_file(stage / "plots" / "img_000_plot.png")
    _write_file(stage / "plots" / "notes.txt")

    stages = Scanner(base).build_index().all_projects()[0].lamellae[0].stages

    assert [plot.name for plot in stages[0].plots] == ["img_000_plot.png"]


def test_plots_folder_alone_does_not_create_a_stage(tmp_path: Path):
    """Stage detection still depends on sem/ or fib/, not on plots/."""
    base = tmp_path / "workspace"
    lamella = _make_project(base)
    _write_file(lamella / "not_a_stage" / "plots" / "img_000_plot.png")

    project = Scanner(base).build_index().all_projects()[0]

    assert project.lamellae == []
