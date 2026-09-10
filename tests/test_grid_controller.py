"""Tests for the Grid view's model: which overviews a project offers.

No Qt: this is the layer that decides which overview opens, what each one is
called, and why nothing is placed where nothing is. Built on scanned projects
whose overviews carry the header Autolamella writes, so the geometry is read
the way it is off real data.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from am_segmentation_manager.scanner import Scanner

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile", "yaml")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile
    import yaml

    from am_segmentation_manager.gui.grid_controller import GridController

POSITIONED = {
    "name": "grid",
    "positions": [{
        "petname": "01-ruling-ibex",
        "state": {"microscope_state": {"stage_position": {
            "x": 20e-6, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0,
        }}},
    }],
}


def header(
    *, beam: str = "ELECTRON", when: str | float | None = None, geometry: bool = True
) -> str:
    """Return the description Autolamella writes into an overview."""
    state: dict = {
        "electron_beam": {"scan_rotation": 0.0},
        "ion_beam": {"scan_rotation": 0.0},
    }
    if when is not None:
        state["timestamp"] = when
    if geometry:
        state["stage_position"] = {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0}
    return json.dumps({
        "image": {"beam_type": beam, "resolution": [256, 256]},
        "pixel_size": {"x": 1e-6, "y": 1e-6},
        "microscope_state": state,
    })


def make_project(tmp_path: Path, *, positioned: bool = True) -> tuple[Path, Path]:
    """Return the scan root and a project with one lamella, holding no overview yet."""
    root = tmp_path / "scan"
    project = root / "AutoLamella-2026-07-28-10-31"
    stage = project / "01-ruling-ibex" / "adaptive_polish_2026-07-28-11-34-24AM" / "sem"
    stage.mkdir(parents=True)
    tifffile.imwrite(str(stage / "img_000_SEM.tif"), np.full((64, 96), 100, dtype=np.uint8))
    (project / "experiment.yaml").write_text(
        yaml.safe_dump(POSITIONED if positioned else {"name": "grid"}), encoding="utf-8"
    )
    return root, project


def add_overview(project: Path, name: str, **described) -> None:
    tifffile.imwrite(
        str(project / name),
        np.full((256, 256), 40, dtype=np.uint8),
        description=header(**described),
    )


def grid_for(root: Path) -> GridController:
    grid = GridController()
    grid.set_project(Scanner(root).build_index().all_projects()[0])
    return grid


def test_the_earliest_overview_that_places_anything_opens(tmp_path: Path):
    """Taken before milling or coating, it is usually the clearest picture of
    where the lamellae are; one that places nothing is offered, not opened."""
    root, project = make_project(tmp_path)
    add_overview(project, "overview-hand-saved.tif", when="2026-07-28 10:00:00", geometry=False)
    add_overview(project, "overview-image-2026-07-28_10-31-00.tif", when="2026-07-28 10:31:00")
    add_overview(project, "overview-image-2026-07-28_12-00-00.tif", when="2026-07-28 12:00:00")

    grid = grid_for(root)

    assert [choice.overview.record.image_path.name for choice in grid.choices] == [
        "overview-hand-saved.tif",
        "overview-image-2026-07-28_10-31-00.tif",
        "overview-image-2026-07-28_12-00-00.tif",
    ]
    assert grid.chosen == 1
    assert [marker.name for marker in grid.placement.markers] == ["01-ruling-ibex"]


def test_choosing_an_overview_shows_that_one(tmp_path: Path):
    root, project = make_project(tmp_path)
    add_overview(project, "overview-image-2026-07-28_10-31-00.tif", when="2026-07-28 10:31:00")
    add_overview(project, "overview-image-2026-07-28_12-00-00.tif", when="2026-07-28 12:00:00")
    grid = grid_for(root)

    grid.choose(1)

    assert grid.current.overview.record.image_path.name == (
        "overview-image-2026-07-28_12-00-00.tif"
    )
    assert grid.placement is not None


def test_an_overview_with_no_positions_recorded_is_offered_and_says_so(tmp_path: Path):
    """The picture is still the grid; only the rings are missing, and why."""
    root, project = make_project(tmp_path, positioned=False)
    add_overview(project, "overview-image-2026-07-28_10-31-00.tif")

    grid = grid_for(root)

    assert grid.current is not None
    assert grid.placement is None
    assert "no lamella positions" in grid.current.reason


def test_nothing_is_placed_on_an_ion_beam_overview(tmp_path: Path):
    """The projection is the electron beam's; a tilted ion column foreshortens
    the grid too far for a mark to point at the right square."""
    root, project = make_project(tmp_path)
    add_overview(project, "overview-image-2026-07-28_10-31-00.tif", beam="ION")

    grid = grid_for(root)

    assert grid.placement is None
    assert "ion-beam" in grid.current.reason


def test_an_overview_is_named_by_when_by_which_beam_and_how_large(tmp_path: Path):
    """Two taken within a minute of each other differ by their size."""
    root, project = make_project(tmp_path)
    add_overview(project, "overview-image-2026-07-28_10-31-00.tif", when="2026-07-28 10:31:00")

    grid = grid_for(root)

    assert grid.current.caption == "2026-07-28 10:31 · SEM · 256 px"


def test_a_project_holding_no_overview_offers_none(tmp_path: Path):
    root, _project = make_project(tmp_path)

    grid = grid_for(root)

    assert grid.choices == ()
    assert grid.current is None
    assert grid.placement is None


def test_a_time_read_as_utc_says_so_in_the_name(tmp_path: Path):
    """A stitched overview's time is seconds since the epoch, read as UTC; left
    unsaid, it reads an hour off the time in its own file name."""
    root, project = make_project(tmp_path)
    add_overview(project, "overview-image-2026-04-08_10-04-12.tif", when=1775639052.187201)

    grid = grid_for(root)

    assert grid.current.caption == "2026-04-08 09:04 UTC · SEM · 256 px"

