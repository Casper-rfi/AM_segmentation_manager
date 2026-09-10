"""Tests for placing a lamella's stage coordinates on a grid overview.

The projection is a reimplementation of fibsemOS's, so the tests that matter
are the ones holding it to fibsemOS's own answers: ``data/*.json`` carry real
overview headers with the coordinates fibsem 0.5.1 computed for them, captured
in the second environment, and the suite here runs with no fibsem installed.
That is what makes "is this still the same projection?" a question this suite
answers rather than an assumption.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from am_segmentation_manager.image_metadata import ImageMetadata
from am_segmentation_manager.models import ImageRecord, Lamella, Project
from am_segmentation_manager.overview import (
    Overview,
    StagePosition,
    is_inside,
    is_overview_name,
    overview_from_metadata,
    place_lamellae,
    project_overviews,
    read_positions,
    reproject,
)

DATA = Path(__file__).parent / "data"

#: How far a reprojected point may sit from fibsemOS's own answer, in pixels.
#: Not a tolerance on the projection but on the arithmetic: the two compute the
#: same expression in the same double precision, and anything above rounding
#: means they have stopped agreeing.
TOLERANCE_PX = 1e-6


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def as_overview(case: dict) -> Overview:
    centre = case["centre"]
    return Overview(
        path=Path(case.get("overview", case.get("label", "overview.tif"))),
        width=case["width"],
        height=case["height"],
        pixel_size_m=case["pixel_size_m"],
        centre=StagePosition(
            x=centre["x"], y=centre["y"], z=centre["z"],
            rotation=centre["r"], tilt=centre["t"],
        ),
        scan_rotation=case["scan_rotation_rad"],
    )


def cases(name: str) -> list[tuple[str, dict, dict]]:
    document = load(name)
    return [
        (f"{case.get('overview', case.get('label'))}::{entry['name']}", case, entry)
        for case in document["cases"]
        for entry in case["positions"]
    ]


REAL = cases("fibsem_reprojection.json")
BRANCHES = cases("fibsem_reprojection_branches.json")


# ---------------------------------------------------------------------------
# Agreement with fibsemOS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, case, entry", REAL, ids=[row[0] for row in REAL])
def test_overview_matches_fibsem(label: str, case: dict, entry: dict):
    """Every lamella of every overview in the reference collection."""
    position = entry["position"]
    point = reproject(
        as_overview(case),
        StagePosition(
            x=position["x"], y=position["y"], z=position["z"],
            rotation=position["r"], tilt=position["t"], name=entry["name"],
        ),
    )

    assert point[0] == pytest.approx(entry["fibsem_x"], abs=TOLERANCE_PX)
    assert point[1] == pytest.approx(entry["fibsem_y"], abs=TOLERANCE_PX)


@pytest.mark.parametrize("label, case, entry", BRANCHES, ids=[row[0] for row in BRANCHES])
def test_overview_matches_fibsem_where_real_data_does_not_reach(label, case, entry):
    """The scan rotation and compustage branches.

    Every real overview carries a half-turn scan rotation and an upright stage,
    so these two sign flips would otherwise rest on a reading of fibsemOS's
    source rather than on running it.
    """
    position = entry["position"]
    point = reproject(
        as_overview(case),
        StagePosition(
            x=position["x"], y=position["y"], z=position["z"],
            rotation=position["r"], tilt=position["t"], name=entry["name"],
        ),
    )

    assert point[0] == pytest.approx(entry["fibsem_x"], abs=TOLERANCE_PX)
    assert point[1] == pytest.approx(entry["fibsem_y"], abs=TOLERANCE_PX)


def test_the_fixtures_cover_every_overview_that_can_place_a_lamella():
    """A shrinking fixture would weaken the comparison without failing it."""
    document = load("fibsem_reprojection.json")

    assert len(document["cases"]) >= 35
    assert sum(len(case["positions"]) for case in document["cases"]) >= 265


# ---------------------------------------------------------------------------
# What the projection does
# ---------------------------------------------------------------------------

def upright(**overrides) -> Overview:
    """An overview with no scan rotation and an upright stage."""
    defaults = dict(
        path=Path("overview.tif"), width=1000, height=800, pixel_size_m=1e-6,
        centre=StagePosition(x=0.0, y=0.0, z=0.0, rotation=0.0, tilt=0.0),
        scan_rotation=0.0,
    )
    return Overview(**{**defaults, **overrides})


def at(x=0.0, y=0.0, z=0.0, rotation=0.0, tilt=0.0, name=None) -> StagePosition:
    return StagePosition(x=x, y=y, z=z, rotation=rotation, tilt=tilt, name=name)


def test_the_centre_of_the_stage_is_the_centre_of_the_image():
    assert reproject(upright(), at()) == (500.0, 400.0)


def test_moving_the_stage_moves_the_point_by_the_pixel_size():
    """One pixel is one pixel_size_m of stage travel."""
    assert reproject(upright(), at(x=10e-6))[0] == pytest.approx(510.0)


def test_height_and_depth_are_seen_as_one_distance():
    """The stage moves in three dimensions and the image has two: raising the
    stage and moving it away look alike from above."""
    flat = reproject(upright(), at(y=50e-6))
    raised = reproject(upright(), at(y=30e-6, z=40e-6))

    assert raised == pytest.approx(flat)


def test_a_half_turn_of_scan_rotation_mirrors_both_axes():
    turned = upright(scan_rotation=math.pi)

    assert reproject(turned, at(x=10e-6, y=10e-6)) == pytest.approx(
        (490.0, 410.0)
    )
    assert reproject(upright(), at(x=10e-6, y=10e-6)) == pytest.approx(
        (510.0, 390.0)
    )


def test_a_compustage_mounted_upside_down_mirrors_only_y():
    inverted = upright(
        centre=StagePosition(x=0.0, y=0.0, z=0.0, rotation=0.0, tilt=math.radians(-180))
    )

    assert reproject(inverted, at(x=10e-6, y=10e-6, tilt=math.radians(-180))) == (
        pytest.approx((510.0, 410.0))
    )


def test_a_lamella_off_the_overview_is_placed_rather_than_refused():
    """A real answer: a grid can be moved between the overview and the milling."""
    point = reproject(upright(), at(x=5e-3))

    assert point[0] > 1000
    assert not is_inside(upright(), point)


def test_a_position_facing_the_other_way_is_refused():
    """fibsemOS transforms these through a calibration measured on one
    microscope, which is not a constant this package can carry."""
    with pytest.raises(ValueError, match="cannot be placed on it"):
        reproject(upright(), at(rotation=math.pi, name="01-ruling-ibex"))


def test_a_small_difference_in_rotation_is_still_the_same_way_round():
    """Recorded rotations differ in the last decimals between one lamella and
    the overview; refusing on that would refuse every real project."""
    assert reproject(upright(), at(rotation=math.radians(1.5))) == (500.0, 400.0)


def test_is_inside_reads_the_overview_bounds():
    overview = upright()

    assert is_inside(overview, (0.0, 0.0))
    assert is_inside(overview, (999.0, 799.0))
    assert not is_inside(overview, (1000.0, 400.0))
    assert not is_inside(overview, (-1.0, 400.0))


# ---------------------------------------------------------------------------
# Reading the geometry out of a header
# ---------------------------------------------------------------------------

def metadata(**overrides) -> ImageMetadata:
    defaults = dict(
        image_width=1000, image_height=800, pixel_size_nm=1000.0,
        stage_x_um=1.0, stage_y_um=2.0, stage_z_um=3.0,
        stage_rotation_deg=180.0, stage_tilt_deg=0.0, scan_rotation_deg=180.0,
    )
    return ImageMetadata(**{**defaults, **overrides})


def test_the_header_units_are_converted_to_the_instrument_s_own():
    """The header states nanometres and micrometres and degrees; the projection
    works in metres and radians, which is what the instrument records."""
    resolved = overview_from_metadata(Path("o.tif"), metadata())

    assert resolved is not None
    assert resolved.pixel_size_m == pytest.approx(1e-6)
    assert resolved.centre.x == pytest.approx(1e-6)
    assert resolved.centre.rotation == pytest.approx(math.pi)
    assert resolved.scan_rotation == pytest.approx(math.pi)


@pytest.mark.parametrize(
    "missing",
    ["image_width", "image_height", "pixel_size_nm", "stage_x_um", "stage_y_um",
     "stage_z_um", "stage_rotation_deg", "stage_tilt_deg", "scan_rotation_deg"],
)
def test_a_header_missing_any_field_places_nothing(missing: str):
    """A projection missing any of them would be a guess. An overview saved by
    hand from the microscope's own interface carries no header at all."""
    assert overview_from_metadata(Path("o.tif"), metadata(**{missing: None})) is None


def test_a_header_stating_a_zero_pixel_size_places_nothing():
    assert overview_from_metadata(Path("o.tif"), metadata(pixel_size_nm=0.0)) is None


# ---------------------------------------------------------------------------
# Reading positions out of experiment.yaml
# ---------------------------------------------------------------------------

def experiment(*positions) -> dict:
    return {"positions": list(positions)}


def position_entry(name: str, **stage) -> dict:
    values = {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0, **stage}
    return {
        "petname": name,
        "state": {"microscope_state": {"stage_position": values}},
    }


def test_positions_are_named_by_the_directory_they_were_written_into():
    found = read_positions(experiment(position_entry("01-ruling-ibex", x=1.0)))

    assert [(one.name, one.x) for one in found] == [("01-ruling-ibex", 1.0)]


def test_a_position_falls_back_to_the_last_part_of_its_recorded_path():
    """The same rule tasks.read_vocabularies uses on the same file."""
    entry = position_entry("", x=1.0)
    entry["petname"] = None
    entry["path"] = "C:/data\\\\AutoLamella-2026\\\\02-ruling-oryx"

    assert [one.name for one in read_positions(experiment(entry))] == ["02-ruling-oryx"]


def test_a_lamella_that_was_never_positioned_is_skipped():
    """Autolamella creates an entry before the stage is anywhere; where it was
    is the whole question, so there is nothing to default to."""
    entry = position_entry("01-ruling-ibex")
    entry["state"] = {"microscope_state": {"stage_position": {"x": None}}}

    assert read_positions(experiment(entry)) == []


def test_an_experiment_predating_positions_yields_nothing():
    """Every dataset older than experiment.yaml, which is why a caller checks
    the list rather than assuming one position per lamella."""
    assert read_positions({}) == []
    assert read_positions(None) == []
    assert read_positions({"positions": "not a list"}) == []


# ---------------------------------------------------------------------------
# Finding a project's overviews
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("overview-image-2026-04-17_11-17-34.tif", True),
        ("CB016A1_SEM_overview.tif", True),
        ("Grid01_overview-2025-04-14_11-03-08.tif", True),
        ("overview-image-2026-04-17_11-17-34-autogamma.tif", False),
        ("tile_0_0_eb.tif", False),
        ("ref_Polishing_start_eb.tif", False),
    ],
)
def test_an_overview_is_recognised_and_its_display_copy_is_not(name: str, expected: bool):
    """Autolamella writes the same picture twice, the second with a gamma curve
    for looking at; both would place the same lamellae."""
    assert is_overview_name(name) is expected


def make_project(*paths: Path) -> Project:
    """A project whose root stands in for a lamella, as an indexed one does."""
    root = Path("/scan/AutoLamella-2026")
    lamella = Lamella(name=root.name, path=root, synthetic=True)
    lamella.stages = []
    project = Project(name=root.name, path=root, lamellae=[lamella])
    from am_segmentation_manager.models import Stage

    lamella.stages = [
        Stage(
            name=root.name, path=root, synthetic=True,
            images=[
                ImageRecord(image_path=path, series="other", modality="SEM")
                for path in paths
            ],
        )
    ]
    return project


def test_a_projects_overviews_are_read_off_the_index():
    root = Path("/scan/AutoLamella-2026")
    project = make_project(
        root / "overview-image-2026-04-17_11-17-34.tif",
        root / "overview-image-2026-04-17_11-17-34-autogamma.tif",
        root / "something_new_eb.tif",
    )

    found = project_overviews(project)

    assert [overview.record.image_path.name for overview in found] == [
        "overview-image-2026-04-17_11-17-34.tif"
    ]


def test_an_overview_is_drawn_from_its_gamma_copy_where_there_is_one():
    """The copy is made for looking at and carries the same header, so it is
    the picture while the file as acquired stays the geometry."""
    root = Path("/scan/AutoLamella-2026")
    project = make_project(
        root / "overview-image-2026-04-17_11-17-34.tif",
        root / "overview-image-2026-04-17_11-17-34-autogamma.tif",
        root / "overview-image-2026-04-17_11-20-05.tif",
    )

    found = project_overviews(project)

    assert [overview.display_path.name for overview in found] == [
        "overview-image-2026-04-17_11-17-34-autogamma.tif",
        "overview-image-2026-04-17_11-20-05.tif",
    ]
    assert found[0].record.image_path.name == "overview-image-2026-04-17_11-17-34.tif"


def test_an_overview_inside_a_lamella_is_not_the_projects():
    """It belongs to the project rather than any lamella, which is why it sits
    at the root; one below is something else's."""
    root = Path("/scan/AutoLamella-2026")
    project = make_project(root / "01-ruling-ibex" / "CB016A1_SEM_overview.tif")

    assert project_overviews(project) == []


# ---------------------------------------------------------------------------
# Placing a project's lamellae
# ---------------------------------------------------------------------------

def test_markers_are_fractions_of_the_picture():
    """So whoever draws it may reduce the picture without recomputing them."""
    markers = place_lamellae(upright(), [at(x=10e-6, name="01-ruling-ibex")])

    assert [(one.name, one.x, one.y) for one in markers] == [
        ("01-ruling-ibex", pytest.approx(0.51), pytest.approx(0.5))
    ]


def test_a_lamella_off_the_overview_is_marked_as_outside():
    markers = place_lamellae(upright(), [at(x=5e-3, name="01-ruling-ibex")])

    assert [one.inside for one in markers] == [False]


def test_a_position_that_cannot_be_placed_is_left_out_rather_than_guessed():
    """The whole value of the picture is that a mark on it is where the
    lamella is."""
    markers = place_lamellae(
        upright(),
        [at(name="01-ruling-ibex"), at(rotation=math.pi, name="02-ruling-oryx")],
    )

    assert [one.name for one in markers] == ["01-ruling-ibex"]


def test_a_position_with_no_name_is_left_out():
    """Nothing could say which lamella such a mark belonged to."""
    assert place_lamellae(upright(), [at(x=10e-6)]) == []
