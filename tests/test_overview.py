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
import statistics
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
    system = case["system"]
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
        beam=case["beam_type"],
        shuttle_pretilt=math.radians(system["shuttle_pre_tilt_deg"]),
        rotation_reference=math.radians(system["rotation_reference_deg"]),
        rotation_180=math.radians(system["rotation_180_deg"]),
        electron_column_tilt=math.radians(system["electron_column_tilt_deg"]),
        ion_column_tilt=math.radians(system["ion_column_tilt_deg"]),
        compustage="Arctis" in (system["model"] or ""),
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
    """The ion beam, the stage facing the ion column, a pre-tilt below 45
    degrees, a compustage either way up and no scan rotation.

    No overview in the collection reaches these, so they would otherwise rest
    on a reading of fibsemOS's source rather than on running it.
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
# Agreement with a person
# ---------------------------------------------------------------------------

#: How far a mark may land from where a person placed the lamella, in
#: micrometres: about one grid square, 125um being a 200-mesh pitch. A mark this
#: close still points a reader at the lamella's square or the next.
MARK_TOLERANCE_UM = 125.0

#: How far the marks on one overview may land from a person's placements, as a
#: median, in micrometres. The flat projection this replaced was 67-80um off on
#: the Aquilos overviews with a pre-tilted shuttle.
OVERVIEW_MEDIAN_UM = 35.0

#: Placements that disagree with the projection for a reason not yet
#: understood, named here rather than hidden by a looser tolerance.
UNEXPLAINED_MARKS = {
    "overview-image-2025-07-22_09-43-03.tif::03-causal-ghost": (
        "placed 268um from its mark where every neighbour is within 35um: the "
        "reference image may not have been taken where its header says"
    ),
}
UNEXPLAINED_OVERVIEWS = {
    "overview-image-2026-04-08_10-04-12.tif": (
        "every mark sits 50-73px above its lamella, a constant offset rather "
        "than a scale error, which points at the overview's recorded centre"
    ),
}


def placed_by_a_person() -> list[tuple[str, dict, dict]]:
    """Return each placement made independently of the projection.

    A placement that is the projection's own first guess, unmoved, says nothing
    about the projection; one the image match found and a person confirmed
    over the overview does.
    """
    found = []
    for case in load("overview_ground_truth.json")["cases"]:
        for entry in case["positions"]:
            if entry["answer"] != "placed":
                continue
            projected_x, projected_y = entry["projected"]
            if entry["first_guess"] == "projected" and math.hypot(
                entry["x"] - projected_x, entry["y"] - projected_y
            ) < 0.5:
                continue
            found.append((f"{Path(case['overview']).name}::{entry['name']}", case, entry))
    return found


def error_um(case: dict, entry: dict) -> float:
    """Return how far the projection lands from a person's placement, in micrometres."""
    position = entry["position"]
    x, y = reproject(
        as_overview(case),
        StagePosition(
            x=position["x"], y=position["y"], z=position["z"],
            rotation=position["r"], tilt=position["t"], name=entry["name"],
        ),
    )
    return math.hypot(x - entry["x"], y - entry["y"]) * case["pixel_size_m"] * 1e6


PLACED = placed_by_a_person()


def _unexplained(reasons: dict, key: str) -> tuple:
    return (pytest.mark.xfail(strict=True, reason=reasons[key]),) if key in reasons else ()


@pytest.mark.parametrize(
    "case, entry",
    [
        pytest.param(case, entry, id=label, marks=_unexplained(UNEXPLAINED_MARKS, label))
        for label, case, entry in PLACED
    ],
)
def test_every_mark_lands_within_a_grid_square_of_where_a_person_placed_it(case, entry):
    """Placed by eye, lining the lamella's own SEM reference image up with the
    overview's grid bars in ``scripts/refine_overview_positions.py``."""
    assert error_um(case, entry) <= MARK_TOLERANCE_UM


def _overview_errors() -> dict[str, list[float]]:
    errors: dict[str, list[float]] = {}
    for _label, case, entry in PLACED:
        errors.setdefault(Path(case["overview"]).name, []).append(error_um(case, entry))
    return errors


@pytest.mark.parametrize(
    "errors",
    [
        pytest.param(errors, id=name, marks=_unexplained(UNEXPLAINED_OVERVIEWS, name))
        for name, errors in _overview_errors().items()
    ],
)
def test_each_overview_places_its_lamellae_to_within_tens_of_microns(errors):
    """What the perspective correction bought, held: the flat projection
    passes the grid-square test on most lamellae and fails this one."""
    assert statistics.median(errors) <= OVERVIEW_MEDIAN_UM


def test_enough_placements_are_independent_of_the_projection():
    """Leaving out unmoved guesses must not leave nothing to test."""
    assert len(PLACED) >= 30


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


def test_a_grid_flat_to_the_beam_is_seen_at_full_length():
    assert reproject(upright(), at(y=50e-6))[1] == pytest.approx(350.0)


def test_a_tilted_grid_is_foreshortened_by_the_angle_the_beam_sees_it_at():
    """The distance along the grid shrinks by the cosine of the angle between
    the grid and the beam."""
    tilted = upright(
        centre=StagePosition(x=0.0, y=0.0, z=0.0, rotation=0.0, tilt=math.radians(30))
    )

    offset = reproject(tilted, at(y=50e-6, tilt=math.radians(30)))[1] - 400.0

    assert offset == pytest.approx(-50.0 * math.cos(math.radians(30)))


def test_an_ion_beam_overview_sees_the_grid_at_its_column_s_angle():
    """A grid flat to the electron column is 52 degrees off the ion one."""
    ion = upright(beam="ION")

    offset = reproject(ion, at(y=50e-6))[1] - 400.0

    assert offset == pytest.approx(-50.0 * math.cos(math.radians(52)))


def test_the_shuttle_pre_tilt_is_taken_out_of_the_stage_move():
    """The shuttle holds the grid at its pre-tilt, so moving the stage along y
    moves the grid further than y along its own surface."""
    pretilted = upright(
        shuttle_pretilt=math.radians(35),
        centre=StagePosition(x=0.0, y=0.0, z=0.0, rotation=0.0, tilt=math.radians(35)),
    )

    offset = reproject(pretilted, at(y=50e-6, tilt=math.radians(35)))[1] - 400.0

    assert offset == pytest.approx(-50.0 / math.cos(math.radians(35)))


def test_a_half_turn_of_scan_rotation_mirrors_both_axes():
    turned = upright(scan_rotation=math.pi)

    assert reproject(turned, at(x=10e-6, y=10e-6)) == pytest.approx(
        (490.0, 410.0)
    )
    assert reproject(upright(), at(x=10e-6, y=10e-6)) == pytest.approx(
        (510.0, 390.0)
    )


def test_a_compustage_at_no_tilt_is_seen_as_a_flat_stage():
    """It images the back of the grid, half a turn over, and at no tilt the
    two turns cancel."""
    assert reproject(upright(compustage=True), at(x=10e-6, y=10e-6)) == pytest.approx(
        reproject(upright(), at(x=10e-6, y=10e-6))
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
        beam_type="ELECTRON", shuttle_pretilt_deg=35.0,
        stage_rotation_reference_deg=0.0, stage_rotation_180_deg=180.0,
        electron_column_tilt_deg=0.0, ion_column_tilt_deg=52.0,
        microscope_model="Aquilos",
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
    assert resolved.shuttle_pretilt == pytest.approx(math.radians(35))
    assert resolved.ion_column_tilt == pytest.approx(math.radians(52))
    assert resolved.compustage is False


def test_an_arctis_is_known_as_a_compustage():
    """fibsemOS recognises one by its model name, the header saying no more."""
    resolved = overview_from_metadata(Path("o.tif"), metadata(microscope_model="Arctis"))

    assert resolved.compustage is True


@pytest.mark.parametrize(
    "missing",
    ["image_width", "image_height", "pixel_size_nm", "stage_x_um", "stage_y_um",
     "stage_z_um", "stage_rotation_deg", "stage_tilt_deg", "scan_rotation_deg",
     "beam_type", "shuttle_pretilt_deg", "stage_rotation_reference_deg",
     "stage_rotation_180_deg", "electron_column_tilt_deg", "ion_column_tilt_deg"],
)
def test_a_header_missing_any_field_places_nothing(missing: str):
    """A projection missing any of them would be a guess. An overview saved
    from the microscope's own interface states no pre-tilt or column geometry."""
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
