"""Tests for the filtering module."""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.filtering import (
    extract_image_number,
    matches_image,
    matches_lamella,
    matches_project,
    matches_stage,
    matches_state,
    parse_scope_number,
    parse_selector_values,
    record_matches_paths,
)
from am_segmentation_manager.models import ImageRecord, Lamella, Project, Stage
from am_segmentation_manager.state import STATE_VALUES


def test_parse_scope_number():
    assert parse_scope_number("1") == "01"
    assert parse_scope_number("01") == "01"
    assert parse_scope_number("001") == "01"
    assert parse_scope_number("abc") is None
    assert parse_scope_number(None) is None


def test_parse_selector_values():
    numerics, names = parse_selector_values(["1-3,5", "all", "special_lamella"])
    assert numerics == {1, 2, 3, 5}
    assert names == {"all", "special_lamella"}


def test_extract_image_number():
    assert extract_image_number("01-lamella_AP_img_013_SEM.tif") == 13
    assert extract_image_number("img_005.png") == 5
    assert extract_image_number("unknown_filename.tif") is None


def test_extract_image_number_reads_the_2023_beam_form():
    """``Image_<beam>_<number>``: both beams of one acquisition give one number."""
    assert extract_image_number("Image_1_000.tif") == 0
    assert extract_image_number("Image_2_000.tif") == 0
    assert extract_image_number("Image_1_044.tif") == 44


def test_extract_image_number_leaves_the_other_naming_eras_alone():
    """One number, or a non-numeric tail, is not the beam form and carries none."""
    assert extract_image_number("Image_000.tif") is None
    assert extract_image_number("Image_000_TLD.tif") is None
    assert extract_image_number("lamella02_01.tif") is None
    # A timestamp must not be read as a beam and an acquisition number.
    assert extract_image_number("Image_2023_0811.tif") is None
    # Plot naming stays unmatched; see docs/04_future_work.md.
    assert extract_image_number("GLM_Plot_0.png") is None
    assert extract_image_number("Image_000.tif_00.png") is None


def test_matches_project():
    proj = Project(name="MyProject", path=Path("/fake/path"))
    proj.number = "001"
    assert matches_project(proj, ["1"]) is True
    assert matches_project(proj, ["MyProject"]) is True
    assert matches_project(proj, ["OtherProject"]) is False


def test_matches_lamella():
    lam = Lamella(name="01-ruling-ibex", path=Path("/fake"), number="01")
    assert matches_lamella(lam, ["1"]) is True
    assert matches_lamella(lam, ["01-ruling-ibex"]) is True
    assert matches_lamella(lam, ["2"]) is False


def test_matches_stage():
    stg = Stage(name="Adaptive_polish_1", path=Path("/fake"), number="01")
    assert matches_stage(stg, ["1"]) is True
    assert matches_stage(stg, ["Adaptive_polish_1"]) is True
    assert matches_stage(stg, ["2"]) is False


def test_matches_image():
    rec = ImageRecord(image_path=Path("/fake/AP_img_013_SEM.tif"))
    assert matches_image(rec, ["13"]) is True
    assert matches_image(rec, ["AP_img_013_SEM.tif"]) is True
    assert matches_image(rec, ["14"]) is False


def test_record_matches_paths():
    p = Path("/fake/img.tif")
    rec = ImageRecord(image_path=p)
    assert record_matches_paths(rec, [str(p)]) is True
    assert record_matches_paths(rec, ["/fake/other.tif"]) is False
    assert record_matches_paths(rec, None) is True


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", STATE_VALUES)
def test_a_record_matches_exactly_one_state(state):
    """The three states are mutually exclusive, so a state filter partitions the
    collection. A record matching two would put one image in two exports."""
    record = ImageRecord(image_path=Path("sem/img.tif"), state=state)

    matched = [value for value in STATE_VALUES if matches_state(record, value)]

    assert matched == [state]


@pytest.mark.parametrize("state", STATE_VALUES)
def test_every_state_matches_all(state):
    assert matches_state(ImageRecord(image_path=Path("sem/img.tif"), state=state), "all")


def test_a_state_that_is_not_one_matches_nothing():
    """A typo in --state narrows to nothing rather than quietly meaning 'all'."""
    record = ImageRecord(image_path=Path("sem/img.tif"))

    assert not matches_state(record, "Selected")
    assert not matches_state(record, "")
