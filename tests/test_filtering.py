"""Tests for the filtering module."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.filtering import (
    extract_image_number,
    matches_image,
    matches_lamella,
    matches_project,
    matches_stage,
    parse_scope_number,
    parse_selector_values,
    record_matches_paths,
)
from am_segmentation_manager.models import AdaptiveMillingStage, ImageRecord, Lamella, Project


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
    stg = AdaptiveMillingStage(name="Adaptive_polish_1", path=Path("/fake"), number="01")
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
