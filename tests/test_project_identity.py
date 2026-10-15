"""Tests for project identity in the scanned index.

Project names are not unique. The real dataset collection holds two different
6 GB projects whose ``experiment.yaml`` files carry identical ``name``,
``created_at`` and ``path`` values, leaving the filesystem path as the only
thing that tells them apart. These tests pin the behaviour that keeps both.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.filtering import iter_matching_records, matches_project
from am_segmentation_manager.models import Project
from am_segmentation_manager.scanner import Scanner

#: The identical metadata the two real duplicate projects share.
SHARED_EXPERIMENT = (
    "created_at: 1753107713.652424\n"
    "method: AutoLamella-OnGrid\n"
    "name: AutoLamella-2025-07-21-15-21_AM01\n"
    "path: Z:/group/user/20250721_AM_vs_nonAM\\AutoLamella-2025-07-21-15-21_AM01\n"
)


def _make_project(root: Path, name: str, image_name: str) -> Path:
    """Build a minimal project whose experiment.yaml is identical to its twin."""
    project = root / name
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / image_name).write_bytes(b"")
    (stage / "sem_labels").mkdir()
    (stage / "sem_labels" / image_name).write_bytes(b"")
    project.mkdir(exist_ok=True)
    (project / "experiment.yaml").write_text(SHARED_EXPERIMENT, encoding="utf-8")
    return project


def test_two_projects_with_identical_metadata_are_both_indexed(tmp_path: Path):
    """Keying the index by name silently discarded one of these."""
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()

    assert len(index.all_projects()) == 2
    assert set(index.projects) == {"AutoLamella-01", "FIB/AutoLamella-01"}


def test_projects_are_retrievable_by_path(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()
    nested = index.find_by_key("FIB/AutoLamella-01")

    assert nested.path == tmp_path / "FIB" / "AutoLamella-01"


def test_lookup_by_an_ambiguous_name_raises_rather_than_guessing(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()

    with pytest.raises(ValueError, match="identify the one you mean by its path"):
        index.find_by_name("AutoLamella-2025-07-21-15-21_AM01")


def test_lookup_by_an_unambiguous_name_still_works(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")

    index = Scanner(tmp_path).build_index()

    assert index.find_by_name("AutoLamella-2025-07-21-15-21_AM01").path.name == "AutoLamella-01"


def test_lookup_by_a_missing_name_raises_key_error(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    index = Scanner(tmp_path).build_index()

    with pytest.raises(KeyError):
        index.find_by_name("nonexistent")


def test_each_project_keeps_its_own_images(tmp_path: Path):
    """A collision must not merge two projects' contents."""
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()
    images = {
        project.key: [
            record.image_path.name
            for lamella in project.lamellae
            for stage in lamella.stages
            for record in stage.images
        ]
        for project in index.all_projects()
    }

    assert images == {
        "AutoLamella-01": ["a.tif"],
        "FIB/AutoLamella-01": ["b.tif"],
    }


def test_display_name_distinguishes_projects_sharing_a_name(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()
    labels = sorted(project.display_name for project in index.all_projects())

    assert labels == ["001: AutoLamella-01", "002: FIB/AutoLamella-01"]


def test_display_name_falls_back_to_the_name_without_a_scan_root():
    project = Project(name="Standalone", path=Path("/collection/standalone"), number="001")
    assert project.display_name == "001: Standalone"


# ---------------------------------------------------------------------------
# Selecting a project on the command line
# ---------------------------------------------------------------------------

def test_a_project_is_selectable_by_relative_path(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()
    selector = "FIB/AutoLamella-01"
    records = list(
        iter_matching_records(
            index, project_selectors=[selector], lamella_selectors=["all"],
            stage_selectors=["all"], image_selectors=["all"],
        )
    )

    assert [record.image_path.name for record in records] == ["b.tif"]


def test_a_relative_path_typed_with_backslashes_selects_the_same_project(tmp_path: Path):
    """Windows writes a path with '\\', so that is what a person there types."""
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()
    records = list(
        iter_matching_records(
            index, project_selectors=["FIB\\AutoLamella-01"], lamella_selectors=["all"],
            stage_selectors=["all"], image_selectors=["all"],
        )
    )

    assert [record.image_path.name for record in records] == ["b.tif"]
    assert index.find_by_key("FIB\\AutoLamella-01") is index.find_by_key("FIB/AutoLamella-01")


def test_a_project_is_selectable_by_scan_order_number(tmp_path: Path):
    _make_project(tmp_path, "AutoLamella-01", "a.tif")
    _make_project(tmp_path / "FIB", "AutoLamella-01", "b.tif")

    index = Scanner(tmp_path).build_index()
    records = list(
        iter_matching_records(
            index, project_selectors=["2"], lamella_selectors=["all"],
            stage_selectors=["all"], image_selectors=["all"],
        )
    )

    assert [record.image_path.name for record in records] == ["b.tif"]


def test_matches_project_still_accepts_a_bare_name():
    project = Project(
        name="MyProject", path=Path("/fake"), relative_path="nested/MyProject",
    )
    assert matches_project(project, ["MyProject"])
    assert matches_project(project, ["nested/MyProject"])
    assert not matches_project(project, ["Other"])
