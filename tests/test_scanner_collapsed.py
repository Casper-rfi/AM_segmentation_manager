"""Scanning datasets whose hierarchy is missing levels.

Autolamella writes project → lamella → stage → image directories. Older
datasets collapse levels: some have no stage directory, and the oldest have
neither a stage nor a lamella directory. Rather than requiring those levels to
be created on disk, the scanner synthesises them, standing the parent directory
in for the missing level and marking it so nothing pretends otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.filtering import iter_matching_records
from am_segmentation_manager.scanner import Scanner


def make_collapsed_stage(root: Path) -> Path:
    """Build ``project/lam_010/SEM_8/...``: a lamella, but no stage directory.

    Modelled on ``20240228_adaptive_milling_ai``, including the shared ``labels``
    directory and the uppercase ``labels_FIB``.
    """
    project = root / "20240228_adaptive_milling_ai"
    lamella = project / "lam_010"
    for directory in ("SEM_8", "SEM_8_fi_1", "FIB", "labels", "labels_FIB"):
        (lamella / directory).mkdir(parents=True)
        for index in (0, 4):
            (lamella / directory / f"img_000{index}.tif").write_bytes(b"")
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
    return project


def make_collapsed_lamella(root: Path) -> Path:
    """Build ``project/SEM/...``: neither a lamella nor a stage directory."""
    project = root / "20231102"
    for directory in ("SEM", "labels"):
        (project / directory).mkdir(parents=True)
        for index in (1, 2):
            (project / directory / f"SEM_{index}.tif").write_bytes(b"")
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
    return project


def _only_project(root: Path):
    projects = Scanner(root).build_index().all_projects()
    assert len(projects) == 1
    return projects[0]


# ---------------------------------------------------------------------------
# A lamella with no stage directory
# ---------------------------------------------------------------------------

def test_a_lamella_without_a_stage_directory_is_indexed(tmp_path: Path):
    make_collapsed_stage(tmp_path)
    project = _only_project(tmp_path)

    assert [lamella.name for lamella in project.lamellae] == ["lam_010"]
    stage = project.lamellae[0].stages[0]
    assert stage.path == project.path / "lam_010"
    assert stage.synthetic is True
    assert project.lamellae[0].synthetic is False


def test_a_synthetic_stage_is_marked_in_its_display_name(tmp_path: Path):
    make_collapsed_stage(tmp_path)
    project = _only_project(tmp_path)

    assert "no stage folder" in project.lamellae[0].stages[0].display_name


def test_the_legacy_uppercase_label_directory_is_matched(tmp_path: Path):
    make_collapsed_stage(tmp_path)
    project = _only_project(tmp_path)

    records = {r.image_path.parent.name: r for r in project.lamellae[0].stages[0].images}
    assert records["FIB"].labels[0].parent.name == "labels_FIB"


def test_every_imaging_condition_is_indexed(tmp_path: Path):
    """``SEM_8`` and ``SEM_8_fi_1`` are one scene under two conditions."""
    make_collapsed_stage(tmp_path)
    stage = _only_project(tmp_path).lamellae[0].stages[0]

    by_directory = {record.image_path.parent.name: record for record in stage.images}
    assert set(by_directory) == {"SEM_8", "SEM_8_fi_1", "FIB"}
    assert by_directory["SEM_8"].variant == "8"
    assert by_directory["SEM_8_fi_1"].variant == "8_fi_1"
    assert by_directory["FIB"].variant is None


def test_the_sem_conditions_share_one_label_and_are_marked_as_such(tmp_path: Path):
    make_collapsed_stage(tmp_path)
    stage = _only_project(tmp_path).lamellae[0].stages[0]

    by_directory = {record.image_path.parent.name: record for record in stage.images}
    assert by_directory["SEM_8"].labels == by_directory["SEM_8_fi_1"].labels
    assert by_directory["SEM_8"].label_shared is True
    assert by_directory["FIB"].label_shared is False


def test_the_conditions_of_one_acquisition_form_a_single_frame(tmp_path: Path):
    from am_segmentation_manager.frames import build_lamella_frames

    make_collapsed_stage(tmp_path)
    lamella = _only_project(tmp_path).lamellae[0]

    frames = build_lamella_frames(lamella)

    assert len(frames) == 2  # img_0000 and img_0004
    assert [record.variant for record in frames[0].sem] == ["8", "8_fi_1"]
    assert len(frames[0].fib) == 1


def test_a_lamella_number_is_parsed_from_a_non_numeric_prefix(tmp_path: Path):
    """``lam_010`` should be selectable as lamella 10."""
    make_collapsed_stage(tmp_path)
    project = _only_project(tmp_path)

    assert project.lamellae[0].number == "10"


# ---------------------------------------------------------------------------
# Neither level present
# ---------------------------------------------------------------------------

def test_a_project_with_no_lamella_or_stage_directory_is_indexed(tmp_path: Path):
    make_collapsed_lamella(tmp_path)
    project = _only_project(tmp_path)

    lamella = project.lamellae[0]
    assert lamella.synthetic is True
    assert lamella.path == project.path
    assert lamella.stages[0].synthetic is True
    assert [record.image_path.name for record in lamella.stages[0].images] == [
        "SEM_1.tif", "SEM_2.tif",
    ]


def test_a_synthetic_lamella_is_marked_and_numbered_one(tmp_path: Path):
    make_collapsed_lamella(tmp_path)
    lamella = _only_project(tmp_path).lamellae[0]

    assert "no lamella folder" in lamella.display_name
    assert lamella.number == "01"


def test_a_synthetic_lamella_is_selectable_as_lamella_one(tmp_path: Path):
    """Selectors must work the same across every layout."""
    make_collapsed_lamella(tmp_path)
    index = Scanner(tmp_path).build_index()

    records = list(
        iter_matching_records(
            index, project_selectors=["all"], lamella_selectors=["1"],
            stage_selectors=["all"], image_selectors=["all"],
        )
    )
    assert len(records) == 2


def test_labels_are_matched_in_the_collapsed_layout(tmp_path: Path):
    make_collapsed_lamella(tmp_path)
    project = _only_project(tmp_path)

    for record in project.lamellae[0].stages[0].images:
        assert len(record.labels) == 1
        assert record.labels[0].name == record.image_path.name


# ---------------------------------------------------------------------------
# The marker file is what makes these indexable
# ---------------------------------------------------------------------------

def test_without_a_marker_a_legacy_dataset_is_not_a_project(tmp_path: Path):
    project = make_collapsed_lamella(tmp_path)
    (project / "AM_metadata.yaml").unlink()

    assert Scanner(tmp_path).build_index().all_projects() == []


def test_am_metadata_alone_is_enough(tmp_path: Path):
    make_collapsed_lamella(tmp_path)
    project = _only_project(tmp_path)

    assert not (project.path / "experiment.yaml").exists()
    assert project.experiment_metadata.sample == "HeLa"


def test_a_project_marked_only_by_am_metadata_falls_back_to_its_folder_name(tmp_path: Path):
    make_collapsed_lamella(tmp_path)
    assert _only_project(tmp_path).name == "20231102"
