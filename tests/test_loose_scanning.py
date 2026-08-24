"""Tests for indexing the images that sit loose in a stage directory.

Builds the 2026 layout, whose reference images sit above the Adaptive
Polishing stages they surround, and asserts the records the scanner makes of
them: what series each belongs to, and that each carries the stage and label
directory that cannot be recovered from a loose image's path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import group_by_acquisition
from am_segmentation_manager.operations.label_groups import group_identity, group_key
from am_segmentation_manager.scanner import Scanner


def touch(directory: Path, *names: str) -> Path:
    """Create *names* as empty files in *directory*, which is created too."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A 2026 project: one lamella, one milling task, one Adaptive stage."""
    root = tmp_path / "AutoLamella-2026-07-28-10-31"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")
    lamella = touch(
        root / "01-ruling-ibex",
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
    )
    task = touch(
        lamella / "Milling" / "Polishing",
        "Polishing_pre-task_10-55-29_eb.tif",
        "Polishing_pre-task_10-55-29_ib.tif",
    )
    stage = task / "adaptive_polish_2026-07-28-11-34-24AM"
    touch(stage / "sem", "01-ruling-ibex_AP_img_000_SEM.tif")
    touch(stage / "fib", "01-ruling-ibex_AP_img_000_FIB.tif")
    return root


@pytest.fixture
def records(project: Path):
    index = Scanner(project.parent).build_index()
    return index.find_by_name("bitmap").image_records


def by_name(records) -> dict[str, object]:
    return {record.image_path.name: record for record in records}


# ---------------------------------------------------------------------------
# What gets indexed
# ---------------------------------------------------------------------------

def test_every_loose_image_is_indexed_under_its_series(records):
    found = {record.image_path.name: record.series for record in records}

    assert found == {
        "ref_Polishing_start_eb.tif": "reference",
        "ref_Polishing_start_ib.tif": "reference",
        "ref_alignment_ib.tif": "alignment",
        "Polishing_pre-task_10-55-29_eb.tif": "reference",
        "Polishing_pre-task_10-55-29_ib.tif": "reference",
        "01-ruling-ibex_AP_img_000_SEM.tif": "adaptive",
        "01-ruling-ibex_AP_img_000_FIB.tif": "adaptive",
    }


def test_the_beam_suffix_decides_the_modality(records):
    found = by_name(records)

    assert found["ref_Polishing_start_eb.tif"].modality == "SEM"
    assert found["ref_Polishing_start_ib.tif"].modality == "FIB"
    assert found["ref_alignment_ib.tif"].modality == "FIB"


def test_a_loose_record_belongs_to_the_lamella_its_images_sit_in(project, records):
    found = by_name(records)

    assert found["ref_Polishing_start_eb.tif"].lamella == "01-ruling-ibex"
    assert found["Polishing_pre-task_10-55-29_eb.tif"].lamella == "01-ruling-ibex"


# ---------------------------------------------------------------------------
# What a record carries rather than derives
# ---------------------------------------------------------------------------

def test_every_record_carries_the_stage_that_actually_holds_it(project, records):
    """The derivation this replaces named a loose image's grandparent."""
    found = by_name(records)
    lamella = project / "01-ruling-ibex"
    task = lamella / "Milling" / "Polishing"

    assert found["ref_Polishing_start_eb.tif"].stage_path == lamella
    assert found["Polishing_pre-task_10-55-29_eb.tif"].stage_path == task
    assert found["01-ruling-ibex_AP_img_000_SEM.tif"].stage_path == (
        task / "adaptive_polish_2026-07-28-11-34-24AM"
    )


def test_every_record_carries_where_its_label_would_go(project, records):
    found = by_name(records)
    lamella = project / "01-ruling-ibex"

    assert found["ref_Polishing_start_eb.tif"].labels_directory == (
        lamella / "reference_sem_labels"
    )
    assert found["ref_alignment_ib.tif"].labels_directory == (
        lamella / "alignment_fib_labels"
    )


def test_a_loose_record_has_a_label_group_to_be_keyed_by(project, records):
    """Without this its state could not be declared, only refused."""
    record = by_name(records)["ref_Polishing_start_eb.tif"]

    assert group_identity(record) == (
        project / "01-ruling-ibex" / "reference_sem_labels" / "ref_Polishing_start_eb.tif"
    )
    assert group_key(record) != str(record.image_path)


def test_loose_records_of_one_stage_group_into_that_stage(records):
    """The old derivation scattered them across the collection."""
    reference = [r for r in records if r.series == "reference"]

    stages = {acquisition.stage_directory for acquisition in group_by_acquisition(reference)}

    assert stages == {r.stage_path for r in reference}
    assert len(stages) == 2


# ---------------------------------------------------------------------------
# What indexing them must not change
# ---------------------------------------------------------------------------

def test_adaptive_records_are_untouched(project, records):
    """Every field the scanner filled before it knew about series."""
    record = by_name(records)["01-ruling-ibex_AP_img_000_SEM.tif"]
    stage = (
        project / "01-ruling-ibex" / "Milling" / "Polishing"
        / "adaptive_polish_2026-07-28-11-34-24AM"
    )

    assert record.series == "adaptive"
    assert record.modality == "SEM"
    assert record.variant is None
    assert record.stage == stage.name
    assert record.labels_directory == stage / "sem_labels"
    assert record.state == "selected"


def test_a_loose_set_is_given_no_predictions(project, records):
    """Resolving would claim the Adaptive pipeline's output in a collapsed tree."""
    stage = project / "01-ruling-ibex"
    touch(stage / "sem_predictions" / "am_seg", "ref_Polishing_start_eb.tif")

    fresh = Scanner(project.parent).build_index().find_by_name("bitmap").image_records

    assert by_name(fresh)["ref_Polishing_start_eb.tif"].predictions == []


def test_scanning_works_without_the_image_stack(project, monkeypatch):
    """A scan is a filesystem walk and has never needed tifffile.

    Only images whose filename states no beam are lost, and they are logged.
    """
    monkeypatch.setattr(
        "am_segmentation_manager.image_metadata.image_stack_problems",
        lambda: [("tifffile", "not installed")],
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    records = Scanner(project.parent).build_index().find_by_name("bitmap").image_records

    assert "ref_PositionReady.tif" not in by_name(records)
    assert "ref_Polishing_start_eb.tif" in by_name(records)


def test_a_beam_read_from_a_header_indexes_the_image(project, monkeypatch):
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.beam_resolver", lambda: (lambda path: "FIB")
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    records = Scanner(project.parent).build_index().find_by_name("bitmap").image_records

    record = by_name(records)["ref_PositionReady.tif"]
    assert record.modality == "FIB"
    assert record.series == "reference"


def test_only_names_stating_no_beam_are_opened(project, monkeypatch):
    """Reading every header would turn a scan into an image pass."""
    opened: list[str] = []
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.beam_resolver",
        lambda: (lambda path: opened.append(path.name) or "FIB"),
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    Scanner(project.parent).build_index()

    assert opened == ["ref_PositionReady.tif"]
