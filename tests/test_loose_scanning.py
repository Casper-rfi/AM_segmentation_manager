"""Tests for indexing the images that sit loose in a stage directory.

Builds the 2026 layout, whose reference images sit above the Adaptive
Polishing stages they surround, and asserts the records the scanner makes of
them: what series each belongs to, and that each carries the stage and label
directory that cannot be recovered from a loose image's path.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from am_segmentation_manager import parallel
from am_segmentation_manager.frames import group_by_acquisition
from am_segmentation_manager.image_metadata import ION, ImageMetadata
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

    Nothing is lost without it; images whose filename states no beam are
    indexed as `other` rather than into their own series, and logged.
    """
    monkeypatch.setattr(
        "am_segmentation_manager.image_metadata.image_stack_problems",
        lambda: [("tifffile", "not installed")],
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    records = Scanner(project.parent).build_index().find_by_name("bitmap").image_records

    assert by_name(records)["ref_PositionReady.tif"].series == "other"
    assert by_name(records)["ref_Polishing_start_eb.tif"].series == "reference"


def test_a_beam_read_from_a_header_indexes_the_image(project, monkeypatch):
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ImageMetadata(beam_type=ION)),
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    records = Scanner(project.parent).build_index().find_by_name("bitmap").image_records

    record = by_name(records)["ref_PositionReady.tif"]
    assert record.modality == "FIB"
    assert record.series == "reference"


def test_only_loose_images_are_opened(project, monkeypatch):
    """Reading every header would turn a scan into an image pass.

    A loose image is opened because nothing else states its beam reliably or
    says when it was taken. An Adaptive Polishing image is not: its filename
    carries both the beam and the acquisition number that orders it.
    """
    opened: list[str] = []
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: (opened.append(path.name), ImageMetadata(beam_type=ION))[1]),
    )

    Scanner(project.parent).build_index()

    assert set(opened) == {
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
        "Polishing_pre-task_10-55-29_eb.tif",
        "Polishing_pre-task_10-55-29_ib.tif",
    }


def test_a_scan_hands_its_headers_over_together(project, monkeypatch):
    """The reads are the one part of a scan that is not a filesystem walk. Handed
    over a stage at a time, most batches are too small for the worker processes
    and are read one after another here, so a scan gathers them across stages
    and projects first."""
    shutil.copytree(project, project.parent / "AutoLamella-2026-07-29-09-00")
    batches: list[list[str]] = []
    real = parallel.map_in_processes

    def read(path):
        return ImageMetadata(beam_type=ION)

    def recording(function, items, **options):
        items = list(items)
        if function is read:
            batches.append(sorted(path.name for path in items))
        return real(function, items, **options)

    monkeypatch.setattr("am_segmentation_manager.scanner.header_reader", lambda: read)
    monkeypatch.setattr(parallel, "map_in_processes", recording)

    Scanner(project.parent).build_index()

    loose = [
        "Polishing_pre-task_10-55-29_eb.tif",
        "Polishing_pre-task_10-55-29_ib.tif",
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
    ]
    assert batches == [sorted(loose * 2)]


def test_a_scan_in_worker_processes_indexes_what_one_here_does(project, monkeypatch):
    """Headers and experiment files read in the pool are the same answers."""
    shutil.copytree(project, project.parent / "AutoLamella-2026-07-29-09-00")
    here = Scanner(project.parent).build_index().all_projects()

    parallel.shutdown_pool()
    monkeypatch.setattr(parallel, "MIN_POOL_ITEMS", 0)
    monkeypatch.setattr(parallel, "POOL_WORKERS", 2)
    monkeypatch.setattr("am_segmentation_manager.scanner.EXPERIMENT_POOL_FILES", 0)
    try:
        pooled = Scanner(project.parent).build_index().all_projects()
        assert parallel._pool is not None
    finally:
        parallel.shutdown_pool()

    assert [p.metadata for p in pooled] == [p.metadata for p in here]
    assert [p.image_records for p in pooled] == [p.image_records for p in here]


def test_the_index_keeps_every_header_the_scan_read(project, monkeypatch):
    """The scan opens each loose image for its beam and time; the rest of what it
    read is kept, so the Export tab need not open those files again."""
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ImageMetadata(beam_type=ION, plasma_gas=path.name)),
    )

    headers = Scanner(project.parent).build_index().headers

    assert {path.name: header.plasma_gas for path, header in headers.items()} == {
        name: name
        for name in (
            "ref_Polishing_start_eb.tif",
            "ref_Polishing_start_ib.tif",
            "ref_alignment_ib.tif",
            "Polishing_pre-task_10-55-29_eb.tif",
            "Polishing_pre-task_10-55-29_ib.tif",
        )
    }


def test_each_image_is_opened_once(project, monkeypatch):
    """The beam and the acquisition time come out of the same read."""
    opened: list[str] = []
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: (opened.append(path.name), ImageMetadata(beam_type=ION))[1]),
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    Scanner(project.parent).build_index()

    assert len(opened) == len(set(opened))
