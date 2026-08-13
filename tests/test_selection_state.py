"""Tests for record state reporting and for records staying accurate after moves.

A record that survives several operations -- as it does in a long-running UI --
must keep pointing at its files. These tests pin that behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.selection import (
    current_state,
    deselect_records,
    mark_corrected_records,
    select_records,
    transition_records,
)


@pytest.fixture
def record(tmp_path: Path) -> ImageRecord:
    """A selected SEM record with one label, both present on disk."""
    stage = tmp_path / "stageA"
    image = stage / "sem" / "img1.tif"
    label = stage / "labels" / "img1.tif"
    for path in (image, label):
        path.parent.mkdir(parents=True)
        path.write_text(path.parent.name)
    return ImageRecord(image_path=image, labels=[label], modality="SEM")


# ---------------------------------------------------------------------------
# current_state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "relative_path, expected",
    [
        ("sem/img1.tif", "selected"),
        ("sem/deselected/img1.tif", "deselected"),
        ("sem/corrected/img1.tif", "corrected"),
        ("fib/deselected/img1.tif", "deselected"),
    ],
)
def test_current_state_reads_the_state_folder(tmp_path, relative_path, expected):
    record = ImageRecord(image_path=tmp_path / relative_path)

    assert current_state(record) == expected


def test_current_state_tracks_transitions(record: ImageRecord):
    assert current_state(record) == "selected"

    deselect_records([record])
    assert current_state(record) == "deselected"

    mark_corrected_records([record], allow_move_from_corrected=True)
    assert current_state(record) == "corrected"

    select_records([record], allow_move_from_corrected=True)
    assert current_state(record) == "selected"


# ---------------------------------------------------------------------------
# Records stay accurate after a move
# ---------------------------------------------------------------------------

def test_deselect_updates_record_paths(record: ImageRecord):
    original_image = record.image_path

    deselect_records([record])

    assert record.image_path == original_image.parent / "deselected" / "img1.tif"
    assert record.labels == [original_image.parent.parent / "labels" / "deselected" / "img1.tif"]
    assert record.image_path.exists()
    assert record.labels[0].exists()


def test_record_paths_survive_a_round_trip(record: ImageRecord):
    original_image = record.image_path
    original_labels = list(record.labels)

    deselect_records([record])
    select_records([record])

    assert record.image_path == original_image
    assert record.labels == original_labels
    assert record.image_path.exists()


@pytest.mark.parametrize("state", ["selected", "deselected", "corrected"])
def test_record_paths_point_at_real_files_after_any_transition(record: ImageRecord, state):
    transition_records([record], state, allow_move_from_corrected=True)

    assert record.image_path.exists(), f"image path stale after -> {state}"
    assert all(label.exists() for label in record.labels), f"label path stale after -> {state}"
    assert current_state(record) == state


def test_transition_from_a_stale_path_corrects_the_record(record: ImageRecord):
    """A record can hold a path in the wrong state folder; the move fixes it."""
    deselect_records([record])
    # Simulate a record built before the move, still naming the selected path.
    stale = ImageRecord(
        image_path=record.image_path.parent.parent / "img1.tif",
        labels=[record.labels[0].parent.parent / "img1.tif"],
        modality="SEM",
    )

    deselect_records([stale])

    assert stale.image_path == record.image_path
    assert stale.labels == record.labels
    assert current_state(stale) == "deselected"


def test_no_op_transition_still_normalises_paths(record: ImageRecord):
    """Re-applying the current state fixes a stale path without moving files."""
    deselect_records([record])
    moved_image = record.image_path
    stale = ImageRecord(image_path=moved_image.parent.parent / "img1.tif", modality="SEM")

    deselect_records([stale])

    assert stale.image_path == moved_image
    assert moved_image.exists()


def test_missing_label_keeps_its_recorded_path(record: ImageRecord):
    """A label that exists nowhere is left as-is rather than invented or dropped."""
    record.labels[0].unlink()
    absent_label = record.labels[0]

    deselect_records([record])

    assert record.labels == [absent_label]
    assert record.image_path.name == "img1.tif"
    assert current_state(record) == "deselected"


# ---------------------------------------------------------------------------
# Corrected data stays protected
# ---------------------------------------------------------------------------

def test_moving_out_of_corrected_still_requires_confirmation(record: ImageRecord):
    mark_corrected_records([record])
    corrected_image = record.image_path

    with pytest.raises(PermissionError):
        select_records([record])

    # Nothing moved, and the record still describes where the files are.
    assert record.image_path == corrected_image
    assert corrected_image.exists()
    assert current_state(record) == "corrected"


def test_curating_a_frame_from_a_scanned_project(tmp_path: Path):
    """Scan -> frame -> transition, the sequence the curation UI performs.

    Covers the case that motivated updating record paths: a frame is built
    once, then used for several state changes without re-scanning.
    """
    from am_segmentation_manager.frames import build_lamella_frames
    from am_segmentation_manager.scanner import Scanner

    project = tmp_path / "AutoLamella-2026-07-28-13-57"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: test")
    stage = project / "01-alive-muskox" / "adaptive_polish_2026-07-28-02-43-32PM"
    files = {
        "sem": "01-alive-muskox_AP_img_000_SEM.tif",
        "labels": "01-alive-muskox_AP_img_000_SEM.tif",
        "fib": "01-alive-muskox_AP_img_000_FIB.tif",
        "labels_fib": "01-alive-muskox_AP_img_000_FIB.tif",
    }
    for folder, name in files.items():
        (stage / folder).mkdir(parents=True)
        (stage / folder / name).write_text(folder)

    lamella = Scanner(tmp_path).build_index().all_projects()[0].lamellae[0]
    frame = build_lamella_frames(lamella)[0]
    assert frame.sem is not None and frame.fib is not None

    # Deselect only the SEM side, as the "apply to: SEM" control will.
    deselect_records([frame.sem])
    assert current_state(frame.sem) == "deselected"
    assert current_state(frame.fib) == "selected"

    # Mark both corrected afterwards, reusing the same records.
    mark_corrected_records(frame.records)
    assert [current_state(record) for record in frame.records] == ["corrected", "corrected"]
    for record in frame.records:
        assert record.image_path.exists()
        assert all(label.exists() for label in record.labels)


def test_failed_batch_leaves_records_untouched(tmp_path: Path):
    """Records are only updated once every move in the batch has succeeded."""
    sem = tmp_path / "sem"
    sem.mkdir()
    (sem / "deselected").mkdir()
    safe_image = sem / "safe.tif"
    safe_image.write_text("safe")
    blocked_image = sem / "blocked.tif"
    blocked_image.write_text("normal")
    (sem / "deselected" / "blocked.tif").write_text("already deselected")

    safe = ImageRecord(image_path=safe_image)
    blocked = ImageRecord(image_path=blocked_image)

    with pytest.raises(FileExistsError):
        deselect_records([safe, blocked])

    assert safe.image_path == safe_image
    assert safe.selected is True
    assert safe_image.exists()
