"""Tests for reporting a record's state, and for changing it.

State is declared in a stage's file, so a change writes there and moves nothing.
What these pin is that the record and the file stay in step, that a record
surviving several operations -- as it does in a long-running UI -- keeps
pointing at its files, and that a batch which cannot be carried out changes
neither.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.selection import (
    UnmigratedStage,
    current_state,
    deselect_records,
    mark_corrected_records,
    select_records,
    transition_records,
)
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state, state_file_path
from conftest import build_dataset


@pytest.fixture
def dataset(tmp_path: Path):
    """One SEM acquisition with a label, nothing declared yet."""
    return build_dataset(tmp_path, modalities=("SEM",), acquisitions=1)


@pytest.fixture
def record(dataset) -> ImageRecord:
    """The dataset's single record, as the scanner reads it."""
    project = Scanner(dataset.root).build_index().all_projects()[0]
    records = project.lamellae[0].stages[0].images
    assert len(records) == 1
    return records[0]


def declared(dataset) -> dict[str, str]:
    """Return what the stage's file declares."""
    return dict(read_state(dataset.stage).states)


# ---------------------------------------------------------------------------
# current_state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "selected, corrected, expected",
    [
        (True, False, "selected"),
        (False, False, "deselected"),
        (False, True, "corrected"),
    ],
)
def test_current_state_reads_the_record(selected, corrected, expected):
    """The record carries the state the scanner read; nothing re-derives it."""
    record = ImageRecord(
        image_path=Path("sem/img1.tif"), selected=selected, corrected=corrected
    )

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
# The file and the record stay in step
# ---------------------------------------------------------------------------

def test_a_change_is_written_to_the_stage_file(dataset, record: ImageRecord):
    deselect_records([record])

    assert list(declared(dataset).values()) == ["deselected"]


def test_the_key_written_names_the_label(dataset, record: ImageRecord):
    deselect_records([record])

    key = next(iter(declared(dataset)))
    assert key == f"{dataset.label_dir('SEM').name}/{record.image_path.name}"


def test_a_rescan_reads_back_what_was_declared(dataset, record: ImageRecord):
    mark_corrected_records([record])

    project = Scanner(dataset.root).build_index().all_projects()[0]
    rescanned = project.lamellae[0].stages[0].images[0]
    assert current_state(rescanned) == "corrected"


def test_returning_to_selected_leaves_nothing_declared(dataset, record: ImageRecord):
    """A state file lists what is not selected, so the entry goes."""
    deselect_records([record])
    select_records([record])

    assert declared(dataset) == {}


def test_nothing_moves(dataset, record: ImageRecord):
    before = sorted(str(p.relative_to(dataset.root)) for p in dataset.root.rglob("*.tif"))

    deselect_records([record])
    mark_corrected_records([record], allow_move_from_corrected=True)

    after = sorted(str(p.relative_to(dataset.root)) for p in dataset.root.rglob("*.tif"))
    assert after == before
    assert record.image_path.exists()
    assert all(label.exists() for label in record.labels)


@pytest.mark.parametrize("state", ["selected", "deselected", "corrected"])
def test_record_paths_stay_real_after_any_transition(record: ImageRecord, state):
    transition_records([record], state, allow_move_from_corrected=True)

    assert record.image_path.exists()
    assert all(label.exists() for label in record.labels)
    assert current_state(record) == state


def test_a_group_with_no_label_can_still_be_declared(tmp_path: Path):
    """Keyed by where its label would go, so the entry survives annotation."""
    dataset = build_dataset(
        tmp_path, modalities=("SEM",), acquisitions=1, labels="none"
    )
    project = Scanner(dataset.root).build_index().all_projects()[0]
    record = project.lamellae[0].stages[0].images[0]

    deselect_records([record])

    key = next(iter(read_state(dataset.stage).states))
    assert key.endswith(record.image_path.name)
    assert not (dataset.stage / key).exists()


# ---------------------------------------------------------------------------
# Corrected data stays protected
# ---------------------------------------------------------------------------

def test_leaving_corrected_still_requires_confirmation(dataset, record: ImageRecord):
    mark_corrected_records([record])

    with pytest.raises(PermissionError):
        select_records([record])

    assert current_state(record) == "corrected"
    assert list(declared(dataset).values()) == ["corrected"]


# ---------------------------------------------------------------------------
# A batch that cannot be carried out changes nothing
# ---------------------------------------------------------------------------

def test_a_missing_image_stops_the_whole_batch(dataset, record: ImageRecord):
    missing = ImageRecord(image_path=dataset.image_dir("SEM") / "absent.tif")

    with pytest.raises(FileNotFoundError):
        deselect_records([record, missing])

    assert current_state(record) == "selected"
    assert declared(dataset) == {}


def test_a_stage_still_keeping_state_in_folders_is_refused(dataset, record: ImageRecord):
    """Declaring one group there would leave the rest unlisted, and unlisted
    reads as selected -- promoting deselected images into a training set."""
    nested = dataset.image_dir("SEM") / "deselected"
    nested.mkdir()
    (nested / "someone_elses.tif").write_bytes(b"")

    with pytest.raises(UnmigratedStage, match="migrate-state"):
        deselect_records([record])

    assert not state_file_path(dataset.stage).exists()
    assert current_state(record) == "selected"


# ---------------------------------------------------------------------------
# The sequence the curation window performs
# ---------------------------------------------------------------------------

def test_curating_a_frame_from_a_scanned_project(tmp_path: Path):
    """Scan -> frame -> transition, without re-scanning in between."""
    from am_segmentation_manager.frames import build_lamella_frames

    dataset = build_dataset(tmp_path, acquisitions=1)
    lamella = Scanner(dataset.root).build_index().all_projects()[0].lamellae[0]
    frame = build_lamella_frames(lamella)[0]
    assert frame.record("SEM") is not None and frame.record("FIB") is not None

    # Deselect only the SEM side, as the "apply to: SEM" control will.
    deselect_records([frame.record("SEM")])
    assert current_state(frame.record("SEM")) == "deselected"
    assert current_state(frame.record("FIB")) == "selected"

    # Mark both corrected afterwards, reusing the same records.
    mark_corrected_records(frame.records)
    assert [current_state(record) for record in frame.records] == ["corrected", "corrected"]
    for record in frame.records:
        assert record.image_path.exists()
        assert all(label.exists() for label in record.labels)
