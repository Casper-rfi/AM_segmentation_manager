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
    StaleState,
    UnmigratedStage,
    apply_plan,
    current_state,
    deselect_records,
    mark_corrected_records,
    plan_transition,
    select_records,
    transition_records,
)
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state, state_file_path, write_state
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

@pytest.mark.parametrize("state", ["selected", "deselected", "corrected"])
def test_current_state_reads_the_record(state):
    """The record carries the state the scanner read; nothing re-derives it."""
    record = ImageRecord(image_path=Path("sem/img1.tif"), state=state)

    assert current_state(record) == state


def test_a_record_starts_selected():
    """An image nothing has declared anything about is selected, matching the
    state file, where an absent entry means the same."""
    assert current_state(ImageRecord(image_path=Path("sem/img1.tif"))) == "selected"


def test_current_state_tracks_transitions(record: ImageRecord):
    assert current_state(record) == "selected"

    deselect_records([record])
    assert current_state(record) == "deselected"

    mark_corrected_records([record], allow_leaving_corrected=True)
    assert current_state(record) == "corrected"

    select_records([record], allow_leaving_corrected=True)
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
    mark_corrected_records([record], allow_leaving_corrected=True)

    after = sorted(str(p.relative_to(dataset.root)) for p in dataset.root.rglob("*.tif"))
    assert after == before
    assert record.image_path.exists()
    assert all(label.exists() for label in record.labels)


@pytest.mark.parametrize("state", ["selected", "deselected", "corrected"])
def test_record_paths_stay_real_after_any_transition(record: ImageRecord, state):
    transition_records([record], state, allow_leaving_corrected=True)

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

    with pytest.raises(UnmigratedStage, match="last-migrate-state"):
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


# ---------------------------------------------------------------------------
# A batch holds together across stages
# ---------------------------------------------------------------------------

def _two_stage_records(tmp_path: Path) -> list[ImageRecord]:
    """One acquisition in each of two lamellae, so a change spans two files."""
    build_dataset(tmp_path, modalities=("SEM",), acquisitions=1, lamellae=2)
    projects = Scanner(tmp_path).build_index().all_projects()
    records = [
        record
        for project in projects
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]
    assert len({record.image_path.parent.parent for record in records}) == 2
    return records


def _stage_files(records: list[ImageRecord]) -> list[Path]:
    return sorted({record.image_path.parent.parent for record in records})


def _curated_elsewhere(plan, stage: Path) -> None:
    """Declare *stage*'s group corrected, as another window would.

    The key comes from the plan rather than being spelled out, so this really
    is the entry the batch is about rather than an unrelated one that merely
    changes the file.
    """
    key = next(
        change.key for change in plan.changes if change.stage_directory == stage
    )
    write_state(read_state(stage), {key: "corrected"})


def test_a_change_spanning_stages_writes_every_one(tmp_path: Path):
    records = _two_stage_records(tmp_path)

    deselect_records(records)

    for stage in _stage_files(records):
        assert list(read_state(stage).states.values()) == ["deselected"]
    assert {record.state for record in records} == {"deselected"}


def test_a_stage_changed_since_planning_stops_the_whole_batch(tmp_path: Path):
    """Two windows on one collection would otherwise discard each other's
    curation, half a batch at a time.

    The stage that goes stale is the *last* one written, so without a check
    over the whole batch first the earlier stage is already on disk by the time
    anything notices.
    """
    records = _two_stage_records(tmp_path)
    plan = plan_transition(records, "deselected")
    first, second = _stage_files(records)
    _curated_elsewhere(plan, second)

    with pytest.raises(StaleState):
        apply_plan(plan)

    assert read_state(first).states == {}
    assert not state_file_path(first).exists()


def test_a_refused_batch_leaves_the_records_alone(tmp_path: Path):
    """A record saying 'deselected' for a change that never happened would send
    the next export off what is in memory rather than what is on disk."""
    records = _two_stage_records(tmp_path)
    plan = plan_transition(records, "deselected")
    _curated_elsewhere(plan, _stage_files(records)[1])

    with pytest.raises(StaleState):
        apply_plan(plan)

    assert {record.state for record in records} == {"selected"}


def test_a_refused_batch_leaves_the_other_curation_alone(tmp_path: Path):
    """Guarded twice: the batch check here, and write_state's own per-file one."""
    records = _two_stage_records(tmp_path)
    plan = plan_transition(records, "deselected")
    second = _stage_files(records)[1]
    _curated_elsewhere(plan, second)

    with pytest.raises(StaleState):
        apply_plan(plan)

    assert list(read_state(second).states.values()) == ["corrected"]


def test_the_refusal_names_the_file_that_moved(tmp_path: Path):
    records = _two_stage_records(tmp_path)
    plan = plan_transition(records, "deselected")
    second = _stage_files(records)[1]
    _curated_elsewhere(plan, second)

    with pytest.raises(StaleState, match=second.name):
        apply_plan(plan)


def test_a_plan_can_be_carried_out_after_being_looked_at(tmp_path: Path):
    """The point of splitting the two: what was previewed is what runs."""
    records = _two_stage_records(tmp_path)
    plan = plan_transition(records, "deselected")

    assert plan.describe()  # the preview a caller would show
    assert apply_plan(plan) == len(records)
    assert {record.state for record in records} == {"deselected"}
