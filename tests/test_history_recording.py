"""Tests for the operations that record what they changed.

Two rules hold across all of them. **A project with no log is left without
one**, so installing this changes nothing about a collection that has not asked
to be recorded. **Recording never fails an operation**: a change that happened
and went unrecorded is a gap in the record, where a save refused because its
entry would not write is lost work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.history import (
    Recorder,
    events_for,
    history_file_path,
    read_log,
)
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
from am_segmentation_manager.state import read_state, write_state
from conftest import start_history

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


def logged(project: Path) -> Path:
    """Start a log in *project*, so operations there record into it."""
    start_history(project)
    return project


def label(value: int = 2, shape: tuple[int, int] = (16, 16)):
    """Return a small label holding one square of *value*."""
    array = np.zeros(shape, dtype=np.uint8)
    array[2:6, 2:6] = value
    return array


def write_label(path: Path, array) -> Path:
    """Write *array* to *path* as a label file."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array)
    return path


# ---------------------------------------------------------------------------
# Opting in
# ---------------------------------------------------------------------------

def test_a_project_with_no_log_is_left_without_one(tmp_path: Path, recorder):
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)

    save_labels(
        [LabelToSave(destination=destination, label=label(), created=True)],
        recorder=recorder,
    )

    assert destination.is_file()
    assert not history_file_path(tmp_path).exists()
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_an_operation_given_no_recorder_records_nothing(tmp_path: Path):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)

    save_labels([LabelToSave(destination=destination, label=label(), created=True)])

    assert read_log(tmp_path).events == ()


# ---------------------------------------------------------------------------
# Saving labels
# ---------------------------------------------------------------------------

def test_a_saved_label_is_recorded_with_what_it_holds(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)

    save_labels(
        [LabelToSave(destination=destination, label=label(), created=True)],
        recorder=recorder,
    )

    (event,) = read_log(tmp_path).events
    assert event.kind == "label-saved"
    assert event.target == "sem_labels/i.tif"
    assert event.by == "casper"
    assert event.data["shape"] == [16, 16] and event.data["empty"] is False
    assert "parent_digest" not in event.data


def test_a_replaced_label_records_what_it_grew_out_of(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    before = label()
    write_label(destination, before)

    after = before.copy()
    after[10:14, 10:14] = 3
    save_labels(
        [
            LabelToSave(
                destination=destination,
                label=after,
                fingerprint=_fingerprint(destination),
            )
        ],
        recorder=recorder,
    )

    (event,) = read_log(tmp_path).events
    assert event.data["parent_digest"] != event.data["digest"]
    assert event.data["changed_fraction"] == pytest.approx(16 / 256)


def test_an_empty_label_is_recorded_as_empty(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)

    save_labels(
        [
            LabelToSave(
                destination=destination,
                label=np.zeros((8, 8), dtype=np.uint8),
                created=True,
            )
        ],
        recorder=recorder,
    )

    (event,) = read_log(tmp_path).events
    assert event.data["empty"] is True


def test_one_save_of_several_labels_shares_a_batch(tmp_path: Path, recorder):
    logged(tmp_path)
    items = []
    for index in range(3):
        destination = tmp_path / "sem_labels" / f"{index}.tif"
        destination.parent.mkdir(parents=True, exist_ok=True)
        items.append(
            LabelToSave(destination=destination, label=label(), created=True)
        )

    save_labels(items, recorder=recorder)

    events = read_log(tmp_path).events
    assert len(events) == 3
    assert len({event.batch for event in events}) == 1


def test_a_refused_save_records_nothing(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = write_label(tmp_path / "sem_labels" / "i.tif", label())

    with pytest.raises(PermissionError):
        save_labels(
            [
                LabelToSave(
                    destination=destination,
                    label=label(3),
                    fingerprint=_fingerprint(destination),
                    corrected=True,
                )
            ],
            recorder=recorder,
        )

    assert read_log(tmp_path).events == ()


def test_a_label_that_will_not_open_does_not_stop_the_save(tmp_path: Path, recorder):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"not a tiff at all")

    save_labels(
        [
            LabelToSave(
                destination=destination,
                label=label(),
                fingerprint=_fingerprint(destination),
            )
        ],
        recorder=recorder,
    )

    (event,) = read_log(tmp_path).events
    assert event.kind == "label-saved" and "parent_digest" not in event.data


def test_a_log_that_cannot_be_written_does_not_stop_the_save(
    tmp_path: Path, recorder, monkeypatch
):
    logged(tmp_path)
    destination = tmp_path / "sem_labels" / "i.tif"
    destination.parent.mkdir(parents=True)

    def refuse(*_args, **_kwargs):
        raise OSError("the disk is full")

    monkeypatch.setattr("am_segmentation_manager.history.append_event", refuse)
    report = save_labels(
        [LabelToSave(destination=destination, label=label(), created=True)],
        recorder=recorder,
    )

    assert report.written == (destination,)
    assert destination.is_file()


# ---------------------------------------------------------------------------
# Changing state
# ---------------------------------------------------------------------------

def test_a_state_change_records_what_it_moved_between(tmp_path: Path, recorder):
    from am_segmentation_manager.operations.selection import transition_records

    logged(tmp_path)
    record = _record_in(tmp_path)
    transition_records([record], "deselected", recorder=recorder)

    (event,) = read_log(tmp_path).events
    assert event.kind == "state-changed"
    assert event.data == {"old": "selected", "new": "deselected"}
    assert event.target == "sem_labels/i.tif"


def test_a_group_already_in_the_state_records_nothing(tmp_path: Path, recorder):
    from am_segmentation_manager.operations.selection import transition_records

    logged(tmp_path)
    record = _record_in(tmp_path)
    transition_records([record], "deselected", recorder=recorder)
    before = len(read_log(tmp_path).events)

    transition_records([record], "deselected", recorder=recorder)

    assert len(read_log(tmp_path).events) == before


def test_a_stale_state_file_stops_the_batch_with_nothing_recorded(
    tmp_path: Path, recorder
):
    from am_segmentation_manager.operations.selection import (
        StaleState,
        apply_plan,
        plan_transition,
    )

    logged(tmp_path)
    record = _record_in(tmp_path)
    plan = plan_transition([record], "deselected")

    stage = record.stage_path
    write_state(read_state(stage), {"sem_labels/other.tif": "deselected"})

    with pytest.raises(StaleState):
        apply_plan(plan, recorder=recorder)
    assert read_log(tmp_path).events == ()


# ---------------------------------------------------------------------------
# Editing metadata
# ---------------------------------------------------------------------------

def test_each_edited_field_is_recorded_on_its_own(tmp_path: Path, recorder):
    from am_segmentation_manager.operations.metadata_edit import update_project_metadata

    logged(tmp_path)
    update_project_metadata(
        tmp_path, {"sample": "yeast", "grid_type": "UltrAuFoil"}, recorder=recorder
    )

    events = read_log(tmp_path).events
    assert {event.data["field"] for event in events} == {"sample", "grid_type"}
    assert all(event.kind == "metadata-changed" for event in events)
    assert all(event.target == "AM_metadata.yaml" for event in events)


def test_a_dry_run_edit_records_nothing(tmp_path: Path, recorder):
    from am_segmentation_manager.operations.metadata_edit import update_project_metadata

    logged(tmp_path)
    update_project_metadata(
        tmp_path, {"sample": "yeast"}, dry_run=True, recorder=recorder
    )

    assert read_log(tmp_path).events == ()


# ---------------------------------------------------------------------------
# Recompressing
# ---------------------------------------------------------------------------

def test_a_recompression_records_the_same_digest_before_and_after(
    tmp_path: Path, recorder
):
    from am_segmentation_manager.operations.label_compression import compress_labels

    logged(tmp_path)
    record = _record_in(tmp_path)
    write_label(record.labels[0], np.zeros((256, 256), dtype=np.uint8))

    results = compress_labels([record], recorder=recorder)

    assert any(result.applied for result in results)
    (event,) = read_log(tmp_path).events
    assert event.kind == "label-compressed"
    assert event.data["digest"] == event.data["parent_digest"]
    assert event.data["bytes_after"] < event.data["bytes_before"]


def test_a_dry_run_compression_records_nothing(tmp_path: Path, recorder):
    from am_segmentation_manager.operations.label_compression import compress_labels

    logged(tmp_path)
    record = _record_in(tmp_path)
    write_label(record.labels[0], np.zeros((256, 256), dtype=np.uint8))

    compress_labels([record], dry_run=True, recorder=recorder)

    assert read_log(tmp_path).events == ()


# ---------------------------------------------------------------------------
# Renaming a label directory
# ---------------------------------------------------------------------------

def test_renaming_a_label_directory_keeps_its_history_reachable(
    tmp_path: Path, recorder
):
    logged(tmp_path)
    old = tmp_path / "labels"
    write_label(old / "i.tif", label())
    recorder.record(old / "i.tif", "state-changed", old="selected", new="corrected")

    new = tmp_path / "sem_labels"
    old.rename(new)
    assert recorder.record_directory_rename(old, new) == 1

    log = read_log(tmp_path)
    moved = events_for(log, "sem_labels/i.tif")
    assert moved[-1].kind == "renamed"
    assert moved[-1].data == {"renamed_from": "labels/i.tif", "scope": "directory"}
    assert events_for(log, "labels/i.tif")[0].kind == "state-changed"


def test_renaming_where_nothing_is_recorded_records_nothing(tmp_path: Path, recorder):
    old = tmp_path / "labels"
    write_label(old / "i.tif", label())
    new = tmp_path / "sem_labels"
    old.rename(new)

    assert recorder.record_directory_rename(old, new) == 0


# ---------------------------------------------------------------------------
# Reaching the log at all
# ---------------------------------------------------------------------------

def test_a_project_reached_through_a_symlink_is_still_recorded_into(
    tmp_path: Path, recorder
):
    """The failure this pins is silent, which is what makes it worth a test.

    ``/tmp`` is a symlink on macOS, so a project found through an unresolved
    parent is not a prefix of the resolved label path -- and every entry about
    it was dropped with nothing said. ``tmp_path`` is already resolved, so only
    a symlink in the test finds this.
    """
    real = tmp_path / "collection"
    (real / "sem_labels").mkdir(parents=True)
    logged(real)
    link = tmp_path / "through"
    link.symlink_to(real, target_is_directory=True)

    destination = link / "sem_labels" / "i.tif"
    save_labels(
        [LabelToSave(destination=destination, label=label(), created=True)],
        recorder=recorder,
    )

    (event,) = read_log(real).events
    assert event.target == "sem_labels/i.tif"


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def test_each_operation_gets_a_batch_of_its_own(recorder):
    assert recorder.for_batch().batch != recorder.for_batch().batch


def test_a_batch_keeps_the_author_it_came_from(recorder):
    batch = recorder.for_batch()
    assert (batch.by, batch.by_source) == (recorder.by, recorder.by_source)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(path: Path):
    from am_segmentation_manager.io.atomic import fingerprint_of

    return fingerprint_of(path)


def _record_in(project: Path):
    """Return a record whose image and label sit in a stage under *project*."""
    from am_segmentation_manager.models import ImageRecord

    image = project / "sem" / "i.tif"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    label_path = project / "sem_labels" / "i.tif"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.touch()
    return ImageRecord(
        image_path=image,
        labels=[label_path],
        modality="SEM",
        stage_path=project,
        labels_directory=project / "sem_labels",
    )
