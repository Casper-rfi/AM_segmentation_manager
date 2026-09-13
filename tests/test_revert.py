"""Tests for putting a selection of labels back before an operation.

The workflow this exists for: convert a project's predictions over labels
somebody has already proofread, look at a few, decide the model is not good
enough, and put the whole project back in one command. So what is pinned is
what it refuses to do -- put back a label whose earlier version was never kept,
one whose pixels are elsewhere, one edited since, and proofread work that
nobody has confirmed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.history import Recorder, read_log
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.label_io import label_fields
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
from am_segmentation_manager.operations.revert import (
    NO_ENTRY,
    NOT_ARCHIVED,
    SUPERSEDED,
    apply_revert,
    plan_revert,
)
from conftest import start_history

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


@pytest.fixture
def converted(tmp_path: Path, recorder):
    """A proofread label with a model's guess written over it.

    The state the workflow this undoes leaves behind: what is on disk is the
    conversion, and the proofread version is in the archive.
    """
    start_history(tmp_path)
    label = tmp_path / "sem_labels" / "img_000.tif"
    _write(label, 2)
    _convert_over(label, _pixels(3), recorder)
    return label


def _pixels(value: int, shape: tuple[int, int] = (16, 16)):
    array = np.zeros(shape, dtype=np.uint8)
    array[2:6, 2:6] = value
    return array


def _write(path: Path, value: int) -> Path:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, _pixels(value))
    return path


def _digest(path: Path) -> str:
    import tifffile

    return str(label_fields(tifffile.imread(path))["digest"])


def _read(path: Path):
    import tifffile

    return tifffile.imread(path)


def _record(label: Path) -> ImageRecord:
    """Return a record whose group is that label."""
    image = label.parents[1] / "sem" / label.name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    return ImageRecord(
        image_path=image, labels=[label], modality="SEM",
        stage_path=label.parents[1], labels_directory=label.parent,
    )


def _convert_over(label: Path, pixels, recorder: Recorder) -> None:
    """Write *pixels* over *label* the way a conversion does, and record it."""
    from am_segmentation_manager.operations.archive import Archivist

    before = label_fields(_read(label))
    archived = Archivist(recorder).keep(
        label, str(before["digest"]), empty=bool(before["empty"])
    )
    import tifffile

    tifffile.imwrite(label, pixels)
    recorder.record(
        label, "label-converted", model="fpn_v4", prediction_kind="am_seg",
        archived_as=archived, **label_fields(pixels),
    )


def _save_over(label: Path, pixels, recorder: Recorder, **kwargs) -> None:
    save_labels(
        [
            LabelToSave(
                destination=label, label=pixels,
                fingerprint=fingerprint_of(label), **kwargs,
            )
        ],
        recorder=recorder, allow_corrected=kwargs.get("corrected", False),
    )


# ---------------------------------------------------------------------------
# The workflow it exists for
# ---------------------------------------------------------------------------

def test_a_conversion_over_proofread_work_can_be_put_back(converted, recorder):
    plan = plan_revert([_record(converted)], "conversion")

    (item,) = plan.items
    assert item.can_revert
    assert plan.revertible


def test_putting_it_back_restores_the_pixels(converted, recorder):
    import tifffile

    plan = plan_revert([_record(converted)], "conversion")

    apply_revert(plan, recorder=recorder)

    assert np.array_equal(tifffile.imread(converted), _pixels(2))


def test_putting_it_back_does_not_keep_the_model_s_untouched_guess(
    converted, recorder
):
    """It is machine output nobody has edited, which the model can make again.

    The rule the archive applies everywhere, and its consequence here: looking
    at that guess again means converting again, not opening the archive. What
    made it is recorded, so that is a re-run rather than a reconstruction.
    """
    from am_segmentation_manager.operations.archive import versions_of

    guess = _digest(converted)

    apply_revert(plan_revert([_record(converted)], "conversion"), recorder=recorder)

    kept = versions_of(converted.parents[1], "sem_labels/img_000.tif")
    assert guess not in {_digest(version.path) for version in kept}


def test_a_whole_selection_goes_back_at_once(tmp_path: Path, recorder):
    start_history(tmp_path)
    labels = []
    for index in range(4):
        label = tmp_path / "sem_labels" / f"img_{index}.tif"
        _write(label, 2)
        _convert_over(label, _pixels(3), recorder)
        labels.append(label)

    plan = plan_revert([_record(label) for label in labels], "conversion")
    report = apply_revert(plan, recorder=recorder)

    assert len(report.restored) == 4
    assert all(_digest(label) == _digest(labels[0]) for label in labels)


def test_a_revert_is_recorded_and_so_can_itself_be_undone(converted, recorder):
    apply_revert(plan_revert([_record(converted)], "conversion"), recorder=recorder)

    kinds = [event.kind for event in read_log(converted.parents[1]).events]
    assert kinds.count("label-restored") == 1


# ---------------------------------------------------------------------------
# What it leaves alone
# ---------------------------------------------------------------------------

def test_a_label_no_conversion_touched_is_left_alone(tmp_path: Path, recorder):
    start_history(tmp_path)
    untouched = _write(tmp_path / "sem_labels" / "img.tif", 2)

    plan = plan_revert([_record(untouched)], "conversion")

    (item,) = plan.items
    assert item.blocked == NO_ENTRY and not plan.revertible


def test_a_conversion_that_kept_nothing_is_left_alone(tmp_path: Path, recorder):
    """Over an empty placeholder nothing is archived, so there is nothing back."""
    start_history(tmp_path)
    label = tmp_path / "sem_labels" / "img.tif"
    _write(label, 0)
    _convert_over(label, _pixels(3), recorder)

    plan = plan_revert([_record(label)], "conversion")

    (item,) = plan.items
    assert item.blocked == NOT_ARCHIVED


def test_a_label_edited_since_the_conversion_is_left_alone(converted, recorder):
    """Putting it back would discard that editing, which nobody asked for."""
    _save_over(converted, _pixels(5), recorder)

    plan = plan_revert([_record(converted)], "conversion")

    (item,) = plan.items
    assert item.blocked == SUPERSEDED


def test_editing_since_can_be_overridden_deliberately(converted, recorder):
    _save_over(converted, _pixels(5), recorder)

    plan = plan_revert([_record(converted)], "conversion", allow_superseded=True)

    assert plan.revertible


def test_proofread_work_is_refused_without_confirmation(converted, recorder):
    from am_segmentation_manager.operations.selection import transition_records

    record = _record(converted)
    transition_records([record], "corrected", recorder=recorder)
    plan = plan_revert([record], "conversion")
    before = _digest(converted)

    with pytest.raises(PermissionError, match="corrected label"):
        apply_revert(plan, recorder=recorder)

    assert _digest(converted) == before


def test_one_proofread_label_stops_the_whole_batch(tmp_path: Path, recorder):
    """Checked before the first write, so the answer is "nothing happened"."""
    from am_segmentation_manager.operations.selection import transition_records

    start_history(tmp_path)
    records = []
    for index in range(3):
        label = tmp_path / "sem_labels" / f"img_{index}.tif"
        _write(label, 2)
        _convert_over(label, _pixels(3), recorder)
        records.append(_record(label))
    transition_records(records[2:], "corrected", recorder=recorder)
    digests = [_digest(record.labels[0]) for record in records]

    with pytest.raises(PermissionError):
        apply_revert(plan_revert(records, "conversion"), recorder=recorder)

    assert [_digest(record.labels[0]) for record in records] == digests


def test_an_unknown_operation_is_refused(converted):
    with pytest.raises(ValueError, match="not something this can undo"):
        plan_revert([_record(converted)], "everything")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_previews_by_default(converted, recorder, capsys):
    from am_segmentation_manager.cli import main

    before = _digest(converted)

    assert main(
        [
            "revert", str(converted.parents[2]), "--project", "all",
            "--lamella", "all", "--stage", "all", "--image", "all",
            "--modality", "all",
        ]
    ) in (0, 2)

    assert _digest(converted) == before


def test_a_merge_appending_older_entries_does_not_confuse_which_run_to_undo(
    converted, recorder
):
    """File order stops being time order the first time a merge happens.

    ``revert`` looks for the last conversion, and for anything that changed the
    label after it. Read by position in the file, an older entry a merge
    appended counts as the most recent thing that happened to the label.
    """
    from am_segmentation_manager.history import (
        append_event,
        build_event,
        containing_project,
    )

    project = containing_project(converted)
    target = converted.resolve().relative_to(project).as_posix()
    expected = plan_revert([_record(converted)], "conversion").revertible[0]

    # What a merge does: append the copy's entries whole, older than ours, and
    # this one kept nothing -- so taking it as the last conversion says there is
    # nothing to put back.
    append_event(
        project,
        build_event(
            "label-converted", by="anna", by_source="flag", target=target,
            at="2020-01-01T00:00:00.000Z",
            digest="0" * 16, dtype="uint8", shape=[16, 16], empty=False,
            model="ancient", prediction_kind="am_seg",
        ),
    )

    plan = plan_revert([_record(converted)], "conversion")

    assert plan.revertible, "the older entry is not the conversion to undo"
    assert plan.revertible[0].version.path == expected.version.path
