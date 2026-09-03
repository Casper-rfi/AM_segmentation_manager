"""Tests for merging a worked-on copy back into its collection.

This is where the work can lose data, so what is pinned is mostly what a merge
refuses: a fork point it cannot verify, a project that is not the same project,
and any target both sides changed differently. The resolution rule is pinned
row by row, because getting one row wrong discards somebody's proofreading
while reporting success.

The other property throughout: **a merge reads the files, not the log.** A label
edited by something that recorded nothing must still be seen to have changed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.history import (
    Recorder,
    append_event,
    build_event,
    history_file_path,
    read_log,
)
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.operations.copy_export import copy_export, plan_copy
from am_segmentation_manager.operations.export_record import (
    build_export_record,
    export_record_path,
    read_export_record,
    write_export_record,
)
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
from am_segmentation_manager.operations.merge import (
    CONFLICT,
    CONVERGED,
    KEEP_OURS,
    TAKE_THEIRS,
    MergeRefused,
    apply_merge,
    plan_merge,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")
pytest.importorskip("cv2")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


@pytest.fixture
def forked(tmp_path: Path, recorder):
    """A collection with a history, and a copy of it carrying a fork point.

    The shape every test here starts from: two trees that agree, so anything a
    test then changes is the only thing that has changed.
    """
    main = tmp_path / "collection"
    main.mkdir()
    dataset = build_dataset(main, lamellae=1, acquisitions=2, labels="annotated")
    history_file_path(dataset.project).touch()
    append_event(
        dataset.project,
        build_event(
            "log-created", by="casper", by_source="flag",
            project_id="9f2c4b11", am_seg_version="0.0.0",
        ),
    )
    _baseline(main, recorder)

    copy = tmp_path / "copy"
    _export(main, copy)
    return _Forked(main=main, copy=copy, project=dataset.project)


class _Forked:
    """Where the two trees are, so a test need not re-derive paths."""

    def __init__(self, main: Path, copy: Path, project: Path) -> None:
        self.main = main
        self.copy = copy
        self.project = project

    def label(self, tree: Path) -> Path:
        """Return the first label of *tree*, whichever side it is."""
        return sorted(tree.rglob("*_labels/*.tif")) [0] if list(
            tree.rglob("*_labels/*.tif")
        ) else sorted(tree.rglob("labels/*.tif"))[0]

    @property
    def target(self) -> str:
        """Return the history target of the label the tests act on."""
        label = self.label(self.main)
        return label.resolve().relative_to(self.project.resolve()).as_posix()


# ---------------------------------------------------------------------------
# The resolution rule, row by row
# ---------------------------------------------------------------------------

def test_neither_side_changing_leaves_nothing_to_do(forked):
    plan = plan_merge(forked.copy, forked.main)

    assert plan.projects[0].labels == ()
    assert not plan.is_refused


def test_only_the_copy_changing_takes_the_copy(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)

    plan = plan_merge(forked.copy, forked.main)

    (item,) = plan.projects[0].labels
    assert item.outcome == TAKE_THEIRS


def test_only_this_collection_changing_keeps_what_is_here(forked, recorder):
    _edit(forked.label(forked.main), 3, recorder)

    plan = plan_merge(forked.copy, forked.main)

    (item,) = plan.projects[0].labels
    assert item.outcome == KEEP_OURS


def test_both_arriving_at_the_same_thing_is_not_a_conflict(forked, recorder):
    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 3, recorder)

    plan = plan_merge(forked.copy, forked.main)

    (item,) = plan.projects[0].labels
    assert item.outcome == CONVERGED
    assert not plan.is_refused


def test_both_changing_differently_is_a_conflict(forked, recorder):
    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)

    plan = plan_merge(forked.copy, forked.main)

    (item,) = plan.projects[0].labels
    assert item.outcome == CONFLICT
    assert plan.is_refused


def test_a_recompression_on_one_side_is_not_a_change(forked, recorder):
    """The digest is over pixels, so re-encoding a file changes nothing."""
    from am_segmentation_manager.operations.label_compression import compress_labels

    _edit(forked.label(forked.copy), 3, recorder)
    records = _records(forked.main)
    compress_labels(records, allow_corrected=True, recorder=recorder)

    plan = plan_merge(forked.copy, forked.main)

    (item,) = plan.projects[0].labels
    assert item.outcome == TAKE_THEIRS


# ---------------------------------------------------------------------------
# Reading the files, not the log
# ---------------------------------------------------------------------------

def test_a_label_changed_with_nothing_recording_it_is_still_seen(forked):
    """The failure this pins keeps the wrong version and reports success."""
    import tifffile

    label = forked.label(forked.copy)
    pixels = tifffile.imread(label)
    pixels[8:12, 8:12] = 3
    tifffile.imwrite(label, pixels)          # no recorder: nothing is logged

    plan = plan_merge(forked.copy, forked.main)

    (item,) = plan.projects[0].labels
    assert item.outcome == TAKE_THEIRS
    assert item.ours != item.theirs


def test_an_unrecorded_edit_on_both_sides_is_a_conflict(forked):
    import tifffile

    for tree, value in ((forked.main, 3), (forked.copy, 4)):
        label = forked.label(tree)
        pixels = tifffile.imread(label)
        pixels[8:12, 8:12] = value
        tifffile.imwrite(label, pixels)

    plan = plan_merge(forked.copy, forked.main)

    assert plan.projects[0].labels[0].outcome == CONFLICT


# ---------------------------------------------------------------------------
# What a merge refuses
# ---------------------------------------------------------------------------

def test_a_copy_with_no_fork_point_is_refused(forked):
    export_record_path(forked.copy).unlink()

    with pytest.raises(MergeRefused, match="no fork point"):
        plan_merge(forked.copy, forked.main)


def test_a_fork_point_naming_an_entry_this_collection_never_held_is_refused(forked):
    """A copy of a copy, or of another collection, still carries a plausible file."""
    record = read_export_record(forked.copy)
    fork = record.projects["project"]
    write_export_record(
        forked.copy,
        type(record)(
            exported_at=record.exported_at, exported_by=record.exported_by,
            source=record.source,
            projects={"project": type(fork)(key="project", project_id=fork.project_id,
                                            tip="ffffffffffffffff", entries=1)},
        ),
    )

    with pytest.raises(MergeRefused, match="not in this collection"):
        plan_merge(forked.copy, forked.main)


def test_two_projects_that_are_not_the_same_project_are_refused(forked):
    record = read_export_record(forked.copy)
    fork = record.projects["project"]
    write_export_record(
        forked.copy,
        type(record)(
            exported_at=record.exported_at, exported_by=record.exported_by,
            source=record.source,
            projects={"project": type(fork)(key="project", project_id="somethingelse",
                                            tip=fork.tip, entries=fork.entries)},
        ),
    )

    with pytest.raises(MergeRefused, match="not the same project"):
        plan_merge(forked.copy, forked.main)


def test_a_project_the_collection_does_not_have_is_refused(forked, tmp_path: Path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    build_dataset(other, name="unrelated", labels="annotated")

    with pytest.raises(MergeRefused, match="not in"):
        plan_merge(forked.copy, other)


def test_one_conflict_stops_the_whole_merge(forked, recorder):
    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)
    plan = plan_merge(forked.copy, forked.main)
    before = _digest(forked.label(forked.main))

    with pytest.raises(MergeRefused, match="undecided"):
        apply_merge(plan, recorder=recorder)

    assert _digest(forked.label(forked.main)) == before


# ---------------------------------------------------------------------------
# Deciding a conflict
# ---------------------------------------------------------------------------

def test_a_conflict_decided_by_hand_is_no_longer_one(forked, recorder):
    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)

    plan = plan_merge(
        forked.copy, forked.main,
        resolutions={f"project/{forked.target}": "theirs"},
    )

    assert plan.projects[0].labels[0].outcome == TAKE_THEIRS
    assert not plan.is_refused


def test_deciding_for_this_collection_takes_nothing(forked, recorder):
    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)

    plan = plan_merge(
        forked.copy, forked.main, resolutions={f"project/{forked.target}": "ours"}
    )

    assert plan.projects[0].labels[0].outcome == KEEP_OURS
    assert not plan.is_refused


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def test_taking_a_label_keeps_the_one_it_replaced(forked, recorder):
    from am_segmentation_manager.operations.archive import versions_of

    _edit(forked.label(forked.copy), 3, recorder)
    ours = _digest(forked.label(forked.main))

    report = apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    assert report.labels_taken == 1
    assert _digest(forked.label(forked.main)) == _digest(forked.label(forked.copy))
    kept = versions_of(forked.project, forked.target)
    assert kept and _digest(kept[-1].path) == ours


def test_taking_a_label_records_where_it_came_from(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)

    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    (event,) = [e for e in read_log(forked.project).events if e.kind == "merged"]
    assert event.data["resolution"] == TAKE_THEIRS
    assert event.data["source_project_id"] == "9f2c4b11"


def test_the_copys_history_comes_across(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)
    before = len(read_log(forked.project).events)

    report = apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    assert report.entries_added >= 1
    assert len(read_log(forked.project).events) > before


def test_merging_twice_changes_nothing_the_second_time(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)
    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)
    after_first = _digest(forked.label(forked.main))
    entries = len(read_log(forked.project).events)

    plan = plan_merge(forked.copy, forked.main)
    if plan.counted(TAKE_THEIRS) or any(p.incoming for p in plan.projects):
        apply_merge(plan, recorder=recorder)

    assert _digest(forked.label(forked.main)) == after_first
    assert len(read_log(forked.project).events) == entries


def test_a_label_only_the_copy_has_is_added(forked, recorder):
    condition = sorted(forked.copy.rglob("*_labels"))
    directory = condition[0] if condition else sorted(forked.copy.rglob("labels"))[0]
    new = directory / "extra_SEM.tif"
    _write(new, 2)
    image_directory = sorted(forked.copy.rglob("sem"))
    if image_directory:
        _write(image_directory[0] / "extra_SEM.tif", 0)

    plan = plan_merge(forked.copy, forked.main)
    taken = [item for item in plan.projects[0].labels if item.outcome == TAKE_THEIRS]

    assert taken, "a label the collection does not have should be brought in"


def test_a_target_the_copy_does_not_hold_is_never_deleted(forked, recorder):
    """A copy holds a fraction of a collection; absence is not a deletion."""
    extra = sorted(forked.main.rglob("*_labels"))
    directory = extra[0] if extra else sorted(forked.main.rglob("labels"))[0]
    only_here = directory / "only_here_SEM.tif"
    _write(only_here, 5)

    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    assert only_here.is_file()


# ---------------------------------------------------------------------------
# Metadata and state
# ---------------------------------------------------------------------------

def test_two_people_editing_different_fields_is_not_a_conflict(forked, recorder):
    from am_segmentation_manager.operations.metadata_edit import update_project_metadata

    update_project_metadata(forked.project, {"sample": "yeast"}, recorder=recorder)
    update_project_metadata(
        forked.copy / "project", {"grid_type": "UltrAuFoil"}, recorder=recorder
    )

    plan = plan_merge(forked.copy, forked.main)

    outcomes = {item.field: item.outcome for item in plan.projects[0].metadata}
    assert outcomes == {"sample": KEEP_OURS, "grid_type": TAKE_THEIRS}
    assert not plan.is_refused


def test_the_same_field_edited_differently_is_a_conflict(forked, recorder):
    from am_segmentation_manager.operations.metadata_edit import update_project_metadata

    update_project_metadata(forked.project, {"sample": "yeast"}, recorder=recorder)
    update_project_metadata(
        forked.copy / "project", {"sample": "HeLa"}, recorder=recorder
    )

    plan = plan_merge(forked.copy, forked.main)

    assert plan.projects[0].metadata[0].outcome == CONFLICT
    assert plan.is_refused


def test_a_state_changed_only_in_the_copy_is_taken(forked, recorder):
    from am_segmentation_manager.operations.selection import transition_records

    transition_records(_records(forked.copy)[:1], "deselected", recorder=recorder)

    plan = plan_merge(forked.copy, forked.main)

    taken = [item for item in plan.projects[0].states if item.outcome == TAKE_THEIRS]
    assert len(taken) == 1 and taken[0].theirs == "deselected"


def test_a_state_both_sides_moved_differently_is_a_conflict(forked, recorder):
    from am_segmentation_manager.operations.selection import transition_records

    transition_records(_records(forked.main)[:1], "deselected", recorder=recorder)
    transition_records(_records(forked.copy)[:1], "corrected", recorder=recorder)

    plan = plan_merge(forked.copy, forked.main)

    assert any(item.outcome == CONFLICT for item in plan.projects[0].states)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_previews_by_default(forked, recorder, capsys):
    from am_segmentation_manager.cli import main

    _edit(forked.label(forked.copy), 3, recorder)
    ours = _digest(forked.label(forked.main))

    assert main(["merge", str(forked.copy), "--into", str(forked.main)]) == 0

    assert "Re-run with --apply" in capsys.readouterr().out
    assert _digest(forked.label(forked.main)) == ours


def test_the_command_refuses_while_a_conflict_stands(forked, recorder):
    from am_segmentation_manager.cli import main

    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)

    assert main(
        ["merge", str(forked.copy), "--into", str(forked.main), "--apply"]
    ) == 2


def test_the_command_takes_a_decision(forked, recorder):
    from am_segmentation_manager.cli import main

    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)
    theirs = _digest(forked.label(forked.copy))

    assert main(
        [
            "merge", str(forked.copy), "--into", str(forked.main), "--apply",
            "--resolve", f"project/{forked.target}=theirs",
        ]
    ) == 0

    assert _digest(forked.label(forked.main)) == theirs


def test_a_malformed_decision_is_refused(forked):
    from am_segmentation_manager.cli import main

    assert main(
        ["merge", str(forked.copy), "--into", str(forked.main), "--resolve", "nonsense"]
    ) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(root: Path):
    return [
        record
        for project in Scanner(root).build_index().all_projects()
        for record in project.image_records
    ]


def _baseline(root: Path, recorder: Recorder) -> None:
    from am_segmentation_manager.operations.init_history import (
        apply_baseline,
        plan_baseline,
    )

    projects = Scanner(root).build_index().all_projects()
    apply_baseline(plan_baseline(projects), recorder=recorder)


def _export(main: Path, copy: Path) -> None:
    projects = Scanner(main).build_index().all_projects()
    records = [record for project in projects for record in project.image_records]
    copy_export(plan_copy(records, copy, main, projects=projects))
    write_export_record(copy, build_export_record(projects, main))


def _edit(label: Path, value: int, recorder: Recorder) -> None:
    """Change a label the way an editor would, recording it."""
    import tifffile

    pixels = tifffile.imread(label)
    pixels[8:12, 8:12] = value
    save_labels(
        [
            LabelToSave(
                destination=label, label=pixels, fingerprint=fingerprint_of(label)
            )
        ],
        recorder=recorder,
    )


def _write(path: Path, value: int) -> Path:
    import tifffile

    array = np.zeros((16, 16), dtype=np.uint8)
    if value:
        array[2:6, 2:6] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array)
    return path


def _digest(path: Path) -> str:
    import tifffile

    from am_segmentation_manager.operations.label_io import label_fields

    return str(label_fields(tifffile.imread(path))["digest"])
