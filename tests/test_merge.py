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
    EVERYTHING,
    KEEP_OURS,
    OURS,
    TAKE_THEIRS,
    THEIRS,
    LabelMerge,
    MergePlan,
    MergeRefused,
    MetadataMerge,
    ProjectMerge,
    StateMerge,
    apply_merge,
    decisions_for,
    plan_merge,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset, start_history

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
    start_history(dataset.project)
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


def test_taking_a_label_records_how_much_it_changed(forked, recorder):
    """A version list is only readable if every row carries this number."""
    _edit(forked.label(forked.copy), 3, recorder)

    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    (event,) = [e for e in read_log(forked.project).events if e.kind == "merged"]
    assert isinstance(event.data.get("changed_fraction"), float)
    assert event.data["changed_fraction"] > 0


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


def test_the_command_decides_a_whole_project_at_once(forked, recorder):
    """Its labels and its fields together, as a project's mark does in the
    window."""
    from am_segmentation_manager.cli import main
    from am_segmentation_manager.io.yaml import read_yaml
    from am_segmentation_manager.operations.metadata_edit import update_project_metadata

    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)
    update_project_metadata(forked.project, {"sample": "yeast"}, recorder=recorder)
    update_project_metadata(
        forked.copy / "project", {"sample": "HeLa"}, recorder=recorder
    )
    theirs = _digest(forked.label(forked.copy))

    assert main(
        [
            "merge", str(forked.copy), "--into", str(forked.main), "--apply",
            "--resolve", "project=theirs",
        ]
    ) == 0

    assert _digest(forked.label(forked.main)) == theirs
    assert read_yaml(forked.project / "AM_metadata.yaml")["metadata"]["sample"] == "HeLa"


def test_a_decision_that_decides_nothing_is_refused(forked, recorder):
    """A mistyped path deciding nothing would pass unnoticed otherwise."""
    from am_segmentation_manager.cli import main

    _edit(forked.label(forked.main), 3, recorder)
    _edit(forked.label(forked.copy), 4, recorder)
    ours = _digest(forked.label(forked.main))

    assert main(
        [
            "merge", str(forked.copy), "--into", str(forked.main), "--apply",
            "--resolve", f"project/{forked.target}=theirs",
            "--resolve", "elsewhere=ours",
        ]
    ) == 2
    assert _digest(forked.label(forked.main)) == ours


# ---------------------------------------------------------------------------
# Deciding by scope
# ---------------------------------------------------------------------------

def _plan_of(*projects) -> MergePlan:
    """Return a plan holding label and field conflicts, with no trees behind it.

    Each project is ``(key, label targets, field names)``.
    """
    return MergePlan(
        source=Path("copy"),
        destination=Path("collection"),
        record=None,
        projects=tuple(
            ProjectMerge(
                key=key,
                our_root=Path(key),
                their_root=Path(key),
                labels=tuple(
                    LabelMerge(
                        target=target, ancestor="a", ours="b", theirs="c",
                        our_path=Path(target), their_path=Path(target),
                        outcome=CONFLICT,
                    )
                    for target in targets
                ),
                metadata=tuple(
                    MetadataMerge(
                        field=name, ancestor=None, ours="x", theirs="y",
                        outcome=CONFLICT,
                    )
                    for name in fields
                ),
            )
            for key, targets, fields in projects
        ),
    )


@pytest.fixture
def scoped() -> MergePlan:
    """Conflicts in two folders, one project's name the start of another's."""
    return _plan_of(
        ("f/p1", ("lam1/s/a.tif", "lam1/s/b.tif", "lam2/s/a.tif"), ("sample",)),
        ("f/p10", ("lam1/s/a.tif",), ()),
        ("g/p3", ("lam1/s/a.tif",), ()),
    )


def test_a_scope_decides_every_conflict_under_it_and_nothing_else(scoped):
    assert decisions_for(scoped, {"f/p1/lam1": THEIRS}) == {
        "f/p1/lam1/s/a.tif": THEIRS,
        "f/p1/lam1/s/b.tif": THEIRS,
    }


def test_a_project_holds_its_fields_and_a_folder_its_projects(scoped):
    by_project = decisions_for(scoped, {"f/p1": OURS})
    assert set(by_project) == {
        "f/p1/lam1/s/a.tif", "f/p1/lam1/s/b.tif", "f/p1/lam2/s/a.tif", "f/p1:sample"
    }, "a project whose name only starts the same, f/p10, is not under it"

    by_folder = decisions_for(scoped, {"f": OURS})
    assert set(by_folder) == {key for key in scoped.conflict_keys if key.startswith("f/")}


def test_everything_is_one_scope(scoped):
    assert set(decisions_for(scoped, {EVERYTHING: THEIRS})) == set(scoped.conflict_keys)


def test_the_narrowest_scope_decides_whatever_order_they_come_in(scoped):
    given = {EVERYTHING: THEIRS, "f/p1": OURS, "f/p1/lam1/s/a.tif": THEIRS}
    for chosen in (given, dict(reversed(list(given.items())))):
        decided = decisions_for(scoped, chosen)
        assert decided["f/p1/lam1/s/a.tif"] == THEIRS
        assert decided["f/p1/lam1/s/b.tif"] == OURS
        assert decided["g/p3/lam1/s/a.tif"] == THEIRS


def test_a_scope_holding_no_conflict_is_refused(scoped):
    with pytest.raises(ValueError, match="f/p2"):
        decisions_for(scoped, {"f/p1": OURS, "f/p2": THEIRS})


def test_a_scope_is_read_as_its_conflicts_are_named(scoped):
    assert decisions_for(scoped, {"./f/p1/lam1/": THEIRS}) == decisions_for(
        scoped, {"f/p1/lam1": THEIRS}
    )


def test_a_side_that_is_neither_is_refused(scoped):
    with pytest.raises(ValueError, match="both"):
        decisions_for(scoped, {"f": "both"})


def test_a_label_and_its_state_are_one_name(scoped):
    project = scoped.projects[0]
    state = StateMerge(
        target="lam1/s/a.tif", ancestor="selected", ours="corrected",
        theirs="deselected", record=None, outcome=CONFLICT,
    )
    plan = MergePlan(
        source=scoped.source, destination=scoped.destination, record=None,
        projects=(
            ProjectMerge(
                key=project.key, our_root=project.our_root,
                their_root=project.their_root, labels=project.labels[:1],
                states=(state,),
            ),
        ),
    )

    assert plan.conflict_keys == ("f/p1/lam1/s/a.tif",)


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


# ---------------------------------------------------------------------------
# Proofread work on the receiving side
# ---------------------------------------------------------------------------

def _mark_corrected(root: Path, recorder: Recorder) -> None:
    """Declare the first group of *root* proofread work."""
    from am_segmentation_manager.operations.selection import transition_records

    transition_records(_records(root)[:1], "corrected", recorder=recorder)


def test_a_merge_over_proofread_work_is_refused_without_confirmation(forked, recorder):
    """CLAUDE.md: no automated operation may replace a corrected label unasked."""
    _edit(forked.label(forked.copy), 3, recorder)
    _mark_corrected(forked.main, recorder)
    ours = _digest(forked.label(forked.main))

    plan = plan_merge(forked.copy, forked.main)
    assert plan.corrected, "the plan has to say so before it is carried out"

    with pytest.raises(PermissionError, match="corrected item"):
        apply_merge(plan, recorder=recorder)

    assert _digest(forked.label(forked.main)) == ours, "nothing was written"


def test_proofread_work_is_taken_when_it_is_asked_for(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)
    _mark_corrected(forked.main, recorder)
    theirs = _digest(forked.label(forked.copy))

    plan = plan_merge(forked.copy, forked.main)
    report = apply_merge(plan, recorder=recorder, allow_corrected=True)

    assert _digest(forked.label(forked.main)) == theirs
    assert report.archived, "and what was here is kept, so it can be undone"


def test_one_proofread_label_stops_the_whole_merge(forked, recorder, tmp_path):
    """Half a merge would make what happened depend on the order of a loop."""
    _edit(forked.label(forked.copy), 3, recorder)
    _mark_corrected(forked.main, recorder)
    before = {
        path: fingerprint_of(path)
        for path in sorted(forked.main.rglob("*.tif"))
    }

    with pytest.raises(PermissionError):
        apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    after = {
        path: fingerprint_of(path)
        for path in sorted(forked.main.rglob("*.tif"))
    }
    assert after == before


def test_a_merge_will_not_silently_drop_a_corrected_claim(tmp_path: Path, recorder):
    """Permission used to be granted exactly when it was needed, which is no guard.

    The claim has to predate the copy, or the two sides moved it differently and
    it is a conflict rather than something taken.
    """
    from am_segmentation_manager.operations.selection import (
        current_state,
        transition_records,
    )

    main = tmp_path / "collection"
    main.mkdir()
    dataset = build_dataset(main, lamellae=1, acquisitions=2, labels="annotated")
    start_history(dataset.project)
    _baseline(main, recorder)
    transition_records(_records(main)[:1], "corrected", recorder=recorder)

    copy = tmp_path / "copy"
    _export(main, copy)
    # Deliberate on the copy, which is why it reaches the merge at all.
    transition_records(
        _records(copy)[:1], "deselected",
        allow_leaving_corrected=True, recorder=recorder,
    )

    plan = plan_merge(copy, main)
    assert [item.outcome for item in plan.projects[0].states] == [TAKE_THEIRS]
    assert plan.corrected, "dropping a corrected claim is proofread work being lost"

    with pytest.raises(PermissionError):
        apply_merge(plan, recorder=recorder)
    assert current_state(_records(main)[0]) == "corrected"

    apply_merge(plan_merge(copy, main), recorder=recorder, allow_corrected=True)
    assert current_state(_records(main)[0]) == "deselected"


def test_a_label_that_changed_since_the_plan_stops_the_merge(forked, recorder):
    """A merge carried out against a tree that moved on would write blind."""
    from am_segmentation_manager.operations.label_save import StaleLabels

    _edit(forked.label(forked.copy), 3, recorder)
    plan = plan_merge(forked.copy, forked.main)
    _edit(forked.label(forked.main), 1, recorder)   # somebody else, in between

    with pytest.raises(StaleLabels):
        apply_merge(plan, recorder=recorder)


def test_taking_a_label_records_a_merge_and_what_it_grew_out_of(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)

    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    merged = [
        event for event in read_log(forked.project).events
        if event.kind == "merged" and event.target == forked.target
    ]
    assert len(merged) == 1
    assert merged[0].data["source_project_id"] == "9f2c4b11"
    assert merged[0].data["archived_as"]
    assert merged[0].data["changed_fraction"] > 0


def test_an_empty_label_here_is_not_archived_when_it_is_replaced(tmp_path: Path, recorder):
    """The archive rule's first row: an empty placeholder holds nothing to keep.

    It has to be empty before the copy is taken, or emptying it is a change and
    the two sides conflict instead.
    """
    main = tmp_path / "collection"
    main.mkdir()
    dataset = build_dataset(main, lamellae=1, acquisitions=2, labels="empty")
    start_history(dataset.project)
    _baseline(main, recorder)

    copy = tmp_path / "copy"
    _export(main, copy)
    _edit(sorted(copy.rglob("*labels/*.tif"))[0], 3, recorder)

    report = apply_merge(plan_merge(copy, main), recorder=recorder)

    assert report.labels_taken == 1
    assert not report.archived, "an empty placeholder is not worth keeping"


# ---------------------------------------------------------------------------
# A copy that is worked on and brought back more than once
# ---------------------------------------------------------------------------

def test_a_copy_worked_on_again_fast_forwards_rather_than_conflicting(forked, recorder):
    """Use case 1, iterated: the laptop keeps its copy and merges back twice.

    Its fork point still names the entry the first export ended at, so an
    ancestor read from that would call every label taken the first time a
    conflict -- ours changed (we took theirs) and theirs changed again.
    """
    _edit(forked.label(forked.copy), 3, recorder)
    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)
    assert _digest(forked.label(forked.main)) == _digest(forked.label(forked.copy))

    _edit(forked.label(forked.copy), 4, recorder)     # more work, same copy

    plan = plan_merge(forked.copy, forked.main)

    assert not plan.conflicts, "a second round is a fast-forward, not a conflict"
    assert [item.outcome for item in plan.projects[0].labels] == [TAKE_THEIRS]

    apply_merge(plan, recorder=recorder)
    assert _digest(forked.label(forked.main)) == _digest(forked.label(forked.copy))


def test_the_second_merge_still_sees_a_real_conflict(forked, recorder):
    """Fast-forwarding must not come from having stopped comparing."""
    _edit(forked.label(forked.copy), 3, recorder)
    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    _edit(forked.label(forked.copy), 4, recorder)
    _edit(forked.label(forked.main), 5, recorder)     # both sides, differently

    plan = plan_merge(forked.copy, forked.main)

    assert plan.conflicts
    assert plan.is_refused


def test_merging_the_same_copy_twice_changes_nothing_the_second_time(forked, recorder):
    _edit(forked.label(forked.copy), 3, recorder)
    apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)
    after = fingerprint_of(forked.label(forked.main))

    report = apply_merge(plan_merge(forked.copy, forked.main), recorder=recorder)

    assert report.labels_taken == 0 and report.entries_added == 0
    assert fingerprint_of(forked.label(forked.main)) == after
