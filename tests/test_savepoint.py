"""Tests for naming a state of the data, and asking what it can still reach.

A savepoint is a claim, and the failure that matters is a claim nobody can act
on: a state it names that no version holds any more. So most of what is pinned
here is the reporting -- that it says what it can reach *before* anything is
written, that it never overstates, and that closing the gap actually closes it.

The other property: a savepoint is a set of per-project entries, not an object.
Coverage is what is found by looking, never a list somebody stored.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.history import Recorder, read_log
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.operations.label_save import LabelToSave, save_labels
from am_segmentation_manager.operations.savepoint import (
    ARCHIVED,
    AT_RISK,
    CURRENT,
    LOST,
    SavepointRefused,
    apply_savepoint,
    find_savepoint,
    list_savepoints,
    plan_savepoint,
    reachability,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset, start_history

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


@pytest.fixture
def recorder() -> Recorder:
    """A recorder attributing everything to a named author."""
    return Recorder(by="casper", by_source="flag")


@pytest.fixture
def collection(tmp_path: Path, recorder):
    """Two projects that record, with their labels baselined."""
    for name in ("project_a", "project_b"):
        dataset = build_dataset(
            tmp_path, name=name, lamellae=1, acquisitions=2, labels="annotated"
        )
        start_history(dataset.project)
    _baseline(tmp_path, recorder)
    return tmp_path


def _projects(root: Path):
    return Scanner(root).build_index().all_projects()


def _baseline(root: Path, recorder: Recorder) -> None:
    from am_segmentation_manager.operations.init_history import (
        apply_baseline,
        plan_baseline,
    )

    apply_baseline(plan_baseline(_projects(root)), recorder=recorder)


def _labels(root: Path) -> list[Path]:
    return sorted(root.rglob("*labels/*.tif"))


def _edit(label: Path, value: int, recorder: Recorder) -> None:
    """Change a label the way the proofreading view does."""
    import tifffile

    pixels = tifffile.imread(label)
    pixels[4:8, 4:8] = value
    save_labels(
        [LabelToSave(destination=label, label=pixels, fingerprint=fingerprint_of(label))],
        recorder=recorder,
    )


def _convert_over(label: Path, value: int, recorder: Recorder) -> None:
    """Write a model's guess over a label, recording it as a conversion."""
    import tifffile

    from am_segmentation_manager.operations.archive import Archivist
    from am_segmentation_manager.operations.label_io import label_fields

    before = label_fields(tifffile.imread(label))
    archived = Archivist(recorder).keep(
        label, str(before["digest"]), empty=bool(before["empty"])
    )
    pixels = tifffile.imread(label)
    pixels[2:5, 2:5] = value
    tifffile.imwrite(label, pixels)
    recorder.record(
        label, "label-converted", model="fpn_v6", prediction_kind="am_seg",
        archived_as=archived, **label_fields(pixels),
    )


# ---------------------------------------------------------------------------
# Taking one
# ---------------------------------------------------------------------------

def test_a_savepoint_is_one_entry_per_project_sharing_an_id(collection, recorder):
    plan = plan_savepoint(_projects(collection), "round 3")
    report = apply_savepoint(plan, recorder=recorder)

    assert report.entries_written == 2
    ids = {
        event.data["savepoint"]
        for project in _projects(collection)
        for event in read_log(project.path).events
        if event.kind == "checkpoint"
    }
    assert ids == {plan.id}, "one id across the projects in scope"


def test_taking_one_writes_nothing_else(collection, recorder):
    import hashlib

    def fingerprints():
        return {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(collection.rglob("*"))
            if path.is_file() and path.name != "AM_history.jsonl"
        }

    before = fingerprints()
    apply_savepoint(
        plan_savepoint(_projects(collection), "round 3"), recorder=recorder
    )

    assert fingerprints() == before, "only the logs are appended to"


def test_a_savepoint_needs_a_title(collection):
    with pytest.raises(SavepointRefused, match="needs a title"):
        plan_savepoint(_projects(collection), "   ")


def test_a_collection_that_records_nothing_has_no_state_to_name(tmp_path: Path):
    build_dataset(tmp_path, name="project_a", labels="annotated")

    with pytest.raises(SavepointRefused, match="no state to name"):
        plan_savepoint(_projects(tmp_path), "round 3")


# ---------------------------------------------------------------------------
# Coverage is found, never stored
# ---------------------------------------------------------------------------

def test_a_project_added_afterwards_can_join_and_coverage_grows(collection, recorder):
    apply_savepoint(
        plan_savepoint(_projects(collection)[:1], "round 3"), recorder=recorder
    )
    (point,) = list_savepoints(_projects(collection))
    assert point.found == 1

    joined = plan_savepoint(
        _projects(collection), "", joining=find_savepoint(_projects(collection), "round 3")
    )
    apply_savepoint(joined, recorder=recorder)

    (point,) = list_savepoints(_projects(collection))
    assert point.found == 2
    assert point.title == "round 3", "joining keeps the name it already had"


def test_the_count_at_creation_is_a_note_and_not_a_denominator(collection, recorder):
    """A machine holding a subset reports fewer, and that is the honest answer."""
    apply_savepoint(
        plan_savepoint(_projects(collection), "round 3"), recorder=recorder
    )

    one = [p for p in _projects(collection) if p.key == "project_a"]
    (point,) = list_savepoints(one)

    assert point.found == 1
    assert point.covered_at_creation == 2
    assert "2 at creation" in point.describe()


def test_joining_adds_only_the_projects_that_lack_it(collection, recorder):
    apply_savepoint(
        plan_savepoint(_projects(collection), "round 3"), recorder=recorder
    )
    point = find_savepoint(_projects(collection), "round 3")

    again = plan_savepoint(_projects(collection), "", joining=point)

    assert again.projects == ()


def test_two_savepoints_sharing_a_title_are_refused_rather_than_guessed_between(
    collection, recorder
):
    for _ in range(2):
        apply_savepoint(
            plan_savepoint(_projects(collection), "round 3"), recorder=recorder
        )

    with pytest.raises(SavepointRefused, match="names 2 savepoints"):
        find_savepoint(_projects(collection), "round 3")


def test_a_savepoint_can_always_be_named_by_its_id(collection, recorder):
    first = apply_savepoint(
        plan_savepoint(_projects(collection), "round 3"), recorder=recorder
    )
    apply_savepoint(
        plan_savepoint(_projects(collection), "round 3"), recorder=recorder
    )

    assert find_savepoint(_projects(collection), first.id).id == first.id


def test_a_name_nothing_carries_says_what_is_here(collection, recorder):
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)

    with pytest.raises(SavepointRefused, match="There is round 3"):
        find_savepoint(_projects(collection), "round 9")


# ---------------------------------------------------------------------------
# What it can still reach
# ---------------------------------------------------------------------------

def test_a_label_still_holding_what_was_named_is_reachable(collection, recorder):
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)

    point = find_savepoint(_projects(collection), "round 3")

    assert point.targets and point.reachable == point.targets
    assert {state.where for project in point.projects for state in project.targets} == {
        CURRENT
    }


def test_a_label_edited_since_is_reachable_through_the_archive(collection, recorder):
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _edit(_labels(collection)[0], 3, recorder)

    point = find_savepoint(_projects(collection), "round 3")

    kept = [
        state
        for project in point.projects
        for state in project.targets
        if state.where == ARCHIVED
    ]
    assert len(kept) == 1 and kept[0].archived_as
    assert point.reachable == point.targets


def test_untouched_machine_output_is_reported_as_at_risk(collection, recorder):
    """The archive skips it, so the next conversion takes the state with it."""
    _convert_over(_labels(collection)[0], 3, recorder)

    plan = plan_savepoint(_projects(collection), "round 3")

    risky = plan.at_risk
    assert len(risky) == 1
    assert risky[0][1].where == AT_RISK
    assert "only until the next conversion" in " ".join(plan.describe())


def test_at_risk_is_reported_before_anything_is_written(collection, recorder):
    _convert_over(_labels(collection)[0], 3, recorder)

    plan = plan_savepoint(_projects(collection), "round 3")

    assert plan.at_risk, "the answer comes with the preview, not after the fact"
    assert not any(
        event.kind == "checkpoint"
        for project in _projects(collection)
        for event in read_log(project.path).events
    )


def test_closing_the_gap_makes_it_reachable(collection, recorder):
    _convert_over(_labels(collection)[0], 3, recorder)
    plan = plan_savepoint(_projects(collection), "round 3")

    report = apply_savepoint(plan, recorder=recorder, close_the_gap=True)

    assert len(report.versions_kept) == 1
    point = find_savepoint(_projects(collection), "round 3")
    assert not point.at_risk, "what was at risk is now kept"
    assert point.reachable == point.targets


def test_closing_the_gap_survives_the_conversion_it_was_for(collection, recorder):
    """The whole point: convert again, and the named state is still there."""
    label = _labels(collection)[0]
    _convert_over(label, 3, recorder)
    apply_savepoint(
        plan_savepoint(_projects(collection), "round 3"),
        recorder=recorder, close_the_gap=True,
    )

    _convert_over(label, 5, recorder)

    point = find_savepoint(_projects(collection), "round 3")
    assert point.reachable == point.targets
    assert not any(
        state.where == LOST for project in point.projects for state in project.targets
    )


def test_without_closing_the_gap_the_next_conversion_loses_it(collection, recorder):
    """A savepoint promises nothing about pixels, and says so rather than lying."""
    label = _labels(collection)[0]
    _convert_over(label, 3, recorder)
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)

    _convert_over(label, 5, recorder)

    point = find_savepoint(_projects(collection), "round 3")
    lost = [
        state
        for project in point.projects
        for state in project.targets
        if state.where == LOST
    ]
    assert len(lost) == 1
    assert point.reachable < point.targets
    assert "reachable" in point.describe()


def test_a_pruned_version_reads_as_lost_rather_than_as_present(collection, recorder):
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    label = _labels(collection)[0]
    _edit(label, 3, recorder)
    _edit(label, 4, recorder)

    project = next(p for p in _projects(collection) if str(label).startswith(str(p.path)))
    for kept in sorted((project.path / "AM_archive").rglob("*.tif")):
        kept.unlink()

    point = find_savepoint(_projects(collection), "round 3")

    assert any(
        state.where == LOST for carried in point.projects for state in carried.targets
    )


def test_the_state_named_is_the_one_before_the_savepoint(collection, recorder):
    """Work done after it must not be read back as part of it."""
    label = _labels(collection)[0]
    _edit(label, 3, recorder)
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _edit(label, 7, recorder)

    point = find_savepoint(_projects(collection), "round 3")
    project = point.projects[0]
    log = read_log(project.project.path)
    at_savepoint = {
        state.target: state.digest for state in project.targets
    }
    edits = [
        event for event in log.events
        if event.kind == "label-saved" and event.target in at_savepoint
    ]
    assert edits, "the fixture edited something"
    assert at_savepoint[edits[0].target] == edits[0].data["digest"], (
        "the first edit is in; the second, after the savepoint, is not"
    )


def test_reachability_of_a_project_that_has_recorded_nothing_is_empty(
    tmp_path: Path, recorder
):
    dataset = build_dataset(tmp_path, name="project_a", labels="annotated")
    start_history(dataset.project)

    assert reachability(dataset.project, read_log(dataset.project)) == ()


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_previews_by_default(collection, capsys):
    from am_segmentation_manager.cli import main

    assert main(["savepoint", "--start-path", str(collection), "--title", "round 3"]) == 0

    assert not any(
        event.kind == "checkpoint"
        for project in _projects(collection)
        for event in read_log(project.path).events
    )


def test_the_command_refuses_inside_a_copy(collection, tmp_path: Path, capsys):
    """A copy may hold part of a project, so a savepoint here would overclaim."""
    from am_segmentation_manager.cli import main
    from am_segmentation_manager.operations.export_record import (
        build_export_record,
        write_export_record,
    )

    write_export_record(
        collection, build_export_record(_projects(collection), collection)
    )

    assert main(
        ["savepoint", "--start-path", str(collection), "--title", "round 3", "--apply"]
    ) == 2


def test_the_listing_says_what_each_covers_and_reaches(collection, recorder, capsys):
    from am_segmentation_manager.cli import main

    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)

    assert main(["savepoints", "--start-path", str(collection)]) == 0

    printed = capsys.readouterr().out
    assert "round 3" in printed and "2 project(s) here" in printed
    assert "labels reachable" in printed


# ---------------------------------------------------------------------------
# Undoing a round by name
# ---------------------------------------------------------------------------

def _records(root: Path):
    return [record for project in _projects(root) for record in project.image_records]


def test_a_round_of_work_goes_back_by_name(collection, recorder):
    import tifffile

    from am_segmentation_manager.operations.label_io import label_fields
    from am_segmentation_manager.operations.revert import apply_revert, plan_revert_to

    label = _labels(collection)[0]
    at_savepoint = label_fields(tifffile.imread(label))["digest"]
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _convert_over(label, 3, recorder)

    point = find_savepoint(_projects(collection), "round 3")
    plan = plan_revert_to(_records(collection), point)
    apply_revert(plan, recorder=recorder, allow_corrected=True)

    assert label_fields(tifffile.imread(label))["digest"] == at_savepoint


def test_a_label_already_holding_what_it_held_then_is_left_alone(collection, recorder):
    """Putting back what is already there would keep a copy of nothing."""
    from am_segmentation_manager.operations.revert import ALREADY_THERE, plan_revert_to

    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)

    plan = plan_revert_to(
        _records(collection), find_savepoint(_projects(collection), "round 3")
    )

    assert not plan.revertible
    assert {item.blocked for item in plan.items} == {ALREADY_THERE}


def test_only_what_changed_since_is_offered(collection, recorder):
    from am_segmentation_manager.operations.revert import plan_revert_to

    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _convert_over(_labels(collection)[0], 3, recorder)

    plan = plan_revert_to(
        _records(collection), find_savepoint(_projects(collection), "round 3")
    )

    assert len(plan.revertible) == 1
    assert plan.revertible[0].label_path == _labels(collection)[0]


def test_a_state_that_cannot_be_reached_is_reported_rather_than_attempted(
    collection, recorder
):
    from am_segmentation_manager.operations.revert import UNREACHABLE, plan_revert_to

    label = _labels(collection)[0]
    _convert_over(label, 3, recorder)
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _convert_over(label, 5, recorder)   # the gap was never closed

    plan = plan_revert_to(
        _records(collection), find_savepoint(_projects(collection), "round 3")
    )

    blocked = [item for item in plan.items if item.blocked == UNREACHABLE]
    assert len(blocked) == 1 and not plan.revertible


def test_reverting_a_round_refuses_proofread_work_without_confirmation(
    collection, recorder
):
    from am_segmentation_manager.operations.revert import apply_revert, plan_revert_to
    from am_segmentation_manager.operations.selection import transition_records

    label = _labels(collection)[0]
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _convert_over(label, 3, recorder)
    group = [r for r in _records(collection) if label in r.labels]
    transition_records(group, "corrected", recorder=recorder)

    plan = plan_revert_to(
        _records(collection), find_savepoint(_projects(collection), "round 3")
    )

    assert plan.corrected
    with pytest.raises(PermissionError):
        apply_revert(plan, recorder=recorder)


def test_the_command_undoes_a_round_by_name(collection, recorder):
    from am_segmentation_manager.cli import main

    label = _labels(collection)[0]
    apply_savepoint(plan_savepoint(_projects(collection), "round 3"), recorder=recorder)
    _convert_over(label, 3, recorder)

    assert main([
        "revert", str(collection), "--modality", "all", "--state", "all",
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
        "--to", "round 3", "--apply", "--confirm-corrected",
    ]) == 0


def test_the_command_refuses_a_savepoint_nobody_took(collection):
    from am_segmentation_manager.cli import main

    assert main([
        "revert", str(collection), "--modality", "all", "--state", "all",
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
        "--to", "round 9",
    ]) == 2
