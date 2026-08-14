"""Tests for shared labels and the state they link together.

The property under test throughout: moving a shared label must never leave the
images that share it behind, because their state is read from that label. These
are the cases where getting it wrong silently corrupts the record of what has
been proofread.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.layout import base_path
from am_segmentation_manager.operations.label_groups import (
    build_group,
    build_groups,
    expand_to_groups,
    find_inconsistent_groups,
    group_key,
)
from am_segmentation_manager.operations.selection import (
    deselect_records,
    mark_corrected_records,
    plan_transition,
    repair_group_states,
    select_records,
    transition_records,
)
from am_segmentation_manager.scanner import Scanner

#: The legacy multi-condition layout: several SEM sets sharing one label set.
SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


def make_project(root: Path, filename: str = "img_0004.tif") -> Path:
    """Build a project where the SEM conditions share a label and FIB does not."""
    project = root / "project"
    lamella = project / "lam_010"
    for directory in (*SEM_VARIANTS, "FIB", "labels", "labels_FIB"):
        (lamella / directory).mkdir(parents=True)
        (lamella / directory / filename).write_bytes(b"")
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
    return project


def scan(root: Path) -> dict[str, object]:
    """Return the scanned records keyed by their image directory name.

    Keyed by the *image set* directory rather than the immediate parent, so a
    record stays findable under the same key once it has moved into a state
    subfolder.
    """
    index = Scanner(root).build_index()
    return {
        base_path(record.image_path).parent.name: record
        for project in index.all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    }


# ---------------------------------------------------------------------------
# Group identity
# ---------------------------------------------------------------------------

def test_images_sharing_a_label_share_a_group(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    keys = {group_key(records[name]) for name in SEM_VARIANTS}
    assert len(keys) == 1
    assert group_key(records["FIB"]) not in keys


def test_a_group_holds_every_image_of_the_scene(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    group = build_group(records["SEM_8"])

    assert group.is_shared is True
    assert {record.image_path.parent.name for record in group.records} == set(SEM_VARIANTS)
    assert group.label_path.name == "img_0004.tif"


def test_a_private_label_forms_a_group_of_one(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    group = build_group(records["FIB"])

    assert group.is_shared is False
    assert len(group.records) == 1


def test_an_image_without_a_label_forms_a_group_of_one(tmp_path: Path):
    project = tmp_path / "project"
    stage = project / "01-lamella" / "adaptive_polish_1"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / "img.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text("name: p\n", encoding="utf-8")

    group = build_group(scan(tmp_path)["sem"])

    assert group.label_path is None
    assert len(group.records) == 1
    assert group.state() == "selected"


def test_the_groups_of_a_whole_scan_collapse_by_label(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    groups = build_groups(records.values())

    # Three SEM sets collapse into one group; FIB is its own.
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def test_expansion_adds_the_images_that_share_a_label(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    expanded = expand_to_groups([records["SEM_8"]])

    assert {r.image_path.parent.name for r in expanded} == set(SEM_VARIANTS)


def test_expansion_preserves_the_objects_it_was_given(tmp_path: Path):
    """A caller holding a record must see that record updated, not a copy."""
    make_project(tmp_path)
    records = scan(tmp_path)
    original = records["SEM_8"]

    expanded = expand_to_groups([original])

    assert any(record is original for record in expanded)


def test_expansion_does_not_duplicate_records_already_supplied(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    expanded = expand_to_groups([records[name] for name in SEM_VARIANTS])

    assert len(expanded) == len(SEM_VARIANTS)


def test_expansion_leaves_unshared_records_alone(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    assert len(expand_to_groups([records["FIB"]])) == 1


# ---------------------------------------------------------------------------
# Linked transitions
# ---------------------------------------------------------------------------

def test_deselecting_one_condition_deselects_the_whole_scene(tmp_path: Path):
    """The label is the unit of proofreading, so the scene moves together."""
    project = make_project(tmp_path)
    records = scan(tmp_path)

    moved = deselect_records([records["SEM_8"]])

    assert moved == len(SEM_VARIANTS)
    for name in SEM_VARIANTS:
        assert (project / "lam_010" / name / "deselected" / "img_0004.tif").exists()
    assert (project / "lam_010" / "labels" / "deselected" / "img_0004.tif").exists()


def test_the_shared_label_is_moved_exactly_once(tmp_path: Path):
    """Three records reference the same label, which must not be moved thrice."""
    make_project(tmp_path)
    records = scan(tmp_path)

    plan = plan_transition([records["SEM_8"]], "deselected")

    label_moves = [move for move in plan.moves if move.source.parent.name == "labels"]
    assert len(label_moves) == 1
    # One move per image, plus the single shared label.
    assert len(plan.moves) == len(SEM_VARIANTS) + 1


def test_an_unshared_modality_is_unaffected(tmp_path: Path):
    project = make_project(tmp_path)
    records = scan(tmp_path)

    deselect_records([records["SEM_8"]])

    assert (project / "lam_010" / "FIB" / "img_0004.tif").exists()
    assert (project / "lam_010" / "labels_FIB" / "img_0004.tif").exists()


def test_a_plan_reports_how_far_it_reaches(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    plan = plan_transition([records["SEM_8"]], "deselected")

    assert plan.reaches_beyond_request is True
    assert {r.image_path.parent.name for r in plan.added} == set(SEM_VARIANTS) - {"SEM_8"}


def test_a_plan_on_an_unshared_record_reaches_no_further(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    plan = plan_transition([records["FIB"]], "deselected")

    assert plan.reaches_beyond_request is False


def test_a_round_trip_restores_every_image(tmp_path: Path):
    project = make_project(tmp_path)

    deselect_records([scan(tmp_path)["SEM_8"]])
    select_records([scan(tmp_path)["SEM_8"]])

    for name in SEM_VARIANTS:
        assert (project / "lam_010" / name / "img_0004.tif").exists()
        assert not (project / "lam_010" / name / "deselected" / "img_0004.tif").exists()
    assert (project / "lam_010" / "labels" / "img_0004.tif").exists()


def test_correcting_the_scene_protects_it_from_a_later_move(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    mark_corrected_records([records["SEM_8"]])
    rescanned = scan(tmp_path)

    with pytest.raises(PermissionError):
        deselect_records([rescanned["SEM_8_fi_1"]])


def test_corrected_data_moves_only_with_explicit_confirmation(tmp_path: Path):
    project = make_project(tmp_path)
    mark_corrected_records([scan(tmp_path)["SEM_8"]])

    deselect_records([scan(tmp_path)["SEM_8"]], allow_move_from_corrected=True)

    assert (project / "lam_010" / "labels" / "deselected" / "img_0004.tif").exists()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_a_dry_run_moves_nothing(tmp_path: Path):
    project = make_project(tmp_path)
    records = scan(tmp_path)
    before = sorted(p.relative_to(project) for p in project.rglob("*"))

    count = transition_records([records["SEM_8"]], "deselected", dry_run=True)

    assert count == len(SEM_VARIANTS)
    assert sorted(p.relative_to(project) for p in project.rglob("*")) == before


def test_a_dry_run_still_refuses_what_the_real_run_would(tmp_path: Path):
    """A rehearsal that skipped validation would be worse than useless."""
    make_project(tmp_path)
    mark_corrected_records([scan(tmp_path)["SEM_8"]])

    with pytest.raises(PermissionError):
        transition_records([scan(tmp_path)["SEM_8"]], "deselected", dry_run=True)


# ---------------------------------------------------------------------------
# State authority and repair
# ---------------------------------------------------------------------------

def _desync(project: Path) -> None:
    """Move one image out of step with its label, as an interrupted run would."""
    lamella = project / "lam_010"
    (lamella / "SEM_8" / "deselected").mkdir()
    (lamella / "SEM_8" / "img_0004.tif").rename(
        lamella / "SEM_8" / "deselected" / "img_0004.tif"
    )


def test_group_state_comes_from_the_label_not_the_image(tmp_path: Path):
    project = make_project(tmp_path)
    _desync(project)

    group = build_group(scan(tmp_path)["SEM_8"])

    # The label never moved, so the scene is still selected.
    assert group.state() == "selected"
    assert group.is_consistent is False


def test_an_inconsistent_group_is_found(tmp_path: Path):
    project = make_project(tmp_path)
    _desync(project)

    groups = find_inconsistent_groups(scan(tmp_path).values())

    assert len(groups) == 1
    assert [
        base_path(record.image_path).parent.name
        for record in groups[0].inconsistent_records()
    ] == ["SEM_8"]


def test_repair_moves_the_image_to_follow_the_label(tmp_path: Path):
    project = make_project(tmp_path)
    _desync(project)

    repaired = repair_group_states(scan(tmp_path).values())

    assert len(repaired) == 1
    assert (project / "lam_010" / "SEM_8" / "img_0004.tif").exists()
    assert not (project / "lam_010" / "SEM_8" / "deselected" / "img_0004.tif").exists()


def test_repair_is_a_no_op_when_everything_agrees(tmp_path: Path):
    make_project(tmp_path)
    assert repair_group_states(scan(tmp_path).values()) == []


def test_a_group_description_names_the_image_directories(tmp_path: Path):
    """Even when an image has moved into a state folder."""
    from am_segmentation_manager.operations.label_groups import describe_group

    project = make_project(tmp_path)
    _desync(project)

    description = describe_group(build_group(scan(tmp_path)["SEM_8"]))

    assert "deselected" not in description
    for name in SEM_VARIANTS:
        assert name in description


def test_repair_dry_run_changes_nothing(tmp_path: Path):
    project = make_project(tmp_path)
    _desync(project)

    repair_group_states(scan(tmp_path).values(), dry_run=True)

    assert (project / "lam_010" / "SEM_8" / "deselected" / "img_0004.tif").exists()


def test_the_survey_reports_a_state_mismatch(tmp_path: Path):
    from am_segmentation_manager.survey import survey_collection

    project = make_project(tmp_path)
    _desync(project)

    survey = survey_collection(tmp_path)
    mismatches = [f for f in survey.all_findings if f.category == "state-mismatch"]

    assert len(mismatches) == 1
    assert mismatches[0].severity == "warning"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _cli(tmp_path: Path, command: str, *extra: str) -> int:
    return main([
        command, "--start-path", str(tmp_path), "--project", "all", "--lamella", "all",
        "--stage", "all", "--image", "all", "--modality", "SEM", "--variant", "8", *extra,
    ])


def test_cli_dry_run_reports_without_moving(tmp_path: Path, capsys):
    """The reach of the change must be visible before it is made."""
    project = make_project(tmp_path)

    assert _cli(tmp_path, "deselect", "--dry-run") == 0

    output = capsys.readouterr().out
    assert (project / "lam_010" / "SEM_8" / "img_0004.tif").exists()
    assert "2 further image(s) share a label" in output
    assert "SEM_8_fi_1" in output and "SEM_8_fi_4" in output


def test_cli_deselect_moves_the_whole_group(tmp_path: Path):
    project = make_project(tmp_path)

    assert _cli(tmp_path, "deselect") == 0

    for name in SEM_VARIANTS:
        assert (project / "lam_010" / name / "deselected" / "img_0004.tif").exists()


def test_cli_repair_state(tmp_path: Path):
    project = make_project(tmp_path)
    _desync(project)

    assert main([
        "repair-state", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all",
    ]) == 0

    assert (project / "lam_010" / "SEM_8" / "img_0004.tif").exists()
