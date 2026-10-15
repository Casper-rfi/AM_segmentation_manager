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
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.label_groups import (
    build_group,
    build_groups,
    expand_to_groups,
    find_matching_label,
    group_key,
    labels_directory_for,
    primary_record,
    record_from_image_path,
)
from am_segmentation_manager.operations.selection import (
    deselect_records,
    mark_corrected_records,
    plan_transition,
    select_records,
    transition_records,
)
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state

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
        record.image_path.parent.name: record
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
    assert group.identity == stage / "sem_labels" / "img.tif"
    assert len(group.records) == 1


def test_the_groups_of_a_whole_scan_collapse_by_label(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    groups = build_groups(records.values())

    # Three SEM sets collapse into one group; FIB is its own.
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# The primary condition
# ---------------------------------------------------------------------------

def _named_primary(root: Path, condition: str) -> None:
    (root / "project" / "AM_families.yaml").write_text(
        f"primary:\n  SEM: {condition}\n", encoding="utf-8"
    )


def test_without_a_named_primary_the_first_condition_stands_for_the_scene(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    assert primary_record(records[name] for name in SEM_VARIANTS) is records["SEM_8"]


def test_the_primary_a_project_names_stands_for_the_scene(tmp_path: Path):
    make_project(tmp_path)
    _named_primary(tmp_path, "sem_8_FI_4")
    records = scan(tmp_path)

    assert primary_record(records[name] for name in SEM_VARIANTS) is records["SEM_8_fi_4"]


def test_a_named_primary_missing_from_a_scene_falls_back_to_the_default(tmp_path: Path):
    make_project(tmp_path)
    _named_primary(tmp_path, "SEM_8_fi_4")
    records = scan(tmp_path)

    chosen = primary_record([records["SEM_8_fi_1"], records["SEM_8"]])

    assert chosen is records["SEM_8"]


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


def test_expansion_reaches_siblings_through_the_records_the_caller_holds(tmp_path: Path):
    """A sibling the caller already holds is that record, so a change shows on it."""
    make_project(tmp_path)
    records = scan(tmp_path)

    expanded = expand_to_groups([records["SEM_8"]], known=records.values())

    for name in SEM_VARIANTS:
        assert any(record is records[name] for record in expanded), name


def test_a_transition_updates_the_siblings_the_caller_holds(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    deselect = transition_records([records["SEM_8"]], "deselected", known=records.values())

    assert deselect == len(SEM_VARIANTS)
    assert {records[name].state for name in SEM_VARIANTS} == {"deselected"}
    assert records["FIB"].state == "selected"


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
    """The label is the unit of proofreading, so the scene changes together."""
    project = make_project(tmp_path)
    records = scan(tmp_path)

    changed = deselect_records([records["SEM_8"]])

    assert changed == len(SEM_VARIANTS)
    assert {record.state for record in scan(tmp_path).values() if record.variant} == {"deselected"}
    assert read_state(project / "lam_010").states == {
        "labels/img_0004.tif": "deselected"
    }


def test_the_shared_label_is_declared_exactly_once(tmp_path: Path):
    """Three records reference the same label, which is one entry, not three."""
    make_project(tmp_path)
    records = scan(tmp_path)

    plan = plan_transition([records["SEM_8"]], "deselected")

    assert len(plan.changes) == 1
    assert plan.changes[0].key == "labels/img_0004.tif"
    assert len(plan.records) == len(SEM_VARIANTS)


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


def test_a_round_trip_leaves_nothing_declared(tmp_path: Path):
    project = make_project(tmp_path)

    deselect_records([scan(tmp_path)["SEM_8"]])
    select_records([scan(tmp_path)["SEM_8"]])

    assert read_state(project / "lam_010").states == {}
    for name in SEM_VARIANTS:
        assert (project / "lam_010" / name / "img_0004.tif").exists()


def test_correcting_the_scene_protects_it_from_a_later_move(tmp_path: Path):
    make_project(tmp_path)
    records = scan(tmp_path)

    mark_corrected_records([records["SEM_8"]])
    rescanned = scan(tmp_path)

    with pytest.raises(PermissionError):
        deselect_records([rescanned["SEM_8_fi_1"]])


def test_corrected_data_changes_only_with_explicit_confirmation(tmp_path: Path):
    project = make_project(tmp_path)
    mark_corrected_records([scan(tmp_path)["SEM_8"]])

    deselect_records([scan(tmp_path)["SEM_8"]], allow_leaving_corrected=True)

    assert read_state(project / "lam_010").states == {
        "labels/img_0004.tif": "deselected"
    }


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
# Command line
# ---------------------------------------------------------------------------

def _cli(tmp_path: Path, command: str, *extra: str) -> int:
    return main([
        command, "--start-path", str(tmp_path), "--project", "all", "--lamella", "all",
        "--stage", "all", "--image", "all", "--modality", "SEM", "--variant", "8", *extra,
    ])


def test_cli_dry_run_reports_without_writing(tmp_path: Path, capsys):
    """The reach of the change must be visible before it is made."""
    project = make_project(tmp_path)

    assert _cli(tmp_path, "deselect", "--dry-run") == 0

    output = capsys.readouterr().out
    assert (project / "lam_010" / "SEM_8" / "img_0004.tif").exists()
    assert "2 further image(s) share a label" in output
    assert "SEM_8_fi_1" in output and "SEM_8_fi_4" in output


def test_cli_deselect_declares_the_whole_group(tmp_path: Path):
    project = make_project(tmp_path)

    assert _cli(tmp_path, "deselect") == 0

    assert read_state(project / "lam_010").states == {
        "labels/img_0004.tif": "deselected"
    }
    assert {record.state for record in scan(tmp_path).values() if record.variant} == {"deselected"}




# ---------------------------------------------------------------------------
# Images sitting loose in their stage
# ---------------------------------------------------------------------------

#: The prototype's Adaptive Polishing pair: one scene, both beams, loose in the
#: lamella directory with no image directory to take a label directory from.
LOOSE_PAIR = (
    "adapt_mill_img_000_electron_000.tif",
    "adapt_mill_img_000_ion_000.tif",
)


def build_loose_lamella(tmp_path: Path) -> Path:
    """Return a lamella holding one loose Adaptive Polishing acquisition."""
    lamella = tmp_path / "project" / "01-enough-bear"
    lamella.mkdir(parents=True)
    for name in LOOSE_PAIR:
        (lamella / name).write_bytes(b"")
    return lamella


def test_a_loose_image_is_keyed_by_its_series_and_beam(tmp_path: Path) -> None:
    """One directory holds two sets, so the answer cannot come from the path."""
    lamella = build_loose_lamella(tmp_path)
    electron, ion = (lamella / name for name in LOOSE_PAIR)

    assert Path(group_key(record_from_image_path(electron))).parts[-2:] == (
        "adaptive_sem_labels", LOOSE_PAIR[0]
    )
    assert Path(group_key(record_from_image_path(ion))).parts[-2:] == (
        "adaptive_fib_labels", LOOSE_PAIR[1]
    )


def test_a_loose_record_carries_where_its_label_belongs(tmp_path: Path) -> None:
    lamella = build_loose_lamella(tmp_path)
    record = record_from_image_path(lamella / LOOSE_PAIR[0])

    assert record.modality == "SEM"
    assert record.series == "adaptive"
    assert record.labels_directory == lamella / "adaptive_sem_labels"
    assert record.stage_path == lamella


def test_a_loose_image_whose_name_says_nothing_still_gets_a_bare_record(
    tmp_path: Path,
) -> None:
    lamella = build_loose_lamella(tmp_path)
    (lamella / "something_else.tif").write_bytes(b"")
    record = record_from_image_path(lamella / "something_else.tif")

    assert record.labels_directory is None
    assert group_key(record) == str(lamella / "something_else.tif")


@pytest.mark.parametrize("name", ["ref_trench_finished_eb.tif", "notes_image.tif"])
def test_an_ancestor_named_like_a_condition_is_not_one(tmp_path: Path, name: str) -> None:
    """A collection folder called ``sem_runs`` never makes a loose image a condition's.

    Only the image's own directory is read as one, as the scan reads it, so the
    record and its group agree and neither points outside the lamella.
    """
    lamella = tmp_path / "sem_runs" / "project" / "lamella-01"
    lamella.mkdir(parents=True)
    (lamella.parent / "AM_metadata.yaml").write_text("", encoding="utf-8")
    (lamella / name).write_bytes(b"")
    record = record_from_image_path(lamella / name)

    assert record.family is None
    assert record.labels_directory in (None, lamella / "reference_sem_labels")
    assert record.labels_directory == labels_directory_for(lamella / name)


def test_a_loose_group_can_be_declared_corrected(tmp_path: Path) -> None:
    """The state file lives in the lamella, which stands in for the stage."""
    lamella = build_loose_lamella(tmp_path)
    records = [record_from_image_path(lamella / name) for name in LOOSE_PAIR]

    assert mark_corrected_records(records) == 2

    assert read_state(lamella).states == {
        f"adaptive_sem_labels/{LOOSE_PAIR[0]}": "corrected",
        f"adaptive_fib_labels/{LOOSE_PAIR[1]}": "corrected",
    }


def test_find_matching_label(tmp_path: Path):
    """A group has one label in one place, so there is nothing to choose."""
    stage = tmp_path / "stage"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    img = sem / "img1.tif"
    label = labels / "img1.tif"
    label.write_text("lbl")

    record = ImageRecord(image_path=img, labels=[label], state="corrected")
    assert find_matching_label(record) == label

    absent = ImageRecord(image_path=img, labels=[labels / "gone.tif"])
    assert find_matching_label(absent) is None
