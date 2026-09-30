"""Tests for changing a project's families, and the checks guarding labels.

Each refusal is the one way a correction could silently cost labels: one left
unread, one swapped for another, one taken from a condition the change never
named, a family split into two directories, or two conditions of a family
answering to one name. The rest pins what a change that goes ahead does to the
file, the labels each condition reads, the state its images are in, and the log.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.families import (
    AM_FAMILIES_FILENAME,
    StaleFamilies,
    read_families,
)
from am_segmentation_manager.history import Recorder, read_log
from am_segmentation_manager.image_metadata import ImageMetadata
from am_segmentation_manager.models import Project
from am_segmentation_manager.operations.families_edit import (
    FamiliesValidationError,
    apply_families_change,
    plan_families_change,
    summarise_families,
    update_families,
)
from am_segmentation_manager.operations.init_history import writing_recorder
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state

IMAGE = "img_0000.tif"


def _stage(root: Path, *directories: str, state: dict[str, str] | None = None) -> Path:
    """Build a one-stage project with *directories*, each holding one file."""
    project = root / "project"
    stage = project / "lam_000" / "adaptive_polish_1"
    for directory in directories:
        (stage / directory).mkdir(parents=True)
        (stage / directory / IMAGE).write_bytes(b"")
    (project / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    if state:
        body = "".join(f"  {key}/{IMAGE}: {value}\n" for key, value in state.items())
        (stage / "AM_state.yaml").write_text(f"version: 1\nstate:\n{body}", encoding="utf-8")
    return stage


def _project(root: Path) -> Project:
    return Scanner(root).build_index().all_projects()[0]


def _labels_of(root: Path) -> dict[str, Path | None]:
    return {
        record.image_path.parent.name: record.labels_directory
        for record in _project(root).image_records
    }


def _refusal(root: Path, **request) -> str:
    with pytest.raises(FamiliesValidationError) as raised:
        plan_families_change(_project(root), **request)
    return "; ".join(raised.value.problems)


# ---------------------------------------------------------------------------
# Changes that go ahead
# ---------------------------------------------------------------------------

def test_a_condition_leaving_its_family_starts_unlabelled(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_fi8", "sem_mag2", "sem_labels")

    change = update_families(_project(tmp_path), families={"sem_mag2": "sem-mag2"})

    assert change.applied
    assert [move.image_directory.name for move in change.moves] == ["sem_mag2"]
    assert read_families(stage.parents[1]).family_of("sem_mag2") == "sem-mag2"
    labels = _labels_of(tmp_path)
    assert labels["sem_mag2"] == stage / "sem-mag2_labels"
    assert labels["sem_fi8"] == stage / "sem_labels"


def test_an_unlabelled_condition_joining_a_family_shares_its_label_and_state(tmp_path: Path):
    stage = _stage(
        tmp_path, "sem-2kV", "sem2kVb", "sem-2kV_labels",
        state={"sem-2kV_labels": "corrected", "sem2kVb_labels": "deselected"},
    )

    change = update_families(_project(tmp_path), families={"sem2kVb": "sem-2kV"})

    assert change.moves[0].labels_shared == 1
    assert [(e.old, e.new) for e in change.states.effects] == [("deselected", "corrected")]
    assert (
        f"lam_000/adaptive_polish_1/sem2kVb/{IMAGE}: deselected -> corrected" in change.describe()[-1]
    )
    assert read_state(stage).states == {f"sem-2kV_labels/{IMAGE}": "corrected"}
    assert _labels_of(tmp_path)["sem2kVb"] == stage / "sem-2kV_labels"


def test_images_forming_a_new_group_keep_a_deselection(tmp_path: Path):
    stage = _stage(
        tmp_path, "sem", "sem_mag2", "sem_labels", state={"sem_labels": "deselected"}
    )

    change = update_families(_project(tmp_path), families={"sem_mag2": "sem-mag2"})

    assert change.states.effects == ()
    assert read_state(stage).states == {
        f"sem_labels/{IMAGE}": "deselected",
        f"sem-mag2_labels/{IMAGE}": "deselected",
    }


def test_leaving_a_corrected_group_needs_confirming_and_drops_the_claim(tmp_path: Path):
    stage = _stage(
        tmp_path, "sem", "sem_mag2", "sem_labels", state={"sem_labels": "corrected"}
    )

    assert "corrected group" in _refusal(tmp_path, families={"sem_mag2": "sem-mag2"})
    change = update_families(
        _project(tmp_path), families={"sem_mag2": "sem-mag2"}, allow_leaving_corrected=True
    )

    assert [(e.old, e.new) for e in change.states.effects] == [("corrected", "selected")]
    assert read_state(stage).states == {f"sem_labels/{IMAGE}": "corrected"}


def test_images_carried_into_one_new_group_keep_a_deselection_either_brings(tmp_path: Path):
    stage = _stage(tmp_path, "sem-a", "sem-b", state={"sem-a_labels": "deselected"})

    update_families(_project(tmp_path), families={"sem-a": "sem-c", "sem-b": "sem-c"})

    assert read_state(stage).states == {f"sem-c_labels/{IMAGE}": "deselected"}


def test_a_group_nothing_reads_any_more_loses_its_state_entry(tmp_path: Path):
    """Left behind, it would be reported as orphaned curation."""
    stage = _stage(tmp_path, "sem", "sem_labels", "sem-x", state={"sem-x_labels": "deselected"})

    update_families(_project(tmp_path), families={"sem-x": "sem"})

    assert read_state(stage).states == {}


def test_names_are_matched_whatever_their_case_and_kept_as_on_disk(tmp_path: Path):
    stage = _stage(tmp_path, "sem-2kV", "sem2kVb")

    update_families(_project(tmp_path), families={"SEM2KVB": "SEM-2KV"})

    assert read_families(stage.parents[1]).families == {"sem2kVb": "sem-2kV"}


def test_clearing_returns_a_condition_to_the_family_its_name_gives(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_mag2", "sem_labels")
    update_families(_project(tmp_path), families={"sem_mag2": "sem-mag2"})

    change = update_families(_project(tmp_path), families={"sem_mag2": None})

    assert [(e.key, e.old, e.new) for e in change.entries] == [("sem_mag2", "sem-mag2", None)]
    assert _labels_of(tmp_path)["sem_mag2"] == stage / "sem_labels"
    assert (stage.parents[1] / AM_FAMILIES_FILENAME).exists()


def test_giving_a_condition_the_family_its_name_gives_is_no_change(tmp_path: Path):
    _stage(tmp_path, "sem", "sem_fi8")

    change = update_families(_project(tmp_path), families={"sem_fi8": "SEM"})

    assert change.is_empty
    assert not (tmp_path / "project" / AM_FAMILIES_FILENAME).exists()


def test_a_primary_is_set_and_cleared_without_moving_a_label(tmp_path: Path):
    project_root = _stage(tmp_path, "sem", "sem_fi8", "sem_labels").parents[1]

    change = update_families(_project(tmp_path), primary={"SEM": "SEM_FI8"})
    assert change.moves == ()
    assert read_families(project_root).primary == {"sem": "sem_fi8"}

    update_families(_project(tmp_path), primary={"sem": None})
    assert read_families(project_root).primary == {}


def test_each_entry_is_recorded(tmp_path: Path):
    project_root = _stage(tmp_path, "sem", "sem_mag2", "sem_labels").parents[1]
    project = _project(tmp_path)
    recorder = writing_recorder([project], Recorder())

    update_families(
        project, families={"sem_mag2": "sem-mag2"}, primary={"sem": "sem"}, recorder=recorder
    )

    recorded = [
        (event.target, *(event.data[name] for name in ("section", "key", "old", "new")))
        for event in read_log(project_root).events
        if event.kind == "family-changed"
    ]
    assert recorded == [
        (AM_FAMILIES_FILENAME, "families", "sem_mag2", None, "sem-mag2"),
        (AM_FAMILIES_FILENAME, "primary", "sem", None, "sem"),
    ]


def test_the_summary_marks_conditions_placed_by_the_file(tmp_path: Path):
    _stage(tmp_path, "sem", "sem_fi8", "sem_mag2", "fib")
    update_families(
        _project(tmp_path), families={"sem_mag2": "sem-mag2"}, primary={"sem": "sem_fi8"}
    )

    summaries = summarise_families(_project(tmp_path))

    assert [line for summary in summaries for line in summary.describe()] == [
        "SEM sem: sem (1), sem_fi8 (1)",
        "    primary: sem_fi8",
        "SEM sem-mag2: sem_mag2* (1)",
        "FIB fib: fib (1)",
    ]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_condition_not_in_the_project_is_refused(tmp_path: Path):
    _stage(tmp_path, "sem")

    assert "nosuch is not an image directory" in _refusal(tmp_path, families={"nosuch": "sem-x"})


def test_a_family_left_with_nobody_reading_its_labels_is_refused(tmp_path: Path):
    _stage(tmp_path, "sem_fi8", "sem_labels")

    assert "no condition would read sem_labels/" in _refusal(
        tmp_path, families={"sem_fi8": "sem-x"}
    )


def test_swapping_one_labelled_directory_for_another_is_refused(tmp_path: Path):
    _stage(tmp_path, "sem", "sem_fi8", "sem_labels", "sem-2kV", "sem-2kV_labels")

    assert "merging them is not supported" in _refusal(
        tmp_path, families={"sem_fi8": "sem-2kV"}
    )


def test_a_condition_not_named_may_not_lose_its_labels(tmp_path: Path):
    """The leaving condition's own directory was standing in for the family."""
    _stage(tmp_path, "sem", "sem_fi8", "sem_mag2", "sem_mag2_labels")

    problems = _refusal(tmp_path, families={"sem_mag2": "sem-mag2"})

    assert "sem would stop reading its labels in sem_mag2_labels/" in problems
    assert "sem_fi8 would stop reading" in problems


def test_a_family_split_between_two_directories_is_refused(tmp_path: Path):
    _stage(tmp_path, "sem", "sem_labels", "sem2kVb", "sem2kVb_labels")

    assert "would have 2 labels directories" in _refusal(tmp_path, families={"sem2kVb": "sem"})


def test_two_conditions_answering_to_one_variant_are_refused(tmp_path: Path):
    _stage(tmp_path, "sem_fi1", "sem-x_fi1")

    assert "would both be sem [fi1]" in _refusal(tmp_path, families={"sem-x_fi1": "sem"})


def test_a_primary_outside_its_family_is_refused(tmp_path: Path):
    _stage(tmp_path, "sem", "sem-2kV")

    assert "belongs to family sem-2kV, not sem" in _refusal(tmp_path, primary={"sem": "sem-2kV"})


def test_a_damaged_file_is_refused_until_corrected_by_hand(tmp_path: Path):
    _stage(tmp_path, "sem", "sem_mag2")
    (tmp_path / "project" / AM_FAMILIES_FILENAME).write_text("families: [\n", encoding="utf-8")

    assert "correct the file by hand" in _refusal(tmp_path, families={"sem_mag2": "sem-x"})


def test_a_file_edited_after_the_preview_is_not_written_over(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_mag2", "sem_labels", state={"sem_labels": "deselected"})
    change = plan_families_change(_project(tmp_path), families={"sem_mag2": "sem-mag2"})
    edited = "families:\n  sem_mag2: sem-other\n"
    (stage.parents[1] / AM_FAMILIES_FILENAME).write_text(edited, encoding="utf-8")
    before = (stage / "AM_state.yaml").read_text(encoding="utf-8")

    with pytest.raises(StaleFamilies):
        apply_families_change(change)

    assert (stage.parents[1] / AM_FAMILIES_FILENAME).read_text(encoding="utf-8") == edited
    assert (stage / "AM_state.yaml").read_text(encoding="utf-8") == before


def _headers(monkeypatch, headers: dict[str, ImageMetadata] | None) -> None:
    """Stand in for the header reader, answering by condition; None for no image stack."""
    import am_segmentation_manager.operations.families_edit as edit_module

    reader = None if headers is None else (lambda path: headers.get(path.parent.name))
    monkeypatch.setattr(edit_module, "header_reader", lambda: reader)


def test_joining_a_family_of_another_image_size_is_refused(tmp_path: Path, monkeypatch):
    _stage(tmp_path, "sem-2kV", "sem2kVb", "sem-2kV_labels")
    _headers(monkeypatch, {
        "sem-2kV": ImageMetadata(image_width=3072, image_height=2048),
        "sem2kVb": ImageMetadata(image_width=1536, image_height=1024),
    })

    assert "one label cannot fit both" in _refusal(tmp_path, families={"sem2kVb": "sem-2kV"})


def test_joining_a_family_at_another_pixel_size_is_warned_about(tmp_path: Path, monkeypatch):
    _stage(tmp_path, "sem-2kV", "sem2kVb", "sem-2kV_labels")
    _headers(monkeypatch, {
        "sem-2kV": ImageMetadata(pixel_size_nm=13.49),
        "sem2kVb": ImageMetadata(pixel_size_nm=26.98),
    })

    change = plan_families_change(_project(tmp_path), families={"sem2kVb": "sem-2kV"})

    assert change.warnings == (
        f"lam_000/adaptive_polish_1: {IMAGE} differs in pixel size: "
        "sem-2kV 13.49 nm, sem2kVb 26.98 nm",
    )
    assert any(line.startswith("  warning: ") for line in change.describe())


def test_a_join_says_when_headers_could_not_be_compared(tmp_path: Path, monkeypatch):
    _stage(tmp_path, "sem-2kV", "sem2kVb")
    _headers(monkeypatch, None)

    change = plan_families_change(_project(tmp_path), families={"sem2kVb": "sem-2kV"})

    assert change.warnings == (
        "image headers were not compared: the image stack is not installed",
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_the_command_previews_unless_told_to_apply(tmp_path: Path, capsys):
    project_root = _stage(tmp_path, "sem", "sem_mag2", "sem_labels").parents[1]
    command = ["edit-families", str(tmp_path), "--project", "project", "--set", "sem_mag2=sem-mag2"]

    assert main(command) == 0
    assert "sem_mag2: family as named -> sem-mag2" in capsys.readouterr().out
    assert not (project_root / AM_FAMILIES_FILENAME).exists()

    assert main([*command, "--apply"]) == 0
    assert read_families(project_root).family_of("sem_mag2") == "sem-mag2"


def test_the_command_refuses_and_writes_nothing(tmp_path: Path, capsys):
    project_root = _stage(tmp_path, "sem_fi8", "sem_labels").parents[1]

    code = main([
        "edit-families", str(tmp_path), "--project", "project",
        "--set", "sem_fi8=sem-x", "--set", "nosuch=sem-y", "--apply",
    ])

    assert code == 2
    assert "nosuch is not an image directory" in capsys.readouterr().out
    assert not (project_root / AM_FAMILIES_FILENAME).exists()


def test_the_command_clears_an_entry_and_a_primary(tmp_path: Path):
    project_root = _stage(tmp_path, "sem", "sem_mag2", "sem_labels").parents[1]
    base = ["edit-families", str(tmp_path), "--project", "project", "--apply"]
    main([*base, "--set", "sem_mag2=sem-mag2", "--primary", "sem=sem"])

    assert main([*base, "--clear", "sem_mag2", "--clear-primary", "sem"]) == 0

    corrections = read_families(project_root)
    assert (corrections.families, corrections.primary) == ({}, {})


def test_show_families_lists_each_family_and_its_conditions(tmp_path: Path, capsys):
    _stage(tmp_path, "sem", "sem_fi8", "sem-2kV")

    assert main(["show-families", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "SEM sem: sem (1), sem_fi8 (1)" in out
    assert "SEM sem-2kV: sem-2kV (1)" in out
