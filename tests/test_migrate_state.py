"""Tests for transcribing folder state into a stage's state file.

Migration is a one-shot with no visible symptom when it goes wrong: a group it
gets wrong reads as selected, which puts a deselected image into a training
set. So what is pinned here is mostly that it refuses -- and that it refuses a
whole stage rather than part of one.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from migrate_state import migrate_projects, migrate_stage, unnest_stage

from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import (
    AM_STATE_FILENAME,
    read_state,
    state_file_path,
)
from conftest import build_dataset


def move_state(path: Path, state: str) -> Path:
    """Move a file into the folder meaning *state*, as the old design did."""
    destination = state_path(path, state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def state_path(path: Path, state: str) -> Path:
    """Where *path* would sit in *state*, as a pre-declaration dataset had it."""
    return path.parent / state / path.name


def _duplicate_into(path: Path, state: str) -> Path:
    """Leave *path* where it is and put a second copy in a state folder."""
    destination = state_path(path, state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, destination)
    return destination


def only_stage(root: Path) -> Path:
    """Return the single stage directory under *root*, scanned fresh."""
    projects = Scanner(root).build_index().all_projects()
    stages = [
        stage
        for project in projects
        for lamella in project.lamellae
        for stage in lamella.stages
    ]
    assert len(stages) == 1, f"expected one stage, found {len(stages)}"
    return stages[0].path


def _records_of(root: Path) -> list:
    """Return every record the scanner indexes under *root*."""
    return [
        record
        for project in Scanner(root).build_index().all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]


def deselect_group(dataset, index: int = 0, modality: str = "SEM") -> None:
    """Put an image and the label it owns in ``deselected/``, as a pair.

    Building the pre-migration layout that ``migrate_state.py`` reads, which is the
    one place folders still mean anything.
    """
    name = dataset.filenames[modality][index]
    move_state(dataset.image_dir(modality) / name, "deselected")
    label = dataset.label_dir(modality) / name
    if label.exists():
        move_state(label, "deselected")


# ---------------------------------------------------------------------------
# What gets written
# ---------------------------------------------------------------------------

def test_a_stage_that_is_selected_throughout_needs_no_file(tmp_path: Path):
    """No entry already means selected, so a file would say nothing."""
    dataset = build_dataset(tmp_path)
    result = migrate_stage(only_stage(dataset.root))
    assert result.outcome == "nothing-to-declare"
    assert not state_file_path(dataset.stage).exists()


def test_a_deselected_group_is_declared(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    deselect_group(dataset)
    name = dataset.filenames["SEM"][0]

    result = migrate_stage(only_stage(dataset.root))

    assert result.outcome == "migrated"
    assert result.applied
    declared = dict(read_state(dataset.stage).states)
    assert list(declared.values()) == ["deselected"]
    assert list(declared)[0].endswith(name)


def test_a_corrected_group_is_declared(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    name = dataset.filenames["SEM"][0]
    move_state(dataset.image_dir("SEM") / name, "corrected")
    move_state(dataset.label_dir("SEM") / name, "corrected")

    migrate_stage(only_stage(dataset.root))

    assert list(read_state(dataset.stage).states.values()) == ["corrected"]


def test_the_key_names_the_label_not_the_image(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    deselect_group(dataset)

    migrate_stage(only_stage(dataset.root))

    key = next(iter(read_state(dataset.stage).states))
    labels_directory = dataset.label_dir("SEM").relative_to(dataset.stage)
    assert key.startswith(f"{labels_directory.as_posix()}/")


def test_conditions_sharing_a_label_get_one_entry(tmp_path: Path):
    """The case the design turns on: five images, one state."""
    dataset = build_dataset(
        tmp_path, layout="no_stage_level",
        sem_variants=("SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_8"),
        modalities=("SEM",),
    )
    name = dataset.filenames["SEM"][0]
    for variant in (None, "SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_8"):
        move_state(dataset.image_dir("SEM", variant) / name, "deselected")
    move_state(dataset.label_dir("SEM") / name, "deselected")

    migrate_stage(only_stage(dataset.root))

    declared = dict(read_state(dataset.stage).states)
    assert len(declared) == 1
    assert list(declared.values()) == ["deselected"]


def test_a_group_with_no_label_yet_is_keyed_by_where_its_label_would_go(tmp_path: Path):
    """Keying on the image would orphan the entry once a label appeared."""
    dataset = build_dataset(tmp_path, labels="none", modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    move_state(dataset.image_dir("SEM") / name, "deselected")

    migrate_stage(only_stage(dataset.root))

    key = next(iter(read_state(dataset.stage).states))
    assert key.endswith(name)
    assert not (dataset.stage / key).exists()


@pytest.mark.parametrize("layout", ["no_stage_level", "no_lamella_level"])
def test_a_collapsed_layout_puts_the_file_beside_the_image_directories(
    tmp_path: Path, layout: str
):
    """The stage level is synthesised, but the directory holding the images is real."""
    dataset = build_dataset(tmp_path, layout=layout, modalities=("SEM",))
    deselect_group(dataset)

    result = migrate_stage(only_stage(dataset.root))

    assert result.stage_directory == dataset.image_dir("SEM").parent
    assert (dataset.image_dir("SEM").parent / AM_STATE_FILENAME).exists()


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------

def test_images_disagreeing_with_their_label_block_the_stage(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    move_state(dataset.image_dir("SEM") / name, "deselected")  # label stays put

    result = migrate_stage(only_stage(dataset.root))

    assert result.outcome == "blocked"
    assert [problem.kind for problem in result.problems] == ["inconsistent"]
    assert "move them to match" in result.problems[0].detail
    assert not state_file_path(dataset.stage).exists()


def test_one_bad_group_blocks_the_whole_stage(tmp_path: Path):
    """The rule that keeps a migration from silently promoting an image.

    Writing the groups that were fine would leave the skipped one unlisted,
    and unlisted reads as selected.
    """
    dataset = build_dataset(tmp_path, acquisitions=3, modalities=("SEM",))
    deselect_group(dataset, index=0)                       # a clean deselection
    move_state(dataset.image_dir("SEM") / dataset.filenames["SEM"][1], "deselected")

    result = migrate_stage(only_stage(dataset.root))

    assert result.outcome == "blocked"
    assert not state_file_path(dataset.stage).exists()


def test_a_file_in_two_state_folders_blocks_the_stage(tmp_path: Path):
    """An ambiguous filesystem must not be transcribed into a confident file."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    _duplicate_into(dataset.image_dir("SEM") / name, "deselected")

    result = migrate_stage(only_stage(dataset.root))

    assert result.outcome == "blocked"
    assert [problem.kind for problem in result.problems] == ["duplicate"]
    assert not state_file_path(dataset.stage).exists()


def test_an_existing_file_that_disagrees_is_a_conflict(tmp_path: Path):
    """Only somebody's curation could have written it."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    state_file_path(dataset.stage).write_text(
        "version: 1\nstate:\n  labels/other.tif: corrected\n", encoding="utf-8"
    )
    before = state_file_path(dataset.stage).read_text(encoding="utf-8")

    result = migrate_stage(only_stage(dataset.root))

    assert result.outcome == "conflict"
    assert [problem.kind for problem in result.problems] == ["existing"]
    assert state_file_path(dataset.stage).read_text(encoding="utf-8") == before


def test_an_unreadable_file_is_a_conflict(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    state_file_path(dataset.stage).write_text("state:\n  - [\n", encoding="utf-8")

    result = migrate_stage(only_stage(dataset.root))

    assert result.outcome == "conflict"
    assert [problem.kind for problem in result.problems] == ["unreadable"]


# ---------------------------------------------------------------------------
# Running it twice, and not running it at all
# ---------------------------------------------------------------------------

def test_a_dry_run_writes_nothing_but_says_what_it_would(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)

    result = migrate_stage(only_stage(dataset.root), dry_run=True)

    assert result.outcome == "migrated"
    assert not result.applied
    assert len(result.declared) == 1
    assert not state_file_path(dataset.stage).exists()


def test_running_it_twice_changes_nothing(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)

    migrate_stage(only_stage(dataset.root))
    before = state_file_path(dataset.stage).read_text(encoding="utf-8")
    second = migrate_stage(only_stage(dataset.root))

    assert second.outcome == "unchanged"
    assert not second.applied
    assert state_file_path(dataset.stage).read_text(encoding="utf-8") == before


def test_a_run_is_undone_by_deleting_what_it_wrote(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    before = sorted(path.name for path in dataset.stage.iterdir())

    migrate_stage(only_stage(dataset.root))
    state_file_path(dataset.stage).unlink()

    assert sorted(path.name for path in dataset.stage.iterdir()) == before


def test_nothing_moves(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    before = sorted(
        str(path.relative_to(dataset.root))
        for path in dataset.root.rglob("*.tif")
    )

    migrate_stage(only_stage(dataset.root))

    after = sorted(
        str(path.relative_to(dataset.root))
        for path in dataset.root.rglob("*.tif")
    )
    assert after == before


# ---------------------------------------------------------------------------
# Across a collection
# ---------------------------------------------------------------------------

def test_every_stage_of_every_project_is_visited(tmp_path: Path):
    build_dataset(tmp_path, name="one", lamellae=2, modalities=("SEM",))
    build_dataset(tmp_path, name="two", lamellae=1, modalities=("SEM",))

    projects = Scanner(tmp_path).build_index().all_projects()
    results = migrate_projects(projects, dry_run=True)

    assert len(results) == 3
    assert {result.outcome for result in results} == {"nothing-to-declare"}


def test_a_blocked_stage_does_not_stop_the_others(tmp_path: Path):
    first = build_dataset(tmp_path, name="one", modalities=("SEM",))
    second = build_dataset(tmp_path, name="two", modalities=("SEM",))
    deselect_group(first)
    move_state(second.image_dir("SEM") / second.filenames["SEM"][0], "deselected")

    projects = Scanner(tmp_path).build_index().all_projects()
    results = migrate_projects(projects)

    assert {result.outcome for result in results} == {"migrated", "blocked"}
    assert state_file_path(first.stage).exists()
    assert not state_file_path(second.stage).exists()


# ---------------------------------------------------------------------------
# Through the script's own entry point
# ---------------------------------------------------------------------------

def test_the_script_migrates_and_reports(tmp_path: Path, capsys):
    from migrate_state import main

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)

    assert main([str(tmp_path)]) == 0

    assert state_file_path(dataset.stage).exists()
    assert "1 group(s) declared" in capsys.readouterr().out


def test_the_script_reports_a_blocked_stage_without_writing(tmp_path: Path, capsys):
    """A run that could not finish must not exit as if it had."""
    from migrate_state import main

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    move_state(dataset.image_dir("SEM") / dataset.filenames["SEM"][0], "deselected")

    assert main([str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "not migrated" in captured.out + captured.err
    assert not state_file_path(dataset.stage).exists()


def test_the_script_dry_run_writes_nothing(tmp_path: Path, capsys):
    from migrate_state import main

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)

    assert main([str(tmp_path), "--dry-run"]) == 0

    assert "Would migrate" in capsys.readouterr().out
    assert not state_file_path(dataset.stage).exists()


def test_the_script_needs_a_project(tmp_path: Path):
    from migrate_state import main

    assert main([str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# Taking the files back out of the state folders
# ---------------------------------------------------------------------------

def test_un_nesting_moves_files_back_beside_their_siblings(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    name = dataset.filenames["SEM"][0]
    migrate_stage(only_stage(dataset.root))

    result = unnest_stage(only_stage(dataset.root))

    assert result.outcome == "moved" and result.applied
    assert (dataset.image_dir("SEM") / name).exists()
    assert (dataset.label_dir("SEM") / name).exists()
    assert not (dataset.image_dir("SEM") / "deselected" / name).exists()


def test_the_declared_state_survives_un_nesting(tmp_path: Path):
    """The point of the whole exercise: the folder stops mattering."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    migrate_stage(only_stage(dataset.root))

    unnest_stage(only_stage(dataset.root))

    records = _records_of(dataset.root)
    deselected = [record for record in records if record.state != "selected"]
    assert len(deselected) == 1
    assert deselected[0].image_path.parent == dataset.image_dir("SEM")


def test_a_stage_whose_state_is_not_declared_is_refused(tmp_path: Path):
    """The folders are the only record there; moving would destroy it."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    name = dataset.filenames["SEM"][0]

    result = unnest_stage(only_stage(dataset.root))

    assert result.outcome == "blocked"
    assert [problem.kind for problem in result.problems] == ["unmigrated"]
    assert (dataset.image_dir("SEM") / "deselected" / name).exists()


def test_a_flat_stage_has_nothing_to_do(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))

    result = unnest_stage(only_stage(dataset.root))

    assert result.outcome == "nothing-nested"
    assert result.moves == ()


def test_an_occupied_destination_blocks_the_whole_stage(tmp_path: Path):
    """A file in two states at once; picking one would be a guess."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    _duplicate_into(dataset.image_dir("SEM") / name, "deselected")
    state_file_path(dataset.stage).write_text(
        "version: 1\nstate:\n", encoding="utf-8"
    )

    result = unnest_stage(only_stage(dataset.root))

    assert result.outcome == "blocked"
    assert [problem.kind for problem in result.problems] == ["collision"]
    assert (dataset.image_dir("SEM") / "deselected" / name).exists()


def test_a_dry_run_moves_nothing(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    name = dataset.filenames["SEM"][0]
    migrate_stage(only_stage(dataset.root))

    result = unnest_stage(only_stage(dataset.root), dry_run=True)

    assert result.outcome == "moved" and not result.applied
    assert len(result.moves) == 2
    assert (dataset.image_dir("SEM") / "deselected" / name).exists()


def test_empty_state_folders_are_removed_only_when_asked(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    migrate_stage(only_stage(dataset.root))

    kept = unnest_stage(only_stage(dataset.root))
    assert kept.removed == ()
    assert (dataset.image_dir("SEM") / "deselected").is_dir()

    deselect_group(dataset, index=1)
    migrate_stage(only_stage(dataset.root))
    result = unnest_stage(only_stage(dataset.root), remove_empty_directories=True)

    assert (dataset.image_dir("SEM") / "deselected") in result.removed
    assert not (dataset.image_dir("SEM") / "deselected").exists()


def test_a_state_folder_holding_something_else_is_left_alone(tmp_path: Path):
    """Only empty directories go; nothing here knows what else is in one."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    migrate_stage(only_stage(dataset.root))
    nested = dataset.image_dir("SEM") / "deselected"
    (nested / "notes.txt").write_text("mine", encoding="utf-8")

    unnest_stage(only_stage(dataset.root), remove_empty_directories=True)

    assert nested.is_dir()
    assert (nested / "notes.txt").exists()


def test_the_script_migrates_and_moves_in_one_run(tmp_path: Path, capsys):
    from migrate_state import main

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    name = dataset.filenames["SEM"][0]

    assert main([str(tmp_path), "--move-files"]) == 0

    output = capsys.readouterr().out
    assert "2 file(s) out of state folders" in output
    assert (dataset.image_dir("SEM") / name).exists()
    assert list(read_state(dataset.stage).states.values()) == ["deselected"]


def test_empty_folders_are_cleared_by_a_later_run(tmp_path: Path):
    """The files may already have been moved by an earlier run without the flag."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    migrate_stage(only_stage(dataset.root))
    unnest_stage(only_stage(dataset.root))
    assert (dataset.image_dir("SEM") / "deselected").is_dir()

    result = unnest_stage(only_stage(dataset.root), remove_empty_directories=True)

    assert result.outcome == "nothing-nested"
    assert (dataset.image_dir("SEM") / "deselected") in result.removed
    assert not (dataset.image_dir("SEM") / "deselected").exists()


def test_a_dry_run_says_which_folders_would_be_cleared(tmp_path: Path):
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    migrate_stage(only_stage(dataset.root))

    result = unnest_stage(
        only_stage(dataset.root), dry_run=True, remove_empty_directories=True
    )

    assert (dataset.image_dir("SEM") / "deselected") in result.removed
    assert (dataset.image_dir("SEM") / "deselected").is_dir(), "a dry run removes nothing"


def test_a_folder_holding_a_stray_file_is_named_not_cleared(tmp_path: Path, caplog):
    """On macOS this is usually a .DS_Store, which is the user's to delete."""
    dataset = build_dataset(tmp_path, modalities=("SEM",))
    deselect_group(dataset)
    migrate_stage(only_stage(dataset.root))
    nested = dataset.image_dir("SEM") / "deselected"
    (nested / ".DS_Store").write_bytes(b"finder")

    with caplog.at_level("WARNING"):
        result = unnest_stage(only_stage(dataset.root), remove_empty_directories=True)

    assert nested.is_dir()
    assert nested not in result.removed
    assert ".DS_Store" in caplog.text
