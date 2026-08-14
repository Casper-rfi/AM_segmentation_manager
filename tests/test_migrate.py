"""Tests for renaming pre-convention label directories.

This renames directories holding proofread work, so the properties that matter
are that it never overwrites, never merges two directories silently, and
reports the whole batch before touching anything.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.operations.migrate import (
    MigrationConflict,
    apply_label_migration,
    plan_label_migration,
)
from am_segmentation_manager.scanner import Scanner


def make_project(root: Path, label_dirs: tuple[str, ...] = ("labels", "labels_fib")) -> Path:
    """A stage with legacy label directory names."""
    project = root / "project"
    stage = project / "01-lamella" / "adaptive_polish"
    for directory in ("sem", "fib", *label_dirs):
        (stage / directory).mkdir(parents=True)
        (stage / directory / "img_000.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text("name: project\n", encoding="utf-8")
    return project


def _stage(project: Path) -> Path:
    return project / "01-lamella" / "adaptive_polish"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_legacy_directories_are_planned_for_rename(tmp_path: Path):
    make_project(tmp_path)

    renames = plan_label_migration(tmp_path)

    assert {(r.source.name, r.destination.name) for r in renames} == {
        ("labels", "sem_labels"),
        ("labels_fib", "fib_labels"),
    }


def test_the_plan_names_the_image_sets_each_directory_serves(tmp_path: Path):
    make_project(tmp_path)

    renames = {r.source.name: r for r in plan_label_migration(tmp_path)}

    assert renames["labels"].image_sets == ("sem",)
    assert renames["labels_fib"].image_sets == ("fib",)


def test_an_uppercase_legacy_name_is_recognised(tmp_path: Path):
    make_project(tmp_path, label_dirs=("labels_FIB",))

    renames = plan_label_migration(tmp_path)

    assert [r.destination.name for r in renames] == ["fib_labels"]


def test_current_convention_directories_are_left_alone(tmp_path: Path):
    make_project(tmp_path, label_dirs=("sem_labels", "fib_labels"))

    assert plan_label_migration(tmp_path) == []


def test_a_shared_legacy_directory_is_renamed_once(tmp_path: Path):
    """The legacy layout: several SEM conditions sharing one 'labels'."""
    project = tmp_path / "project"
    lamella = project / "lam_010"
    for directory in ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "labels"):
        (lamella / directory).mkdir(parents=True)
        (lamella / directory / "img_0004.tif").write_bytes(b"")
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: X\n", encoding="utf-8")

    renames = plan_label_migration(tmp_path)

    assert len(renames) == 1
    assert set(renames[0].image_sets) == {"SEM_8", "SEM_8_fi_1", "SEM_8_fi_4"}


# ---------------------------------------------------------------------------
# Refusing to lose data
# ---------------------------------------------------------------------------

def test_a_rename_onto_an_existing_directory_is_refused(tmp_path: Path):
    """Renaming would replace a directory that already holds labels."""
    make_project(tmp_path, label_dirs=("labels", "sem_labels"))

    with pytest.raises(MigrationConflict) as excinfo:
        plan_label_migration(tmp_path)

    assert "already exists" in str(excinfo.value)


def test_a_conflict_aborts_the_whole_batch(tmp_path: Path):
    """A partly migrated collection is harder to reason about than none."""
    make_project(tmp_path, label_dirs=("labels", "sem_labels", "labels_fib"))

    with pytest.raises(MigrationConflict):
        plan_label_migration(tmp_path)

    # The clean rename in the same batch must not have been performed either.
    assert (_stage(tmp_path / "project") / "labels_fib").exists()


def test_every_conflict_is_reported_not_just_the_first(tmp_path: Path):
    make_project(tmp_path, label_dirs=("labels", "sem_labels", "labels_fib", "fib_labels"))

    with pytest.raises(MigrationConflict) as excinfo:
        plan_label_migration(tmp_path)

    assert len(excinfo.value.conflicts) == 2


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def test_applying_renames_the_directories(tmp_path: Path):
    project = make_project(tmp_path)

    apply_label_migration(plan_label_migration(tmp_path))

    stage = _stage(project)
    assert (stage / "sem_labels" / "img_000.tif").exists()
    assert (stage / "fib_labels" / "img_000.tif").exists()
    assert not (stage / "labels").exists()
    assert not (stage / "labels_fib").exists()


def test_the_scanner_still_matches_labels_after_migration(tmp_path: Path):
    """The point of the exercise: nothing about indexing changes."""
    make_project(tmp_path)

    def labels_of(record):
        return [path.name for path in record.labels]

    before = {
        r.modality: labels_of(r)
        for p in Scanner(tmp_path).build_index().all_projects()
        for lam in p.lamellae for s in lam.stages for r in s.images
    }
    apply_label_migration(plan_label_migration(tmp_path))
    after = {
        r.modality: labels_of(r)
        for p in Scanner(tmp_path).build_index().all_projects()
        for lam in p.lamellae for s in lam.stages for r in s.images
    }

    assert before == after == {"SEM": ["img_000.tif"], "FIB": ["img_000.tif"]}


def test_migration_is_idempotent(tmp_path: Path):
    make_project(tmp_path)

    apply_label_migration(plan_label_migration(tmp_path))

    assert plan_label_migration(tmp_path) == []


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_previews_without_applying(tmp_path: Path, capsys):
    """Renaming user data on a bare command would be the wrong default."""
    project = make_project(tmp_path)

    assert main(["migrate-layout", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Would rename" in output
    assert "--apply" in output
    assert (_stage(project) / "labels").exists()


def test_cli_applies_when_asked(tmp_path: Path):
    project = make_project(tmp_path)

    assert main(["migrate-layout", str(tmp_path), "--apply"]) == 0

    assert (_stage(project) / "sem_labels").exists()
    assert not (_stage(project) / "labels").exists()


def test_cli_reports_a_conflict_and_changes_nothing(tmp_path: Path, capsys):
    project = make_project(tmp_path, label_dirs=("labels", "sem_labels"))

    assert main(["migrate-layout", str(tmp_path), "--apply"]) == 2

    output = capsys.readouterr().out
    assert "already exists" in output
    assert "Nothing was renamed" in output
    assert (_stage(project) / "labels").exists()


def test_cli_on_an_already_migrated_collection(tmp_path: Path, capsys):
    make_project(tmp_path, label_dirs=("sem_labels", "fib_labels"))

    assert main(["migrate-layout", str(tmp_path)]) == 0
    assert "No pre-convention label directories" in capsys.readouterr().out
