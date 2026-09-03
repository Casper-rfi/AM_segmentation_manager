"""Tests for the fork point an export writes.

What a merge needs from a copy is the entry each project's log ended at when
the copy was taken. So what is pinned here is that the log always travels, that
the tip names a real entry, and that a fork point which cannot be trusted is
refused rather than half-read -- a merge comparing against the wrong ancestor
discards work while reporting success.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.history import (
    AM_HISTORY_FILENAME,
    Recorder,
    append_event,
    build_event,
    history_file_path,
    read_log,
)
from am_segmentation_manager.layout import ARCHIVE_DIRECTORY
from am_segmentation_manager.operations.copy_export import (
    PROJECT_FILES,
    copy_export,
    plan_copy,
)
from am_segmentation_manager.operations.export_record import (
    AM_EXPORT_FILENAME,
    FILE_VERSION,
    build_export_record,
    export_record_path,
    read_export_record,
    write_export_record,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

#: Every indexed operation asks which projects, lamellae, stages and images it
#: covers; an export of everything says so rather than defaulting.
SCOPE = ["--project", "all", "--lamella", "all", "--stage", "all", "--image", "all"]

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")
pytest.importorskip("cv2")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A collection of one project, with a history and a kept label version."""
    root = tmp_path / "collection"
    root.mkdir()
    dataset = build_dataset(root, lamellae=1, acquisitions=2, labels="annotated")
    history_file_path(dataset.project).touch()

    recorder = Recorder(by="casper", by_source="flag")
    append_event(
        dataset.project,
        build_event(
            "log-created", by="casper", by_source="flag",
            project_id="9f2c4b11", am_seg_version="0.0.0",
        ),
    )
    condition = next(iter(dataset.label_dirs))
    label = next(dataset.label_dirs[condition].glob("*.tif"))
    _save_over(label, recorder)
    return root


def scanned(root: Path):
    """Return the records and projects of a scan."""
    projects = Scanner(root).build_index().all_projects()
    return (
        [record for project in projects for record in project.image_records],
        projects,
    )


def exported(source: Path, destination: Path, **kwargs):
    """Copy *source* into *destination* and return the plan and projects."""
    records, projects = scanned(source)
    plan = plan_copy(records, destination, source, projects=projects, **kwargs)
    copy_export(plan)
    return plan, projects


# ---------------------------------------------------------------------------
# What travels
# ---------------------------------------------------------------------------

def test_the_history_always_travels(source: Path, tmp_path: Path):
    """Without it a merge back has no common ancestor to compare against."""
    assert AM_HISTORY_FILENAME in PROJECT_FILES

    destination = tmp_path / "out"
    exported(source, destination)

    assert (destination / "project" / AM_HISTORY_FILENAME).is_file()


def test_the_copied_history_is_the_one_that_was_there(source: Path, tmp_path: Path):
    destination = tmp_path / "out"
    exported(source, destination)

    assert [event.id for event in read_log(destination / "project").events] == [
        event.id for event in read_log(source / "project").events
    ]


def test_the_archive_stays_behind_unless_asked_for(source: Path, tmp_path: Path):
    """A subset export dragging every kept version along would balloon."""
    assert (source / "project" / ARCHIVE_DIRECTORY).is_dir()

    destination = tmp_path / "out"
    exported(source, destination)

    assert not (destination / "project" / ARCHIVE_DIRECTORY).exists()


def test_the_archive_travels_when_it_is_asked_for(source: Path, tmp_path: Path):
    destination = tmp_path / "out"
    exported(source, destination, with_archive=True)

    kept = sorted(
        path.name
        for path in (destination / "project" / ARCHIVE_DIRECTORY).rglob("*")
        if path.is_file()
    )
    assert kept == sorted(
        path.name
        for path in (source / "project" / ARCHIVE_DIRECTORY).rglob("*")
        if path.is_file()
    )


def test_the_copy_still_scans_as_a_collection(source: Path, tmp_path: Path):
    destination = tmp_path / "out"
    exported(source, destination, with_archive=True)

    before, _ = scanned(source)
    after, _ = scanned(destination)
    assert len(after) == len(before)


# ---------------------------------------------------------------------------
# The fork point
# ---------------------------------------------------------------------------

def test_the_tip_names_the_last_entry_the_project_held(source: Path, tmp_path: Path):
    _, projects = scanned(source)

    record = build_export_record(projects, source)

    fork = record.projects["project"]
    log = read_log(source / "project")
    assert fork.tip == log.events[-1].id
    assert fork.entries == len(log.events)
    assert fork.project_id == "9f2c4b11"


def test_a_project_keeping_no_history_says_so_rather_than_being_left_out(
    tmp_path: Path
):
    """A merge must tell a project it has no ancestor for from one not copied."""
    root = tmp_path / "collection"
    root.mkdir()
    build_dataset(root, labels="annotated")
    _, projects = scanned(root)

    record = build_export_record(projects, root)

    assert record.projects["project"].tip is None
    assert record.without_history == ("project",)


def test_a_fork_point_round_trips(source: Path, tmp_path: Path):
    _, projects = scanned(source)
    record = build_export_record(projects, source, scope="lam_000", with_archive=True)
    destination = tmp_path / "out"
    destination.mkdir()

    write_export_record(destination, record)
    read_back = read_export_record(destination)

    assert read_back.projects == record.projects
    assert read_back.exported_by == record.exported_by
    assert read_back.scope == "lam_000"
    assert read_back.with_archive is True


def test_a_copy_with_no_fork_point_reads_as_none(tmp_path: Path):
    assert read_export_record(tmp_path) is None


def test_a_fork_point_from_a_later_version_is_refused(tmp_path: Path):
    """Half-read, it would have a merge compare against the wrong ancestor."""
    export_record_path(tmp_path).write_text(
        f"version: {FILE_VERSION + 1}\nprojects: {{}}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="later version"):
        read_export_record(tmp_path)


def test_a_fork_point_that_is_not_a_mapping_is_refused(tmp_path: Path):
    export_record_path(tmp_path).write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        read_export_record(tmp_path)


def test_a_fork_point_with_no_version_is_refused(tmp_path: Path):
    export_record_path(tmp_path).write_text("projects: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="whole number"):
        read_export_record(tmp_path)


def test_a_project_entry_that_is_not_a_mapping_is_refused(tmp_path: Path):
    export_record_path(tmp_path).write_text(
        f"version: {FILE_VERSION}\nprojects:\n  one: not-a-mapping\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not hold a mapping"):
        read_export_record(tmp_path)


# ---------------------------------------------------------------------------
# Through the command
# ---------------------------------------------------------------------------

def test_the_command_writes_a_fork_point(source: Path, tmp_path: Path):
    from am_segmentation_manager.cli import main

    destination = tmp_path / "out"
    assert main(
        ["export-files", str(source), *SCOPE, "-o", str(destination)]
    ) == 0

    record = read_export_record(destination)
    assert record is not None
    assert record.projects["project"].tip
    assert (destination / AM_EXPORT_FILENAME).is_file()


def test_a_dry_run_writes_no_fork_point(source: Path, tmp_path: Path):
    from am_segmentation_manager.cli import main

    destination = tmp_path / "out"
    main(["export-files", str(source), *SCOPE, "-o", str(destination), "--dry-run"])

    assert not destination.exists() or read_export_record(destination) is None


def test_the_command_says_when_a_project_has_no_history(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    root = tmp_path / "collection"
    root.mkdir()
    build_dataset(root, labels="annotated")

    main(["export-files", str(root), *SCOPE, "-o", str(tmp_path / "out")])

    assert "no common ancestor" in capsys.readouterr().out


def test_the_fork_point_names_only_the_projects_the_copy_holds(
    source: Path, tmp_path: Path
):
    """One it does not hold would have a merge look for an ancestor in a tree
    that is not there."""
    from am_segmentation_manager.cli import main

    build_dataset(source, name="untouched", labels="annotated")
    destination = tmp_path / "out"

    main(
        ["export-files", str(source), "--project", "project",
         "--lamella", "all", "--stage", "all", "--image", "all",
         "-o", str(destination)]
    )

    assert set(read_export_record(destination).projects) == {"project"}


def test_the_fork_point_says_what_the_export_covered(source: Path, tmp_path: Path):
    from am_segmentation_manager.cli import main

    destination = tmp_path / "out"
    main(
        ["export-files", str(source), "--project", "project",
         "--lamella", "all", "--stage", "all", "--image", "all",
         "-o", str(destination)]
    )

    scope = read_export_record(destination).scope
    assert "--project project" in scope and "--lamella all" in scope


def test_the_command_records_who_exported(source: Path, tmp_path: Path):
    from am_segmentation_manager.cli import main

    destination = tmp_path / "out"
    main(["--user", "anna", "export-files", str(source), *SCOPE, "-o", str(destination)])

    assert read_export_record(destination).exported_by == "anna"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_over(label: Path, recorder: Recorder) -> None:
    """Replace *label* the way an editor would, so a version is kept."""
    import tifffile

    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.label_save import LabelToSave, save_labels

    edited = tifffile.imread(label)
    edited[0:2, 0:2] = 3
    save_labels(
        [
            LabelToSave(
                destination=label,
                label=edited,
                fingerprint=fingerprint_of(label),
            )
        ],
        recorder=recorder,
    )
