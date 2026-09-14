"""A project file that will not read as YAML is reported, and never stops a scan.

The project is indexed without that file's metadata, the survey reports it, and the
metadata exports refuse the project rather than writing its fields blank. The file
itself is never rewritten.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager import parallel
from am_segmentation_manager.aggregate import project_lookup
from am_segmentation_manager.cli import main
from am_segmentation_manager.io.yaml import UnreadableYAMLError
from am_segmentation_manager.metadata import AM_METADATA_FILENAME, EXPERIMENT_FILENAME
from am_segmentation_manager.operations.metadata_csv import (
    UnreadableMetadataError,
    export_image_metadata_csv,
    export_project_metadata_csv,
)
from am_segmentation_manager.operations.metadata_edit import update_project_metadata
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.survey import survey_collection
from conftest import build_dataset

#: The two ways a file can be there and still not read.
UNREADABLE = {
    "not-a-mapping": "typed by hand\n",
    "does-not-parse": "sample: [HeLa\n",
}

BOTH_FILES = [AM_METADATA_FILENAME, EXPERIMENT_FILENAME]


def _collection(root: Path, broken: str, content: str) -> Path:
    """Build a readable project ``good`` and a project ``bad`` whose *broken* file holds *content*."""
    build_dataset(root, name="good")
    bad = build_dataset(root, name="bad").project
    (bad / broken).write_text(content, encoding="utf-8")
    return bad / broken


def _projects(root: Path) -> dict:
    return {project.path.name: project for project in Scanner(root).build_index().all_projects()}


@pytest.mark.parametrize("broken", BOTH_FILES)
@pytest.mark.parametrize("content", UNREADABLE.values(), ids=UNREADABLE.keys())
def test_an_unreadable_project_file_leaves_the_rest_of_the_scan_alone(tmp_path, broken, content):
    path = _collection(tmp_path, broken, content)
    before = path.read_bytes()

    projects = _projects(tmp_path)

    assert projects["good"].metadata_problems == ()
    assert projects["good"].image_records
    assert len(projects["bad"].metadata_problems) == 1
    assert broken in projects["bad"].metadata_problems[0]
    assert projects["bad"].image_records, "its images are still indexed"
    assert path.read_bytes() == before


def test_an_unreadable_experiment_comes_back_from_a_worker_process(tmp_path, monkeypatch):
    """Experiment files are parsed in the pool, where one that raised would fail the batch."""
    _collection(tmp_path, EXPERIMENT_FILENAME, UNREADABLE["not-a-mapping"])

    parallel.shutdown_pool()
    monkeypatch.setattr(parallel, "_pool_broken", False)
    monkeypatch.setattr(parallel, "POOL_WORKERS", 2)
    monkeypatch.setattr("am_segmentation_manager.scanner.EXPERIMENT_POOL_FILES", 0)
    try:
        projects = _projects(tmp_path)
        assert parallel._pool is not None
    finally:
        parallel.shutdown_pool()

    assert projects["good"].metadata_problems == ()
    assert len(projects["bad"].metadata_problems) == 1


@pytest.mark.parametrize("broken", BOTH_FILES)
def test_the_survey_reports_an_unreadable_project_file_as_an_error(tmp_path, broken):
    _collection(tmp_path, broken, UNREADABLE["does-not-parse"])

    survey = survey_collection(tmp_path, check_labels=False)
    findings = {project.path.name: project.findings for project in survey.projects}

    unreadable = [f for f in findings["bad"] if f.category == "unreadable-yaml"]
    assert [(f.path.name, f.severity) for f in unreadable] == [(broken, "error")]
    assert not [f for f in findings["bad"] if f.category.startswith("metadata-")]
    assert not [f for f in findings["good"] if f.category == "unreadable-yaml"]


def test_an_unreadable_am_metadata_is_never_edited(tmp_path):
    path = _collection(tmp_path, AM_METADATA_FILENAME, UNREADABLE["not-a-mapping"])
    before = path.read_bytes()

    with pytest.raises(UnreadableYAMLError):
        update_project_metadata(path.parent, {"sample": "HeLa"})

    assert path.read_bytes() == before


def test_the_metadata_exports_refuse_a_project_with_an_unreadable_file(tmp_path):
    _collection(tmp_path, AM_METADATA_FILENAME, UNREADABLE["not-a-mapping"])
    projects = list(_projects(tmp_path).values())
    records = [record for project in projects for record in project.image_records]
    output = tmp_path / "exports"
    output.mkdir()

    with pytest.raises(UnreadableMetadataError, match=AM_METADATA_FILENAME):
        export_project_metadata_csv(projects, tmp_path, output / "projects.csv")
    with pytest.raises(UnreadableMetadataError, match=AM_METADATA_FILENAME):
        export_image_metadata_csv(
            records, tmp_path, output / "images.csv",
            include_headers=False, projects=project_lookup(projects),
        )

    assert not list(output.iterdir())


def test_an_image_export_of_readable_projects_alone_goes_ahead(tmp_path):
    _collection(tmp_path, AM_METADATA_FILENAME, UNREADABLE["not-a-mapping"])
    projects = _projects(tmp_path)

    output = export_image_metadata_csv(
        projects["good"].image_records, tmp_path, tmp_path / "images.csv",
        include_headers=False, projects=project_lookup(projects.values()),
    )

    assert output.exists()


def test_the_command_line_export_refuses_with_exit_status_two(tmp_path):
    _collection(tmp_path, AM_METADATA_FILENAME, UNREADABLE["not-a-mapping"])
    output = tmp_path / "projects.csv"

    assert main(["export-projects-csv", str(tmp_path), "-o", str(output)]) == 2
    assert not output.exists()
