"""Tests for AM_metadata.yaml template generation.

The safety property that matters here is that nothing is ever guessed: a
plausible but wrong ``sample`` looks exactly like a correct one in the output,
so fields that vary between projects must stay visible placeholders until a
person fills them in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.io.yaml import read_yaml
from am_segmentation_manager.operations.init_project import (
    init_project,
    init_projects,
    read_defaults,
)
from am_segmentation_manager.scanner import Scanner


def _make_project(root: Path, name: str, method: str | None = "AutoLamella-OnGrid") -> Path:
    project = root / name
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / "img.tif").write_bytes(b"")
    body = f"name: {name}\n"
    if method is not None:
        body += f"method: {method}\n"
    (project / "experiment.yaml").write_text(body, encoding="utf-8")
    return project


# ---------------------------------------------------------------------------
# Template contents
# ---------------------------------------------------------------------------

def test_a_template_is_valid_yaml_with_every_field(tmp_path: Path):
    project = _make_project(tmp_path, "project")
    init_project(project)

    document = read_yaml(project / "AM_metadata.yaml")
    assert set(document["metadata"]) == {
        "description", "organisation", "project", "user", "sample", "grid_type",
        "lamella_type",
    }


def test_derivable_fields_are_filled_in(tmp_path: Path):
    project = _make_project(tmp_path, "project", method="AutoLiftout")
    result = init_project(project)

    document = read_yaml(project / "AM_metadata.yaml")
    assert document["metadata"]["lamella_type"] == "lift-out"
    assert result.derived == ("lamella_type",)


def test_underivable_fields_are_left_as_placeholders(tmp_path: Path):
    project = _make_project(tmp_path, "project")
    result = init_project(project)

    document = read_yaml(project / "AM_metadata.yaml")
    assert document["metadata"]["sample"] == "TODO"
    assert document["metadata"]["user"] == "TODO"
    assert "sample" in result.todo


def test_a_2026_project_without_method_leaves_lamella_type_to_fill_in(tmp_path: Path):
    project = _make_project(tmp_path, "project", method=None)
    result = init_project(project)

    document = read_yaml(project / "AM_metadata.yaml")
    assert document["metadata"]["lamella_type"] == "TODO"
    assert "lamella_type" in result.todo


def test_a_dataset_without_experiment_yaml_still_gets_a_template(tmp_path: Path):
    """This is what makes a pre-Autolamella dataset indexable at all."""
    project = tmp_path / "20240228_adaptive_milling_ai"
    (project / "lam_010" / "SEM_8").mkdir(parents=True)
    (project / "lam_010" / "SEM_8" / "img_0000.tif").write_bytes(b"")

    init_project(project)

    assert (project / "AM_metadata.yaml").exists()
    assert Scanner(tmp_path).build_index().all_projects()[0].path == project


def test_the_template_names_the_project_it_describes(tmp_path: Path):
    project = _make_project(tmp_path, "AutoLamella-01")
    init_project(project)

    assert "AutoLamella-01" in (project / "AM_metadata.yaml").read_text()


# ---------------------------------------------------------------------------
# Not overwriting
# ---------------------------------------------------------------------------

def test_an_existing_file_is_never_overwritten(tmp_path: Path):
    project = _make_project(tmp_path, "project")
    original = "metadata:\n  sample: HeLa\n"
    (project / "AM_metadata.yaml").write_text(original, encoding="utf-8")

    result = init_project(project)

    assert (project / "AM_metadata.yaml").read_text() == original
    assert result.skipped_existing is True
    assert result.written is False


def test_a_dry_run_writes_nothing(tmp_path: Path):
    project = _make_project(tmp_path, "project")

    result = init_project(project, dry_run=True)

    assert not (project / "AM_metadata.yaml").exists()
    assert result.written is False
    assert result.todo


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_defaults_are_copied_into_each_file(tmp_path: Path):
    """Copied at generation time, not resolved at scan time."""
    defaults_file = tmp_path / "shared.yaml"
    defaults_file.write_text("metadata:\n  organisation: RFI\n  user: Casper\n", encoding="utf-8")
    project = _make_project(tmp_path, "project")

    init_project(project, defaults=read_defaults(defaults_file))

    document = read_yaml(project / "AM_metadata.yaml")
    assert document["metadata"]["organisation"] == "RFI"
    assert document["metadata"]["user"] == "Casper"


def test_defaults_may_be_written_without_a_metadata_block(tmp_path: Path):
    defaults_file = tmp_path / "shared.yaml"
    defaults_file.write_text("organisation: RFI\n", encoding="utf-8")

    assert read_defaults(defaults_file) == {"organisation": "RFI"}


@pytest.mark.parametrize("field", ["sample", "description", "grid_type"])
def test_fields_that_vary_between_projects_cannot_be_defaulted(tmp_path: Path, field: str):
    """Projects in one directory may come from different samples, and a grid
    belongs to a specimen as much as the sample does -- one collection here
    holds a yeast project and a virus one side by side."""
    defaults_file = tmp_path / "shared.yaml"
    defaults_file.write_text(f"metadata:\n  {field}: HeLa\n", encoding="utf-8")

    with pytest.raises(ValueError, match="varies between projects"):
        read_defaults(defaults_file)


def test_an_unknown_default_field_is_rejected(tmp_path: Path):
    defaults_file = tmp_path / "shared.yaml"
    defaults_file.write_text("metadata:\n  smaple: typo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown field"):
        read_defaults(defaults_file)


def test_a_missing_defaults_file_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_defaults(tmp_path / "absent.yaml")


def test_a_derived_value_beats_a_default(tmp_path: Path):
    """A project's own file is better evidence than a shared assumption."""
    defaults_file = tmp_path / "shared.yaml"
    defaults_file.write_text("metadata:\n  lamella_type: waffle\n", encoding="utf-8")
    project = _make_project(tmp_path, "project", method="AutoLiftout")

    init_project(project, defaults=read_defaults(defaults_file))

    assert read_yaml(project / "AM_metadata.yaml")["metadata"]["lamella_type"] == "lift-out"


# ---------------------------------------------------------------------------
# Bulk generation and the CLI
# ---------------------------------------------------------------------------

def test_every_project_gets_its_own_file(tmp_path: Path):
    collection = tmp_path / "collection"
    collection.mkdir()
    projects = [_make_project(collection, f"project-{n}") for n in range(3)]

    results = init_projects(projects)

    assert all(result.written for result in results)
    assert all((project / "AM_metadata.yaml").exists() for project in projects)


def test_cli_recursive_writes_one_file_per_project(tmp_path: Path):
    collection = tmp_path / "collection"
    collection.mkdir()
    for n in range(3):
        _make_project(collection, f"project-{n}")

    assert main(["init-project", str(collection), "--recursive"]) == 0

    written = sorted(path.parent.name for path in collection.rglob("AM_metadata.yaml"))
    assert written == ["project-0", "project-1", "project-2"]


def test_cli_dry_run_reports_without_writing(tmp_path: Path):
    project = _make_project(tmp_path, "project")

    assert main(["init-project", str(project), "--dry-run"]) == 0
    assert not (project / "AM_metadata.yaml").exists()


def test_cli_reports_a_bad_defaults_file_instead_of_writing(tmp_path: Path):
    project = _make_project(tmp_path, "project")
    defaults_file = tmp_path / "shared.yaml"
    defaults_file.write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")

    assert main(["init-project", str(project), "--defaults", str(defaults_file)]) == 2
    assert not (project / "AM_metadata.yaml").exists()


def test_cli_recursive_on_a_directory_with_no_projects_fails(tmp_path: Path):
    assert main(["init-project", str(tmp_path), "--recursive"]) == 2


def test_generated_templates_are_reported_as_a_checklist(tmp_path: Path):
    """The remaining work is visible in the survey, not something to remember."""
    from am_segmentation_manager.survey import survey_collection

    project = _make_project(tmp_path, "project")
    init_project(project)

    survey = survey_collection(tmp_path)
    todo = [f for f in survey.all_findings if f.category == "metadata-todo"]

    assert len(todo) == 1
    assert "sample" in todo[0].message


def test_init_project_refuses_a_collection_directory(tmp_path: Path, capsys):
    """A marker here would make every dataset under it one project's lamellae,
    merging several samples under one metadata record."""
    from am_segmentation_manager.cli import main

    collection = tmp_path / "collection"
    project = collection / "AutoLamella-2026-01-01-00-00"
    (project / "01-lamella" / "stage" / "sem").mkdir(parents=True)
    (project / "01-lamella" / "stage" / "sem" / "img.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    legacy = collection / "20240101_legacy" / "lam_001" / "sem"
    legacy.mkdir(parents=True)
    (legacy / "img.tif").write_bytes(b"")

    assert main(["init-project", str(collection)]) == 2

    assert not (collection / "AM_metadata.yaml").exists()
    assert "collection rather than a project" in capsys.readouterr().out


def test_init_project_recursive_reaches_the_datasets_inside(tmp_path: Path):
    """What the refusal points at: each dataset gets its own file."""
    from am_segmentation_manager.cli import main

    collection = tmp_path / "collection"
    project = collection / "AutoLamella-2026-01-01-00-00"
    (project / "01-lamella" / "stage" / "sem").mkdir(parents=True)
    (project / "01-lamella" / "stage" / "sem" / "img.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    legacy = collection / "20240101_legacy" / "lam_001" / "sem"
    legacy.mkdir(parents=True)
    (legacy / "img.tif").write_bytes(b"")

    assert main(["init-project", str(collection), "--recursive"]) == 0

    assert not (collection / "AM_metadata.yaml").exists()
    assert (collection / "20240101_legacy" / "AM_metadata.yaml").exists()
    assert (project / "AM_metadata.yaml").exists()


def test_init_project_still_writes_into_a_plain_project(tmp_path: Path):
    """The refusal must not catch a directory that really is one project."""
    from am_segmentation_manager.cli import main

    dataset = tmp_path / "20240101_legacy"
    (dataset / "lam_001" / "sem").mkdir(parents=True)
    (dataset / "lam_001" / "sem" / "img.tif").write_bytes(b"")

    assert main(["init-project", str(dataset)]) == 0

    assert (dataset / "AM_metadata.yaml").exists()
