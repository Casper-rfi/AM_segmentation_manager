"""Tests for ``scripts/refresh_metadata_header.py``.

What it must not get wrong is a value: only comments may change, and a file it
does not recognise is left exactly as it is.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
import refresh_metadata_header
from refresh_metadata_header import (
    OLD_HEADER,
    OLD_VARIANTS_EXAMPLE,
    ValuesWouldChange,
    main,
    plan_refresh,
)

from am_segmentation_manager.io.yaml import read_yaml
from am_segmentation_manager.operations.init_project import header_lines, init_project

BODY = "metadata:\n  sample: HeLa\n  # Who acquired the data\n  user: TODO\n"


def old_file(project: Path, name: str = "20240128_adaptive_milling_ai") -> Path:
    """Write an ``AM_metadata.yaml`` as the earlier template wrote it."""
    project.mkdir(parents=True, exist_ok=True)
    path = project / "AM_metadata.yaml"
    header = "\n".join([f"# Metadata for {name}", "#", *OLD_HEADER]) + "\n"
    path.write_text(header + BODY + OLD_VARIANTS_EXAMPLE, encoding="utf-8")
    return path


def test_an_old_file_gets_the_current_header_and_loses_the_example(tmp_path: Path):
    path = old_file(tmp_path / "project")

    plan = plan_refresh(path)

    assert plan.header_replaced and plan.example_removed
    assert plan.new_text == "\n".join(header_lines("20240128_adaptive_milling_ai")) + "\n" + BODY


def test_the_project_name_is_kept_from_the_file_not_the_directory(tmp_path: Path):
    path = old_file(tmp_path / "renamed_since", name="original_name")

    assert plan_refresh(path).new_text.startswith("# Metadata for original_name\n")


def test_a_file_written_by_the_current_template_is_left_as_it_is(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    init_project(project)

    plan = plan_refresh(project / "AM_metadata.yaml")

    assert not plan.changes and not plan.problem


def test_a_header_edited_by_hand_is_left_alone(tmp_path: Path):
    path = old_file(tmp_path / "project")
    path.write_text(path.read_text().replace("delete the line", "remove the line"))

    plan = plan_refresh(path)

    assert "header" in plan.problem and not plan.changes


def test_an_edited_variants_example_is_left_alone(tmp_path: Path):
    path = old_file(tmp_path / "project")
    path.write_text(path.read_text().replace("5 kV landing energy", "5 kV"))

    assert "example" in plan_refresh(path).problem


def test_a_variants_section_with_values_is_left_alone(tmp_path: Path):
    path = old_file(tmp_path / "project")
    path.write_text(path.read_text() + "variants:\n  fi4: four frames\n")

    assert "variants section" in plan_refresh(path).problem


def test_a_rewrite_that_would_change_a_value_stops_the_run(tmp_path: Path, monkeypatch):
    path = old_file(tmp_path / "project")
    monkeypatch.setattr(refresh_metadata_header, "header_lines", lambda name: ["extra: 1"])

    with pytest.raises(ValuesWouldChange):
        plan_refresh(path)
    assert main([str(tmp_path), "--apply"]) == 2
    assert path.read_text().startswith("# Metadata for")


def test_a_preview_writes_nothing(tmp_path: Path):
    path = old_file(tmp_path / "project")
    before = path.read_text()

    assert main([str(tmp_path)]) == 0
    assert path.read_text() == before


def test_apply_rewrites_the_file_keeps_its_values_and_lists_it(tmp_path: Path):
    path = old_file(tmp_path / "project")
    values = read_yaml(path)
    manifest = tmp_path / "manifest.csv"

    assert main([str(tmp_path), "--apply", "--manifest", str(manifest)]) == 0

    assert read_yaml(path) == values
    assert "variants" not in path.read_text()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [Path(row["file"]) for row in rows] == [path]


def test_a_file_changed_after_the_plan_is_not_overwritten(tmp_path: Path):
    path = old_file(tmp_path / "project")
    plan = plan_refresh(path)
    path.write_text(path.read_text() + "# a note added meanwhile\n")

    assert refresh_metadata_header.apply_refresh([plan]) == 0
    assert path.read_text().endswith("# a note added meanwhile\n")
