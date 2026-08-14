"""Tests for typed project metadata and its resolution rules.

The resolution rules matter more than they look: a wrong ``sample`` is
invisible in the output, so the tests here pin that nothing is inherited from
a parent directory, that placeholders are not mistaken for values, and that
``AM_metadata.yaml`` overrides per key rather than wholesale.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.metadata import (
    ExperimentMetadata,
    build_experiment_metadata,
    lamella_type_from_method,
)
from am_segmentation_manager.scanner import Scanner

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_am_metadata_overrides_experiment_per_key():
    """Overriding one field must not discard the others."""
    experiment = {"metadata": {"sample": "HeLa", "user": "Original", "organisation": "RFI"}}
    am_metadata = {"metadata": {"user": "Corrected"}}

    resolved = build_experiment_metadata(experiment, am_metadata)

    assert resolved.user == "Corrected"
    assert resolved.sample == "HeLa"
    assert resolved.organisation == "RFI"


def test_sources_record_which_file_supplied_each_value(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yaml"
    am_path = tmp_path / "AM_metadata.yaml"

    resolved = build_experiment_metadata(
        {"metadata": {"sample": "HeLa", "user": "Original"}},
        {"metadata": {"user": "Corrected"}},
        experiment_path=experiment_path,
        am_metadata_path=am_path,
    )

    assert resolved.sources["sample"] == experiment_path
    assert resolved.sources["user"] == am_path


@pytest.mark.parametrize(
    "method, expected",
    [
        ("AutoLamella-OnGrid", "cells-on-grid"),
        ("AutoLiftout", "lift-out"),
        ("autolamella-waffle", "waffle"),
        ("Something-New", None),
        (None, None),
        ("", None),
    ],
)
def test_lamella_type_is_derived_from_method(method, expected):
    assert lamella_type_from_method(method) == expected


def test_derived_lamella_type_is_used_when_not_stated():
    resolved = build_experiment_metadata({"method": "AutoLiftout"}, {})
    assert resolved.lamella_type == "lift-out"


def test_an_explicit_lamella_type_beats_the_derived_one():
    resolved = build_experiment_metadata(
        {"method": "AutoLiftout"}, {"metadata": {"lamella_type": "waffle"}}
    )
    assert resolved.lamella_type == "waffle"


def test_the_2026_schema_without_method_yields_no_lamella_type():
    """All four 2026 projects in the reference collection are in this case."""
    experiment = {"_id": "abc", "created_at": 1783589555.68676, "metadata": {}}
    resolved = build_experiment_metadata(experiment, {})

    assert resolved.lamella_type is None
    assert "lamella_type" in resolved.missing_fields()


def test_an_empty_experiment_metadata_block_is_harmless():
    """Autolamella 2026 writes `metadata: {}` -- the same key this package uses."""
    resolved = build_experiment_metadata({"metadata": {}}, {"metadata": {"sample": "HeLa"}})
    assert resolved.sample == "HeLa"


def test_todo_placeholders_are_not_values():
    resolved = build_experiment_metadata({}, {"metadata": {"sample": "TODO", "user": "Casper"}})

    assert resolved.sample is None
    assert resolved.todo_fields == ("sample",)
    assert resolved.user == "Casper"


def test_a_filled_in_field_clears_its_todo_state():
    resolved = build_experiment_metadata(
        {"metadata": {"sample": "TODO"}}, {"metadata": {"sample": "HeLa"}}
    )
    assert resolved.sample == "HeLa"
    assert resolved.todo_fields == ()


def test_blank_values_are_treated_as_absent():
    resolved = build_experiment_metadata({}, {"metadata": {"sample": "   ", "user": None}})
    assert resolved.sample is None
    assert resolved.user is None


def test_an_unrecognised_lamella_type_is_kept_but_warned(caplog):
    with caplog.at_level("WARNING"):
        resolved = build_experiment_metadata({}, {"metadata": {"lamella_type": "banana"}})

    assert resolved.lamella_type == "banana"
    assert "Unrecognised lamella_type" in caplog.text


def test_variant_descriptions_are_read_and_fall_back():
    resolved = build_experiment_metadata({}, {"variants": {"5kv": "5 kV landing energy"}})

    assert resolved.variant_description("5kv") == "5 kV landing energy"
    assert resolved.variant_description("lowmag") == "lowmag"
    assert resolved.variant_description(None) == "base"


def test_empty_metadata_reports_itself_as_empty():
    assert ExperimentMetadata().is_empty
    assert not ExperimentMetadata(sample="HeLa").is_empty


# ---------------------------------------------------------------------------
# Nothing is inherited from a parent directory
# ---------------------------------------------------------------------------

def _make_project(root: Path, name: str, am_metadata: str | None = None) -> Path:
    project = root / name
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / "img.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    if am_metadata is not None:
        (project / "AM_metadata.yaml").write_text(am_metadata, encoding="utf-8")
    return project


def test_metadata_is_not_inherited_from_a_collection_directory(tmp_path: Path):
    """Projects in one directory may be from different samples.

    A collection-level file must not supply values to the projects beneath it,
    because attributing one project's sample to another would be invisible.
    """
    collection = tmp_path / "20250414_adaptive_milling"
    collection.mkdir()
    _make_project(collection, "project-a", "metadata:\n  sample: HeLa\n")
    _make_project(collection, "project-b")
    (collection / "AM_metadata.yaml").write_text(
        "metadata:\n  sample: ShouldNotLeak\n", encoding="utf-8"
    )

    index = Scanner(tmp_path).build_index()
    by_name = {project.path.name: project for project in index.all_projects()}

    assert by_name["project-a"].experiment_metadata.sample == "HeLa"
    assert by_name["project-b"].experiment_metadata.sample is None


def test_a_marker_in_a_collection_directory_does_not_swallow_its_projects(tmp_path: Path):
    """A collection-level metadata file must not merge the projects beneath it.

    This is the mistake most likely to be made by hand, since writing one file
    per collection is the obvious shortcut. Left unhandled it absorbs every
    project's stages into a single pseudo-project, losing the boundaries between
    them entirely.
    """
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "AM_metadata.yaml").write_text("metadata:\n  sample: X\n", encoding="utf-8")
    _make_project(collection, "project-a")
    _make_project(collection, "project-b")

    index = Scanner(tmp_path).build_index()
    by_name = {project.path.name: project for project in index.all_projects()}

    assert set(by_name) == {"collection", "project-a", "project-b"}
    # Each real project keeps its own stage.
    assert len(by_name["project-a"].lamellae) == 1
    assert len(by_name["project-b"].lamellae) == 1
    # The collection itself has no images of its own.
    assert by_name["collection"].lamellae == []


def test_the_scanner_attaches_typed_metadata(tmp_path: Path):
    _make_project(tmp_path, "project", "metadata:\n  sample: HeLa\n  user: Casper\n")

    project = Scanner(tmp_path).build_index().all_projects()[0]

    assert project.experiment_metadata.sample == "HeLa"
    assert project.experiment_metadata.user == "Casper"
    assert project.metadata["name"] == "project"
