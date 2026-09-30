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
    FIELDS,
    ExperimentMetadata,
    build_experiment_metadata,
    format_field_value,
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

    assert resolved.user == ("Corrected",)
    assert resolved.sample == ("HeLa",)
    assert resolved.organisation == ("RFI",)


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
    assert resolved.sample == ("HeLa",)


def test_todo_placeholders_are_not_values():
    resolved = build_experiment_metadata({}, {"metadata": {"sample": "TODO", "user": "Casper"}})

    assert resolved.sample == ()
    assert resolved.todo_fields == ("sample",)
    assert resolved.user == ("Casper",)


def test_a_filled_in_field_clears_its_todo_state():
    resolved = build_experiment_metadata(
        {"metadata": {"sample": "TODO"}}, {"metadata": {"sample": "HeLa"}}
    )
    assert resolved.sample == ("HeLa",)
    assert resolved.todo_fields == ()


def test_blank_values_are_treated_as_absent():
    resolved = build_experiment_metadata({}, {"metadata": {"sample": "   ", "user": None}})
    assert resolved.sample == ()
    assert resolved.user == ()


def test_an_unrecognised_lamella_type_is_kept_but_warned(caplog):
    with caplog.at_level("WARNING"):
        resolved = build_experiment_metadata({}, {"metadata": {"lamella_type": "banana"}})

    assert resolved.lamella_type == "banana"
    assert "Unrecognised lamella_type" in caplog.text


def test_a_leftover_variants_section_is_ignored():
    """Files written before the section was dropped may still hold one."""
    resolved = build_experiment_metadata(
        {}, {"metadata": {"sample": "HeLa"}, "variants": {"5kv": "5 kV landing energy"}}
    )

    assert resolved.sample == ("HeLa",)


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

    assert by_name["project-a"].experiment_metadata.sample == ("HeLa",)
    assert by_name["project-b"].experiment_metadata.sample == ()


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

    assert project.experiment_metadata.sample == ("HeLa",)
    assert project.experiment_metadata.user == ("Casper",)
    assert project.metadata["name"] == "project"


# ---------------------------------------------------------------------------
# The field table
#
# Adding a descriptive field is a row in metadata.FIELDS, and nothing outside
# it may list the fields or single one out. These pin the reach of that table:
# each is a place that used to carry its own copy of the answer.
# ---------------------------------------------------------------------------

def test_every_field_is_an_attribute_of_the_resolved_metadata():
    """A row with no attribute behind it reads as permanently unset, and
    nothing says so -- the field simply never holds anything."""
    for entry in FIELDS:
        assert hasattr(ExperimentMetadata(), entry.name), entry.name


def test_the_fields_run_in_the_order_the_add_data_tab_shows_them():
    """A section's fields are consecutive rows, so one order serves the tab,
    a template and the per-project CSV alike."""
    from am_segmentation_manager.metadata import (
        GENERAL,
        LAMELLA_PREPARATION,
        SAMPLE_PREPARATION,
    )

    sections = [GENERAL, SAMPLE_PREPARATION, LAMELLA_PREPARATION]

    assert [entry.name for entry in FIELDS] == [
        "description", "organisation", "project", "user",
        "sample", "sample_state", "sample_medium", "grid_type", "grid_id",
        "freezing_method", "gis_deposition_time_s", "lamella_type",
    ]
    order = [sections.index(entry.section) for entry in FIELDS]
    assert order == sorted(order)


def test_the_table_is_where_the_fields_and_their_guidance_come_from():
    from am_segmentation_manager.metadata import DESCRIPTIVE_FIELDS, FIELD_COMMENTS

    assert DESCRIPTIVE_FIELDS == tuple(entry.name for entry in FIELDS)
    assert FIELD_COMMENTS == {entry.name: entry.comment for entry in FIELDS}
    assert all(entry.comment for entry in FIELDS), "a field with no guidance"


def test_the_table_decides_what_a_defaults_file_may_pre_fill():
    from am_segmentation_manager.operations.init_project import NEVER_DEFAULTED

    assert NEVER_DEFAULTED == {
        entry.name for entry in FIELDS if not entry.defaultable
    }


def test_the_table_decides_which_fields_are_offered_the_values_in_use():
    from am_segmentation_manager.gui.metadata_controller import SUGGESTED_FIELDS

    assert SUGGESTED_FIELDS == tuple(
        entry.name for entry in FIELDS if entry.suggested
    )


def test_a_field_that_can_be_counted_by_gets_a_breakdown_of_its_own():
    """One row in the table, one count, one --by selector and one Export filter
    row. A field that had to be added to aggregate.py separately is a field
    somebody forgets to add."""
    from am_segmentation_manager.aggregate import BREAKDOWNS

    for entry in FIELDS:
        if entry.breakdown_label is None:
            assert entry.breakdown_name not in BREAKDOWNS
            continue
        breakdown = BREAKDOWNS[entry.breakdown_name]
        assert breakdown.kind == "project"
        assert breakdown.label == entry.breakdown_label


def test_a_vocabulary_is_rejected_by_name_rather_than_by_a_named_field():
    """The check used to name lamella_type in four places. A second field with
    a vocabulary has to be rejected the same way without editing any of them."""
    from am_segmentation_manager.operations.metadata_edit import (
        MetadataValidationError,
        _validate,
    )

    for entry in FIELDS:
        if not entry.choices:
            continue
        _validate({entry.name: entry.choices[0]})
        with pytest.raises(MetadataValidationError, match=entry.name):
            _validate({entry.name: "not-a-real-value"})


# ---------------------------------------------------------------------------
# Numeric fields
# ---------------------------------------------------------------------------

def test_a_numeric_field_resolves_to_a_number():
    """Not to the text of one: it is deposited in a CSV beside other numbers,
    and a column that cannot be summed is worth less than the value in it."""
    resolved = build_experiment_metadata(
        {}, {"metadata": {"gis_deposition_time_s": 30}}
    )

    assert resolved.gis_deposition_time_s == 30.0


def test_a_numeric_field_that_is_not_a_number_is_dropped_and_warned(caplog):
    """Kept as text it would not fit the attribute; kept as zero it would be a
    deposition time nobody performed."""
    with caplog.at_level("WARNING"):
        resolved = build_experiment_metadata(
            {}, {"metadata": {"gis_deposition_time_s": "3o"}}
        )

    assert resolved.gis_deposition_time_s is None
    assert "not a number" in caplog.text


def test_zero_is_a_recorded_value_and_not_an_empty_project():
    """No GIS deposition is a thing that happened, and it is falsy."""
    resolved = build_experiment_metadata(
        {}, {"metadata": {"gis_deposition_time_s": 0}}
    )

    assert resolved.is_empty is False
    assert "gis_deposition_time_s" not in resolved.missing_fields()


def test_a_number_reads_back_as_it_was_typed():
    """30 shown as "30.0" would register as an edit on every project merely
    looked at, and would head a breakdown column as a number nobody typed."""
    assert format_field_value(30.0) == "30"
    assert format_field_value(2.5) == "2.5"
    assert format_field_value(None) == ""
