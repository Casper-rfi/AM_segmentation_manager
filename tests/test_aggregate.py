"""Tests for counting records, and for the metadata filters and CSV exports.

The aggregation layer is deliberately dull -- it groups and counts -- so most
of what is worth testing is the edges: values the data does not record, keys
that need a lookup because record.project is a name and names are not unique,
and filters that must not quietly admit images of unknown provenance.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager import aggregate as ag
from am_segmentation_manager.filtering import (
    METADATA_FILTERS,
    filter_by_metadata,
    filter_property,
    matches_metadata,
)
from am_segmentation_manager.image_metadata import ImageMetadata
from am_segmentation_manager.models import (
    AdaptiveMillingStage,
    ImageRecord,
    Lamella,
    Project,
)
from am_segmentation_manager.operations.metadata_csv import (
    export_image_metadata_csv,
    export_project_metadata_csv,
)
from am_segmentation_manager.reporting import format_counts, format_cross_tabulation


def record(name: str, modality: str = "SEM", state: str = "selected") -> ImageRecord:
    suffix = "" if state == "selected" else f"/{state}"
    directory = "sem" if modality == "SEM" else "fib"
    return ImageRecord(
        image_path=Path(f"stage/{directory}{suffix}/{name}.tif"), modality=modality
    )


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def test_counts_are_ordered_largest_first():
    records = [record("a"), record("b"), record("c", "FIB")]

    assert list(ag.count_by(records, ag.by_modality)) == ["SEM", "FIB"]


def test_an_unrecorded_value_counts_as_its_own_group():
    """A large "not recorded" row is a finding: it says the archive cannot
    answer the question being asked of it. Dropping those rows would hide it."""
    records = [record("a"), ImageRecord(image_path=Path("x.tif"))]

    counts = ag.count_by(records, ag.by_modality)

    assert counts == {"SEM": 1, ag.MISSING: 1}
    assert sum(counts.values()) == len(records)


def test_counting_an_empty_selection_is_not_an_error():
    assert ag.count_by([], ag.by_modality) == {}


def test_cross_tabulation_counts_pairs():
    records = [record("a", "SEM"), record("b", "SEM", "corrected"), record("c", "FIB")]
    metadata = {r.image_path: ImageMetadata(microscope_model="Arctis") for r in records}

    table = ag.cross_tabulate(records, ag.by_microscope_model(metadata), ag.by_modality)

    assert table == {("Arctis", "FIB"): 1, ("Arctis", "SEM"): 2}


def test_metadata_keys_report_missing_for_images_without_headers():
    records = [record("a"), record("b")]
    metadata = {records[0].image_path: ImageMetadata(microscope_model="Arctis")}

    assert ag.count_by(records, ag.by_microscope_model(metadata)) == {
        "Arctis": 1, ag.MISSING: 1,
    }


def test_one_beam_setting_is_one_voltage_group():
    """A header sometimes records a measured voltage rather than the set point,
    which left exact makes a row per image of a table about beam settings."""
    records = [record("a"), record("b"), record("c")]
    voltages = (30.0, 29.984666704217595, 29.99588191484701)
    metadata = {
        record_.image_path: ImageMetadata(acceleration_voltage_kv=voltage)
        for record_, voltage in zip(records, voltages)
    }

    assert ag.count_by(records, ag.by_voltage(metadata)) == {30.0: 3}


def test_a_voltage_no_header_recorded_is_still_reported_as_missing():
    records = [record("a")]

    assert ag.count_by(records, ag.by_voltage({})) == {ag.MISSING: 1}


# ---------------------------------------------------------------------------
# Keys that need the record's project
# ---------------------------------------------------------------------------

def make_project(name: str, key: str, sample: str | None, records: list) -> Project:
    from am_segmentation_manager.metadata import ExperimentMetadata

    return Project(
        name=name,
        path=Path(key),
        relative_path=key,
        experiment_metadata=ExperimentMetadata(sample=sample),
        lamellae=[
            Lamella(
                name="01-lam", path=Path(key) / "01-lam",
                stages=[AdaptiveMillingStage(name="s", path=Path(key), images=records)],
            )
        ],
    )


def test_project_properties_are_looked_up_by_path_not_by_name():
    """Two projects in the reference collection share a name and differ only by
    path, so grouping by name would merge them -- and merge their samples."""
    first, second = record("a"), record("b")
    object.__setattr__(first, "image_path", Path("one/sem/a.tif"))
    object.__setattr__(second, "image_path", Path("two/sem/b.tif"))
    projects = [
        make_project("same-name", "one", "HeLa", [first]),
        make_project("same-name", "two", "S. cerevisiae", [second]),
    ]
    lookup = ag.project_lookup(projects)

    counts = ag.count_by([first, second], ag.by_sample(lookup))

    assert counts == {"HeLa": 1, "S. cerevisiae": 1}


def test_a_project_without_metadata_reports_missing():
    only = record("a")
    lookup = ag.project_lookup([make_project("p", "p", None, [only])])

    assert ag.count_by([only], ag.by_sample(lookup)) == {ag.MISSING: 1}


# ---------------------------------------------------------------------------
# Resolving breakdowns by name
# ---------------------------------------------------------------------------

def test_an_unknown_breakdown_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown breakdown"):
        ag.resolve_breakdown("colour")


def test_every_declared_breakdown_resolves():
    """A name in the table that cannot be resolved would only fail at runtime,
    in the CLI, for whichever option nobody tried."""
    for name in ag.BREAKDOWNS:
        assert callable(ag.resolve_breakdown(name, {}, {}))


def test_only_the_header_breakdowns_are_marked_as_needing_metadata():
    """The CLI reads headers only when a chosen breakdown needs them, so this
    set decides whether a structural question stays a filesystem walk."""
    assert "microscope" in ag.NEEDS_METADATA
    assert "voltage" in ag.NEEDS_METADATA
    assert "modality" not in ag.NEEDS_METADATA
    assert "sample" not in ag.NEEDS_METADATA


# ---------------------------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------------------------

def test_numbers_match_across_int_and_float_spellings():
    entry = ImageMetadata(acceleration_voltage_kv=30.0)

    assert matches_metadata(entry, {"voltage": ["30"]})
    assert not matches_metadata(entry, {"voltage": ["2"]})


def test_a_filter_is_offered_with_the_unit_it_takes_values_in():
    """A filter takes the value in the unit the field is stored in, so a label
    reading "acceleration voltage kv" invites a value in volts."""
    assert filter_property("voltage") == "acceleration voltage (kV)"
    assert filter_property("microscope") == "microscope model"


def test_every_filter_can_be_described():
    for name in METADATA_FILTERS:
        assert filter_property(name)


def test_text_matches_case_insensitively():
    entry = ImageMetadata(microscope_model="Arctis")

    assert matches_metadata(entry, {"microscope": ["arctis"]})


def test_filters_are_and_across_properties_or_within_one():
    entry = ImageMetadata(microscope_model="Arctis", acceleration_voltage_kv=2.0)

    assert matches_metadata(entry, {"voltage": ["2", "2.5"]})
    assert not matches_metadata(entry, {"microscope": ["Arctis"], "voltage": ["30"]})


def test_an_image_with_no_header_matches_no_filter():
    """Including it would put images of unknown provenance into a set defined
    by provenance."""
    assert not matches_metadata(None, {"microscope": ["Arctis"]})
    assert matches_metadata(None, {})


def test_filtering_keeps_only_matching_records():
    keep, drop = record("a"), record("b")
    metadata = {
        keep.image_path: ImageMetadata(microscope_model="Arctis"),
        drop.image_path: ImageMetadata(microscope_model="Aquilos"),
    }

    assert filter_by_metadata([keep, drop], metadata, {"microscope": ["Arctis"]}) == [keep]


def test_no_filters_returns_everything_untouched():
    records = [record("a"), record("b")]

    assert filter_by_metadata(records, {}, None) == records


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def test_the_project_export_has_no_duplicate_columns(tmp_path: Path):
    """Regression: the identity column and a descriptive field are both called
    "project", so DictWriter let one silently win and the identity came out
    blank. The identity keeps the plain name; the descriptive field is the one
    qualified."""
    projects = [make_project("p", "p", "HeLa", [record("a")])]
    output = export_project_metadata_csv(projects, tmp_path, tmp_path / "p.csv")

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert len(header) == len(set(header))
    row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))
    assert row["project"] == "p"
    assert row["sample"] == "HeLa"
    assert "metadata_project" in header


def test_the_project_export_records_which_file_supplied_each_value(tmp_path: Path):
    from am_segmentation_manager.metadata import ExperimentMetadata

    project = make_project("p", "p", "HeLa", [record("a")])
    object.__setattr__(
        project, "experiment_metadata",
        ExperimentMetadata(sample="HeLa", sources={"sample": Path("AM_metadata.yaml")}),
    )
    output = export_project_metadata_csv([project], tmp_path, tmp_path / "p.csv")
    row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))

    assert row["sample_source"] == "AM_metadata.yaml"


def test_the_image_export_can_skip_headers(tmp_path: Path):
    """A purely structural listing should not pay to open every file."""
    only = record("a")
    output = export_image_metadata_csv(
        [only], tmp_path, tmp_path / "images.csv", include_headers=False
    )
    row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))

    assert row["modality"] == "SEM"
    assert row["acceleration_voltage_kv"] == ""


def test_the_image_export_reports_label_state_not_just_presence(tmp_path: Path):
    """An empty label is not an annotation, and an export that conflated them
    would overstate how much of the archive is finished."""
    pytest.importorskip("tifffile")
    import numpy as np
    import tifffile

    labels = tmp_path / "sem_labels"
    labels.mkdir()
    empty, annotated = labels / "a.tif", labels / "b.tif"
    tifffile.imwrite(str(empty), np.zeros((8, 8), dtype=np.uint8))
    filled = np.zeros((8, 8), dtype=np.uint8)
    filled[2:6, 2:6] = 2
    tifffile.imwrite(str(annotated), filled)

    records = [
        ImageRecord(image_path=tmp_path / "sem" / "a.tif", labels=[empty], modality="SEM"),
        ImageRecord(image_path=tmp_path / "sem" / "b.tif", labels=[annotated], modality="SEM"),
        ImageRecord(image_path=tmp_path / "sem" / "c.tif", modality="SEM"),
    ]
    output = export_image_metadata_csv(
        records, tmp_path, tmp_path / "images.csv", include_headers=False
    )
    states = [row["label_state"] for row in csv.DictReader(output.open(encoding="utf-8"))]

    assert states == ["empty", "annotated", "missing"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def test_counts_are_formatted_with_a_share_of_the_total():
    lines = format_counts({"Arctis": 3, "Aquilos": 1}, "Images per microscope")

    assert "4 image(s)" in lines[0]
    assert "75.0%" in lines[1]


def test_a_cross_tabulation_carries_row_and_column_totals():
    table = {("Arctis", "SEM"): 2, ("Arctis", "FIB"): 1, ("Aquilos", "SEM"): 1}
    lines = format_cross_tabulation(table, "microscope", "modality")
    text = "\n".join(lines)

    assert "= 3" in text      # the Arctis row
    assert "= 4" in text      # the grand total


def test_formatting_nothing_says_so_rather_than_printing_an_empty_table():
    assert "nothing to count" in format_counts({}, "Images")[0]
    assert "nothing to count" in format_cross_tabulation({}, "a", "b")[0]


# ---------------------------------------------------------------------------
# Through the CLI
# ---------------------------------------------------------------------------

def built(tmp_path: Path):
    from conftest import build_dataset

    return build_dataset(tmp_path, acquisitions=2)


SCOPE = ["--project", "all", "--lamella", "all", "--stage", "all", "--image", "all"]


def test_cli_stats_counts_by_one_property(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    assert main(["stats", str(tmp_path), *SCOPE, "--by", "modality"]) == 0
    output = capsys.readouterr().out

    assert "Images per modality" in output
    assert "SEM" in output and "FIB" in output


def test_cli_stats_cross_tabulates_two(tmp_path: Path, capsys):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    assert main(["stats", str(tmp_path), *SCOPE, "--by", "modality", "--by", "state"]) == 0

    assert "modality x state" in capsys.readouterr().out


def test_cli_stats_rejects_more_than_two_properties(tmp_path: Path):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    assert main([
        "stats", str(tmp_path), *SCOPE,
        "--by", "modality", "--by", "state", "--by", "variant",
    ]) == 2


def test_cli_stats_rejects_an_unknown_property(tmp_path: Path):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    assert main(["stats", str(tmp_path), *SCOPE, "--by", "colour"]) == 2


def test_cli_stats_writes_a_csv(tmp_path: Path):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    output = tmp_path / "counts.csv"
    assert main(["stats", str(tmp_path), *SCOPE, "--by", "modality", "-o", str(output)]) == 0

    rows = list(csv.reader(output.open(encoding="utf-8")))
    assert rows[0] == ["modality", "images"]
    assert len(rows) == 3


def test_cli_export_projects_csv(tmp_path: Path):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    output = tmp_path / "projects.csv"
    assert main(["export-projects-csv", str(tmp_path), "-o", str(output)]) == 0

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 1
    assert int(rows[0]["sem_images"]) == 2


def test_cli_export_metadata_csv_without_headers(tmp_path: Path):
    from am_segmentation_manager.cli import main

    built(tmp_path)
    output = tmp_path / "images.csv"
    assert main([
        "export-metadata-csv", str(tmp_path), *SCOPE, "--no-headers", "-o", str(output)
    ]) == 0

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 4
    assert {row["modality"] for row in rows} == {"SEM", "FIB"}


def test_cli_refuses_metadata_filters_without_headers(tmp_path: Path):
    """The filters need the headers they would be filtering on."""
    from am_segmentation_manager.cli import main

    built(tmp_path)
    assert main([
        "export-metadata-csv", str(tmp_path), *SCOPE,
        "--no-headers", "--microscope", "Arctis", "-o", str(tmp_path / "x.csv"),
    ]) == 2
