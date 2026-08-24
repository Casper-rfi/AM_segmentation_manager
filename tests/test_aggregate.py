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
from am_segmentation_manager.config import load_segmentation_config
from am_segmentation_manager.filtering import (
    METADATA_FILTERS,
    filter_by_metadata,
    filter_property,
    matches_metadata,
)
from am_segmentation_manager.image_metadata import ImageMetadata
from am_segmentation_manager.models import (
    ImageRecord,
    Lamella,
    Project,
    Stage,
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


def test_one_magnification_is_one_pixel_size_group():
    """A pixel size is derived from the magnification, so one field of view
    comes back as 13.020833333333336 and another as ...32. Left exact, each is
    its own row of a table about which magnifications were used."""
    first, second = record("a"), record("b")
    metadata = {
        first.image_path: ImageMetadata(pixel_size_nm=13.020833333333336),
        second.image_path: ImageMetadata(pixel_size_nm=13.020833333333332),
    }

    counts = ag.count_by([first, second], ag.by_pixel_size(metadata))

    assert counts == {13.02: 2}


def test_pixel_sizes_a_decade_apart_stay_apart():
    """Significant figures rather than decimal places: rounding a 0.5 nm pixel
    and a 500 nm one to the same decimal place loses one of them."""
    small, large = record("a"), record("b")
    metadata = {
        small.image_path: ImageMetadata(pixel_size_nm=0.5234567),
        large.image_path: ImageMetadata(pixel_size_nm=523.4567),
    }

    counts = ag.count_by([small, large], ag.by_pixel_size(metadata))

    assert set(counts) == {0.5235, 523.5}


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
                stages=[Stage(name="s", path=Path(key), images=records)],
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


def test_lamellae_of_two_projects_are_not_counted_as_one():
    """Almost every project has an "01-", so an unqualified lamella row merges
    lamellae that have nothing to do with each other."""
    first, second = record("a"), record("b")
    object.__setattr__(first, "image_path", Path("one/sem/a.tif"))
    object.__setattr__(second, "image_path", Path("two/sem/b.tif"))
    first.lamella = second.lamella = "01-lam"
    lookup = ag.project_lookup([
        make_project("same-name", "one", "HeLa", [first]),
        make_project("same-name", "two", "HeLa", [second]),
    ])

    counts = ag.count_by([first, second], ag.by_lamella(lookup))

    assert counts == {"one / 01-lam": 1, "two / 01-lam": 1}


def test_a_stage_is_named_by_the_lamella_it_belongs_to():
    """Stage names are either shared by every lamella (adaptive_polish) or a
    bare timestamp. Neither says where to go and look."""
    only = record("a")
    object.__setattr__(only, "image_path", Path("one/sem/a.tif"))
    only.lamella, only.stage = "01-lam", "adaptive_polish"
    lookup = ag.project_lookup([make_project("p", "one", None, [only])])

    counts = ag.count_by([only], ag.by_stage(lookup))

    assert counts == {"one / 01-lam / adaptive_polish": 1}


def test_a_record_outside_the_lookup_still_lands_somewhere_named():
    """A record the lookup does not hold falls back to the name it carries,
    rather than collapsing into "(not recorded)" with every other orphan."""
    only = record("a")
    only.project, only.lamella = "loose-project", "01-lam"

    counts = ag.count_by([only], ag.by_lamella({}))

    assert counts == {"loose-project / 01-lam": 1}


def test_lamella_and_stage_are_listed_as_workflow_not_project_metadata():
    """They need the project lookup to be named, but a lamella is not something
    read out of a project's metadata file, and the headings say where a value
    comes from."""
    grouped = dict(ag.grouped_breakdowns())

    workflow = [name for name, _ in grouped[ag.GROUP_TITLES["record"]]]

    assert "lamella" in workflow
    assert "stage" in workflow


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


def test_every_breakdown_is_named_for_a_reader():
    """A bare key is what the dropdown used to show: "dwell" does not say
    dwell of what, and "current" reads as the current selection."""
    for entry in ag.BREAKDOWNS.values():
        assert entry.label
        assert entry.kind in ag.GROUP_TITLES

    assert ag.BREAKDOWNS["dwell"].label == "dwell time (ns)"
    assert ag.BREAKDOWNS["current"].label == "beam current (pA)"
    assert ag.BREAKDOWNS["serial"].label == "microscope serial number"


def test_a_breakdown_carrying_a_unit_says_which():
    """A count of "30" is unreadable without knowing it is kV."""
    for name in ("voltage", "current", "dwell", "pixel-size"):
        assert "(" in ag.BREAKDOWNS[name].label


def test_the_groups_offer_every_breakdown_exactly_once():
    grouped = ag.grouped_breakdowns()
    offered = [name for _, rows in grouped for name, _ in rows]

    assert sorted(offered) == sorted(ag.BREAKDOWNS)
    assert [title for title, _ in grouped] == list(ag.GROUP_TITLES.values())


def test_rows_within_a_group_are_ordered_by_what_is_read():
    """The label is what a reader scans down, not the identifier behind it."""
    for _, rows in ag.grouped_breakdowns():
        assert [label for _, label in rows] == sorted(label for _, label in rows)


# ---------------------------------------------------------------------------
# Class areas, binned
# ---------------------------------------------------------------------------

DECADES = ag.area_bin_edges()
DECADE_BINS = ag.area_bins(DECADES)
DECADE_ORDER = ag.area_order(DECADES)


def binned(area: float, edges=DECADES) -> str:
    return ag._area_bin(area, edges, ag.area_bins(edges))


def test_an_area_falls_in_the_bin_that_contains_it():
    assert binned(0.005) == "< 0.01"
    assert binned(0.5) == "0.1 - 1"
    assert binned(132.0) == "100 - 1000"
    assert binned(5000.0) == ">= 1000"


def test_a_bin_edge_belongs_to_the_bin_above_it():
    """Otherwise an area exactly on an edge falls in both or neither."""
    assert binned(0.1) == "0.1 - 1"
    assert binned(1.0) == "1 - 10"


def test_an_edge_holds_at_every_resolution():
    """Regression: 3.0 * 10**-1 is 0.30000000000000004, and an edge a hair
    above its round value puts an area of exactly 0.3 in the bin below."""
    for per_decade in ag.AREA_BINS_PER_DECADE:
        edges = ag.area_bin_edges(per_decade)
        for edge in edges[:-1]:
            assert binned(edge, edges).startswith(f"{edge:g} -")


def test_a_class_that_is_not_in_the_label_is_absent_not_zero():
    """"No GIS was deposited here" is a finding; a zero area reads as one."""
    assert binned(0.0) == ag.ABSENT


def test_a_label_that_was_never_measured_keeps_its_own_status():
    """No area is not the same as a small one, so it groups under why."""
    key = ag.by_class_area(2, DECADES)({})

    assert key(record("a")) == "no-label"


def test_the_bins_run_smallest_first_with_the_failures_last():
    order = DECADE_ORDER

    assert order[0] == ag.ABSENT
    assert list(order[1 : 1 + len(DECADE_BINS)]) == list(DECADE_BINS)
    assert order[-3:] == ("empty", "no-pixel-size", "no-label")


# -- how finely the scale is cut --------------------------------------------

def test_the_default_bins_are_decades():
    """The setting a caller that says nothing gets, and what the CLI keeps."""
    assert DECADES == (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def test_raising_the_resolution_splits_a_bin_rather_than_moving_its_edges():
    """A lamella is hundreds of um^2, so at one bin per decade nearly all of
    them land in "100 - 1000" and the breakdown says nothing about their
    spread. Every decade edge has to survive being cut finer, or a count taken
    at one setting means nothing at another."""
    finer = ag.area_bin_edges(3)

    assert set(DECADES) < set(finer)
    assert binned(132.0) == "100 - 1000"
    assert binned(132.0, finer) == "100 - 200"


def test_every_supported_resolution_keeps_its_edges_round():
    for per_decade in ag.AREA_BINS_PER_DECADE:
        edges = ag.area_bin_edges(per_decade)

        assert list(edges) == sorted(edges)
        assert len(set(edges)) == len(edges)
        # Round numbers, not floating-point residue like 0.30000000000000004.
        assert all(float(f"{edge:.2g}") == edge for edge in edges)


def test_an_unsupported_resolution_is_rejected():
    with pytest.raises(ValueError, match="Bins per decade"):
        ag.area_bin_edges(7)


def test_a_resolution_says_what_it_cuts_a_decade_at():
    """The control has to offer a scale rather than a number: "3" is something
    to look up, "1, 2, 5" is something a reader recognises."""
    assert ag.bin_mantissas(1) == (1.0,)
    assert ag.bin_mantissas(3) == (1.0, 2.0, 5.0)

    with pytest.raises(ValueError, match="Bins per decade"):
        ag.bin_mantissas(4)


def test_the_offered_cuts_are_the_ones_the_edges_are_built_from():
    """Two statements of the same fact would let the control describe a scale
    the breakdown does not use."""
    for per_decade in ag.AREA_BINS_PER_DECADE:
        edges = ag.area_bin_edges(per_decade)
        mantissas = ag.bin_mantissas(per_decade)

        assert len(edges) % len(mantissas) == 0
        assert [round(edge, 10) for edge in edges[:len(mantissas)]] == [
            round(value * 0.01, 10) for value in mantissas
        ]


def test_the_area_breakdowns_are_built_at_the_chosen_resolution():
    config = load_segmentation_config()

    table = ag.breakdowns_for(config, area_bins_per_decade=3)

    assert "100 - 200" in table["area-lamella"].order
    assert "100 - 1000" not in table["area-lamella"].order


def test_every_measured_class_gets_an_area_breakdown():
    """Which classes exist is the segmentation config's to say, so the table
    cannot be fixed in code without hardcoding one project's classes."""
    config = load_segmentation_config()
    table = ag.breakdowns_for(config)

    assert set(table) - set(ag.BREAKDOWNS) == {
        "area-gis", "area-lamella", "area-crack-hole",
    }
    assert table["area-lamella"].label == "lamella area (um^2)"
    assert table["area-lamella"].kind == "measurement"


def test_a_class_name_becomes_a_command_line_safe_breakdown_name():
    assert ag._slug("crack/hole") == "crack-hole"
    assert ag._slug("GIS") == "gis"


# ---------------------------------------------------------------------------
# Declared ordering
# ---------------------------------------------------------------------------

def test_counts_keep_a_declared_order_rather_than_the_largest_first():
    """Bins of increasing area shuffled by population stop reading as a
    distribution."""
    records = [record(str(index)) for index in range(5)]
    sizes = iter([1.0, 1.0, 1.0, 0.005, 0.5])

    counts = ag.count_by(records, lambda _: binned(next(sizes)), DECADE_ORDER)

    assert list(counts) == ["< 0.01", "0.1 - 1", "1 - 10"]


def test_an_unordered_breakdown_still_leads_with_its_largest_group():
    records = [record("a"), record("b"), record("c", "FIB")]

    assert list(ag.count_by(records, ag.by_modality)) == ["SEM", "FIB"]


def test_a_cross_tabulated_axis_can_be_ordered_too():
    table = {("1 - 10", "a"): 1, ("< 0.01", "a"): 1}

    rows, _ = ag.totals(table, row_order=DECADE_ORDER)

    assert rows == ["< 0.01", "1 - 10"]


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

def test_the_two_projects_are_different_breakdowns():
    """One is the directory the files are in, the other the project the work
    was part of. Several directories usually belong to one wider project, so
    counting by either answers a different question."""
    assert ag.BREAKDOWNS["autolamella-project"].group == "record"
    assert ag.BREAKDOWNS["project"].group == "project"


def test_counting_by_the_two_projects_gives_different_answers(tmp_path: Path):
    from am_segmentation_manager.metadata import ExperimentMetadata

    projects = [
        make_project(f"dir-{index}", f"dir-{index}", "HeLa", [record(f"a{index}")])
        for index in range(2)
    ]
    for project in projects:
        object.__setattr__(
            project, "experiment_metadata", ExperimentMetadata(project="adaptive milling")
        )
    lookup = ag.project_lookup(projects)
    records = [
        image for project in projects for image in project.image_records
    ]

    directories = ag.count_by(records, ag.by_autolamella_project(lookup))
    wider = ag.count_by(records, ag.by_project(lookup))

    assert set(directories) == {"dir-0", "dir-1"}
    assert wider == {"adaptive milling": 2}


def test_the_image_export_names_both_projects_without_colliding(tmp_path: Path):
    """project_project said nothing the shorter name does not, and there is
    nothing left to disambiguate it from now the directory is qualified."""
    from am_segmentation_manager.metadata import ExperimentMetadata

    project = make_project("dir", "dir", "HeLa", [record("a")])
    object.__setattr__(
        project, "experiment_metadata",
        ExperimentMetadata(sample="HeLa", project="adaptive milling"),
    )
    output = tmp_path / "images.csv"
    export_image_metadata_csv(
        project.image_records, tmp_path, output,
        include_headers=False, projects=ag.project_lookup([project]),
    )

    row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))

    assert row["autolamella_project"] == "dir"
    assert row["project"] == "adaptive milling"
    assert row["project_sample"] == "HeLa"
    assert "project_project" not in row


def test_the_project_export_has_no_duplicate_columns(tmp_path: Path):
    """The directory and the wider project it belongs to are different facts,
    and both are columns here. Sharing a name let DictWriter pick a winner
    silently, and the identity came out blank."""
    projects = [make_project("p", "p", "HeLa", [record("a")])]
    output = export_project_metadata_csv(projects, tmp_path, tmp_path / "p.csv")

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert len(header) == len(set(header))
    row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))
    assert row["autolamella_project"] == "p"
    assert "project" in row
    assert row["sample"] == "HeLa"


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


def test_a_cross_tabulation_can_be_headed_with_what_was_counted():
    """The window counts one particular set and has to say which; the axes go
    on naming themselves down the side and along the top either way."""
    table = {("Arctis", "SEM"): 2}
    lines = format_cross_tabulation(
        table, "microscope", "modality", title="Exportable images"
    )

    assert lines[0] == "Exportable images"
    assert lines[1].startswith("microscope")


def test_formatting_nothing_says_so_rather_than_printing_an_empty_table():
    assert "nothing to count" in format_counts({}, "Images")[0]
    assert "nothing to count" in format_cross_tabulation({}, "a", "b")[0]
    assert format_cross_tabulation({}, "a", "b", title="Exportable")[0].startswith(
        "Exportable"
    )


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
    """One row per image file, which is what a repository deposition wants."""
    from am_segmentation_manager.cli import main

    built(tmp_path)
    output = tmp_path / "images.csv"
    assert main([
        "export-metadata-csv", str(tmp_path), *SCOPE, "--no-headers",
        "--modality", "SEM", "-o", str(output),
    ]) == 0

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 2
    assert {row["modality"] for row in rows} == {"SEM"}


def test_cli_export_metadata_csv_pairs_both_modalities(tmp_path: Path):
    """Matching the training export's shape, so a row here describes a row
    there. Each beam's own values carry which beam they came from."""
    from am_segmentation_manager.cli import main

    built(tmp_path)
    output = tmp_path / "paired.csv"
    assert main([
        "export-metadata-csv", str(tmp_path), *SCOPE, "--no-headers", "-o", str(output)
    ]) == 0

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 2, "two acquisitions, both beams on one row each"
    assert "modality" not in rows[0], "a paired row has no single modality"
    assert rows[0]["sem_image"] and rows[0]["fib_image"]
    assert rows[0]["lamella"], "what the beams share is said once"


def test_cli_refuses_metadata_filters_without_headers(tmp_path: Path):
    """The filters need the headers they would be filtering on."""
    from am_segmentation_manager.cli import main

    built(tmp_path)
    assert main([
        "export-metadata-csv", str(tmp_path), *SCOPE,
        "--no-headers", "--microscope", "Arctis", "-o", str(tmp_path / "x.csv"),
    ]) == 2
