"""Tests for the export and metadata view models.

Both are Qt-free, so these run without a display. What they mostly pin is the
arithmetic behind the live count -- the exclusions that were previously silent
on the command line, and which the Export tab exists to make visible.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager.gui.export_controller import ExportController
from am_segmentation_manager.gui.metadata_controller import (
    SUGGESTION_LIMIT,
    MetadataController,
    MetadataValidationError,
)
from am_segmentation_manager.image_metadata import ELECTRON, ION, ImageMetadata
from am_segmentation_manager.metadata import DESCRIPTIVE_FIELDS, build_experiment_metadata
from am_segmentation_manager.models import Project
from am_segmentation_manager.operations.csv import export_training_csv
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset


def scanned(root: Path):
    return Scanner(root).build_index().all_projects()


@pytest.fixture
def exporter(tmp_path: Path) -> ExportController:
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    return controller


@pytest.fixture
def headers(exporter: ExportController, monkeypatch) -> ExportController:
    """The same scan, with headers, without writing any.

    Each beam gets a property the other cannot have -- a plasma gas for the ion
    column, an emitter for the electron one -- and the two FIB images get
    different gases, which is what the filter lists are asserted against.
    """
    from am_segmentation_manager.gui import export_controller as module

    def read(records) -> dict[Path, ImageMetadata]:
        gases = ["Argon", "Xenon"]
        entries = {}
        for record in records:
            if record.modality == "FIB":
                entry = ImageMetadata(beam_type=ION, plasma_gas=gases[len(entries) % 2])
            else:
                entry = ImageMetadata(beam_type=ELECTRON, electron_source="FEG")
            entries[record.image_path] = entry
        return entries

    monkeypatch.setattr(module, "read_records_metadata", read)
    return exporter


# ---------------------------------------------------------------------------
# Export: the live count
# ---------------------------------------------------------------------------

def test_the_preview_counts_what_would_be_exported(exporter: ExportController):
    preview = exporter.preview()

    assert preview.matched == 4          # two acquisitions, two modalities
    assert preview.annotated == 4
    assert preview.exportable == 4


def test_empty_labels_are_shown_as_excluded_rather_than_silently_dropped(tmp_path: Path):
    """The reason this view exists: on the command line this exclusion is
    reported only in the log, after the file has been written."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="empty")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    preview = controller.preview()

    assert preview.matched == 4
    assert preview.empty_labels == 4
    assert preview.exportable == 0
    assert "empty label" in preview.summary()


def test_including_empty_labels_makes_them_exportable(tmp_path: Path):
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="empty")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    controller.filters.include_empty_labels = True

    assert controller.preview().exportable == 4


def test_images_without_labels_are_counted_separately(tmp_path: Path):
    build_dataset(tmp_path, acquisitions=2, labels="none")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    preview = controller.preview()

    assert preview.missing_labels == 4
    assert preview.exportable == 0


def test_one_condition_per_scene_is_visible_before_exporting(tmp_path: Path):
    """Several imaging conditions share a label, so exporting all of them
    multiplies the dataset by copies of one scene."""
    pytest.importorskip("tifffile")
    build_dataset(
        tmp_path, layout="no_stage_level", sem_variants=("SEM_8_fi_1", "SEM_8_fi_4"),
        modalities=("SEM",), acquisitions=1, labels="annotated",
    )
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    primary = controller.preview()
    controller.filters.conditions = "all"
    everything = controller.preview()

    assert primary.matched == 3
    assert primary.after_conditions == 1
    assert everything.after_conditions == 3


# ---------------------------------------------------------------------------
# Export: the training split
# ---------------------------------------------------------------------------

def csv_rows(path: Path | None) -> int:
    if path is None:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1   # less the header


def written_split(exporter: ExportController, output: Path) -> tuple[int, int]:
    """Export for real, and return how the two files came out."""
    train, validation = export_training_csv(
        records=exporter.selected_records(),
        start_path=exporter.start_path,
        output_dir=output,
        prefix=exporter.training.prefix,
        train_ratio=exporter.training.train_ratio,
        seed=exporter.training.seed,
        include_empty_labels=exporter.filters.include_empty_labels,
        conditions=exporter.filters.conditions,
    )
    return csv_rows(train), csv_rows(validation)


def test_the_split_defaults_to_what_the_command_line_does(exporter: ExportController):
    """The same selection exported from either interface must give the same two
    files, so the defaults cannot drift apart."""
    assert (exporter.training.train_ratio, exporter.training.seed) == (0.8, 42)
    assert exporter.training.prefix == "dataset"


def test_the_preview_and_the_export_agree_on_the_split(
    exporter: ExportController, tmp_path: Path
):
    """The count shown is produced by the same split as the files, which is the
    only way a preview of a shuffle can be trusted."""
    preview = exporter.preview()

    assert (preview.train, preview.validation) == written_split(
        exporter, tmp_path / "exported"
    )
    assert preview.train and preview.validation   # not vacuously equal at 0
    assert preview.train + preview.validation == preview.exportable


def test_a_full_training_share_leaves_nothing_to_validate(exporter: ExportController):
    exporter.training.train_ratio = 1.0
    preview = exporter.preview()

    assert (preview.train, preview.validation) == (preview.exportable, 0)


def test_an_unseeded_split_still_places_every_pair(
    exporter: ExportController, tmp_path: Path
):
    exporter.training.seed = None
    preview = exporter.preview()

    assert preview.train + preview.validation == preview.exportable
    assert (preview.train, preview.validation) == written_split(
        exporter, tmp_path / "unseeded"
    )


def test_the_summary_reports_the_split_it_would_write(exporter: ExportController):
    assert "train / 1 validation" in exporter.preview().summary()


def test_nothing_to_export_reports_no_split(exporter: ExportController):
    exporter.filters.state = "corrected"

    assert "train" not in exporter.preview().summary()


def test_filters_narrow_the_selection(exporter: ExportController):
    exporter.filters.modality = "FIB"

    assert exporter.preview().matched == 2


def test_a_selection_matching_nothing_says_so(exporter: ExportController):
    exporter.filters.state = "corrected"

    assert exporter.preview().matched == 0
    assert "Nothing matches" in exporter.preview().summary()


# ---------------------------------------------------------------------------
# Export: breakdowns and metadata
# ---------------------------------------------------------------------------

def test_breakdowns_count_the_current_selection(exporter: ExportController):
    exporter.filters.modality = "SEM"

    assert exporter.breakdown("modality") == {"SEM": 2}


def test_headers_are_only_read_when_something_needs_them(exporter: ExportController):
    """A structural breakdown must stay a filesystem walk."""
    exporter.breakdown("modality")
    assert not exporter.metadata_loaded()

    exporter.breakdown("voltage")
    assert exporter.metadata_loaded()


def test_headers_are_not_re_read_when_none_could_be_parsed(exporter: ExportController):
    """Regression: "loaded but found nothing" looked the same as "not loaded",
    so a collection of unreadable headers was re-read on every filter change."""
    assert exporter.load_metadata() == 0
    assert exporter.metadata_loaded()

    calls = []
    import am_segmentation_manager.gui.export_controller as module

    original = module.read_records_metadata
    module.read_records_metadata = lambda records: calls.append(1) or {}
    try:
        exporter.load_metadata()
    finally:
        module.read_records_metadata = original

    assert calls == []


def test_metadata_filter_values_come_from_the_data(exporter: ExportController):
    """Offering what the archive holds is the difference between a filter you
    can explore and one you must already know the answer to."""
    assert exporter.metadata_values("microscope") == []   # synthetic TIFFs, no header


def test_a_filter_offers_only_what_the_selection_still_holds(headers: ExportController):
    """Regression: the lists were built from the whole scan, so looking at SEM
    images offered Argon as a plasma gas -- which reads as the SEM claiming it."""
    headers.filters.modality = "SEM"

    assert headers.metadata_values("plasma-gas") == []
    assert headers.metadata_values("electron-source") == ["FEG"]


def test_a_filter_does_not_narrow_its_own_list(headers: ExportController):
    """Choosing one gas must leave the others reachable, or a filter list
    collapses to the single value already chosen."""
    headers.set_metadata_filter("plasma-gas", ["Argon"])

    assert headers.metadata_values("plasma-gas") == ["Argon", "Xenon"]


def test_a_chosen_value_stays_offered_when_something_else_excludes_it(
    headers: ExportController,
):
    """A filter that vanished from its own list could not be seen or cleared,
    and would go on excluding everything."""
    headers.set_metadata_filter("plasma-gas", ["Argon"])
    headers.filters.modality = "SEM"

    assert headers.metadata_values("plasma-gas") == ["Argon"]
    assert headers.preview().matched == 0


def test_an_unknown_metadata_filter_is_rejected(exporter: ExportController):
    with pytest.raises(ValueError, match="Unknown metadata filter"):
        exporter.set_metadata_filter("colour", ["red"])


def test_the_export_view_reuses_the_scan_rather_than_repeating_it(exporter: ExportController):
    """A second scan for this view would double the slowest thing the
    application does."""
    assert exporter.records
    assert exporter.projects


# ---------------------------------------------------------------------------
# Metadata tab
# ---------------------------------------------------------------------------

@pytest.fixture
def metadata(tmp_path: Path) -> MetadataController:
    build_dataset(tmp_path, acquisitions=1)
    controller = MetadataController()
    controller.set_projects(scanned(tmp_path))
    return controller


def test_values_and_sources_are_reported_together(metadata: MetadataController):
    """Showing which file supplied a value is most of the point of the tab."""
    metadata.editing_enabled = True
    metadata.apply({"sample": "HeLa"})

    assert metadata.values()["sample"] == "HeLa"
    assert metadata.sources()["sample"] == "AM_metadata.yaml"


def test_editing_is_refused_until_it_is_enabled(metadata: MetadataController):
    with pytest.raises(PermissionError):
        metadata.apply({"sample": "HeLa"})


def test_an_invalid_edit_is_rejected(metadata: MetadataController):
    metadata.editing_enabled = True

    with pytest.raises(MetadataValidationError):
        metadata.apply({"lamella_type": "banana"})


def test_the_view_refreshes_after_a_write(metadata: MetadataController):
    """Otherwise the tab keeps showing what was there before the edit, which is
    how a user learns not to trust the tool."""
    metadata.editing_enabled = True
    metadata.apply({"sample": "HeLa"})

    assert metadata.values()["sample"] == "HeLa"
    # Two: the sample just set, plus lamella_type derived from the project's
    # experiment.yaml method.
    assert "2 of 6" in metadata.summary()


def test_planning_writes_nothing(metadata: MetadataController):
    change = metadata.plan({"sample": "HeLa"})

    assert not change.is_empty
    assert metadata.values()["sample"] == ""


def test_outstanding_fields_are_reported(metadata: MetadataController):
    metadata.editing_enabled = True
    metadata.apply({"sample": "TODO"})

    assert "sample" in metadata.outstanding()


# ---------------------------------------------------------------------------
# Suggestions, which are read off the projects rather than off the disk
# ---------------------------------------------------------------------------

def with_samples(*samples: str) -> MetadataController:
    """Return a controller over one project per sample, selecting the first."""
    controller = MetadataController()
    controller.set_projects([
        Project(
            name=f"project-{position}",
            path=Path(f"/collection/project-{position}"),
            relative_path=f"project-{position}",
            experiment_metadata=build_experiment_metadata(
                {}, {"metadata": {"sample": sample}}
            ),
        )
        for position, sample in enumerate(samples)
    ])
    return controller


def test_suggestions_put_the_most_used_value_first():
    controller = with_samples("HeLa", "yeast", "COS-7", "HeLa", "COS-7", "HeLa")

    assert controller.suggestions("sample") == ["HeLa", "COS-7", "yeast"]


def test_equally_used_values_are_ordered_alphabetically():
    """Otherwise the list reshuffles between projects for no visible reason."""
    controller = with_samples("yeast", "COS-7", "HeLa")

    assert controller.suggestions("sample") == ["COS-7", "HeLa", "yeast"]


def test_a_field_with_nothing_recorded_offers_nothing():
    controller = with_samples()

    assert controller.suggestions("sample") == []


def test_fields_that_do_not_repeat_are_not_offered_a_list():
    """`description` is prose, and `project` names this project rather than a
    shared one, so neither is worth a list of other projects' values."""
    controller = with_samples("HeLa", "HeLa")

    assert controller.suggestions("description") == []
    assert controller.suggestions("project") == []


def test_the_list_is_capped():
    controller = with_samples(*(f"sample-{n:02d}" for n in range(SUGGESTION_LIMIT + 5)))

    assert len(controller.suggestions("sample")) == SUGGESTION_LIMIT


def test_the_selected_value_survives_the_cap():
    """A box offering a list that omits what it is showing invites replacing
    the value by accident."""
    rare = "zzz-rare-sample"
    controller = with_samples(
        rare, *(f"sample-{n:02d}" for n in range(SUGGESTION_LIMIT))
    )

    offered = controller.suggestions("sample")
    assert len(offered) == SUGGESTION_LIMIT + 1
    assert offered[-1] == rare


def test_a_saved_value_becomes_a_suggestion(metadata: MetadataController):
    """The point of rebuilding the list on every refresh: type a sample once
    and pick it on the next project."""
    assert metadata.suggestions("sample") == []
    metadata.editing_enabled = True
    metadata.apply({"sample": "HeLa"})

    assert metadata.suggestions("sample") == ["HeLa"]


# ---------------------------------------------------------------------------
# The project's own directory, offered as its project name
# ---------------------------------------------------------------------------

def at(relative: str) -> MetadataController:
    """Return a controller over one project at *relative* within a scan root."""
    controller = MetadataController()
    controller.set_projects([
        Project(
            name="recorded-name",
            path=Path("/collection") / relative,
            relative_path=str(Path(relative)),
        )
    ])
    return controller


def test_the_directory_holding_the_yaml_files_is_offered_as_the_project_name():
    controller = at("20260728_CB_AM_bitmap/AutoLamella-2026-07-28-13-57")

    assert controller.suggested_project_name() == "AutoLamella-2026-07-28-13-57"


def test_the_directory_is_preferred_over_the_name_experiment_yaml_records():
    """A project copied to a second location keeps the recorded name of the
    first; the directory is what is actually there."""
    controller = at("AutoLamella-2026-07-28-13-57")

    assert controller.projects[0].name == "recorded-name"
    assert controller.suggested_project_name() == "AutoLamella-2026-07-28-13-57"


def test_nothing_is_offered_without_a_project():
    assert MetadataController().suggested_project_name() is None


def test_the_suggestion_is_not_a_resolved_value():
    """It is a starting point for a person, not a value: until it is saved the
    field is still unset, and a project nobody opens keeps it that way."""
    controller = at("20260728_CB_AM_bitmap/AutoLamella-2026-07-28-13-57")

    assert controller.suggested_project_name() == "AutoLamella-2026-07-28-13-57"
    assert controller.values()["project"] == ""


# ---------------------------------------------------------------------------
# Which projects still need describing, and where they are
# ---------------------------------------------------------------------------

def test_a_project_with_no_metadata_file_is_missing_every_field():
    """It is in the same position as one whose fields are all placeholders."""
    controller = at("AutoLamella-2026-07-28-13-57")

    summary = controller.project_summaries()[0]
    assert summary.filled == 0
    assert summary.is_complete is False
    assert set(summary.missing) == set(DESCRIPTIVE_FIELDS)


def test_a_placeholder_counts_as_missing(metadata: MetadataController):
    """A TODO is a note that nobody has filled the field in, not a value."""
    metadata.editing_enabled = True
    metadata.apply({"sample": "TODO", "user": "Casper"})

    summary = metadata.project_summaries()[0]
    assert "sample" in summary.missing
    assert "user" not in summary.missing


def test_a_summary_names_the_fields_still_wanted():
    controller = at("AutoLamella-2026-07-28-13-57")

    assert "missing: description" in controller.project_summaries()[0].describe()


def test_a_complete_project_is_reported_as_complete(metadata: MetadataController):
    metadata.editing_enabled = True
    metadata.apply({
        field: "filled" if field != "lamella_type" else "waffle"
        for field in DESCRIPTIVE_FIELDS
    })

    summary = metadata.project_summaries()[0]
    assert summary.is_complete is True
    assert summary.filled == summary.total
    assert "missing" not in summary.describe()


def test_every_project_gets_a_row_not_only_the_selected_one():
    """The point of the marks: seeing what is left without clicking each."""
    controller = with_samples("HeLa", "yeast", "COS-7")

    assert len(controller.project_summaries()) == 3


def test_a_project_path_is_absolute():
    """A path pasted into a terminal has to be, or copying one is no use."""
    controller = at("20260728_CB_AM_bitmap/AutoLamella-2026-07-28-13-57")

    assert controller.project_path(0) == Path(
        "/collection/20260728_CB_AM_bitmap/AutoLamella-2026-07-28-13-57"
    )


def test_a_path_is_not_invented_for_a_row_that_is_not_there():
    assert MetadataController().project_path(0) is None
    assert at("project").project_path(7) is None
