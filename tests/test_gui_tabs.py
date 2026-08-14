"""Tests for the export and metadata view models.

Both are Qt-free, so these run without a display. What they mostly pin is the
arithmetic behind the live count -- the exclusions that were previously silent
on the command line, and which the Export tab exists to make visible.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.gui.export_controller import ExportController
from am_segmentation_manager.gui.metadata_controller import (
    MetadataController,
    MetadataValidationError,
)
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
