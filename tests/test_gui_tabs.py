"""Tests for the export and metadata view models.

Both are Qt-free, so these run without a display. What they mostly pin is the
arithmetic behind the live count -- the exclusions that were previously silent
on the command line, and which the Export tab exists to make visible.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager import aggregate as ag
from am_segmentation_manager.aggregate import MISSING
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
from am_segmentation_manager.operations.selection import deselect_records
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


def test_a_deselected_image_that_would_be_exported_is_counted(tmp_path: Path):
    """The state filter opens on "all", so an image somebody decided against can
    reach a training set if it was labelled before being deselected. The
    empty-label rule does not catch that one, so the count has to show it."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, modalities=("SEM",), labels="annotated")
    projects = scanned(tmp_path)
    records = [
        record
        for project in projects
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]
    deselect_records(records[:1])

    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    preview = controller.preview()

    assert preview.deselected == 1
    assert "1 deselected" in preview.summary()


def test_nothing_deselected_says_nothing_about_it(tmp_path: Path):
    """A line that always appears is a line nobody reads."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, modalities=("SEM",), labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert controller.preview().deselected == 0
    assert "deselected" not in controller.preview().summary()


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


def test_an_empty_label_is_never_exportable_from_the_window(tmp_path: Path):
    """Looking at the empty ones is what the label filter is for; writing
    them into a training file is a deliberate act the command line keeps."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="empty")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    controller.filters.label = "empty"

    assert controller.preview().matched == 4      # visible, and countable
    assert controller.preview().exportable == 0   # but never written


def test_images_without_labels_are_counted_separately(tmp_path: Path):
    build_dataset(tmp_path, acquisitions=2, labels="none")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    preview = controller.preview()

    assert preview.missing_labels == 4
    assert preview.exportable == 0


def test_a_corrected_image_with_an_empty_label_is_flagged_before_export(
    tmp_path: Path,
):
    """Corrected claims a label was proofread. The export is right to drop one
    that holds nothing, but a training set silently short of the images you
    believe you annotated is the failure this is here to prevent."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=1, modalities=("SEM",), labels="empty")
    _mark_corrected(tmp_path)
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    found = controller.unexportable_corrected()

    assert len(found.empty) == 1
    assert found.summary() == "1 with an empty label"


def test_a_corrected_image_with_no_label_is_flagged_before_export(tmp_path: Path):
    """A group with no label at all cannot be caught by looking inside one, so
    the export's own check is the only place it surfaces."""
    build_dataset(tmp_path, acquisitions=1, modalities=("SEM",), labels="none")
    _mark_corrected(tmp_path)
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    found = controller.unexportable_corrected()

    assert len(found.missing) == 1


def test_an_unlabelled_image_that_is_not_corrected_is_not_flagged(tmp_path: Path):
    """An archive being worked through is mostly unlabelled. Warning about that
    on every export is a warning nobody reads by the third time."""
    build_dataset(tmp_path, acquisitions=2, labels="none")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert not controller.unexportable_corrected()


def test_a_corrected_annotated_image_is_not_flagged(tmp_path: Path):
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=1, modalities=("SEM",), labels="annotated")
    _mark_corrected(tmp_path)
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert not controller.unexportable_corrected()


def _mark_corrected(root: Path) -> None:
    """Declare everything under *root* corrected, as the window would.

    Through the real operation rather than by setting the field, so what the
    export reads is what a state change actually wrote.
    """
    from am_segmentation_manager.operations.selection import mark_corrected_records

    mark_corrected_records(
        [
            record
            for project in scanned(root)
            for lamella in project.lamellae
            for stage in lamella.stages
            for record in stage.images
        ]
    )


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

    everything = controller.preview()
    controller.filters.conditions = "primary"
    primary = controller.preview()

    # The rule is a filter, so it narrows what "matched" means rather than
    # being reported as a separate step further down the line.
    assert primary.matched == 1
    assert everything.matched == 3


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
        # As the window calls it: the filters have already applied the dedup
        # rule, and the window never writes blank masks.
        include_empty_labels=False,
        conditions="all",
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


def test_a_breakdown_counts_what_the_filters_match(tmp_path: Path):
    """One tab, one set. A tick that halves the export has to halve the table
    under it, or the two numbers on screen describe different collections."""
    pytest.importorskip("tifffile")
    build_dataset(
        tmp_path, layout="no_stage_level", sem_variants=("SEM_8_fi_1", "SEM_8_fi_4"),
        modalities=("SEM",), acquisitions=1, labels="annotated",
    )
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert controller.breakdown("variant") == {"8": 1, "8_fi_1": 1, "8_fi_4": 1}

    controller.filters.conditions = "primary"

    assert controller.breakdown("variant") == {"8": 1}


def test_an_image_with_no_label_is_counted_but_not_exportable(tmp_path: Path):
    """The tab has to answer "what have I got" as well as "what can I write":
    an archive that is mostly unlabelled is the normal case, and a table that
    showed none of it would be describing a collection nobody has."""
    build_dataset(tmp_path, acquisitions=2, labels="none")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert sum(controller.breakdown("modality").values()) == 4
    assert controller.preview().matched == 4
    assert controller.preview().exportable == 0


def test_the_label_filter_decides_what_the_table_counts(tmp_path: Path):
    """Finding the empty ones is the point of the filter: they are what you
    would go and correct or remove, and they cannot be found if nothing counts
    them."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="empty")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert sum(controller.breakdown("modality").values()) == 4   # "any"

    controller.filters.label = "annotated"
    assert controller.breakdown("modality") == {}

    controller.filters.label = "empty"
    assert sum(controller.breakdown("modality").values()) == 4

    controller.filters.label = "no label yet"
    assert controller.breakdown("modality") == {}


def test_the_label_states_present_can_be_counted(tmp_path: Path):
    """One row set answering "how much of this collection is annotated"."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="empty")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)

    assert controller.breakdown("label") == {"empty": 4}


def test_two_properties_cross_tabulate(headers: ExportController):
    """The question a training set raises is rarely one-dimensional: not how
    many SEM images, but which plasma gas the FIB ones were taken with."""
    table = headers.cross_tabulation("modality", "plasma-gas")

    # The fixture gives each beam a property the other cannot have, so the SEM
    # row lands under "not recorded" and the two FIB images under a gas each.
    assert table == {
        ("FIB", "Argon"): 1,
        ("FIB", "Xenon"): 1,
        ("SEM", MISSING): 2,
    }
    assert sum(table.values()) == len(headers.exportable_records())


def test_a_cross_tabulation_reads_the_headers_it_needs(exporter: ExportController):
    """Either half needing headers is enough to have to read them."""
    exporter.cross_tabulation("modality", "state")
    assert not exporter.metadata_loaded()

    exporter.cross_tabulation("modality", "voltage")
    assert exporter.metadata_loaded()


def test_labels_are_never_measured_without_being_asked(exporter: ExportController):
    """Measuring is tens of milliseconds a label. Doing it because a breakdown
    was chosen would fire on every press of an arrow key down the list."""
    exporter.breakdown("has-crack")

    assert not exporter.measurements_loaded()


def test_measuring_says_how_far_it_has_got(exporter: ExportController):
    """Tens of milliseconds a label over a whole collection, so the dialog has
    to say it is advancing rather than only that it is busy."""
    seen: list[tuple[int, int]] = []

    exporter.load_measurements(lambda done, total: seen.append((done, total)))

    assert seen, "measuring reported nothing"
    assert [done for done, _ in seen] == list(range(1, len(seen) + 1))
    assert {total for _, total in seen} == {seen[-1][0]}


def test_a_crack_breakdown_counts_nothing_until_the_labels_are_measured(
    exporter: ExportController,
):
    """Regression: the key function was built without the measurements, so
    every image came back "no-label" -- which reads as a finding about the data
    rather than as a question that was never asked."""
    assert exporter.needs_measurements("has-crack")
    assert exporter.breakdown("has-crack") == {"no-label": 4}

    exporter.load_measurements()

    assert exporter.breakdown("has-crack") != {"no-label": 4}


def test_the_crack_threshold_changes_what_counts_as_cracked(tmp_path: Path):
    """A crack is a physical feature with a size, and where the line falls is a
    judgement about the sample rather than something to hardcode."""
    cv2 = pytest.importorskip("cv2")
    controller = _measured_with_one_small_crack(tmp_path, cv2)

    # The speck is about 0.04 um^2, so it is a crack under a 0.01 threshold and
    # noise over a 0.1 one.
    controller.set_crack_threshold(0.01)
    assert controller.breakdown("has-crack") == {"cracked": 1}

    controller.set_crack_threshold(0.1)
    assert controller.breakdown("has-crack") == {"intact": 1}


def test_changing_the_threshold_re_reads_no_labels(tmp_path: Path):
    """The threshold is applied to component areas read_label_pixels caches
    *without* it, so a new one costs arithmetic rather than a second pass over
    the archive. Without this the control would be unusable on a real
    collection."""
    cv2 = pytest.importorskip("cv2")
    from am_segmentation_manager.operations import label_measure

    controller = _measured_with_one_small_crack(tmp_path, cv2)
    reads: list[Path] = []
    original = label_measure._count_pixels

    def counted(path, crack_class):
        reads.append(path)
        return original(path, crack_class)

    label_measure._count_pixels = counted
    try:
        controller.set_crack_threshold(0.02)
    finally:
        label_measure._count_pixels = original

    assert controller.measurements_loaded()
    assert reads == []


def test_setting_the_same_threshold_again_does_nothing(exporter: ExportController):
    exporter.load_measurements()
    before = exporter.measurements_loaded()

    exporter.set_crack_threshold(exporter.min_crack_area_um2)

    assert exporter.measurements_loaded() == before


def test_a_negative_crack_threshold_is_rejected(exporter: ExportController):
    with pytest.raises(ValueError, match="cannot be negative"):
        exporter.set_crack_threshold(-1.0)


def test_the_area_scale_can_be_cut_more_finely(exporter: ExportController):
    """A lamella is hundreds of um^2, so at one bin per decade nearly every one
    lands in the same bin and the breakdown says nothing about the spread."""
    assert "100 - 1000" in exporter.breakdowns["area-lamella"].order

    exporter.set_area_bins(3)

    assert exporter.area_bins_per_decade == 3
    assert "100 - 200" in exporter.breakdowns["area-lamella"].order


def test_recutting_the_scale_measures_nothing_again(exporter: ExportController):
    """How a number is grouped is not a property of the number."""
    exporter.load_measurements()

    exporter.set_area_bins(2)

    assert exporter.measurements_loaded()


def _measured_with_one_small_crack(tmp_path: Path, cv2) -> ExportController:
    """A scan of one image whose label holds a single small crack component."""
    import numpy as np
    import tifffile

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.image_metadata import ImageMetadata
    from am_segmentation_manager.operations.label_measure import clear_cache

    clear_cache()
    build_dataset(tmp_path, acquisitions=1, modalities=("SEM",), labels="annotated")
    label = next(tmp_path.rglob("*_labels/*.tif"), None) or next(
        tmp_path.rglob("labels/*.tif")
    )
    array = np.zeros((64, 64), dtype=np.uint8)
    array[10:30, 10:30] = 2      # the lamella
    array[40:42, 40:42] = 3      # a 4-pixel speck of crack/hole
    tifffile.imwrite(str(label), array)

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 2: (0, 1, 0), 3: (1, 0, 0)},
        class_labels={0: "background", 2: "lamella", 3: "crack/hole"},
        crack_class=3,
    )
    controller = ExportController(config=config)
    controller.set_projects(scanned(tmp_path), tmp_path)
    # 100 nm pixels, so the 4-pixel speck is 0.04 um^2.
    controller._metadata = {
        record.image_path: ImageMetadata(pixel_size_nm=100.0, image_width=64)
        for record in controller.records
    }
    controller._metadata_loaded = True
    controller.load_measurements()
    return controller


def test_measuring_reports_what_it_covered(exporter: ExportController):
    assert exporter.measurable_labels() == 4
    assert exporter.load_measurements() == 4
    assert exporter.measurements_loaded()


def test_a_second_scan_forgets_what_was_measured(exporter: ExportController, tmp_path):
    """The measurements belong to the scan they were taken from."""
    exporter.load_measurements()

    exporter.set_projects(scanned(tmp_path), tmp_path)

    assert not exporter.measurements_loaded()


def test_class_areas_are_offered_for_the_classes_the_config_measures(
    exporter: ExportController,
):
    """The dropdown is built from the config, so a project that renamed its
    classes gets its own areas rather than this project's."""
    offered = {
        name for _, rows in exporter.available_breakdowns() for name, _ in rows
    }

    assert {"area-gis", "area-lamella", "area-crack-hole"} <= offered


def test_an_area_breakdown_keeps_its_bins_in_order(exporter: ExportController):
    """Ordered by population, bins of increasing area stop reading as a
    distribution."""
    exporter.load_measurements()
    order = exporter.breakdowns["area-lamella"].order

    counts = exporter.breakdown("area-lamella")

    positions = [order.index(key) for key in counts]
    assert positions == sorted(positions)


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


def test_filter_values_come_from_the_data(exporter: ExportController):
    """Offering what the archive holds is the difference between a filter you
    can explore and one you must already know the answer to."""
    # Synthetic TIFFs carry no header, and "no header" is a value you can ask
    # for rather than a gap in the list: it is how the unreadable ones are found.
    assert exporter.choice_values("microscope") == [ag.MISSING]


def test_a_filter_offers_only_what_the_selection_still_holds(headers: ExportController):
    """Regression: the lists were built from the whole scan, so looking at SEM
    images offered Argon as a plasma gas -- which reads as the SEM claiming it."""
    headers.filters.modality = "SEM"

    assert "Argon" not in headers.choice_values("plasma-gas")
    assert headers.choice_values("electron-source") == ["FEG"]


def test_images_with_no_header_can_be_filtered_for(headers: ExportController):
    """"Not recorded" is a value the list offers, not a gap in it. Before, a
    metadata filter excluded every image whose header could not be read and
    there was no way to ask which those were; now selecting it finds them."""
    headers.filters.modality = "FIB"
    headers.set_choice_filter("electron-source", [ag.MISSING])

    assert headers.preview().matched == 2   # the FIB images, which have no emitter


def test_a_filter_does_not_narrow_its_own_list(headers: ExportController):
    """Choosing one gas must leave the others reachable, or a filter list
    collapses to the single value already chosen."""
    headers.set_choice_filter("plasma-gas", ["Argon"])

    assert {"Argon", "Xenon"} <= set(headers.choice_values("plasma-gas"))


def test_a_chosen_value_stays_offered_when_something_else_excludes_it(
    headers: ExportController,
):
    """A filter that vanished from its own list could not be seen or cleared,
    and would go on excluding everything."""
    headers.set_choice_filter("plasma-gas", ["Argon"])
    headers.filters.modality = "SEM"

    assert "Argon" in headers.choice_values("plasma-gas")
    assert headers.preview().matched == 0


def test_an_unknown_filter_is_rejected(exporter: ExportController):
    with pytest.raises(ValueError, match="Unknown filter"):
        exporter.set_choice_filter("colour", ["red"])


def test_a_measured_property_is_not_offered_as_a_filter(exporter: ExportController):
    """It cannot answer until the labels are measured, and a filter that keeps
    nothing until a button elsewhere is pressed is worse than no filter."""
    offered = dict(exporter.filterable_breakdowns())

    assert "has-crack" not in offered
    with pytest.raises(ValueError, match="measured property"):
        exporter.set_choice_filter("has-crack", ["cracked"])


def test_a_property_with_its_own_control_is_not_also_a_list(
    exporter: ExportController,
):
    """Modality is three fixed choices, not whatever the data holds, and the
    label filter asks five questions rather than offering four states."""
    offered = dict(exporter.filterable_breakdowns())

    assert "modality" not in offered
    assert "state" not in offered
    assert "label" not in offered


def test_the_project_and_lamella_can_be_filtered_on(exporter: ExportController):
    """The last thing the command line could narrow that the window could not."""
    offered = dict(exporter.filterable_breakdowns())

    assert "project" in offered
    assert "lamella" in offered
    assert "variant" in offered


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
