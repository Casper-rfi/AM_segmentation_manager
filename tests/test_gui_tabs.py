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
from am_segmentation_manager.gui import export_controller as ec
from am_segmentation_manager.gui.export_controller import ExportController
from am_segmentation_manager.gui.metadata_controller import (
    SUGGESTION_LIMIT,
    MetadataController,
    MetadataValidationError,
)
from am_segmentation_manager.image_metadata import ELECTRON, ION, ImageMetadata
from am_segmentation_manager.metadata import (
    DESCRIPTIVE_FIELDS,
    FIELDS,
    NUMBER,
    build_experiment_metadata,
)
from am_segmentation_manager.models import Project
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
def measured(exporter: ExportController) -> ExportController:
    """The same scan with its labels read, as pressing Measure labels does.

    Nothing below the first two counts in the summary can be answered without
    opening every label, so a test about those has to have paid for them --
    which is the point of the button.
    """
    exporter.load_measurements()
    return exporter


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
# Export: forgetting a label that has been replaced
# ---------------------------------------------------------------------------

def test_a_replaced_label_is_read_again_rather_than_remembered(
    measured: ExportController,
):
    """Converting can overwrite a label this view has already opened and
    measured, and all three answers are keyed by path -- so without this the
    tab goes on reporting the state and the area of the label that is gone.
    The scan after converting used to reset it by handing the projects over
    again."""
    from am_segmentation_manager.operations.csv import find_matching_label

    # A *label* path, which is what a conversion reports writing. The three
    # caches are not keyed alike, so a test picking whichever key came to hand
    # would pass while two of them kept their answer.
    record = measured.selected_records()[0]
    label = find_matching_label(record)
    assert label in measured._label_empty
    assert record.image_path in measured._measurements

    forgotten = measured.forget_labels([label])

    assert forgotten == 1
    assert label not in measured._label_empty
    assert record.image_path not in measured._measurements
    assert record.image_path not in measured._label_states


def test_forgetting_one_label_forgets_every_image_that_shares_it(
    measured: ExportController,
):
    """A conversion writes one label for a scene, and the measurement is held
    per image -- so a condition left remembering it would keep reporting the
    area of the label that was replaced."""
    from am_segmentation_manager.operations.csv import find_matching_label

    record = measured.selected_records()[0]
    label = find_matching_label(record)
    sharing = [r for r in measured.records if label in r.labels]
    assert len(sharing) >= 1
    for other in sharing:
        assert other.image_path in measured._measurements

    measured.forget_labels([label])

    assert not any(r.image_path in measured._measurements for r in sharing)


def test_forgetting_a_label_nothing_was_read_from_changes_nothing(
    measured: ExportController, tmp_path: Path
):
    """Every conversion reports its destinations, most of which this view has
    never opened."""
    assert measured.forget_labels([tmp_path / "never-read.tif"]) == 0


# ---------------------------------------------------------------------------
# Export: the live count
# ---------------------------------------------------------------------------

def test_the_preview_counts_what_would_be_exported(measured: ExportController):
    """Both modalities are in play, so a row carries an acquisition rather than
    an image: four images are two acquisitions, side by side."""
    preview = measured.preview()

    assert preview.matched == 4          # two acquisitions, two modalities
    assert preview.annotated == 4
    assert preview.paired
    assert preview.exportable == 2
    assert (preview.sem_only, preview.fib_only) == (0, 0)


def test_choosing_one_modality_counts_images_again(measured: ExportController):
    """A single-beam export is one row per image, as it always was."""
    measured.filters.modality = "SEM"
    preview = measured.preview()

    assert not preview.paired
    assert preview.exportable == 2


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
    controller.load_measurements()
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
    controller.load_measurements()

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
    """Export for real, and return how the two files came out.

    Through the controller, which is what the window calls: which records,
    which shape, how split and what named are all its decisions.
    """
    train, validation = exporter.export_training(output)
    return csv_rows(train), csv_rows(validation)


def test_the_split_defaults_to_what_the_command_line_does(exporter: ExportController):
    """The same selection exported from either interface must give the same two
    files, so the defaults cannot drift apart."""
    assert (exporter.training.train_ratio, exporter.training.seed) == (0.8, 42)
    assert exporter.training.prefix == "dataset"


def test_the_preview_and_the_export_agree_on_the_split(
    measured: ExportController, tmp_path: Path
):
    """The count shown is produced by the same split as the files, which is the
    only way a preview of a shuffle can be trusted."""
    preview = measured.preview()

    assert (preview.train, preview.validation) == written_split(
        measured, tmp_path / "exported"
    )
    assert preview.train + preview.validation == preview.exportable


def test_a_full_training_share_leaves_nothing_to_validate(exporter: ExportController):
    exporter.training.train_ratio = 1.0
    preview = exporter.preview()

    assert (preview.train, preview.validation) == (preview.exportable, 0)


def test_an_unseeded_split_still_places_every_pair(
    measured: ExportController, tmp_path: Path
):
    measured.training.seed = None
    preview = measured.preview()

    assert preview.train + preview.validation == preview.exportable
    assert (preview.train, preview.validation) == written_split(
        measured, tmp_path / "unseeded"
    )


def test_the_summary_reports_the_split_it_would_write(measured: ExportController):
    summary = measured.preview().summary()

    assert "paired row(s)" in summary
    assert "train /" in summary and "validation" in summary


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


def test_the_project_can_be_filtered_on(exporter: ExportController):
    """The last thing the command line could narrow that the window could not."""
    offered = dict(exporter.filterable_breakdowns())

    assert "project" in offered
    assert "variant" in offered


def test_a_property_whose_values_nobody_could_choose_from_is_not_offered(
    exporter: ExportController,
):
    """A lamella is one row per lamella and a milling stage one per timestamped
    folder, both running to hundreds; an electron source does not vary here at
    all. All three stay countable, and the command line still filters on
    them -- what goes is the list in the window."""
    offered = dict(exporter.filterable_breakdowns())

    for name in ("lamella", "stage", "electron-source"):
        assert name not in offered
        assert name in exporter.breakdowns, "still a breakdown, just not a filter"


def test_the_filter_dropdown_is_grouped_as_the_breakdown_one_is(
    exporter: ExportController,
):
    """A property should sit in the same place whether it is being counted by
    or filtered on."""
    groups = exporter.filterable_groups()
    headings = [heading for heading, _ in groups]
    offered = {name for _, rows in groups for name, _ in rows}

    assert headings == [h for h, _ in exporter.available_breakdowns() if h in headings]
    assert offered == set(dict(exporter.filterable_breakdowns()))
    assert all(rows for _, rows in groups), "an empty heading is not shown"


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
    assert f"2 of {len(DESCRIPTIVE_FIELDS)}" in metadata.summary()


def test_planning_writes_nothing(metadata: MetadataController):
    """One change per project in scope, which for one project is one."""
    changes = metadata.plan({"sample": "HeLa"})

    assert len(changes) == 1
    assert not changes[0].is_empty
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


def test_the_wider_project_is_not_guessed_from_the_directory():
    """The directory *is* the Autolamella project, and every export records it
    as one. Copying it into the field naming the wider project the work belongs
    to would fill a human-only answer with a machine guess, and put the same
    fact in the collection twice under two meanings."""
    controller = at("20260728_CB_AM_bitmap/AutoLamella-2026-07-28-13-57")

    assert controller.values()["project"] == ""
    assert not hasattr(controller, "suggested_project_name")


def test_the_wider_project_is_offered_from_the_values_already_in_use():
    """It repeats across the datasets of one project by definition, which is
    what makes typing it twice the way a collection ends up with two of it."""
    from am_segmentation_manager.gui.metadata_controller import SUGGESTED_FIELDS

    assert "project" in SUGGESTED_FIELDS


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


def _acceptable(entry) -> str:
    """Return a value the field *entry* describes will accept.

    Read off the table so a field added with a kind or a vocabulary of its own
    does not have to be remembered here as well.
    """
    if entry.choices:
        return entry.choices[0]
    if entry.kind == NUMBER:
        return "30"
    return "filled"


def test_a_complete_project_is_reported_as_complete(metadata: MetadataController):
    metadata.editing_enabled = True
    metadata.apply({entry.name: _acceptable(entry) for entry in FIELDS})

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


# ---------------------------------------------------------------------------
# Reading the labels once, off the interface thread
# ---------------------------------------------------------------------------
#
# The tab cannot show a single number without knowing what is inside each
# label, and over a real collection that is half a minute of reading. What
# these pin is that it happens once, is counted in files, and that nothing
# afterwards opens a label again.

def _counting_reads(monkeypatch) -> list[Path]:
    """Record every label file opened, in order."""
    opened: list[Path] = []
    real = ec.is_label_file_empty

    def counted(path):
        opened.append(path)
        return real(path)

    monkeypatch.setattr(ec, "is_label_file_empty", counted)
    return opened


def test_warming_reports_progress_in_label_files_not_images(
    tmp_path: Path, monkeypatch,
):
    """Several imaging conditions of one scene share a label and cost one read
    between them, so a counter over images would overstate the work and then
    jump."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=3, labels="annotated",
                  sem_variants=("SEM_8_fi_1",))
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    seen: list[tuple[int, int]] = []

    read = controller.warm_labels(lambda done, total: seen.append((done, total)))

    assert seen == [(n, read) for n in range(1, read + 1)]
    assert read < len(controller.records), "a shared label is read once"


def test_warming_reads_each_label_once(tmp_path: Path, monkeypatch):
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=3, labels="annotated",
                  sem_variants=("SEM_8_fi_1",))
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    opened = _counting_reads(monkeypatch)

    controller.warm_labels()

    assert len(opened) == len(set(opened))
    assert controller.labels_known()


def test_nothing_opens_a_label_again_once_warm(tmp_path: Path, monkeypatch):
    """The whole point: after this the tab redraws on every keystroke without
    touching the disk."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=3, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    controller.warm_labels()
    opened = _counting_reads(monkeypatch)

    controller.preview()
    controller.breakdown("label")
    controller.filters.label = "annotated"
    controller.selected_records()

    assert opened == []


def test_measuring_answers_what_is_in_a_label_without_a_second_read(
    tmp_path: Path, monkeypatch,
):
    """Measuring opens the label anyway, and whether there is anything in it is
    the same read -- so the exportable count and the label-state breakdown come
    out of the one pass the user asked for."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=3, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    controller.load_measurements()
    opened = _counting_reads(monkeypatch)

    controller.preview()
    controller.breakdown("label")
    controller.filters.label = "annotated"
    controller.selected_records()

    assert opened == []
    assert controller.measurements_loaded()

def test_a_rescan_makes_the_labels_cold_again(tmp_path: Path):
    """They are read from disk, and a rescan is what follows something that
    wrote to it."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    controller.warm_labels()

    controller.set_projects(scanned(tmp_path), tmp_path)

    assert not controller.labels_known()


# ---------------------------------------------------------------------------
# What opening the tab costs, and what it does not
# ---------------------------------------------------------------------------

def test_the_summary_says_what_it_cannot_answer_yet(tmp_path: Path, monkeypatch):
    """The counts below the first two need every label opened. Showing them as
    zero would read as a finding rather than as a question not yet asked."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    opened = _counting_reads(monkeypatch)

    preview = controller.preview()

    assert opened == [], "the summary must open no label"
    assert not preview.measured
    assert preview.matched == 4
    assert "Measure labels" in preview.summary()
    assert "exportable" not in preview.summary()


def test_measuring_covers_what_the_filters_match_and_no_more(tmp_path: Path):
    """An export is chosen by narrowing, so measuring an archive to answer a
    question about one project of it is work nobody asked for."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, name="wanted", acquisitions=2, labels="annotated")
    build_dataset(tmp_path, name="other", acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    everything = controller.measurable_labels()

    controller.set_choice_filter("autolamella-project", ["wanted"])

    assert controller.measurable_labels() < everything
    controller.load_measurements()
    assert controller.measurements_loaded()


def test_widening_a_filter_asks_for_the_labels_it_brings_in(tmp_path: Path):
    """A tab that went on answering from what it measured before would be
    answering about a different set than the one on screen."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, name="wanted", acquisitions=2, labels="annotated")
    build_dataset(tmp_path, name="other", acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    controller.set_choice_filter("autolamella-project", ["wanted"])
    controller.load_measurements()
    assert controller.measurements_loaded()

    controller.set_choice_filter("autolamella-project", [])

    assert not controller.measurements_loaded()
    assert controller.measurable_labels() > 0
    assert not controller.preview().measured


def test_measuring_leaves_nothing_for_an_export_to_read(tmp_path: Path, monkeypatch):
    """Which is what makes writing the file instant once the numbers are on
    screen: the export asks the same question of the same cache."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    controller.load_measurements()
    opened = _counting_reads(monkeypatch)

    controller.export_training(tmp_path / "out")

    assert opened == []


# ---------------------------------------------------------------------------
# Filtering on a property an image has several of
# ---------------------------------------------------------------------------

@pytest.fixture
def several(tmp_path: Path) -> ExportController:
    """A scan whose project holds two organisations and two sample states."""
    pytest.importorskip("tifffile")
    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    project = next(tmp_path.rglob("experiment.yaml")).parent
    (project / "AM_metadata.yaml").write_text(
        "metadata:\n"
        "  organisation: [RFI, Diamond]\n"
        "  sample_state: [infected, differentiated]\n",
        encoding="utf-8",
    )
    controller = ExportController()
    controller.set_projects(scanned(tmp_path), tmp_path)
    return controller


def test_a_filter_offers_each_value_rather_than_the_set(several: ExportController):
    """"Diamond; RFI" as a single entry could not be used to find every project
    either of them touched, which is the point of holding them apart."""
    assert several.choice_values("organisation") == ["Diamond", "RFI"]


def test_filtering_on_one_value_keeps_images_holding_it_among_others(
    several: ExportController,
):
    """An export narrowed to infected samples wants every image of one, not
    only those of samples that are nothing else."""
    everything = several.preview().matched

    several.set_choice_filter("sample-state", ["infected"])

    assert several.preview().matched == everything
    assert everything > 0


def test_filtering_on_a_value_nothing_holds_keeps_nothing(several: ExportController):
    several.set_choice_filter("organisation", ["EMBL"])

    assert several.preview().matched == 0


def test_the_breakdown_counts_an_image_under_each_of_its_values(
    several: ExportController,
):
    counts = several.breakdown("organisation")

    assert counts == {"Diamond": 4, "RFI": 4}
    assert sum(counts.values()) > several.preview().matched


# ---------------------------------------------------------------------------
# Describing a whole folder at once
# ---------------------------------------------------------------------------

@pytest.fixture
def session(tmp_path: Path) -> MetadataController:
    """Three projects filed under one session folder, as Autolamella leaves them."""
    folder = tmp_path / "20250428_AM"
    projects = []
    for name in ("run-one", "run-two", "run-three"):
        root = folder / name
        root.mkdir(parents=True)
        (root / "AM_metadata.yaml").write_text("metadata:\n  organisation: RFI\n")
        projects.append(
            Project(
                name=name,
                path=root,
                relative_path=f"20250428_AM/{name}",
                experiment_metadata=build_experiment_metadata(
                    {}, {"metadata": {"organisation": "RFI"}},
                    am_metadata_path=root / "AM_metadata.yaml",
                ),
            )
        )
    controller = MetadataController()
    controller.set_projects(projects)
    controller.editing_enabled = True
    return controller


def test_choosing_a_folder_describes_every_project_under_it(session):
    session.select_folder("20250428_AM")

    assert len(session.scope) == 3
    assert session.scope_folder == "20250428_AM"
    assert session.project is None, "several projects means no single one to name"


def test_a_field_they_agree_on_reads_as_that_value(session):
    session.select_folder("20250428_AM")
    session.apply({"sample": "HeLa"})

    assert session.values()["sample"] == "HeLa"


def test_a_field_they_disagree_on_reads_as_mixed(session):
    """The rule: fields that differ show mixed and are left alone unless
    touched."""
    from am_segmentation_manager.gui.metadata_controller import MIXED_MARKER

    session.select(0)
    session.apply({"sample": "HeLa"})
    session.select_folder("20250428_AM")

    assert session.values()["sample"] == MIXED_MARKER
    assert "sample" in session.mixed_fields()


def test_saving_writes_to_every_project_in_the_folder(session):
    session.select_folder("20250428_AM")

    changes = session.apply({"sample": "HeLa"})

    assert len(changes) == 3
    for project in session.scope:
        assert "HeLa" in (project.path / "AM_metadata.yaml").read_text()


def test_an_invalid_edit_writes_nothing_at_all(session):
    """A folder is a batch: stopping half way through would leave six projects
    described and six not, with nothing on screen saying which."""
    session.select_folder("20250428_AM")
    before = [
        (project.path / "AM_metadata.yaml").read_text() for project in session.scope
    ]

    with pytest.raises(MetadataValidationError):
        session.apply({"lamella_type": "not-a-lamella-type"})

    after = [
        (project.path / "AM_metadata.yaml").read_text() for project in session.scope
    ]
    assert after == before


def test_a_source_is_named_only_where_they_all_agree_on_it(session):
    """A value shared across a folder can still have come from a different file
    in each, and naming one would be a claim about provenance that is untrue."""
    session.select(0)
    session.apply({"sample": "HeLa"})
    session.select(1)
    session.apply({"sample": "HeLa"})
    session.select(2)
    session.apply({"sample": "HeLa"})
    session.select_folder("20250428_AM")

    assert session.values()["sample"] == "HeLa"
    assert session.sources()["sample"] == "AM_metadata.yaml"


def test_a_placeholder_in_one_project_is_outstanding_for_the_folder(session):
    """Any, not all: a badge that appeared only once every project agreed would
    go quiet exactly as the folder was nearly finished."""
    session.select(1)
    session.apply({"sample": "TODO"})
    session.select_folder("20250428_AM")

    assert "sample" in session.outstanding()


def test_a_folder_is_summarised_by_count_rather_than_read_out(session):
    session.select_folder("20250428_AM")

    assert session.summary() == "20250428_AM: 3 projects, 0 fully described"


def test_describing_one_project_again_narrows_the_scope_back(session):
    session.select_folder("20250428_AM")
    session.select(1)

    assert session.scope_folder is None
    assert session.project is session.projects[1]
