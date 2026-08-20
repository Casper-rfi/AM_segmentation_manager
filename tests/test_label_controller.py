"""Tests for the proofreading view's model.

No Qt and no display: this is the layer that decides what the Label tab shows,
and all of it is exercised against a scanned project on disk.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from am_segmentation_manager.config import SegmentationConfig
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state
from conftest import build_dataset

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile

    from am_segmentation_manager.display import DisplayImage
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.label_controller import (
        FULL_RESOLUTION,
        LabelController,
        load_full_resolution,
    )


CONFIG = SegmentationConfig(
    class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
    class_labels={0: "background", 1: "GIS", 2: "lamella"},
)


def stub_cache() -> FrameCache:
    """A cache that fabricates frames instead of reading them.

    The controller's job is deciding *which* frame to show; making that
    testable without decoding anything is why the cache takes a loader.
    """
    def load(image_path: Path, label_path: Path | None) -> DisplayImage:
        return DisplayImage(
            image=np.zeros((8, 8), dtype=np.uint8),
            label=np.zeros((8, 8), dtype=np.uint8) if label_path else None,
            label_status="empty" if label_path else "missing",
            source=image_path,
        )

    return FrameCache(load_image=load, image_limit=4)


def scan_lamella(root: Path):
    """Return the first lamella of the project under *root*."""
    projects = Scanner(root).build_index().all_projects()
    return projects[0].lamellae[0]


def deselect(path: Path) -> Path:
    """Move a file into its ``deselected`` folder, as a state change would."""
    destination = path.parent / "deselected" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


@pytest.fixture
def controller(tmp_path: Path) -> LabelController:
    """A controller pointed at a four-acquisition lamella."""
    build_dataset(tmp_path, acquisitions=4)
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    return controller


# ---------------------------------------------------------------------------
# Which frames are visited
# ---------------------------------------------------------------------------

def test_every_selected_frame_is_visitable(controller):
    assert controller.count == 4
    assert controller.position == 0


def test_deselected_frames_are_not_visited(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=4)
    deselect(dataset.image("SEM", 1))
    deselect(dataset.image("SEM", 2))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.count == 2, "a deselected image is one decided against"


def test_corrected_frames_stay_visitable(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=2)
    image = dataset.image("SEM", 0)
    corrected = image.parent / "corrected" / image.name
    corrected.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image), str(corrected))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.count == 2, "finished work must still be reviewable"


def test_a_lamella_with_nothing_selected_reports_it(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=2, modalities=("SEM",))
    for index in range(2):
        deselect(dataset.image("SEM", index))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.count == 0
    assert controller.frame is None
    assert "no SEM images to proofread" in controller.status_text()


def test_nothing_is_shown_before_a_lamella_is_chosen():
    controller = LabelController(CONFIG, cache=stub_cache())

    assert controller.frame is None
    assert controller.count == 0
    assert controller.status_text() == "No lamella selected"


# ---------------------------------------------------------------------------
# Moving about
# ---------------------------------------------------------------------------

def test_stepping_moves_one_frame(controller):
    controller.step(1)
    assert controller.position == 1


def test_stepping_stops_at_either_end(controller):
    controller.step(-1)
    assert controller.position == 0

    controller.select(99)
    assert controller.position == controller.count - 1


def test_positions_skip_over_a_deselected_frame(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=4)
    deselect(dataset.image("SEM", 1))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    first = controller.record.image_path
    controller.step(1)

    assert controller.count == 3
    assert controller.record.image_path != first
    assert "img_001" not in controller.record.image_path.name


# ---------------------------------------------------------------------------
# Modality
# ---------------------------------------------------------------------------

def test_both_modalities_are_offered_when_both_are_present(controller):
    assert controller.available_modalities() == ["SEM", "FIB"]


def test_a_modality_with_nothing_to_proofread_is_not_offered(tmp_path: Path):
    build_dataset(tmp_path, acquisitions=2, modalities=("SEM",))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.available_modalities() == ["SEM"]


def test_switching_modality_stays_on_the_same_acquisition(controller):
    controller.select(2)
    frame = controller.frame_index

    controller.set_modality("FIB")

    assert controller.modality == "FIB"
    assert controller.frame_index == frame, "the beam changed, the moment should not"


def test_an_unknown_modality_is_rejected(controller):
    with pytest.raises(ValueError, match="Unknown modality"):
        controller.set_modality("EBSD")


# ---------------------------------------------------------------------------
# Re-pointing at a lamella
# ---------------------------------------------------------------------------

def test_returning_to_the_same_lamella_keeps_the_position(controller, tmp_path: Path):
    controller.select(2)

    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.position == 2, "the tab was re-entered, not moved"


def test_moving_to_another_lamella_starts_at_its_first_frame(tmp_path: Path):
    build_dataset(tmp_path, acquisitions=3, lamellae=2)
    projects = Scanner(tmp_path).build_index().all_projects()
    first, second = projects[0].lamellae[:2]

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(first)
    controller.select(2)
    controller.set_lamella(second)

    assert controller.position == 0


def test_re_pointing_picks_up_a_state_change_made_since(controller, tmp_path: Path):
    dataset_image = controller.record.image_path
    deselect(dataset_image)

    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.count == 3, "the rebuilt frames must reflect the move"


# ---------------------------------------------------------------------------
# Preparing frames
# ---------------------------------------------------------------------------

def test_the_current_frame_is_requested_first(controller):
    controller.select(2)
    requests = controller.pending_loads()

    assert requests[0].frame_index == controller.frame_index
    assert all(request.modality == "SEM" for request in requests)


def test_prepared_frames_are_not_requested_again(controller):
    controller.current_image()

    requests = controller.pending_loads()

    assert all(
        request.frame_index != controller.frame_index for request in requests
    )


def test_read_ahead_covers_only_the_visitable_frames(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=4)
    deselect(dataset.image("SEM", 1))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    requested = {request.frame_index for request in controller.pending_loads(radius=None)}

    assert len(requested) == 3


def test_nothing_is_requested_for_an_empty_lamella():
    controller = LabelController(CONFIG, cache=stub_cache())

    assert controller.pending_loads() == []


def test_frames_are_read_at_full_resolution(tmp_path: Path):
    """The premise of the whole tab: no reduction on the way in."""
    dataset = build_dataset(tmp_path, acquisitions=1, modalities=("SEM",))
    image_path = dataset.image("SEM", 0)
    label_path = dataset.label("SEM", 0)
    # Larger than the curation view's 1024 px limit, so a reduction would show.
    tifffile.imwrite(str(image_path), np.zeros((1200, 1600), dtype=np.uint8))
    tifffile.imwrite(str(label_path), np.zeros((1200, 1600), dtype=np.uint8))

    prepared = load_full_resolution(image_path, label_path)

    assert FULL_RESOLUTION == 0
    assert prepared.image.shape == (1200, 1600)
    assert prepared.scale == 1.0
    assert prepared.label.shape == (1200, 1600)


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------

def test_the_timeline_covers_the_visitable_frames_only(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=4)
    deselect(dataset.image("SEM", 0))

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    assert len(controller.timeline_states()["SEM"]) == 3
    assert len(controller.timeline_labels()["SEM"]) == 3


def test_the_timeline_reports_only_the_modality_on_screen(controller):
    assert set(controller.timeline_states()) == {"SEM"}

    controller.set_modality("FIB")

    assert set(controller.timeline_states()) == {"FIB"}


def test_an_empty_label_is_reported_as_a_placeholder(controller):
    record = controller.record
    tifffile.imwrite(str(record.labels[0]), np.zeros((8, 8), dtype=np.uint8))
    controller.forget_label_state(record)

    assert controller.timeline_labels()["SEM"][0] == "empty"


def test_an_annotated_label_is_reported_as_present(controller):
    record = controller.record
    label = record.labels[0]
    tifffile.imwrite(str(label), np.full((8, 8), 2, dtype=np.uint8))
    controller.forget_label_state(record)

    assert controller.timeline_labels()["SEM"][0] == "present"


def test_a_label_is_read_only_once(controller, monkeypatch):
    reads = []
    import am_segmentation_manager.operations.label_io as label_io

    original = label_io.label_state

    def counted(record):
        reads.append(record.image_path)
        return original(record)

    monkeypatch.setattr(
        "am_segmentation_manager.gui.label_controller.label_io.label_state", counted
    )
    controller.timeline_labels()
    first = len(reads)
    controller.timeline_labels()

    assert first > 0
    assert len(reads) == first, "label states must be cached across refreshes"


# ---------------------------------------------------------------------------
# The status line
# ---------------------------------------------------------------------------

def test_the_status_line_says_where_the_reviewer_is(controller):
    text = controller.status_text()

    assert "SEM 1 / 4" in text
    assert "selected" in text


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def unlabelled_controller(root: Path):
    """A controller on a lamella whose images have no label file yet.

    Named by the current convention, because with no label files on disk there
    is no legacy directory for the layout to resolve to.
    """
    dataset = build_dataset(
        root, acquisitions=2, labels="none", label_naming="convention"
    )
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(root))
    return controller, dataset


def annotate(controller) -> None:
    """Paint one pixel on the frame the controller is showing."""
    from am_segmentation_manager.operations.label_edit import paint

    controller.current_image()  # the working label comes from the prepared frame
    label = controller.working_label()
    controller.record_edit(paint(label, [(1, 1)], 0, 2))


def test_saving_names_a_created_label_on_its_records(tmp_path: Path):
    controller, dataset = unlabelled_controller(tmp_path)
    record = controller.record
    assert not record.labels, "the scan found no label to start from"

    annotate(controller)
    controller.save()

    assert record.labels == [dataset.label("SEM", 0)], (
        "a record that does not name its label cannot have it moved with the image"
    )


def test_marking_corrected_declares_a_label_created_while_proofreading(tmp_path: Path):
    controller, dataset = unlabelled_controller(tmp_path)
    image, label = dataset.image("SEM", 0), dataset.label("SEM", 0)

    annotate(controller)
    controller.save(mark_corrected=True)

    assert read_state(dataset.stage).states == {
        f"{label.parent.name}/{label.name}": "corrected"
    }
    assert image.exists() and label.exists(), (
        "declaring a state must not disturb the proofread work"
    )


def test_marking_corrected_declares_an_existing_label(tmp_path: Path):
    dataset = build_dataset(tmp_path, acquisitions=2)
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    image = controller.record.image_path
    label = controller.current_key

    annotate(controller)
    controller.save(mark_corrected=True)

    assert read_state(dataset.stage).states[f"{label.parent.name}/{label.name}"] == "corrected"
    assert image.exists() and label.exists()


def test_a_corrected_label_is_still_looked_up_where_it_was_written(tmp_path: Path):
    """Nothing moves, so a destination worked out before the change stays right."""
    controller, dataset = unlabelled_controller(tmp_path)

    annotate(controller)
    controller.save(mark_corrected=True)

    assert controller.key_for(controller.record) == dataset.label("SEM", 0)


# ---------------------------------------------------------------------------
# Imaging conditions
# ---------------------------------------------------------------------------

def variant_controller(root: Path):
    """A controller on a lamella whose SEM images have two extra conditions."""
    build_dataset(
        root, acquisitions=2, modalities=("SEM",),
        sem_variants=("sem_8_fi_1", "sem_8_fi_4"),
    )
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(root))
    return controller


def test_the_base_image_is_what_opens(tmp_path: Path):
    controller = variant_controller(tmp_path)

    assert controller.variant is None
    assert controller.available_variants()[0] is None


def test_every_condition_of_the_acquisition_is_offered(tmp_path: Path):
    controller = variant_controller(tmp_path)

    assert set(controller.available_variants()) == {None, "8_fi_1", "8_fi_4"}


def test_choosing_a_condition_changes_the_image_read(tmp_path: Path):
    controller = variant_controller(tmp_path)
    base = controller.record.image_path

    controller.set_variant("8_fi_4")

    assert controller.record.image_path != base
    assert "8_fi_4" in str(controller.record.image_path)


def test_a_condition_is_proofread_through_the_same_label(tmp_path: Path):
    controller = variant_controller(tmp_path)
    label = controller.current_key

    controller.set_variant("8_fi_1")

    assert controller.current_key == label, (
        "the conditions of one acquisition are one segmentation"
    )


def test_choosing_a_condition_leaves_the_frames_alone(tmp_path: Path):
    controller = variant_controller(tmp_path)
    controller.step(1)

    controller.set_variant("8_fi_1")

    assert controller.count == 2, "state follows the label, not the condition"
    assert controller.position == 1, "and the reviewer stays where they were"


def test_a_frame_without_the_condition_shows_its_base_image(tmp_path: Path):
    dataset = build_dataset(
        tmp_path, acquisitions=2, modalities=("SEM",), sem_variants=("sem_8_fi_1",),
    )
    # Only the first acquisition was captured under the condition.
    # Keyed by directory name here; the scanner parses that to the variant.
    dataset.image("SEM", 1, variant="sem_8_fi_1").unlink()
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    controller.set_variant("8_fi_1")

    controller.select(1)

    assert controller.record.variant is None, "blanking the pane helps nobody"


def test_the_loads_asked_for_name_the_condition(tmp_path: Path):
    controller = variant_controller(tmp_path)
    controller.set_variant("8_fi_4")

    assert all(request.variant == "8_fi_4" for request in controller.pending_loads())


# ---------------------------------------------------------------------------
# Specks
# ---------------------------------------------------------------------------

def paint_speck(controller, value: int = 2) -> None:
    """Put a single stray pixel on the label on screen."""
    from am_segmentation_manager.operations.label_edit import paint

    controller.current_image()
    label = controller.working_label()
    controller.record_edit(paint(label, [(3, 3)], 0, value))


def test_a_speck_painted_by_hand_is_found(controller):
    assert controller.specks() == ()

    paint_speck(controller)

    specks = controller.specks()
    assert [(speck.pixels, speck.x, speck.y) for speck in specks] == [(1, 3, 3)], (
        "measured on the array being edited, not on the file"
    )


def test_a_speck_painted_out_stops_being_reported(controller):
    from am_segmentation_manager.operations.label_edit import paint

    paint_speck(controller)
    assert controller.specks()

    label = controller.working_label()
    controller.record_edit(paint(label, [(3, 3)], 0, 0))

    assert controller.specks() == (), "an edit drops what was counted before it"


def test_the_count_is_held_until_something_changes_it(controller, monkeypatch):
    calls = []
    import am_segmentation_manager.gui.label_controller as module

    original = module.find_specks
    monkeypatch.setattr(
        module, "find_specks",
        lambda label, **kwargs: calls.append(1) or original(label, **kwargs),
    )
    paint_speck(controller)

    controller.specks()
    controller.specks()

    assert len(calls) == 1, "stepping back and forth must not re-measure"


def test_changing_the_size_measures_again(controller):
    paint_speck(controller)
    assert controller.specks()

    controller.set_speck_size(1)
    assert controller.specks(), "a one-pixel speck is still a speck at one pixel"

    controller.set_speck_size(64)
    assert controller.specks()


def test_the_size_is_at_least_one_pixel(controller):
    controller.set_speck_size(0)

    assert controller.speck_size == 1
