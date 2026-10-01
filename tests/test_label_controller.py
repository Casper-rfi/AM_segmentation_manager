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

    Big enough that the background is not itself a small component: a frame of
    a few dozen pixels makes everything on it a speck, which is a property of
    the fixture and of nothing real.
    """
    def load(image_path: Path, label_path: Path | None) -> DisplayImage:
        return DisplayImage(
            image=np.zeros((64, 64), dtype=np.uint8),
            label=np.zeros((64, 64), dtype=np.uint8) if label_path else None,
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
    from am_segmentation_manager.state import read_state, write_state

    dataset = build_dataset(tmp_path, acquisitions=2)
    label = dataset.label("SEM", 0)
    write_state(
        read_state(dataset.stage), {f"{label.parent.name}/{label.name}": "corrected"}
    )

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
# Which series is proofread
# ---------------------------------------------------------------------------
#
# Annotating a lamella means annotating one series of it: the reference images
# taken around a milling task are a different job from the Adaptive Polishing
# frames, and a list holding both is not one anybody works through.

@pytest.fixture
def three_series(tmp_path: Path) -> LabelController:
    """A lamella holding reference and alignment images beside its stage."""
    dataset = build_dataset(tmp_path, acquisitions=2)
    image = np.zeros((8, 8), dtype=np.uint8)
    for name in (
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
    ):
        tifffile.imwrite(str(dataset.lamella / name), image)

    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    return controller


def test_the_tab_opens_on_adaptive_polishing(three_series: LabelController):
    assert three_series.series == "adaptive"
    assert three_series.available_series() == ["adaptive", "reference", "alignment"]
    assert three_series.count == 2, "the stage's frames, not the loose ones too"


def test_choosing_a_series_reviews_that_one(three_series: LabelController):
    three_series.set_series("reference")

    assert three_series.count == 1, "one acquisition, both beams of it"
    assert three_series.frame.series == "reference"


def test_the_beam_follows_the_series(three_series: LabelController):
    """An alignment run is imaged with the ion beam alone."""
    three_series.set_series("alignment")

    assert three_series.modality == "FIB"
    assert three_series.count == 1


def test_the_beam_that_was_chosen_comes_back(three_series: LabelController):
    three_series.set_series("alignment")

    three_series.set_series("adaptive")

    assert three_series.modality == "SEM"


def test_a_series_change_starts_at_the_first_frame(three_series: LabelController):
    three_series.step(1)
    assert three_series.position == 1

    three_series.set_series("reference")

    assert three_series.position == 0, "a frame index means nothing across lists"


def test_a_series_change_keeps_the_edits_waiting_to_be_saved(
    three_series: LabelController,
):
    """They belong to the lamella, which has not changed."""
    annotate(three_series)
    assert three_series.pending_count() == 1

    three_series.set_series("reference")

    assert three_series.pending_count() == 1


def test_an_unknown_series_is_refused(three_series: LabelController):
    with pytest.raises(ValueError, match="Unknown series"):
        three_series.set_series("refrence")


def test_a_lamella_without_the_chosen_series_shows_one_it_holds(tmp_path: Path):
    """As in the curation view: the choice is remembered, not overwritten."""
    build_dataset(tmp_path, acquisitions=2)
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    controller.set_series("reference")

    assert controller.series == "adaptive", "all this lamella holds"
    assert controller.available_series() == ["adaptive"]


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

def test_proofreading_opens_on_the_primary_the_project_names(tmp_path: Path):
    lamella_path = tmp_path / "project" / "lam_010"
    for directory in ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "labels"):
        (lamella_path / directory).mkdir(parents=True)
        (lamella_path / directory / "img_0004.tif").write_bytes(b"")
    (tmp_path / "project" / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    (tmp_path / "project" / "AM_families.yaml").write_text(
        "version: 1\nprimary:\n  SEM: SEM_8_fi_1\n", encoding="utf-8"
    )
    projects = Scanner(tmp_path).build_index().all_projects()
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_projects(projects)

    controller.set_lamella(projects[0].lamellae[0])

    assert controller.record.condition == "SEM_8_fi_1"


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
    controller.known_labels.forget(records=[record])

    assert controller.timeline_labels()["SEM"][0] == "empty"


def test_an_annotated_label_is_reported_as_present(controller):
    record = controller.record
    label = record.labels[0]
    tifffile.imwrite(str(label), np.full((8, 8), 2, dtype=np.uint8))
    controller.known_labels.forget(records=[record])

    assert controller.timeline_labels()["SEM"][0] == "present"


def test_a_label_is_read_only_once(controller, monkeypatch):
    reads = []
    import am_segmentation_manager.gui.known_labels as known_labels

    original = known_labels.is_label_file_empty

    def counted(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(known_labels, "is_label_file_empty", counted)
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
# A label rewritten by another program
# ---------------------------------------------------------------------------

def read_ahead_controller(root: Path, *, labels: str = "empty"):
    """A controller on real files, with the frame after this one already read.

    Real files and the real loader, because what is at stake is the moment a
    label's pixels were read against the moment its file changed.
    """
    dataset = build_dataset(
        root, acquisitions=2, modalities=("SEM",), labels=labels,
        label_naming="convention",
    )
    for index in range(2):
        tifffile.imwrite(str(dataset.image("SEM", index)), np.zeros((32, 32), np.uint8))
        if labels != "none":
            tifffile.imwrite(str(dataset.label("SEM", index)), np.zeros((32, 32), np.uint8))
    controller = LabelController(CONFIG)
    controller.set_lamella(scan_lamella(root))
    for request in controller.pending_loads():
        controller.cache.ensure(request)
    controller.working_label()
    return controller, dataset


def rewrite(path: Path) -> None:
    """Write a label with a mark in it, as another program would.

    Compressed, so that the file changes *size*. An uncompressed replacement is
    byte-for-byte as long as what it replaces, and a same-size write in place
    keeps the file's identity -- leaving only the modification time to tell
    them apart, which on Windows moves in steps of about 15 ms and so may not
    have moved at all. :func:`~..io.atomic.fingerprint_of` documents that blind
    spot; a test of what the window does with a rewrite must not sit in it.
    """
    label = np.zeros((32, 32), np.uint8)
    label[4:8, 4:8] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), label, compression="zlib")


def test_a_frame_read_ahead_opens_what_was_written_since(tmp_path: Path):
    controller, dataset = read_ahead_controller(tmp_path)
    rewrite(dataset.label("SEM", 1))

    controller.step(1)
    assert controller.cached_image() is None, "the frame read before the rewrite was kept"
    controller.current_image()

    assert controller.working_label()[4:8, 4:8].all(), (
        "editing started from pixels older than the file"
    )


def test_a_frame_returned_to_opens_what_was_written_since(tmp_path: Path):
    controller, dataset = read_ahead_controller(tmp_path)
    controller.step(1)
    controller.working_label()
    controller.step(-1)
    controller.working_label()  # opening frame 0 puts frame 1 away, unedited
    rewrite(dataset.label("SEM", 1))

    controller.step(1)
    controller.current_image()

    assert controller.working_label()[4:8, 4:8].all()


def test_saving_after_a_rewrite_keeps_what_was_written(tmp_path: Path):
    """The whole path the reviewer takes, ending where the damage would be."""
    from am_segmentation_manager.operations.label_edit import paint

    controller, dataset = read_ahead_controller(tmp_path)
    rewrite(dataset.label("SEM", 1))

    controller.step(1)
    controller.current_image()
    label = controller.working_label()
    controller.record_edit(paint(label, [(20, 20)], 0, 2))
    controller.save()

    assert tifffile.imread(str(dataset.label("SEM", 1)))[4:8, 4:8].all()


def test_a_label_created_after_the_scan_is_not_overwritten(tmp_path: Path):
    """Reading again cannot help, the record naming no label to read: the
    save is what refuses."""
    from am_segmentation_manager.operations.label_edit import paint
    from am_segmentation_manager.operations.label_save import StaleLabels

    controller, dataset = read_ahead_controller(tmp_path, labels="none")
    rewrite(dataset.label("SEM", 1))

    controller.step(1)
    label = controller.working_label()
    controller.record_edit(paint(label, [(20, 20)], 0, 2))

    with pytest.raises(StaleLabels):
        controller.save()
    assert tifffile.imread(str(dataset.label("SEM", 1)))[4:8, 4:8].all()


def test_a_rewrite_after_editing_starts_still_stops_the_save(tmp_path: Path):
    from am_segmentation_manager.operations.label_edit import paint
    from am_segmentation_manager.operations.label_save import StaleLabels

    controller, dataset = read_ahead_controller(tmp_path)
    label = controller.working_label()
    controller.record_edit(paint(label, [(20, 20)], 0, 2))
    rewrite(dataset.label("SEM", 0))

    assert controller.cached_image() is not None, "an edited frame is not read again"
    with pytest.raises(StaleLabels):
        controller.save()


def test_a_frame_saved_twice_is_not_refused_by_its_own_save(tmp_path: Path):
    from am_segmentation_manager.operations.label_edit import paint

    controller, dataset = read_ahead_controller(tmp_path)
    for point in ((20, 20), (22, 22)):
        label = controller.working_label()
        controller.record_edit(paint(label, [point], 0, 2))
        controller.save()

    assert tifffile.imread(str(dataset.label("SEM", 0)))[22, 22] == 2


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

    assert controller.condition is None
    first = controller.condition_choices()[0]
    assert (first.text, first.condition) == ("sem", "sem")
    assert controller.record.variant is None


def test_every_condition_of_the_acquisition_is_offered(tmp_path: Path):
    controller = variant_controller(tmp_path)

    assert [choice.condition for choice in controller.condition_choices()] == [
        "sem", "sem_8_fi_1", "sem_8_fi_4",
    ]


def test_choosing_a_condition_changes_the_image_read(tmp_path: Path):
    controller = variant_controller(tmp_path)
    base = controller.record.image_path

    controller.set_condition("sem_8_fi_4")

    assert controller.record.image_path != base
    assert "8_fi_4" in str(controller.record.image_path)


def test_a_condition_is_proofread_through_the_same_label(tmp_path: Path):
    controller = variant_controller(tmp_path)
    label = controller.current_key

    controller.set_condition("sem_8_fi_1")

    assert controller.current_key == label, (
        "the conditions of one acquisition are one segmentation"
    )


def test_choosing_a_condition_leaves_the_frames_alone(tmp_path: Path):
    controller = variant_controller(tmp_path)
    controller.step(1)

    controller.set_condition("sem_8_fi_1")

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
    controller.set_condition("sem_8_fi_1")

    controller.select(1)

    assert controller.record.variant is None, "blanking the pane helps nobody"


def test_the_loads_asked_for_name_the_condition(tmp_path: Path):
    controller = variant_controller(tmp_path)
    controller.set_condition("sem_8_fi_4")

    assert all(request.condition == "sem_8_fi_4" for request in controller.pending_loads())


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


# ---------------------------------------------------------------------------
# A series with no labels
# ---------------------------------------------------------------------------

def other_lamella(root: Path):
    """Return a lamella holding one image no naming rule reaches."""
    project = root / "AutoLamella-2026-07-28-10-31"
    image = project / "02-ruling-oryx" / "something_new_eb.tif"
    image.parent.mkdir(parents=True)
    tifffile.imwrite(str(image), np.full((64, 96), 100, dtype=np.uint8))
    (project / "experiment.yaml").write_text("name: residue\n", encoding="utf-8")
    return scan_lamella(root)


def test_an_image_with_nowhere_for_a_label_has_no_key(tmp_path: Path):
    """The view asks for a key on every redraw, so it may not raise where the
    write paths refuse: there is no label, and nowhere one could go."""
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(other_lamella(tmp_path))

    assert controller.series == "other"
    assert controller.record is not None
    assert controller.current_key is None
    assert controller.series_is_labelled is False


def test_a_labelled_series_is_labelled(controller):
    assert controller.series_is_labelled is True
    assert controller.current_key is not None


def test_a_series_switched_off_is_not_offered_for_proofreading(three_series):
    three_series.set_shown_series(("adaptive", "alignment"))

    assert "reference" not in three_series.available_series()


def test_switching_off_the_series_being_proofread_moves_to_one_shown(three_series):
    three_series.set_shown_series(("reference",))

    assert three_series.series == "reference"


def test_forgetting_a_label_reaches_every_view_sharing_it(tmp_path: Path):
    """A save forgets what it wrote for the Curate and Export views as well.

    They hold the same :class:`KnownLabels`, so a timeline that read a label as
    missing before it was drawn would otherwise go on saying so after the save.
    """
    from am_segmentation_manager.gui.known_labels import KnownLabels

    build_dataset(tmp_path, acquisitions=2, labels="none", label_naming="convention")
    shared = KnownLabels()
    controller = LabelController(CONFIG, cache=stub_cache(), known_labels=shared)
    controller.set_lamella(scan_lamella(tmp_path))
    record = controller.record
    assert shared.state(record) == "missing", "as another view asked before the save"

    annotate(controller)
    controller.save()

    assert shared.state(record) == "annotated"


def test_a_save_leaves_no_measurement_of_the_label_it_replaced(tmp_path: Path):
    """The Export tab keys measurements by image, which the shared label states do
    not reach, so the window hands it what a save wrote -- as done here."""
    pytest.importorskip("tifffile")
    from am_segmentation_manager.gui.export_controller import ExportController
    from am_segmentation_manager.gui.known_labels import KnownLabels

    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    shared = KnownLabels()
    exporter = ExportController(known_labels=shared)
    exporter.set_projects(Scanner(tmp_path).build_index().all_projects(), tmp_path)
    exporter.load_measurements()
    assert exporter.measurable_labels() == 0, "every label measured before the edit"

    controller = LabelController(CONFIG, cache=stub_cache(), known_labels=shared)
    controller.set_lamella(scan_lamella(tmp_path))
    edited = controller.current_key
    annotate(controller)
    saved = controller.save()
    exporter.forget_labels(saved.written)

    assert saved.written == (edited,)
    assert exporter.measurable_labels() == 1, "the saved label waits to be measured again"


def test_a_save_stopped_partway_settles_what_it_wrote_and_keeps_the_rest(
    tmp_path: Path, monkeypatch
):
    """Left pending, a written label would be refused next time as changed on
    disk -- by this window's own save."""
    pytest.importorskip("tifffile")
    from am_segmentation_manager.gui import label_controller as module
    from am_segmentation_manager.operations.label_save import SaveReport, SaveStopped

    build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    first = controller.current_key
    annotate(controller)
    controller.step(1)
    second = controller.current_key
    annotate(controller)

    def stopped(items, **kwargs):
        raise SaveStopped("disk full", SaveReport(written=(first,)))

    monkeypatch.setattr(module, "save_labels", stopped)
    with pytest.raises(SaveStopped):
        controller.save(mark_corrected=True)

    assert controller.session.edited_keys() == {second}


def test_a_write_that_fails_partway_leaves_the_rest_to_save_again(
    tmp_path: Path, monkeypatch
):
    """The real failure rather than a stub: the second label's write is refused.
    Saving again afterwards must write it, not refuse the first as changed on
    disk by this window's own save."""
    from am_segmentation_manager.operations import label_save
    from am_segmentation_manager.operations.label_save import SaveStopped

    dataset = build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    annotate(controller)
    controller.step(1)
    annotate(controller)
    labels = [dataset.label("SEM", 0), dataset.label("SEM", 1)]
    before = {label: label.read_bytes() for label in labels}

    real = label_save.save_label_image
    calls = []

    def second_write_fails(mask, path, dtype):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        real(mask, path, dtype)

    monkeypatch.setattr(label_save, "save_label_image", second_write_fails)
    with pytest.raises(SaveStopped, match="disk full") as stopped:
        controller.save()

    [written] = stopped.value.report.written
    [unwritten] = [label for label in labels if label != written]
    assert written.read_bytes() != before[written]
    assert unwritten.read_bytes() == before[unwritten]
    assert controller.session.edited_keys() == {unwritten}

    monkeypatch.setattr(label_save, "save_label_image", real)
    saved = controller.save()

    assert saved.written == (unwritten,)
    assert unwritten.read_bytes() != before[unwritten]


def test_labels_saved_but_not_marked_corrected_are_settled_and_say_why(
    tmp_path: Path, monkeypatch
):
    """Marking corrected is a second write. Its failing must not leave the
    labels the first one wrote pending, to be refused as changed on disk."""
    from am_segmentation_manager.gui import label_controller as module
    from am_segmentation_manager.state import StaleState

    dataset = build_dataset(tmp_path, acquisitions=2, labels="annotated")
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    label = controller.current_key
    before = label.read_bytes()
    annotate(controller)

    def curated_elsewhere(*args, **kwargs):
        raise StaleState("AM_state.yaml changed since it was read")

    monkeypatch.setattr(module, "transition_records", curated_elsewhere)
    saved = controller.save(mark_corrected=True)

    assert label.read_bytes() != before
    assert saved.written == (label,)
    assert "changed since it was read" in saved.problem
    assert "not marked corrected" in saved.headline
    assert not controller.session.edited_keys()
    assert read_state(dataset.stage).states == {}


# ---------------------------------------------------------------------------
# Families: proofreading follows the family on screen
# ---------------------------------------------------------------------------

#: The conditions each acquisition of the two-family lamella was imaged under.
TWO_FAMILIES = {
    "img_0004.tif": ("sem", "sem_fi4", "sem-5kV", "sem-5kV_fi4"),
    "img_0005.tif": ("sem", "sem-5kV"),
    "img_0006.tif": ("sem", "sem-5kV"),
}


def two_family_controller(tmp_path: Path, deselect_5kv: tuple[str, ...] = ()) -> LabelController:
    """Return a controller on a lamella of the sem and sem-5kV families.

    The sem-5kV groups of the images named in *deselect_5kv* are deselected.
    """
    from am_segmentation_manager.operations.selection import transition_records

    lamella = tmp_path / "project" / "lam_010"
    for name, conditions in TWO_FAMILIES.items():
        for directory in (*conditions, "sem_labels", "sem-5kV_labels"):
            (lamella / directory).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(lamella / directory / name), np.zeros((8, 8), np.uint8))
    (tmp_path / "project" / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    records = scan_lamella(tmp_path).image_records
    transition_records(
        [
            record for record in records
            if record.condition == "sem-5kV" and record.image_path.name in deselect_5kv
        ],
        "deselected",
    )
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))
    return controller


def test_the_frames_to_proofread_follow_the_family_on_screen(tmp_path: Path):
    controller = two_family_controller(tmp_path, deselect_5kv=("img_0005.tif",))
    assert controller.count == 3

    controller.set_condition("sem-5kV_fi4")

    assert controller.count == 2
    assert controller.record.condition == "sem-5kV_fi4", "stays on the acquisition"
    controller.step(1)
    assert controller.record.image_path.name == "img_0006.tif"
    assert controller.record.condition == "sem-5kV", "the family's own, where fi4 is absent"


def test_a_family_with_nothing_to_proofread_shows_one_that_has(tmp_path: Path):
    """Otherwise no frame would be left, and no control to choose another with."""
    controller = two_family_controller(
        tmp_path, deselect_5kv=("img_0004.tif", "img_0005.tif", "img_0006.tif")
    )

    controller.set_condition("sem-5kV")

    assert controller.count == 3
    assert controller.record.family == "sem"
    assert controller.condition == "sem-5kV", "the choice is kept"


def test_switching_family_keeps_the_other_familys_edits(tmp_path: Path):
    """Edits are held by label, so this is no different from changing frame."""
    controller = two_family_controller(tmp_path)
    annotate(controller)

    controller.set_condition("sem-5kV")
    annotate(controller)

    assert controller.pending_count() == 2
    assert {save.destination.parent.name for save in controller.pending_saves()} == {
        "sem_labels", "sem-5kV_labels",
    }


def test_the_status_names_the_conditions_sharing_the_label(tmp_path: Path):
    controller = two_family_controller(tmp_path)

    controller.set_condition("sem-5kV_fi4")

    assert controller.label_sharers() == ["sem-5kV", "sem-5kV_fi4"]
    assert controller.status_text().endswith("label shared by sem-5kV, sem-5kV_fi4")


def test_one_family_per_beam_names_no_sharers(tmp_path: Path):
    """Every condition shares the label then, as the Variant tooltip says."""
    lamella = tmp_path / "project" / "lam_010"
    for directory in ("SEM_8", "SEM_8_fi_1", "labels"):
        (lamella / directory).mkdir(parents=True)
        tifffile.imwrite(str(lamella / directory / "img_0004.tif"), np.zeros((8, 8), np.uint8))
    (tmp_path / "project" / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    controller = LabelController(CONFIG, cache=stub_cache())
    controller.set_lamella(scan_lamella(tmp_path))

    assert controller.label_sharers() == ["SEM_8", "SEM_8_fi_1"]
    assert "label shared by" not in controller.status_text()
