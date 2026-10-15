"""Tests for the labelling guide: the steps, and the Label tab's model of them.

No Qt. The labelling guide is held in memory only, so nothing here asserts on a file
except where a save it feeds writes one.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from am_segmentation_manager.config import SegmentationConfig
from am_segmentation_manager.gui.labelling_guide import (
    EXAMPLE_FILES,
    EXAMPLES,
    STEPS,
    LabellingGuide,
    read_example,
)
from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.state import read_state
from conftest import build_dataset

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

FIRST = Path("lamella/sem_labels/image_000.tif")
SECOND = Path("lamella/sem_labels/image_001.tif")


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

def test_every_step_says_what_to_do_and_how():
    for step in STEPS:
        assert step.title and step.instructions and step.explanation, step.title


def test_the_last_step_is_saving():
    assert "Save" in STEPS[-1].title
    assert LabellingGuide().save_step == len(STEPS) - 1


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def test_a_label_starts_on_the_first_step():
    guide = LabellingGuide()

    assert guide.current(FIRST) == 0
    assert not guide.is_started(FIRST)


def test_advancing_ticks_the_steps_in_order():
    guide = LabellingGuide()

    assert guide.advance(FIRST)
    assert guide.advance(FIRST)

    assert guide.ticked(FIRST) == {0, 1}
    assert guide.current(FIRST) == 2


def test_advancing_stops_at_the_save_which_a_save_ticks():
    guide = LabellingGuide()
    for _ in range(guide.save_step):
        assert guide.advance(FIRST)

    assert not guide.advance(FIRST), "ticking the save by hand would claim it was saved"
    assert guide.current(FIRST) == guide.save_step
    assert guide.is_checked(FIRST)


def test_a_step_ticked_out_of_order_is_skipped_by_advancing():
    guide = LabellingGuide()
    guide.set_ticked(FIRST, 1, True)

    guide.advance(FIRST)

    assert guide.ticked(FIRST) == {0, 1}
    assert guide.current(FIRST) == 2


def test_going_back_unticks_the_step_before_the_current_one():
    guide = LabellingGuide()
    guide.advance(FIRST)
    guide.advance(FIRST)

    assert guide.retreat(FIRST)

    assert guide.ticked(FIRST) == {0}
    assert guide.current(FIRST) == 1


def test_going_back_from_the_first_step_does_nothing():
    assert not LabellingGuide().retreat(FIRST)


def test_each_label_has_its_own_progress():
    guide = LabellingGuide()
    guide.advance(FIRST)

    assert guide.ticked(SECOND) == frozenset()
    assert guide.current(SECOND) == 0


def test_a_label_is_checked_only_with_every_step_before_the_save_ticked():
    guide = LabellingGuide()
    for step in range(guide.save_step - 1):
        guide.set_ticked(FIRST, step, True)

    assert not guide.is_checked(FIRST)
    guide.set_ticked(FIRST, guide.save_step - 1, True)
    assert guide.is_checked(FIRST)


def test_a_save_ticks_the_save_step():
    guide = LabellingGuide()

    guide.mark_saved([FIRST])

    assert guide.ticked(FIRST) == {guide.save_step}


def test_forgetting_a_label_unticks_everything_for_it_alone():
    guide = LabellingGuide()
    guide.advance(FIRST)
    guide.advance(SECOND)

    guide.forget([FIRST])

    assert not guide.is_started(FIRST)
    assert guide.ticked(SECOND) == {0}


def test_there_is_no_step_past_the_last():
    with pytest.raises(IndexError):
        LabellingGuide().set_ticked(FIRST, len(STEPS), True)


# ---------------------------------------------------------------------------
# The Label tab's model
# ---------------------------------------------------------------------------

image_stack = pytest.mark.skipif(not HAS_IMAGE_STACK, reason="Optional image libs not installed")

CONFIG = SegmentationConfig(
    class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
    class_labels={0: "background", 1: "GIS", 2: "lamella"},
)


def _stub_cache():
    """A cache that fabricates frames, as the controller's own tests use."""
    import numpy as np

    from am_segmentation_manager.display import DisplayImage
    from am_segmentation_manager.gui.cache import FrameCache

    def load(image_path: Path, label_path: Path | None) -> DisplayImage:
        return DisplayImage(
            image=np.zeros((64, 64), dtype=np.uint8),
            label=np.zeros((64, 64), dtype=np.uint8) if label_path else None,
            label_status="empty" if label_path else "missing",
            source=image_path,
        )

    return FrameCache(load_image=load, image_limit=4)


@pytest.fixture
def proofread(tmp_path: Path):
    """A controller on a two-acquisition lamella, its first frame read."""
    from am_segmentation_manager.gui.label_controller import LabelController

    dataset = build_dataset(tmp_path, acquisitions=2)
    controller = LabelController(CONFIG, cache=_stub_cache())
    controller.set_lamella(Scanner(tmp_path).build_index().all_projects()[0].lamellae[0])
    controller.current_image()
    return controller, dataset


def _tick_through(controller) -> None:
    while controller.labelling_guide.current(controller.guide_key) < controller.labelling_guide.save_step:
        controller.advance_guide()


def _paint(controller) -> None:
    from am_segmentation_manager.operations.label_edit import paint

    controller.record_edit(paint(controller.working_label(), [(1, 1)], 0, 2))


@image_stack
def test_the_labelling_guide_is_about_the_label_on_screen(proofread):
    controller, _dataset = proofread

    controller.advance_guide()

    assert controller.labelling_guide.ticked(controller.current_key) == {0}


@image_stack
def test_advancing_to_the_save_says_saving_is_next(proofread):
    controller, _dataset = proofread
    for _ in range(controller.labelling_guide.save_step - 1):
        assert not controller.advance_guide()

    assert not controller.advance_guide(), "that tick was the last before the save"
    assert controller.advance_guide(), "now the only step left is saving"


@image_stack
def test_a_label_ticked_through_is_marked_corrected_without_an_edit(proofread):
    controller, dataset = proofread
    label = controller.current_key
    _tick_through(controller)

    assert controller.pending_count() == 0, "nothing was painted"
    controller.save(mark_corrected=True)

    assert read_state(dataset.stage).states[f"{label.parent.name}/{label.name}"] == "corrected"
    assert controller.labelling_guide.save_step in controller.labelling_guide.ticked(label)


@image_stack
def test_a_label_ticked_part_way_is_not_marked_corrected(proofread):
    controller, dataset = proofread
    controller.advance_guide()

    controller.save(mark_corrected=True)

    assert read_state(dataset.stage).states == {}


@image_stack
def test_a_corrected_label_is_not_offered_for_marking_again(proofread):
    controller, _dataset = proofread
    _tick_through(controller)
    controller.save(mark_corrected=True)

    assert controller.checked_keys() == set()


@image_stack
def test_saving_without_marking_leaves_the_save_step_unticked(proofread):
    controller, _dataset = proofread
    _tick_through(controller)
    _paint(controller)

    controller.save(mark_corrected=False)

    assert controller.labelling_guide.save_step not in controller.labelling_guide.ticked(
        controller.current_key
    )


@image_stack
def test_progress_outlives_moving_to_another_frame(proofread):
    controller, _dataset = proofread
    first = controller.current_key
    controller.advance_guide()

    controller.step(1)
    controller.current_image()
    controller.step(-1)

    assert controller.labelling_guide.ticked(first) == {0}


@image_stack
def test_discarding_edits_forgets_the_labelling_guide_of_the_labels_edited(proofread):
    controller, _dataset = proofread
    edited = controller.current_key
    controller.advance_guide()
    _paint(controller)
    controller.step(1)
    controller.current_image()
    untouched = controller.current_key
    controller.advance_guide()

    controller.discard_edits()

    assert not controller.labelling_guide.is_started(edited), "the steps ticked were the work discarded"
    assert controller.labelling_guide.ticked(untouched) == {0}


@image_stack
def test_an_unfinished_labelling_guide_is_counted_only_for_a_label_being_saved(proofread):
    controller, _dataset = proofread
    controller.advance_guide()
    assert controller.unfinished_guides() == 0, "nothing of this label is being saved"

    _paint(controller)

    assert controller.unfinished_guides() == 1


@image_stack
def test_a_frame_that_cannot_be_painted_has_no_labelling_guide(tmp_path: Path):
    from am_segmentation_manager.gui.label_controller import LabelController

    build_dataset(tmp_path, acquisitions=1)
    controller = LabelController(CONFIG, cache=_stub_cache())
    controller.set_lamella(Scanner(tmp_path).build_index().all_projects()[0].lamellae[0])

    assert controller.guide_key is None, "the frame has not been read yet"
    assert not controller.advance_guide()


@image_stack
def test_marking_an_unchanged_label_corrected_is_recorded(tmp_path: Path):
    """Nothing is written but the state, and the record must still start first."""
    from am_segmentation_manager.gui.label_controller import LabelController

    build_dataset(tmp_path, acquisitions=2)
    projects = Scanner(tmp_path).build_index().all_projects()
    controller = LabelController(CONFIG, cache=_stub_cache())
    controller.set_projects(projects)
    controller.set_lamella(projects[0].lamellae[0])
    controller.current_image()
    _tick_through(controller)

    controller.save(mark_corrected=True)

    log = projects[0].path / "AM_history.jsonl"
    assert '"kind":"state-changed"' in log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The examples
# ---------------------------------------------------------------------------

def _write_example(folder: Path, shapes=((4, 6), (4, 6), (4, 6))) -> None:
    import cv2
    import numpy as np

    folder.mkdir(parents=True)
    for name, shape in zip(EXAMPLE_FILES, shapes):
        cv2.imwrite(str(folder / name), np.zeros(shape, dtype=np.uint8))


@image_stack
def test_an_example_is_its_image_and_both_labels(tmp_path: Path):
    _write_example(tmp_path / "step")

    example = read_example("step", root=tmp_path)

    assert example is not None
    assert example.image.shape == example.before.shape == example.after.shape == (4, 6)


def test_a_step_without_an_example_has_none(tmp_path: Path):
    assert read_example("", root=tmp_path) is None
    assert read_example("not_there", root=tmp_path) is None


@image_stack
def test_an_example_missing_a_file_has_none(tmp_path: Path):
    _write_example(tmp_path / "step")
    (tmp_path / "step" / "after.png").unlink()

    assert read_example("step", root=tmp_path) is None


@image_stack
def test_an_example_cropped_unalike_is_refused(tmp_path: Path):
    _write_example(tmp_path / "step", shapes=((4, 6), (4, 6), (5, 6)))

    with pytest.raises(ValueError, match="cropped alike"):
        read_example("step", root=tmp_path)


@image_stack
def test_every_example_a_step_names_is_shipped():
    named = [step.example for step in STEPS if step.example]

    assert named, "at least one step has its example"
    assert [name for name in named if read_example(name) is None] == []


def test_every_example_shipped_is_named_by_a_step():
    named = {step.example for step in STEPS}

    assert sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir()
                  and path.name not in named) == []


@image_stack
def test_a_label_ticked_through_with_no_annotation_is_counted_before_marking(tmp_path: Path):
    """The claim Curate asks about before making it, reached here by ticking."""
    from am_segmentation_manager.gui.label_controller import LabelController

    build_dataset(tmp_path, acquisitions=1, labels="none")
    controller = LabelController(CONFIG, cache=_stub_cache())
    controller.set_lamella(Scanner(tmp_path).build_index().all_projects()[0].lamellae[0])
    controller.current_image()
    _tick_through(controller)

    assert controller.unannotated_checked() == 1


@image_stack
def test_a_label_painted_on_is_not_counted_as_unannotated(tmp_path: Path):
    from am_segmentation_manager.gui.label_controller import LabelController

    build_dataset(tmp_path, acquisitions=1, labels="none")
    controller = LabelController(CONFIG, cache=_stub_cache())
    controller.set_lamella(Scanner(tmp_path).build_index().all_projects()[0].lamellae[0])
    controller.current_image()
    _tick_through(controller)
    _paint(controller)

    assert controller.unannotated_checked() == 0, "what is written is what will be marked"
