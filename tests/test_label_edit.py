"""Tests for the label drawing tools.

Everything here runs on small synthetic arrays: no display, no GUI toolkit and
no filesystem. The tools are the only part of proofreading that can silently
destroy an annotation, so the emphasis is on what they must leave alone.
"""
from __future__ import annotations

import importlib.util

import pytest

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np

    from am_segmentation_manager.operations.label_edit import (
        UndoStack,
        fill,
        new_label,
        paint,
        patch_between,
    )


@pytest.fixture
def label():
    """A 20x30 background-only label."""
    return new_label((20, 30))


def patched(label, patch):
    """Return the region of *label* the patch covers."""
    top, left, height, width = patch.bbox
    return label[top : top + height, left : left + width]


# ---------------------------------------------------------------------------
# Brush
# ---------------------------------------------------------------------------

def test_a_dab_paints_a_disc_of_the_right_radius(label):
    paint(label, [(10, 8)], radius=3, value=2)

    # Centre, and the four points exactly on the rim, are painted.
    assert label[8, 10] == 2
    for y, x in ((5, 10), (11, 10), (8, 7), (8, 13)):
        assert label[y, x] == 2
    # One pixel beyond the rim is not.
    for y, x in ((4, 10), (12, 10), (8, 6), (8, 14)):
        assert label[y, x] == 0


def test_a_dab_changes_nothing_outside_its_patch(label):
    before = label.copy()
    patch = paint(label, [(10, 8)], radius=3, value=2)

    top, left, height, width = patch.bbox
    outside = np.ones(label.shape, dtype=bool)
    outside[top : top + height, left : left + width] = False
    np.testing.assert_array_equal(label[outside], before[outside])


def test_a_zero_radius_stroke_paints_a_single_pixel(label):
    paint(label, [(10, 8)], radius=0, value=2)

    assert label[8, 10] == 2
    assert int((label != 0).sum()) == 1


def test_a_drag_leaves_a_continuous_line_not_separate_dabs(label):
    # Two points far enough apart that discs alone would leave a gap.
    paint(label, [(4, 10), (24, 10)], radius=1, value=3)

    painted = label[10, 4:25]
    assert (painted == 3).all(), "the stroke is broken between its endpoints"


def test_a_stroke_wholly_outside_the_label_changes_nothing(label):
    before = label.copy()

    assert paint(label, [(200, 200), (250, 210)], radius=2, value=1) is None
    np.testing.assert_array_equal(label, before)


def test_a_stroke_repainting_its_own_class_is_not_an_edit(label):
    paint(label, [(10, 8)], radius=3, value=2)

    assert paint(label, [(10, 8)], radius=3, value=2) is None


def test_an_empty_stroke_changes_nothing(label):
    assert paint(label, [], radius=2, value=1) is None


def test_a_negative_radius_is_rejected(label):
    with pytest.raises(ValueError, match="radius"):
        paint(label, [(1, 1)], radius=-1, value=1)


# ---------------------------------------------------------------------------
# Locked classes
# ---------------------------------------------------------------------------

def test_a_locked_class_survives_a_stroke_drawn_across_it(label):
    label[:, 15] = 4  # a vertical line of the class being protected

    paint(label, [(5, 10), (25, 10)], radius=1, value=2, protected={4})

    assert (label[:, 15] == 4).all(), "the locked class was painted over"
    assert label[10, 5] == 2 and label[10, 25] == 2, "the rest of the stroke was lost"


def test_locking_one_class_does_not_protect_another(label):
    label[10, 12] = 1
    label[10, 13] = 4

    paint(label, [(12, 10), (13, 10)], radius=0, value=2, protected={4})

    assert label[10, 12] == 2
    assert label[10, 13] == 4


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def test_a_fill_stops_at_a_boundary_of_another_class(label):
    label[:, 15] = 1  # split the frame in two

    fill(label, (5, 10), value=2)

    assert (label[:, :15] == 2).all(), "the fill did not cover its own side"
    assert (label[:, 15] == 1).all(), "the fill crossed the boundary"
    assert (label[:, 16:] == 0).all(), "the fill leaked past the boundary"


def test_a_fill_does_not_leak_through_a_diagonal_boundary(label):
    # A one-pixel diagonal wall: 8-connectivity would slip between its pixels.
    for offset in range(20):
        label[offset, offset] = 1

    # Seeded below the diagonal, in the lower-left triangle.
    fill(label, (0, 19), value=2)

    assert label[19, 0] == 2, "the fill did not cover its own side"
    assert label[0, 29] == 0, "the fill leaked through the diagonal"


def test_a_fill_seeded_on_a_locked_class_changes_nothing(label):
    label[:, 15] = 4
    before = label.copy()

    assert fill(label, (15, 10), value=2, protected={4}) is None
    np.testing.assert_array_equal(label, before)


def test_a_fill_seeded_outside_the_label_changes_nothing(label):
    before = label.copy()

    assert fill(label, (99, 99), value=2) is None
    np.testing.assert_array_equal(label, before)


def test_a_fill_onto_the_same_class_is_not_an_edit(label):
    assert fill(label, (5, 5), value=0) is None


def test_an_unknown_connectivity_is_rejected(label):
    with pytest.raises(ValueError, match="Connectivity"):
        fill(label, (5, 5), value=1, connectivity=6)


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def test_undo_restores_the_array_exactly(label):
    label[10, 12] = 1
    before = label.copy()
    stack = UndoStack()

    stack.push(paint(label, [(10, 10), (20, 10)], radius=2, value=3))
    assert not np.array_equal(label, before)

    stack.undo(label)
    np.testing.assert_array_equal(label, before)


def test_undo_then_redo_returns_to_the_edited_state(label):
    stack = UndoStack()
    stack.push(fill(label, (5, 5), value=2))
    edited = label.copy()

    stack.undo(label)
    stack.redo(label)

    np.testing.assert_array_equal(label, edited)


def test_edits_are_undone_most_recent_first(label):
    stack = UndoStack()
    stack.push(paint(label, [(5, 5)], radius=0, value=1))
    after_first = label.copy()
    stack.push(paint(label, [(6, 5)], radius=0, value=2))

    stack.undo(label)

    np.testing.assert_array_equal(label, after_first)
    assert stack.can_undo


def test_a_new_edit_discards_the_undone_ones(label):
    stack = UndoStack()
    stack.push(paint(label, [(5, 5)], radius=0, value=1))
    stack.undo(label)
    assert stack.can_redo

    stack.push(paint(label, [(6, 5)], radius=0, value=2))

    assert not stack.can_redo


def test_a_no_op_edit_is_not_pushed(label):
    stack = UndoStack()

    stack.push(paint(label, [], radius=1, value=1))

    assert not stack.can_undo


def test_the_oldest_edit_is_forgotten_past_the_limit(label):
    stack = UndoStack(max_entries=2)
    for x in range(3):
        stack.push(paint(label, [(x, 0)], radius=0, value=1))

    assert len(stack) == 2
    for _ in range(2):
        stack.undo(label)

    assert not stack.can_undo
    assert label[0, 0] == 1, "the forgotten edit was rolled back anyway"


def test_undoing_an_empty_stack_is_harmless(label):
    before = label.copy()
    stack = UndoStack()

    assert stack.undo(label) is None
    assert stack.redo(label) is None
    np.testing.assert_array_equal(label, before)


def test_a_stack_must_hold_at_least_one_edit():
    with pytest.raises(ValueError, match="max_entries"):
        UndoStack(max_entries=0)


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_every_operation_preserves_the_label_dtype(dtype):
    label = new_label((20, 30), dtype=np.dtype(dtype))
    stack = UndoStack()

    stack.push(paint(label, [(5, 5), (10, 10)], radius=2, value=3))
    stack.push(fill(label, (0, 0), value=1))
    stack.undo(label)

    assert label.dtype == np.dtype(dtype)


def test_a_class_too_large_for_the_dtype_is_rejected():
    label = new_label((10, 10))

    with pytest.raises(ValueError, match="uint8"):
        paint(label, [(5, 5)], radius=1, value=300)
    with pytest.raises(ValueError, match="uint8"):
        fill(label, (5, 5), value=300)


def test_a_uint16_label_accepts_a_class_a_uint8_one_could_not():
    label = new_label((10, 10), dtype=np.uint16)

    paint(label, [(5, 5)], radius=0, value=300)

    assert label[5, 5] == 300


# ---------------------------------------------------------------------------
# New labels
# ---------------------------------------------------------------------------

def test_a_new_label_is_all_background_at_the_image_shape():
    label = new_label((480, 640))

    assert label.shape == (480, 640)
    assert label.dtype == np.uint8
    assert not label.any()


def test_a_new_label_takes_only_the_first_two_axes_of_a_shape():
    assert new_label((480, 640, 3)).shape == (480, 640)


# ---------------------------------------------------------------------------
# Reducing a stroke to one edit
# ---------------------------------------------------------------------------

def test_a_stroke_drawn_in_segments_becomes_one_patch(label):
    """How the canvas turns a drag into a single undo step."""
    before = label.copy()
    for start, end in (((4, 4), (10, 4)), ((10, 4), (16, 4))):
        paint(label, [start, end], radius=1, value=2)

    patch = patch_between(before, label)

    assert patch is not None
    top, left, height, width = patch.bbox
    assert (top, left) == (3, 3)
    assert width == 16 - 4 + 3, "the patch should span the whole stroke"


def test_reversing_a_stroke_patch_restores_the_array(label):
    before = label.copy()
    paint(label, [(4, 4), (16, 9)], radius=3, value=2)
    stack = UndoStack()

    stack.push(patch_between(before, label))
    stack.undo(label)

    np.testing.assert_array_equal(label, before)


def test_an_unchanged_array_yields_no_patch(label):
    assert patch_between(label.copy(), label) is None


def test_comparing_differently_shaped_labels_is_rejected(label):
    with pytest.raises(ValueError, match="Cannot compare"):
        patch_between(new_label((5, 5)), label)
