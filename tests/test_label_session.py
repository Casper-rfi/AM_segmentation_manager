"""Tests for the edits the Label tab holds before they are saved.

No Qt and no filesystem: the session is where unsaved proofreading lives, and
losing an edit here is the worst thing this feature can do, so the emphasis is
on what survives navigating away and back.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

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

    from am_segmentation_manager.gui.label_session import (
        LabelSession,
        decode_label,
        encode_label,
    )
    from am_segmentation_manager.operations.label_edit import new_label, paint

FIRST = Path("/labels/one.tif")
SECOND = Path("/labels/two.tif")


@pytest.fixture
def session():
    return LabelSession()


@pytest.fixture
def blank():
    return new_label((20, 30))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_a_label_survives_being_encoded_and_decoded(dtype):
    label = new_label((32, 48), dtype=np.dtype(dtype))
    label[4:12, 6:20] = 3

    restored = decode_label(encode_label(label))

    np.testing.assert_array_equal(restored, label)
    assert restored.dtype == label.dtype


def test_encoding_a_label_is_far_smaller_than_holding_it():
    label = new_label((512, 512))
    label[100:400, 100:400] = 2

    assert len(encode_label(label)) < label.nbytes // 10


# ---------------------------------------------------------------------------
# Opening and painting
# ---------------------------------------------------------------------------

def test_painting_does_not_write_through_to_the_source(session, blank):
    working = session.open(FIRST, blank)

    paint(working, [(5, 5)], radius=2, value=1)

    assert working[5, 5] == 1
    assert blank[5, 5] == 0, "the cached frame must not be edited in place"


def test_the_same_label_reopens_the_array_already_in_hand(session, blank):
    working = session.open(FIRST, blank)

    assert session.open(FIRST, blank) is working


def test_an_edit_survives_moving_to_another_label_and_back(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=2, value=1))

    session.open(SECOND, blank)
    returned = session.open(FIRST, blank)

    assert returned[5, 5] == 1, "a pending edit was lost on navigation"


def test_an_untouched_label_is_not_kept(session, blank):
    session.open(FIRST, blank)
    session.open(SECOND, blank)

    assert len(session) == 0
    assert session.state_of(FIRST) is None


def test_a_label_with_no_file_is_recorded_as_created(session, blank):
    working = session.open(FIRST, blank, created=True)
    session.record(paint(working, [(5, 5)], radius=1, value=1))

    assert session.state_of(FIRST).created


# ---------------------------------------------------------------------------
# What is pending
# ---------------------------------------------------------------------------

def test_an_edit_marks_the_label_edited_and_reviewed(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=1, value=1))

    assert session.edited_keys() == {FIRST}
    assert session.reviewed_keys() == {FIRST}
    assert session


def test_a_label_can_be_passed_without_being_edited(session):
    session.mark_reviewed(FIRST)

    assert session.reviewed_keys() == {FIRST}
    assert session.edited_keys() == set()
    assert len(session) == 1


def test_a_stroke_that_changed_nothing_is_not_pending(session, blank):
    working = session.open(FIRST, blank)

    session.record(paint(working, [(5, 5)], radius=2, value=0))

    assert not session


def test_every_pending_label_is_yielded_with_its_array(session, blank):
    first = session.open(FIRST, blank)
    session.record(paint(first, [(5, 5)], radius=1, value=1))
    second = session.open(SECOND, blank)
    session.record(paint(second, [(9, 9)], radius=1, value=2))

    found = {entry.key: array for entry, array in session.labels()}

    assert set(found) == {FIRST, SECOND}
    assert found[FIRST][5, 5] == 1
    assert found[SECOND][9, 9] == 2, "the open label must come from memory"


def test_discarding_forgets_everything(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=1, value=1))

    session.discard()

    assert not session
    assert session.open_key is None


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def test_undo_reverses_the_last_stroke(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=2, value=1))

    assert session.undo()
    assert working[5, 5] == 0


def test_redo_puts_the_stroke_back(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=2, value=1))
    session.undo()

    assert session.redo()
    assert working[5, 5] == 1


def test_undoing_every_stroke_leaves_nothing_to_save(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=2, value=1))

    session.undo()

    assert session.edited_keys() == set(), "the label is as it was read"


def test_a_label_undone_to_the_start_is_still_reviewed(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=2, value=1))

    session.undo()

    assert session.reviewed_keys() == {FIRST}, "looking and deciding is a judgement"


def test_undo_applies_to_the_label_in_hand_only(session, blank):
    first = session.open(FIRST, blank)
    session.record(paint(first, [(5, 5)], radius=2, value=1))
    session.open(SECOND, blank)

    assert not session.can_undo, "the other label's history is not this one's"
    assert not session.undo()


def test_undo_history_survives_moving_away_and_back(session, blank):
    working = session.open(FIRST, blank)
    session.record(paint(working, [(5, 5)], radius=2, value=1))
    session.open(SECOND, blank)

    returned = session.open(FIRST, blank)

    assert session.can_undo
    session.undo()
    assert returned[5, 5] == 0


def test_undoing_nothing_is_harmless(session):
    assert not session.undo()
    assert not session.redo()
    assert not session.can_undo
