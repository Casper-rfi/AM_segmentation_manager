"""The selection model: what a state change applies to.

Qt-free, so these run without a display. The one-level-at-a-time rule and the
gesture arithmetic are the whole of the model; the widgets only draw it and
report clicks, and those are covered in the window tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.gui.batch_scope import (
    LEVEL_CURRENT_FRAME,
    LEVEL_FRAMES,
    LEVEL_LAMELLAE,
    LEVEL_PROJECTS,
    Selection,
    selection_key,
)

# ---------------------------------------------------------------------------
# One level at a time
# ---------------------------------------------------------------------------

def test_nothing_selected_means_the_frame_on_screen():
    assert Selection().level == LEVEL_CURRENT_FRAME
    assert Selection().is_empty is True


def test_ticking_a_lamella_clears_the_ticked_projects():
    """Otherwise a change would be decided by ticks the reach line ignores."""
    selection = Selection()
    selection.check(LEVEL_PROJECTS, "/p", True)

    selection.check(LEVEL_LAMELLAE, "/p/lam", True)

    assert selection.projects == set()
    assert selection.level == LEVEL_LAMELLAE


def test_ticking_a_project_clears_the_ticked_lamellae():
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/p/lam", True)

    selection.check(LEVEL_PROJECTS, "/p", True)

    assert selection.lamellae == set()
    assert selection.level == LEVEL_PROJECTS


def test_picking_a_frame_clears_both_lists():
    selection = Selection()
    selection.check(LEVEL_PROJECTS, "/p", True)
    selection.check(LEVEL_LAMELLAE, "/p/lam", True)

    selection.toggle(3)

    assert selection.projects == set()
    assert selection.lamellae == set()
    assert selection.level == LEVEL_FRAMES


def test_extending_over_frames_clears_the_lists_too():
    """A Shift-drag selects without ever going through toggle()."""
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/p/lam", True)

    selection.begin_extend(2)
    selection.extend_to(4)

    assert selection.lamellae == set()
    assert selection.frames == {2, 3, 4}


def test_unticking_clears_nothing_else():
    """Taking something out can only make a change smaller, so it is not a
    gesture that needs to reset anything."""
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/a", True)
    selection.check(LEVEL_LAMELLAE, "/b", True)

    selection.check(LEVEL_PROJECTS, "/p", False)

    assert selection.lamellae == {"/a", "/b"}


def test_ticking_every_lamella_clears_the_ticked_projects():
    selection = Selection()
    selection.check(LEVEL_PROJECTS, "/p", True)

    selection.check_all(LEVEL_LAMELLAE, ["/p/1", "/p/2"], True)

    assert selection.projects == set()
    assert selection.lamellae == {"/p/1", "/p/2"}


def test_clearing_the_frames_leaves_nothing_selected():
    """There is no coarser level left to fall back to: picking the frames out
    cleared it."""
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/p/lam", True)
    selection.toggle(3)

    selection.clear_frames()

    assert selection.is_empty


# ---------------------------------------------------------------------------
# Ticking
# ---------------------------------------------------------------------------

def test_a_key_is_the_resolved_path(tmp_path: Path):
    """Keyed by path, not by row: a row means a different lamella as soon as
    the project changes."""
    assert selection_key(tmp_path / "a" / ".." / "a") == str(tmp_path.resolve() / "a")


def test_unticking_removes_only_that_one():
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/a", True)
    selection.check(LEVEL_LAMELLAE, "/b", True)

    selection.check(LEVEL_LAMELLAE, "/a", False)

    assert selection.lamellae == {"/b"}


def test_unticking_something_never_ticked_is_not_an_error():
    selection = Selection()

    selection.check(LEVEL_LAMELLAE, "/a", False)

    assert selection.lamellae == set()


def test_clearing_the_ticks_touches_only_the_keys_named():
    """"None" clears the list on screen; nothing else is named, so nothing else
    moves. The controller is what keeps that list the whole of the selection."""
    selection = Selection()

    selection.check_all(LEVEL_LAMELLAE, ["/here/1", "/here/2"], True)
    selection.check_all(LEVEL_LAMELLAE, ["/here/1"], False)

    assert selection.lamellae == {"/here/2"}


def test_an_uncheckable_level_is_refused():
    with pytest.raises(ValueError):
        Selection().check(LEVEL_FRAMES, "/a", True)


# ---------------------------------------------------------------------------
# Picking frames out
# ---------------------------------------------------------------------------

def test_toggling_adds_then_removes():
    selection = Selection()

    selection.toggle(4)
    assert selection.frames == {4}

    selection.toggle(4)
    assert selection.frames == set()


def test_an_extension_covers_everything_between_the_anchor_and_here():
    selection = Selection()
    selection.toggle(3)          # anchor lands on 3

    selection.begin_extend(6)
    selection.extend_to(6)

    assert selection.frames == {3, 4, 5, 6}


def test_an_extension_reaches_backwards_too():
    selection = Selection()
    selection.toggle(6)

    selection.begin_extend(3)
    selection.extend_to(3)

    assert selection.frames == {3, 4, 5, 6}


def test_dragging_back_shrinks_the_range_again():
    """Each extension describes the whole range rather than adding to it, or a
    drag would leave its widest reach behind."""
    selection = Selection()
    selection.toggle(1)
    selection.begin_extend(6)

    selection.extend_to(6)
    selection.extend_to(3)

    assert selection.frames == {1, 2, 3}


def test_an_extension_keeps_what_was_already_picked():
    """The 1, 3, 4, 5, 6 case: two Ctrl-clicks and then a Shift-click."""
    selection = Selection()
    selection.toggle(1)
    selection.toggle(3)

    selection.begin_extend(6)
    selection.extend_to(6)

    assert selection.frames == {1, 3, 4, 5, 6}


def test_extending_with_nothing_picked_yet_starts_here():
    """Shift on a fresh lamella selects one frame rather than nothing."""
    selection = Selection()

    selection.begin_extend(5)
    selection.extend_to(5)

    assert selection.frames == {5}


def test_navigating_moves_the_anchor():
    """So a Shift-extension measures from where the user is, not from wherever
    they last selected something."""
    selection = Selection()
    selection.toggle(1)

    selection.point_at(8)
    selection.begin_extend(10)
    selection.extend_to(10)

    assert selection.frames == {1, 8, 9, 10}


def test_clearing_frames_forgets_the_anchor_too():
    selection = Selection()
    selection.toggle(4)

    selection.clear_frames()

    assert selection.frames == set()
    assert selection.anchor is None


def test_clearing_everything_clears_every_level():
    selection = Selection()
    selection.check(LEVEL_PROJECTS, "/p", True)
    selection.check(LEVEL_LAMELLAE, "/l", True)
    selection.toggle(1)

    selection.clear()

    assert selection.is_empty


# ---------------------------------------------------------------------------
# Saying what is selected, which is what a confirmation names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda s: None, "this frame"),
        (lambda s: s.toggle(1), "1 frame"),
        (lambda s: [s.toggle(1), s.toggle(2)], "2 frames"),
        (lambda s: s.check(LEVEL_LAMELLAE, "/a", True), "1 lamella"),
        (
            lambda s: [s.check(LEVEL_LAMELLAE, k, True) for k in ("/a", "/b")],
            "2 lamellae",
        ),
        (lambda s: s.check(LEVEL_PROJECTS, "/a", True), "1 project"),
        (
            lambda s: [s.check(LEVEL_PROJECTS, k, True) for k in ("/a", "/b")],
            "2 projects",
        ),
    ],
)
def test_the_selection_describes_itself(build, expected):
    selection = Selection()
    build(selection)

    assert selection.describe() == expected


# ---------------------------------------------------------------------------
# Putting a selection back after a rescan
# ---------------------------------------------------------------------------

def test_restoring_replaces_every_level_at_once():
    selection = Selection()
    selection.check(LEVEL_PROJECTS, "/p", True)

    selection.restore(lamellae=["/p/lam"])

    assert selection.projects == set()
    assert selection.lamellae == {"/p/lam"}
    assert selection.level == LEVEL_LAMELLAE


def test_restoring_nothing_leaves_the_frame_on_screen():
    """An operation whose whole selection has gone falls back to the safest
    reach there is, rather than to a stale one."""
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/p/lam", True)

    selection.restore()

    assert selection.is_empty


def test_restoring_two_levels_is_refused():
    """The invariant the model rests on. A caller that hands over two levels
    has produced a selection showing one thing and acting on another."""
    selection = Selection()

    with pytest.raises(ValueError, match="one level at a time"):
        selection.restore(projects=["/p"], frames=[1])


def test_restoring_frames_forgets_the_gesture_that_built_them():
    """An extension measures from the last frame pointed at, and after a
    rescan that is wherever the view landed -- not wherever it was before."""
    selection = Selection()
    selection.toggle(4)

    selection.restore(frames=[4])

    assert selection.frames == {4}
    assert selection.anchor is None
