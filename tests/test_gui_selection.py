"""The selection model: what a state change applies to.

Qt-free, so these run without a display. The precedence and the gesture
arithmetic are the whole of the model; the widgets only draw it and report
clicks, and those are covered in the window tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.gui.selection import (
    LEVEL_CURRENT_FRAME,
    LEVEL_FRAMES,
    LEVEL_LAMELLAE,
    LEVEL_PROJECTS,
    Selection,
    selection_key,
)

# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

def test_nothing_selected_means_the_frame_on_screen():
    assert Selection().level == LEVEL_CURRENT_FRAME
    assert Selection().is_empty is True


def test_the_finest_level_present_wins():
    """Finest-first is the fail-safe order: a forgotten frame selection makes a
    change smaller than intended, where the other order would make it larger."""
    selection = Selection()
    selection.check(LEVEL_PROJECTS, "/p", True)
    assert selection.level == LEVEL_PROJECTS

    selection.check(LEVEL_LAMELLAE, "/p/lam", True)
    assert selection.level == LEVEL_LAMELLAE

    selection.toggle(3)
    assert selection.level == LEVEL_FRAMES


def test_clearing_the_finest_level_falls_back_to_the_next():
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/p/lam", True)
    selection.toggle(3)

    selection.clear_frames()

    assert selection.level == LEVEL_LAMELLAE


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


def test_ticking_all_leaves_other_projects_lamellae_alone():
    """The list only shows one project's, and un-ticking what is not on screen
    would be a change nobody could see."""
    selection = Selection()
    selection.check(LEVEL_LAMELLAE, "/elsewhere", True)

    selection.check_all(LEVEL_LAMELLAE, ["/here/1", "/here/2"], True)
    selection.check_all(LEVEL_LAMELLAE, ["/here/1", "/here/2"], False)

    assert selection.lamellae == {"/elsewhere"}


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
