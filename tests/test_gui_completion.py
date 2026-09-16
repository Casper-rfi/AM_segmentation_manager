"""What a suggestion box does with what has been half-typed.

A completer that matches anywhere and completes inline rewrites the field: it
keeps the typed prefix and appends the rest of a match that never started with
it. The values here are exported metadata, so the property pinned is that no
suggestion mechanism can put a word in the field that nobody typed.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import destroy

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QCompleter

    from am_segmentation_manager.gui.widgets.controls import ChipEditor
    from am_segmentation_manager.gui.widgets.metadata_panel import _suggestion_box

#: Two grid types told apart by their middles, not by how they start -- which
#: is why matching anywhere is worth keeping.
GRID_TYPES = ["UltrAuFoil Au R2/2 200 mesh", "Quantifoil Cu R2/1 200 mesh"]


@pytest.fixture(params=["suggestion box", "chip editor"])
def box(qt_app, request):
    """Both editors that offer values already in use, built as the panels do."""
    if request.param == "suggestion box":
        box = _suggestion_box()
        box.addItems(["", *GRID_TYPES])
        yield box
        destroy(box)
        return
    editor = ChipEditor()
    editor.set_suggestions(GRID_TYPES)
    yield editor.entry
    destroy(editor)


def test_a_suggestion_matches_the_middle_of_a_value(box):
    completer = box.completer()
    completer.setCompletionPrefix("foil")

    assert completer.completionCount() == len(GRID_TYPES)


def test_matching_anywhere_never_completes_in_place(box):
    """The fix, stated as the thing that made it necessary.

    Inline completion keeps what was typed and appends the tail of the match,
    which is only correct when the match is a prefix. Typing 'a' against
    'UltrAuFoil Au R2/2 200 mesh' wrote 'altrAuFoil Au R2/2 200 mesh' into the
    field. A popup replaces the whole text instead.
    """
    completer = box.completer()

    assert completer.filterMode() == Qt.MatchContains
    assert completer.completionMode() == QCompleter.PopupCompletion


def test_what_a_suggestion_offers_is_a_whole_value(box):
    """Why replacing is right and appending is not.

    The completion for 'a' is the entire grid type, which does not begin with
    an 'a'. Anything that keeps the typed character and adds this to it writes
    a value that is in no list -- which is what the popup is here to prevent.
    """
    completer = box.completer()
    completer.setCompletionPrefix("a")

    completion = completer.currentCompletion()
    assert completion in GRID_TYPES
    assert not completion.lower().startswith("a")


# ---------------------------------------------------------------------------
# Taking a suggestion
# ---------------------------------------------------------------------------

#: Where a Tab press is delivered. **Both are real.** An open popup grabs the
#: keyboard and gets the key, which is what happens in the window; the offscreen
#: platform cannot grab one and delivers to the field instead, which is what
#: happens in this suite. A test covering only the second passed against a
#: window in which Tab did nothing at all.
ROUTES = ("popup", "field")


@pytest.fixture(params=ROUTES)
def route(request) -> str:
    return request.param


def _press_tab(qt_app, box, route: str) -> None:
    """Send Tab where the platform would put it."""
    from qtpy.QtGui import QKeyEvent

    target = box.completer().popup() if route == "popup" else box.lineEdit()
    for kind in (QKeyEvent.KeyPress, QKeyEvent.KeyRelease):
        qt_app.sendEvent(target, QKeyEvent(kind, Qt.Key_Tab, Qt.NoModifier, "\t"))
    qt_app.processEvents()


def _offer(qt_app, box, typed: str) -> None:
    """Type *typed* and put the popup on screen, as the field does."""
    box.lineEdit().setText(typed)
    box.completer().setCompletionPrefix(typed)
    box.completer().complete()
    qt_app.processEvents()


def test_tab_takes_the_suggestion_the_popup_is_showing(qt_app, box, route):
    """A popup answers Return and a click; Tab is not its key.

    Left alone, Tab moved the focus and the half-typed letter stayed behind as
    the value -- obvious enough to notice, and still a value nobody typed.
    """
    box.show()
    _offer(qt_app, box, "a")

    _press_tab(qt_app, box, route)

    assert box.currentText() in GRID_TYPES


def test_tab_is_left_alone_when_nothing_is_being_offered(qt_app, box):
    """It is still the key that moves to the next field."""
    box.show()
    box.lineEdit().setText("nothing matches this")

    _press_tab(qt_app, box, "field")

    assert box.currentText() == "nothing matches this"


def test_a_chip_is_the_whole_value_and_not_the_letter_that_found_it(qt_app, route):
    """The editor that keeps what it is given: what Tab leaves is committed."""
    editor = ChipEditor()
    editor.set_suggestions(GRID_TYPES)
    editor.show()
    _offer(qt_app, editor.entry, "a")

    _press_tab(qt_app, editor.entry, route)
    editor.commit_pending()

    # Not compared against the box, which committing empties.
    assert len(editor.values()) == 1
    assert editor.values()[0] in GRID_TYPES
    destroy(editor)
