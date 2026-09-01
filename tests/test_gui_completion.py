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

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QCompleter

    from am_segmentation_manager.gui.widgets.common import ChipEditor
    from am_segmentation_manager.gui.widgets.metadata_panel import _suggestion_box

#: Two grid types told apart by their middles, not by how they start -- which
#: is why matching anywhere is worth keeping.
GRID_TYPES = ["UltrAuFoil Au R2/2 200 mesh", "Quantifoil Cu R2/1 200 mesh"]


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["suggestion box", "chip editor"])
def box(qt_app, request):
    """Both editors that offer values already in use, built as the panels do."""
    if request.param == "suggestion box":
        box = _suggestion_box()
        box.addItems(["", *GRID_TYPES])
        return box
    editor = ChipEditor()
    editor.set_suggestions(GRID_TYPES)
    return editor.entry


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
