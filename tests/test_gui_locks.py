"""What the padlocks on the label tab say about themselves.

A lock button carries a state, so its shackle has to report that state. A
checked button drawn with a closed padlock beside an unchecked one drawn the
same way says nothing at all -- and what it is reporting is whether a class is
protected from the brush.
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
    from qtpy.QtCore import QSize

    from am_segmentation_manager.config import BACKGROUND_CLASS, SegmentationConfig
    from am_segmentation_manager.gui.label_controller import LabelController
    from am_segmentation_manager.gui.widgets.label_panel import (
        MARK,
        LabelPanel,
        lock_icon,
    )

#: The size the shackle is compared at. The icons are drawn at one size, so
#: this only has to be that size.
ICON = QSize(16, 16) if HAS_QT else None


@pytest.fixture
def panel(qt_app) -> LabelPanel:
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )
    panel = LabelPanel(LabelController(config=config))
    yield panel
    destroy(panel)


def _drawn(icon) -> bytes:
    """Return the pixels of *icon*, so two icons can be told apart."""
    image = icon.pixmap(ICON).toImage()
    return image.constBits().tobytes()


def test_the_two_padlocks_are_different_pictures(qt_app):
    """Everything below rests on this, so it is asserted rather than assumed."""
    assert _drawn(lock_icon(True, MARK)) != _drawn(lock_icon(False, MARK))


def test_a_class_starts_unlocked_and_says_so(panel):
    for index, button in panel.lock_buttons.items():
        assert not button.isChecked()
        assert _drawn(button.icon()) == _drawn(lock_icon(False, MARK)), index


def test_locking_a_class_closes_its_shackle_and_no_other(panel):
    index, button = next(iter(panel.lock_buttons.items()))

    button.setChecked(True)

    assert _drawn(button.icon()) == _drawn(lock_icon(True, MARK))
    for other, others_button in panel.lock_buttons.items():
        if other != index:
            assert _drawn(others_button.icon()) == _drawn(lock_icon(False, MARK)), other


def test_unlocking_opens_it_again(panel):
    button = next(iter(panel.lock_buttons.values()))
    button.setChecked(True)

    button.setChecked(False)

    assert _drawn(button.icon()) == _drawn(lock_icon(False, MARK))


def test_locking_everything_from_the_keyboard_closes_every_shackle_but_background(panel):
    """A lock changed by the 'A' key must look like one changed by clicking."""
    foreground = [
        button for index, button in panel.lock_buttons.items()
        if index != BACKGROUND_CLASS
    ]
    panel.toggle_all_locks()

    assert all(button.isChecked() for button in foreground)
    for button in foreground:
        assert _drawn(button.icon()) == _drawn(lock_icon(True, MARK))
    background = panel.lock_buttons[BACKGROUND_CLASS]
    assert _drawn(background.icon()) == _drawn(lock_icon(False, MARK))

    panel.toggle_all_locks()

    for button in panel.lock_buttons.values():
        assert _drawn(button.icon()) == _drawn(lock_icon(False, MARK))


def test_the_shackle_agrees_with_what_the_canvas_protects(panel):
    """The icon is a report on the canvas, not a decoration beside it."""
    index, button = next(iter(panel.lock_buttons.items()))
    button.setChecked(True)

    assert index in panel.canvas.protected
    assert _drawn(button.icon()) == _drawn(lock_icon(True, MARK))
