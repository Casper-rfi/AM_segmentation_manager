"""The control that is on or off, and where it is used.

A tick box and a switch answer different questions. A row of a list is ticked --
it is one of many, and the tick says which ones a change covers. A standalone
control is switched: there is one of it, and it is on or off. The window used a
tick box for both, so the two questions looked alike.
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
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication, QCheckBox

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.export_controller import ExportController
    from am_segmentation_manager.gui.label_controller import LabelController
    from am_segmentation_manager.gui.metadata_controller import MetadataController
    from am_segmentation_manager.gui.widgets.controls import ToggleSwitch
    from am_segmentation_manager.gui.widgets.export_panel import ExportPanel
    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel
    from am_segmentation_manager.gui.widgets.label_panel import LabelPanel
    from am_segmentation_manager.gui.widgets.lists import CheckList
    from am_segmentation_manager.gui.widgets.metadata_panel import MetadataPanel
    from am_segmentation_manager.gui.widgets.state_panel import StatePanel



@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config():
    return SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )


def _drawn(widget) -> bytes:
    """Return the pixels the widget paints, so two states can be compared."""
    widget.resize(widget.sizeHint())
    return widget.grab().toImage().constBits().tobytes()


# ---------------------------------------------------------------------------
# The control itself
# ---------------------------------------------------------------------------

def test_it_is_a_check_box(qt_app):
    """The whole reason the swap was one word per call site.

    Every panel and some forty tests drive these through setChecked, isChecked
    and toggled. Painting differently must not cost any of that.
    """
    assert issubclass(ToggleSwitch, QCheckBox)


def test_on_and_off_are_different_pictures(qt_app):
    switch = ToggleSwitch("Show label")

    switch.setChecked(False)
    off = _drawn(switch)
    switch.setChecked(True)

    assert _drawn(switch) != off


def test_a_switch_that_cannot_be_used_says_so(qt_app):
    switch = ToggleSwitch("Show label")
    switch.setChecked(True)
    enabled = _drawn(switch)

    switch.setEnabled(False)

    assert _drawn(switch) != enabled


def test_the_caption_is_given_room(qt_app):
    """A switch drawn over its own caption is one nobody can read."""
    bare = ToggleSwitch("")
    captioned = ToggleSwitch("One image per label")

    assert captioned.sizeHint().width() > bare.sizeHint().width()


def test_the_focus_ring_fits_inside_the_widget(qt_app):
    """Reserved room, not drawn over the edge.

    The ring is painted outside the track, so without room kept for it the
    left side is clipped flat where it should curve -- which reads as a
    drawing fault rather than as focus.
    """
    from am_segmentation_manager.gui.widgets import theme

    switch = ToggleSwitch("Enable editing")
    switch.resize(switch.sizeHint())
    track = switch._track()

    assert track.left() >= theme.SWITCH_FOCUS
    assert track.top() >= theme.SWITCH_FOCUS
    assert track.bottom() <= switch.height() - theme.SWITCH_FOCUS


def test_the_keyboard_still_works(qt_app):
    """Space is how a switch is reached without a pointer, and it comes free
    from QCheckBox -- which is worth asserting, since the painting is not."""
    switch = ToggleSwitch("Enable editing")
    seen: list[bool] = []
    switch.toggled.connect(seen.append)

    for kind in (QKeyEvent.KeyPress, QKeyEvent.KeyRelease):
        qt_app.sendEvent(
            switch, QKeyEvent(kind, Qt.Key_Space, Qt.NoModifier, " ")
        )

    assert switch.isChecked()
    assert seen == [True]


# ---------------------------------------------------------------------------
# Where it is used
# ---------------------------------------------------------------------------

def _panels(config):
    """One of every panel that carries an on/off control."""
    return [
        StatePanel(),
        ImagePanel(config),
        LabelPanel(LabelController(config=config)),
        MetadataPanel(MetadataController()),
        ExportPanel(ExportController()),
    ]


def test_every_standalone_on_off_control_is_a_switch(qt_app, config):
    """The coherence that was asked for: one control, the same everywhere.

    Found by searching the panels rather than by listing the controls, so a
    tick box added later is caught here rather than noticed on screen.
    """
    for panel in _panels(config):
        plain = [
            box
            for box in panel.findChildren(QCheckBox)
            if not isinstance(box, ToggleSwitch)
        ]
        assert not plain, [box.text() for box in plain]


def test_a_row_of_a_list_is_still_ticked(qt_app):
    """A tick says which of many rows a change covers. A switch cannot."""
    listing = CheckList()
    listing.set_values(["one", "two"], ticked=[True, False])

    assert not isinstance(listing, ToggleSwitch)
    assert listing.item(0).checkState() == Qt.Checked
    assert listing.item(1).checkState() == Qt.Unchecked
