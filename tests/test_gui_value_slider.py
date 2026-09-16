"""The one-handled slider, drawn as the range slider beside it is.

Fusion's slider has a taller groove and a square handle, so an opacity slider
paired above Min / Max read as a different kind of control. What is pinned here
is that it is still Qt's slider underneath, that a press lands where the pointer
is, and that its pixels are the range slider's.
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
    from qtpy.QtCore import QEvent, QPointF, QRect, Qt
    from qtpy.QtGui import QColor, QImage, QMouseEvent
    from qtpy.QtWidgets import QApplication, QSlider

    from am_segmentation_manager.gui.widgets.controls import RangeSlider, ValueSlider


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def slider(qt_app) -> ValueSlider:
    widget = ValueSlider()
    widget.setRange(0, 255)
    widget.resize(200, widget.sizeHint().height())
    yield widget
    destroy(widget)


def _drawn(widget) -> QImage:
    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(QColor("white"))
    widget.render(image)
    return image


def test_it_is_a_slider(slider):
    """Its keys, wheel and signal are Qt's, so only the drawing is new."""
    seen: list[int] = []
    slider.valueChanged.connect(seen.append)

    slider.setValue(40)

    assert isinstance(slider, QSlider)
    assert seen == [40]


def test_a_press_puts_the_handle_under_the_pointer(slider):
    point = QPointF(slider._x_of(180), slider.height() / 2)

    slider.mousePressEvent(
        QMouseEvent(
            QEvent.MouseButtonPress, point, point,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        )
    )

    assert slider.value() == 180


def test_it_is_drawn_as_the_range_slider_is(slider):
    """Compared on the pixels: the right half of each, where both hold the
    filled groove and one handle at the end."""
    pair = RangeSlider(0, 255, 0, 255)
    pair.resize(slider.size())
    slider.setValue(255)

    half = QRect(slider.width() // 2, 0, slider.width() // 2, slider.height())
    one, two = _drawn(slider).copy(half), _drawn(pair).copy(half)

    assert slider.sizeHint() == pair.sizeHint()
    assert one.constBits().tobytes() == two.constBits().tobytes()


def test_the_groove_is_filled_up_to_the_handle(slider):
    from am_segmentation_manager.gui.widgets.theme import active_accent

    slider.setValue(128)
    image = _drawn(slider)
    middle = slider.height() // 2

    assert image.pixelColor(round(slider._x_of(40)), middle) == QColor(active_accent().colour)
    assert image.pixelColor(round(slider._x_of(230)), middle) != QColor(active_accent().colour)


def test_no_stylesheet_rule_names_the_control(qt_app):
    from am_segmentation_manager.gui.widgets import theme

    sheet = theme.stylesheet(theme.active_accent())
    assert "ValueSlider" not in sheet
    assert "QSlider" not in sheet
