"""The control that sets a low and a high limit at once.

Two sliders for one interval leave the reader working out which is which, and
let the two be dragged past each other. What is pinned here is that the pair
cannot cross whichever way it is moved, that a handle can be put back on its
own, and that the control draws the interval rather than only its ends.
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
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QColor, QImage, QMouseEvent

    from am_segmentation_manager.gui.widgets.controls import RangeSlider


@pytest.fixture
def slider(qt_app) -> RangeSlider:
    """A slider over 0..255 resting wide open, laid out so x maps to a value."""
    widget = RangeSlider(0, 255, 0, 255)
    widget.resize(200, 20)
    yield widget
    destroy(widget)


def _double_click_at(slider, x: float) -> None:
    """Send a double-click at *x* along the track."""
    # QPointF, not QPoint: the integer overload is deprecated in Qt 6 and warns.
    point = QPointF(x, slider.height() / 2)
    slider.mouseDoubleClickEvent(
        QMouseEvent(
            QEvent.MouseButtonDblClick, point, point,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        )
    )


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------

def test_it_opens_where_it_was_told_to(slider):
    assert slider.values() == (0, 255)


def test_setting_a_pair_reports_it_once(slider):
    seen = []
    slider.range_changed.connect(lambda low, high: seen.append((low, high)))

    slider.set_range(40, 200)

    assert slider.values() == (40, 200)
    assert seen == [(40, 200)]


def test_setting_the_pair_it_already_holds_reports_nothing(slider):
    slider.set_range(40, 200)
    seen = []
    slider.range_changed.connect(lambda *values: seen.append(values))

    slider.set_range(40, 200)

    assert not seen, "a control that reports a change that did not happen redraws for nothing"


def test_reflecting_a_change_made_elsewhere_reports_nothing(slider):
    """Or the panel that set it hears its own change back and round again."""
    seen = []
    slider.range_changed.connect(lambda *values: seen.append(values))

    slider.set_range(40, 200, notify=False)

    assert slider.values() == (40, 200)
    assert not seen


def test_the_limits_are_clamped_into_the_range(slider):
    slider.set_range(-50, 900)

    assert slider.values() == (0, 255)


# ---------------------------------------------------------------------------
# Held apart
# ---------------------------------------------------------------------------

def test_a_pair_that_would_cross_is_pushed_apart(slider):
    """A picture that turned into a threshold would read as a fault rather
    than as a setting, so the control refuses instead."""
    slider.set_range(150, 100)

    low, high = slider.values()
    assert low < high


def test_dragging_the_low_handle_past_the_high_one_stops_it(slider):
    slider.set_range(0, 100)
    slider._active = "low"

    slider._drag_to(slider._x_of(200))

    low, high = slider.values()
    assert (low, high) == (99, 100), "the handle dragged gives way, not the other"


def test_dragging_the_high_handle_past_the_low_one_stops_it(slider):
    slider.set_range(100, 255)
    slider._active = "high"

    slider._drag_to(slider._x_of(20))

    low, high = slider.values()
    assert (low, high) == (100, 101)


def test_a_press_takes_hold_of_the_nearer_handle(slider):
    slider.set_range(50, 200)
    point = QPointF(slider._x_of(190), slider.height() / 2)

    slider.mousePressEvent(
        QMouseEvent(
            QEvent.MouseButtonPress, point, point,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        )
    )

    assert slider._active == "high"
    assert slider.values()[0] == 50, "the far handle must not have moved"


# ---------------------------------------------------------------------------
# Going back
# ---------------------------------------------------------------------------

def test_a_double_click_puts_back_only_the_handle_it_lands_on(slider):
    """Reaching an end by dragging is a thing nobody quite manages, which for
    this control means being left slightly off it without noticing."""
    slider.set_range(70, 200)

    _double_click_at(slider, slider._x_of(70))

    assert slider.values() == (0, 200)


def test_a_double_click_near_the_high_handle_puts_that_one_back(slider):
    slider.set_range(70, 200)

    _double_click_at(slider, slider._x_of(200))

    assert slider.values() == (70, 255)


def test_a_double_click_does_not_leave_the_handle_held(slider):
    """The press that opened it took hold; leaving it held would drag the
    handle wherever the pointer went next."""
    slider.set_range(70, 200)

    _double_click_at(slider, slider._x_of(70))

    assert not slider._dragging


def test_reset_opens_the_range_again(slider):
    slider.set_range(70, 200)

    slider.reset()

    assert slider.values() == (0, 255)


# ---------------------------------------------------------------------------
# What it draws
# ---------------------------------------------------------------------------

def _drawn(slider) -> QImage:
    image = QImage(slider.size(), QImage.Format_ARGB32)
    image.fill(QColor("white"))
    slider.render(image)
    return image


def test_the_interval_is_drawn_and_not_only_its_ends(slider):
    """The filled span is the answer; the empty ends are what is thrown away."""
    from am_segmentation_manager.gui.widgets.theme import active_accent

    slider.set_range(100, 160)
    image = _drawn(slider)
    middle = slider.height() // 2

    inside = image.pixelColor(round(slider._x_of(130)), middle)
    outside = image.pixelColor(round(slider._x_of(30)), middle)

    assert inside == QColor(active_accent().colour)
    assert outside != QColor(active_accent().colour)


def test_moving_a_handle_changes_the_picture(slider):
    """A control whose value moved and whose pixels did not is one nobody can
    read; a palette read back in Python would say nothing about either."""
    before = _drawn(slider).constBits().tobytes()

    slider.set_range(60, 190)

    assert _drawn(slider).constBits().tobytes() != before


def test_no_stylesheet_rule_names_the_control(qt_app):
    """It paints everything itself, so a rule would send it back through
    QStyle and draw a second control underneath this one."""
    from am_segmentation_manager.gui.widgets import theme

    assert "RangeSlider" not in theme.stylesheet(theme.active_accent())
