"""Tests for the label lines and mismatch wash on the frame timeline.

The timeline answers two different questions at once -- what workflow state an
image is in, and whether it has been annotated -- so the test that matters is
that the two are drawn distinctly and cannot be read as each other.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


HAS_QT = importlib.util.find_spec("qtpy") is not None
pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtGui import QImage
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.gui.widgets.frame_strip import (
        LABEL_COLOURS,
        LABEL_LINE_HEIGHT,
        ROW_HEIGHT,
        ROW_MODALITIES,
        STATE_COLOURS,
        FrameTimeline,
    )


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def timeline(qt_app):
    widget = FrameTimeline()
    widget.resize(400, widget.height())
    yield widget


def _render(widget) -> QImage:
    """Capture the widget's pixels so they can be asserted on.

    Uses ``QWidget.grab()`` rather than creating a QPainter and calling
    ``render()`` into it. That older approach relies on painter redirection:
    ``paintEvent`` opens its own QPainter on the widget while the external one
    is still active. Qt5 tolerated the nesting; Qt6 does not, and under PySide6
    it produced "QPaintDevice: Cannot destroy paint device that is being
    painted" and then a segmentation fault during garbage collection --
    reported from pytest's traceback formatting, far from the cause.

    grab() is the supported way to capture a widget and works on both bindings.
    """
    pixmap = widget.grab()
    image = pixmap.toImage()
    # On a HiDPI screen grab() returns more device pixels than the widget has
    # logical ones. The assertions below are in widget coordinates, so bring
    # the image back to that scale rather than doubling every expected x and y.
    if image.width() != widget.width() or image.height() != widget.height():
        image = image.scaled(widget.width(), widget.height())
    image.setDevicePixelRatio(1.0)
    return image


def _column(index: int, widget) -> int:
    """Return an x inside the given frame's column, clear of the cursor.

    The position marker is drawn down the middle of the current frame, so
    sampling the centre would read the cursor rather than the state beneath it.
    """
    width = widget.width() / max(1, widget._count)
    return int(index * width + width / 4)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_each_modality_gets_a_label_line_and_a_state_row(timeline):
    expected = len(ROW_MODALITIES) * (ROW_HEIGHT + LABEL_LINE_HEIGHT + 1) + 2
    assert timeline.height() == expected


def test_the_label_lines_sit_on_the_outside_edges(timeline):
    """SEM's line above its row, FIB's below, so neither can be misread."""
    sem_line = FrameTimeline._label_line_top(0)
    sem_row = FrameTimeline._row_top(0)
    fib_row = FrameTimeline._row_top(1)
    fib_line = FrameTimeline._label_line_top(1)

    assert sem_line < sem_row < fib_row < fib_line


def test_the_label_line_is_thinner_than_a_state_row(timeline):
    """It should read as a rule, not as a third and fourth state bar."""
    assert LABEL_LINE_HEIGHT < ROW_HEIGHT


# ---------------------------------------------------------------------------
# What gets drawn
# ---------------------------------------------------------------------------

def test_an_annotated_label_draws_its_line(timeline):
    timeline.set_states(
        {"SEM": ["selected"], "FIB": ["selected"]}, 0,
        labels={"SEM": ["present"], "FIB": ["present"]},
    )
    image = _render(timeline)

    y = FrameTimeline._label_line_top(0) + 1
    assert image.pixelColor(_column(0, timeline), y) == LABEL_COLOURS["present"]


def test_an_empty_label_draws_a_different_colour(timeline):
    timeline.set_states(
        {"SEM": ["selected"], "FIB": ["selected"]}, 0,
        labels={"SEM": ["present"], "FIB": ["empty"]},
    )
    image = _render(timeline)

    sem_y = FrameTimeline._label_line_top(0) + 1
    fib_y = FrameTimeline._label_line_top(1) + 1
    sem = image.pixelColor(_column(0, timeline), sem_y)
    fib = image.pixelColor(_column(0, timeline), fib_y)

    assert sem == LABEL_COLOURS["present"]
    assert fib == LABEL_COLOURS["empty"]
    assert sem != fib


def test_an_empty_label_is_clearly_lighter_than_the_background(timeline):
    """The distinction that has to survive a glance.

    A dark colour against a near black background is nearly indistinguishable
    from nothing drawn at all, which collapses "placeholder" into "no label".
    """
    from am_segmentation_manager.gui.widgets.frame_strip import BACKGROUND

    empty = LABEL_COLOURS["empty"]

    assert empty.lightness() > BACKGROUND.lightness() + 60
    assert empty.lightness() < LABEL_COLOURS["present"].lightness()


def test_a_missing_label_draws_nothing(timeline):
    timeline.set_states(
        {"SEM": ["selected"], "FIB": ["selected"]}, 0,
        labels={"SEM": ["missing"], "FIB": ["missing"]},
    )
    image = _render(timeline)

    y = FrameTimeline._label_line_top(0) + 1
    colour = image.pixelColor(_column(0, timeline), y)

    assert colour != LABEL_COLOURS["present"]
    assert colour != LABEL_COLOURS["empty"]


def test_label_state_does_not_disturb_the_workflow_state_row(timeline):
    """The two questions are drawn in separate places."""
    timeline.set_states(
        {"SEM": ["corrected"], "FIB": ["corrected"]}, 0,
        labels={"SEM": ["empty"], "FIB": ["empty"]},
    )
    image = _render(timeline)

    row_y = FrameTimeline._row_top(0) + ROW_HEIGHT // 2
    assert image.pixelColor(_column(0, timeline), row_y) == STATE_COLOURS["corrected"]


def test_omitting_labels_entirely_still_draws_the_states(timeline):
    """Callers that know nothing about labels must keep working."""
    timeline.set_states({"SEM": ["deselected"], "FIB": ["selected"]}, 0)
    image = _render(timeline)

    row_y = FrameTimeline._row_top(0) + ROW_HEIGHT // 2
    assert image.pixelColor(_column(0, timeline), row_y) == STATE_COLOURS["deselected"]


# ---------------------------------------------------------------------------
# The mismatch wash
# ---------------------------------------------------------------------------

def test_a_mismatched_frame_is_tinted_across_its_whole_column(timeline):
    timeline.set_states(
        {"SEM": ["selected", "selected"], "FIB": ["selected", "selected"]}, 0,
        mismatched={1},
        labels={"SEM": ["present", "present"], "FIB": ["present", "present"]},
    )
    image = _render(timeline)

    row_y = FrameTimeline._row_top(0) + ROW_HEIGHT // 2
    clean = image.pixelColor(_column(0, timeline), row_y)
    tinted = image.pixelColor(_column(1, timeline), row_y)

    assert clean == STATE_COLOURS["selected"]
    assert tinted != clean
    # Warmer than the untinted cell, since the wash is amber.
    assert tinted.red() > clean.red()


def test_the_wash_covers_the_label_lines_too(timeline):
    """A mismatch is about the frame, not about one row."""
    timeline.set_states(
        {"SEM": ["selected"], "FIB": ["selected"]}, 0,
        mismatched={0},
        labels={"SEM": ["present"], "FIB": ["present"]},
    )
    image = _render(timeline)

    y = FrameTimeline._label_line_top(1) + 1
    assert image.pixelColor(_column(0, timeline), y) != LABEL_COLOURS["present"]


def test_no_mismatch_leaves_the_strip_untinted(timeline):
    timeline.set_states({"SEM": ["selected"], "FIB": ["selected"]}, 0)
    image = _render(timeline)

    row_y = FrameTimeline._row_top(0) + ROW_HEIGHT // 2
    assert image.pixelColor(_column(0, timeline), row_y) == STATE_COLOURS["selected"]
