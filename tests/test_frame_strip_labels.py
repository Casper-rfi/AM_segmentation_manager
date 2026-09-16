"""Tests for the label lines and mismatch wash on the frame timeline.

The timeline answers two different questions at once -- what workflow state an
image is in, and whether it has been annotated -- so the test that matters is
that the two are drawn distinctly and cannot be read as each other.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import destroy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


HAS_QT = importlib.util.find_spec("qtpy") is not None
pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtGui import QImage

    from am_segmentation_manager.gui.widgets.frame_strip import (
        CURATE_BAND,
        LABEL_BAND,
        LABEL_COLOURS,
        LABEL_LINE_HEIGHT,
        ROW_HEIGHT,
        ROW_MODALITIES,
        SELECTION_BAND,
        SPECK_INK,
        STATE_COLOURS,
        FrameTimeline,
        timeline_legend,
    )


@pytest.fixture
def timeline(qt_app):
    widget = FrameTimeline()
    widget.resize(400, widget.height())
    yield widget
    destroy(widget)


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
    bands = len(ROW_MODALITIES) * (ROW_HEIGHT + LABEL_LINE_HEIGHT + 1) + 2

    assert timeline.height() == SELECTION_BAND + bands


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


def test_no_mismatch_leaves_the_strip_untinted(timeline):
    timeline.set_states({"SEM": ["selected"], "FIB": ["selected"]}, 0)
    image = _render(timeline)

    row_y = FrameTimeline._row_top(0) + ROW_HEIGHT // 2
    assert image.pixelColor(_column(0, timeline), row_y) == STATE_COLOURS["selected"]


# ---------------------------------------------------------------------------
# Picking frames out
#
# Plain click and drag still seek; the modifiers select instead. The two are
# separate on purpose: building a selection must not set a lamella's worth of
# images loading.
# ---------------------------------------------------------------------------

def _mouse_event(widget, index: int, kind, button, buttons, modifier):
    """Build a mouse event over a frame's column.

    The global position is passed even though nothing here reads it: the
    overload without one is deprecated in Qt 6 and warns under PySide6, while
    this form exists in both Qt 5 and Qt 6.
    """
    from qtpy.QtCore import QPoint
    from qtpy.QtGui import QMouseEvent

    point = QPoint(_column(index, widget), widget.height() // 2)
    return QMouseEvent(
        kind, point, widget.mapToGlobal(point), button, buttons, modifier
    )


def _click(widget, index: int, modifier=None):
    """Send a real press at a frame's column, with an optional modifier."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QMouseEvent

    widget.mousePressEvent(_mouse_event(
        widget, index, QMouseEvent.MouseButtonPress,
        Qt.LeftButton, Qt.LeftButton, modifier if modifier is not None else Qt.NoModifier,
    ))


def _drag_to(widget, index: int, modifier):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QMouseEvent

    widget.mouseMoveEvent(_mouse_event(
        widget, index, QMouseEvent.MouseMove, Qt.NoButton, Qt.LeftButton, modifier,
    ))


@pytest.fixture
def loaded(timeline):
    timeline.set_states(
        {"SEM": ["selected"] * 8, "FIB": ["selected"] * 8}, 0,
    )
    return timeline


def test_a_plain_click_seeks_rather_than_selecting(loaded):
    from qtpy.QtCore import Qt

    seeks, picks = [], []
    loaded.frame_selected.connect(seeks.append)
    loaded.frame_toggled.connect(picks.append)

    _click(loaded, 3, Qt.NoModifier)

    assert seeks == [3]
    assert picks == []


def test_ctrl_click_picks_a_frame_without_moving_to_it(loaded):
    from qtpy.QtCore import Qt

    seeks, picks = [], []
    loaded.frame_selected.connect(seeks.append)
    loaded.frame_toggled.connect(picks.append)

    _click(loaded, 3, Qt.ControlModifier)

    assert picks == [3]
    assert seeks == []


def test_shift_click_starts_and_reports_an_extension(loaded):
    from qtpy.QtCore import Qt

    started, reached = [], []
    loaded.extend_started.connect(started.append)
    loaded.extend_reached.connect(reached.append)

    _click(loaded, 5, Qt.ShiftModifier)

    assert started == [5]
    assert reached == [5]


def test_dragging_with_a_modifier_widens_the_extension(loaded):
    from qtpy.QtCore import Qt

    reached = []
    loaded.extend_reached.connect(reached.append)

    _click(loaded, 2, Qt.ShiftModifier)
    _drag_to(loaded, 4, Qt.ShiftModifier)
    _drag_to(loaded, 6, Qt.ShiftModifier)

    assert reached == [2, 4, 6]


def test_dragging_without_a_modifier_still_scrubs(loaded):
    from qtpy.QtCore import Qt

    seeks = []
    loaded.frame_selected.connect(seeks.append)

    _click(loaded, 1, Qt.NoModifier)
    _drag_to(loaded, 5, Qt.NoModifier)

    assert seeks == [1, 5]


def test_picked_frames_are_drawn_as_a_band_on_top(loaded):
    from am_segmentation_manager.gui.widgets.frame_strip import SELECTION

    loaded.set_states(
        {"SEM": ["selected"] * 8, "FIB": ["selected"] * 8}, 0, selected={5},
    )
    image = _render(loaded)

    assert image.pixelColor(_column(5, loaded), 1) == SELECTION
    assert image.pixelColor(_column(4, loaded), 1) != SELECTION


def test_the_band_does_not_cover_a_state_row(loaded):
    """It sits above everything, so a picked frame still shows its state."""
    loaded.set_states(
        {"SEM": ["corrected"] * 8, "FIB": ["corrected"] * 8}, 0, selected={5},
    )
    image = _render(loaded)

    row_y = FrameTimeline._row_top(0) + ROW_HEIGHT // 2
    assert image.pixelColor(_column(5, loaded), row_y) == STATE_COLOURS["corrected"]


# ---------------------------------------------------------------------------
# Speck counts, drawn over the state rows while the filter is on
# ---------------------------------------------------------------------------

def _ink_pixels(image, top: int, frame: int, frames: int) -> int:
    """Count speck ink inside one frame's cell of one modality's state row.

    Scoped to the cell rather than the whole row because the ink is the
    cursor's colour -- deliberately, both being marks the reviewer put there --
    so a count across the row would find the cursor as well. Every test below
    therefore keeps the cursor on a frame it is not looking at.
    """
    from am_segmentation_manager.gui.widgets.frame_strip import ROW_HEIGHT

    width = image.width() / frames
    return sum(
        image.pixelColor(x, y).rgb() == SPECK_INK.rgb()
        for y in range(top, top + ROW_HEIGHT - 1)
        for x in range(int(frame * width), int((frame + 1) * width))
    )


def test_no_marks_are_drawn_when_no_counts_are_given(timeline):
    """Which is what the filter being off means: the marks exist because the
    reviewer asked for them."""
    timeline.set_states(
        {"SEM": ["selected"] * 4}, current=3, labels={"SEM": ["present"] * 4}
    )
    image = _render(timeline)

    assert _ink_pixels(image, FrameTimeline._row_top(0), 1, 4) == 0


def test_a_frame_with_no_specks_is_left_blank(timeline):
    """Only frames with something get a mark, which is what keeps them sparse
    enough to read as the problems rather than as a row to be scanned."""
    timeline.set_states(
        {"SEM": ["selected"] * 4}, current=3,
        labels={"SEM": ["present"] * 4}, specks={"SEM": [0, 0, 0, 0]},
    )
    image = _render(timeline)

    assert _ink_pixels(image, FrameTimeline._row_top(0), 1, 4) == 0


def test_a_frame_with_specks_is_marked(timeline):
    timeline.set_states(
        {"SEM": ["selected"] * 4}, current=3,
        labels={"SEM": ["present"] * 4}, specks={"SEM": [0, 3, 0, 0]},
    )
    image = _render(timeline)

    assert _ink_pixels(image, FrameTimeline._row_top(0), 1, 4) > 0
    assert _ink_pixels(image, FrameTimeline._row_top(0), 2, 4) == 0


def test_a_frame_nothing_has_counted_yet_is_left_blank(timeline):
    """None is "not counted", and it must not read as "none": the sweep fills
    them in one at a time, and a blank that meant zero would be a wrong answer
    for as long as it took."""
    timeline.set_states(
        {"SEM": ["selected"] * 4}, current=3,
        labels={"SEM": ["present"] * 4}, specks={"SEM": [None, None, None, None]},
    )
    image = _render(timeline)

    assert _ink_pixels(image, FrameTimeline._row_top(0), 1, 4) == 0


def test_the_marks_go_on_the_modality_they_belong_to(timeline):
    """A label belongs to one beam; the other beam's row must stay clean."""
    timeline.set_states(
        {"SEM": ["selected"] * 4, "FIB": ["selected"] * 4}, current=3,
        labels={"SEM": ["present"] * 4, "FIB": ["present"] * 4},
        specks={"SEM": [0, 3, 0, 0], "FIB": [0, 0, 0, 0]},
    )
    image = _render(timeline)

    assert _ink_pixels(image, FrameTimeline._row_top(0), 1, 4) > 0
    assert _ink_pixels(image, FrameTimeline._row_top(1), 1, 4) == 0


def test_a_count_too_wide_for_its_frame_still_marks_it(qt_app):
    """The digits give way to a block rather than to nothing: a lamella dense
    enough to refuse the number is still one where the frame has to be
    findable."""
    widget = FrameTimeline()
    widget.resize(60, widget.height())          # 60 frames of one pixel each
    widget.set_states(
        {"SEM": ["selected"] * 60}, current=59,
        labels={"SEM": ["present"] * 60},
        specks={"SEM": [0] * 30 + [27] + [0] * 29},
    )
    image = _render(widget)

    assert _ink_pixels(image, FrameTimeline._row_top(0), 30, 60) > 0


def test_the_count_on_the_frame_you_are_on_clears_the_cursor(timeline):
    """The cursor is drawn down the middle of the cell and in the same white,
    so a count centred there has a line through it."""
    from am_segmentation_manager.gui.widgets.frame_strip import CURSOR_WIDTH

    timeline.set_states(
        {"SEM": ["selected"] * 4}, current=1,
        labels={"SEM": ["present"] * 4}, specks={"SEM": [0, 8, 0, 0]},
    )
    image = _render(timeline)

    width = timeline.width() / 4
    middle = int(1 * width + width / 2)
    top = FrameTimeline._row_top(0)
    struck = sum(
        image.pixelColor(x, y).rgb() == SPECK_INK.rgb()
        for y in range(top + 2, top + ROW_HEIGHT - 3)
        for x in range(middle - CURSOR_WIDTH, middle + CURSOR_WIDTH + 1)
    )
    drawn = _ink_pixels(image, top, 1, 4)

    assert drawn > 0, "the count should still be drawn"
    # Only the cursor's own pixels, no glyph among them.
    assert struck <= (ROW_HEIGHT - 5) * CURSOR_WIDTH


def test_a_count_elsewhere_stays_centred(timeline):
    """Only the frame being looked at is moved: shifting every count would
    leave the row reading as though it were out of step with its frames."""
    timeline.set_states(
        {"SEM": ["selected"] * 4}, current=3,
        labels={"SEM": ["present"] * 4}, specks={"SEM": [0, 8, 0, 0]},
    )
    image = _render(timeline)

    width = timeline.width() / 4
    top = FrameTimeline._row_top(0)
    left_half = sum(
        image.pixelColor(x, y).rgb() == SPECK_INK.rgb()
        for y in range(top, top + ROW_HEIGHT - 1)
        for x in range(int(width), int(width + width / 2))
    )

    assert left_half > 0, "a centred count straddles the middle of its cell"


# ---------------------------------------------------------------------------
# Hovering: what the frame under the pointer holds, above what the marks mean.
# ---------------------------------------------------------------------------

def _hover(widget, index: int) -> None:
    """Move the pointer over a frame's column with no button held."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QMouseEvent

    widget.mouseMoveEvent(_mouse_event(
        widget, index, QMouseEvent.MouseMove, Qt.NoButton, Qt.NoButton, Qt.NoModifier,
    ))


def test_a_frame_is_described_as_it_is_drawn(timeline):
    timeline.set_states(
        {"SEM": ["selected", "corrected"], "FIB": ["deselected", None]},
        0,
        labels={"SEM": ["present", "empty"], "FIB": ["missing", None]},
        selected={1},
        specks={"SEM": [0, 3]},
    )

    assert timeline.describe_frame(0) == [
        "Frame 1 of 2 (where you are)",
        "SEM: selected, label holds an annotation",
        "FIB: deselected, no label file",
    ]
    assert timeline.describe_frame(1) == [
        "Frame 2 of 2",
        "SEM: corrected, label is empty, 3 specks",
        "FIB: no image",
        CURATE_BAND,
    ]


def test_hovering_a_frame_shows_what_it_holds_above_the_legend(loaded):
    """The legend is drawn by what draws the strip, as the F1 list's is."""
    _hover(loaded, 3)

    card = loaded.card
    assert not card.isHidden()
    assert card.frame.text() == "\n".join(loaded.describe_frame(3))
    assert card.meanings == [meaning for _, meaning in timeline_legend(CURATE_BAND)[1:]]


def test_the_card_follows_the_pointer_to_another_frame(loaded):
    _hover(loaded, 1)
    _hover(loaded, 5)

    assert loaded.card.frame.text().startswith("Frame 6 of 8")


def test_the_card_goes_when_the_pointer_leaves(loaded):
    from qtpy.QtCore import QEvent

    _hover(loaded, 2)
    loaded.leaveEvent(QEvent(QEvent.Leave))

    assert loaded.card.isHidden()


def test_no_card_while_pressing_or_dragging(loaded):
    """A card over the strip while scrubbing would cover what is being scrubbed."""
    from qtpy.QtCore import Qt

    _hover(loaded, 2)
    _click(loaded, 2)
    assert loaded.card.isHidden()

    _drag_to(loaded, 4, Qt.NoModifier)
    assert loaded.card.isHidden()


def test_a_state_changing_under_the_pointer_is_described_at_once(loaded):
    _hover(loaded, 2)

    loaded.set_states({"SEM": ["corrected"] * 8, "FIB": ["selected"] * 8}, 0)

    assert "SEM: corrected" in loaded.card.frame.text()


def test_the_label_tab_s_card_names_its_own_band(qt_app):
    widget = FrameTimeline(band=LABEL_BAND)
    widget.resize(400, widget.height())
    widget.set_states({"SEM": ["selected"] * 2, "FIB": ["selected"] * 2}, 0, selected={1})

    _hover(widget, 1)

    assert LABEL_BAND in widget.card.meanings
    assert widget.describe_frame(1)[-1] == LABEL_BAND


def test_the_strip_has_no_tooltip_to_show_beside_the_card(timeline):
    assert timeline.toolTip() == ""
