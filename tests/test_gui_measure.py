"""The scale bar every acquisition and grid overview carries, and measuring on one.

Where something is drawn is asserted on the pixels, as `docs/display.md` asks of
anything about how something renders; a size is asserted on what the pane works
out, the arithmetic behind it being `display`'s.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import class_config, destroy, settle

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

pytestmark = pytest.mark.skipif(not (HAS_QT and HAS_NUMPY), reason="Qt binding or numpy not installed")

if HAS_QT and HAS_NUMPY:
    import numpy as np
    from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
    from qtpy.QtGui import QImage, QKeyEvent, QMouseEvent

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.display import DisplayImage
    from am_segmentation_manager.gui.cache import FrameImage
    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel
    from am_segmentation_manager.gui.widgets.image_view import (
        ImageView,
        LinkedImageViews,
        Measurement,
    )
    from am_segmentation_manager.gui.widgets.label_canvas import TOOL_MEASURE, LabelCanvas
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE, TAB_LABEL

#: Grey enough that no pixel of the picture can be taken for the bar.
PICTURE_GREY = 100


def _view(qt_app, size: tuple[int, int], field_width_um: float | None) -> ImageView:
    """Return a pane of *size* showing a 300 x 200 grey frame, fitted."""
    view = ImageView()
    view.resize(*size)
    view.show()
    qt_app.processEvents()
    view.set_array(np.full((200, 300), PICTURE_GREY, dtype=np.uint8))
    view.fit()
    view.set_field_width(field_width_um)
    qt_app.processEvents()
    return view


def _white(view: ImageView) -> list[tuple[int, int]]:
    """Return every pure white pixel in the pane, as (x, y)."""
    image = view.viewport().grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    ratio = image.devicePixelRatio()
    return [
        (int(x / ratio), int(y / ratio))
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) & 0xFFFFFF == 0xFFFFFF
    ]


def _picture(view: ImageView):
    """Return the picture's corners in the pane's pixels."""
    rect = view._item.sceneBoundingRect()
    return view.mapFromScene(rect.topLeft()), view.mapFromScene(rect.bottomRight())


@pytest.mark.parametrize(
    "size, where",
    [((400, 600), "below"), ((700, 300), "beside")],
    ids=["room-below", "room-beside-only"],
)
def test_the_scale_bar_sits_off_the_picture_by_its_bottom_right_corner(qt_app, size, where):
    """Under the corner where the pane has room, else beside it: never over the picture."""
    view = _view(qt_app, size, field_width_um=30.0)
    try:
        top_left, bottom_right = _picture(view)
        white = _white(view)

        assert white, "a frame with a scale is drawn with a bar"
        assert not [
            (x, y) for x, y in white
            if top_left.x() <= x <= bottom_right.x() and top_left.y() <= y <= bottom_right.y()
        ], "nothing of the bar is drawn over the picture"
        if where == "below":
            assert all(y > bottom_right.y() for _, y in white)
            assert max(x for x, _ in white) <= bottom_right.x()
        else:
            assert all(x > bottom_right.x() for x, _ in white)
            assert max(y for _, y in white) <= bottom_right.y()
    finally:
        destroy(view)


@pytest.mark.parametrize(
    "field_width_um, zoom", [(None, 1.0), (30.0, 6.0)], ids=["no-scale", "corner-off-the-pane"]
)
def test_no_scale_bar_is_drawn_without_a_scale_or_room_beside_the_picture(
    qt_app, field_width_um, zoom
):
    view = _view(qt_app, (400, 600), field_width_um)
    try:
        view.zoom_by(zoom)
        qt_app.processEvents()

        assert view.scale_bar_layout() is None
        assert _white(view) == []
    finally:
        destroy(view)


@pytest.mark.parametrize("field_width_um", [7.0, 30.0, 1800.0])
def test_fitted_every_frame_s_bar_is_about_a_fifth_of_its_width(qt_app, field_width_um):
    """Rounded to 1, 2 or 5, so between two fifths of the target and all of it."""
    view = _view(qt_app, (400, 600), field_width_um)
    try:
        layout = view.scale_bar_layout()
        shown = view._item.sceneBoundingRect().width() * view.transform().m11()

        assert 0.08 * shown <= layout.bar.width() <= 0.2 * shown + 1
    finally:
        destroy(view)


def test_zooming_in_chooses_a_shorter_bar(qt_app):
    view = _view(qt_app, (400, 600), field_width_um=30.0)
    try:
        fitted = view.scale_bar_layout().length_um
        view.zoom_by(1.25 ** 3)
        # Zoomed in about the middle the corner leaves the pane, and the bar
        # with it, so the corner is brought back to where the bar has room.
        view.centerOn(view._item.sceneBoundingRect().bottomRight())

        assert view.scale_bar_layout().length_um < fitted
    finally:
        destroy(view)


def test_clearing_a_pane_takes_its_scale_with_it(qt_app):
    view = _view(qt_app, (400, 600), field_width_um=30.0)
    try:
        view.clear()

        assert view.field_width_um is None
    finally:
        destroy(view)


def test_both_curate_panes_are_given_the_frame_s_scale(qt_app):
    panel = ImagePanel(SegmentationConfig(class_colors={0: (0, 0, 0)}, class_labels={0: "background"}))
    try:
        panel.show_image(
            FrameImage(DisplayImage(
                image=np.zeros((64, 96), dtype=np.uint8), field_width_um=40.0,
            )),
            show_overlay=True,
        )

        assert panel.pair.left.field_width_um == panel.pair.right.field_width_um == 40.0
    finally:
        destroy(panel)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def _mouse(kind, position: QPoint, button=None, modifiers=None):
    """Build a mouse event under either Qt version."""
    button = button if button is not None else Qt.MouseButton.LeftButton
    modifiers = modifiers if modifiers is not None else Qt.KeyboardModifier.NoModifier
    point = QPointF(position)
    try:
        return QMouseEvent(kind, point, point, point, button, button, modifiers)
    except TypeError:  # pragma: no cover - depends on the installed binding
        return QMouseEvent(kind, point, button, button, modifiers)


def _drag(view: ImageView, start, end, *, button=None, modifiers=None) -> None:
    """Drag between two points of the picture, given in its own pixels."""
    begin = view.mapFromScene(QPointF(*start))
    finish = view.mapFromScene(QPointF(*end))
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, begin, button, modifiers))
    view.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, finish, button, modifiers))
    view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, finish, button, modifiers))


def _measuring_view(qt_app, field_width_um=30.0, source_width=None) -> ImageView:
    """A 300-pixel-wide frame, 0.1 um a pixel with the default scale, measuring."""
    view = _view(qt_app, (400, 600), field_width_um)
    view.set_field_width(field_width_um, source_width)
    view.set_measuring(True)
    # One pane pixel to a picture pixel, so the points dragged between land
    # exactly where they are asked for rather than on the nearest pane pixel.
    view.resetTransform()
    view.centerOn(view._item)
    return view


@pytest.mark.parametrize(
    "shift, sizes", [(False, ("5 µm",)), (True, ("3 µm", "4 µm"))], ids=["line", "rectangle"]
)
def test_a_drag_measures_a_line_and_with_shift_a_rectangle(qt_app, shift, sizes):
    view = _measuring_view(qt_app)
    try:
        _drag(
            view, (50, 50), (80, 90),
            modifiers=Qt.KeyboardModifier.ShiftModifier if shift else None,
        )
        qt_app.processEvents()

        assert view.measurement.rectangle is shift
        assert view.measured_lengths() == sizes
        top_left, bottom_right = _picture(view)
        assert [
            (x, y) for x, y in _white(view)
            if top_left.x() <= x <= bottom_right.x() and top_left.y() <= y <= bottom_right.y()
        ], "the measurement is drawn over the picture"
    finally:
        destroy(view)


def test_without_a_scale_a_length_is_in_the_file_s_pixels(qt_app):
    """The pane shows a 300-pixel copy of a 600-pixel file."""
    view = _measuring_view(qt_app, field_width_um=None, source_width=600)
    try:
        _drag(view, (50, 50), (80, 90))

        assert view.measured_lengths() == ("100 px",)
    finally:
        destroy(view)


@pytest.mark.parametrize("how", ["switched-off", "clicked"])
def test_a_measurement_goes_when_measuring_stops_or_on_a_click(qt_app, how):
    view = _measuring_view(qt_app)
    try:
        _drag(view, (50, 50), (80, 90))
        if how == "switched-off":
            view.set_measuring(False)
        else:
            _drag(view, (120, 120), (120, 120))

        assert view.measurement is None
    finally:
        destroy(view)


@pytest.mark.parametrize("shape, kept", [((200, 300), True), ((100, 300), False)])
def test_a_measurement_is_kept_for_a_frame_of_the_same_shape_only(qt_app, shape, kept):
    view = _measuring_view(qt_app)
    try:
        _drag(view, (50, 50), (80, 90))

        view.set_array(np.full(shape, PICTURE_GREY, dtype=np.uint8))

        assert (view.measurement is not None) is kept
    finally:
        destroy(view)


def test_the_other_curate_pane_shows_what_one_measures(qt_app):
    pair = LinkedImageViews("image", "image + label")
    try:
        pair.resize(800, 600)
        pair.show()
        for pane in (pair.left, pair.right):
            pane.set_array(np.full((200, 300), PICTURE_GREY, dtype=np.uint8))
        pair.set_field_width(30.0)
        pair.set_measuring(True)
        qt_app.processEvents()

        _drag(pair.right, (50, 50), (80, 90))

        assert pair.left.measurement == pair.right.measurement is not None
    finally:
        destroy(pair)


def test_the_middle_button_pans_while_measuring(qt_app):
    view = _measuring_view(qt_app)
    try:
        view.zoom_by(2.0)
        before = view.horizontalScrollBar().value()

        _drag(view, (100, 100), (60, 100), button=Qt.MouseButton.MiddleButton)

        assert view.horizontalScrollBar().value() != before
        assert view.measurement is None
    finally:
        destroy(view)


def test_space_held_gives_the_left_button_back_to_panning(qt_app):
    view = _measuring_view(qt_app)
    try:
        view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))

        assert view.dragMode() == view.DragMode.ScrollHandDrag

        view.keyReleaseEvent(QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))

        assert view.dragMode() == view.DragMode.NoDrag
    finally:
        destroy(view)


# ---------------------------------------------------------------------------
# The Measure button on each tab
# ---------------------------------------------------------------------------

def _line() -> Measurement:
    return Measurement((10.0, 10.0), (40.0, 30.0))


def _press(window, tab: str, key: str) -> None:
    """Do what *key* does on *tab*."""
    next(b for b in window._keys.bindings if b.tab == tab and b.key == key).action()


def test_m_switches_measuring_on_and_off_in_every_curate_view(window):
    panel = window.curate_panel.image_panel
    views = (panel.pair.left, panel.pair.right, panel.grid_view)

    _press(window, TAB_CURATE, "M")

    assert panel.measure_button.isChecked()
    assert all(view.measuring for view in views)

    _press(window, TAB_CURATE, "M")

    assert not any(view.measuring for view in views)


def test_plots_cannot_be_measured(window):
    panel = window.curate_panel.image_panel
    panel.measure_button.setChecked(True)

    window.curate_panel.set_mode("plot", from_user=True)

    assert not panel.measure_button.isChecked()
    assert not panel.measure_button.isEnabled()
    assert not panel.pair.left.measuring


def test_another_view_takes_the_measurement_away(window):
    panel = window.curate_panel.image_panel
    panel.measure_button.setChecked(True)
    panel.pair.left.show_measurement(_line())

    window.curate_panel.set_mode("fib", from_user=True)

    assert panel.pair.left.measurement is None
    assert panel.measure_button.isChecked(), "measuring stays on in the new view"


@pytest.fixture
def label_window(qt_app, window):
    """The window with the Label tab in front and a frame drawn on it."""
    window.tabs.setCurrentWidget(window.label_panel)
    settle(qt_app, lambda: not window.label_panel.canvas._item.pixmap().isNull())
    return window


def test_m_measures_and_again_goes_back_to_the_tool_before(label_window):
    panel = label_window.label_panel
    panel.set_tool("fill")

    _press(label_window, TAB_LABEL, "M")

    assert panel.canvas.tool == TOOL_MEASURE and panel.canvas.measuring
    assert panel.tool_buttons[TOOL_MEASURE].isChecked()

    _press(label_window, TAB_LABEL, "M")

    assert panel.canvas.tool == "fill" and not panel.canvas.measuring


def test_clicking_measure_while_it_is_chosen_goes_back_to_the_tool_before(label_window):
    """An exclusive group never unchecks a button by clicking it, so the panel does."""
    panel = label_window.label_panel
    button = panel.tool_buttons[TOOL_MEASURE]

    button.click()
    assert panel.canvas.tool == TOOL_MEASURE

    button.click()
    assert panel.canvas.tool == "brush"


def test_choosing_another_tool_takes_the_measurement_away(label_window):
    panel = label_window.label_panel
    _press(label_window, TAB_LABEL, "M")
    panel.canvas.show_measurement(_line())

    _press(label_window, TAB_LABEL, "D")

    assert panel.canvas.measurement is None


@pytest.mark.parametrize("editable", [True, False], ids=["editing", "view-only"])
def test_measuring_on_the_label_canvas_writes_nothing(qt_app, editable):
    """Shift draws a rectangle while measuring, rather than picking a class."""
    canvas = LabelCanvas(class_config())
    try:
        canvas.resize(400, 400)
        canvas.show()
        label = np.zeros((60, 80), dtype=np.uint8)
        canvas.set_frame(np.zeros((60, 80), dtype=np.uint8), label)
        canvas.set_editable(editable)
        canvas.set_tool(TOOL_MEASURE)
        picked = []
        canvas.class_picked.connect(picked.append)
        qt_app.processEvents()

        _drag(canvas, (10, 10), (40, 30), modifiers=Qt.KeyboardModifier.ShiftModifier)

        assert canvas.measurement is not None and canvas.measurement.rectangle
        assert not label.any() and picked == []
    finally:
        destroy(canvas)
