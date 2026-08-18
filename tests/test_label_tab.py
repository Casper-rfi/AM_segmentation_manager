"""Tests for the Label tab: its canvas, its panel, and its place in the window.

Qt's offscreen platform, so no display is needed. The frames are real files on
disk, because the point of the tab is that it reads them at full size.
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

from conftest import build_dataset

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGE_STACK),
    reason="Qt binding or optional image libs not installed",
)

if HAS_QT and HAS_IMAGE_STACK:
    import numpy as np
    import tifffile
    from qtpy.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
    from qtpy.QtGui import QImage, QMouseEvent
    from qtpy.QtWidgets import QApplication, QGraphicsView

    from am_segmentation_manager.config import (
        BACKGROUND_CLASS,
        UNKNOWN_CLASS_COLOR,
        SegmentationConfig,
    )
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.label_canvas import (
        MAX_BRUSH_RADIUS,
        TOOL_BRUSH,
        TOOL_FILL,
        LabelCanvas,
        label_colour_table,
        label_to_qimage,
    )
    from am_segmentation_manager.gui.widgets.main_window import (
        TAB_CURATE,
        TAB_LABEL,
        CurationWindow,
    )
    from am_segmentation_manager.operations.label_edit import new_label

CONFIG_COLOURS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)}


@pytest.fixture(scope="session")
def qt_app():
    """One QApplication for the whole session; Qt dislikes several."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config():
    return SegmentationConfig(
        class_colors=dict(CONFIG_COLOURS),
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A one-lamella project whose images and labels are real TIFFs."""
    dataset = build_dataset(tmp_path, acquisitions=3)
    for modality in ("SEM", "FIB"):
        for index in range(3):
            image = np.full((48, 64), 120, dtype=np.uint8)
            tifffile.imwrite(str(dataset.image(modality, index)), image)
            label = np.zeros((48, 64), dtype=np.uint8)
            label[:10] = 2
            tifffile.imwrite(str(dataset.label(modality, index)), label)
    return tmp_path


@pytest.fixture
def window(qt_app, project: Path, config, tmp_path: Path):
    """A window with the project scanned and the Label tab in front."""
    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller,
        settings=QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat),
    )
    window.scan(str(project))
    window.tabs.setCurrentWidget(window.label_panel)
    _settle(qt_app, lambda: window.label_panel.controller.cached_image() is not None)
    yield window
    window.close()


def _settle(app, condition=None, timeout: float = 10.0) -> None:
    """Run the event loop until *condition* holds, or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if condition is not None and condition():
            return
        time.sleep(0.01)
    if condition is not None and not condition():
        pytest.fail("Timed out waiting for the Label tab to settle")


# ---------------------------------------------------------------------------
# The colour table
# ---------------------------------------------------------------------------

def test_the_background_class_is_transparent(config):
    table = label_colour_table(config)

    assert table[BACKGROUND_CLASS] == 0, "background must not be painted over"


def test_a_configured_class_takes_its_own_colour_opaquely(config):
    table = label_colour_table(config)

    red, green, blue = CONFIG_COLOURS[2]
    assert table[2] == (0xFF << 24) | (red << 16) | (green << 8) | blue


def test_an_undeclared_class_is_visible_rather_than_invisible(config):
    table = label_colour_table(config)

    red, green, blue = UNKNOWN_CLASS_COLOR
    assert table[200] == (0xFF << 24) | (red << 16) | (green << 8) | blue


def test_the_table_covers_every_value_a_byte_can_hold(config):
    assert len(label_colour_table(config)) == 256


# ---------------------------------------------------------------------------
# Drawing the label
# ---------------------------------------------------------------------------

def test_a_label_becomes_an_indexed_image_of_the_same_size(config):
    label = np.zeros((30, 40), dtype=np.uint8)

    image = label_to_qimage(label, label_colour_table(config))

    assert image.format() == QImage.Format_Indexed8
    assert (image.width(), image.height()) == (40, 30)


def test_a_16_bit_label_is_drawn_without_changing_the_array(config):
    label = np.zeros((8, 8), dtype=np.uint16)
    label[0, 0] = 2

    image = label_to_qimage(label, label_colour_table(config))

    assert label.dtype == np.uint16, "the array behind the view must be untouched"
    assert image.pixelIndex(0, 0) == 2


def test_a_class_too_large_to_draw_shows_as_unknown_not_as_another_class(config):
    label = np.zeros((4, 4), dtype=np.uint16)
    label[0, 0] = 4000

    image = label_to_qimage(label, label_colour_table(config))

    assert image.pixelIndex(0, 0) == 255, "wrapping would show class 160 instead"


def test_the_image_does_not_share_the_array_it_came_from(config):
    label = np.zeros((8, 8), dtype=np.uint8)
    image = label_to_qimage(label, label_colour_table(config))

    label[0, 0] = 2

    assert image.pixelIndex(0, 0) == 0, "the buffer must be copied, not referenced"


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

def test_the_canvas_draws_both_layers(qt_app, config):
    canvas = LabelCanvas(config)

    canvas.set_frame(
        np.zeros((20, 30), dtype=np.uint8), np.zeros((20, 30), dtype=np.uint8)
    )

    assert not canvas._item.pixmap().isNull()
    assert not canvas._label_item.pixmap().isNull()
    assert canvas._label_item.isVisible()


def test_a_frame_with_no_label_draws_only_the_image(qt_app, config):
    canvas = LabelCanvas(config)

    canvas.set_frame(np.zeros((20, 30), dtype=np.uint8), None)

    assert not canvas._item.pixmap().isNull()
    assert not canvas._label_item.isVisible()


def test_hiding_the_label_leaves_the_image_alone(qt_app, config):
    canvas = LabelCanvas(config)
    canvas.set_frame(
        np.zeros((20, 30), dtype=np.uint8), np.zeros((20, 30), dtype=np.uint8)
    )

    canvas.set_label_shown(False)

    assert not canvas._label_item.isVisible()
    assert not canvas._item.pixmap().isNull()
    assert not canvas.label_shown


def test_a_hidden_label_stays_hidden_when_the_frame_changes(qt_app, config):
    canvas = LabelCanvas(config)
    canvas.set_label_shown(False)

    canvas.set_frame(
        np.zeros((20, 30), dtype=np.uint8), np.zeros((20, 30), dtype=np.uint8)
    )

    assert not canvas._label_item.isVisible()


def test_opacity_is_the_label_layer_s_own(qt_app, config):
    canvas = LabelCanvas(config)

    canvas.set_label_opacity(0.75)

    assert canvas._label_item.opacity() == pytest.approx(0.75)


def test_an_impossible_opacity_is_rejected(qt_app, config):
    canvas = LabelCanvas(config)

    with pytest.raises(ValueError, match="alpha"):
        canvas.set_label_opacity(1.5)


def test_clearing_empties_both_layers(qt_app, config):
    canvas = LabelCanvas(config)
    canvas.set_frame(
        np.zeros((20, 30), dtype=np.uint8), np.zeros((20, 30), dtype=np.uint8)
    )

    canvas.clear()

    assert canvas._item.pixmap().isNull()
    assert canvas._label_item.pixmap().isNull()


# ---------------------------------------------------------------------------
# The tab in the window
# ---------------------------------------------------------------------------

def test_the_tab_shows_the_lamella_the_curation_view_is_on(window):
    assert window.label_controller.lamella is window.controller.lamella
    assert not window.label_panel.canvas._item.pixmap().isNull()


def test_the_frame_is_drawn_at_full_resolution(window):
    prepared = window.label_controller.cached_image()

    assert prepared.display.image.shape == (48, 64)
    assert prepared.display.scale == 1.0


def test_the_counter_follows_the_position(qt_app, window):
    assert window.label_panel.frame_label.text() == "1 / 3"

    window.label_panel.step_frame(1)

    assert window.label_panel.frame_label.text() == "2 / 3"


def test_stepping_past_the_end_stays_on_the_last_frame(window):
    window.label_panel.step_frame(9)

    assert window.label_panel.frame_label.text() == "3 / 3"


def test_switching_modality_redraws(qt_app, window):
    window.label_panel.set_modality("FIB")
    _settle(qt_app, lambda: window.label_controller.cached_image() is not None)

    assert window.label_controller.modality == "FIB"
    assert not window.label_panel.canvas._item.pixmap().isNull()


def test_the_label_can_be_hidden_from_the_panel(window):
    window.label_panel.toggle_label()

    assert not window.label_panel.canvas.label_shown

    window.label_panel.toggle_label()

    assert window.label_panel.canvas.label_shown


def test_the_opacity_slider_drives_the_label_layer(window):
    window.label_panel.alpha_slider.setValue(90)

    assert window.label_panel.canvas._label_item.opacity() == pytest.approx(0.9)


def test_the_tab_keeps_its_place_when_it_is_left_and_returned_to(qt_app, window):
    window.label_panel.step_frame(2)
    window.tabs.setCurrentWidget(window.curate)
    window.tabs.setCurrentWidget(window.label_panel)

    assert window.label_panel.frame_label.text() == "3 / 3"


def test_moving_lamella_from_the_tab_moves_both_views(qt_app, config, tmp_path):
    root = tmp_path / "two"
    build_dataset(root, lamellae=2, acquisitions=2)
    for tiff in root.rglob("*.tif"):
        tifffile.imwrite(str(tiff), np.zeros((16, 16), dtype=np.uint8))

    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller, settings=QSettings(str(tmp_path / "p2.ini"), QSettings.IniFormat)
    )
    try:
        window.scan(str(root))
        window.tabs.setCurrentWidget(window.label_panel)
        first = window.controller.lamella
        assert len(window.controller.lamellae) == 2

        window._step_label_lamella(1)

        assert window.controller.lamella is not first
        assert window.label_controller.lamella is window.controller.lamella
        assert window.label_panel.frame_label.text() == "1 / 2"
    finally:
        window.close()


def test_the_tab_s_keys_are_live_only_on_it(window):
    window.tabs.setCurrentWidget(window.label_panel)
    live = {
        binding.key
        for binding, shortcut in zip(window._bindings, window._shortcuts)
        if shortcut.isEnabled()
    }
    assert "Right" in live

    window.tabs.setCurrentWidget(window.curate)
    label_keys = [b for b in window._bindings if b.tab == TAB_LABEL]

    assert label_keys, "the tab should have keys of its own"
    assert all(
        not shortcut.isEnabled()
        for binding, shortcut in zip(window._bindings, window._shortcuts)
        if binding.tab == TAB_LABEL
    )


def test_the_help_lists_the_tab_s_own_keys(window):
    help_text = window.shortcut_help(TAB_LABEL)

    assert "Next frame" in help_text
    assert "Show / hide the label" in help_text
    assert "Set state: corrected" not in help_text, "that is the Curate tab's"


def test_the_curate_tab_keeps_its_own_keys(window):
    assert "Set state: corrected" in window.shortcut_help(TAB_CURATE)


# ---------------------------------------------------------------------------
# Painting on the canvas
# ---------------------------------------------------------------------------

def mouse_event(kind, position, button=None, buttons=None):
    """Build a mouse event under either Qt version.

    Qt 6 requires a scene and a global position that Qt 5 does not take, so the
    two signatures are tried in turn.
    """
    button = button if button is not None else Qt.LeftButton
    buttons = buttons if buttons is not None else button
    point = QPointF(position)
    try:
        return QMouseEvent(kind, point, point, point, button, buttons, Qt.NoModifier)
    except TypeError:  # pragma: no cover - depends on the installed binding
        return QMouseEvent(kind, point, button, buttons, Qt.NoModifier)


@pytest.fixture
def canvas(qt_app, config):
    """A canvas with a frame loaded and painting switched on."""
    canvas = LabelCanvas(config)
    canvas.resize(200, 200)
    canvas.set_frame(np.zeros((60, 80), dtype=np.uint8), new_label((60, 80)))
    canvas.set_editable(True)
    return canvas


def drag(canvas, start, end):
    """Press, move and release, returning the patches emitted."""
    patches = []
    canvas.edited.connect(patches.append)
    canvas.mousePressEvent(mouse_event(QEvent.MouseButtonPress, start))
    canvas.mouseMoveEvent(mouse_event(QEvent.MouseMove, end))
    canvas.mouseReleaseEvent(mouse_event(QEvent.MouseButtonRelease, end))
    return patches


def test_a_drag_paints_and_reports_one_edit(canvas):
    patches = drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert len(patches) == 1, "a drag is one undo step, not one per mouse move"
    assert (canvas._label_array != 0).any(), "nothing was painted"


def test_the_brush_writes_the_chosen_class(canvas):
    canvas.set_paint_value(2)

    drag(canvas, QPoint(100, 100), QPoint(110, 110))

    assert set(np.unique(canvas._label_array)) == {0, 2}


def test_a_locked_class_survives_a_stroke_across_it(canvas):
    canvas._label_array[:] = 1
    canvas.set_protected({1})
    canvas.set_paint_value(2)

    drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert not (canvas._label_array == 2).any(), "the lock was ignored"


def test_nothing_is_painted_when_editing_is_off(canvas):
    canvas.set_editable(False)

    patches = drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert patches == []
    assert not (canvas._label_array != 0).any()


def test_a_frame_with_no_label_cannot_be_painted(qt_app, config):
    canvas = LabelCanvas(config)
    canvas.resize(200, 200)
    canvas.set_frame(np.zeros((60, 80), dtype=np.uint8), None)

    canvas.set_editable(True)

    assert not canvas.editable


def test_holding_space_pans_instead_of_painting(canvas):
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))

    patches = drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert patches == []
    assert not (canvas._label_array != 0).any()
    assert canvas.dragMode() == QGraphicsView.ScrollHandDrag


def test_releasing_space_gives_the_brush_back(canvas):
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))
    canvas.keyReleaseEvent(key_event(QEvent.KeyRelease, Qt.Key_Space))

    assert canvas.dragMode() == QGraphicsView.NoDrag

    patches = drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert len(patches) == 1


def test_the_middle_button_pans_without_painting(canvas):
    patches = []
    canvas.edited.connect(patches.append)

    canvas.mousePressEvent(
        mouse_event(QEvent.MouseButtonPress, QPoint(80, 80), Qt.MiddleButton)
    )
    canvas.mouseMoveEvent(
        mouse_event(QEvent.MouseMove, QPoint(120, 120), Qt.MiddleButton)
    )
    canvas.mouseReleaseEvent(
        mouse_event(QEvent.MouseButtonRelease, QPoint(120, 120), Qt.MiddleButton)
    )

    assert patches == []
    assert not (canvas._label_array != 0).any()


def test_the_brush_size_is_bounded(canvas):
    canvas.set_brush_radius(10_000)
    assert canvas.brush_radius == MAX_BRUSH_RADIUS

    canvas.set_brush_radius(-5)
    assert canvas.brush_radius == 0


def key_event(kind, key):
    """Build a key event, without auto-repeat."""
    from qtpy.QtGui import QKeyEvent

    return QKeyEvent(kind, key, Qt.NoModifier)


# ---------------------------------------------------------------------------
# Editing through the tab
# ---------------------------------------------------------------------------

def paint_on(window, value=2):
    """Paint a stroke on the frame the window is showing."""
    canvas = window.label_panel.canvas
    canvas.resize(200, 200)
    canvas.set_paint_value(value)
    return drag(canvas, QPoint(90, 90), QPoint(110, 110))


def test_a_stroke_is_recorded_against_the_label_on_screen(window):
    assert window.label_controller.editable

    paint_on(window)

    assert window.label_controller.pending_count() == 1
    assert window.label_controller.session.edited_keys() == {
        window.label_controller.current_key
    }


def test_an_edit_is_not_written_to_disk(window):
    key = window.label_controller.current_key
    before = key.read_bytes()

    paint_on(window)

    assert key.read_bytes() == before, "slice 3 holds edits, it does not save them"


def test_the_window_says_edits_are_unsaved(window):
    paint_on(window)

    assert "not yet written to disk" in window.label_panel.pending_label.text()


def test_an_edit_survives_stepping_away_and_back(window):
    paint_on(window)
    edited = window.label_controller.working_label().copy()

    window.label_panel.step_frame(1)
    window.label_panel.step_frame(-1)

    np.testing.assert_array_equal(window.label_controller.working_label(), edited)


def test_undo_reverses_a_stroke_through_the_panel(window):
    paint_on(window)
    assert window.label_panel.undo_button.isEnabled()

    window.label_panel.undo()

    assert window.label_controller.pending_count() == 1, "still reviewed, not edited"
    assert window.label_controller.session.edited_keys() == set()
    assert window.label_panel.redo_button.isEnabled()


def test_the_timeline_marks_the_frames_with_unsaved_edits(window):
    paint_on(window)

    assert window.label_controller.edited_positions() == {0}


def test_the_class_list_offers_background_as_an_eraser(window):
    names = [
        window.label_panel.class_combo.itemText(index)
        for index in range(window.label_panel.class_combo.count())
    ]

    assert any("erase" in name for name in names)


def test_choosing_a_class_sets_what_the_brush_writes(window):
    combo = window.label_panel.class_combo
    combo.setCurrentIndex(combo.count() - 1)

    assert window.label_panel.canvas.paint_value == combo.currentData()


def test_locking_a_class_reaches_the_canvas(window):
    class_index, check = next(iter(window.label_panel.lock_checks.items()))

    check.setChecked(True)

    assert class_index in window.label_panel.canvas.protected


def test_the_brush_slider_sets_the_radius(window):
    window.label_panel.brush_slider.setValue(25)

    assert window.label_panel.canvas.brush_radius == 25
    assert window.label_panel.brush_value.text() == "25"


def test_an_image_with_no_label_can_still_be_annotated(qt_app, config, tmp_path):
    root = tmp_path / "unlabelled"
    build_dataset(root, acquisitions=1, modalities=("SEM",), labels="none")
    for tiff in root.rglob("*.tif"):
        tifffile.imwrite(str(tiff), np.zeros((32, 40), dtype=np.uint8))

    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller, settings=QSettings(str(tmp_path / "p3.ini"), QSettings.IniFormat)
    )
    try:
        window.scan(str(root))
        window.tabs.setCurrentWidget(window.label_panel)
        _settle(qt_app, lambda: window.label_controller.cached_image() is not None)

        working = window.label_controller.working_label()

        assert working is not None, "an unlabelled image must be annotatable"
        assert working.shape == (32, 40)
        assert not working.any()
        assert window.label_controller.editable
        assert window.label_controller.current_key.name.endswith(".tif")
    finally:
        window.close()


def test_switching_lamella_does_not_carry_edits_across(qt_app, config, tmp_path):
    root = tmp_path / "two"
    build_dataset(root, lamellae=2, acquisitions=1)
    for tiff in root.rglob("*.tif"):
        tifffile.imwrite(str(tiff), np.zeros((32, 40), dtype=np.uint8))

    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller, settings=QSettings(str(tmp_path / "p4.ini"), QSettings.IniFormat)
    )
    try:
        window.scan(str(root))
        window.tabs.setCurrentWidget(window.label_panel)
        _settle(qt_app, lambda: window.label_controller.cached_image() is not None)
        paint_on(window)
        assert window.label_controller.pending_count() == 1

        window._step_label_lamella(1)

        assert window.label_controller.pending_count() == 0
    finally:
        window.close()


# ---------------------------------------------------------------------------
# The fill tool
# ---------------------------------------------------------------------------

def click(canvas, position, button=None):
    """Press and release at one point, returning the patches emitted."""
    patches = []
    canvas.edited.connect(patches.append)
    canvas.mousePressEvent(mouse_event(QEvent.MouseButtonPress, position, button))
    canvas.mouseReleaseEvent(mouse_event(QEvent.MouseButtonRelease, position, button))
    return patches


def test_a_click_floods_the_area_it_landed_in(canvas):
    canvas.set_tool(TOOL_FILL)
    canvas.set_paint_value(2)

    patches = click(canvas, QPoint(100, 100))

    assert len(patches) == 1
    assert (canvas._label_array == 2).all(), "an all-background label fills entirely"


def test_a_fill_stops_at_a_boundary(canvas):
    canvas._label_array[:, 40:] = 1
    canvas.set_tool(TOOL_FILL)
    canvas.set_paint_value(2)

    click(canvas, QPoint(100, 100))

    assert (canvas._label_array[:, 40:] == 1).all(), "the fill crossed the boundary"
    assert (canvas._label_array[:, :40] == 2).all()


def test_a_fill_respects_a_locked_class(canvas):
    canvas._label_array[:] = 1
    canvas.set_protected({1})
    canvas.set_tool(TOOL_FILL)
    canvas.set_paint_value(2)

    patches = click(canvas, QPoint(100, 100))

    assert patches == []
    assert (canvas._label_array == 1).all()


def test_a_fill_onto_its_own_class_is_not_an_edit(canvas):
    canvas.set_tool(TOOL_FILL)
    canvas.set_paint_value(0)

    assert click(canvas, QPoint(100, 100)) == []


def test_dragging_with_the_fill_tool_does_not_draw_a_stroke(canvas):
    canvas.set_tool(TOOL_FILL)
    canvas.set_paint_value(2)
    canvas._label_array[:] = 1

    patches = drag(canvas, QPoint(90, 90), QPoint(110, 110))

    assert len(patches) == 1, "the press fills; the drag adds nothing"


def test_the_fill_is_refused_when_editing_is_off(canvas):
    canvas.set_tool(TOOL_FILL)
    canvas.set_editable(False)

    assert click(canvas, QPoint(100, 100)) == []


def test_holding_space_pans_rather_than_filling(canvas):
    canvas.set_tool(TOOL_FILL)
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))

    assert click(canvas, QPoint(100, 100)) == []


def test_an_unknown_tool_is_rejected(canvas):
    with pytest.raises(ValueError, match="Unknown tool"):
        canvas.set_tool("smudge")


def test_the_brush_outline_is_hidden_while_filling(canvas):
    canvas.set_tool(TOOL_FILL)

    canvas.mouseMoveEvent(mouse_event(QEvent.MouseMove, QPoint(100, 100), Qt.NoButton))

    assert not canvas._brush_item.isVisible()


def test_a_fill_can_be_undone(window):
    window.label_panel.set_tool(TOOL_FILL)
    window.label_panel.canvas.resize(200, 200)
    window.label_panel.canvas.set_paint_value(1)
    click(window.label_panel.canvas, QPoint(100, 100))
    assert window.label_controller.session.edited_keys()

    window.label_panel.undo()

    assert window.label_controller.session.edited_keys() == set()


def test_choosing_the_fill_switches_off_the_brush_size(window):
    window.label_panel.set_tool(TOOL_FILL)

    assert window.label_panel.canvas.tool == TOOL_FILL
    assert not window.label_panel.brush_slider.isEnabled()

    window.label_panel.set_tool(TOOL_BRUSH)

    assert window.label_panel.brush_slider.isEnabled()


def test_the_help_lists_both_tools(window):
    help_text = window.shortcut_help(TAB_LABEL)

    assert "Brush tool" in help_text
    assert "Fill tool" in help_text
