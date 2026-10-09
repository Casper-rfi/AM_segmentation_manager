"""Tests for the Label tab: its canvas, its panel, and its place in the window.

Qt's offscreen platform, so no display is needed. The frames are real files on
disk, because the point of the tab is that it reads them at full size.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from conftest import build_dataset, destroy, settle

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
    from qtpy.QtGui import (
        QEnterEvent,
        QFocusEvent,
        QImage,
        QMouseEvent,
        QWheelEvent,
    )
    from qtpy.QtWidgets import (
        QGraphicsView,
        QLabel,
        QMessageBox,
        QPushButton,
        QWidget,
    )

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
    from am_segmentation_manager.gui.widgets.label_panel import BRUSH_WIDTHS
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE, TAB_LABEL
    from am_segmentation_manager.operations.label_edit import new_label

CONFIG_COLOURS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)}


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
def unlabelled(qt_app, config, tmp_path: Path):
    """A window on a project whose images have no label file at all.

    What proves a save can create a label rather than only replace one, which
    is the case the record has to be told about.
    """
    root = tmp_path / "bare"
    dataset = build_dataset(root, acquisitions=1, labels="none")
    for modality in ("SEM", "FIB"):
        tifffile.imwrite(
            str(dataset.image(modality, 0)), np.full((48, 64), 120, dtype=np.uint8)
        )
    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller,
        settings=QSettings(str(tmp_path / "bare.ini"), QSettings.IniFormat),
    )
    window.scan(str(root))
    window.tabs.setCurrentWidget(window.label_panel)
    settle(qt_app, lambda: drawn(window))
    yield window
    window.label_controller.discard_edits()
    destroy(window)


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
    settle(qt_app, lambda: drawn(window))
    yield window
    window.label_controller.discard_edits()
    destroy(window)


def drawn(window) -> bool:
    """Whether a frame has reached the canvas, rather than merely the cache.

    The loader fills the cache on its own thread and *then* reports the frame
    with a queued signal that the window turns into a draw, so a cached frame
    is not yet a drawn one. Waiting on the cache is what made these tests
    flaky.
    """
    return not window.label_panel.canvas._item.pixmap().isNull()


def draws(window):
    """Return a counter of the frames delivered to the Label tab.

    Connected after the window's own handler, so Qt runs it second: once this
    has counted a frame, the canvas has already been given it. For a redraw,
    where the canvas holds the previous frame's pixmap and :func:`drawn`
    therefore answers True before anything has changed.
    """
    seen: list[object] = []
    window.label_loader.worker.loaded.connect(lambda *_: seen.append(None))
    return lambda: len(seen)


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


def test_opening_the_tab_gives_the_canvas_the_keyboard(window):
    window.tabs.setCurrentWidget(window.image_panel)
    window.tabs.setCurrentWidget(window.label_panel)

    assert window.focusWidget() is window.label_panel.canvas, (
        "space has to pan without clicking the pane first"
    )


def test_the_tools_do_not_take_the_keyboard(window):
    panel = window.label_panel

    taking = [
        widget for widget in panel.findChildren(QWidget)
        if widget is not panel.canvas
        and widget.focusPolicy() != Qt.NoFocus
        # A dropdown's own list is exempt: it lives in a popup window of its
        # own, exists only while that popup is open, and has to take the arrow
        # keys while it is. Nothing on the row itself may.
        and widget.window() is panel.window()
    ]

    assert taking == [], "a control holding focus swallows the space that pans"


@pytest.fixture
def variants(qt_app, config, tmp_path: Path):
    """A window on a lamella whose SEM images have a second condition."""
    root = tmp_path / "conditions"
    build_dataset(
        root, acquisitions=2, modalities=("SEM",), sem_variants=("sem_8_fi_1",),
    )
    for tiff in root.rglob("*.tif"):
        tifffile.imwrite(str(tiff), np.zeros((32, 40), dtype=np.uint8))

    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller, settings=QSettings(str(tmp_path / "v.ini"), QSettings.IniFormat)
    )
    window.scan(str(root))
    window.tabs.setCurrentWidget(window.label_panel)
    settle(qt_app, lambda: drawn(window))
    yield window
    window.label_controller.discard_edits()
    destroy(window)


def test_two_families_are_offered_under_their_headings(qt_app, config, tmp_path: Path):
    """Choosing the other family proofreads its label, which the status line names."""
    root = tmp_path / "families"
    build_dataset(
        root, acquisitions=2, modalities=("SEM",), sem_variants=("sem-5kV", "sem-5kV_fi4"),
    )
    for tiff in root.rglob("*.tif"):
        tifffile.imwrite(str(tiff), np.zeros((32, 40), dtype=np.uint8))
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config),
        settings=QSettings(str(tmp_path / "f.ini"), QSettings.IniFormat),
    )
    try:
        window.scan(str(root))
        window.tabs.setCurrentWidget(window.label_panel)
        settle(qt_app, lambda: drawn(window))
        panel = window.label_panel
        combo = panel.variant_combo
        assert [combo.itemText(row) for row in range(combo.count())] == [
            "sem", "sem", "sem-5kV", "sem-5kV", "sem-5kV_fi4",
        ]

        combo.setCurrentIndex(4)

        assert window.label_controller.record.condition == "sem-5kV_fi4"
        assert panel.status.full_text().endswith("label shared by sem-5kV, sem-5kV_fi4")
    finally:
        window.label_controller.discard_edits()
        destroy(window)


@pytest.fixture
def three_series_window(qt_app, config, tmp_path: Path):
    """A window whose lamella holds reference and alignment images too."""
    dataset = build_dataset(tmp_path, acquisitions=2)
    for modality in ("SEM", "FIB"):
        for index in range(2):
            tifffile.imwrite(
                str(dataset.image(modality, index)),
                np.full((48, 64), 120, dtype=np.uint8),
            )
            tifffile.imwrite(
                str(dataset.label(modality, index)), np.zeros((48, 64), dtype=np.uint8)
            )
    for name in ("ref_Polishing_start_eb.tif", "ref_alignment_ib.tif"):
        tifffile.imwrite(
            str(dataset.lamella / name), np.full((48, 64), 90, dtype=np.uint8)
        )

    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config),
        settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat),
    )
    window.scan(str(tmp_path))
    window.tabs.setCurrentWidget(window.label_panel)
    settle(qt_app, lambda: drawn(window))
    yield window
    window.label_controller.discard_edits()
    destroy(window)


def test_the_series_control_is_greyed_when_a_lamella_holds_one(window):
    combo = window.label_panel.series_combo

    assert combo.isHidden() is False
    assert combo.isEnabled() is False
    assert combo.currentText() == "Adaptive Polishing"


def test_the_tab_reviews_one_series_at_a_time(qt_app, three_series_window):
    """It used to open every series of the lamella as one list, which is two
    unrelated jobs to work through in one pass."""
    window = three_series_window
    panel = window.label_panel

    assert panel.series_combo.isEnabled() is True
    assert window.label_controller.count == 2, "the stage's frames alone"

    panel.series_combo.setCurrentIndex(panel.series_combo.findData("reference"))
    settle(qt_app, lambda: window.label_controller.series == "reference")

    assert window.label_controller.series == "reference"
    assert window.label_controller.count == 1


def test_choosing_the_alignment_series_shows_the_beam_it_was_imaged_with(
    qt_app, three_series_window
):
    window = three_series_window
    panel = window.label_panel

    panel.series_combo.setCurrentIndex(panel.series_combo.findData("alignment"))
    settle(qt_app, lambda: window.label_controller.modality == "FIB")

    assert window.label_controller.modality == "FIB"
    assert panel.modality_buttons["FIB"].isChecked() is True


def test_the_condition_control_is_greyed_when_there_is_no_choice(window):
    """Greyed rather than hidden, as in the curation view: it still says which
    condition is being annotated, and the row does not move."""
    # isHidden() rather than isVisible(): a child of a window that was never
    # shown is not visible either way, which would pass whatever the code did.
    assert window.label_panel.variant_combo.isHidden() is False
    assert window.label_panel.variant_label.isHidden() is False
    assert window.label_panel.variant_combo.isEnabled() is False


def test_the_conditions_of_an_acquisition_are_offered(variants):
    combo = variants.label_panel.variant_combo

    assert combo.isHidden() is False
    assert [combo.itemData(i) for i in range(combo.count())] == ["sem", "sem_8_fi_1"]
    assert combo.itemText(0) == "sem"


def test_choosing_a_condition_draws_it(qt_app, variants):
    panel = variants.label_panel
    panel.variant_combo.setCurrentIndex(1)
    settle(qt_app, lambda: variants.label_controller.cached_image() is not None)

    assert variants.label_controller.condition == "sem_8_fi_1"
    assert "8_fi_1" in str(variants.label_controller.record.image_path)


def test_hiding_the_image_leaves_the_label_on_screen(window):
    panel = window.label_panel

    panel.image_check.setChecked(False)

    assert not panel.canvas._item.isVisible(), "the acquisition is hidden"
    assert panel.canvas._label_item.isVisible(), (
        "the label is what you are looking for against the black"
    )


def test_the_image_comes_back(window):
    panel = window.label_panel
    panel.image_check.setChecked(False)

    panel.image_check.setChecked(True)

    assert panel.canvas._item.isVisible()


def test_the_hidden_image_survives_stepping_to_another_frame(qt_app, window):
    panel = window.label_panel
    panel.image_check.setChecked(False)

    panel.step_frame(1)
    settle(qt_app, lambda: window.label_controller.cached_image() is not None)
    panel.draw_current()

    assert not panel.canvas._item.isVisible(), (
        "a view choice is not undone by moving through the lamella"
    )


def test_finding_specks_sets_the_view_that_shows_them(window):
    panel = window.label_panel
    panel.alpha_slider.setValue(40)

    panel.speck_check.setChecked(True)

    assert panel.image_check.isChecked() is False, "the label alone, against black"
    assert panel.overlay_check.isChecked() is True
    assert panel.alpha_slider.value() == 100


def test_switching_the_filter_off_puts_the_view_back(window):
    panel = window.label_panel
    panel.alpha_slider.setValue(40)
    panel.image_check.setChecked(True)

    panel.speck_check.setChecked(True)
    panel.speck_check.setChecked(False)

    assert panel.image_check.isChecked() is True
    assert panel.alpha_slider.value() == 40, (
        "a filter that leaves the window rearranged is one you stop using"
    )


def test_a_speck_is_ringed_where_it_is(window):
    panel = window.label_panel
    paint_on(window, value=2)   # a stroke, not a speck
    panel.canvas.set_paint_value(1)
    stray(panel, 7, 9)

    panel.speck_check.setChecked(True)

    rings = panel.canvas._speck_items
    assert len(rings) == 1
    assert (rings[0].pos().x(), rings[0].pos().y()) == (7.5, 9.5)


def test_the_canvas_zooms_one_step_out_past_fit(window):
    """A label is easiest to leave a pixel short at the frame's edge, so that
    edge has to come clear of the pane's border -- but only just clear."""
    from am_segmentation_manager.gui.widgets.image_view import (
        LABEL_MIN_FIT_FRACTION,
    )

    canvas = window.label_panel.canvas
    canvas.fit()

    for _ in range(10):
        canvas.zoom_by(0.5)

    assert canvas.transform().m11() == pytest.approx(
        canvas.fit_scale() * LABEL_MIN_FIT_FRACTION, rel=0.02
    )
    # About a twentieth of the pane on each side: enough to see the edge, and
    # not the fifth of it the curation panes used to leave.
    assert LABEL_MIN_FIT_FRACTION == pytest.approx(1 / 1.1)


def _drawn(icon) -> bytes:
    """Return an icon's pixels, a drawn mark being testable no other way."""
    from am_segmentation_manager.gui.widgets.icons import ICON

    return icon.pixmap(ICON, ICON).toImage().constBits().tobytes()


def test_the_action_buttons_carry_their_own_marks(window):
    """These three are reached for without reading, so each needs a mark --
    and each a different one, or the marks say nothing."""
    from am_segmentation_manager.gui.widgets.label_panel import (
        MARK,
        redo_icon,
        save_icon,
        undo_icon,
    )

    panel = window.label_panel
    for button, icon in (
        (panel.save_button, save_icon(MARK)),
        (panel.undo_button, undo_icon(MARK)),
        (panel.redo_button, redo_icon(MARK)),
    ):
        assert not button.icon().isNull()
        assert _drawn(button.icon()) == _drawn(icon)

    marks = {_drawn(save_icon()), _drawn(undo_icon()), _drawn(redo_icon())}
    assert len(marks) == 3, "undo and redo are mirrored, not the same picture"


def test_a_drawn_mark_puts_ink_on_the_pixmap(qt_app):
    """A painter that draws nothing still returns an icon; only the pixels
    say whether there is a picture on it."""
    from am_segmentation_manager.gui.widgets.icons import (
        ICON,
        redo_icon,
        save_icon,
        undo_icon,
    )

    for icon in (save_icon(), undo_icon(), redo_icon()):
        image = icon.pixmap(ICON, ICON).toImage()
        painted = sum(
            image.pixelColor(x, y).alpha() > 0
            for x in range(ICON)
            for y in range(ICON)
        )
        assert painted > ICON, "too little ink to be a picture"


def test_the_pointer_is_polled_only_while_there_is_an_outline(window):
    """The poll exists to move the brush outline, so it runs exactly while
    there is one -- not while filling, panning or looking at a locked frame."""
    canvas = window.label_panel.canvas
    assert canvas.editable
    assert canvas._pointer_poll.isActive(), "the brush tool draws an outline"

    canvas.set_tool(TOOL_FILL)
    assert not canvas._pointer_poll.isActive()

    canvas.set_tool(TOOL_BRUSH)
    canvas.set_editable(False)
    assert not canvas._pointer_poll.isActive()


def test_the_poll_leaves_an_active_window_alone(window, monkeypatch):
    """While the window is in front, real pointer events arrive and the poll
    must not spend anything looking the pointer up itself."""
    canvas = window.label_panel.canvas
    synced = []
    monkeypatch.setattr(canvas, "_sync_brush", lambda: synced.append(None))
    monkeypatch.setattr(canvas, "isActiveWindow", lambda: True)

    canvas._poll_pointer()

    assert not synced

    monkeypatch.setattr(canvas, "isActiveWindow", lambda: False)
    canvas._poll_pointer()

    assert synced, "a window that is not in front gets no pointer motion at all"


def test_the_wheel_holds_the_point_it_is_pointed_at(window):
    """Zooming has to keep the pixel under the pointer under the pointer.

    Anchored on the wheel event's own position rather than on the scene point
    Qt recorded from the last mouse event the view handled, which is a
    different thing whenever no such event arrived.
    """
    canvas = window.label_panel.canvas
    canvas.resize(200, 200)
    canvas.fit()
    at = QPoint(150, 60)
    before = canvas.mapToScene(at)

    canvas.wheelEvent(wheel_event(at, 120))

    after = canvas.mapToScene(at)
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()), abs=1.0)


def test_painting_does_not_move_where_the_wheel_zooms(window):
    """The canvas accepts a mouse move without passing it on for the whole of
    a stroke, so Qt's own recorded anchor freezes wherever the stroke began.
    Nothing may depend on it."""
    canvas = window.label_panel.canvas
    canvas.resize(200, 200)
    canvas.fit()
    drag(canvas, QPoint(30, 30), QPoint(60, 60))
    at = QPoint(170, 20)
    before = canvas.mapToScene(at)

    canvas.wheelEvent(wheel_event(at, 120))

    after = canvas.mapToScene(at)
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()), abs=1.0)


def test_zooming_out_holds_the_point_too(window):
    """Both directions, until the floor takes over and centres instead."""
    canvas = window.label_panel.canvas
    canvas.resize(200, 200)
    canvas.fit()
    for _ in range(4):
        canvas.zoom_by(1.25)
    at = QPoint(40, 150)
    before = canvas.mapToScene(at)

    canvas.wheelEvent(wheel_event(at, -120))

    after = canvas.mapToScene(at)
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()), abs=1.0)


def test_zooming_fully_out_centres_the_frame(window):
    """The wheel keeps the point under the pointer fixed, which is what makes
    zooming *in* follow the eye -- and what left a fully zoomed-out frame in
    whichever corner the pointer happened to be over."""
    canvas = window.label_panel.canvas
    for _ in range(4):
        canvas.zoom_by(1.25)
    canvas.horizontalScrollBar().setValue(canvas.horizontalScrollBar().maximum())
    canvas.verticalScrollBar().setValue(canvas.verticalScrollBar().maximum())

    for _ in range(20):
        canvas.zoom_by(0.8)

    image = canvas._item.sceneBoundingRect().center()
    pane = canvas.mapToScene(canvas.viewport().rect().center())
    assert (pane.x(), pane.y()) == pytest.approx((image.x(), image.y()), abs=1.0)


def test_a_ring_keeps_its_size_on_screen_at_any_zoom(window):
    panel = window.label_panel
    stray(panel, 7, 9)
    panel.speck_check.setChecked(True)
    ring = panel.canvas._speck_items[0]
    before = ring.deviceTransform(panel.canvas.viewportTransform()).mapRect(
        ring.boundingRect()
    ).width()

    panel.canvas.zoom_by(4.0)

    after = ring.deviceTransform(panel.canvas.viewportTransform()).mapRect(
        ring.boundingRect()
    ).width()
    assert after == pytest.approx(before), (
        "a ring scaled with the image would be smaller than the speck inside it"
    )


def test_the_series_shows_no_counts_until_the_filter_is_on(window):
    """The marks exist because the reviewer asked for them: a series quietly
    keeping them would answer a question nobody put."""
    assert window.label_panel.controller.timeline_specks() == {}


def test_switching_the_filter_on_asks_for_the_counts(window):
    """The panel does not read the labels itself -- a sweep of a lamella is
    seconds of file reading, and the panel draws."""
    asked = []
    window.label_panel.speck_counts_needed.connect(lambda: asked.append(1))

    window.label_panel.speck_check.setChecked(True)

    assert asked


def test_a_frame_nothing_has_counted_reports_none_rather_than_zero(window):
    """Which is what lets the timeline leave it blank while the sweep runs;
    zero would be a wrong answer for as long as the reading took."""
    controller = window.label_panel.controller
    controller.finding_specks = True

    counts = controller.timeline_specks()[controller.modality]

    assert any(count is None for count in counts), (
        "a frame the sweep has not reached must not read as having none"
    )


def test_a_counted_frame_is_not_asked_for_again(window):
    """The sweep is resumable, which is what makes abandoning one cheap."""
    controller = window.label_panel.controller
    controller.finding_specks = True
    wanted = controller.uncounted_labels()
    assert wanted, "nothing to count in the fixture"

    key, _path = wanted[0]
    controller.record_speck_count(key, 3)

    assert key not in [asked for asked, _ in controller.uncounted_labels()]
    assert controller.timeline_specks()[controller.modality].count(3) == 1


def test_the_edited_frame_keeps_its_own_count_over_the_sweep(window):
    """What is on screen carries strokes that are not saved, so the file the
    sweep reads no longer says the same thing."""
    controller = window.label_panel.controller
    controller.finding_specks = True
    key = controller.current_key
    controller.specks()                     # counts the frame on screen
    live = controller._speck_counts[key]

    controller.record_speck_count(key, live + 99)

    assert controller._speck_counts[key] == live


def test_a_new_size_forgets_every_count(window):
    """Every count in the series is wrong at a new size, not just this one's."""
    controller = window.label_panel.controller
    controller.finding_specks = True
    controller.specks()
    controller.record_speck_count(Path("elsewhere/label.tif"), 4)
    assert controller._speck_counts

    controller.set_speck_size(controller.speck_size + 1)

    assert controller._speck_counts == {}


def test_dragging_the_size_does_not_sweep_per_pixel(window):
    """A sweep per value of the handle would read the series once a pixel."""
    asked = []
    window.label_panel.speck_counts_needed.connect(lambda: asked.append(1))
    window.label_panel.speck_check.setChecked(True)
    asked.clear()

    for size in range(20, 30):
        window.label_panel.speck_slider.setValue(size)

    assert not asked, "the sweep must wait for the handle to stop"


def test_the_filter_opens_wide_enough_to_sweep_a_label(window):
    """One size answers both questions now: the report over the collection
    starts where proofreading settled, false positives included."""
    from am_segmentation_manager.operations.label_measure import (
        DEFAULT_SPECK_PIXELS,
    )

    assert window.label_panel.controller.speck_size == DEFAULT_SPECK_PIXELS == 128
    assert window.label_panel.speck_slider.value() == DEFAULT_SPECK_PIXELS
    assert window.label_panel.speck_slider.maximum() >= DEFAULT_SPECK_PIXELS, (
        "the slider has to reach the size it starts at, and past it"
    )


def test_switching_the_filter_off_takes_the_rings_away(window):
    panel = window.label_panel
    stray(panel, 7, 9)

    panel.speck_check.setChecked(True)
    panel.speck_check.setChecked(False)

    assert panel.canvas._speck_items == []


def test_the_frame_on_screen_gets_its_count_onto_the_timeline(window):
    """There is no count beside the slider any more: the timeline carries one
    per frame, and this frame's is the one that replaced it."""
    panel = window.label_panel
    stray(panel, 7, 9)

    panel.speck_check.setChecked(True)

    counts = panel.controller.timeline_specks()[panel.controller.modality]
    assert counts[panel.controller.position] == 1


def test_painting_out_the_last_speck_takes_its_mark_away(window):
    """The reported fault: the count reached the controller and the timeline
    went on showing the old one, so a frame stayed marked after it was put
    right."""
    panel = window.label_panel
    stray(panel, 7, 9)
    panel.speck_check.setChecked(True)
    position = panel.controller.position
    modality = panel.controller.modality
    assert panel.timeline._specks[modality][position] == 1

    from am_segmentation_manager.operations.label_edit import paint

    label = panel.controller.working_label()
    panel.controller.record_edit(paint(label, [(7, 9)], 0, 0))
    panel._refresh_specks()

    # Asked of the widget, not the controller: the controller was right all
    # along, and what was wrong was that nothing redrew the timeline with it.
    assert panel.timeline._specks[modality][position] == 0


def test_an_edit_lists_the_label_as_saved_until_undone(window):
    """Holding B mid-edit shows what the edits started from, which until a save
    is in no other row."""
    from am_segmentation_manager.operations.label_edit import paint

    panel = window.label_panel
    controller = panel.controller
    on_disk = controller.working_label().copy()
    assert panel.versions_controller.saved is None

    panel._stroke_finished(paint(controller.working_label(), [(30, 30)], 0, 1))

    assert panel.versions_controller.rows[0].source == "saved"
    assert panel.versions.chosen == 0, "a comparison starts on the label as saved"
    assert not panel.versions.restore_button.isEnabled()
    panel._compare_key_held(True)
    assert np.array_equal(panel._label_to_draw(), on_disk)
    assert not np.array_equal(controller.working_label(), on_disk)
    panel._compare_key_held(False)

    panel.undo()

    assert panel.versions_controller.saved is None
    assert all(row.source != "saved" for row in panel.versions_controller.rows)


def test_leaving_a_frame_before_the_recount_still_fixes_its_mark(window):
    """The reported fault. A stroke defers the recount by 400ms so it does not
    run mid-drag; move on inside that window and the recount fires for the
    frame you moved *to*, leaving the one you edited showing what it held
    before -- and it is never asked again, a frame with a count being one the
    sweep passes over."""
    from am_segmentation_manager.operations.label_edit import paint

    panel = window.label_panel
    controller = panel.controller
    panel.speck_check.setChecked(True)
    key = controller.current_key
    before = controller.timeline_specks()[controller.modality][controller.position]

    # A stroke through the panel, as the canvas makes one, then away at once.
    panel._stroke_finished(paint(controller.working_label(), [(7, 9)], 0, 1))
    position = controller.position
    controller.step(1)
    controller.working_label()

    sources = dict(controller.uncounted_labels())
    for asked, source in sources.items():
        controller.record_speck_count(asked, 99 if asked == key else 0)
    after = controller.timeline_specks()[controller.modality][position]

    assert after != before, (
        "the frame kept the count it had before the stroke, and nothing asks again"
    )


def test_stepping_away_hands_the_edited_label_to_the_sweep(window):
    """End to end, through the panel's own navigation: `refresh` draws the new
    frame before it asks for loads, and drawing is what puts the edited label
    away in the session -- so by the time the sweep runs there is a copy of it
    to count, and the mark is right one frame later rather than two."""
    from am_segmentation_manager.operations.label_edit import paint

    panel = window.label_panel
    controller = panel.controller
    panel.speck_check.setChecked(True)
    key = controller.current_key
    position = controller.position

    panel._stroke_finished(paint(controller.working_label(), [(7, 9)], 0, 1))
    panel.step_frame(1)                      # the panel's own way of moving

    sources = dict(controller.uncounted_labels())
    assert isinstance(sources.get(key), bytes), (
        "the edited frame should reach the sweep as the session's own copy"
    )
    controller.record_speck_count(key, sources and 1)

    assert controller.timeline_specks()[controller.modality][position] == 1


def test_undoing_a_stroke_puts_the_count_back(window):
    """Undo changes the label as surely as the stroke did, and left the count
    saying what the stroke had made it."""
    panel = window.label_panel
    controller = panel.controller
    panel.speck_check.setChecked(True)
    position = controller.position
    modality = controller.modality
    clean = controller.timeline_specks()[modality][position]

    stray(panel, 7, 9)
    panel._refresh_specks()
    assert controller.timeline_specks()[modality][position] == clean + 1

    panel.undo()
    panel._refresh_specks()

    assert controller.timeline_specks()[modality][position] == clean


def test_an_edited_frame_is_counted_from_memory_not_from_its_file(window):
    """The reported fault. Widening the size clears every count, so the sweep
    that follows re-reads frames the reviewer has already edited -- and their
    files still hold the specks that were painted out, which came back onto the
    timeline until the edits were saved."""
    panel = window.label_panel
    controller = panel.controller
    stray(panel, 7, 9)                       # an unsaved stroke on this frame
    panel.speck_check.setChecked(True)
    key = controller.current_key
    assert key in controller.session.edited_keys()

    # Move away and draw the next frame, which is what puts the edit away in
    # the session: the session releases the open label only when another is
    # opened, and opening is what drawing does.
    controller.step(1)
    controller.working_label()
    controller.set_speck_size(controller.speck_size + 1)

    sources = dict(controller.uncounted_labels())
    assert isinstance(sources.get(key), bytes), (
        "an edited frame must be counted from the session's copy, not its file"
    )


def test_an_unedited_frame_is_counted_from_its_file(window):
    """Which is the ordinary case, and the cheap one: no array is held for it
    and none needs to be."""
    panel = window.label_panel
    controller = panel.controller
    controller.finding_specks = True

    sources = [source for _key, source in controller.uncounted_labels()]

    assert sources, "nothing to count in the fixture"
    assert all(isinstance(source, Path) for source in sources)


def test_stepping_brings_a_speck_into_the_middle(window):
    panel = window.label_panel
    stray(panel, 40, 30)
    panel.speck_check.setChecked(True)
    panel.canvas.zoom_by(8.0)

    panel.step_speck(1)

    centre = panel.canvas.mapToScene(panel.canvas.viewport().rect().center())
    assert centre.x() == pytest.approx(40.5, abs=1.5)
    assert centre.y() == pytest.approx(30.5, abs=1.5)


def stray(panel, x, y):
    """Paint a single pixel, which is what a speck is."""
    from am_segmentation_manager.operations.label_edit import paint

    label = panel.controller.working_label()
    panel.controller.record_edit(paint(label, [(x, y)], 0, 1))


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
    delivered = draws(window)

    window.label_panel.set_modality("FIB")
    settle(qt_app, delivered)

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


def test_returning_to_the_tab_draws_a_label_converted_meanwhile(qt_app, unlabelled):
    """The reported fault: a blank opened for an unlabelled image stood in for
    the label Curate converted there, until the reviewer stepped away and back."""
    window = unlabelled
    canvas = window.label_panel.canvas
    assert not canvas._label_array.any(), "the image opens on a blank label"
    window.tabs.setCurrentWidget(window.curate)
    converted = window.label_controller.current_key
    label = np.zeros((48, 64), dtype=np.uint8)
    label[:10] = 2
    converted.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(converted), label)
    window.controller.attach_written([converted])

    count = draws(window)
    window.tabs.setCurrentWidget(window.label_panel)
    settle(qt_app, lambda: count() >= 1, what="the converted frame")

    assert (canvas._label_array[:10] == 2).all()


def test_a_frame_still_loading_cannot_be_painted(window):
    """The canvas still holds the frame before it, whose label is no longer
    the one a stroke would be recorded against."""
    panel = window.label_panel
    assert panel.canvas.editable

    window.label_controller.cache.clear()
    panel.draw_current()

    assert not panel.canvas.editable


def test_a_scan_from_the_tab_paints_into_the_label_the_session_holds(
    qt_app, window, project
):
    """A stroke on any other array is drawn and then never saved."""
    count = draws(window)
    window.scan(str(project))
    settle(qt_app, lambda: count() >= 1, what="the frame after the scan")

    assert window.label_panel.canvas._label_array is window.label_controller.session.open_label


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
        destroy(window)


def test_the_tab_s_keys_are_live_only_on_it(window):
    window.tabs.setCurrentWidget(window.label_panel)
    live = {
        binding.key
        for binding, shortcut in zip(window._keys.bindings, window._keys.shortcuts)
        if shortcut.isEnabled()
    }
    assert "Right" in live

    window.tabs.setCurrentWidget(window.curate)
    label_keys = [b for b in window._keys.bindings if b.tab == TAB_LABEL]

    assert label_keys, "the tab should have keys of its own"
    assert all(
        not shortcut.isEnabled()
        for binding, shortcut in zip(window._keys.bindings, window._keys.shortcuts)
        if binding.tab == TAB_LABEL
    )


def keys_of(window, tab):
    """Return the keys bound on *tab*, by what they do."""
    return {b.description: b.key for b in window._keys.bindings if b.tab == tab}


def test_every_class_sits_beside_its_own_padlock(window):
    """One cell each, so which lock protects which class needs no reading."""
    panel = window.label_panel

    assert set(panel.lock_buttons) == set(panel.paint_buttons)
    for class_index, paint in panel.paint_buttons.items():
        lock = panel.lock_buttons[class_index]
        assert lock.parentWidget() is paint.parentWidget(), f"class {class_index}"
        assert lock.height() == paint.height(), f"class {class_index}"


def test_a_long_class_name_does_not_make_its_cell_wider(window):
    """Background's caption says it erases, and its cell is no wider for it."""
    cells = {lock.parentWidget() for lock in window.label_panel.lock_buttons.values()}

    assert len({cell.width() for cell in cells}) == 1


def test_a_padlock_is_big_enough_to_hit_beside_the_canvas(window):
    from am_segmentation_manager.gui.widgets.label_panel import PADLOCK

    assert PADLOCK >= 28
    for class_index, lock in window.label_panel.lock_buttons.items():
        assert lock.width() >= PADLOCK and lock.height() >= PADLOCK, class_index


def test_a_class_button_is_its_name_alone(window):
    """The row is compact enough for one line; what painting background does
    is on its tooltip rather than in its caption."""
    from qtpy.QtWidgets import QLabel

    panel = window.label_panel
    captions = {label.text() for label in panel.findChildren(QLabel)}

    assert not {"Edit", "Paint"} & captions
    assert panel.paint_buttons[BACKGROUND_CLASS].text() == "background"
    assert "erases" in panel.paint_buttons[BACKGROUND_CLASS].toolTip()


def test_the_tools_and_the_classes_are_one_row_that_wraps(window):
    from am_segmentation_manager.gui.widgets.controls import FlowArea

    panel = window.label_panel

    assert isinstance(panel.tools, FlowArea)
    assert panel.tools.layout().minimumSize().width() == 0
    for button in (*panel.tool_buttons.values(), *panel.paint_buttons.values()):
        assert panel.tools.isAncestorOf(button)


def test_undo_redo_and_save_are_marks_under_the_beam_buttons(window):
    """Where image software keeps them; the tooltip carries the name."""
    panel = window.label_panel
    holder = panel.modality_control.parentWidget()

    for button, word in (
        (panel.undo_button, "Undo"), (panel.redo_button, "Redo"), (panel.save_button, "Save"),
    ):
        assert holder.isAncestorOf(button)
        assert button.text() == ""
        assert word in button.toolTip()


def test_every_mark_on_the_bars_is_drawn_one_size(window):
    """Balanced: a padlock larger than the tools beside it read as the more
    important control, and the tools' captions smaller than their marks."""
    from qtpy.QtCore import QSize

    from am_segmentation_manager.gui.widgets.label_panel import ACTION_BUTTON, MARK

    panel = window.label_panel
    marked = [
        panel.undo_button, panel.redo_button, panel.save_button,
        *panel.tool_buttons.values(), *panel.paint_buttons.values(),
        *panel.lock_buttons.values(),
    ]

    assert {button.iconSize() for button in marked} == {QSize(MARK, MARK)}
    for button in (panel.undo_button, panel.redo_button, panel.save_button):
        assert (button.width(), button.height()) == ACTION_BUTTON
    plain = panel.fit_button.font().pointSizeF()
    for button in (*panel.tool_buttons.values(), *panel.paint_buttons.values()):
        assert button.font().pointSizeF() == plain + 1


def test_the_padlock_is_not_shrunk_by_its_button(window):
    """Asserted on the pixels: the icon size was set once before and the
    button's padding still drew the padlock ten pixels wide."""
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui.widgets.icons import ICON
    from am_segmentation_manager.gui.widgets.theme import TEXT

    window.resize(1512, 900)
    window.show()
    window.tabs.setCurrentWidget(window.label_panel)
    lock = next(iter(window.label_panel.lock_buttons.values()))
    image = lock.grab().toImage()
    ink = QColor(TEXT).rgb()
    columns = [
        x for x in range(image.width())
        if any(image.pixel(x, y) == ink for y in range(image.height()))
    ]

    assert columns and columns[-1] - columns[0] + 1 > ICON * 0.6


def test_nothing_to_say_about_the_label_takes_no_room(window):
    """An empty badge line still holds a row of the canvas's height, which is
    what put these in the tools row instead of on one of their own."""
    panel = window.label_panel
    panel._set_badges([])
    assert panel.badges.isHidden()

    panel._set_badges([("something happened", "warning")])

    assert not panel.badges.isHidden()


def test_the_status_line_elides_rather_than_being_cut_off(window):
    """It shares a row now, and a stage name is a timestamp: clipping would
    take the ends off without saying it had."""
    panel = window.label_panel
    panel.status.setFixedWidth(60)

    panel.status.setText("a" * 200)

    assert panel.status.full_text() == "a" * 200
    assert len(panel.status.text()) < 200
    assert panel.status.toolTip() == "a" * 200


def test_the_tools_sit_under_the_hand_that_paints(window):
    keys = keys_of(window, TAB_LABEL)

    assert keys["Brush tool"] == "D"
    assert keys["Fill tool"] == "F"


def test_the_overlay_and_fit_use_the_same_keys_in_both_tabs(window):
    label, curate = keys_of(window, TAB_LABEL), keys_of(window, TAB_CURATE)

    assert label["Show / hide the label"] == curate["Show / hide the label overlay"]
    assert label["Fit the image to the pane"] == curate["Fit the view to the pane"]


def test_a_picked_class_is_the_one_painted_and_the_one_shown(window):
    panel = window.label_panel
    panel._select_class(1)

    panel.canvas.class_picked.emit(2)

    assert panel.canvas.paint_value == 2
    assert panel.paint_buttons[2].isChecked()


def test_a_class_the_configuration_does_not_define_is_not_picked(window):
    """It is drawn in the colour kept for an unknown class, and has no button."""
    panel = window.label_panel
    panel._select_class(1)
    notices = []
    panel.notice.connect(notices.append)

    panel.canvas.class_picked.emit(7)

    assert panel.canvas.paint_value == 1
    assert panel.paint_buttons[1].isChecked()
    assert notices and "7" in notices[0]


def test_a_class_is_chosen_by_one_key_rather_than_stepped_to(window):
    from am_segmentation_manager.gui.widgets.label_panel import CLASS_KEYS

    panel = window.label_panel
    bound = [b for b in window._keys.bindings if b.tab == TAB_LABEL and b.key in CLASS_KEYS]

    assert len(bound) == len(panel.class_names())
    for binding in bound:
        binding.action()
        assert panel.paint_buttons[panel.paint_class].text() in binding.description


def test_shift_and_a_class_key_lock_that_class_alone(window):
    """The padlock a key reaches is read off the key that paints its class."""
    from am_segmentation_manager.gui.widgets.label_panel import LOCK_KEYS

    panel = window.label_panel
    bound = {
        binding.key: binding for binding in window._keys.bindings
        if binding.tab == TAB_LABEL and binding.key in LOCK_KEYS
    }

    assert len(bound) == len(panel.class_names())
    for position, class_index in enumerate(panel._classes):
        bound[LOCK_KEYS[position]].action()
        assert panel.canvas.protected == {class_index}
        bound[LOCK_KEYS[position]].action()
        assert panel.canvas.protected == frozenset()


def test_one_key_locks_and_unlocks_every_class_but_background(window):
    panel = window.label_panel

    panel.toggle_all_locks()
    assert panel.canvas.protected == set(panel.lock_buttons) - {BACKGROUND_CLASS}

    panel.toggle_all_locks()
    assert panel.canvas.protected == frozenset()


def test_the_lock_key_leaves_background_s_own_lock_as_it_was(window):
    """Locking background changes what a stroke can reach, a choice of its own."""
    panel = window.label_panel
    panel.lock_buttons[BACKGROUND_CLASS].setChecked(True)

    panel.toggle_all_locks()
    panel.toggle_all_locks()

    assert panel.canvas.protected == {BACKGROUND_CLASS}


def test_locking_everything_leaves_the_eraser_alone(window):
    panel = window.label_panel

    panel.toggle_all_locks()

    assert BACKGROUND_CLASS not in panel.canvas.protected, (
        "painting background is how a stroke erases"
    )


def test_the_help_lists_the_tab_s_own_keys(window):
    help_text = window._keys.help_html(TAB_LABEL)

    assert "Next frame" in help_text
    assert "Show / hide the label" in help_text
    assert "Set state: corrected" not in help_text, "that is the Curate tab's"


def test_the_help_says_a_class_can_be_picked_off_the_label(window):
    help_text = window._keys.help_html(TAB_LABEL)

    assert "Shift-click the label" in help_text
    assert "Shift-click the label" not in window._keys.help_html(TAB_CURATE)


def test_the_curate_tab_keeps_its_own_keys(window):
    assert "Set state: corrected" in window._keys.help_html(TAB_CURATE)


# ---------------------------------------------------------------------------
# Painting on the canvas
# ---------------------------------------------------------------------------

def mouse_event(kind, position, button=None, buttons=None, modifiers=None):
    """Build a mouse event under either Qt version.

    Qt 6 requires a scene and a global position that Qt 5 does not take, so the
    two signatures are tried in turn.
    """
    button = button if button is not None else Qt.LeftButton
    buttons = buttons if buttons is not None else button
    modifiers = modifiers if modifiers is not None else Qt.NoModifier
    point = QPointF(position)
    try:
        return QMouseEvent(kind, point, point, point, button, buttons, modifiers)
    except TypeError:  # pragma: no cover - depends on the installed binding
        return QMouseEvent(kind, point, button, buttons, modifiers)


@pytest.fixture
def canvas(qt_app, config):
    """A canvas with a frame loaded and painting switched on."""
    canvas = LabelCanvas(config)
    canvas.resize(200, 200)
    canvas.set_frame(np.zeros((60, 80), dtype=np.uint8), new_label((60, 80)))
    canvas.set_editable(True)
    yield canvas
    destroy(canvas)


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


def test_a_locked_background_keeps_a_stroke_to_what_is_already_drawn(canvas):
    canvas._label_array[:, :40] = 1
    canvas.set_protected({BACKGROUND_CLASS})
    canvas.set_paint_value(2)

    drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert not (canvas._label_array[:, 40:] == 2).any(), "background was painted over"
    assert (canvas._label_array[:, :40] == 2).any(), "the class already drawn was not"


def test_a_locked_background_still_lets_a_stroke_erase(canvas):
    canvas._label_array[:] = 1
    canvas.set_protected({BACKGROUND_CLASS})
    canvas.set_paint_value(BACKGROUND_CLASS)

    drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert (canvas._label_array == BACKGROUND_CLASS).any(), "the eraser was locked too"


def test_nothing_is_painted_when_editing_is_off(canvas):
    canvas.set_editable(False)

    patches = drag(canvas, QPoint(80, 80), QPoint(120, 120))

    assert patches == []
    assert not (canvas._label_array != 0).any()


def shift_click(canvas, x, y):
    """Shift-click label pixel (x, y), returning the classes picked and the edits made."""
    picked, patches = [], []
    canvas.class_picked.connect(picked.append)
    canvas.edited.connect(patches.append)
    position = canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))
    for kind, buttons in (
        (QEvent.MouseButtonPress, Qt.LeftButton), (QEvent.MouseButtonRelease, Qt.NoButton)
    ):
        event = mouse_event(kind, position, Qt.LeftButton, buttons, Qt.ShiftModifier)
        if kind == QEvent.MouseButtonPress:
            canvas.mousePressEvent(event)
        else:
            canvas.mouseReleaseEvent(event)
    return picked, patches


# Named in full: the decorator is read where Qt, and so the canvas, is not installed.
@pytest.mark.parametrize("tool", ["brush", "fill"])
def test_shift_click_picks_the_class_under_the_pointer_and_paints_nothing(canvas, tool):
    canvas.set_tool(tool)
    canvas.set_paint_value(1)
    canvas._label_array[20:30, 30:40] = 2
    before = canvas._label_array.copy()

    picked, patches = shift_click(canvas, 35, 25)

    assert picked == [2]
    assert patches == [] and np.array_equal(canvas._label_array, before)


def test_shift_click_picks_background_too(canvas):
    """Which is how a stroke erases, so it is a class like any other here."""
    picked, _ = shift_click(canvas, 5, 5)

    assert picked == [BACKGROUND_CLASS]


def test_nothing_is_picked_when_editing_is_off(canvas):
    canvas.set_editable(False)

    picked, _ = shift_click(canvas, 5, 5)

    assert picked == []


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


def test_holding_space_shows_the_panning_hand(canvas):
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))

    assert canvas.viewport().cursor().shape() == Qt.OpenHandCursor


def test_the_panning_hand_survives_a_drag_across_the_frame(canvas):
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))
    # What Qt does as the pointer crosses a scene with no item cursor of its
    # own, which is what took the hand away mid-pan.
    canvas.viewport().setCursor(Qt.ArrowCursor)

    canvas.mouseMoveEvent(
        mouse_event(QEvent.MouseMove, QPoint(100, 100), Qt.NoButton)
    )

    assert canvas.viewport().cursor().shape() == Qt.OpenHandCursor


def test_releasing_space_takes_the_hand_away(canvas):
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))
    canvas.keyReleaseEvent(key_event(QEvent.KeyRelease, Qt.Key_Space))

    assert canvas.viewport().cursor().shape() != Qt.OpenHandCursor


def test_a_frame_that_cannot_be_painted_on_shows_the_hand(canvas):
    canvas.set_editable(False)

    assert canvas.viewport().cursor().shape() == Qt.OpenHandCursor, (
        "the left button pans there, so the hand is what it does"
    )


def test_the_fill_tool_shows_a_crosshair(canvas):
    canvas.set_tool(TOOL_FILL)

    assert canvas.viewport().cursor().shape() == Qt.CrossCursor


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


def test_the_brush_appears_where_the_pointer_already_is(canvas):
    canvas.set_editable(False)
    canvas._pointer_position = lambda: QPoint(100, 100)

    canvas.set_editable(True)

    assert canvas._brush_item.isVisible(), (
        "a frame finishes loading under a pointer that is not moving"
    )


def test_the_brush_comes_back_when_the_pan_key_is_let_go(canvas):
    """Holding space takes the ring away, and releasing it used to leave the
    pane with no ring and no cursor until the pointer was nudged."""
    canvas._pointer_position = lambda: QPoint(100, 100)
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))
    assert not canvas._brush_item.isVisible()

    canvas.keyReleaseEvent(key_event(QEvent.KeyRelease, Qt.Key_Space))

    assert canvas._brush_item.isVisible()


def test_the_brush_comes_back_when_the_brush_tool_is_chosen_again(canvas):
    """The fill has a crosshair and no ring; going back to the brush is a
    state change like any other, not something to wait for a mouse move."""
    canvas._pointer_position = lambda: QPoint(100, 100)
    canvas.set_tool(TOOL_FILL)
    assert not canvas._brush_item.isVisible()

    canvas.set_tool(TOOL_BRUSH)

    assert canvas._brush_item.isVisible()


def test_the_brush_follows_the_scene_when_the_view_moves_under_it(canvas):
    """The ring is drawn in scene coordinates, so anything that scrolls the
    view -- a middle-button pan, a fit, going to the next speck -- leaves it
    round a pixel the pointer is no longer over."""
    canvas._pointer_position = lambda: QPoint(100, 100)
    canvas.mouseMoveEvent(mouse_event(QEvent.MouseMove, QPoint(100, 100), Qt.NoButton))
    before = canvas._brush_item.rect().center()

    canvas.centre_on(5, 5)

    after = canvas._brush_item.rect().center()
    assert canvas.mapFromScene(after) == QPoint(100, 100)
    assert after != before, "the scene moved, so the ring is over a new pixel"


def test_the_brush_stays_hidden_when_the_pointer_is_elsewhere(canvas):
    canvas.set_editable(False)
    canvas._pointer_position = lambda: QPoint(-40, -40)

    canvas.set_editable(True)

    assert not canvas._brush_item.isVisible()


def test_the_pointer_arriving_takes_the_keyboard_and_shows_the_brush(canvas):
    canvas.clearFocus()

    canvas.enterEvent(enter_event(QPoint(90, 90)))

    # focusWidget rather than hasFocus: an offscreen window is never activated,
    # so Qt records which widget asked for the keyboard but grants it to none.
    assert canvas.focusWidget() is canvas, "space pans, and this widget reads it"
    assert canvas._brush_item.isVisible()


def test_losing_the_keyboard_stops_a_pan(canvas):
    canvas.keyPressEvent(key_event(QEvent.KeyPress, Qt.Key_Space))
    assert canvas.dragMode() == QGraphicsView.ScrollHandDrag

    canvas.focusOutEvent(QFocusEvent(QEvent.FocusOut))

    assert canvas.dragMode() == QGraphicsView.NoDrag, (
        "the release of a held space arrives wherever focus went"
    )


def enter_event(position):
    """Build a pointer-entered event, under either Qt version."""
    point = QPointF(position)
    try:  # Qt 6 wants the position three times over.
        return QEnterEvent(point, point, point)
    except TypeError:  # pragma: no cover - depends on the installed binding
        return QEnterEvent(position, point, point)


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
    assert window.label_panel.save_button.property("pending") is True


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


def test_the_class_row_starts_on_what_the_brush_writes(window):
    panel = window.label_panel
    checked = [index for index, b in panel.paint_buttons.items() if b.isChecked()]

    assert checked == [panel.canvas.paint_value]
    assert checked != [BACKGROUND_CLASS], (
        "a bar reading 'erase' while the brush paints a class costs an image")


def test_choosing_a_class_sets_what_the_brush_writes(window):
    panel = window.label_panel
    last = list(panel.paint_buttons)[-1]

    panel.paint_buttons[last].click()

    assert panel.canvas.paint_value == last


def test_the_paint_buttons_carry_the_class_colours(window):
    """They are the legend, so a separate one would repeat them."""
    panel = window.label_panel

    assert all(not button.icon().isNull() for button in panel.paint_buttons.values())
    assert not hasattr(panel, "legend")


def test_background_can_be_locked_like_any_other_class(window):
    panel = window.label_panel

    panel.lock_buttons[BACKGROUND_CLASS].setChecked(True)

    assert BACKGROUND_CLASS in panel.canvas.protected
    assert not hasattr(panel, "lock_all_button")


def test_the_tools_carry_an_icon_each(window):
    panel = window.label_panel

    assert all(not button.icon().isNull() for button in panel.tool_buttons.values())


def test_the_brush_size_belongs_to_the_brush(window):
    panel = window.label_panel

    panel.set_tool(TOOL_FILL)
    assert not panel.brush_slider.isEnabled(), "the fill has no size"

    panel.set_tool(TOOL_BRUSH)
    assert panel.brush_slider.isEnabled()


def test_locking_a_class_reaches_the_canvas(window):
    class_index, button = next(iter(window.label_panel.lock_buttons.items()))

    button.setChecked(True)

    assert class_index in window.label_panel.canvas.protected


def test_the_brush_slider_sets_the_width(window):
    panel = window.label_panel
    panel.brush_slider.setValue(6)

    assert panel.brush_width == BRUSH_WIDTHS[6]
    assert panel.canvas.brush_radius == (BRUSH_WIDTHS[6] - 1) // 2


def test_the_brush_reads_out_in_pixels_painted(window):
    panel = window.label_panel
    panel.brush_slider.setValue(1)

    assert panel.brush_width == 3
    assert panel.brush_value.text() == "3 px"


def test_the_smallest_brush_is_a_single_pixel(window):
    panel = window.label_panel
    panel.brush_slider.setValue(0)

    assert panel.brush_width == 1
    assert panel.canvas.brush_radius == 0


def test_stepping_the_brush_moves_one_offered_size(window):
    panel = window.label_panel
    panel.brush_slider.setValue(0)

    panel.step_brush(1)

    assert panel.brush_width == 3, (
        "one press from a single pixel is three across, not five"
    )


def test_shift_and_the_wheel_resize_the_brush(window):
    panel = window.label_panel
    panel.brush_slider.setValue(4)
    before = panel.brush_width

    panel.canvas.wheelEvent(wheel_event(QPoint(50, 50), 120, Qt.ShiftModifier))

    assert panel.brush_width > before


def test_shift_and_the_wheel_do_not_zoom(window):
    canvas = window.label_panel.canvas
    before = canvas.transform().m11()

    canvas.wheelEvent(wheel_event(QPoint(50, 50), 120, Qt.ShiftModifier))

    assert canvas.transform().m11() == before


def test_a_sideways_shifted_wheel_still_resizes_the_brush(window):
    # macOS reports Shift and the wheel as a horizontal scroll.
    panel = window.label_panel
    panel.brush_slider.setValue(4)
    before = panel.brush_width

    panel.canvas.wheelEvent(
        wheel_event(QPoint(50, 50), 120, Qt.ShiftModifier, sideways=True)
    )

    assert panel.brush_width > before


def wheel_event(position, delta, modifiers=None, *, sideways=False):
    """Build a wheel event, under either Qt version.

    *modifiers* defaults to none held. Spelling that as ``Qt.NoModifier`` in the
    signature would evaluate it at import, before the file's own skip applies.
    """
    modifiers = Qt.NoModifier if modifiers is None else modifiers
    angle = QPoint(delta, 0) if sideways else QPoint(0, delta)
    point = QPointF(position)
    try:  # Qt 6 wants the phase and the inversion as well.
        return QWheelEvent(
            point, point, QPoint(0, 0), angle,
            Qt.NoButton, modifiers, Qt.NoScrollPhase, False,
        )
    except TypeError:  # pragma: no cover - depends on the installed binding
        return QWheelEvent(position, delta, Qt.NoButton, modifiers, Qt.Vertical)
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
        settle(qt_app, lambda: window.label_controller.cached_image() is not None)

        working = window.label_controller.working_label()

        assert working is not None, "an unlabelled image must be annotatable"
        assert working.shape == (32, 40)
        assert not working.any()
        assert window.label_controller.editable
        assert window.label_controller.current_key.name.endswith(".tif")
    finally:
        destroy(window)


@pytest.fixture
def two_lamellae(qt_app, config, tmp_path):
    """A window on a two-lamella project, with the Label tab in front."""
    root = tmp_path / "two"
    build_dataset(root, lamellae=2, acquisitions=1)
    for tiff in root.rglob("*.tif"):
        tifffile.imwrite(str(tiff), np.zeros((32, 40), dtype=np.uint8))

    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(
        controller, settings=QSettings(str(tmp_path / "p4.ini"), QSettings.IniFormat)
    )
    window.scan(str(root))
    window.tabs.setCurrentWidget(window.label_panel)
    settle(qt_app, lambda: drawn(window))
    yield window
    window.label_controller.discard_edits()
    destroy(window)


def answer_warning(monkeypatch, answer):
    """Answer the unsaved-edits warning, recording that it was asked."""
    asked = []

    def warned(*args, **kwargs):
        asked.append(args[2] if len(args) > 2 else "")
        return answer

    monkeypatch.setattr("qtpy.QtWidgets.QMessageBox.warning", warned)
    return asked


def test_leaving_a_lamella_with_unsaved_edits_asks_first(two_lamellae, monkeypatch):
    paint_on(two_lamellae)
    asked = answer_warning(monkeypatch, QMessageBox.Cancel)
    first = two_lamellae.label_controller.lamella

    two_lamellae._step_label_lamella(1)

    assert asked, "the edits would have been thrown away silently"
    assert two_lamellae.label_controller.lamella is first
    assert two_lamellae.label_controller.pending_count() == 1


def test_discarding_unsaved_edits_moves_on(two_lamellae, monkeypatch):
    paint_on(two_lamellae)
    answer_warning(monkeypatch, QMessageBox.Discard)
    first = two_lamellae.label_controller.lamella

    two_lamellae._step_label_lamella(1)

    assert two_lamellae.label_controller.lamella is not first
    assert two_lamellae.label_controller.pending_count() == 0


def test_moving_lamella_with_nothing_pending_asks_nothing(two_lamellae, monkeypatch):
    asked = answer_warning(monkeypatch, QMessageBox.Cancel)

    two_lamellae._step_label_lamella(1)

    assert not asked


def test_closing_with_unsaved_edits_asks_first(two_lamellae, monkeypatch):
    paint_on(two_lamellae)
    answer_warning(monkeypatch, QMessageBox.Cancel)

    two_lamellae.close()

    assert two_lamellae.isVisible() or two_lamellae.label_controller.pending_count() == 1


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_with(window, mark_corrected):
    """Answer the save dialog without opening it."""
    window._ask_how_to_save = lambda plan: mark_corrected
    window.save_labels()


def saved(window):
    """Whether the window has caught up with a finished save.

    Waited on rather than the session's own count, which is cleared on the
    worker thread before the interface has redrawn.
    """
    return not window.label_panel.save_button.isEnabled()


def test_saving_writes_the_edited_label(qt_app, window):
    key = window.label_controller.current_key
    paint_on(window, value=2)
    before = key.read_bytes()

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    assert key.read_bytes() != before
    assert (tifffile.imread(str(key)) == 2).any(), "the stroke did not reach the file"


def test_the_canvas_shows_what_was_saved_rather_than_what_it_replaced(
    qt_app, window
):
    """A prepared frame holds the pixels its label had when it was read, and a
    save replaces the file underneath it. Without dropping that frame the
    canvas goes on drawing the label the save overwrote -- which reads as the
    save having silently failed."""
    key = window.label_controller.current_key
    paint_on(window, value=2)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    on_disk = tifffile.imread(str(key))
    on_canvas = window.label_controller.working_label()
    assert np.array_equal(np.asarray(on_canvas), on_disk), (
        "the canvas is still drawing the label the save replaced"
    )


def curate_label(window):
    """The label Curate holds for the SEM image on screen, or None if not held."""
    controller = window.controller
    prepared = controller.cache.cached_image(
        controller.frame_position, "SEM", controller.current_condition("SEM")
    )
    return None if prepared is None else prepared.display.label


def test_curate_shows_what_the_label_tab_saved(qt_app, window):
    """The reported fault: Curate went on drawing the label a save replaced,
    until the reviewer moved to another lamella and back."""
    key = window.label_controller.current_key
    assert window.controller.frame.record("SEM").labels == [key]
    settle(qt_app, lambda: curate_label(window) is not None, what="Curate's frame")
    paint_on(window, value=1)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))
    window.tabs.setCurrentWidget(window.curate)

    settle(
        qt_app,
        lambda: np.array_equal(curate_label(window), tifffile.imread(str(key))),
        what="the saved label on Curate",
    )


def test_curate_shows_a_label_rewritten_while_it_was_behind(qt_app, window):
    """A restore, or another program: nothing tells Curate, so arriving asks."""
    key = window.label_controller.current_key
    settle(qt_app, lambda: curate_label(window) is not None, what="Curate's frame")
    label = np.zeros((48, 64), dtype=np.uint8)
    label[20:30] = 1
    tifffile.imwrite(str(key), label, compression="zlib")

    window.tabs.setCurrentWidget(window.curate)

    settle(
        qt_app,
        lambda: np.array_equal(curate_label(window), label),
        what="the rewritten label on Curate",
    )


def test_saving_starts_the_record_it_writes_into(qt_app, window):
    """The Label tab has no editing guard, so the save is the door itself.

    Recording is always on: nobody switches it on and nothing asks. The first
    change made to a project starts its log and records what the whole project
    held first, so the change has something to be a change *from*.
    """
    from am_segmentation_manager.history import history_file_path, read_log

    project = window.controller.projects[0]
    assert not history_file_path(project.path).is_file(), "nothing recorded yet"
    paint_on(window, value=2)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    kinds = [event.kind for event in read_log(project.path).events]
    assert "log-created" in kinds and "baseline" in kinds
    assert "label-saved" in kinds, kinds


def test_saving_clears_what_was_waiting(qt_app, window):
    paint_on(window)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    assert window.label_panel.pending_label.text() == ""
    assert not window.label_panel.save_button.isEnabled()


def test_saving_alone_does_not_move_anything(qt_app, window):
    image = window.label_controller.record.image_path
    paint_on(window)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    assert image.exists(), "save only must not move the image"
    assert "corrected" not in str(image)


def test_marking_corrected_declares_the_whole_group(qt_app, window):
    from am_segmentation_manager.state import read_state

    window.controller.set_editing_enabled(True)
    record = window.label_controller.record
    image = record.image_path
    label = window.label_controller.current_key
    paint_on(window)

    save_with(window, mark_corrected=True)
    settle(qt_app, lambda: saved(window))

    assert image.exists() and label.exists(), "declaring a state moves nothing"
    assert record.state == "corrected"
    stage = image.parent.parent
    assert read_state(stage).states[f"{label.parent.name}/{label.name}"] == "corrected"


def test_labels_saved_but_not_marked_corrected_are_said_to_be(qt_app, window, monkeypatch):
    """A flash in the status line is gone in four seconds; a claim the reviewer
    asked for and did not get has to be said where it cannot be missed."""
    from am_segmentation_manager.gui import label_controller as module
    from am_segmentation_manager.state import StaleState

    def curated_elsewhere(*args, **kwargs):
        raise StaleState("AM_state.yaml changed since it was read")

    warnings = []
    monkeypatch.setattr(module, "transition_records", curated_elsewhere)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda parent, title, text, *args: warnings.append(text),
    )
    window.controller.set_editing_enabled(True)
    paint_on(window)

    save_with(window, mark_corrected=True)
    settle(qt_app, lambda: saved(window))
    settle(qt_app, lambda: bool(warnings), what="the warning")

    assert "saved, but not marked corrected" in warnings[0]
    assert "changed since it was read" in warnings[0]


def test_saving_leaves_the_reviewer_on_the_lamella_they_saved(qt_app, two_lamellae):
    two_lamellae._step_label_lamella(1)
    settle(qt_app, lambda: drawn(two_lamellae))
    lamella = two_lamellae.label_controller.lamella.path
    paint_on(two_lamellae)

    save_with(two_lamellae, mark_corrected=False)
    settle(qt_app, lambda: saved(two_lamellae))

    assert two_lamellae.label_controller.lamella.path == lamella, (
        "a save re-reads the collection; it is not a request to start over"
    )
    assert two_lamellae.controller.lamella.path == lamella


def test_a_save_leaves_the_two_views_on_one_lamella(qt_app, window):
    paint_on(window)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    assert window.label_controller.lamella is window.controller.lamella
    assert window.label_panel.frame_label.text() == "1 / 3"


def test_saving_does_not_read_the_collection_again(qt_app, window, monkeypatch):
    """A save writes what is already in memory, so it must not scan.

    The whole effect of a save is known where it happens: the labels it created
    are named on their records and a declared state carries its own records
    with it. Scanning to find that out again is the difference between a save
    that is instant and one that costs a walk of the whole collection.
    """
    scans = []
    monkeypatch.setattr(
        window.controller, "rescan", lambda: scans.append(None) or []
    )
    paint_on(window)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    assert not scans, "a save must not re-read the collection"


def test_a_label_a_save_created_is_known_without_a_rescan(qt_app, unlabelled):
    """An image annotated from nothing carries its new label straight away.

    Nothing re-reads the collection after a save, so the record has to be told
    where the label it did not have now is -- or the next state change would be
    keyed on a label the window does not know exists.
    """
    window = unlabelled
    record = window.label_controller.record
    assert not record.labels, "the fixture must start with no label"
    key = window.label_controller.current_key
    paint_on(window)

    save_with(window, mark_corrected=False)
    settle(qt_app, lambda: saved(window))

    assert key.exists(), "the label was not written"
    assert record.labels == [key], "the record does not know its own new label"


def test_marking_corrected_is_visible_without_a_rescan(qt_app, window, monkeypatch):
    """The lists recount from the records the save moved, not from a new scan."""
    window.controller.set_editing_enabled(True)
    scans = []
    monkeypatch.setattr(
        window.controller, "rescan", lambda: scans.append(None) or []
    )
    paint_on(window)

    save_with(window, mark_corrected=True)
    settle(qt_app, lambda: saved(window))

    assert not scans
    assert window.label_controller.record.state == "corrected"
    assert window.label_panel.frame_label.text() == "1 / 3", (
        "a corrected frame is still one to proofread"
    )


def test_saving_nothing_writes_nothing(window, monkeypatch):
    asked = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.exec_", lambda box: asked.append(box) or 0
    )

    window.save_labels()

    assert not asked, "there was nothing to ask about"


def save_dialogs(window, monkeypatch):
    """Open the save dialog without waiting for anyone to answer it.

    Returns a list the opened dialogs are appended to. Modal, so exec_ has to
    be intercepted: offscreen there is nobody to press a button, and the run
    would hang rather than fail.
    """
    opened = []

    def captured(dialog):
        opened.append(dialog)
        return 0  # QDialog.Rejected

    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.label_tab.SaveLabelsDialog.exec_",
        captured,
    )
    return opened


def button_texts(dialog):
    """Return the dialog's buttons, left to right as laid out."""
    return [
        button.text()
        for button in sorted(dialog.findChildren(QPushButton), key=lambda b: b.x())
    ]


def test_the_save_dialog_offers_marking_corrected_only_when_editing_is_on(
    window, monkeypatch
):
    opened = save_dialogs(window, monkeypatch)
    paint_on(window)

    window.controller.set_editing_enabled(False)
    window._ask_how_to_save(window.label_controller.save_plan())
    window.controller.set_editing_enabled(True)
    window._ask_how_to_save(window.label_controller.save_plan())

    locked, unlocked = (button_texts(dialog) for dialog in opened)
    assert not any("mark corrected" in text for text in locked)
    assert any("mark corrected" in text for text in unlocked)


def test_the_save_dialog_ends_on_the_button_to_press(window, monkeypatch):
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(True)
    paint_on(window)

    window._ask_how_to_save(window.label_controller.save_plan())
    opened[0].show()

    assert button_texts(opened[0]) == [
        "Cancel", "Save only", "Save and mark corrected",
    ], "the row ends with the button the reviewer almost always wants"


def test_the_save_dialog_recommends_marking_corrected(window, monkeypatch):
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(True)
    paint_on(window)

    window._ask_how_to_save(window.label_controller.save_plan())

    default = [b for b in opened[0].findChildren(QPushButton) if b.isDefault()]
    assert [b.text() for b in default] == ["Save and mark corrected"]


def test_a_locked_save_does_not_describe_what_it_cannot_do(window, monkeypatch):
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(False)
    paint_on(window)

    window._ask_how_to_save(window.label_controller.save_plan())

    said = " ".join(label.text() for label in opened[0].findChildren(QLabel))
    assert "would move" not in said, "that describes an option not on offer"
    assert "Enable editing" in said, "but it says how to reach it"


def test_a_label_changed_on_disk_stops_the_save(qt_app, window, monkeypatch):
    key = window.label_controller.current_key
    paint_on(window)
    tifffile.imwrite(str(key), np.full((48, 64), 3, dtype=np.uint8))

    reported = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda *args, **kwargs: reported.append(args[2]),
    )
    save_with(window, mark_corrected=False)

    assert reported and "changed on disk" in reported[0]
    assert (tifffile.imread(str(key)) == 3).all(), "the file was overwritten"
    assert window.label_controller.pending_count() == 1, "the edit is still held"


def test_a_frame_read_before_its_label_changed_is_read_again(qt_app, window):
    """The frame reported ready holds pixels older than the file: the window
    asks for it again rather than drawing them, or sitting on *loading*."""

    controller = window.label_controller
    ahead = controller._positions[1]
    settle(qt_app, lambda: controller.cache.holds(ahead, controller.modality))
    controller.step(1)
    tifffile.imwrite(str(controller.current_key), np.full((48, 64), 1, dtype=np.uint8))

    window._on_label_loaded(window.label_loader.token, ahead)

    canvas = window.label_panel.canvas
    settle(
        qt_app,
        lambda: canvas._label_array is not None and (canvas._label_array == 1).all(),
        what="the rewritten label on the canvas",
    )


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
    help_text = window._keys.help_html(TAB_LABEL)

    assert "Brush tool" in help_text
    assert "Fill tool" in help_text


def test_every_class_button_is_named_for_the_ink_ring(window):
    """The stylesheet rings a chosen class in ink only on buttons carrying the name."""
    from am_segmentation_manager.gui.widgets.theme import CLASS_BUTTON

    buttons = window.label_panel.paint_buttons.values()
    assert buttons
    assert all(button.objectName() == CLASS_BUTTON for button in buttons)


def test_a_families_change_asks_about_label_edits_waiting_in_its_project(
    window, monkeypatch, tmp_path: Path
):
    """A move can change which label a condition reads, so they are settled first."""
    paint_on(window)
    asked: list[str] = []

    def warning(parent, title, text, *args):
        asked.append(title)
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "warning", warning)
    project = window.controller.project
    elsewhere = type(project)(name="other", path=tmp_path / "elsewhere")

    assert window._may_change_families(elsewhere) is True
    assert asked == []
    assert window._may_change_families(project) is False
    assert asked == ["Unsaved label edits"]
    assert window.label_controller.pending_count() == 1, "cancelled, so kept"


# ---------------------------------------------------------------------------
# The proofreading checklist
# ---------------------------------------------------------------------------

def press(window, key):
    """Do what the Label tab's *key* does."""
    binding = next(b for b in window._keys.bindings if b.tab == TAB_LABEL and b.key == key)
    binding.action()


def test_the_first_c_opens_the_checklist_and_the_next_ticks_a_step(window):
    panel = window.label_panel
    assert not panel.checklist_check.isChecked(), "an aid somebody turns to, not a default"

    press(window, "C")
    assert panel.checklist_check.isChecked()
    assert panel.checklist.current == 0, "opening it ticks nothing"

    press(window, "C")
    assert panel.checklist.is_ticked(0)
    assert panel.checklist.current == 1


def test_shift_c_unticks_the_step_before(window):
    panel = window.label_panel
    panel.checklist_check.setChecked(True)
    press(window, "C")
    press(window, "C")

    press(window, "Shift+C")

    assert [panel.checklist.is_ticked(step) for step in (0, 1)] == [True, False]
    assert panel.checklist.current == 1


def test_the_current_step_shows_what_to_do_in_it(window):
    panel = window.label_panel
    panel.checklist_check.setChecked(True)
    press(window, "C")

    said = panel.checklist.instructions.text()

    assert "Step 2" in said
    assert panel.checklist.steps[1].instructions[0] in said


def test_any_step_can_be_read_by_selecting_it(window):
    checklist = window.label_panel.checklist

    checklist.list.setCurrentRow(5)

    assert checklist.steps[5].instructions[0] in checklist.instructions.text()


def test_ticking_a_row_ticks_its_step(window):
    panel = window.label_panel
    panel.checklist_check.setChecked(True)

    panel.checklist.list.item(3).setCheckState(Qt.CheckState.Checked)

    assert window.label_controller.checklist.ticked(window.label_controller.current_key) == {3}


def test_the_save_step_cannot_be_ticked_by_hand(window):
    checklist = window.label_panel.checklist
    last = checklist.list.item(len(checklist.steps) - 1)

    assert not last.flags() & Qt.ItemFlag.ItemIsUserCheckable, (
        "a tick there would claim a save nothing made"
    )


def test_the_checklist_follows_the_frame_on_screen(qt_app, window):
    panel = window.label_panel
    panel.checklist_check.setChecked(True)
    press(window, "C")

    panel.step_frame(1)
    settle(qt_app, lambda: window.label_controller.checklist_key is not None and drawn(window))
    assert not panel.checklist.is_ticked(0), "each label has its own progress"

    panel.step_frame(-1)
    settle(qt_app, lambda: window.label_controller.checklist_key is not None and drawn(window))
    assert panel.checklist.is_ticked(0)


def test_c_at_the_save_step_asks_how_to_save(window, monkeypatch):
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(True)
    panel = window.label_panel
    panel.checklist_check.setChecked(True)
    for _ in range(window.label_controller.checklist.save_step):
        press(window, "C")
    assert not opened

    press(window, "C")

    assert len(opened) == 1
    assert button_texts(opened[0]) == ["Cancel", "Mark corrected"], (
        "nothing was painted, so marking is the whole of the save"
    )


def test_a_checked_label_cannot_be_marked_with_editing_off(window, monkeypatch):
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(False)
    window.label_panel.checklist_check.setChecked(True)
    for _ in range(window.label_controller.checklist.save_step):
        press(window, "C")

    window.save_labels()

    assert not opened, "with nothing to write and no marking allowed, there is nothing to ask"


def test_the_save_dialog_names_labels_with_steps_not_ticked(window, monkeypatch):
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(True)
    window.label_panel.checklist_check.setChecked(True)
    press(window, "C")
    paint_on(window)

    window._ask_how_to_save(window.label_controller.save_plan())

    said = " ".join(label.text() for label in opened[0].findChildren(QLabel))
    assert "checklist steps not yet ticked" in said
    assert button_texts(opened[0])[-1] == "Save and mark corrected", "a reminder, not a block"


def test_a_step_s_help_explains_it(window):
    checklist = window.label_panel.checklist
    checklist.list.setCurrentRow(2)

    checklist.help_button.click()

    assert checklist.help.isVisible()
    assert checklist.steps[2].title in checklist.help.title.text()
    assert checklist.steps[2].explanation == checklist.help.explanation.text()


def test_f1_lists_the_checklist_keys(window):
    help_text = window._keys.help_html(TAB_LABEL)

    assert "Proofreading checklist" in help_text
    assert "Tick the current step of the checklist" in help_text


def test_a_step_s_help_shows_its_example_before_and_after(window):
    checklist = window.label_panel.checklist
    step = next(index for index, step in enumerate(checklist.steps) if step.example)
    help_window = checklist.help

    checklist.show_help(step)
    before = help_window.example.pixmap().toImage()
    help_window.example_view.buttons["after"].click()
    after = help_window.example.pixmap().toImage()

    assert help_window.example.isVisibleTo(help_window)
    assert not before.isNull() and before != after, "the switch shows the other label"


def test_a_step_without_an_example_shows_its_explanation_alone(window):
    checklist = window.label_panel.checklist
    step = next(index for index, step in enumerate(checklist.steps) if not step.example)

    checklist.show_help(step)

    assert not checklist.help.example.isVisibleTo(checklist.help)
    assert not checklist.help.example_view.isVisibleTo(checklist.help)


def test_a_narrow_example_is_drawn_larger_by_a_whole_factor():
    from am_segmentation_manager.gui.widgets.checklist_panel import EXAMPLE_WIDTH, magnification

    assert magnification(EXAMPLE_WIDTH) == 1
    assert magnification(EXAMPLE_WIDTH // 2) == 2
    assert magnification(EXAMPLE_WIDTH // 2 + 1) == 1, "a fraction would blur the label's pixels"
    assert magnification(2 * EXAMPLE_WIDTH) == 1, "a wide crop is never shrunk"


def test_every_example_is_drawn_about_as_wide_as_the_help_expects(window):
    from am_segmentation_manager.gui.widgets.checklist_panel import EXAMPLE_WIDTH

    checklist = window.label_panel.checklist
    for index, step in enumerate(checklist.steps):
        if not step.example:
            continue
        checklist.show_help(index)
        width = checklist.help.example.pixmap().width()
        assert EXAMPLE_WIDTH // 2 < width <= 2 * EXAMPLE_WIDTH, (step.example, width)


def test_marking_an_unannotated_checked_label_says_so(unlabelled, monkeypatch):
    window = unlabelled
    opened = save_dialogs(window, monkeypatch)
    window.controller.set_editing_enabled(True)
    window.label_panel.checklist_check.setChecked(True)
    for _ in range(window.label_controller.checklist.save_step):
        press(window, "C")

    window.save_labels()

    said = " ".join(label.text() for label in opened[0].findChildren(QLabel))
    assert "hold no annotation" in said
