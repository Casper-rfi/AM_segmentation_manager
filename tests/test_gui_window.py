"""Smoke tests for the curation window.

These drive the real widgets against a small synthetic project on disk, using
Qt's offscreen platform so they need no display. They are skipped when Qt or the
optional image dependencies are not installed, which keeps a minimal
installation -- and CI without Qt -- able to run the rest of the suite.

pytest-qt is deliberately not used: a session-wide QApplication and a small
event-loop helper are all these tests need, and avoiding the dependency keeps
the test requirements the same as the package's.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import pytest

from conftest import destroy, settle

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
    from qtpy.QtCore import QSettings, Qt
    from qtpy.QtWidgets import QApplication, QMessageBox, QToolButton

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.attribution import Attribution
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.metadata_controller import MetadataController
    from am_segmentation_manager.gui.widgets.branding import PRODUCT_NAME
    from am_segmentation_manager.gui.widgets.controls import ChipEditor
    from am_segmentation_manager.gui.widgets.image_view import MIN_FIT_FRACTION
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow
    from am_segmentation_manager.gui.widgets.metadata_panel import (
        MetadataPanel,
        editor_text,
        set_editor_text,
    )
    from am_segmentation_manager.gui.widgets.theme import BADGE_STYLES
    from am_segmentation_manager.metadata import (
        DESCRIPTIVE_FIELDS,
        FIELDS,
        build_experiment_metadata,
    )
    from am_segmentation_manager.models import Project
    from am_segmentation_manager.progress import Cancellation

CONFIG = None


@pytest.fixture(scope="session")
def qt_app():
    """One QApplication for the whole session; Qt dislikes several."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A two-lamella project with real (tiny) image, label and plot files."""
    import cv2

    root = tmp_path / "scan"
    project_root = root / "AutoLamella-2026-07-28-13-57"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("name: test-project")

    for lamella_index, (lamella, frames) in enumerate(
        (("01-alpha", 4), ("02-beta", 2)), start=1
    ):
        stage = project_root / lamella / "adaptive_polish_2026-07-28-11-15-36AM"
        for number in range(frames):
            image = np.full((64, 96), 100, dtype=np.uint8)
            tifffile.imwrite(str(_ensure(stage / "sem" / f"{lamella}_AP_img_{number:03d}_SEM.tif")), image)
            tifffile.imwrite(str(_ensure(stage / "fib" / f"{lamella}_AP_img_{number:03d}_FIB.tif")), image)

            label = np.zeros((64, 96), dtype=np.uint8)
            label[:20] = 2
            tifffile.imwrite(str(_ensure(stage / "labels" / f"{lamella}_AP_img_{number:03d}_SEM.tif")), label)

            # Larger than any test pane, as the real 3000x4500 figures are:
            # a plot smaller than the pane would be scaled *up* for display,
            # which exercises the opposite path from the one that matters.
            plot = np.full((1400, 2100, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(_ensure(stage / "plots" / f"{lamella}_AP_img_{number:03d}_plot.png")), plot)
    return root


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    """A throwaway preferences file.

    The window remembers its scan root and its layout. Left on the default,
    every test run would read and overwrite the preferences of the real window
    on this machine.
    """
    return QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)


@pytest.fixture
def window(qt_app, project: Path, settings: QSettings):
    """A window with the synthetic project scanned and its first frame ready."""
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )
    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(controller, settings=settings)
    window.scan(str(project))
    settle(qt_app, lambda: window.image_panel.pair.left._item.pixmap().isNull() is False)
    yield window
    destroy(window)


# ---------------------------------------------------------------------------
# Loading and display
# ---------------------------------------------------------------------------

def test_scanning_populates_the_window(window):
    assert window.controller.project.name == "test-project"
    assert window.frame_label.text() == "1 / 4"
    assert len(list(window.project_panel.project_list.project_items())) == 1
    assert window.project_panel.lamella_list.count() == 2
    assert "(1/2)" in window.image_panel.status.full_text()


def test_both_panes_are_drawn(window):
    """The plain image and the overlaid image are shown side by side."""
    assert not window.image_panel.pair.left._item.pixmap().isNull()
    assert not window.image_panel.pair.right._item.pixmap().isNull()


def test_frames_are_prepared_ahead_of_the_user(qt_app, window):
    settle(qt_app, lambda: window.controller.cache.cached_image(1, "SEM") is not None)

    assert window.controller.cache.cached_image(1, "SEM") is not None


def test_stepping_frames_updates_the_view(qt_app, window):
    window._step_frame(1)
    settle(qt_app, lambda: window.frame_label.text() == "2 / 4")

    assert window.controller.frame_position == 1
    assert window.timeline._current == 1


def test_frame_navigation_stops_at_the_end(window):
    window._step_frame(99)

    assert window.frame_label.text() == "4 / 4"


def test_changing_lamella_resets_to_its_first_frame(qt_app, window):
    window._step_frame(2)
    window._step_lamella(1)
    settle(qt_app, lambda: window.controller.lamella.name == "02-beta")

    assert window.frame_label.text() == "1 / 2"
    assert "(2/2)" in window.image_panel.status.full_text()


def test_missing_fib_label_is_reported_not_hidden(qt_app, window):
    """This project has no labels_fib, which the badge must make plain."""
    window._set_mode("fib")
    settle(qt_app, lambda: "no label" in window.image_panel.badges.text())

    assert "no label" in window.image_panel.badges.text()


def test_plot_mode_draws_the_plot(qt_app, window):
    window._set_mode("plot")
    settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())

    assert window.image_panel.stack.currentIndex() == 1


def test_overlay_can_be_switched_off(window):
    window.image_panel.overlay_check.setChecked(False)

    assert window.controller.show_overlay is False
    assert not window.image_panel.pair.right._item.pixmap().isNull()


def test_the_overlay_pane_can_drop_the_image_under_the_label(window):
    window.image_panel.image_check.setChecked(False)

    assert window.controller.show_image is False
    left = window.image_panel.pair.left._item.pixmap()
    right = window.image_panel.pair.right._item.pixmap()
    assert not left.isNull(), "the left pane is the acquisition as recorded"
    assert not right.isNull()
    assert right.toImage() != left.toImage(), (
        "the overlay pane draws the label against black instead"
    )


# ---------------------------------------------------------------------------
# Resizable divisions
# ---------------------------------------------------------------------------

def test_every_division_that_moves_is_marked_as_one(window):
    """Which divisions can be dragged is found by eye, so one that looks like
    an ordinary line is one nobody discovers. Asserted over the same mapping
    the window saves its layout through, so a division added without a grip is
    a division added without a saved size as well."""
    from am_segmentation_manager.gui.widgets.controls import GripSplitter

    divisions = window._splitters()

    assert divisions, "the window has resizable divisions"
    assert all(
        isinstance(splitter, GripSplitter) for splitter in divisions.values()
    ), sorted(
        name for name, splitter in divisions.items()
        if not isinstance(splitter, GripSplitter)
    )


# ---------------------------------------------------------------------------
# Fitting the screen
# ---------------------------------------------------------------------------

#: The smallest display the window is expected to open on, in logical pixels.
#: A 13" MacBook Air; the 14" MacBook Pro reports 1512x982 and every external
#: monitor here is larger. Not a guess at the smallest screen in the world --
#: it is the floor this package holds itself to, and a control row that cannot
#: wrap breaks it long before anything exotic does.
SMALLEST_SUPPORTED_SCREEN = (1440, 900)


def test_no_tab_needs_more_room_than_the_smallest_screen_supported(window):
    """A window cannot be made narrower than its widest unwrappable row.

    Qt clamps a resize to the minimum, so a tab needing more width than the
    screen has does not scroll or wrap: the far edge sits off the display with
    no way to drag it back. That is not a layout blemish but an unreachable
    control, and it is invisible to anyone developing on a larger monitor --
    the Label tab's view controls needed 1560px, against the 1512 a 14"
    MacBook Pro has, and the right of every tab was cut off there.
    """
    wide, tall = SMALLEST_SUPPORTED_SCREEN
    too_big = {
        window.tabs.tabText(index): (
            window.tabs.widget(index).minimumSizeHint().width(),
            window.tabs.widget(index).minimumSizeHint().height(),
        )
        for index in range(window.tabs.count())
        if window.tabs.widget(index).minimumSizeHint().width() > wide
        or window.tabs.widget(index).minimumSizeHint().height() > tall
    }

    assert not too_big, f"needs more than {wide}x{tall}: {too_big}"
    assert window.minimumSizeHint().width() <= wide
    assert window.minimumSizeHint().height() <= tall


def test_the_view_controls_wrap_rather_than_widening_the_window(window):
    """The Label tab's controls are the row that broke the rule above.

    Asserted on the row rather than only through the window, so the reason a
    tab fits stays visible: it fits because this row wraps, and a control
    added to a plain box layout here would put it back over the edge.
    """
    from am_segmentation_manager.gui.widgets.controls import FlowArea, FlowLayout

    controls = window.label_panel.controls

    assert isinstance(controls, FlowArea)
    assert isinstance(controls.layout(), FlowLayout)
    assert controls.layout().minimumSize().width() == 0


def test_a_long_caption_does_not_widen_the_curate_tab(window):
    """The line under the frame names the project path and the image file,
    which on a grid overview runs past a screen's width. A line that cannot
    shorten is a floor under the tab, and the splitter takes that width from
    the lamella list; so the caption elides and the badges wrap."""
    panel = window.image_panel
    before = panel.minimumSizeHint().width()
    caption = (
        "001: " + "/".join(["AutoLamella-2026-04-08-10-01_virussample"] * 8)
        + " overview-image-2026-04-08_10-04-53-autogamma.tif"
    )

    panel.set_status(caption)
    panel._set_badges([("positions are approximate", "warning")] * 12)

    assert panel.minimumSizeHint().width() <= before
    assert panel.status.full_text() == caption
    assert panel.status.toolTip() == caption


def test_a_window_larger_than_the_screen_is_pulled_onto_it(window):
    """Tested against a screen this machine does not have.

    The fault only appears on a display smaller than the one it is developed
    on, so asserting it through the real screen would assert nothing here.
    """
    from qtpy.QtCore import QRect

    window.resize(1600, 950)
    window._fit_within(QRect(0, 0, 1512, 982))

    assert window.width() <= 1512
    assert window.height() <= 982


def test_a_window_remembered_off_the_edge_is_moved_back_on(window):
    """A geometry saved on a second monitor is restored without one attached.

    Qt puts back what it was given; nothing in ``restoreGeometry`` asks
    whether the screen it was recorded on is the screen it is being put back
    on, and a window opening off the edge cannot be dragged back.
    """
    from qtpy.QtCore import QRect

    room = QRect(0, 0, 1512, 982)
    window.resize(1100, 800)
    window.move(2400, 200)

    window._fit_within(room)

    assert room.contains(window.frameGeometry())


# ---------------------------------------------------------------------------
# The editing guard
# ---------------------------------------------------------------------------

def test_state_controls_start_locked(window):
    assert window.state_panel.state_box.isEnabled() is False
    assert window.controller.editing_enabled is False


def test_choosing_what_to_change_is_not_guarded(window):
    """Ticking rows and picking a modality write nothing, so they stay usable
    with the guard on -- and the reach line recounts as they change."""
    assert window.controller.editing_enabled is False

    assert window.state_panel.target_buttons["FIB"].isEnabled() is True
    assert window.project_panel.lamella_list.isEnabled() is True


def test_enabling_editing_unlocks_the_controls(window):
    window.state_panel.enable_check.setChecked(True)

    assert window.state_panel.state_box.isEnabled() is True
    assert window.controller.editing_enabled is True


def test_a_state_change_is_declared_and_shown_in_the_readout(qt_app, window):
    from am_segmentation_manager.state import read_state

    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    record = window.controller.frame.record("SEM")
    original = record.image_path
    assert original.exists()

    window.apply_state("deselected")
    settle(qt_app, lambda: window.controller.state_of("SEM") == "deselected")

    assert original.exists(), "declaring a state moves nothing"
    assert record.image_path == original
    assert "SEM: deselected" in window.state_panel.readout.text()
    assert "FIB: selected" in window.state_panel.readout.text()

    stage = original.parent.parent
    assert list(read_state(stage).states.values()) == ["deselected"]


def test_applying_to_both_declares_both_modalities(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["both"].setChecked(True)

    window.apply_state("deselected")
    settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    for record in window.controller.frame.records:
        assert record.state != "selected"
        assert record.image_path.exists()


def test_the_change_is_reported_where_it_was_made(window):
    """A state change is the one thing on this tab that writes without a dialog
    to report it, so the buttons say what they just did."""
    window.state_panel.enable_check.setChecked(True)

    window.apply_state("deselected")

    assert "deselected" in window.state_panel.last_change.text()


def test_timeline_reflects_the_lamella(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["both"].setChecked(True)
    window.apply_state("deselected")
    settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    assert window.timeline._rows["SEM"][0] == "deselected"
    assert window.timeline._rows["FIB"][0] == "deselected"
    assert window.timeline._rows["SEM"][1] == "selected"


def test_mixed_state_shows_no_option_selected(qt_app, window):
    """A frame whose modalities disagree must not claim to be in one state."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("deselected")
    settle(qt_app, lambda: window.controller.combined_state() == "mixed")

    window.state_panel.target_buttons["both"].setChecked(True)
    window._refresh_state_panel()

    assert not any(
        button.isChecked() for button in window.state_panel.state_buttons.values()
    )
    assert "modalities differ" in window.state_panel.readout.text()


# ---------------------------------------------------------------------------
# Linked zoom (regression: the panes stopped following each other once an
# image fitted its pane, because zoom rode on scrollbar signals)
# ---------------------------------------------------------------------------

def test_zoom_is_mirrored_to_the_other_pane(window):
    pair = window.image_panel.pair
    pair.left.zoom_by(2.0)

    assert pair.right.transform().m11() == pytest.approx(pair.left.transform().m11())


def test_zoom_stays_linked_when_the_image_fits_its_pane(window):
    """The case that used to desynchronise: no scrollbars, so no scroll events."""
    pair = window.image_panel.pair
    pair.left.zoom_by(4.0)
    assert pair.right.transform().m11() == pytest.approx(pair.left.transform().m11())

    # Zoom back out past the point where the image fits.
    for _ in range(8):
        pair.left.zoom_by(0.5)

    assert pair.right.transform().m11() == pytest.approx(pair.left.transform().m11())


def test_zoom_out_stops_at_fit(window):
    """Reviewing is done at fit, so there is nothing below it to see: shrinking
    the frame further only strands it in an empty pane."""
    view = window.image_panel.pair.left
    view.fit()
    fitted = view.transform().m11()

    for _ in range(10):
        view.zoom_by(0.5)

    assert view.transform().m11() == pytest.approx(fitted, rel=0.001), (
        "zoomed all the way out and fitted have to be the same picture"
    )
    assert view.min_fit_fraction == MIN_FIT_FRACTION


def test_the_pane_can_be_scrolled_past_the_edges_of_the_frame(window):
    view = window.image_panel.pair.left
    image = view._item.boundingRect()
    scene = view.sceneRect()

    assert scene.left() < image.left() and scene.right() > image.right()
    assert scene.top() < image.top() and scene.bottom() > image.bottom(), (
        "an edge that cannot leave the border of the pane is one you paint blind"
    )


def test_the_panes_have_no_scrollbars(window):
    view = window.image_panel.pair.left

    assert view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_timeline_shows_each_modality_separately(qt_app, window):
    """A frame with one modality changed must not read as an unlabelled 'mixed'."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("deselected")
    settle(qt_app, lambda: window.timeline._rows["SEM"][0] == "deselected")

    assert window.timeline._rows["SEM"][0] == "deselected"
    assert window.timeline._rows["FIB"][0] == "selected"


def test_clicking_the_timeline_seeks(window):
    window.timeline.resize(400, 24)
    window.timeline.set_states(window.controller.modality_states(), 0)

    window.timeline.frame_selected.emit(2)

    assert window.controller.frame_position == 2


def test_timeline_handles_an_empty_lamella(window):
    window.timeline.set_states({"SEM": [], "FIB": []}, 0)

    assert window.timeline._count == 0


# ---------------------------------------------------------------------------
# State panel
# ---------------------------------------------------------------------------

def test_state_options_are_in_workflow_order(window):
    order = list(window.state_panel.state_buttons)

    assert order == ["deselected", "selected", "corrected"]


def test_state_options_carry_a_colour_swatch(window):
    for button in window.state_panel.state_buttons.values():
        assert not button.icon().isNull()


def test_state_tooltips_name_their_shortcut(window):
    assert "D" in window.state_panel.state_buttons["deselected"].toolTip()
    assert "C" in window.state_panel.state_buttons["corrected"].toolTip()


# ---------------------------------------------------------------------------
# Long lists
# ---------------------------------------------------------------------------

def test_filtering_hides_non_matching_lamellae(window):
    window.project_panel.lamella_filter.setText("beta")

    hidden = [
        window.project_panel.lamella_list.item(row).isHidden()
        for row in range(window.project_panel.lamella_list.count())
    ]
    assert hidden == [True, False]


def test_filtering_keeps_row_numbers_aligned_with_lamellae(window):
    """Rows must still map to real indices, or selection would target the wrong one."""
    window.project_panel.lamella_filter.setText("beta")
    window._select_lamella(1)

    assert window.controller.lamella.name == "02-beta"


def test_clearing_the_filter_restores_everything(window):
    window.project_panel.lamella_filter.setText("beta")
    window.project_panel.lamella_filter.setText("")

    hidden = [
        window.project_panel.lamella_list.item(row).isHidden()
        for row in range(window.project_panel.lamella_list.count())
    ]
    assert hidden == [False, False]


def test_list_titles_show_counts(window):
    assert "(1)" in window.project_panel.project_box.title()
    assert "(2)" in window.project_panel.lamella_box.title()


# ---------------------------------------------------------------------------
# Status chips
#
# The colour is the information, so Qt must not be left to re-tint it.
# ---------------------------------------------------------------------------

def test_a_chip_is_the_same_pixmap_in_every_icon_mode(qt_app):
    """Left to itself, Qt generates a Selected variant blended with the
    highlight colour, which washes the chip out on the row the user is on."""
    from qtpy.QtGui import QIcon, QPixmap

    from am_segmentation_manager.gui.widgets.common import chip_icon

    pixmap = QPixmap(14, 10)
    pixmap.fill(Qt.red)
    icon = chip_icon(pixmap)

    normal = icon.pixmap(14, 10, QIcon.Normal).toImage()
    for mode in (QIcon.Selected, QIcon.Active, QIcon.Disabled):
        assert icon.pixmap(14, 10, mode).toImage() == normal


def _status(**cells):
    """Build a status from the cells named, the rest holding no images.

    Keys are ``series_MODALITY``: ``adaptive_SEM="done"``.
    """
    from am_segmentation_manager.gui.controller import CurationStatus
    from am_segmentation_manager.layout import MODALITIES, SERIES

    grid = {(s, m): "empty" for s in SERIES for m in MODALITIES}
    grid.update({tuple(key.rsplit("_", 1)): value for key, value in cells.items()})
    return CurationStatus(grid)


def _chip_image(**cells):
    """Render one status chip on its own, at its natural size."""
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import SWATCH_HEIGHT, SWATCH_WIDTH

    icon = _status_icon(_status(**cells))
    return icon.pixmap(SWATCH_WIDTH, SWATCH_HEIGHT).toImage()


def test_a_chip_has_a_column_per_series_it_is_given(qt_app):
    """A series switched off takes no room, rather than an empty column."""
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import SWATCH_WIDTH, swatch_width
    from am_segmentation_manager.layout import SERIES

    icon = _status_icon(_status(), ("adaptive", "reference"))

    assert swatch_width(len(SERIES)) == SWATCH_WIDTH
    assert icon.availableSizes()[0].width() == swatch_width(2)


def _cell_centre(column: int, row: int):
    """Return a point inside the cell at *column* and *row* of a chip."""
    from am_segmentation_manager.gui.widgets.theme import SWATCH_CELL, SWATCH_GAP

    step, middle = SWATCH_CELL + SWATCH_GAP, SWATCH_CELL // 2
    return column * step + middle, row * step + middle


def _luminance(colour) -> float:
    """Return the relative luminance of *colour*, as the eye weights it."""
    def channel(value: int) -> float:
        value /= 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(colour.red())
        + 0.7152 * channel(colour.green())
        + 0.0722 * channel(colour.blue())
    )


def test_the_two_modalities_are_stacked_not_side_by_side(qt_app):
    """SEM above FIB, the same arrangement as the timeline's two rows, so a
    beam is found in the same place in both."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS

    image = _chip_image(adaptive_SEM="done", adaptive_FIB="not started")

    assert image.pixelColor(*_cell_centre(0, 0)) == CELL_COLOURS["done"]
    assert image.pixelColor(*_cell_centre(0, 1)) == CELL_COLOURS["not started"]


def test_a_part_finished_modality_gets_a_colour_rather_than_a_split(qt_app):
    """A split showed two pixels of each colour once the outline had taken the
    rest, and the two were the same brightness. The cell is one flat colour."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS
    from am_segmentation_manager.gui.widgets.theme import SWATCH_CELL

    image = _chip_image(adaptive_SEM="in progress")
    row = SWATCH_CELL // 2
    across = {image.pixelColor(x, row).name() for x in range(SWATCH_CELL)}

    assert across == {CELL_COLOURS["in progress"].name()}


def test_the_progress_steps_are_told_apart_by_brightness(qt_app):
    """Hue is the channel the eye cannot use in a cell this small, so the three
    steps have to separate in luminance whatever their colour."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS

    ramp = [
        _luminance(CELL_COLOURS[state])
        for state in ("not started", "in progress", "done")
    ]

    assert ramp == sorted(ramp, reverse=True)
    assert all(
        (before + 0.05) / (after + 0.05) >= 1.5
        for before, after in zip(ramp, ramp[1:])
    )


def test_a_cell_with_no_images_is_the_palest_thing_on_the_chip(qt_app):
    """Absent has to read as absent, not as a state: many older datasets have
    no FIB whatsoever, and a quarter of the archive no Adaptive Polishing. It is
    drawn rather than left hollow, so it does not depend on what is behind."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS

    image = _chip_image(adaptive_SEM="done")
    inside = image.pixelColor(*_cell_centre(0, 1))

    assert inside == CELL_COLOURS["empty"]
    assert inside.alpha() == 255
    assert _luminance(inside) > max(
        _luminance(colour)
        for state, colour in CELL_COLOURS.items()
        if state != "empty"
    )


def test_the_chip_is_an_even_number_of_pixels_tall(qt_app):
    """Qt centres a row's check indicator on the row but rounds the decoration
    differently, so an odd-height icon lands half a pixel below the tick beside
    it. The chip was 13 and sat visibly low."""
    from am_segmentation_manager.gui.widgets.theme import SWATCH_HEIGHT

    assert SWATCH_HEIGHT % 2 == 0


def test_a_chip_keeps_its_colour_on_the_row_being_looked_at(qt_app):
    """The regression: a 'done' teal came out blue-green on the current row,
    against a blue-tinted background -- so a finished lamella looked unfinished
    exactly when you selected it to check."""
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QListWidget

    from am_segmentation_manager.gui.widgets.project_panel import (
        CELL_COLOURS,
        _status_icon,
    )
    from am_segmentation_manager.gui.widgets.theme import apply_theme

    stylesheet, palette = qt_app.styleSheet(), qt_app.palette()
    apply_theme(qt_app)
    try:
        done = _status(adaptive_SEM="done", adaptive_FIB="done")
        widget = QListWidget()
        widget.addItems(["current", "other"])
        for row in range(2):
            widget.item(row).setIcon(_status_icon(done))
        widget.setCurrentRow(0)
        widget.resize(300, 60)
        image = widget.grab().toImage()

        teal = CELL_COLOURS["done"].name()
        found = [
            any(
                QColor(image.pixel(x, y)).name() == teal
                for y in range(4 + row * 22, 24 + row * 22)
                for x in range(4, 40)
            )
            for row in range(2)
        ]
        assert found == [True, True]
    finally:
        qt_app.setStyleSheet(stylesheet)
        qt_app.setPalette(palette)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def test_tabs_are_in_workflow_order(window):
    titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]

    assert titles == ["Add data", "Curate", "Label", "Export"]


def test_curate_is_the_tab_that_opens(window):
    """Left-most is the workflow's first step; open on the one used most."""
    assert window.tabs.currentIndex() == _curate_tab(window)
    assert window._current_tab() == "curate"


# ``focusWidget`` rather than ``hasFocus``: these windows are never shown, and
# Qt defers real focus until a window is activated. What is asserted is the
# widget that will hold the keyboard, which is the same either way.

def test_curate_opens_with_the_keyboard_on_the_project_list(window):
    """Not on the scan box. It is the first focusable widget in the tab, and a
    focused QLineEdit claims Left and Right for its own cursor -- so the keys
    that step through frames never reach the window until something is
    clicked."""
    assert window.focusWidget() is window.project_panel.project_list


def test_returning_to_curate_takes_the_keyboard_back(window):
    window.tabs.setCurrentWidget(window.metadata_panel)
    window.tabs.setCurrentIndex(_curate_tab(window))

    assert window.focusWidget() is window.project_panel.project_list


def test_the_scan_box_is_what_would_have_swallowed_the_frame_keys(window):
    """The reason the fix is a focus change rather than a rebinding. Asserted
    on the widgets themselves so a Qt version that changed its mind about which
    keys a line edit claims is reported here."""
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QKeyEvent

    def claims(widget, key) -> bool:
        event = QKeyEvent(QEvent.ShortcutOverride, key, Qt.NoModifier)
        event.ignore()
        QApplication.sendEvent(widget, event)
        return event.isAccepted()

    assert claims(window.project_panel.path_edit, Qt.Key_Left)
    assert not claims(window.project_panel.project_list, Qt.Key_Left)
    assert not claims(window.project_panel.project_list, Qt.Key_Down)


def test_opening_export_reads_the_headers_and_not_the_labels(window, monkeypatch):
    """Half a millisecond an image, against ten times that per label. The
    headers are what the acquisition filters are made of, so they are worth
    paying for on arrival; the labels answer a different question and stay
    behind the button that asks it."""
    submitted: list[str] = []
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: (
            submitted.append(description), on_done(work())
        ),
    )

    window.tabs.setCurrentWidget(window.export_panel)

    assert submitted and "header" in submitted[0]
    assert window.export_controller.metadata_loaded()
    assert not window.export_controller.measurements_loaded(), (
        "opening the tab must not open a single label"
    )


def test_returning_to_export_does_not_read_the_headers_again(window, monkeypatch):
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    window.tabs.setCurrentWidget(window.export_panel)
    assert window.export_controller.metadata_loaded()

    monkeypatch.setattr(
        window, "_run_operation",
        lambda *args, **kwargs: pytest.fail("the headers were already read"),
    )
    window.tabs.setCurrentWidget(window.curate)
    window.tabs.setCurrentWidget(window.export_panel)


def test_the_scan_says_what_it_is_doing_before_it_blocks(window, project: Path):
    """A wait cursor does not say whether a window is working or stuck, and the
    scan is what stops anything else being painted."""
    announced: list[str] = []
    original = window.project_panel.announce
    window.project_panel.announce = lambda text: (
        announced.append(text), original(text)
    )
    try:
        window.scan(str(project))
    finally:
        del window.project_panel.announce

    assert announced and "Scanning" in announced[0]
    assert project.name in announced[0]


def test_the_add_data_tab_opens_before_anything_is_scanned(qt_app, settings):
    """Reachable from a fresh window, which has no projects to describe yet."""
    controller = CurationController(cache=FrameCache(), config=SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    ))
    fresh = CurationWindow(controller, settings=settings)
    try:
        fresh.tabs.setCurrentWidget(fresh.metadata_panel)

        assert fresh.metadata_panel.summary.text() == "No project selected"
    finally:
        destroy(fresh)


# ---------------------------------------------------------------------------
# Remembering the layout
# ---------------------------------------------------------------------------

def test_the_splitter_positions_survive_closing_the_window(qt_app, project, settings):
    """Dragging a divider is a preference, and losing it every session is what
    makes a window feel like it is not listening."""
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    )
    first = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    first.curate.setSizes([400, 800, 400])
    sizes = first.curate.sizes()
    first.close()
    destroy(first)

    second = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert second.curate.sizes() == sizes
    finally:
        destroy(second)


def test_a_window_with_no_saved_layout_uses_its_defaults(qt_app, settings):
    """A first run has nothing to restore, which must not leave a collapsed
    pane behind."""
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    )
    fresh = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert all(size > 0 for size in fresh.curate.sizes())
    finally:
        destroy(fresh)


#: Longest tooltip worth reading in a hover, as the Export tab already holds
#: itself to. Past this it is documentation, and a tooltip nobody finishes
#: explains nothing.
MAX_TOOLTIP = 200


def test_no_tooltip_in_the_window_is_longer_than_anyone_reads(window):
    """The ceiling is stated in gui/widgets/__init__.py. Enforced on one tab
    only, the timeline grew a 488-character legend nobody was reading."""
    from qtpy.QtWidgets import QWidget

    too_long = {
        widget.toolTip()
        for widget in window.findChildren(QWidget)
        if len(widget.toolTip()) > MAX_TOOLTIP
    }

    assert not too_long


# ---------------------------------------------------------------------------
# Shortcut reference
# ---------------------------------------------------------------------------

def test_every_shortcut_is_documented(window):
    assert len(window._keys.bindings) == len(window._keys.shortcuts)
    assert all(binding.description for binding in window._keys.bindings)
    assert any(binding.key == "F1" for binding in window._keys.bindings)


def test_a_key_is_live_only_on_the_tab_it_belongs_to(window):
    """Curate keys firing behind another tab is what stopped the Metadata list
    ever seeing an arrow key."""
    window.tabs.setCurrentWidget(window.metadata_panel)

    live = {b.key for b, s in zip(window._keys.bindings, window._keys.shortcuts) if s.isEnabled()}

    assert live == {"Up", "Down", "F1"}


def test_curate_keys_come_back_with_the_curate_tab(window):
    window.tabs.setCurrentWidget(window.metadata_panel)
    window.tabs.setCurrentIndex(_curate_tab(window))

    live = {b.key for b, s in zip(window._keys.bindings, window._keys.shortcuts) if s.isEnabled()}

    assert {"Left", "Right", "V", "D", "F1"} <= live


def _curate_tab(window) -> int:
    """Return the curate tab's index, whatever order the tabs are in."""
    others = {window.metadata_panel, window.export_panel}
    return next(
        index for index in range(window.tabs.count())
        if window.tabs.widget(index) not in others
    )


def test_f1_puts_the_list_away_again(window):
    window.show_shortcuts()
    assert window._keys.help.isVisible()
    # is_in_front rather than the real thing: an offscreen window is never
    # activated, so nothing here is ever the window in front.
    window._keys.help.is_in_front = lambda: True

    window.show_shortcuts()

    assert not window._keys.help.isVisible(), "the key that opened it closes it"


def test_f1_fetches_back_a_list_left_behind_the_window(window):
    """It closed on visibility alone, and a list buried behind the window is
    visible -- so F1 dismissed, unseen, the one thing that answers "what can I
    press?", and pressing it again just put it back. That is most of what made
    F1 look as though it worked only sometimes."""
    window.show_shortcuts()
    window._keys.help.is_in_front = lambda: False

    window.show_shortcuts()

    assert window._keys.help.isVisible()


def test_f1_is_ignored_while_an_operation_holds_the_window(window, monkeypatch):
    """The key reaches the window from anywhere now, and a list opened over an
    application-modal dialog can be neither read nor closed until it ends."""
    monkeypatch.setattr(
        QApplication, "activeModalWidget", staticmethod(lambda: window)
    )

    window.show_shortcuts()

    assert window._keys.help is None


def test_f1_works_from_the_list_it_opened(window):
    """A window-scoped shortcut is dead while a second top-level window holds
    the keyboard, and the list is one -- so the key that opens it could not
    close it again."""
    from qtpy.QtCore import Qt

    f1 = next(
        shortcut for binding, shortcut in zip(window._keys.bindings, window._keys.shortcuts)
        if binding.key == "F1"
    )

    assert f1.context() == Qt.ApplicationShortcut


def test_the_window_asks_to_be_the_one_in_front(window):
    """Raised and activated, not just raised: a window on top but without the
    keyboard is one macOS sends neither key presses nor pointer motion to, so
    F1, every tooltip and the brush outline are all dead until it is clicked.
    Asserted on the calls, an offscreen window never being activated."""
    asked: list[str] = []
    window.raise_ = lambda: asked.append("raise")
    window.activateWindow = lambda: asked.append("activate")

    window.bring_to_front()

    assert asked == ["raise", "activate"]


def test_losing_the_keyboard_is_recorded(window, caplog):
    """Whether the window was in front is not recoverable after the fact, and
    it is the first thing to establish about a report that the keyboard or the
    pointer stopped working."""
    from qtpy.QtCore import QEvent

    with caplog.at_level(logging.DEBUG, logger="am_segmentation_manager.gui.widgets"):
        window.changeEvent(QEvent(QEvent.ActivationChange))

    assert "deactivated" in caplog.text


def test_the_shortcut_list_does_not_block_the_window(window):
    window.show_shortcuts()

    assert not window._keys.help.isModal(), (
        "a list you must dismiss before trying what it says is read once"
    )


def test_the_shortcut_list_follows_the_tab(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_LABEL

    window.show_shortcuts()
    window.tabs.setCurrentWidget(window.label_panel)

    assert window._keys.help._text.text() == window._keys.help_html(TAB_LABEL)


def test_the_help_lists_only_the_active_tab(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_ADD_DATA, TAB_CURATE

    curate = window._keys.help_html(TAB_CURATE)
    metadata = window._keys.help_html(TAB_ADD_DATA)

    assert "Next frame" in curate and "Next project" not in curate
    assert "Next project" in metadata and "Next frame" not in metadata
    # The editing guard only governs the curate tab's keys.
    assert "Enable editing" in curate and "Enable editing" not in metadata


def test_the_help_describes_the_mouse_gestures_too(window):
    """Picking frames out has no key to press, so listing only keys left the
    whole selection mechanism undiscoverable."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    help_text = window._keys.help_html(TAB_CURATE)

    assert "Ctrl-click" in help_text
    assert "Shift-click" in help_text
    assert "Tick a project or lamella" in help_text


def test_every_gesture_is_described(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_GESTURES

    for gestures in TAB_GESTURES.values():
        assert all(
            group and action and description
            for group, action, description in gestures
        )


def _headings() -> set[str]:
    """The headings F1 is allowed to file a row under, named one by one.

    Deliberately not derived from the module: a heading is added by writing it
    here as well, which is the moment to ask whether the list needs another
    one or the row belongs under a heading that already exists.
    """
    from am_segmentation_manager.gui.widgets import shortcuts as shortcuts_module

    return {
        shortcuts_module.GROUP_VIEW,
        shortcuts_module.GROUP_CHOOSE,
        shortcuts_module.GROUP_STATE,
        shortcuts_module.GROUP_TOOLS,
        shortcuts_module.GROUP_CLASSES,
        shortcuts_module.GROUP_SPECKS,
        shortcuts_module.GROUP_EDITS,
        shortcuts_module.GROUP_ELSEWHERE,
        shortcuts_module.GROUP_HELP,
    }


def test_every_key_and_gesture_is_filed_under_a_known_heading(window):
    """A row with nowhere to go is one the reader has to place themselves."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_GESTURES

    headings = _headings()
    assert {binding.group for binding in window._keys.bindings} <= headings
    for gestures in TAB_GESTURES.values():
        assert {group for group, _action, _description in gestures} <= headings


def test_the_three_ways_to_resize_the_brush_are_listed_together(window):
    """[ and ] and Shift with the wheel all do one thing, so they read as one
    group -- and the mouse rows still run together, this one leading them."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_LABEL

    help_text = window._keys.help_html(TAB_LABEL)

    assert (
        help_text.index("Larger brush")
        < help_text.index("Shift + scroll wheel")
        < help_text.index("Drag with the left button")
        < help_text.index("Click with the fill tool")
    )


def test_a_heading_is_not_drawn_like_the_rows_under_it(window):
    """Every left-hand cell is bold, being the column a reader scans, so bold
    alone left a heading looking like one more key."""
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_TOOLS, TAB_LABEL
    from am_segmentation_manager.gui.widgets.theme import HEADING

    help_text = window._keys.help_html(TAB_LABEL)

    heading = help_text[: help_text.index(GROUP_TOOLS)].rsplit("<p", 1)[-1]
    assert HEADING in heading, "a heading takes the group-title colour"
    assert "font-size" in heading and "border-bottom" in heading
    assert f"<b>{GROUP_TOOLS}</b>" not in help_text, (
        "a heading has to differ from a key, not merely repeat its weight"
    )


def test_the_curate_keys_are_grouped_by_what_they_are_for(window):
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_CHOOSE,
        GROUP_STATE,
        GROUP_VIEW,
        TAB_CURATE,
    )

    help_text = window._keys.help_html(TAB_CURATE)

    for heading in (GROUP_VIEW, GROUP_CHOOSE, GROUP_STATE):
        assert heading in help_text
    # In the order the work is done: look, choose, then change.
    assert (
        help_text.index(GROUP_VIEW)
        < help_text.index(GROUP_CHOOSE)
        < help_text.index(GROUP_STATE)
    )


def test_a_gesture_sits_under_the_same_heading_as_the_keys_it_belongs_with(window):
    """Picking frames out is one idea, done with Esc or with Ctrl-click.

    Listing the keys and then the gestures separately made the reader do the
    grouping the list is for.
    """
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_CHOOSE, GROUP_STATE, TAB_CURATE

    help_text = window._keys.help_html(TAB_CURATE)

    choose = help_text.index(GROUP_CHOOSE)
    state = help_text.index(GROUP_STATE)
    assert choose < help_text.index("Esc") < state
    assert choose < help_text.index("Ctrl-click a frame") < state


def test_the_key_that_opens_the_list_is_listed_last(window):
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_HELP, TAB_CURATE, TAB_LABEL

    for tab in (TAB_CURATE, TAB_LABEL):
        help_text = window._keys.help_html(tab)
        heading = help_text.index(GROUP_HELP)
        assert all(
            heading > help_text.index(other)
            for other in _headings()
            if other != GROUP_HELP and other in help_text
        ), tab


def test_a_tab_with_no_keys_of_its_own_says_so(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_EXPORT

    help_text = window._keys.help_html(TAB_EXPORT)

    assert "no keys of its own" in help_text
    assert "F1" in help_text


def test_the_project_arrows_are_scoped_to_the_project_list(window):
    """Window-wide, they would take Up/Down from the editable combo boxes on
    the same tab."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_ADD_DATA

    scoped = [b for b in window._keys.bindings if b.tab == TAB_ADD_DATA]

    assert {b.key for b in scoped} == {"Up", "Down"}
    assert all(b.scope is window.metadata_panel.projects for b in scoped)


# ---------------------------------------------------------------------------
# Zoom (regression: zooming in was refused whenever the current scale sat
# below the fit scale, which it does after every fit)
# ---------------------------------------------------------------------------

def test_zoom_in_works_from_the_fitted_scale(window):
    view = window.image_panel.pair.left
    view.fit()
    before = view.transform().m11()

    view.zoom_by(1.25)

    assert view.transform().m11() > before


def test_zoom_in_works_even_when_well_below_the_fit_scale(window):
    """The fit limit constrains zooming out; it must not block zooming in."""
    from qtpy.QtGui import QTransform

    view = window.image_panel.pair.left
    fit = view.fit_scale()
    view.setTransform(QTransform.fromScale(fit * 0.5, fit * 0.5))

    view.zoom_by(1.25)

    assert view.transform().m11() > fit * 0.5


def test_repeated_zoom_in_keeps_working(window):
    view = window.image_panel.pair.left
    scales = []
    for _ in range(4):
        view.zoom_by(1.25)
        scales.append(view.transform().m11())

    assert scales == sorted(scales)
    assert scales[-1] > scales[0]


# ---------------------------------------------------------------------------
# The edit target follows the view
# ---------------------------------------------------------------------------

def test_switching_view_points_the_target_at_that_modality(window):
    window._set_mode("fib")

    assert window.controller.target == "FIB"
    assert window.state_panel.target_buttons["FIB"].isChecked()

    window._set_mode("sem")
    assert window.controller.target == "SEM"
    assert window.state_panel.target_buttons["SEM"].isChecked()


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------

def test_corrected_confirmation_names_only_corrected_files(qt_app, window):
    """With a corrected SEM and a deselected FIB, only the SEM is at risk."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("corrected")
    window.state_panel.target_buttons["FIB"].setChecked(True)
    window.apply_state("deselected")
    settle(qt_app, lambda: window.controller.state_of("FIB") == "deselected")

    window.state_panel.target_buttons["both"].setChecked(True)
    at_risk = window.controller.corrected_records()

    assert [record.modality for record in at_risk] == ["SEM"]


def test_marking_corrected_without_an_annotation_asks_first(window, monkeypatch):
    """Corrected is the claim that a label was proofread, and a training export
    trusts it. The fixture's FIB images have no label at all, so calling them
    corrected is a claim about work that does not exist."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["FIB"].setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.Cancel,
    )

    window.apply_state("corrected")

    assert len(asked) == 1
    assert "1 with no label" in asked[0]
    assert window.controller.state_of("FIB") != "corrected"


def test_the_warning_names_the_lamella_so_it_can_be_found(window, monkeypatch):
    """Image filenames repeat between lamellae, so a list of them is not
    something anyone can act on."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["FIB"].setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.Cancel,
    )

    window.apply_state("corrected")

    assert "01-alpha" in asked[0]


def test_marking_an_annotated_label_corrected_asks_nothing(window, monkeypatch):
    """The guard has to be silent in the normal case, or it becomes a dialog
    people dismiss without reading."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: pytest.fail("asked about an annotated label"),
    )

    window.apply_state("corrected")

    assert window.controller.state_of("SEM") == "corrected"


def test_the_warning_does_not_apply_to_the_other_states(window, monkeypatch):
    """Deselecting an unlabelled image is the ordinary case, not a mistake."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["FIB"].setChecked(True)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: pytest.fail("asked about a deselection"),
    )

    window.apply_state("deselected")

    assert window.controller.state_of("FIB") == "deselected"


def test_cancelling_the_export_warning_writes_nothing(window, monkeypatch, tmp_path):
    """The gate has to stop the write, not merely appear before it."""
    from am_segmentation_manager.models import ImageRecord

    monkeypatch.setattr(
        window.export_controller, "selected_records",
        lambda *a, **k: [ImageRecord(image_path=tmp_path / "img.tif")],
    )
    monkeypatch.setattr(window, "_confirm_dropping_corrected", lambda: False)
    monkeypatch.setattr(
        "am_segmentation_manager.operations.csv.export_training_csv",
        lambda **kwargs: pytest.fail("wrote a training set the user cancelled"),
    )

    window._export_training(str(tmp_path))

    assert list(tmp_path.glob("*.csv")) == []


def _tick_lamella(window, row: int = 0) -> None:
    """Tick a lamella through the widget, as a user would."""
    window.project_panel.lamella_list.item(row).setCheckState(Qt.Checked)


def test_ticking_a_lamella_drives_the_controller(window):
    _tick_lamella(window)

    assert window.controller.lamella_checks() == [True, False]
    assert window.controller.describe_selection() == "1 lamella"


def test_a_tick_widens_what_a_change_would_move(window):
    """The reach line is the number that has to agree with the ticks."""
    frame_only = window.controller.reach().images

    _tick_lamella(window)

    assert window.controller.reach().images > frame_only
    assert str(window.controller.reach().images) in window.state_panel.reach.text()
    assert "1 lamella" in window.state_panel.applies_to.text()


def test_a_ticked_selection_is_confirmed_before_anything_moves(qt_app, window, monkeypatch):
    """More than the frame on screen must not move without an explicit yes."""
    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    asked = []
    monkeypatch.setattr(window, "_confirm_bulk", lambda state: asked.append(state) or False)

    window.apply_state("deselected")

    assert asked == ["deselected"]
    assert all(state == "selected" for state in window.controller.frame_states())


def test_a_confirmed_change_applies_to_the_whole_ticked_lamella(qt_app, window, monkeypatch):
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["both"].setChecked(True)
    window.metadata_panel.enable.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(window, "_confirm_bulk", lambda state: True)

    window.apply_state("deselected")
    settle(qt_app, lambda: window.controller.frame_states()[0] == "deselected")

    assert all(state == "deselected" for state in window.controller.frame_states())
    assert "deselected" in window.state_panel.last_change.text()


def test_nothing_ticked_is_not_confirmed(window, monkeypatch):
    """The common case -- one frame -- must not ask every time."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["both"].setChecked(True)
    monkeypatch.setattr(window, "_confirm_bulk", lambda state: pytest.fail("should not ask"))

    window.apply_state("deselected")

    assert window.controller.combined_state() == "deselected"


def test_ticks_survive_a_list_rebuild(window):
    """Every refresh rebuilds the rows, and the ticks decide what moves -- so
    losing them would silently narrow the next change."""
    _tick_lamella(window)

    window.refresh(reload_lists=True)

    assert window.project_panel.lamella_list.item(0).checkState() == Qt.Checked
    assert window.controller.lamella_checks() == [True, False]


def test_tick_all_and_none_cover_the_list(window):
    window.project_panel.lamellae_check_all.emit(True)
    assert window.controller.lamella_checks() == [True, True]

    window.project_panel.lamellae_check_all.emit(False)
    assert window.controller.lamella_checks() == [False, False]


def test_the_title_says_how_many_are_ticked(window):
    """The lists scroll, so a tick out of view still has to be countable."""
    _tick_lamella(window)

    assert window.project_panel.lamella_box.title() == "Lamellae (1 of 2 ticked)"


def test_the_summary_follows_the_lamella_being_looked_at(window):
    """It counts this lamella's frames, so leaving it on the previous one
    reports a lamella nobody is looking at."""
    assert window.project_panel.summary.text().startswith("4 frames")

    window._select_lamella(1)

    assert window.project_panel.summary.text().startswith("2 frames")


def test_ticking_a_lamella_unticks_the_project_box_on_screen(window):
    """The clearing has to reach the widget, not only the model: a box still
    drawn ticked is exactly the tick somebody counts on."""
    window.project_panel.project_list.project_item(0).setCheckState(0, Qt.Checked)

    _tick_lamella(window)

    assert (
        window.project_panel.project_list.project_item(0).checkState(0) == Qt.Unchecked
    )
    assert window.project_panel.project_box.title() == "Projects (1)"


def test_picking_a_frame_unticks_the_lamella_box_on_screen(window):
    _tick_lamella(window)

    window._toggle_frame(0)

    assert window.project_panel.lamella_list.item(0).checkState() == Qt.Unchecked
    assert window.controller.describe_selection() == "1 frame"


def test_the_modality_control_sits_with_the_buttons_that_act(window):
    """Which rows is chosen on the left; which modality and which state are
    read together, beside the buttons that move them."""
    assert window.state_panel.target_buttons["SEM"].parent() is not None
    assert window.state_panel.isAncestorOf(window.state_panel.target_buttons["SEM"])
    assert not hasattr(window.project_panel, "target_buttons")


def test_the_reach_line_names_the_lamellae_and_the_modalities(window):
    """The count the user asked for before running an operation: how many
    lamellae, and how many images of each beam."""
    window.state_panel.target_buttons["both"].setChecked(True)
    _tick_lamella(window)

    text = window.state_panel.reach.text()

    assert "1 lamella" in text
    assert "SEM" in text and "FIB" in text


# ---------------------------------------------------------------------------
# Operations run against the same selection a state change does
# ---------------------------------------------------------------------------

@pytest.fixture
def inline_operations(window, monkeypatch):
    """Run operations on the calling thread and accept every dialog.

    ``_run_operation`` is the one seam between the window and its worker
    thread; substituting it leaves the whole preview-confirm-apply ladder
    running for real, which is where the behaviour worth testing is.
    """
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    monkeypatch.setattr(window, "_show_report", lambda title, report: None)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window, "_confirm_report",
        lambda title, report, action: asked.append((title, action)) or True,
    )
    return asked


def test_copying_the_selection_writes_it_into_the_chosen_folder(
    window, inline_operations, tmp_path: Path
):
    """The Export tab's copy: the filters decide what, the dialog decides
    where, and the same preview-then-apply ladder decides whether."""
    destination = tmp_path / "deposit"

    window._copy_files(str(destination))

    assert inline_operations == [("Export images and labels", "Apply")]
    copied = sorted(path.name for path in destination.rglob("*.tif"))
    assert copied, "the images and their labels land under the folder"
    assert (destination / "AutoLamella-2026-07-28-13-57" / "experiment.yaml").exists(), (
        "the file that makes the copy indexable comes too"
    )


def test_looking_at_a_collection_does_not_record_anything(window):
    """The rule the whole opt-in property now rests on: only a write starts a
    log, so opening and reading a collection leaves it exactly as it was."""
    from am_segmentation_manager.history import history_file_path

    root = Path(window.controller.start_path)
    window.rescan()

    assert not any(
        history_file_path(project.path).is_file()
        for project in window.controller.projects
    ), f"a scan of {root} wrote a log"


def test_changing_a_state_starts_the_record_and_writes_into_it(window):
    """A state change is one of the three doors a change comes through, and it
    recorded nothing at all before -- not even where a log already existed."""
    from am_segmentation_manager.history import read_log

    window.state_panel.enable_check.setChecked(True)
    window.apply_state("deselected")

    project = window.controller.projects[0]
    kinds = [event.kind for event in read_log(project.path).events]
    assert "log-created" in kinds, "the write started the log"
    assert "baseline" in kinds, "and recorded what the project held first"
    assert "state-changed" in kinds, "and then recorded the change itself"


def test_a_copy_made_here_can_be_merged_back(
    window, inline_operations, tmp_path: Path
):
    """The window's own export has to close the round trip it starts.

    A copy carrying no fork point is refused by `merge` outright, so an export
    that wrote one only from the command line meant the window could make
    copies it could never take back.
    """
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.operations.init_history import (
        apply_baseline,
        plan_baseline,
    )
    from am_segmentation_manager.operations.merge import plan_merge

    root = Path(window.controller.start_path)
    apply_baseline(
        plan_baseline(window.export_controller.projects), recorder=Recorder.for_run()
    )
    destination = tmp_path / "laptop"

    window._copy_files(str(destination))

    assert plan_merge(destination, root) is not None


def test_a_copy_records_what_it_was_narrowed_to(
    window, inline_operations, tmp_path: Path
):
    """In the tab's own words: a copy made here was narrowed by these controls,
    and writing it as command-line flags would describe a run nobody made."""
    from am_segmentation_manager.operations.export_record import read_export_record

    destination = tmp_path / "subset"

    window._copy_files(str(destination))

    record = read_export_record(destination)
    assert record is not None
    assert record.scope == window.export_controller.describe_filters()


def test_merging_waits_on_the_curate_editing_guard(
    window, inline_operations, monkeypatch, tmp_path: Path
):
    """A merge writes labels into this collection, which is what that guard
    protects -- not Add data's, which governs metadata."""
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )

    window._merge_copy(str(tmp_path / "copy"))

    assert shown == ["Editing is off"]
    assert inline_operations == []


def test_merging_a_folder_that_is_not_a_copy_is_refused(
    window, inline_operations, tmp_path: Path
):
    """Refused rather than half-read: the wrong ancestor discards work while
    reporting success."""
    window.state_panel.enable_check.setChecked(True)
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    reports: list = []
    window._show_report = lambda title, report: reports.append(report)

    window._merge_copy(str(stranger))

    assert inline_operations == [], "nothing was offered to apply"
    assert reports and "fork point" in reports[0].headline


def test_copying_previews_before_it_writes(window, inline_operations, monkeypatch):
    """Refusing the preview must leave the folder untouched, gigabytes being
    the usual size of this one."""
    monkeypatch.setattr(window, "_confirm_report", lambda *args: False)
    destination = Path(window.controller.start_path).parent / "refused"

    window._copy_files(str(destination))

    assert not destination.exists()


def test_copying_into_the_collection_is_refused(window, inline_operations):
    """It would be walking into its own output, and the copy would be scanned
    as part of the collection it came from."""
    inside = Path(window.controller.start_path) / "export"

    window._copy_files(str(inside))

    assert inline_operations == [], "a refusal is shown, not offered"
    assert not inside.exists()


# ---------------------------------------------------------------------------
# Reverting, which acts on the selection like everything else in that panel
# ---------------------------------------------------------------------------

def revert_with(window, undoing="conversion", savepoint=""):
    """Answer the revert dialog without opening it."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import operation_actions as module

    class _Answered:
        def __init__(self, *_args):
            self.undoing, self.savepoint = undoing, savepoint

        def exec_(self):
            return QDialog.Accepted

    original = module.RevertDialog
    module.RevertDialog = _Answered
    try:
        window._revert()
    finally:
        module.RevertDialog = original


def test_reverting_waits_on_the_editing_guard(window, inline_operations, monkeypatch):
    """It writes labels, so it is behind the guard every write in that panel is."""
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )

    revert_with(window)

    assert shown == ["Editing is off"]
    assert inline_operations == []


def test_reverting_reports_what_it_cannot_reach(window, inline_operations):
    """Nothing here has been converted, so nothing has a version to go back to
    -- and saying which and why beats a count of zero."""
    window.state_panel.enable_check.setChecked(True)
    reports: list = []
    window._show_report = lambda title, report: reports.append(report)

    revert_with(window)

    assert inline_operations == [], "nothing was offered to apply"
    assert reports and "Nothing to put back" in reports[0].headline


def test_reverting_proofread_work_is_refused_until_it_is_named(
    window, inline_operations, monkeypatch
):
    """CLAUDE.md's rule: no automated operation replaces a corrected label
    without explicit confirmation, and the window must not make that a
    checkbox somebody clicks past."""
    from am_segmentation_manager.gui.report import OperationReport

    window.state_panel.enable_check.setChecked(True)
    asked: list = []
    monkeypatch.setattr(
        window.history_controller, "revert_labels",
        lambda records, **kwargs: (
            asked.append(kwargs.get("allow_corrected"))
            or OperationReport(
                "2 label(s) are proofread work.", ("a", "b"), 2, blocked="corrected"
            )
        ),
    )
    monkeypatch.setattr(window, "_confirm_report", lambda *args: False)

    revert_with(window)

    assert asked == [False], "the first run asked without permission, and stopped"


# ---------------------------------------------------------------------------
# Savepoints, which are collection-wide and so live beside the other two
# ---------------------------------------------------------------------------

def take_savepoint(window, title="round 1", note="", gap=False):
    """Answer the savepoint dialog without opening it."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import operation_actions as module

    class _Answered:
        def __init__(self, *_args):
            self.title, self.description, self.closing_gap = title, note, gap

        def exec_(self):
            return QDialog.Accepted

    original = module.SavepointDialog
    module.SavepointDialog = _Answered
    try:
        window._take_savepoint()
    finally:
        module.SavepointDialog = original


def test_a_savepoint_names_every_project_in_the_scan(window, inline_operations):
    """A round is over the dataset a model was trained on, so there is nothing
    for a savepoint to narrow to and no selection is consulted."""
    from am_segmentation_manager.operations.savepoint import list_savepoints

    window.metadata_panel.enable.setChecked(True)
    take_savepoint(window, title="round 1", note="as proofread in August")

    assert inline_operations == [("Savepoint", "Apply")]
    points = list_savepoints(window.controller.projects)
    assert [point.title for point in points] == ["round 1"]
    assert points[0].found == len(window.controller.projects)


def test_a_savepoint_starts_the_record_it_writes_into(window, inline_operations):
    """It is a write like any other, so it comes through the same door."""
    from am_segmentation_manager.history import history_file_path

    window.metadata_panel.enable.setChecked(True)
    take_savepoint(window)

    assert all(
        history_file_path(project.path).is_file()
        for project in window.controller.projects
    )


def test_a_savepoint_waits_on_the_editing_guard(window, inline_operations, monkeypatch):
    """It writes, so it is behind the same guard recompressing is."""
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )

    take_savepoint(window)

    assert shown == ["Editing is off"]
    assert inline_operations == [], "nothing was previewed, let alone applied"


def restore_savepoint(window, savepoint="round 1"):
    """Answer the restore chooser without opening it."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import operation_actions as module

    class _Answered:
        def __init__(self, *_args, **_kwargs):
            self.undoing, self.savepoint = "", savepoint

        def exec_(self):
            return QDialog.Accepted

    original = module.RevertDialog
    module.RevertDialog = _Answered
    try:
        window._restore_savepoint()
    finally:
        module.RevertDialog = original


def test_a_savepoint_can_be_put_back_over_the_whole_collection(
    window, inline_operations
):
    """It is taken over the whole collection from this tab, so putting the
    whole collection back belongs here too -- going to Curate and ticking
    everything first was the asymmetry this closes."""
    import numpy as np
    import tifffile

    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.label_save import (
        LabelToSave,
        save_labels,
    )

    window.metadata_panel.enable.setChecked(True)
    take_savepoint(window, title="round 1")
    inline_operations.clear()

    # Through the ordinary save, so what it replaces is archived -- a label
    # changed any other way has nothing kept to go back to.
    label = next(
        label
        for project in window.controller.projects
        for record in project.image_records
        for label in record.labels
    )
    before = tifffile.imread(str(label))
    save_labels(
        [
            LabelToSave(
                destination=label,
                label=np.zeros_like(before),
                fingerprint=fingerprint_of(label),
            )
        ],
        allow_corrected=True,
        recorder=Recorder.for_run(),
    )

    restore_savepoint(window, "round 1")

    assert inline_operations, "the restore was never previewed"
    assert inline_operations[-1][0] == "Restore savepoint"
    assert np.array_equal(tifffile.imread(str(label)), before), "it did not go back"


def test_restoring_says_so_when_no_savepoint_has_been_taken(window, monkeypatch):
    """Rather than opening a chooser with nothing in it."""
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )
    window.metadata_panel.enable.setChecked(True)

    restore_savepoint(window)

    assert shown == ["Restore savepoint"]


def test_restoring_waits_on_the_editing_guard(window, inline_operations, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )

    restore_savepoint(window)

    assert shown == ["Editing is off"]
    assert inline_operations == []


def test_reclaiming_keeps_everything_while_no_round_is_closed(
    window, inline_operations
):
    """The savepoints are the whole policy, so with none taken the archive is
    one open round and nothing goes. Saying why beats reporting zero."""
    window.metadata_panel.enable.setChecked(True)
    reports: list = []
    window._show_report = lambda title, report: reports.append(report)

    window._reclaim_archive()

    assert inline_operations == [], "nothing was offered to apply"
    assert reports and "savepoint" in reports[0].headline


def test_reclaiming_waits_on_the_editing_guard(window, inline_operations, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )

    window._reclaim_archive()

    assert shown == ["Editing is off"]


def test_the_archive_is_not_measured_while_nothing_keeps_versions(
    window, monkeypatch
):
    """One stat per project rather than a walk of each: the ordinary case is a
    collection that has never replaced a label, and it must cost nothing."""
    walked: list = []
    monkeypatch.setattr(
        window.history_controller, "measure_archive",
        lambda projects: walked.append(projects),
    )

    window.tabs.setCurrentWidget(window.metadata_panel)

    assert walked == [], "opening the tab walked the archive"
    assert window.metadata_panel.archive.text() == ""


def test_compressing_acts_on_the_whole_collection(window, inline_operations):
    """Recompression preserves the pixels exactly, so there is no label it
    should be withheld from and nothing to select."""
    window.metadata_panel.enable.setChecked(True)

    window._compress_labels()

    assert inline_operations == [("Compress labels", "Apply")]
    assert "Compress labels" in window.state_panel.last_change.text()


def test_compressing_reaches_a_lamella_that_is_not_on_screen(
    window, inline_operations, monkeypatch
):
    """The fixture's second lamella is never selected, and its labels are
    exactly the ones a per-selection button would have left uncompressed."""
    window.metadata_panel.enable.setChecked(True)
    seen: list = []
    real = window.operations_controller.compress_labels
    monkeypatch.setattr(
        window.operations_controller, "compress_labels",
        lambda records, **kwargs: seen.append(list(records)) or real(records, **kwargs),
    )

    window._compress_labels()

    assert seen, "the operation never ran"
    assert {image.lamella for image in seen[0]} == {"01-alpha", "02-beta"}


def test_compressing_waits_for_the_add_data_guard(window, monkeypatch):
    """It writes files, and it is offered on the tab whose guard governs
    writing to the collection."""
    monkeypatch.setattr(
        window, "_run_operation", lambda *args: pytest.fail("should not have started")
    )
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information", lambda *args, **kwargs: None
    )

    window._compress_labels()


def test_the_predict_button_is_withheld_where_inference_is_not_installed(window):
    """A separate install, so the window must say so on the button rather than
    at the end of a dialog opened expecting it to work."""
    window.operations_panel.set_prediction_available(False)

    assert not window.operations_panel.predict_button.isEnabled()
    assert "setup_ml.sh" in window.operations_panel.predict_button.toolTip()


def test_browsing_for_a_model_predicts_nothing_by_itself(window):
    """Choosing a model fills the box. Nothing runs until Predict is pressed,
    which is what lets the window check that editing is on before a file is
    ever asked for."""
    started: list[str] = []
    window.operations_panel.predict_requested.connect(started.append)

    window.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    assert window.operations_panel.model() == "/models/AM_SEM_All_V01.pth"
    assert started == []


# ---------------------------------------------------------------------------
# The model box follows the view
# ---------------------------------------------------------------------------
#
# An SEM model over a FIB image returns a confident, entirely wrong prediction,
# and the series differ from each other as much again. Which model is right
# stays the user's to say; what these pin is that the box is not offering the
# answer to a different question.

def test_the_model_box_is_kept_per_series_and_beam(window, settings):
    window.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    window._set_mode("fib", from_user=True)

    assert window.operations_panel.model() == ""
    assert settings.value("model/adaptive/SEM", "", type=str) == (
        "/models/AM_SEM_All_V01.pth"
    )


def test_the_model_comes_back_when_the_view_does(window):
    window.operations_panel.set_model("/models/AM_SEM_All_V01.pth")
    window._set_mode("fib", from_user=True)
    window.operations_panel.set_model("/models/AM_FIB_All_V02.pth")

    window._set_mode("sem", from_user=True)
    assert window.operations_panel.model() == "/models/AM_SEM_All_V01.pth"

    window._set_mode("fib", from_user=True)
    assert window.operations_panel.model() == "/models/AM_FIB_All_V02.pth"


def test_the_plot_view_leaves_the_model_box_alone(window):
    """It names no beam, so there is no slot to put a typed path into."""
    window.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    window._set_mode("plot", from_user=True)

    assert window.operations_panel.model() == "/models/AM_SEM_All_V01.pth"


def test_a_slot_never_filled_opens_on_the_one_path_kept_before(qt_app, settings):
    """An existing window opens on the model it always did, rather than on an
    empty box because the preference gained a shape."""
    settings.setValue("model_directory", "/models/AM_SEM_All_V01.pth")
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert window.operations_panel.model() == "/models/AM_SEM_All_V01.pth"
    finally:
        destroy(window)


def test_closing_keeps_what_was_typed_into_the_box(window, settings):
    """Predicting is not the only way a path gets chosen, and losing one that
    was browsed for means browsing for it again next session."""
    window.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    window.close()

    assert settings.value("model/adaptive/SEM", "", type=str) == (
        "/models/AM_SEM_All_V01.pth"
    )


def test_predicting_with_editing_off_says_so_without_asking_for_anything(
    window, monkeypatch
):
    """The complaint this order exists to prevent: being asked for a model
    before being told the operation was never going to run."""
    window.state_panel.enable_check.setChecked(False)
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda parent, title, text, *a, **k: shown.append(title),
    )

    window._predict("")

    assert shown == ["Editing is off"]


def test_pressing_predict_with_an_empty_box_asks_for_a_model_last(window, monkeypatch):
    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems", lambda: []
    )
    flashed: list[str] = []
    monkeypatch.setattr(window, "_flash", flashed.append)

    window._predict("")

    assert flashed == ["Choose a model to predict with"]


def test_pressing_predict_does_not_import_torch_into_this_process(
    window, monkeypatch, inline_operations, tmp_path: Path
):
    """The window predicts in a child precisely so that torch is never loaded
    here -- and asking whether torch *would* load is asking by loading it. The
    cheap probe that greys out the button locates the modules instead."""
    from am_segmentation_manager.gui import operations_controller
    from am_segmentation_manager.operations.predict_process import ProcessOutcome

    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems",
        lambda: pytest.fail("the window must not import torch to check"),
    )
    monkeypatch.setattr(
        operations_controller, "predict_out_of_process",
        lambda *args, **kwargs: ProcessOutcome(written=0, total=0),
    )
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"x")

    window._predict(str(weights))


def test_a_missing_inference_stack_comes_back_from_the_child(
    window, monkeypatch, inline_operations, tmp_path: Path
):
    """It is the process that has to import torch that finds out, so the news
    arrives as a report rather than as a check this process could make."""
    from am_segmentation_manager.gui import operations_controller
    from am_segmentation_manager.operations.predict_process import ProcessOutcome

    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(
        operations_controller, "predict_out_of_process",
        lambda *args, **kwargs: ProcessOutcome(
            error="RuntimeError: Loading a model needs torch: not installed"
        ),
    )
    reported: list[str] = []
    monkeypatch.setattr(
        window, "_show_report",
        lambda title, report: reported.append(report.headline),
    )
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"x")

    window._predict(str(weights))

    assert reported and "needs torch" in reported[0]


def test_only_the_applying_run_is_given_something_to_stop(window, monkeypatch):
    """A preview opens no file and answers in milliseconds, so a Stop button on
    one is a button with nothing to stop."""
    from am_segmentation_manager.gui.report import OperationReport

    given: list[object] = []
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: (
            given.append(cancellation) or on_done(work())
        ),
    )
    monkeypatch.setattr(window, "_show_report", lambda title, report: None)
    monkeypatch.setattr(window, "_confirm_report", lambda title, report, action: True)
    monkeypatch.setattr(window, "_operation_applied", lambda *a, **k: None)
    cancellation = Cancellation()

    window._operate(
        "Predict labels",
        lambda allow, dry: OperationReport("preview" if dry else "applied", count=1),
        cancellation=cancellation,
    )

    assert given == [None, cancellation]


def _accept_dialog(monkeypatch, chosen, *, accepted=True, opened=None):
    """Stand in for the Add projects dialog, which is driven in its own tests."""

    class FakeDialog:
        def __init__(self, parent, controller, root):
            if opened is not None:
                opened.append(root)

        def exec_(self):
            return accepted

        def chosen(self):
            return list(chosen)

    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.operation_actions.AddProjectsDialog", FakeDialog
    )


def _unlabelled_record(window):
    """Return a record the scan found no label for, as a conversion's target."""
    return next(
        record
        for project in window.controller.projects
        for record in project.image_records
        if not record.labels
    )


def test_converting_does_not_walk_the_collection_again(
    project: Path, window, inline_operations, monkeypatch
):
    """The scan that used to follow cost a full read of the archive, which on a
    real collection is most of the time the operation takes. What it went
    looking for is the label the run just wrote, and the run says where."""
    from am_segmentation_manager.gui.report import OperationReport
    from am_segmentation_manager.operations.label_io import label_destination

    record = _unlabelled_record(window)
    written = label_destination(record)
    monkeypatch.setattr(
        window.operations_controller, "convert_predictions",
        lambda records, allow_overwrite=False, dry_run=False: OperationReport(
            "Wrote 1 label(s) from predictions.", (), 1,
            applied=not dry_run,
            written=() if dry_run else (written,),
        ),
    )
    monkeypatch.setattr(
        window, "rescan", lambda: pytest.fail("converting should not rescan")
    )
    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)

    window._convert_predictions()

    assert record.labels == [written]


def test_predicting_does_not_walk_the_collection_again(
    project: Path, window, inline_operations, monkeypatch
):
    """Preparing labels is predict then convert against the same selection, so
    the scan was paid twice for one decision. Convert reads the predictions off
    the records, so this is also what makes the second step see the first."""
    from am_segmentation_manager.gui.report import OperationReport
    from am_segmentation_manager.operations.predict import prediction_destination

    record = _unlabelled_record(window)
    written = prediction_destination(record)
    monkeypatch.setattr(
        window.operations_controller, "predict",
        lambda *args, **kwargs: OperationReport(
            "Predicted 1 SEM image(s).", (), 1,
            applied=not kwargs.get("dry_run"),
            written=() if kwargs.get("dry_run") else (written,),
        ),
    )
    monkeypatch.setattr(
        window, "rescan", lambda: pytest.fail("predicting should not rescan")
    )
    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)

    window._predict(str(project / "model.pth"))

    assert record.predictions == [written]


def test_creating_projects_rescans_so_the_new_projects_appear(
    project: Path, window, inline_operations, monkeypatch
):
    """Making an unindexed dataset into a project is the point of this.

    A refresh recounts what is already in memory; only a scan can see the
    project the operation just created, and without one the Add data list keeps
    showing the collection as it was before.
    """
    unindexed = project / "loose-dataset" / "01-lamella" / "stage" / "sem"
    unindexed.mkdir(parents=True)
    (unindexed / "img.tif").write_bytes(b"i")
    window.metadata_panel.enable.setChecked(True)
    assert len(window.controller.projects) == 1

    _accept_dialog(monkeypatch, [project / "loose-dataset"])
    window._add_projects()

    assert (project / "loose-dataset" / "AM_metadata.yaml").exists()
    assert len(window.controller.projects) == 2
    assert len(window.metadata_panel.projects.project_items()) == 2


def test_cancelling_the_dialog_writes_nothing(project: Path, window, monkeypatch):
    """The dialog is where the decision is made, so dismissing it is the whole
    answer -- nothing behind it should have started."""
    unindexed = project / "loose-dataset" / "01-lamella" / "stage" / "sem"
    unindexed.mkdir(parents=True)
    (unindexed / "img.tif").write_bytes(b"i")
    window.metadata_panel.enable.setChecked(True)

    _accept_dialog(monkeypatch, [project / "loose-dataset"], accepted=False)
    window._add_projects()

    assert not (project / "loose-dataset" / "AM_metadata.yaml").exists()


def test_creating_projects_needs_editing_to_be_on(project: Path, window, monkeypatch):
    """The same guard everything that writes is behind."""
    shown: list = []
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.main_window.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args),
    )
    opened: list = []
    _accept_dialog(monkeypatch, [], opened=opened)

    window._add_projects()

    assert shown and not opened


def test_an_operation_previews_before_it_applies(window, monkeypatch):
    """The preview must be a real run of the operation, not a guess at one."""
    window.metadata_panel.enable.setChecked(True)
    reports: list = []
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    monkeypatch.setattr(window, "_show_report", lambda title, report: None)
    monkeypatch.setattr(
        window, "_confirm_report",
        lambda title, report, action: reports.append(report) or False,
    )

    window._compress_labels()

    assert len(reports) == 1
    assert reports[0].applied is False
    assert reports[0].count > 0


def test_corrected_labels_are_confirmed_separately_before_being_touched(
    qt_app, window, monkeypatch
):
    """Two confirmations, in order of what is at stake.

    The same ladder apply_state uses: first that this is more than the frame on
    screen, then that proofread work is at risk. Recompressing rewrites the
    file, so a corrected label must not be reached by one Apply.
    """
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["both"].setChecked(True)
    window.metadata_panel.enable.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(window, "_confirm_bulk", lambda state: True)
    # The fixture's FIB images have no labels, which this ladder is not about.
    monkeypatch.setattr(window, "_confirm_unannotated", lambda: True)
    window.apply_state("corrected")
    settle(qt_app, lambda: window.controller.frame_states()[0] == "corrected")

    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    monkeypatch.setattr(window, "_show_report", lambda title, report: None)
    asked: list[str] = []

    def refuse(title, report, action):
        asked.append(action)
        return False

    monkeypatch.setattr(window, "_confirm_report", refuse)
    window._compress_labels()
    assert asked == ["Include them"]

    # Confirming the corrected question re-previews and then asks to apply.
    asked.clear()
    monkeypatch.setattr(
        window, "_confirm_report",
        lambda title, report, action: asked.append(action) or True,
    )
    window._compress_labels()

    assert asked == ["Include them", "Apply"]


def test_finding_specks_needs_no_editing_guard(window, inline_operations, monkeypatch):
    """It reads and writes nothing, and asking someone to switch editing on to
    look at their own data would be a lock on the wrong door."""
    shown = []
    monkeypatch.setattr(window, "_show_report", lambda title, report: shown.append(title))
    _tick_lamella(window)
    assert not window.state_panel.enable_check.isChecked()

    window._find_specks(64)

    assert shown == ["Find specks"]


def test_finding_specks_needs_something_ticked(window, monkeypatch):
    """Unticked it would sweep whatever happened to be on screen, which is not
    what a report about datasets is for."""
    monkeypatch.setattr(window.controller, "scoped_records", list)
    monkeypatch.setattr(
        window, "_run_operation",
        lambda *args, **kwargs: pytest.fail("should not have started"),
    )

    window._find_specks(64)


def test_an_operation_will_not_run_with_editing_off(window, monkeypatch):
    """The same guard the state buttons have: writing files is writing files."""
    _tick_lamella(window)
    monkeypatch.setattr(
        window, "_run_operation",
        lambda *args: pytest.fail("should not have started"),
    )
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information", lambda *args, **kwargs: None
    )

    window._convert_predictions()


def test_an_operation_needs_something_selected(window, monkeypatch):
    """With no project scanned there is nothing for an operation to act on."""
    window.state_panel.enable_check.setChecked(True)
    monkeypatch.setattr(window.controller, "scoped_records", list)
    monkeypatch.setattr(
        window, "_run_operation",
        lambda *args: pytest.fail("should not have started"),
    )

    window._convert_predictions()


def test_the_collection_actions_wait_for_a_folder(qt_app, settings):
    """All three act on the whole scan root, so none is offered before one
    exists. Only recompressing then waits on the editing guard as well: the
    other two write nothing by themselves, and naming projects is how a
    collection gets its first row -- a disabled button there cannot say what it
    is waiting for, so the window answers that when it is pressed."""
    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        panel = fresh.metadata_panel
        assert not panel.survey_button.isEnabled()
        assert not panel.init_button.isEnabled()
        assert not panel.compress_button.isEnabled()

        panel.set_root(Path("/somewhere"))
        assert panel.survey_button.isEnabled()
        assert panel.init_button.isEnabled()
        assert not panel.compress_button.isEnabled()

        panel.enable.setChecked(True)
        assert panel.compress_button.isEnabled()
    finally:
        destroy(fresh)


def test_a_first_run_opens_on_add_data(qt_app, settings):
    """Curate is the right tab to come back to and the wrong one to arrive on.

    With nothing ever scanned it holds an empty tree, no cue about what comes
    first, and none of the three buttons that could do anything about it.
    """
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        assert fresh.tabs.currentWidget() is fresh.metadata_panel
    finally:
        destroy(fresh)


def test_the_first_scan_hands_over_to_curate(qt_app, project: Path, settings):
    """Opening on Add data is only half of it: once a folder has been scanned
    there is something to curate, and leaving the reviewer on the tab they
    started from would hide the tree they just filled."""
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        assert fresh.tabs.currentWidget() is fresh.metadata_panel

        fresh.scan(str(project))

        assert fresh.tabs.currentWidget() is fresh.curate
    finally:
        destroy(fresh)


def test_a_later_scan_leaves_the_reviewer_where_they_are(window, project: Path):
    """The hand-over happens once. Pressing Scan from Export is how a reviewer
    picks up files changed outside the window, and being thrown to Curate for
    it would cost them the filters they had set."""
    window.tabs.setCurrentWidget(window.export_panel)

    window.scan(str(project))

    assert window.tabs.currentWidget() is window.export_panel


def test_the_title_names_the_folder_that_is_open(window, project: Path):
    """Two windows on two collections are told apart by this and nothing else."""
    assert project.name in window.windowTitle()
    assert PRODUCT_NAME in window.windowTitle()


def test_a_remembered_folder_still_opens_on_curate(qt_app, settings, tmp_path):
    """Which is the ordinary case: a scan root, once chosen, is remembered, so
    this changes where a first run lands and nothing else."""
    from am_segmentation_manager.gui.widgets.main_window import SETTING_SCAN_ROOT, CurationWindow

    settings.setValue(SETTING_SCAN_ROOT, str(tmp_path))
    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        assert fresh.tabs.currentWidget() is fresh.curate
    finally:
        destroy(fresh)


def _menu_items(window, title: str) -> dict[str, str]:
    """Return ``{item text: shortcut}`` for one menu of the bar.

    Plain strings rather than live QActions. PySide6 keeps a wrapper alive only
    while the wrapper that vended it lives, so a QMenu read out of the bar and
    returned past its QAction -- or a QAction returned past its QMenu -- is an
    already-deleted object by the time anything asks it a question. Everything
    is read while the whole chain is still in scope. To *press* an item, go
    through ``window._menu_actions``, which the window holds for this reason.
    """
    bar_action = next(
        action
        for action in window.menuBar().actions()
        if action.text().replace("&", "") == title
    )
    menu = bar_action.menu()
    return {
        item.text(): item.shortcut().toString()
        for item in menu.actions()
        if not item.isSeparator()
    }


# ---------------------------------------------------------------------------
# Who changes are recorded under
# ---------------------------------------------------------------------------

def test_the_window_says_which_name_it_is_recording_under(window, monkeypatch):
    """Beside the mark, on every tab. A name reaches every entry written from
    anywhere in the window, so no one tab is the place to say it and there are
    far too many actions to warn at."""
    from am_segmentation_manager.history import resolve_author

    assert window._corner.author.text() in {
        "Recording without a name",
        f"Recording as {resolve_author()[0]}",
    }


#: What the window is known to hold that records. Named so this test fails
#: loudly if the search below ever stops finding anything, rather than passing
#: on an empty set; the search is what catches a *new* one.
KNOWN_RECORDERS = frozenset({
    "controller",
    "history_controller",
    "label_controller",
    "metadata_controller",
    "operations_controller",
    "label_panel.versions_controller",
})


def _things_that_record(window) -> dict[str, object]:
    """Return everything the window reaches that carries a recorder.

    Instance attributes one level into the panels as well as the window's own:
    the versions controller belongs to the Label tab rather than to the window,
    and a search that stopped at the window would miss exactly the one that was
    hardest to remember. The shared
    :class:`~am_segmentation_manager.gui.attribution.Attribution` is itself
    skipped -- it is what they record *through*, not a thing that records.
    """
    found: dict[str, object] = {}

    def offer(name: str, held: object) -> None:
        if isinstance(held, Attribution) or not hasattr(held, "recorder"):
            return
        found[name] = held

    for name, held in vars(window).items():
        offer(name, held)
        for inner, deeper in vars(held).items() if hasattr(held, "__dict__") else ():
            offer(f"{name}.{inner}", deeper)
    return found


def test_setting_a_name_reaches_everything_that_records(window, monkeypatch):
    """Including the curation controller, which records state changes -- the
    one the first version of this forgot.

    Asserted as "they all carry the same name as the line says", not as a
    literal: whether this machine has a name set is not this test's business.
    """
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module
    from am_segmentation_manager.gui.widgets.settings_dialog import author_summary

    class _Answered:
        def __init__(self, *_args):
            self.author = "Ada"
            # The dialog carries the preferences too; this test is about the
            # name, so it offers none to write.
            self.values: dict[str, int | bool] = {}

        def exec_(self):
            return QDialog.Accepted

    written: list[str] = []
    monkeypatch.setattr(module, "SettingsDialog", _Answered)
    monkeypatch.setattr(module, "set_author", written.append)

    window.show_settings()

    assert written == ["Ada"], "the name was not written"
    # Found rather than listed. A list here would have to be extended by
    # whoever adds a controller, which is the same omission this test exists
    # to catch -- so it asks the window what it holds.
    recording = _things_that_record(window)
    missing = KNOWN_RECORDERS - set(recording)
    assert not missing, f"the search stopped finding {sorted(missing)}"
    # Identity, not equality: two recorders built from the same file agree by
    # accident, and would go on agreeing while one of them was never updated.
    carried = {id(holder.recorder) for holder in recording.values()}
    assert len(carried) == 1, "something that records did not pick the name up"
    assert window._corner.author.text() == author_summary(
        window.controller.recorder.by
    )


def test_a_name_can_be_taken_off_again(window, monkeypatch):
    """A name stated once would otherwise be stated for ever, and somebody who
    set the wrong one needs an undo that is not typing something else."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module

    class _Answered:
        def __init__(self, *_args):
            self.author = ""
            self.values: dict[str, int | bool] = {}

        def exec_(self):
            return QDialog.Accepted

    cleared: list[bool] = []
    monkeypatch.setattr(module, "SettingsDialog", _Answered)
    monkeypatch.setattr(module, "clear_author", lambda: cleared.append(True))
    monkeypatch.setattr(module, "set_author", lambda name: pytest.fail("wrote a name"))

    window.show_settings()

    assert cleared == [True]


def test_the_line_says_plainly_when_nothing_is_recorded(window):
    """The case somebody is most likely to be in without knowing it."""
    from am_segmentation_manager.gui.widgets.settings_dialog import author_summary

    assert author_summary(None) == "Recording without a name"
    assert author_summary("Ada") == "Recording as Ada"
    assert "carry nobody" in author_summary(None, asked=True)


def test_the_window_has_the_menus_a_desktop_application_has(window):
    """Nothing here is a second way to drive the window: it is where someone
    looks for the things a window is expected to have and this one had nowhere
    to put."""
    assert set(_menu_items(window, "File")) == {
        "Open folder...", "Rescan", "Settings...", "Quit",
    }
    assert set(_menu_items(window, "Help")) == {
        "Keyboard shortcuts", f"About {PRODUCT_NAME}",
    }


def test_the_shortcut_menu_item_binds_no_key_of_its_own(window):
    """F1 is already an application shortcut, and Qt answers two bindings of
    one key with an ambiguous overload where neither fires reliably. The list
    teaches its own key instead -- it holds a row for F1."""
    assert _menu_items(window, "Help")["Keyboard shortcuts"] == ""


def test_the_shortcut_menu_item_opens_the_same_list_f1_does(window, qt_app):
    window._menu_actions["Keyboard shortcuts"].trigger()
    try:
        assert window._keys.help is not None
        assert not window._keys.help.isHidden()
    finally:
        if window._keys.help is not None:
            window._keys.help.close()


def test_a_stored_default_is_what_every_control_opens_at(qt_app, settings):
    """One store, four panels, and no control left on the packaged answer.

    The window is what reads the settings and hands them round, so a panel
    built on its own still opens at what the package ships -- which is why
    this asserts through a window rather than through a panel.
    """
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    preferences.write(settings, {
        "speck_pixels": 21,
        "label_opacity": 15,
        "brush_radius": 3,
        "train_share": 60,
        "seed": 7,
        "one_per_label": True,
    })
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert window.operations_panel.speck_size.value() == 21
        assert window.label_panel.speck_slider.value() == 21
        # Held on the controller as well: a slider moved without it would
        # report sizes nothing was counted at.
        assert window.label_panel.controller.speck_size == 21
        assert window.image_panel.alpha_slider.value() == 15
        assert window.label_panel.alpha_slider.value() == 15
        assert window.export_panel.train_share.value() == 60
        assert window.export_panel.seed.value() == 7
        assert window.export_panel.conditions.isChecked()
    finally:
        destroy(window)


def test_a_default_decides_what_a_control_opens_at_and_nothing_else(qt_app, settings):
    """The rule the whole idea rests on, and the reason it is safe to keep
    these on one machine: what a run produces comes from the packaged
    constants and the controls as they stand, never from a preference reached
    behind them. So changing one while a control is on screen changes nothing
    there -- the dialog says so, and this holds it."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.image_panel.alpha_slider.setValue(90)

        preferences.write(settings, {"label_opacity": 10})

        assert window.image_panel.alpha_slider.value() == 90, (
            "a control somebody is using is not moved under them"
        )
    finally:
        destroy(window)


def test_the_settings_dialog_offers_every_preference(qt_app, settings):
    """Built from the table, so a row added there reaches the dialog without
    an edit here -- and one that did not would be a setting nobody could
    change."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        offered = dialog.values

        assert sorted(offered) == sorted(entry.key for entry in preferences.PREFERENCES)
        assert offered == preferences.read(settings)
    finally:
        destroy(dialog)


def test_every_group_of_settings_gets_the_height_its_rows_need(qt_app, settings):
    """A word-wrapped note reports a one-line hint, and the squeeze lands elsewhere.

    Qt asks a label for its size hint before it knows how wide it will be, so
    a dialog sized from that comes up short and every widget below is pushed
    towards its minimum -- which showed as the Export box drawing four
    switches on top of the three rows under them, some way from the label that
    caused it. `controls.WrappingLabel` is the fix; this is what noticed.
    """
    from qtpy.QtWidgets import QGroupBox

    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        dialog.show()
        qt_app.processEvents()
        dialog.grab()

        short = {
            box.title(): (box.height(), box.layout().sizeHint().height())
            for box in dialog.findChildren(QGroupBox)
            if box.height() < box.layout().sizeHint().height()
        }

        assert not short, f"boxes shorter than their rows need: {short}"
    finally:
        destroy(dialog)


def test_nothing_in_the_filters_column_is_clipped_at_the_width_it_opens_at(
    qt_app, settings
):
    """Every compaction here trades height for width, and the width has to be there.

    Captions beside their boxes, a run of segments for the beam and two short
    switches on one line each save a row and cost columns -- and squeezed, Qt
    does not shorten a caption but cuts it from both ends. The first cut of
    this column drew the beam's segments blank and "Adaptive Polishing" as
    "Adaptive Po", which reads as a rendering fault rather than as a column
    that wants dragging.
    """
    from qtpy.QtWidgets import QAbstractButton, QLabel

    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.tabs.setCurrentWidget(window.export_panel)
        window.resize(1512, 900)
        window.show()
        qt_app.processEvents()
        window.grab()

        # PySide6 takes one type, not a tuple of them.
        block = window.export_panel.fixed_filters
        shown = [
            *block.findChildren(QLabel),
            *block.findChildren(QAbstractButton),
        ]
        clipped = {
            widget.text(): (widget.width(), widget.sizeHint().width())
            for widget in shown
            if widget.text() and widget.width() < widget.sizeHint().width()
        }

        assert not clipped, f"cut off in the Filters column: {clipped}"
    finally:
        destroy(window)


def test_the_settings_button_opens_what_f2_opens(window, monkeypatch):
    """The corner is where the name is shown, so it is where it is changed.

    F2 is the only other way in and nothing on screen said so -- the same gap
    the help button beside it fills for F1.
    """
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module

    opened: list[bool] = []

    class _Dismissed:
        def __init__(self, *_args):
            self.author = "Ada"
            self.values: dict[str, int | bool] = {}

        def exec_(self):
            opened.append(True)
            return QDialog.Rejected

    monkeypatch.setattr(module, "SettingsDialog", _Dismissed)

    window._corner.settings_button.click()

    assert opened == [True]


def test_the_product_name_is_not_drawn_like_the_line_beside_it(window):
    """Two facts of different kinds sat in one run of muted text.

    Left of the rule is whose window this is; right of it is what this machine
    is set to. Written alike they read as one phrase, and a reader had to work
    out where one ended and the other began. Asserted on the pixels, a
    stylesheet being the one thing that can be set and not take effect.
    """
    from qtpy.QtGui import QColor

    name = window._corner.product_name
    author = window._corner.author
    assert name.text() and author.text(), "both say something to compare"

    def darkest(widget) -> int:
        image = widget.grab().toImage()
        return min(
            QColor(image.pixel(x, y)).lightness()
            for y in range(image.height())
            for x in range(image.width())
        )

    assert darkest(name) < darkest(author), (
        "the product name is drawn more strongly than the machine's own state"
    )


def test_the_help_button_in_the_corner_opens_the_list(window):
    """F1 is the only way in, and nothing on screen mentions F1."""
    corner = window.tabs.cornerWidget(Qt.TopRightCorner)
    button = next(
        child for child in corner.findChildren(QToolButton) if child.text() == "?"
    )

    button.click()
    try:
        assert window._keys.help is not None
    finally:
        if window._keys.help is not None:
            window._keys.help.close()


def test_about_says_the_version_and_where_the_log_went(window, monkeypatch, tmp_path):
    """The two things anybody reporting a problem is asked for first, and
    neither was anywhere on screen. The log is read back from the handlers, so
    a run started with --log-file reports where its output actually went."""
    import logging

    from am_segmentation_manager import __version__

    shown = {}

    def capture(self):
        shown["text"] = self.text()
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", capture, raising=False)
    handler = logging.FileHandler(tmp_path / "run.log")
    logging.getLogger().addHandler(handler)
    try:
        window._show_about()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    assert __version__ in shown["text"]
    assert "run.log" in shown["text"]


def test_the_tab_says_how_much_of_the_folder_is_not_indexed(
    project: Path, window
):
    """The project list shows what is indexed, which is silent about the data
    sitting beside it that nothing has indexed -- the state a collection is in
    before any of this has been done."""
    unindexed = project / "loose-dataset" / "01-lamella" / "stage" / "sem"
    unindexed.mkdir(parents=True)
    (unindexed / "img.tif").write_bytes(b"i")

    window.rescan()

    said = window.metadata_panel.unindexed.text()
    assert "1 directory(ies) here hold 1 image(s)" in said
    assert "Add projects" in said


def test_a_fully_indexed_folder_says_nothing_about_unindexed_data(project: Path, window):
    window.rescan()

    assert window.metadata_panel.unindexed.text() == ""


# ---------------------------------------------------------------------------
# The dialog a long operation runs behind
# ---------------------------------------------------------------------------

def test_the_busy_dialog_does_not_process_events(window):
    """QProgressDialog.setValue() calls processEvents() whenever the dialog is
    modal, which re-enters the event loop from inside a queued-signal handler:
    the worker's "finished" can then be delivered in the middle of a progress
    update and close the dialog under the code still using it. It showed as a
    freeze on the final item."""
    from qtpy.QtWidgets import QProgressDialog

    from am_segmentation_manager.gui.widgets.busy_dialog import BusyDialog

    window._run_operation("Long thing", lambda: None, lambda result: None)
    try:
        assert isinstance(window._busy, BusyDialog)
        assert not isinstance(window._busy, QProgressDialog)
    finally:
        window._take_pending_operation()


def test_progress_arriving_after_the_work_returned_is_harmless(window):
    """Both are queued signals from the same thread, so the last report can
    land after the result. This is the shape of the freeze: the handler must
    find no dialog rather than one being taken apart."""
    window._run_operation("Long thing", lambda: None, lambda result: None)
    window._take_pending_operation()

    window._operation_progressed(9, 9)

    assert window._busy is None


def test_the_busy_dialog_reports_how_far_the_work_has_got(window):
    window._run_operation("Long thing", lambda: None, lambda result: None)
    try:
        assert window._busy.bar.maximum() == 0, "opens indeterminate"

        window._operation_progressed(3, 9)

        assert window._busy.bar.maximum() == 9
        assert window._busy.bar.value() == 3
        assert "3 of 9" in window._busy.message.text()
    finally:
        window._take_pending_operation()


def test_escape_does_not_dismiss_an_operation_that_cannot_be_stopped(window):
    """The dialog is the only thing keeping the window out of files that are
    being written."""
    window._run_operation("Long thing", lambda: None, lambda result: None)
    try:
        window._busy.reject()

        # isHidden() rather than isVisible(): a dialog shown on a window that
        # was never shown is never "visible", so isVisible() would pass here
        # whether or not Escape had dismissed it.
        assert window._busy.isHidden() is False
    finally:
        window._take_pending_operation()


def test_escape_on_a_stoppable_operation_asks_it_to_stop(window):
    from am_segmentation_manager.progress import Cancellation

    cancellation = Cancellation()
    window._run_operation(
        "Long thing", lambda: None, lambda result: None, cancellation=cancellation
    )
    try:
        window._busy.reject()

        assert cancellation.requested is True
    finally:
        window._take_pending_operation()


def test_the_collection_actions_fit_the_pane_they_sit_in(window):
    """A QPushButton clips its caption rather than eliding it, and clipping
    takes the ends off both sides at once -- so a row too wide for the pane
    reads as "nspect layou" rather than as something obviously cut off."""
    from am_segmentation_manager.gui.widgets.metadata_panel import (
        PROJECT_PANE_MIN_WIDTH,
    )

    needed = window.metadata_panel.collection_box.minimumSizeHint().width()

    assert needed <= PROJECT_PANE_MIN_WIDTH


def test_scanning_names_the_collection_the_actions_would_write_to(window, project: Path):
    assert window.metadata_panel.root == project
    assert project.name in window.metadata_panel.collection_box.title()


def test_the_operations_box_says_what_it_would_act_on(window):
    _tick_lamella(window)

    assert "1 lamella" in window.operations_panel.box.title()


def test_the_operations_box_says_whether_there_is_anything_to_convert(window):
    """Answered before the button is pressed, not by pressing it.

    The count comes from the selection, so it has to survive a refresh: this is
    the path that recomputes it, and a line wired up only at construction would
    pass a direct call to `show_predictions` and still be permanently stale.
    """
    _tick_lamella(window)
    window.refresh()

    assert window.operations_panel.predictions.text()


def test_an_operation_really_runs_off_the_interface_thread(qt_app, window, monkeypatch):
    """The seam every other test substitutes, exercised once for real.

    Asserted on the thread the work ran on: a queued connection that degraded
    to a direct call would still deliver a result, and would freeze the window
    for the minutes a compression takes.

    The failure path is redirected because the window reports it in a modal
    box, which in a test run waits for a click that never comes -- an
    unexpected failure here has to fail the test, not stop the suite.
    """
    import threading

    failures: list = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda parent, title, text, *args: failures.append(text),
    )
    results: list = []
    window._run_operation("Test", lambda: threading.current_thread().name, results.append)
    settle(qt_app, lambda: bool(results) or bool(failures))

    assert not failures
    assert results[0] != threading.current_thread().name
    # The buttons locked for the duration are usable again.
    assert window.operations_panel.convert_button.isEnabled()


def test_a_failing_operation_is_reported_and_leaves_the_window_usable(
    qt_app, window, monkeypatch
):
    reported: list = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda parent, title, text, *args: reported.append(text),
    )

    window._run_operation("Test", lambda: 1 / 0, lambda result: pytest.fail("no result"))
    settle(qt_app, lambda: bool(reported))

    assert "division by zero" in reported[0]
    assert window.operations_panel.convert_button.isEnabled()


def test_a_report_dialog_shows_its_lines_and_asks_only_when_it_can_act(qt_app):
    """The dialog every operation reports through, on both of its two jobs."""
    from qtpy.QtWidgets import QDialogButtonBox

    from am_segmentation_manager.gui.widgets.report_dialog import ReportDialog

    asking = ReportDialog(None, "Compress labels", ["headline", "a/b.tif"], "Apply")
    assert asking.text() == "headline\na/b.tif"
    assert asking.buttons.button(QDialogButtonBox.Ok).text() == "Apply"
    # Cancel is the default, so Return cannot apply an operation by accident.
    assert asking.buttons.button(QDialogButtonBox.Cancel).isDefault()
    # exec() rather than exec_(): present and undeprecated under both bindings.
    assert callable(asking.exec)

    telling = ReportDialog(None, "Inspect layout", ["headline"])
    assert telling.buttons.button(QDialogButtonBox.Ok) is None


# ---------------------------------------------------------------------------
# Plots are prepared at the size they are shown at
# ---------------------------------------------------------------------------

def test_stepping_draws_a_plot_sized_to_the_pane(qt_app, window):
    """A plot is ~180 ms to read however it is asked for, so stepping can only
    be smooth if every one is already held -- which bounds them to the pane."""
    window._set_mode("plot")
    settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())

    stepping = window.controller.cache.plot(0).plot

    assert max(stepping.shape[:2]) < 2100, "the fixture writes 1400x2100"


def test_the_figure_is_read_whole_for_the_frame_being_looked_at(qt_app, window):
    """Reducing it on read put a ceiling on what zooming in could ever show,
    which is what these are opened for."""
    window._set_mode("plot")
    settle(
        qt_app,
        lambda: window.controller.cache.cached_plot_detail(0) is not None,
    )

    assert window.controller.cache.plot_detail(0).plot.shape[:2] == (1400, 2100)


def test_swapping_resolution_repeatedly_does_not_drift(qt_app):
    """Stepping through a series while zoomed swaps copies twice a frame. Each
    swap used to re-derive where the view was looking through the scroll
    position, which is a whole number, losing a fraction of a pixel the same
    way every time -- so the picture crept out of the pane."""
    import numpy as np

    from am_segmentation_manager.gui.widgets.image_view import ImageView

    small = np.zeros((300, 400, 3), dtype=np.uint8)
    large = np.zeros((1200, 1600, 3), dtype=np.uint8)
    view = ImageView()
    view.resize(400, 300)
    view.show()
    qt_app.processEvents()
    try:
        view.set_array(small)
        view.zoom_by(1.25)
        view.zoom_by(1.25)

        def looking_at():
            rect = view._item.boundingRect()
            middle = view.mapToScene(view.viewport().rect().center())
            return (middle.x() / rect.width(), middle.y() / rect.height())

        for _ in range(3):                       # settles on the first swap
            view.set_array(large, same_picture=True)
            view.set_array(small, same_picture=True)
        settled = looking_at()

        for _ in range(20):
            view.set_array(large, same_picture=True)
            view.set_array(small, same_picture=True)

        assert looking_at() == pytest.approx(settled, abs=0.001)
    finally:
        view.close()


def test_a_change_of_resolution_keeps_the_region_on_screen(qt_app):
    """The mechanism the swap rests on, on its own: the same picture at more
    pixels is the same view at a smaller scale about a larger centre."""
    import numpy as np

    from am_segmentation_manager.gui.widgets.image_view import ImageView

    view = ImageView()
    view.resize(400, 300)
    view.show()
    qt_app.processEvents()
    try:
        view.set_array(np.zeros((300, 400, 3), dtype=np.uint8))
        view.zoom_by(1.25)
        qt_app.processEvents()
        drawn = view._item.boundingRect().width() * view.transform().m11()
        centre = view.mapToScene(view.viewport().rect().center())

        view.set_array(np.zeros((1200, 1600, 3), dtype=np.uint8), same_picture=True)

        after = view.mapToScene(view.viewport().rect().center())
        assert view._item.boundingRect().width() * view.transform().m11() == (
            pytest.approx(drawn, rel=0.01)
        )
        assert (after.x(), after.y()) == pytest.approx(
            (centre.x() * 4, centre.y() * 4), rel=0.01
        )
    finally:
        view.close()


def test_swapping_to_the_whole_figure_draws_it_at_the_size_the_zoom_asked_for(
    qt_app, window
):
    """The swap happens inside a zoom, so it has to be invisible: the same
    region, at the size that zoom asked for, drawn from more pixels."""
    window._set_mode("plot")
    settle(qt_app, lambda: window.controller.cache.cached_plot_detail(0) is not None)
    window._render_plot()
    view = window.image_panel.plot_view
    drawn = view._item.boundingRect().width() * view.transform().m11()
    pane_copy = view._item.boundingRect().width()

    view.zoom_by(1.25)

    assert view._item.boundingRect().width() > pane_copy, "swapped to the whole figure"
    assert view._item.boundingRect().width() * view.transform().m11() == pytest.approx(
        drawn * 1.25, rel=0.02
    )


def test_a_figure_of_another_size_starts_again_rather_than_inheriting_the_zoom(
    qt_app, window
):
    """Holding the view across a change of resolution is for the same figure
    read whole. A figure of another size is another picture, and carrying a
    zoom onto it would show it at a scale chosen for something else."""
    import numpy as np

    from am_segmentation_manager.gui.cache import FramePlot

    window._set_mode("plot")
    settle(qt_app, lambda: window.controller.cache.cached_plot_detail(0) is not None)
    window._render_plot()
    panel = window.image_panel
    panel.plot_view.zoom_by(1.25)
    panel.plot_view.zoom_by(1.25)
    assert not panel.plot_view.transform().isIdentity()

    panel.show_plot(FramePlot(plot=np.zeros((240, 320, 3), dtype=np.uint8)))

    # Identity is one array pixel per physical pixel, which is what a copy
    # prepared for this pane is drawn at.
    assert panel.plot_view.transform().isIdentity()


def test_a_plot_opens_centred_in_the_pane(qt_app, window):
    """Resetting the transform says nothing about where the view is looking,
    and the scene is deliberately larger than the picture -- so on its own it
    left the figure off to one side with a band of empty pane beside it."""
    window._set_mode("plot")
    settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())
    view = window.image_panel.plot_view

    picture = view.mapFromScene(view._item.boundingRect().center())
    pane = view.viewport().rect().center()

    assert (picture.x(), picture.y()) == pytest.approx((pane.x(), pane.y()), abs=2)


def test_a_plot_prepared_for_another_pane_still_fills_this_one(qt_app, window):
    """The copy is reduced to a box measured in physical pixels, so trusting
    the screen's ratio at drawing time leaves it at half or twice the size
    whenever the two disagree -- which is what a band of empty pane above and
    below the figure was."""
    import numpy as np

    from am_segmentation_manager.gui.cache import FramePlot

    window._set_mode("plot")
    settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())
    panel = window.image_panel
    viewport = panel.plot_view.viewport().rect()

    for ratio in (1, 2, 3):
        wide = round(viewport.width() * ratio)
        tall = round(viewport.height() * ratio)
        panel.show_plot(FramePlot(plot=np.zeros((tall, wide, 3), dtype=np.uint8)))

        item = panel.plot_view._item.boundingRect()
        drawn = item.width() * panel.plot_view.transform().m11()
        assert drawn == pytest.approx(viewport.width(), rel=0.02), f"ratio {ratio}"


def test_stepping_to_another_plot_does_not_move_the_picture(qt_app, window):
    """Every plot in a lamella is the same size, so a step should not disturb
    the view at all -- it used to refit, then shift again as the whole copy
    landed."""
    window._set_mode("plot")
    settle(qt_app, lambda: window.controller.cache.cached_plot_detail(0) is not None)
    window._render_plot()
    view = window.image_panel.plot_view
    before = view.transform().m11(), view._item.boundingRect().width()

    window._step_frame(1)
    settle(qt_app, lambda: window.controller.frame_position == 1)

    assert (view.transform().m11(), view._item.boundingRect().width()) == before


def test_stepping_through_plots_finds_them_ready(qt_app, window):
    """The behaviour that was stuttering: holding an arrow key through plots."""
    window._set_mode("plot")
    settle(
        qt_app,
        lambda: all(
            window.controller.cache.cached_plot(i) is not None
            for i in range(len(window.controller.frames))
        ),
    )

    missing = 0
    for _ in range(len(window.controller.frames) - 1):
        window._step_frame(1)
        if window.controller.cache.cached_plot(window.controller.frame_position) is None:
            missing += 1

    assert missing == 0


def test_changing_a_state_updates_the_status_chips(window, monkeypatch):
    """The chips exist to report curation progress, so the moment progress is
    made is exactly when they must not go stale.

    Regression: apply_state ended with refresh() rather than
    refresh(reload_lists=True), so the timeline updated and the chips did not.
    """
    controller = window.controller
    position = controller.lamella_position
    assert controller.lamella_statuses()[position].of("adaptive", "SEM") == (
        "not started"
    )

    monkeypatch.setattr(window, "_confirm_bulk", lambda state: True)
    controller.set_editing_enabled(True)
    controller.set_target("SEM")
    controller.set_lamella_checked(position, True)
    window.apply_state("corrected")

    assert controller.lamella_statuses()[position].of("adaptive", "SEM") == "done"
    # The widget, not just the controller, reflects it.
    item = window.project_panel.lamella_list.item(position)
    assert "SEM: done" in item.toolTip()


# ---------------------------------------------------------------------------
# The metadata tab is read before it is written
# ---------------------------------------------------------------------------

@pytest.fixture
def metadata_panel(qt_app, tmp_path: Path):
    """A metadata panel over two projects, built without scanning for them.

    The YAML files are written even though the projects are built by hand: a
    save goes to disk and re-reads what landed there, so a fixture that only
    existed in memory would lose half its metadata the first time one happened.
    """
    experiment = {"metadata": {"organisation": "Rosalind Franklin Institute"}}

    def project(name: str, own: dict) -> Project:
        # Nested in a session folder, as Autolamella projects are: several
        # experiments from one sitting share a directory.
        path = tmp_path / "20260728_CB_AM_bitmap" / name
        path.mkdir(parents=True)
        (path / "experiment.yaml").write_text(
            "metadata:\n  organisation: Rosalind Franklin Institute\n"
        )
        (path / "AM_metadata.yaml").write_text(
            "metadata:\n" + "".join(f"  {key}: {value}\n" for key, value in own.items())
        )
        return Project(
            name=name,
            path=path,
            relative_path=f"20260728_CB_AM_bitmap/{name}",
            experiment_metadata=build_experiment_metadata(
                experiment,
                {"metadata": own},
                experiment_path=path / "experiment.yaml",
                am_metadata_path=path / "AM_metadata.yaml",
            ),
        )

    controller = MetadataController()
    controller.set_projects([
        project("described", {"sample": "HeLa", "user": "Casper"}),
        project("outstanding", {"sample": "TODO"}),
    ])
    panel = MetadataPanel(controller)
    panel.set_projects(controller.project_summaries())
    yield panel
    panel.close()


def test_a_locked_field_still_shows_its_value(metadata_panel):
    """The tab opens locked and is mostly used for reading.

    That the value is also drawn *legibly* while locked is a property of the
    theme, and is asserted on the pixels in ``test_gui_theme.py``.
    """
    assert editor_text(metadata_panel.editors["sample"]) == "HeLa"
    assert all(not editor.isEnabled() for editor in metadata_panel.editors.values())


def test_every_field_says_what_is_expected_in_it(metadata_panel):
    """Hovering either the name or the box has to explain the field."""
    for entry in FIELDS:
        assert metadata_panel.captions[entry.name].toolTip() == entry.guidance
        assert metadata_panel.editors[entry.name].toolTip() == entry.guidance


def test_a_field_holding_several_says_so_without_being_hovered(metadata_panel):
    """A hover explains a control to someone who already suspects it exists.
    Which fields take more than one answer has to be readable at a glance, or
    the ones that do look exactly like the ones that do not."""
    plural = {"(s)", "media"}
    for entry in FIELDS:
        caption = metadata_panel.captions[entry.name].text()
        reads_plural = any(caption.endswith(ending) for ending in plural)
        assert reads_plural == entry.multi, caption
        says_so = "Holds several" in metadata_panel.editors[entry.name].toolTip()
        assert says_so == entry.multi, entry.name


def test_a_value_is_annotated_with_the_file_it_came_from(metadata_panel):
    assert metadata_panel.notes["sample"].text() == "[AM_metadata.yaml]"
    assert metadata_panel.notes["organisation"].text() == "[experiment.yaml]"
    assert metadata_panel.notes["sample"].styleSheet() == BADGE_STYLES["info"]


def _current_project(panel) -> int | None:
    """Return the project number the tree's cursor is on."""
    item = panel.projects.currentItem()
    return panel.projects.project_of(item) if item is not None else None


def test_stepping_moves_to_the_next_project_and_stops_at_the_ends(metadata_panel):
    """What Up/Down are bound to on this tab. It steps through *projects*: the
    folders are rows too, and stepping onto one would describe nothing."""
    metadata_panel.step_project(1)
    assert _current_project(metadata_panel) == 1
    assert metadata_panel.controller.position == 1

    # The last project: stepping past it stays there rather than wrapping.
    metadata_panel.step_project(1)
    assert _current_project(metadata_panel) == 1

    metadata_panel.step_project(-1)
    assert _current_project(metadata_panel) == 0
    assert metadata_panel.controller.position == 0


def test_an_outstanding_field_is_marked_rather_than_dimmed(metadata_panel):
    """A placeholder is the state that most needs seeing; it used to be drawn
    in the dimmest colour on the panel, in the same grey as provenance."""
    metadata_panel.projects.setCurrentItem(metadata_panel.projects.project_item(1))

    note = metadata_panel.notes["sample"]
    assert note.text() == "[TODO]"
    assert note.styleSheet() == BADGE_STYLES["warning"]
    assert editor_text(metadata_panel.editors["sample"]) == ""


def test_repeating_fields_offer_the_values_already_in_use(metadata_panel):
    box = metadata_panel.editors["sample"].entry

    assert box.isEditable()
    # A blank first entry, so "unset" stays reachable from the list.
    assert [box.itemText(row) for row in range(box.count())] == ["", "HeLa"]
    assert metadata_panel.editors["description"].__class__.__name__ == "QLineEdit"


def test_a_single_valued_field_keeps_its_one_box(metadata_panel):
    """Only a field that holds several gets chips. A grid type is one thing,
    and offering to add a second would say it might not be."""
    from qtpy.QtWidgets import QComboBox

    assert isinstance(metadata_panel.editors["grid_type"], QComboBox)
    assert isinstance(metadata_panel.editors["lamella_type"], QComboBox)


def test_a_value_not_in_the_list_can_still_be_typed(metadata_panel):
    """The list is a shortcut, not a vocabulary -- the first project to use a
    sample has nothing to pick. And it is added to what is there rather than
    replacing it, which is the whole reason the field holds several."""
    metadata_panel.enable.setChecked(True)
    editor = metadata_panel.editors["sample"]
    editor.entry.setCurrentText("Chlamydomonas")
    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["HeLa", "Chlamydomonas"]
    assert metadata_panel._edits() == {"sample": "HeLa; Chlamydomonas"}
    # Typing must not grow the list, or a typo would be offered back later.
    assert editor.entry.count() == 2


def test_a_value_typed_and_not_confirmed_is_still_saved(metadata_panel):
    """Regression: the box held what you typed until Return or focus left it,
    so typing a value and reaching straight for Save discarded it silently --
    which is every first attempt at using the field."""
    metadata_panel.enable.setChecked(True)
    metadata_panel.editors["organisation"].entry.setCurrentText("Diamond")

    assert metadata_panel._edits() == {
        "organisation": "Rosalind Franklin Institute; Diamond"
    }


def test_leaving_the_box_turns_what_is_in_it_into_a_value(metadata_panel):
    """So that what was typed is visibly a value before Save is pressed,
    rather than only becoming one when Save reads the editor."""
    metadata_panel.enable.setChecked(True)
    editor = metadata_panel.editors["project"]
    editor.entry.setCurrentText("Adaptive milling")

    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["Adaptive milling"]
    assert editor.entry.currentText() == ""


def test_the_usual_answers_are_offered_before_anyone_has_given_one(metadata_panel):
    """A default is offered, where a vocabulary is required: plunge and high
    pressure cover almost everything, and jet freezing still has to be typable."""
    from am_segmentation_manager.metadata import FREEZING_METHODS

    box = metadata_panel.editors["freezing_method"]
    offered = [box.itemText(row) for row in range(box.count())]

    assert offered == ["", *FREEZING_METHODS]
    assert box.isEditable(), "an unusual method is typed, not refused"


def test_a_chip_can_be_taken_off_again(metadata_panel):
    """And not while editing is locked: the cross is a control on a value, and
    the lock covers every other way of changing one."""
    editor = metadata_panel.editors["sample"]
    assert editor.values() == ["HeLa"]
    editor._chip_layout.itemAt(0).widget().click()
    assert editor.values() == ["HeLa"], "locked, so the cross does nothing"

    metadata_panel.enable.setChecked(True)

    editor._chip_layout.itemAt(0).widget().click()

    assert editor.values() == []
    assert metadata_panel._edits() == {"sample": None}


def test_the_wider_project_holds_several_like_the_rest(metadata_panel):
    """It names the project several datasets belong to, so it repeats across
    them -- which is what the value lists are for. It used to carry a button
    copying the directory name in, which is a different fact entirely."""
    editor = metadata_panel.editors["project"]

    assert isinstance(editor, ChipEditor)
    assert editor.entry.isEditable()
    assert not hasattr(metadata_panel, "folder_button")


def test_the_list_says_which_projects_still_need_describing(metadata_panel):
    """Without this you have to click every project to find the empty ones."""
    described = metadata_panel.projects.project_item(0)
    outstanding = metadata_panel.projects.project_item(1)

    total = len(DESCRIPTIVE_FIELDS)
    assert f"3 of {total} fields set" in described.toolTip(0)
    assert "missing: description, project, " in described.toolTip(0)
    assert "grid_type, " in described.toolTip(0)
    # Only the organisation, from experiment.yaml: its sample is a placeholder.
    assert f"1 of {total} fields set" in outstanding.toolTip(0)
    assert "sample" in outstanding.toolTip(0)
    assert not described.icon(0).isNull()


def test_the_projects_are_grouped_under_their_session_folder(metadata_panel):
    """The same tree the Curate tab lists them in. Describing a collection is
    done a session at a time, and an Autolamella project is named for the
    minute it started."""
    tree = metadata_panel.projects
    folders = [
        tree.topLevelItem(row).text(0)
        for row in range(tree.topLevelItemCount())
        if tree.project_of(tree.topLevelItem(row)) is None
    ]

    assert folders == ["20260728_CB_AM_bitmap"]
    assert len(tree.project_items()) == 2


def test_add_data_offers_expand_and_collapse(metadata_panel):
    """Describing a collection means working down one session at a time, which
    needs its folder opened; thirty left open is the other half of that."""
    tree = metadata_panel.projects
    tree.collapseAll()
    assert not any(
        tree.topLevelItem(row).isExpanded()
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).childCount()
    )

    from qtpy.QtWidgets import QPushButton

    buttons = metadata_panel.findChildren(QPushButton)
    expand = next(b for b in buttons if b.text() == "Expand")
    collapse = next(b for b in buttons if b.text() == "Collapse")
    expand.click()

    assert all(
        tree.topLevelItem(row).isExpanded()
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).childCount()
    )
    collapse.click()
    assert not any(
        tree.topLevelItem(row).isExpanded()
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).childCount()
    )


def test_the_add_data_tree_has_no_tick_boxes(metadata_panel):
    """This tab describes one project at a time, so a box would offer a set it
    cannot act on."""
    from qtpy.QtCore import Qt as _Qt

    for item in metadata_panel.projects.project_items():
        assert not item.flags() & _Qt.ItemIsUserCheckable


def test_a_row_carries_its_own_name_and_not_the_folder_again(metadata_panel):
    """The folder above has said it once, and repeating it on every row is
    what made the flat list a column of identical prefixes."""
    item = metadata_panel.projects.project_item(0)

    assert "20260728_CB_AM_bitmap" not in item.text(0)
    assert "20260728_CB_AM_bitmap" in item.toolTip(0)


def test_a_long_row_is_elided_rather_than_scrolled(metadata_panel):
    """These paths differ in the timestamp they end with. With a horizontal
    scrollbar the view scrolls instead of eliding, and a run of projects from
    one session is cut off at the edge into identical-looking rows."""
    item = metadata_panel.projects.project_item(0)

    assert metadata_panel.projects.textElideMode() == Qt.ElideMiddle
    assert metadata_panel.projects.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    # The folder is a row of its own now, so the tooltip carries the whole
    # relative path where the row carries only the name under it.
    assert item.text(0) in item.toolTip(0)


def test_saving_metadata_starts_the_record_it_writes_into(metadata_panel):
    """The third door a change comes through, and the one that never reaches
    the window: the panel calls its controller directly."""
    from am_segmentation_manager.history import history_file_path, read_log

    project = metadata_panel.controller.scope[0]
    assert not history_file_path(project.path).is_file(), "nothing recorded yet"
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")

    metadata_panel._save()

    kinds = [event.kind for event in read_log(project.path).events]
    assert "log-created" in kinds, "the write started the log"
    assert "metadata-changed" in kinds, kinds


def test_saving_updates_the_mark_in_the_list(metadata_panel):
    """Saving is exactly when a project stops being incomplete, so leaving the
    list describing the state before it is how a user learns to distrust it."""
    total = len(DESCRIPTIVE_FIELDS)
    assert f"3 of {total} fields set" in (
        metadata_panel.projects.project_item(0).toolTip(0)
    )
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")
    metadata_panel._save()

    assert f"4 of {total} fields set" in (
        metadata_panel.projects.project_item(0).toolTip(0)
    )


def test_right_clicking_a_project_asks_for_its_path(metadata_panel):
    """Reusing the curate tab's menu: the row is emitted, and resolving the
    path stays the controller's job."""
    rows: list[int] = []
    metadata_panel.project_path_requested.connect(rows.append)
    metadata_panel.project_path_requested.emit(1)

    assert rows == [1]
    assert metadata_panel.controller.project_path(1).is_absolute()


# ---------------------------------------------------------------------------
# Choosing what to predict on
# ---------------------------------------------------------------------------
#
# Developed on a Mac, which has MPS and a CPU. A workstation has several CUDA
# devices and no MPS, so nothing here may assume the machine it runs on.

def test_the_device_box_offers_what_every_machine_has(window):
    """Before torch is loaded, the only honest answers are "the best one" and
    "the one everything has". Asking torch costs an import, which a dropdown
    may not spend."""
    offered = [
        window.operations_panel.device_box.itemText(index)
        for index in range(window.operations_panel.device_box.count())
    ]

    assert offered == ["auto", "cpu"]
    assert window.operations_panel.device() == "auto"


def test_the_device_box_takes_the_machines_own_devices(window):
    """A CUDA workstation, as torch would report it."""
    window.operations_panel.show_devices(("cuda:0", "cuda:1", "cpu"))

    offered = [
        window.operations_panel.device_box.itemText(index)
        for index in range(window.operations_panel.device_box.count())
    ]

    assert offered == ["auto", "cuda:0", "cuda:1", "cpu"]


def test_a_device_this_package_has_never_seen_can_be_typed(window):
    """resolve_device passes a named device through unexamined, so support for
    a new accelerator is not something anyone has to wait for."""
    window.operations_panel.device_box.setCurrentText("xpu:0")

    assert window.operations_panel.device() == "xpu:0"


def test_filling_the_box_keeps_what_was_chosen(window):
    """It is filled when the box is opened, which must not undo a choice made
    before the machine's devices were known."""
    window.operations_panel.set_device("cuda:1")

    window.operations_panel.show_devices(("cuda:0", "cuda:1", "cpu"))

    assert window.operations_panel.device() == "cuda:1"


def test_the_chosen_device_reaches_the_run(window, monkeypatch, tmp_path: Path):
    """Every prediction from the window used to be 'auto'."""
    seen: list[str] = []
    monkeypatch.setattr(
        window.operations_controller, "predict",
        lambda records, weights, device="auto", **kwargs: seen.append(device),
    )
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems", lambda: []
    )
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done, cancellation=None: work(),
    )
    window.controller.editing_enabled = True
    window.operations_panel.set_device("cpu")
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"x")

    window._predict(str(weights))

    assert seen == ["cpu"]
    assert window.settings.value("predict_device", "", type=str) == "cpu"


def test_predicting_stays_off_where_it_was_never_available(window):
    """Whether the inference stack is installed is asked once. Unlocking the
    buttons after an operation must not answer it again by accident."""
    window.operations_panel.set_prediction_available(False)

    window.operations_panel.set_busy(True)
    window.operations_panel.set_busy(False)

    assert not window.operations_panel.predict_button.isEnabled()
    assert not window.operations_panel.device_box.isEnabled()


# ---------------------------------------------------------------------------
# The chip editor, as a widget
# ---------------------------------------------------------------------------

def test_chips_wrap_onto_another_row_rather_than_off_the_edge(qt_app):
    """Values that do not fit used to be laid out below the bottom edge, where
    they are addressable and invisible -- so a field would report two
    organisations while showing one."""
    editor = ChipEditor()
    editor._chips.resize(90, 0)
    editor.set_values(["alpha"])
    one_row = editor._chips.height()

    editor.set_values(["alpha", "beta", "gamma", "delta"])

    assert one_row > 0
    assert editor._chips.height() > one_row
    assert editor._chip_layout.count() == 4


def test_a_chip_is_tall_enough_to_be_seen_as_soon_as_it_exists(qt_app):
    """Regression, and the worst kind: adding a widget to a layout reparents
    it, reparenting hides it, and a hidden widget measures as nothing -- so the
    row of chips was given no height. The values were held, addressable and
    saved, and simply not on screen, which reads as a field that refuses to
    take a value and as one that loses the value already in it."""
    editor = ChipEditor()
    editor.show()
    editor._chips.resize(300, 0)

    editor.set_values(["RFI"])

    assert editor._chips.isVisible()
    assert editor._chips.height() > 0
    assert editor._chip_layout.heightForWidth(300) > 0


def test_chips_come_back_after_a_project_that_had_none(qt_app):
    """What stepping through a collection does to one editor. A project with
    nothing in the field hides the chips, and the next project has to be able
    to show them again -- which is the case that failed, because a chip cannot
    be shown into a parent that is still hidden."""
    editor = ChipEditor()
    editor.show()
    editor._chips.resize(300, 0)

    for values in (["RFI"], [], ["RFI", "Diamond"], [], ["EMBL"]):
        editor.set_values(values)

        assert (editor._chips.height() > 0) == bool(values), values
        assert editor.values() == values


def test_a_chip_area_takes_no_minimum_width(qt_app):
    """A minimum is a floor. Asked for the widest value in it, the pane could
    never be narrower than the longest organisation anybody had typed."""
    editor = ChipEditor()
    editor.set_values(["a value considerably longer than the pane it sits in"])

    assert editor._chip_layout.minimumSize().width() == 0
    assert editor._chip_layout.sizeHint().width() > 0


def test_typing_several_values_at_once_makes_several_chips(qt_app):
    """Pasting a separated line is how anyone would try it first."""
    editor = ChipEditor()
    editor.entry.setCurrentText("infected; differentiated")
    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["infected", "differentiated"]
    assert editor.entry.currentText() == "", "the box empties into the chips"


def test_a_value_already_there_is_not_added_twice(qt_app):
    editor = ChipEditor()
    editor.set_values(["HeLa"])
    reported: list[int] = []
    editor.changed.connect(lambda: reported.append(1))

    editor.entry.setCurrentText("HeLa")
    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["HeLa"]
    assert reported == [], "nothing changed, so nothing is reported"


def test_setting_the_values_is_a_redraw_rather_than_an_edit(qt_app):
    """The panel fills every editor whenever it refreshes, and a refresh that
    reported edits would have the window saving what it had just read."""
    editor = ChipEditor()
    reported: list[int] = []
    editor.changed.connect(lambda: reported.append(1))

    editor.set_values(["RFI", "Diamond"])

    assert editor.values() == ["RFI", "Diamond"]
    assert reported == []


# ---------------------------------------------------------------------------
# The legend
#
# A mark and its explanation must be drawn by the same code, or they drift.
# ---------------------------------------------------------------------------

def _legend_images(help_text: str) -> list[bytes]:
    """Return the pictures embedded in a help list, decoded."""
    import base64
    import re

    return [
        base64.b64decode(match)
        for match in re.findall(r'<img src="data:image/png;base64,([^"]+)">', help_text)
    ]


def test_the_legend_shows_the_marks_rather_than_describing_them(window, qt_app):
    """Every legend row carries a picture drawn by whatever draws the real
    mark. A described colour and a drawn one drift apart; the same pixmap
    cannot."""
    from qtpy.QtGui import QImage

    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    images = _legend_images(window._keys.help_html(TAB_CURATE))

    assert len(images) >= 10
    for encoded in images:
        picture = QImage.fromData(encoded, "PNG")
        assert not picture.isNull()
        assert picture.width() > 0 and picture.height() > 0


def test_the_chip_legend_is_the_chip_the_rows_carry(window, qt_app):
    """The point of the legend: the fill named here is the fill drawn there."""
    from qtpy.QtGui import QImage

    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    shown = {
        QImage.fromData(encoded, "PNG").pixelColor(1, 1).name()
        for encoded in _legend_images(window._keys.help_html(TAB_CURATE))
    }

    assert {colour.name() for colour in CELL_COLOURS.values()} <= shown


def test_the_legend_says_what_the_tab_actually_draws(window):
    """The white band means frames picked out while curating and unsaved edits
    while proofreading. One mark, two meanings: listing both on both tabs would
    make it mean neither."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE, TAB_LABEL

    curate = window._keys.help_html(TAB_CURATE)
    label = window._keys.help_html(TAB_LABEL)

    assert "Picked out for a state change" in curate
    assert "unsaved edits" not in curate
    assert "unsaved edits" in label
    assert "Picked out for a state change" not in label


def test_a_tab_with_no_marks_gets_no_legend(window):
    """Export draws neither a chip nor the bar, and a heading with nothing
    under it is a question raised and not answered."""
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_ROW_MARK,
        GROUP_STRIP_MARK,
        TAB_EXPORT,
    )

    help_text = window._keys.help_html(TAB_EXPORT)

    assert GROUP_ROW_MARK not in help_text
    assert GROUP_STRIP_MARK not in help_text


def test_the_legend_comes_after_the_keys_and_before_this_list(window):
    """A reader looking for a key should not have to read past the pictures,
    and the key that opened the list is the one they have already found."""
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_HELP,
        GROUP_ROW_MARK,
        GROUP_STRIP_MARK,
        GROUP_VIEW,
        TAB_CURATE,
    )

    help_text = window._keys.help_html(TAB_CURATE)
    places = [
        help_text.index(group)
        for group in (GROUP_VIEW, GROUP_ROW_MARK, GROUP_STRIP_MARK, GROUP_HELP)
    ]

    assert places == sorted(places)


def test_the_shortcut_list_is_capped_to_the_screen(window, qt_app):
    """The list runs past 800 pixels on two tabs now the legend is in it, and a
    dialog merely sized past the screen has no scrollbar the user can reach
    either. Built directly rather than through F1, which does nothing while any
    modal dialog is up."""
    from am_segmentation_manager.gui.widgets.shortcuts import SCREEN_MARGIN, TAB_LABEL, ShortcutHelp

    dialog = ShortcutHelp(window)
    dialog.show_for("Label", window._keys.help_html(TAB_LABEL))
    room = qt_app.primaryScreen().availableGeometry()

    assert dialog.height() <= room.height() - SCREEN_MARGIN
    assert dialog.width() <= room.width() - SCREEN_MARGIN
    # And the rows past the cap are still reachable rather than cut off.
    assert dialog._text.sizeHint().height() > dialog.height()
    assert dialog._scroll.verticalScrollBarPolicy() != Qt.ScrollBarAlwaysOff


def test_the_timeline_legend_describes_nothing_the_timeline_omits(qt_app):
    """`MISMATCH` was commented as a wash over frames disagreeing with their
    label, and `_paint` has never drawn one. A legend row for it would have
    been a mark the reader could look for for ever."""
    from am_segmentation_manager.gui.widgets.frame_strip import timeline_legend

    meanings = " ".join(text for _, text in timeline_legend("picked out")).lower()

    assert "amber" not in meanings
    assert "disagree" not in meanings


def test_a_mark_survives_the_round_trip_into_the_list(qt_app):
    """`QBuffer(QByteArray())` lets the array die under the buffer and
    segfaults PySide6, so the bytes have to be held in a name."""
    import base64
    import re

    from qtpy.QtGui import QColor, QImage, QPixmap

    from am_segmentation_manager.gui.widgets.common import mark_image

    pixmap = QPixmap(25, 16)
    pixmap.fill(QColor("#12604f"))
    encoded = re.search(r'base64,([^"]+)"', mark_image(pixmap)).group(1)
    restored = QImage.fromData(base64.b64decode(encoded), "PNG")

    assert restored.size() == pixmap.size()
    assert restored.pixelColor(12, 8) == QColor("#12604f")


# ---------------------------------------------------------------------------
# Describing a folder of projects at once
# ---------------------------------------------------------------------------

def _choose_folder(panel, folder: str) -> None:
    """Move the tree onto a folder row, as clicking one would."""
    tree = panel.projects
    row = next(
        tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).text(0) == folder
    )
    tree.setCurrentItem(row)


def test_choosing_a_folder_describes_its_projects_together(metadata_panel):
    """The point of grouping them: a session's projects share their sample,
    their grid and their operator."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")

    assert len(metadata_panel.controller.scope) == 2
    assert metadata_panel.controller.scope_folder == "20260728_CB_AM_bitmap"


def test_a_field_the_projects_disagree_on_shows_mixed(metadata_panel):
    from am_segmentation_manager.gui.metadata_controller import MIXED_MARKER

    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")

    assert editor_text(metadata_panel.editors["sample"]) == MIXED_MARKER
    assert metadata_panel.notes["sample"].text() == "[mixed]"


def test_a_mixed_field_left_alone_is_not_written(metadata_panel):
    """The whole rule. The marker is what the box holds *and* what the value
    resolves to, so an untouched field compares equal and never reaches a
    file -- accepted knowing it means a field can be edited without being
    written."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")

    assert metadata_panel._edits() == {}


def test_typing_over_mixed_writes_it_to_every_project(metadata_panel):
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    set_editor_text(metadata_panel.editors["sample"], "HeLa")

    assert metadata_panel._edits() == {"sample": "HeLa"}

    metadata_panel.enable.setChecked(True)
    metadata_panel._save()

    for project in metadata_panel.controller.scope:
        assert "HeLa" in (project.path / "AM_metadata.yaml").read_text()


def test_the_summary_counts_the_folder_rather_than_reading_it_out(metadata_panel):
    """A dozen project names do not fit and would not help."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    metadata_panel.refresh()

    assert metadata_panel.summary.text() == (
        "20260728_CB_AM_bitmap: 2 projects, 0 fully described"
    )


def test_clearing_a_field_across_a_folder_is_confirmed_first(metadata_panel, monkeypatch):
    """A field the projects disagree on shows as a chip with a cross on it,
    which looks like tidying away a placeholder rather than like emptying that
    field in every project under the folder."""
    from qtpy.QtWidgets import QMessageBox

    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["sample"], "")
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.No)[1],
    )

    metadata_panel._save()

    assert asked and "all 2 projects" in asked[0]
    assert metadata_panel.status.text() == "Not saved"


def test_describing_one_project_needs_no_such_confirmation(metadata_panel, monkeypatch):
    """The edit is visible in the box, on the project on screen."""
    from qtpy.QtWidgets import QMessageBox

    metadata_panel.enable.setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.Yes)[1],
    )
    set_editor_text(metadata_panel.editors["user"], "")

    metadata_panel._save()

    assert not asked


def test_writing_a_value_across_a_folder_is_not_confirmed(metadata_panel, monkeypatch):
    """That is what describing them together is for."""
    from qtpy.QtWidgets import QMessageBox

    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    metadata_panel.enable.setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.Yes)[1],
    )
    set_editor_text(metadata_panel.editors["sample"], "HeLa")

    metadata_panel._save()

    assert not asked
    for project in metadata_panel.controller.scope:
        assert "HeLa" in (project.path / "AM_metadata.yaml").read_text()


# ---------------------------------------------------------------------------
# The minimum/maximum range
# ---------------------------------------------------------------------------

def test_the_shown_range_says_so_on_screen(window):
    """The one control that makes the picture differ from the file. A reviewer
    who has forgotten it is on is deciding about an image no export contains."""
    panel = window.image_panel

    panel.intensity_slider.set_range(40, 200)

    assert "shown 40-200" in panel.badges.text()
    assert BADGE_STYLES["warning"] in panel.badges.text()


def test_the_whole_range_says_nothing(window):
    """A line that is always there is a line nobody reads."""
    window.image_panel.reset_intensity_range()

    assert "shown" not in window.image_panel.badges.text()


def test_the_two_limits_cannot_be_dragged_past_each_other(window):
    """A picture that turned into a threshold would read as a fault rather than
    as a setting, so the control refuses instead."""
    panel = window.image_panel
    slider = panel.intensity_slider
    slider.resize(200, 20)
    slider.set_range(0, 100)

    # Dragging the low handle past the high one, which is the gesture that
    # would produce a threshold if the control let it.
    slider._active = "low"
    slider._drag_to(slider._x_of(150))

    low, high = slider.values()
    assert low < high
    assert panel._intensity_range == (low, high)


def test_a_limit_returns_to_its_end_on_a_double_click(window):
    """Reaching an end by dragging is a thing nobody quite manages, which for
    this control means being left slightly off it without noticing."""
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    panel = window.image_panel
    slider = panel.intensity_slider
    slider.resize(200, 20)
    slider.set_range(70, 200)
    at = slider._x_of(70)
    # QPointF, not QPoint: the integer overload is deprecated in Qt 6 and warns.
    event = QMouseEvent(
        QEvent.MouseButtonDblClick, QPointF(at, 10), QPointF(at, 10),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    )

    slider.mouseDoubleClickEvent(event)

    assert slider.values() == (0, 200), "only the handle clicked goes back"


def test_the_range_opens_on_everything(window):
    """A frame opens showing exactly what was recorded."""
    from am_segmentation_manager.display import DEFAULT_INTENSITY_RANGE

    assert window.image_panel._intensity_range == DEFAULT_INTENSITY_RANGE


def test_the_range_is_kept_when_the_frame_changes(window):
    """Comparing one frame with the next is what it is for, and a setting lost
    on every step could not do that."""
    panel = window.image_panel
    panel.intensity_slider.set_range(30, 210)

    window._step_frame(1)

    assert panel._intensity_range == (30, 210)
    assert "shown 30-210" in panel.badges.text()


def test_moving_a_limit_redraws_without_rereading(window, monkeypatch):
    """The whole reason it is applied after the cache: a slider movement must
    not invalidate a frame and send the loader back to re-read the file."""
    from am_segmentation_manager.gui import cache as cache_module

    reads: list = []
    monkeypatch.setattr(
        cache_module, "prepare_display_image",
        lambda *args, **kwargs: reads.append(args) or None,
    )

    window.image_panel.intensity_slider.set_range(40, 255)

    assert reads == []


def test_the_overlay_is_composed_from_the_ranged_image(window, monkeypatch):
    """The two panes must not disagree about how the same acquisition looks, so
    the transform runs once and the label is drawn over its result -- not over
    the array as it came out of the cache."""
    from am_segmentation_manager.gui.widgets import image_panel as module

    panel = window.image_panel
    shown: list = []
    composed: list = []
    monkeypatch.setattr(
        module, "apply_intensity_range",
        lambda image, low, high: shown.append(image) or image,
    )
    monkeypatch.setattr(
        module, "compose_overlay",
        lambda image, *args, **kwargs: composed.append(image) or image,
    )

    panel.intensity_slider.set_range(40, 255)

    assert composed, "the frame under test carries a label to draw"
    assert composed[0] is shown[0], "composed from what the control produced"


def test_a_caption_is_never_a_flow_item_of_its_own(qt_app):
    """A wrapping row may not put a caption on one line and its control on the next.

    The failure this stops is not a control going missing but a row going
    unreadable: "Minimum / maximum" ended a line with its slider starting the
    next one, where it read as a heading for whatever followed. Asserted on
    the structure rather than on a width, because which pairs get split is
    whatever the pane happens to be -- it looked right on the monitor it was
    built on and wrong on the one beside it.
    """
    from qtpy.QtWidgets import QLabel

    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    # Held in a local: a PySide6 wrapper lives only as long as the wrapper that
    # vended it, so `ImagePanel(config).controls` hands back an object already
    # deleted. `docs/display.md` § The binding.
    panel = ImagePanel(config)
    try:
        layout = panel.controls.layout()
        loose = [
            layout.itemAt(index).widget().text()
            for index in range(layout.count())
            if isinstance(layout.itemAt(index).widget(), QLabel)
        ]

        assert not loose, f"captions that can wrap away from their control: {loose}"
    finally:
        destroy(panel)


def test_the_view_controls_are_one_block_on_the_smallest_screen_supported(
    qt_app, settings, project
):
    """Paired into two lines, they fit the centre pane rather than wrapping.

    The pane is what is left of the window after the project list and the
    state panel, so these controls have far less width than the tab does.
    Loose, they needed 1441px against the 922 a laptop leaves, and wrapped
    into an emergent shape that moved with every window size. Paired, the
    block is the same on every screen it is asked to open on, which is what
    makes it readable rather than merely reachable.
    """
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.tabs.setCurrentWidget(window.curate)
        heights = []
        for width in (SMALLEST_SUPPORTED_SCREEN[0], 1920):
            window.resize(width, 900)
            window.show()
            qt_app.processEvents()
            window.grab()
            heights.append(window.image_panel.controls.height())

        assert heights[0] == heights[1], (
            f"the row wraps differently at {SMALLEST_SUPPORTED_SCREEN[0]} "
            f"than at 1920: {heights}"
        )
    finally:
        destroy(window)


def test_a_control_row_still_wraps_where_it_has_to(qt_app):
    """Pairing halves the width it needs; it does not make wrapping optional.

    Answered from the width the row actually has: asked at one width and
    handed another, the second line lands below the bottom edge, present and
    invisible.
    """
    from am_segmentation_manager.gui.widgets.controls import FlowLayout
    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    panel = ImagePanel(config)
    layout = panel.controls.layout()
    assert isinstance(layout, FlowLayout)
    # A minimum with no width is what lets it wrap; a floor would force the
    # pane wider than the longest row instead.
    assert layout.minimumSize().width() == 0

    panel.resize(1900, 400)
    panel.show()
    qt_app.processEvents()
    wide = panel.controls.height()

    panel.resize(500, 400)
    qt_app.processEvents()

    assert panel.controls.height() > wide, "narrower means more lines"


def test_hiding_the_image_leaves_a_ground_the_size_of_the_frame(qt_app):
    """The proofreading canvas hides an item where Curate composes an array.

    Two mechanisms, one rule: with the acquisition gone something has to say
    where the frame ends, or a label a row of pixels short along an edge is
    indistinguishable from one that runs to it. The canvas keeps the label as
    its own layer so it can be painted into, and that layer's background is
    transparent -- so the ground has to be an item under it rather than a fill
    inside it.
    """
    import numpy as np

    from am_segmentation_manager.display import HIDDEN_IMAGE_GREY
    from am_segmentation_manager.gui.widgets.image_view import ImageView

    view = ImageView()
    try:
        view.set_array(np.zeros((40, 60), dtype=np.uint8))
        backdrop = view._backdrop

        assert not backdrop.isVisible(), "nothing to stand in for while it shows"

        view.set_image_shown(False)

        assert backdrop.isVisible()
        assert backdrop.rect() == view._item.boundingRect(), (
            "the ground is the frame, so its edge is the frame's edge"
        )
        assert backdrop.brush().color().getRgb()[:3] == HIDDEN_IMAGE_GREY
        assert backdrop.zValue() < view._item.zValue(), "under the picture"

        view.set_image_shown(True)

        assert not backdrop.isVisible()
    finally:
        destroy(view)


# ---------------------------------------------------------------------------
# The keyboard diagram
# ---------------------------------------------------------------------------

def test_the_list_opens_with_a_keyboard(window):
    """Which of the keys under your hands do anything here is a question a
    picture answers faster than a list."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    help_text = window._keys.help_html(TAB_CURATE)

    assert help_text.startswith("<p><img src=\"data:image/png;base64,")


def test_a_tab_with_no_keys_of_its_own_gets_no_keyboard(window):
    """An almost empty keyboard says nothing the sentence beneath it does not."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_EXPORT

    help_text = window._keys.help_html(TAB_EXPORT)

    assert "<img" not in help_text
    assert "no keys of its own" in help_text


def test_a_chord_lights_its_modifiers_too(window):
    """Ctrl+Shift+Z is three keys somebody has to find, and a diagram showing
    only the letter would be saying the shortcut is one."""
    from am_segmentation_manager.gui.widgets.shortcuts import Binding, _keys_of

    def binding(key: str) -> Binding:
        return Binding(key, "", lambda: None, "any", "group")

    assert _keys_of([binding("Ctrl+Shift+Z")]) == {"Ctrl", "Shift", "Z"}
    assert _keys_of([binding("Left"), binding("Ctrl+S")]) == {"Left", "Ctrl", "S"}


def test_the_keyboard_picks_out_the_keys_it_is_given(qt_app):
    """Drawn from the bindings, so a key added to the window appears here
    without anything else being edited."""
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui.widgets.keyboard import keyboard_diagram
    from am_segmentation_manager.gui.widgets.theme import ACCENT, apply_theme

    stylesheet, palette = qt_app.styleSheet(), qt_app.palette()
    apply_theme(qt_app)
    try:
        plain = keyboard_diagram(set()).toImage()
        marked = keyboard_diagram({"G"}).toImage()

        assert plain.size() == marked.size(), "one key does not change the shape"
        assert plain != marked
        accent = QColor(ACCENT).rgb()
        assert not any(
            plain.pixel(x, y) == accent
            for x in range(plain.width())
            for y in range(plain.height())
        )
        assert any(
            marked.pixel(x, y) == accent
            for x in range(marked.width())
            for y in range(marked.height())
        )
    finally:
        qt_app.setStyleSheet(stylesheet)
        qt_app.setPalette(palette)


def test_every_bound_key_has_somewhere_to_be_drawn(window):
    """A key the diagram has no cap for is one the reader is told about in the
    table and cannot find on the picture."""
    from am_segmentation_manager.gui.widgets.keyboard import KEY_ROWS
    from am_segmentation_manager.gui.widgets.shortcuts import _keys_of

    drawable = {name for row in KEY_ROWS for name in row}
    drawable |= {"Up", "Down", "Left", "Right"}

    assert _keys_of(window._keys.bindings) <= drawable



# ---------------------------------------------------------------------------
# The conflict chooser
# ---------------------------------------------------------------------------

def test_the_chooser_offers_nothing_until_every_row_is_answered(qt_app, window):
    """A default would be a rule about which tree wins, which is the thing a
    conflict exists because there is not."""
    from qtpy.QtWidgets import QDialogButtonBox

    from am_segmentation_manager.gui.report import ConflictRow
    from am_segmentation_manager.gui.widgets.conflict_dialog import ConflictDialog

    rows = (
        ConflictRow(key="p/a.tif", title="p/a.tif", details=("the label",)),
        ConflictRow(key="p/b.tif", title="p/b.tif", details=("the label",)),
    )
    dialog = ConflictDialog(window, rows)
    try:
        merge = dialog.buttons.button(QDialogButtonBox.Ok)
        assert merge.isEnabled() is False
        assert dialog.resolutions == {}

        dialog._controls["p/a.tif"].buttons["ours"].setChecked(True)
        assert merge.isEnabled() is False, "one of two is not all of them"

        dialog._controls["p/b.tif"].buttons["theirs"].setChecked(True)
        assert merge.isEnabled() is True
        assert dialog.resolutions == {"p/a.tif": "ours", "p/b.tif": "theirs"}
    finally:
        destroy(dialog)


def test_the_chooser_says_which_rows_are_proofread_work(qt_app, window):
    """On the row, not counted at the top: which of these is proofread work
    decides how the choice is made, and a count does not say which row."""
    from qtpy.QtWidgets import QLabel

    from am_segmentation_manager.gui.report import ConflictRow
    from am_segmentation_manager.gui.widgets.conflict_dialog import ConflictDialog

    rows = (ConflictRow(key="p/a.tif", title="p/a.tif", corrected=True),)
    dialog = ConflictDialog(window, rows)
    try:
        said = [
            child.text()
            for child in dialog.findChildren(QLabel)
            if "Proofread" in child.text()
        ]
        assert said, "nothing on the row says it is proofread work"
    finally:
        destroy(dialog)


def test_the_viewer_draws_both_labels_over_the_same_acquisition(
    qt_app, window, tmp_path: Path
):
    """Flicked between in one place, not laid side by side: side by side halves
    the resolution of each, and a crack is a fraction of a percent of a frame."""
    import numpy as np
    import tifffile
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeyEvent

    from am_segmentation_manager.gui.report import ConflictRow
    from am_segmentation_manager.gui.widgets.conflict_viewer import ConflictViewer

    image = tmp_path / "img.tif"
    ours = tmp_path / "ours.tif"
    theirs = tmp_path / "theirs.tif"
    tifffile.imwrite(str(image), np.zeros((32, 32), dtype=np.uint8))
    tifffile.imwrite(str(ours), np.zeros((32, 32), dtype=np.uint8))
    mine = np.zeros((32, 32), dtype=np.uint8)
    mine[4:8, 4:8] = 2
    tifffile.imwrite(str(theirs), mine)

    row = ConflictRow(
        key="k", title="p/a.tif", ours=ours, theirs=theirs, image=image
    )
    viewer = ConflictViewer(window, row, window.controller.config)
    try:
        assert viewer._showing_theirs is False
        held = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_B, Qt.NoModifier)
        viewer.keyPressEvent(held)
        assert viewer._showing_theirs is True, "B did not swap to the copy's"
        assert "copy" in viewer.badge.text(), "nothing said which is on screen"

        released = QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_B, Qt.NoModifier)
        viewer.keyReleaseEvent(released)
        assert viewer._showing_theirs is False
        assert viewer.badge.text() == ""
    finally:
        destroy(viewer)


def test_neither_version_is_the_one_the_viewer_prefers(qt_app, window, tmp_path: Path):
    """Qt makes the first accepting button the default and draws it in the
    accent. A default here is a rule about which tree wins, which is the thing
    a conflict exists because there is not."""
    import numpy as np
    import tifffile

    from am_segmentation_manager.gui.report import ConflictRow
    from am_segmentation_manager.gui.widgets.conflict_viewer import ConflictViewer

    ours = tmp_path / "ours.tif"
    theirs = tmp_path / "theirs.tif"
    for path in (ours, theirs):
        tifffile.imwrite(str(path), np.zeros((8, 8), dtype=np.uint8))

    row = ConflictRow(key="k", title="t", ours=ours, theirs=theirs)
    viewer = ConflictViewer(window, row, window.controller.config)
    try:
        assert viewer.keep.isDefault() is False
        assert viewer.take.isDefault() is False
        assert viewer.decision is None, "nothing is chosen before anyone chooses"
    finally:
        destroy(viewer)


def test_labels_of_different_sizes_are_still_shown(qt_app, window, tmp_path: Path):
    """Seeing that one is the wrong size is the finding. Only the pixel to
    pixel mask has to go, because there is no such pairing to draw."""
    import numpy as np
    import tifffile

    from am_segmentation_manager.gui.report import ConflictRow
    from am_segmentation_manager.gui.widgets.conflict_viewer import ConflictViewer

    ours = tmp_path / "ours.tif"
    theirs = tmp_path / "theirs.tif"
    tifffile.imwrite(str(ours), np.zeros((32, 32), dtype=np.uint8))
    tifffile.imwrite(str(theirs), np.zeros((16, 16), dtype=np.uint8))

    row = ConflictRow(key="k", title="p/a.tif", ours=ours, theirs=theirs)
    viewer = ConflictViewer(window, row, window.controller.config)
    try:
        assert viewer._comparable() is False
        assert viewer.differences.isEnabled() is False
        assert "different sizes" in viewer.differences.toolTip()
        # Both are still drawable: the viewer swaps without complaint.
        viewer._showing_theirs = True
        viewer._draw()
    finally:
        destroy(viewer)


def test_a_state_row_offers_nothing_to_look_at(qt_app, window):
    """Two words against two words, both already on the row."""
    from am_segmentation_manager.gui.report import ConflictRow

    label = ConflictRow(key="k", title="t", ours=Path("a"), theirs=Path("b"))
    state = ConflictRow(key="k", title="t", details=("the state: a, b",))

    assert label.is_label is True
    assert state.is_label is False
