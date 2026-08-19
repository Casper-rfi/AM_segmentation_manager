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
import os
import time
from pathlib import Path

import pytest

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
    from qtpy.QtWidgets import QApplication, QMessageBox

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.metadata_controller import MetadataController
    from am_segmentation_manager.gui.widgets.image_view import MIN_FIT_FRACTION
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow
    from am_segmentation_manager.gui.widgets.metadata_panel import (
        MetadataPanel,
        editor_text,
        set_editor_text,
    )
    from am_segmentation_manager.gui.widgets.theme import BADGE_STYLES
    from am_segmentation_manager.metadata import (
        FIELD_COMMENTS,
        build_experiment_metadata,
    )
    from am_segmentation_manager.models import Project

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
    _settle(qt_app, lambda: window.image_panel.pair.left._item.pixmap().isNull() is False)
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
        pytest.fail("Timed out waiting for the window to settle")


# ---------------------------------------------------------------------------
# Loading and display
# ---------------------------------------------------------------------------

def test_scanning_populates_the_window(window):
    assert window.controller.project.name == "test-project"
    assert window.frame_label.text() == "1 / 4"
    assert window.project_panel.project_list.count() == 1
    assert window.project_panel.lamella_list.count() == 2
    assert "(1/2)" in window.image_panel.status.text()


def test_both_panes_are_drawn(window):
    """The plain image and the overlaid image are shown side by side."""
    assert not window.image_panel.pair.left._item.pixmap().isNull()
    assert not window.image_panel.pair.right._item.pixmap().isNull()


def test_frames_are_prepared_ahead_of_the_user(qt_app, window):
    _settle(qt_app, lambda: window.controller.cache.cached_image(1, "SEM") is not None)

    assert window.controller.cache.cached_image(1, "SEM") is not None


def test_stepping_frames_updates_the_view(qt_app, window):
    window._step_frame(1)
    _settle(qt_app, lambda: window.frame_label.text() == "2 / 4")

    assert window.controller.frame_position == 1
    assert window.timeline._current == 1


def test_frame_navigation_stops_at_the_end(window):
    window._step_frame(99)

    assert window.frame_label.text() == "4 / 4"


def test_changing_lamella_resets_to_its_first_frame(qt_app, window):
    window._step_frame(2)
    window._step_lamella(1)
    _settle(qt_app, lambda: window.controller.lamella.name == "02-beta")

    assert window.frame_label.text() == "1 / 2"
    assert "(2/2)" in window.image_panel.status.text()


def test_missing_fib_label_is_reported_not_hidden(qt_app, window):
    """This project has no labels_fib, which the badge must make plain."""
    window._set_mode("fib")
    _settle(qt_app, lambda: "no label" in window.image_panel.badges.text())

    assert "no label" in window.image_panel.badges.text()


def test_plot_mode_draws_the_plot(qt_app, window):
    window._set_mode("plot")
    _settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())

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
# The editing guard
# ---------------------------------------------------------------------------

def test_state_controls_start_locked(window):
    assert window.state_panel.state_box.isEnabled() is False
    assert window.controller.editing_enabled is False


def test_choosing_what_to_change_is_not_guarded(window):
    """Ticking rows and picking a modality write nothing, so they stay usable
    with the guard on -- and the reach line recounts as they change."""
    assert window.controller.editing_enabled is False

    assert window.project_panel.target_buttons["FIB"].isEnabled() is True
    assert window.project_panel.lamella_list.isEnabled() is True


def test_enabling_editing_unlocks_the_controls(window):
    window.state_panel.enable_check.setChecked(True)

    assert window.state_panel.state_box.isEnabled() is True
    assert window.controller.editing_enabled is True


def test_state_change_moves_files_and_updates_the_readout(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["SEM"].setChecked(True)
    record = window.controller.frame.record("SEM")
    original = record.image_path
    assert original.exists()

    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.state_of("SEM") == "deselected")

    assert not original.exists()
    assert record.image_path.parent.name == "deselected"
    assert record.image_path.exists()
    assert "SEM: deselected" in window.state_panel.readout.text()
    assert "FIB: selected" in window.state_panel.readout.text()


def test_applying_to_both_moves_both_modalities(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["both"].setChecked(True)

    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    for record in window.controller.frame.records:
        assert record.image_path.parent.name == "deselected"


def test_the_change_is_logged(window):
    window.state_panel.enable_check.setChecked(True)

    window.apply_state("deselected")

    assert "deselected" in window.changes_log.text()


def test_timeline_reflects_the_lamella(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["both"].setChecked(True)
    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    assert window.timeline._rows["SEM"][0] == "deselected"
    assert window.timeline._rows["FIB"][0] == "deselected"
    assert window.timeline._rows["SEM"][1] == "selected"


def test_mixed_state_shows_no_option_selected(qt_app, window):
    """A frame whose modalities disagree must not claim to be in one state."""
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.combined_state() == "mixed")

    window.project_panel.target_buttons["both"].setChecked(True)
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


def test_zoom_out_stops_just_below_fit(window):
    view = window.image_panel.pair.left
    for _ in range(10):
        view.zoom_by(0.5)

    assert view.transform().m11() == pytest.approx(
        view.fit_scale() * MIN_FIT_FRACTION, rel=0.02
    ), "the frame stops with a margin round it, not pressed against the pane"


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
    window.project_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("deselected")
    _settle(qt_app, lambda: window.timeline._rows["SEM"][0] == "deselected")

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


def _chip_image(sem: str, fib: str):
    """Render one status chip on its own, at its natural size."""
    from am_segmentation_manager.gui.controller import CurationStatus
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import SWATCH_HEIGHT, SWATCH_WIDTH

    icon = _status_icon(CurationStatus(sem=sem, fib=fib))
    return icon.pixmap(SWATCH_WIDTH, SWATCH_HEIGHT).toImage()


def test_the_two_modalities_are_stacked_not_side_by_side(qt_app):
    """Progress fills a band left to right, so the modalities must not also be
    split left to right: a chip was up to four segments in a row and nothing
    said which boundary separated the modalities. SEM above FIB also matches
    the timeline's rows."""
    from am_segmentation_manager.gui.widgets.frame_strip import STATE_COLOURS
    from am_segmentation_manager.gui.widgets.theme import (
        SWATCH_BAND,
        SWATCH_GAP,
        SWATCH_WIDTH,
    )

    image = _chip_image(sem="done", fib="not started")
    middle = SWATCH_WIDTH // 2

    assert image.pixelColor(middle, 2) == STATE_COLOURS["corrected"]
    assert image.pixelColor(middle, SWATCH_BAND + SWATCH_GAP + 2) == (
        STATE_COLOURS["selected"]
    )


def test_a_part_finished_modality_reads_as_a_progress_bar(qt_app):
    """Filled from the left in the finished colour, which is the one thing a
    half-and-half band can be without needing a fourth colour explained."""
    from am_segmentation_manager.gui.widgets.frame_strip import STATE_COLOURS
    from am_segmentation_manager.gui.widgets.theme import SWATCH_WIDTH

    image = _chip_image(sem="in progress", fib="empty")

    assert image.pixelColor(2, 2) == STATE_COLOURS["corrected"]
    assert image.pixelColor(SWATCH_WIDTH - 3, 2) == STATE_COLOURS["selected"]


def test_a_modality_with_no_images_draws_no_band(qt_app):
    """Absent has to read as absent, not as a state: many older datasets have
    no FIB whatsoever."""
    from am_segmentation_manager.gui.widgets.theme import (
        SWATCH_BAND,
        SWATCH_GAP,
        SWATCH_WIDTH,
    )

    image = _chip_image(sem="done", fib="empty")
    band = image.pixelColor(SWATCH_WIDTH // 2, SWATCH_BAND + SWATCH_GAP + 2)

    assert band.alpha() == 0


def test_a_chip_keeps_its_colour_on_the_row_being_looked_at(qt_app):
    """The regression: a 'done' teal came out blue-green on the current row,
    against a blue-tinted background -- so a finished lamella looked unfinished
    exactly when you selected it to check."""
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QListWidget

    from am_segmentation_manager.gui.controller import CurationStatus
    from am_segmentation_manager.gui.widgets.frame_strip import STATE_COLOURS
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import apply_theme

    stylesheet, palette = qt_app.styleSheet(), qt_app.palette()
    apply_theme(qt_app)
    try:
        done = CurationStatus(sem="done", fib="done")
        widget = QListWidget()
        widget.addItems(["current", "other"])
        for row in range(2):
            widget.item(row).setIcon(_status_icon(done))
        widget.setCurrentRow(0)
        widget.resize(300, 60)
        image = widget.grab().toImage()

        teal = STATE_COLOURS["corrected"].name()
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
        fresh.close()


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
    first.splitter.setSizes([400, 800, 400])
    sizes = first.splitter.sizes()
    first.close()

    second = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert second.splitter.sizes() == sizes
    finally:
        second.close()


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
        assert all(size > 0 for size in fresh.splitter.sizes())
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# Shortcut reference
# ---------------------------------------------------------------------------

def test_every_shortcut_is_documented(window):
    assert len(window._bindings) == len(window._shortcuts)
    assert all(binding.description for binding in window._bindings)
    assert any(binding.key == "F1" for binding in window._bindings)


def test_a_key_is_live_only_on_the_tab_it_belongs_to(window):
    """Curate keys firing behind another tab is what stopped the Metadata list
    ever seeing an arrow key."""
    window.tabs.setCurrentWidget(window.metadata_panel)

    live = {b.key for b, s in zip(window._bindings, window._shortcuts) if s.isEnabled()}

    assert live == {"Up", "Down", "F1"}


def test_curate_keys_come_back_with_the_curate_tab(window):
    window.tabs.setCurrentWidget(window.metadata_panel)
    window.tabs.setCurrentIndex(_curate_tab(window))

    live = {b.key for b, s in zip(window._bindings, window._shortcuts) if s.isEnabled()}

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
    assert window._help.isVisible()

    window.show_shortcuts()

    assert not window._help.isVisible(), "the key that opened it closes it"


def test_the_shortcut_list_does_not_block_the_window(window):
    window.show_shortcuts()

    assert not window._help.isModal(), (
        "a list you must dismiss before trying what it says is read once"
    )


def test_the_shortcut_list_follows_the_tab(window):
    from am_segmentation_manager.gui.widgets.main_window import TAB_LABEL

    window.show_shortcuts()
    window.tabs.setCurrentWidget(window.label_panel)

    assert window._help._text.text() == window.shortcut_help(TAB_LABEL)


def test_the_help_lists_only_the_active_tab(window):
    from am_segmentation_manager.gui.widgets.main_window import TAB_ADD_DATA, TAB_CURATE

    curate = window.shortcut_help(TAB_CURATE)
    metadata = window.shortcut_help(TAB_ADD_DATA)

    assert "Next frame" in curate and "Next project" not in curate
    assert "Next project" in metadata and "Next frame" not in metadata
    # The editing guard only governs the curate tab's keys.
    assert "Enable editing" in curate and "Enable editing" not in metadata


def test_the_help_describes_the_mouse_gestures_too(window):
    """Picking frames out has no key to press, so listing only keys left the
    whole selection mechanism undiscoverable."""
    from am_segmentation_manager.gui.widgets.main_window import TAB_CURATE

    help_text = window.shortcut_help(TAB_CURATE)

    assert "Ctrl-click" in help_text
    assert "Shift-click" in help_text
    assert "Tick a project or lamella" in help_text


def test_every_gesture_is_described(window):
    from am_segmentation_manager.gui.widgets.main_window import TAB_GESTURES

    for gestures in TAB_GESTURES.values():
        assert all(action and description for action, description in gestures)


def test_a_tab_with_no_keys_of_its_own_says_so(window):
    from am_segmentation_manager.gui.widgets.main_window import TAB_EXPORT

    help_text = window.shortcut_help(TAB_EXPORT)

    assert "no keys of its own" in help_text
    assert "F1" in help_text


def test_the_project_arrows_are_scoped_to_the_project_list(window):
    """Window-wide, they would take Up/Down from the editable combo boxes on
    the same tab."""
    from am_segmentation_manager.gui.widgets.main_window import TAB_ADD_DATA

    scoped = [b for b in window._bindings if b.tab == TAB_ADD_DATA]

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
    assert window.project_panel.target_buttons["FIB"].isChecked()

    window._set_mode("sem")
    assert window.controller.target == "SEM"
    assert window.project_panel.target_buttons["SEM"].isChecked()


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------

def test_corrected_confirmation_names_only_corrected_files(qt_app, window):
    """With a corrected SEM and a deselected FIB, only the SEM is at risk."""
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("corrected")
    window.project_panel.target_buttons["FIB"].setChecked(True)
    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.state_of("FIB") == "deselected")

    window.project_panel.target_buttons["both"].setChecked(True)
    at_risk = window.controller.corrected_records()

    assert [record.modality for record in at_risk] == ["SEM"]


def test_marking_corrected_without_an_annotation_asks_first(window, monkeypatch):
    """Corrected is the claim that a label was proofread, and a training export
    trusts it. The fixture's FIB images have no label at all, so calling them
    corrected is a claim about work that does not exist."""
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["FIB"].setChecked(True)
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
    window.project_panel.target_buttons["FIB"].setChecked(True)
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
    window.project_panel.target_buttons["SEM"].setChecked(True)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: pytest.fail("asked about an annotated label"),
    )

    window.apply_state("corrected")

    assert window.controller.state_of("SEM") == "corrected"


def test_the_warning_does_not_apply_to_the_other_states(window, monkeypatch):
    """Deselecting an unlabelled image is the ordinary case, not a mistake."""
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["FIB"].setChecked(True)
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
    window.project_panel.target_buttons["both"].setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(window, "_confirm_bulk", lambda state: True)

    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.frame_states()[0] == "deselected")

    assert all(state == "deselected" for state in window.controller.frame_states())
    assert "deselected" in window.changes_log.text()


def test_nothing_ticked_is_not_confirmed(window, monkeypatch):
    """The common case -- one frame -- must not ask every time."""
    window.state_panel.enable_check.setChecked(True)
    window.project_panel.target_buttons["both"].setChecked(True)
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

    assert window.project_panel.lamella_box.title() == "Lamellae (2, 1 ticked)"


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
        lambda description, work, on_done: on_done(work()),
    )
    monkeypatch.setattr(window, "_show_report", lambda title, report: None)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window, "_confirm_report",
        lambda title, report, action: asked.append((title, action)) or True,
    )
    return asked


def test_compressing_acts_on_the_ticked_lamella(window, inline_operations):
    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)

    window._compress_labels()

    assert inline_operations == [("Compress labels", "Apply")]
    assert "Compress labels" in window.changes_log.text()


def test_writing_metadata_files_rescans_so_the_new_projects_appear(
    project: Path, window, inline_operations
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

    window._add_metadata_files()

    assert (project / "loose-dataset" / "AM_metadata.yaml").exists()
    assert len(window.controller.projects) == 2
    assert window.metadata_panel.projects.count() == 2


def test_an_operation_previews_before_it_applies(window, monkeypatch):
    """The preview must be a real run of the operation, not a guess at one."""
    window.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    reports: list = []
    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done: on_done(work()),
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
    window.project_panel.target_buttons["both"].setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(window, "_confirm_bulk", lambda state: True)
    # The fixture's FIB images have no labels, which this ladder is not about.
    monkeypatch.setattr(window, "_confirm_unannotated", lambda: True)
    window.apply_state("corrected")
    _settle(qt_app, lambda: window.controller.frame_states()[0] == "corrected")

    monkeypatch.setattr(
        window, "_run_operation",
        lambda description, work, on_done: on_done(work()),
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

    window._compress_labels()


def test_an_operation_needs_something_selected(window, monkeypatch):
    """With no project scanned there is nothing for an operation to act on."""
    window.state_panel.enable_check.setChecked(True)
    monkeypatch.setattr(window.controller, "scoped_records", list)
    monkeypatch.setattr(
        window, "_run_operation",
        lambda *args: pytest.fail("should not have started"),
    )

    window._compress_labels()


def test_the_collection_actions_wait_for_a_folder_and_the_guard(qt_app, settings):
    """Both act on the whole scan root, so neither is offered before one exists."""
    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        panel = fresh.metadata_panel
        assert not panel.survey_button.isEnabled()
        assert not panel.init_button.isEnabled()

        panel.set_root(Path("/somewhere"))
        assert panel.survey_button.isEnabled()
        # Inspecting changes nothing; writing metadata files still waits.
        assert not panel.init_button.isEnabled()

        panel.enable.setChecked(True)
        assert panel.init_button.isEnabled()
    finally:
        fresh.close()


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
    _settle(qt_app, lambda: bool(results) or bool(failures))

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
    _settle(qt_app, lambda: bool(reported))

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

def test_plot_view_sets_the_cache_to_the_pane_size(qt_app, window):
    window._set_mode("plot")
    _settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())

    view = window.image_panel.plot_view
    pane = view.viewport().rect()
    ratio = view.devicePixelRatioF()
    assert window.controller.cache.plot_max_size == (
        round(pane.width() * ratio), round(pane.height() * ratio),
    )


def test_plot_is_not_resampled_again_by_the_view(qt_app, window):
    """The whole point of matching the pane: the view scales by about 1.0."""
    window._set_mode("plot")
    _settle(qt_app, lambda: not window.image_panel.plot_view._item.pixmap().isNull())

    assert window.image_panel.plot_view.transform().m11() == pytest.approx(1.0, abs=0.05)


def test_stepping_through_plots_finds_them_ready(qt_app, window):
    """The behaviour that was stuttering: holding an arrow key through plots."""
    window._set_mode("plot")
    _settle(
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
    assert controller.lamella_statuses()[position].sem == "not started"

    monkeypatch.setattr(window, "_confirm_bulk", lambda state: True)
    controller.set_editing_enabled(True)
    controller.set_target("SEM")
    controller.set_lamella_checked(position, True)
    window.apply_state("corrected")

    assert controller.lamella_statuses()[position].sem == "done"
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
    for field, comment in FIELD_COMMENTS.items():
        assert metadata_panel.captions[field].toolTip() == comment
        assert metadata_panel.editors[field].toolTip() == comment


def test_a_value_is_annotated_with_the_file_it_came_from(metadata_panel):
    assert metadata_panel.notes["sample"].text() == "[AM_metadata.yaml]"
    assert metadata_panel.notes["organisation"].text() == "[experiment.yaml]"
    assert metadata_panel.notes["sample"].styleSheet() == BADGE_STYLES["info"]


def test_stepping_moves_to_the_next_project_and_stops_at_the_ends(metadata_panel):
    """What Up/Down are bound to on this tab."""
    metadata_panel.step_project(1)
    assert metadata_panel.projects.currentRow() == 1
    assert metadata_panel.controller.position == 1

    # The last project: stepping past it stays there rather than wrapping.
    metadata_panel.step_project(1)
    assert metadata_panel.projects.currentRow() == 1

    metadata_panel.step_project(-1)
    assert metadata_panel.projects.currentRow() == 0
    assert metadata_panel.controller.position == 0


def test_an_outstanding_field_is_marked_rather_than_dimmed(metadata_panel):
    """A placeholder is the state that most needs seeing; it used to be drawn
    in the dimmest colour on the panel, in the same grey as provenance."""
    metadata_panel.projects.setCurrentRow(1)

    note = metadata_panel.notes["sample"]
    assert note.text() == "[TODO]"
    assert note.styleSheet() == BADGE_STYLES["warning"]
    assert editor_text(metadata_panel.editors["sample"]) == ""


def test_repeating_fields_offer_the_values_already_in_use(metadata_panel):
    box = metadata_panel.editors["sample"]

    assert box.isEditable()
    # A blank first entry, so "unset" stays reachable from the list.
    assert [box.itemText(row) for row in range(box.count())] == ["", "HeLa"]
    assert metadata_panel.editors["description"].__class__.__name__ == "QLineEdit"


def test_a_value_not_in_the_list_can_still_be_typed(metadata_panel):
    """The list is a shortcut, not a vocabulary -- the first project to use a
    sample has nothing to pick."""
    box = metadata_panel.editors["sample"]
    box.setCurrentText("Chlamydomonas")

    assert metadata_panel._edits() == {"sample": "Chlamydomonas"}
    # Typing must not grow the list, or a typo would be offered back later.
    assert box.count() == 2


def test_the_folder_name_is_offered_but_not_filled_in(metadata_panel):
    """Pre-filling it would save a guess nobody looked at as though someone
    had; the placeholder shows it and the button applies it."""
    editor = metadata_panel.editors["project"]

    assert editor.placeholderText() == "described (folder name)"
    assert editor_text(editor) == ""
    assert metadata_panel._edits() == {}


def test_the_folder_button_follows_the_editing_guard(metadata_panel):
    assert metadata_panel.folder_button.isEnabled() is False

    metadata_panel.enable.setChecked(True)
    assert metadata_panel.folder_button.isEnabled() is True

    metadata_panel.folder_button.click()
    assert editor_text(metadata_panel.editors["project"]) == "described"
    assert metadata_panel._edits() == {"project": "described"}


def test_the_list_says_which_projects_still_need_describing(metadata_panel):
    """Without this you have to click every project to find the empty ones."""
    described = metadata_panel.projects.item(0)
    outstanding = metadata_panel.projects.item(1)

    assert "3 of 6 fields set" in described.toolTip()
    assert "missing: description, project, lamella_type" in described.toolTip()
    # Only the organisation, from experiment.yaml: its sample is a placeholder.
    assert "1 of 6 fields set" in outstanding.toolTip()
    assert "sample" in outstanding.toolTip()
    assert not described.icon().isNull()


def test_a_long_row_is_elided_rather_than_scrolled(metadata_panel):
    """These paths differ in the timestamp they end with. With a horizontal
    scrollbar the view scrolls instead of eliding, and a run of projects from
    one session is cut off at the edge into identical-looking rows."""
    item = metadata_panel.projects.item(0)

    assert metadata_panel.projects.textElideMode() == Qt.ElideMiddle
    assert metadata_panel.projects.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert item.toolTip().startswith(item.text())


def test_saving_updates_the_mark_in_the_list(metadata_panel):
    """Saving is exactly when a project stops being incomplete, so leaving the
    list describing the state before it is how a user learns to distrust it."""
    assert "3 of 6 fields set" in metadata_panel.projects.item(0).toolTip()
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")
    metadata_panel._save()

    assert "4 of 6 fields set" in metadata_panel.projects.item(0).toolTip()


def test_right_clicking_a_project_asks_for_its_path(metadata_panel):
    """Reusing the curate tab's menu: the row is emitted, and resolving the
    path stays the controller's job."""
    rows: list[int] = []
    metadata_panel.project_path_requested.connect(rows.append)
    metadata_panel.project_path_requested.emit(1)

    assert rows == [1]
    assert metadata_panel.controller.project_path(1).is_absolute()
