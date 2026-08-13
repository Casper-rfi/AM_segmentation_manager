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
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

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

            plot = np.full((40, 60, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(_ensure(stage / "plots" / f"{lamella}_AP_img_{number:03d}_plot.png")), plot)
    return root


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def window(qt_app, project: Path):
    """A window with the synthetic project scanned and its first frame ready."""
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )
    controller = CurationController(cache=FrameCache(), config=config)
    window = CurationWindow(controller)
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


# ---------------------------------------------------------------------------
# The editing guard
# ---------------------------------------------------------------------------

def test_state_controls_start_locked(window):
    assert window.state_panel.state_box.isEnabled() is False
    assert window.state_panel.target_box.isEnabled() is False
    assert window.controller.editing_enabled is False


def test_enabling_editing_unlocks_the_controls(window):
    window.state_panel.enable_check.setChecked(True)

    assert window.state_panel.state_box.isEnabled() is True
    assert window.controller.editing_enabled is True


def test_state_change_moves_files_and_updates_the_readout(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    record = window.controller.frame.sem
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
    window.state_panel.target_buttons["both"].setChecked(True)

    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    for record in window.controller.frame.records:
        assert record.image_path.parent.name == "deselected"


def test_the_change_is_logged(window):
    window.state_panel.enable_check.setChecked(True)

    window.apply_state("deselected")

    assert "deselected" in window.state_panel.log.toPlainText()


def test_timeline_reflects_the_lamella(qt_app, window):
    window.state_panel.enable_check.setChecked(True)
    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    assert window.timeline._rows["SEM"][0] == "deselected"
    assert window.timeline._rows["FIB"][0] == "deselected"
    assert window.timeline._rows["SEM"][1] == "selected"


def test_mixed_state_shows_no_option_selected(qt_app, window):
    """A frame whose modalities disagree must not claim to be in one state."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
    window.apply_state("deselected")
    _settle(qt_app, lambda: window.controller.combined_state() == "mixed")

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
    view = window.image_panel.pair.left
    for _ in range(10):
        view.zoom_by(0.5)

    assert view.transform().m11() == pytest.approx(view.fit_scale(), rel=0.02)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_timeline_shows_each_modality_separately(qt_app, window):
    """A frame with one modality changed must not read as an unlabelled 'mixed'."""
    window.state_panel.enable_check.setChecked(True)
    window.state_panel.target_buttons["SEM"].setChecked(True)
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
# Shortcut reference
# ---------------------------------------------------------------------------

def test_every_shortcut_is_documented(window):
    documented = dict(window._bindings)

    assert len(documented) == len(window._shortcuts)
    assert documented["F1"] == "Show this list"
    assert all(description for description in documented.values())
