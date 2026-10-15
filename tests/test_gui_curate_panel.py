"""Tests for the Curate tab: moving through the collection, drawing it, and changing state.

They drive the real window, as ``test_gui_window.py`` does, against the same small
synthetic project; each claim is the Curate tab's own. Skipped where Qt or the
optional image libraries are not installed.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import settle

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
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMessageBox

    from am_segmentation_manager.gui.widgets.theme import BADGE_STYLES
    from test_gui_window import _tick_lamella


# ---------------------------------------------------------------------------
# Loading and display
# ---------------------------------------------------------------------------

def test_scanning_populates_the_window(window):
    assert window.controller.project.name == "test-project"
    assert window.curate_panel.frame_label.text() == "1 / 4"
    assert len(list(window.curate_panel.project_panel.project_list.project_items())) == 1
    assert window.curate_panel.project_panel.lamella_list.count() == 2
    assert "(1/2)" in window.curate_panel.image_panel.status.full_text()


def test_both_panes_are_drawn(window):
    """The plain image and the overlaid image are shown side by side."""
    assert not window.curate_panel.image_panel.pair.left._item.pixmap().isNull()
    assert not window.curate_panel.image_panel.pair.right._item.pixmap().isNull()


def test_frames_are_prepared_ahead_of_the_user(qt_app, window):
    settle(qt_app, lambda: window.controller.cache.cached_image(1, "SEM") is not None)

    assert window.controller.cache.cached_image(1, "SEM") is not None


def test_stepping_frames_updates_the_view(qt_app, window):
    window.curate_panel.step_frame(1)
    settle(qt_app, lambda: window.curate_panel.frame_label.text() == "2 / 4")

    assert window.controller.frame_position == 1
    assert window.curate_panel.timeline._current == 1


def test_the_status_line_says_which_lamella_is_in_view(qt_app, window):
    window.curate_panel.step_lamella(1)
    settle(qt_app, lambda: window.controller.lamella.name == "02-beta")

    assert "(2/2)" in window.curate_panel.image_panel.status.full_text()


def test_missing_fib_label_is_reported_not_hidden(qt_app, window):
    """This project has no labels_fib, which the badge must make plain."""
    window.curate_panel.set_mode("fib")
    settle(qt_app, lambda: "no label" in window.curate_panel.image_panel.badges.text())

    assert "no label" in window.curate_panel.image_panel.badges.text()


def test_plot_mode_draws_the_plot(qt_app, window):
    window.curate_panel.set_mode("plot")
    settle(qt_app, lambda: not window.curate_panel.image_panel.plot_view._item.pixmap().isNull())

    assert window.curate_panel.image_panel.stack.currentIndex() == 1


def test_overlay_can_be_switched_off(window):
    window.curate_panel.image_panel.overlay_check.setChecked(False)

    assert window.controller.show_overlay is False
    assert not window.curate_panel.image_panel.pair.right._item.pixmap().isNull()


def test_the_overlay_pane_can_drop_the_image_under_the_label(window):
    window.curate_panel.image_panel.image_check.setChecked(False)

    assert window.controller.show_image is False
    left = window.curate_panel.image_panel.pair.left._item.pixmap()
    right = window.curate_panel.image_panel.pair.right._item.pixmap()
    assert not left.isNull(), "the left pane is the acquisition as recorded"
    assert not right.isNull()
    assert right.toImage() != left.toImage(), (
        "the overlay pane draws the label against black instead"
    )


# ---------------------------------------------------------------------------
# The editing guard
# ---------------------------------------------------------------------------

def test_state_controls_start_locked(window):
    assert window.curate_panel.state_panel.state_box.isEnabled() is False
    assert window.controller.editing_enabled is False


def test_choosing_what_to_change_is_not_guarded(window):
    """Ticking rows and picking a modality write nothing, so they stay usable
    with the guard on -- and the reach line recounts as they change."""
    assert window.controller.editing_enabled is False

    assert window.curate_panel.state_panel.target_buttons["FIB"].isEnabled() is True
    assert window.curate_panel.project_panel.lamella_list.isEnabled() is True


def test_enabling_editing_unlocks_the_controls(window):
    window.curate_panel.state_panel.enable_check.setChecked(True)

    assert window.curate_panel.state_panel.state_box.isEnabled() is True
    assert window.controller.editing_enabled is True


def test_a_state_change_is_declared_and_shown_in_the_readout(qt_app, window):
    from am_segmentation_manager.state import read_state

    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["SEM"].setChecked(True)
    record = window.controller.frame.record("SEM")
    original = record.image_path
    assert original.exists()

    window.curate_panel.apply_state("deselected")
    settle(qt_app, lambda: window.controller.state_of("SEM") == "deselected")

    assert original.exists(), "declaring a state moves nothing"
    assert record.image_path == original
    assert "SEM: deselected" in window.curate_panel.state_panel.readout.text()
    assert "FIB: selected" in window.curate_panel.state_panel.readout.text()

    stage = original.parent.parent
    assert list(read_state(stage).states.values()) == ["deselected"]


def test_the_change_is_reported_where_it_was_made(window):
    """A state change is the one thing on this tab that writes without a dialog
    to report it, so the buttons say what they just did."""
    window.curate_panel.state_panel.enable_check.setChecked(True)

    window.curate_panel.apply_state("deselected")

    assert "deselected" in window.curate_panel.state_panel.last_change.text()


def test_timeline_reflects_the_lamella(qt_app, window):
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["both"].setChecked(True)
    window.curate_panel.apply_state("deselected")
    settle(qt_app, lambda: window.controller.combined_state() == "deselected")

    assert window.curate_panel.timeline._rows["SEM"][0] == "deselected"
    assert window.curate_panel.timeline._rows["FIB"][0] == "deselected"
    assert window.curate_panel.timeline._rows["SEM"][1] == "selected"


def test_mixed_state_shows_no_option_selected(qt_app, window):
    """A frame whose modalities disagree must not claim to be in one state."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["SEM"].setChecked(True)
    window.curate_panel.apply_state("deselected")
    settle(qt_app, lambda: window.controller.combined_state() == "mixed")

    window.curate_panel.state_panel.target_buttons["both"].setChecked(True)
    window.curate_panel.refresh_state_panel()

    assert not any(
        button.isChecked() for button in window.curate_panel.state_panel.state_buttons.values()
    )
    assert "the images it covers differ" in window.curate_panel.state_panel.readout.text()


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_timeline_shows_each_modality_separately(qt_app, window):
    """A frame with one modality changed must not read as an unlabelled 'mixed'."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["SEM"].setChecked(True)
    window.curate_panel.apply_state("deselected")
    settle(qt_app, lambda: window.curate_panel.timeline._rows["SEM"][0] == "deselected")

    assert window.curate_panel.timeline._rows["SEM"][0] == "deselected"
    assert window.curate_panel.timeline._rows["FIB"][0] == "selected"


def test_clicking_the_timeline_seeks(window):
    window.curate_panel.timeline.resize(400, 24)
    window.curate_panel.timeline.set_states(window.controller.modality_states(), 0)

    window.curate_panel.timeline.frame_selected.emit(2)

    assert window.controller.frame_position == 2


# ---------------------------------------------------------------------------
# Long lists
# ---------------------------------------------------------------------------

def test_filtering_keeps_row_numbers_aligned_with_lamellae(window):
    """Rows must still map to real indices, or selection would target the wrong one."""
    window.curate_panel.project_panel.lamella_filter.setText("beta")
    window.curate_panel._select_lamella(1)

    assert window.controller.lamella.name == "02-beta"


# ---------------------------------------------------------------------------
# The edit target follows the view
# ---------------------------------------------------------------------------

def test_switching_view_points_the_target_at_that_modality(window):
    window.curate_panel.set_mode("fib")

    assert window.controller.target == "FIB"
    assert window.curate_panel.state_panel.target_buttons["FIB"].isChecked()

    window.curate_panel.set_mode("sem")
    assert window.controller.target == "SEM"
    assert window.curate_panel.state_panel.target_buttons["SEM"].isChecked()


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------

def _corrected_sem(window) -> None:
    """Mark the frame in view's SEM image corrected; its label is annotated."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["SEM"].setChecked(True)
    window.curate_panel.apply_state("corrected")
    assert window.controller.state_of("SEM") == "corrected"


def test_taking_an_image_out_of_corrected_asks_and_cancel_leaves_it(window, monkeypatch):
    """The dialog itself, not a stand-in for it: what it names is what is at
    stake, and Cancel is the answer it is given unless somebody chooses."""
    _corrected_sem(window)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append((args[1], args[2])) or QMessageBox.Cancel,
    )

    window.curate_panel.apply_state("deselected")

    [(title, text)] = asked
    assert title == "Change the state of corrected images?"
    assert "1 of these images" in text
    assert window.controller.corrected_records()[0].image_path.name in text
    assert "'deselected'" in text
    assert window.controller.state_of("SEM") == "corrected"


def test_an_image_leaves_corrected_once_that_is_confirmed(window, monkeypatch):
    _corrected_sem(window)
    asked: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append(args[1]) or QMessageBox.Yes,
    )

    window.curate_panel.apply_state("deselected")

    assert asked == ["Change the state of corrected images?"]
    assert window.controller.state_of("SEM") == "deselected"


def test_marking_corrected_without_an_annotation_asks_first(window, monkeypatch):
    """Corrected is the claim that a label was proofread, and a training export
    trusts it. The fixture's FIB images have no label at all, so calling them
    corrected is a claim about work that does not exist."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["FIB"].setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.Cancel,
    )

    window.curate_panel.apply_state("corrected")

    assert len(asked) == 1
    assert "1 with no label" in asked[0]
    assert window.controller.state_of("FIB") != "corrected"


def test_the_warning_names_the_lamella_so_it_can_be_found(window, monkeypatch):
    """Image filenames repeat between lamellae, so a list of them is not
    something anyone can act on."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["FIB"].setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.Cancel,
    )

    window.curate_panel.apply_state("corrected")

    assert "01-alpha" in asked[0]


def test_marking_an_annotated_label_corrected_asks_nothing(window, monkeypatch):
    """The guard has to be silent in the normal case, or it becomes a dialog
    people dismiss without reading."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["SEM"].setChecked(True)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: pytest.fail("asked about an annotated label"),
    )

    window.curate_panel.apply_state("corrected")

    assert window.controller.state_of("SEM") == "corrected"


def test_the_warning_does_not_apply_to_the_other_states(window, monkeypatch):
    """Deselecting an unlabelled image is the ordinary case, not a mistake."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["FIB"].setChecked(True)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: pytest.fail("asked about a deselection"),
    )

    window.curate_panel.apply_state("deselected")

    assert window.controller.state_of("FIB") == "deselected"


def test_ticking_a_lamella_drives_the_controller_and_the_title(window):
    """The lists scroll, so a tick out of view still has to be countable."""
    _tick_lamella(window)

    assert window.controller.lamella_checks() == [True, False]
    assert window.controller.describe_selection() == "1 lamella"
    assert window.curate_panel.project_panel.lamella_box.title() == "Lamellae (1 of 2 ticked)"


def test_a_tick_is_said_in_words_beside_the_state_buttons(window):
    _tick_lamella(window)

    assert "1 lamella" in window.curate_panel.state_panel.applies_to.text()


def test_a_ticked_selection_is_confirmed_before_anything_moves(qt_app, window, monkeypatch):
    """More than the frame on screen must not move without an explicit yes."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    asked = []
    monkeypatch.setattr(window.curate_panel, "_confirm_bulk", lambda state: asked.append(state) or False)

    window.curate_panel.apply_state("deselected")

    assert asked == ["deselected"]
    assert all(state == "selected" for state in window.controller.frame_states())


def test_nothing_ticked_is_not_confirmed(window, monkeypatch):
    """The common case -- one frame -- must not ask every time."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["both"].setChecked(True)
    monkeypatch.setattr(window.curate_panel, "_confirm_bulk", lambda state: pytest.fail("should not ask"))

    window.curate_panel.apply_state("deselected")

    assert window.controller.combined_state() == "deselected"


def test_ticks_survive_a_list_rebuild(window):
    """Every refresh rebuilds the rows, and the ticks decide what moves -- so
    losing them would silently narrow the next change."""
    _tick_lamella(window)

    window.curate_panel.refresh(reload_lists=True)

    assert window.curate_panel.project_panel.lamella_list.item(0).checkState() == Qt.Checked
    assert window.controller.lamella_checks() == [True, False]


def test_tick_all_and_none_cover_the_list(window):
    window.curate_panel.project_panel.lamellae_check_all.emit(True)
    assert window.controller.lamella_checks() == [True, True]

    window.curate_panel.project_panel.lamellae_check_all.emit(False)
    assert window.controller.lamella_checks() == [False, False]


def test_the_summary_follows_the_lamella_being_looked_at(window):
    """It counts this lamella's frames, so leaving it on the previous one
    reports a lamella nobody is looking at."""
    assert window.curate_panel.project_panel.summary.text().startswith("4 frames")

    window.curate_panel._select_lamella(1)

    assert window.curate_panel.project_panel.summary.text().startswith("2 frames")


def test_ticking_a_lamella_unticks_the_project_box_on_screen(window):
    """The clearing has to reach the widget, not only the model: a box still
    drawn ticked is exactly the tick somebody counts on."""
    window.curate_panel.project_panel.project_list.project_item(0).setCheckState(0, Qt.Checked)

    _tick_lamella(window)

    assert (
        window.curate_panel.project_panel.project_list.project_item(0).checkState(0) == Qt.Unchecked
    )
    assert window.curate_panel.project_panel.project_box.title() == "Projects (1)"


def test_ticking_a_project_unticks_the_lamella_box_on_screen(window):
    _tick_lamella(window)

    window.curate_panel.project_panel.project_list.project_item(0).setCheckState(0, Qt.Checked)

    assert window.curate_panel.project_panel.lamella_list.item(0).checkState() == Qt.Unchecked
    assert window.curate_panel.project_panel.lamella_box.title() == "Lamellae (2)"
    assert window.controller.describe_selection() == "1 project"


def test_picking_a_frame_unticks_the_lamella_box_on_screen(window):
    _tick_lamella(window)

    window.curate_panel._toggle_frame(0)

    assert window.curate_panel.project_panel.lamella_list.item(0).checkState() == Qt.Unchecked
    assert window.controller.describe_selection() == "1 frame"


def test_the_modality_control_sits_with_the_buttons_that_act(window):
    """Which rows is chosen on the left; which modality and which state are
    read together, beside the buttons that move them."""
    assert window.curate_panel.state_panel.target_buttons["SEM"].parent() is not None
    assert window.curate_panel.state_panel.isAncestorOf(window.curate_panel.state_panel.target_buttons["SEM"])
    assert not hasattr(window.curate_panel.project_panel, "target_buttons")


def test_the_reach_line_names_the_lamellae_and_the_modalities(window):
    """The count the user asked for before running an operation: how many
    lamellae, and how many images of each beam."""
    window.curate_panel.state_panel.target_buttons["both"].setChecked(True)
    _tick_lamella(window)

    text = window.curate_panel.state_panel.reach.text()

    assert "1 lamella" in text
    assert "SEM" in text and "FIB" in text


# ---------------------------------------------------------------------------
# The dialog a long operation runs behind
# ---------------------------------------------------------------------------

def test_the_operations_box_says_what_it_would_act_on(window):
    _tick_lamella(window)

    assert "1 lamella" in window.curate_panel.operations_panel.box.title()


def test_the_operations_box_says_whether_there_is_anything_to_convert(window):
    """Answered before the button is pressed, not by pressing it.

    The count comes from the selection, so it has to survive a refresh: this is
    the path that recomputes it, and a line wired up only at construction would
    pass a direct call to `show_predictions` and still be permanently stale.
    """
    _tick_lamella(window)
    window.curate_panel.refresh()

    assert window.curate_panel.operations_panel.predictions.text()


# ---------------------------------------------------------------------------
# Plots are prepared at the size they are shown at
# ---------------------------------------------------------------------------

def test_stepping_draws_a_plot_sized_to_the_pane(qt_app, window):
    """A plot is ~180 ms to read however it is asked for, so stepping can only
    be smooth if every one is already held -- which bounds them to the pane."""
    window.curate_panel.set_mode("plot")
    settle(qt_app, lambda: not window.curate_panel.image_panel.plot_view._item.pixmap().isNull())

    stepping = window.controller.cache.plot(0).plot

    assert max(stepping.shape[:2]) < 2100, "the fixture writes 1400x2100"


def test_the_figure_is_read_whole_for_the_frame_being_looked_at(qt_app, window):
    """Reducing it on read put a ceiling on what zooming in could ever show,
    which is what these are opened for."""
    window.curate_panel.set_mode("plot")
    settle(
        qt_app,
        lambda: window.controller.cache.cached_plot_detail(0) is not None,
    )

    assert window.controller.cache.plot_detail(0).plot.shape[:2] == (1400, 2100)


def test_swapping_to_the_whole_figure_draws_it_at_the_size_the_zoom_asked_for(
    qt_app, window
):
    """The swap happens inside a zoom, so it has to be invisible: the same
    region, at the size that zoom asked for, drawn from more pixels."""
    window.curate_panel.set_mode("plot")
    settle(qt_app, lambda: window.controller.cache.cached_plot_detail(0) is not None)
    window.curate_panel._render_plot()
    view = window.curate_panel.image_panel.plot_view
    drawn = view._item.boundingRect().width() * view.transform().m11()
    pane_copy = view._item.boundingRect().width()

    view.zoom_by(1.25)

    assert view._item.boundingRect().width() > pane_copy, "swapped to the whole figure"
    assert view._item.boundingRect().width() * view.transform().m11() == pytest.approx(
        drawn * 1.25, rel=0.02
    )


def test_a_plot_opens_centred_in_the_pane(qt_app, window):
    """Resetting the transform says nothing about where the view is looking,
    and the scene is deliberately larger than the picture -- so on its own it
    left the figure off to one side with a band of empty pane beside it."""
    window.curate_panel.set_mode("plot")
    settle(qt_app, lambda: not window.curate_panel.image_panel.plot_view._item.pixmap().isNull())
    view = window.curate_panel.image_panel.plot_view

    picture = view.mapFromScene(view._item.boundingRect().center())
    pane = view.viewport().rect().center()

    assert (picture.x(), picture.y()) == pytest.approx((pane.x(), pane.y()), abs=2)


def test_stepping_to_another_plot_does_not_move_the_picture(qt_app, window):
    """Every plot in a lamella is the same size, so a step should not disturb
    the view at all -- it used to refit, then shift again as the whole copy
    landed."""
    window.curate_panel.set_mode("plot")
    settle(qt_app, lambda: window.controller.cache.cached_plot_detail(0) is not None)
    window.curate_panel._render_plot()
    view = window.curate_panel.image_panel.plot_view
    before = view.transform().m11(), view._item.boundingRect().width()

    window.curate_panel.step_frame(1)
    settle(qt_app, lambda: window.controller.frame_position == 1)

    assert (view.transform().m11(), view._item.boundingRect().width()) == before


def test_stepping_through_plots_finds_them_ready(qt_app, window):
    """The behaviour that was stuttering: holding an arrow key through plots."""
    window.curate_panel.set_mode("plot")
    settle(
        qt_app,
        lambda: all(
            window.controller.cache.cached_plot(i) is not None
            for i in range(len(window.controller.frames))
        ),
    )

    missing = 0
    for _ in range(len(window.controller.frames) - 1):
        window.curate_panel.step_frame(1)
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

    monkeypatch.setattr(window.curate_panel, "_confirm_bulk", lambda state: True)
    controller.set_editing_enabled(True)
    controller.set_target("SEM")
    controller.set_lamella_checked(position, True)
    window.curate_panel.apply_state("corrected")

    assert controller.lamella_statuses()[position].of("adaptive", "SEM") == "done"
    # The widget, not just the controller, reflects it.
    item = window.curate_panel.project_panel.lamella_list.item(position)
    assert "SEM: done" in item.toolTip()


# ---------------------------------------------------------------------------
# The minimum/maximum range
# ---------------------------------------------------------------------------

def test_the_shown_range_says_so_on_screen(window):
    """The one control that makes the picture differ from the file. A reviewer
    who has forgotten it is on is deciding about an image no export contains."""
    panel = window.curate_panel.image_panel

    panel.intensity_slider.set_range(40, 200)

    assert "shown 40-200" in panel.badges.text()
    assert BADGE_STYLES["warning"] in panel.badges.text()


def test_the_range_is_kept_when_the_frame_changes(window):
    """Comparing one frame with the next is what it is for, and a setting lost
    on every step could not do that."""
    panel = window.curate_panel.image_panel
    panel.intensity_slider.set_range(30, 210)

    window.curate_panel.step_frame(1)

    assert panel._intensity_range == (30, 210)
    assert "shown 30-210" in panel.badges.text()


def test_moving_a_limit_redraws_without_rereading(qt_app, window, monkeypatch):
    """The whole reason it is applied after the cache: a slider movement must
    not invalidate a frame and send the loader back to re-read the file."""
    cache = window.controller.cache

    def read_ahead_is_done() -> bool:
        return not window.controller.pending_loads()

    # The window is handed over with its first frame drawn and the loader still
    # reading the neighbours, and those reads are not the slider's.
    settle(qt_app, read_ahead_is_done, what="the read-ahead before the limit moved")
    held = cache.cached_image(0, "SEM")
    assert held is not None, "the frame in view was read before the limit moved"
    reads: list = []
    # On the cache itself: it took its reader when it was built, so one put in
    # the module afterwards is never the one it calls.
    monkeypatch.setattr(
        cache, "_load_image", lambda *args, **kwargs: reads.append(args) or held.display
    )

    window.curate_panel.image_panel.intensity_slider.set_range(40, 255)

    assert cache.cached_image(0, "SEM") is held, "the frame was dropped to be read again"
    # A read the movement asked for happens on the loader's thread, so it is
    # waited for rather than missed by looking too soon.
    settle(qt_app, read_ahead_is_done, what="the loader after the limit moved")
    assert reads == []
