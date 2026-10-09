"""Tests for the Train tab's widgets: what they show of the controller, and what they ask first.

The controller is checked without a window in ``test_training_controller``; here it is
the wiring -- that the form says what is missing, that nothing is written or deleted
until a dialog has been answered, and that the tab reads the runs again only while it
is showing and one is live. Dialogs are answered by the test rather than opened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("qtpy")

from am_segmentation_manager.gui.training_controller import TrainingController  # noqa: E402
from am_segmentation_manager.gui.widgets.training_panel import TrainingPanel  # noqa: E402
from am_segmentation_manager.operations import training_process, training_run  # noqa: E402
from conftest import destroy  # noqa: E402
from test_training_controller import _write_epochs  # noqa: E402
from test_training_process import CHILD, _settled, export_of_placeholders  # noqa: E402


class Answers:
    """What the test answers each dialog with, and what each was asked."""

    def __init__(self) -> None:
        self.go = True
        self.destination: Path | None = None
        self.asked: list[tuple[str, str]] = []


@pytest.fixture
def panel(qt_app, tmp_path: Path, monkeypatch):
    """A Train tab set up to train a new model, its dialogs answered by ``panel.answers``."""
    monkeypatch.setattr(training_run, "label_problem", lambda path, classes: None)
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    how = {"training": "finish"}

    def start(directory: Path):
        return training_process.start_training(
            directory, [sys.executable, str(script), how["training"]]
        )

    controller = TrainingController(stack_present=lambda: True, start=start)
    controller.manifest = export_of_placeholders(tmp_path)
    controller.training_directory = tmp_path / "runs"
    made = TrainingPanel(controller)
    answers = Answers()

    def confirm(title, lines, action):
        answers.asked.append((title, "\n".join(lines)))
        return answers.go

    def ask(title, text, action):
        answers.asked.append((title, text))
        return answers.go

    def say(title, text):
        answers.asked.append((title, text if isinstance(text, str) else "\n".join(text)))

    made._confirm, made._ask, made._report, made._warn = confirm, ask, say, say
    made._ask_destination = lambda suggested: answers.destination
    made.answers, made.how = answers, how
    yield made
    for run in controller.runs():
        controller.stop(run.directory)
    destroy(made)


def _started(panel: TrainingPanel) -> Path:
    """Start the run set up on the panel, wait for it to end, and return its directory."""
    panel.start()
    run = panel.controller.selected
    assert run is not None, panel.answers.asked
    _settled(run)
    panel.refresh()
    return run


def _with_checkpoints(panel: TrainingPanel, epochs=(1, 2, 3)) -> Path:
    run = _started(panel)
    written = _write_epochs(run, max(epochs), run_id=f"{run.name}_val")
    for epoch in epochs:
        (written / f"{run.name}_val_epoch{epoch:03}.pth").write_bytes(b"weights")
    panel.refresh()
    return run


def _select(panel: TrainingPanel, *epochs: int) -> None:
    for row in range(panel.checkpoints_list.count()):
        item = panel.checkpoints_list.item(row)
        item.setSelected(item.data(0x0100) in epochs)  # Qt.UserRole


# -- the form -----------------------------------------------------------------


def test_the_form_opens_on_a_new_model_with_a_name_suggested(panel):
    assert panel.mode.current() == "new"
    assert panel.name_edit.text().endswith("_new_01")
    assert not panel.weights_row.isVisibleTo(panel)
    assert panel.start_button.isEnabled()
    assert panel.set_summary.text() == "1 training and 1 validation pair(s)"


def test_refining_shows_the_weights_and_asks_for_them(panel):
    panel._mode_chosen("refine")

    assert panel.weights_row.isVisibleTo(panel)
    assert panel.name_edit.text().endswith("_refine_01")
    assert "Choose the model weights to refine" in panel.problems_label.text()
    assert not panel.start_button.isEnabled()
    assert panel.max_epochs.value() == 50 and panel.initial_rate.text() == "2e-06"


def test_a_name_the_person_typed_is_kept_when_the_mode_changes(panel):
    panel.name_edit.setText("arctis_round3")
    panel._name_edited("arctis_round3")

    panel._mode_chosen("refine")

    assert panel.name_edit.text() == "arctis_round3"


def test_a_name_that_cannot_be_one_stops_the_start(panel):
    panel._name_edited("has space")

    assert not panel.start_button.isEnabled()
    assert "letters, digits" in panel.problems_label.text()


def test_without_the_training_stack_the_form_says_what_to_install(qt_app, tmp_path: Path):
    controller = TrainingController(stack_present=lambda: False)
    controller.training_directory = tmp_path
    made = TrainingPanel(controller)
    try:
        assert not made.start_button.isEnabled()
        assert "--extra train" in made.problems_label.text()
    finally:
        destroy(made)


def test_an_advanced_setting_reaches_the_run(panel):
    panel.max_epochs.setValue(12)
    panel.save_all.setChecked(True)
    panel.initial_rate.setText("5e-6")
    panel._rate_edited("initial_learning_rate", panel.initial_rate)

    settings = panel.controller.settings

    assert (settings.max_epochs, settings.save_all) == (12, True)
    assert settings.initial_learning_rate == 5e-6


def test_a_learning_rate_that_is_not_a_number_is_put_back(panel):
    panel.initial_rate.setText("fast")
    panel._rate_edited("initial_learning_rate", panel.initial_rate)

    assert panel.initial_rate.text() == "1e-05"
    assert panel.controller.settings.initial_learning_rate == 1e-5


def test_default_settings_puts_every_setting_back(panel):
    panel.max_epochs.setValue(12)

    panel.reset_button.click()

    assert panel.reset_button.text() == "Default settings"
    assert panel.max_epochs.value() == 100


def test_the_training_batch_says_to_lower_it_when_memory_runs_out(panel):
    assert panel.batch.value() == 6
    assert "runs out of memory" in panel.batch.toolTip()


# -- starting -----------------------------------------------------------------


def test_starting_says_what_will_be_done_and_asks(panel):
    run = _started(panel)

    title, said = panel.answers.asked[0]
    assert title == "Start training"
    assert "1 training and 1 validation pair(s)" in said
    assert "goes on after this window is closed" in said
    assert panel.controller.selected_run().status.state == "finished"
    assert run.name in panel.runs_tree.project_item(0).text(0)


def test_declining_writes_and_starts_nothing(panel):
    panel.answers.go = False

    panel.start()

    assert not panel.controller.training_directory.exists()
    assert panel.runs_tree.project_items() == []


def test_a_run_that_cannot_be_prepared_gives_its_reasons_and_writes_nothing(panel):
    panel.controller.manifest.with_name("set_val.csv").unlink()

    panel.start()

    title, said = panel.answers.asked[0]
    assert title == "This run cannot be started"
    assert "validation list" in said
    assert not panel.controller.training_directory.exists()


def test_after_a_start_the_next_run_gets_a_new_name(panel):
    first = _started(panel)

    assert panel.name_edit.text() != first.name
    assert panel.name_edit.text().endswith("_new_02")


# -- the runs -----------------------------------------------------------------


def test_stop_is_offered_only_while_a_run_is_running_and_asks_first(panel):
    panel.how["training"] = "sleep"
    panel.start()
    panel.refresh()
    assert panel.stop_button.isEnabled()

    panel.answers.go = False
    panel.stop_selected()
    assert panel.controller.selected_run().status.state == "running"

    panel.answers.go = True
    panel.stop_selected()

    assert panel.controller.selected_run().status.state == "stopped"
    assert not panel.stop_button.isEnabled()
    assert "Stop training?" in [title for title, _ in panel.answers.asked]


def test_the_tab_reads_again_only_while_showing_and_a_run_is_live(panel, monkeypatch):
    panel.how["training"] = "sleep"
    panel.start()
    reads: list[int] = []
    monkeypatch.setattr(panel, "_show_runs", lambda runs=None: reads.append(1))

    panel.poll()
    assert reads == [], "hidden: nothing to redraw for"

    panel.show()
    panel.poll()
    assert reads == [1]

    panel.controller.stop(panel.controller.selected)
    panel.poll()
    assert reads == [1], "nothing live: the files will not change"


def test_curves_are_redrawn_only_when_an_epoch_has_been_added(panel, monkeypatch):
    """A redraw replaces the canvas, and with it any zoom the person had."""
    run = _started(panel)
    drawn: list[int] = []
    monkeypatch.setattr(panel.chart, "show_chart", lambda build, **_: drawn.append(1))
    _write_epochs(run, 2, run_id=f"{run.name}_val")

    panel.refresh()
    panel.refresh()
    assert drawn == [1]

    _write_epochs(run, 3, run_id=f"{run.name}_val")
    panel.refresh()
    assert drawn == [1, 1]


def test_a_run_with_no_epoch_says_so_rather_than_drawing(panel):
    _started(panel)

    assert "wrote no epoch" in panel.chart.message.text()


# -- checkpoints --------------------------------------------------------------


def test_checkpoints_are_listed_with_the_best_marked(panel):
    _with_checkpoints(panel)

    listed = [panel.checkpoints_list.item(row).text() for row in range(3)]

    assert listed == [
        "epoch 1 (incomplete)", "epoch 2 (incomplete)", "epoch 3 (run's best, incomplete)",
    ]


def test_a_copy_is_offered_for_exactly_one_checkpoint(panel):
    _with_checkpoints(panel)

    assert not panel.copy_button.isEnabled()
    _select(panel, 2)
    assert panel.copy_button.isEnabled()
    _select(panel, 1, 2)
    assert not panel.copy_button.isEnabled()


def test_saving_a_copy_puts_the_chosen_epoch_where_the_person_said(panel, tmp_path: Path):
    import zipfile

    run = _with_checkpoints(panel)
    source = next(run.rglob("*_epoch002.pth"))
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("weights/data.pkl", b"epoch two")
    panel.refresh()
    _select(panel, 2)
    panel.answers.destination = tmp_path / "models" / "AM_SEM_round3.pth"

    panel.copy_selected()

    assert panel.answers.destination.read_bytes() == source.read_bytes()
    assert source.exists()
    assert "choose that file as the model on Curate" in panel.answers.asked[-1][1]


def test_cancelling_the_file_dialog_copies_nothing(panel):
    _with_checkpoints(panel)
    _select(panel, 2)
    asked = len(panel.answers.asked)

    panel.copy_selected()

    assert len(panel.answers.asked) == asked


def test_deleting_the_others_says_how_many_and_asks_first(panel):
    run = _with_checkpoints(panel)
    _select(panel, 2)
    panel.answers.go = False

    panel.delete_others()

    title, said = panel.answers.asked[-1]
    assert title == "Delete the other checkpoints?"
    assert "Delete 1 checkpoint(s)" in said and "keeping epoch(s) 2" in said
    assert "Epoch 3 (the run's best) is kept as well." in said
    assert "cannot be undone" in said
    assert len(list(run.rglob("*.pth"))) == 3


def test_deleting_the_others_keeps_what_is_selected(panel):
    run = _with_checkpoints(panel)
    _select(panel, 1, 3)

    panel.delete_others()

    assert sorted(path.name[-12:] for path in run.rglob("*.pth")) == ["epoch001.pth", "epoch003.pth"]
    assert panel.checkpoints_list.count() == 2


def test_delete_is_not_offered_with_nothing_or_everything_selected(panel):
    _with_checkpoints(panel)

    assert not panel.delete_button.isEnabled()
    _select(panel, 1, 2, 3)
    assert not panel.delete_button.isEnabled()
    _select(panel, 1)
    assert panel.delete_button.isEnabled()


def test_delete_is_not_offered_while_the_run_is_training(panel):
    panel.how["training"] = "sleep"
    panel.start()
    run = panel.controller.selected
    written = _write_epochs(run, 2, run_id=f"{run.name}_val")
    for epoch in (1, 2):
        (written / f"{run.name}_val_epoch{epoch:03}.pth").write_bytes(b"weights")
    panel.refresh()
    _select(panel, 1)

    assert not panel.delete_button.isEnabled()


# -- runs am-seg did not start ------------------------------------------------


def test_a_folder_of_am_model_training_s_own_runs_is_listed_and_drawn(panel, monkeypatch):
    from test_training_controller import _bare_run

    _bare_run(panel.controller.training_directory, "wide_seed2")
    drawn: list[int] = []
    monkeypatch.setattr(panel.chart, "show_chart", lambda build, **_: drawn.append(1))

    panel.refresh()

    assert panel.runs_tree.project_item(0).text(0).startswith("wide_seed2 · finished")
    assert "Not started by am-seg" in panel.status_label.text()
    assert drawn == [1]
    assert not panel.stop_button.isEnabled()


# -- the runs as a tree -------------------------------------------------------


def _three_runs(panel: TrainingPanel) -> None:
    """List two runs under one folder and one beside it, the newest first."""
    from test_training_controller import _bare_run

    root = panel.controller.training_directory
    _bare_run(root, "step8/fold1", "260103_000000")
    _bare_run(root, "step8/fold2", "260102_000000")
    _bare_run(root, "alone", "260101_000000")
    panel.refresh()


def _folder(panel: TrainingPanel, name: str):
    return next(top for top in panel.runs_tree._top_items() if top.text(0) == name)


def test_runs_are_gathered_under_the_folder_they_sit_in(panel):
    _three_runs(panel)

    folder = _folder(panel, "step8")

    assert [folder.child(row).text(0).split(" · ")[0] for row in range(2)] == ["fold1", "fold2"]
    assert panel.runs_tree.project_item(2).parent() is None


def test_ticking_a_folder_ticks_the_runs_in_it(panel, qt_app):
    from qtpy.QtCore import Qt

    _three_runs(panel)

    _folder(panel, "step8").setCheckState(0, Qt.CheckState.Checked)

    assert sorted(run.label for run in panel.controller.ticked_runs()) == ["fold1", "fold2"]


def test_tick_all_and_none_reach_every_run(panel):
    _three_runs(panel)

    panel._tick_all(True)
    assert panel.runs_tree.ticked_rows() == [0, 1, 2]
    assert len(panel.controller.ticked_runs()) == 3

    panel._tick_all(False)
    assert panel.runs_tree.ticked_rows() == []
    assert panel.controller.ticked == set()


def test_reading_the_runs_again_keeps_the_ticks_the_folds_and_the_run_looked_at(panel):
    _three_runs(panel)
    panel.runs_tree.setCurrentItem(panel.runs_tree.project_item(1))
    panel.runs_tree.project_item(2).setCheckState(0, _checked())
    _folder(panel, "step8").setExpanded(True)

    panel.refresh()

    assert panel.controller.selected_run().label == "fold2"
    assert panel.runs_tree.currentItem() is panel.runs_tree.project_item(1)
    assert panel.runs_tree.ticked_rows() == [2]
    assert _folder(panel, "step8").isExpanded()


def test_the_row_looked_at_decides_what_is_shown_and_a_tick_does_not(panel):
    _three_runs(panel)
    shown = panel.status_label.text()

    panel.runs_tree.project_item(2).setCheckState(0, _checked())
    assert panel.status_label.text() == shown

    panel.runs_tree.setCurrentItem(panel.runs_tree.project_item(2))
    assert panel.status_label.text().startswith("alone:")


def test_a_run_s_row_gives_the_path_to_copy_or_reveal(panel):
    _three_runs(panel)

    assert panel.run_path(2) == panel.controller.training_directory / "alone" / "training_260101_000000"
    assert panel.run_path(7) is None


def _checked():
    from qtpy.QtCore import Qt

    return Qt.CheckState.Checked


# -- the chip beside a run ----------------------------------------------------


def test_the_chip_fills_by_epochs_trained_and_by_weights_still_on_disk(panel):
    from am_segmentation_manager.gui.widgets.training_panel import (
        CHIP_FINISHED,
        CHIP_WEIGHTS,
        chip_fills,
    )

    run = _with_checkpoints(panel, epochs=(1, 2, 3, 4))
    _select(panel, 4)
    panel.delete_others()

    epochs, weights = chip_fills(panel.controller.summary(run))

    assert epochs == (4 / 100, CHIP_FINISHED)
    assert weights == (1 / 4, CHIP_WEIGHTS)


def test_a_run_that_ended_early_or_may_still_be_training_is_coloured_apart(panel):
    from am_segmentation_manager.gui.widgets.training_panel import (
        CHIP_ENDED_EARLY,
        CHIP_LIVE,
        chip_fills,
    )
    from test_training_controller import _aged, _bare_run

    root = panel.controller.training_directory
    live = _bare_run(root, "live", finished=False)
    ended = _bare_run(root, "ended", "260101_000000", finished=False)
    _aged(ended, 7200)

    assert chip_fills(panel.controller.summary(live))[0] == (0.0, CHIP_LIVE)
    assert chip_fills(panel.controller.summary(ended))[0][1] == CHIP_ENDED_EARLY


def test_each_run_carries_its_chip_and_says_it_in_words_on_hover(panel):
    run = _with_checkpoints(panel)

    item = panel.runs_tree.project_item(0)

    assert not item.icon(0).isNull()
    assert item.toolTip(0) == (
        f"{run}\n3 of at most 100 epochs trained\n3 of 3 saved weights on disk, 0.0 GB"
    )


def test_the_key_list_says_what_a_run_s_chip_means(qt_app):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_TRAIN, _tab_marks

    meanings = " ".join(meaning for _, _, meaning in _tab_marks(TAB_TRAIN))

    assert "epochs trained" in meanings and "still on disk" in meanings


# -- classes, and the point under the pointer ---------------------------------


@pytest.fixture
def drawn(panel):
    """The panel showing the real run's curves."""
    import shutil

    pytest.importorskip("matplotlib")
    real = Path(__file__).parent / "data" / "amt_run" / "training_260927_093830"
    shutil.copytree(real, panel.controller.training_directory / "real" / real.name)
    panel.refresh()
    return panel


def _curves(panel: TrainingPanel) -> set[str]:
    _, metric_axes = panel.chart._figure.axes
    return {line.get_label() for line in metric_axes.get_lines() if line.get_gid() == "curve"}


def test_a_box_is_offered_for_each_class_of_the_run_all_but_background_ticked(drawn):
    assert list(drawn.class_boxes) == ["background", "gis", "lamella", "crack", "vacuum"]
    assert [name for name, box in drawn.class_boxes.items() if not box.isChecked()] == ["background"]
    assert "crack" in _curves(drawn) and "background" not in _curves(drawn)

    drawn.class_boxes["background"].setChecked(True)
    assert "background" in _curves(drawn)


def test_unticking_a_class_takes_its_curve_out_and_leaves_the_averages(drawn):
    drawn.class_boxes["crack"].setChecked(False)

    curves = _curves(drawn)

    assert "crack" not in curves and "gis" in curves
    assert "weighted average, validation" in curves


def test_a_class_unticked_is_still_unticked_when_the_runs_are_read_again(drawn):
    drawn.class_boxes["vacuum"].setChecked(False)

    drawn.refresh()

    assert not drawn.class_boxes["vacuum"].isChecked()
    assert "vacuum" not in _curves(drawn)


def test_pointing_at_a_curve_names_the_class_the_epoch_and_the_value(drawn, qt_app):
    chart = drawn.chart
    chart._canvas.draw()
    _, metric_axes = chart._figure.axes
    line = next(line for line in metric_axes.get_lines() if line.get_label() == "gis")
    x, y = line.get_transform().transform(line.get_xydata()[0])

    chart.point_at(x, y)
    assert chart.hovered().startswith("gis · epoch 1 · IoU 0.")

    chart.point_at(1, 1)
    assert chart.hovered() == ""


def test_nothing_is_named_while_panning(drawn, qt_app):
    chart = drawn.chart
    chart._canvas.draw()
    _, metric_axes = chart._figure.axes
    line = next(line for line in metric_axes.get_lines() if line.get_label() == "gis")
    x, y = line.get_transform().transform(line.get_xydata()[0])

    chart.pan_button.click()
    chart.point_at(x, y)

    assert chart.hovered() == ""


# -- the epoch marked as best -------------------------------------------------


def test_marking_the_selected_epoch_writes_it_beside_the_run_and_says_so(panel):
    from am_segmentation_manager.operations.training_choice import read_chosen_best

    run = _with_checkpoints(panel)
    _select(panel, 2)

    panel.mark_best()

    assert read_chosen_best(run).epoch == 2
    assert panel.checkpoints_list.item(1).text() == "epoch 2 (marked best, incomplete)"
    assert "Marked as best: epoch 2" in panel.status_label.text()
    assert "marked 2" in panel.runs_tree.project_item(0).text(0)
    assert panel.unmark_button.isEnabled()


def test_with_no_epoch_selected_marking_asks_which(panel):
    run = _with_checkpoints(panel)
    asked: list[str] = []
    panel._ask_epoch = lambda listed: asked.append(listed.name) or 1

    panel.mark_best()

    assert asked == [run.name]
    assert panel.controller.summary(run).chosen.epoch == 1


def test_cancelling_the_question_marks_nothing(panel):
    run = _with_checkpoints(panel)
    panel._ask_epoch = lambda listed: None

    panel.mark_best()

    assert panel.controller.summary(run).chosen is None


def test_clearing_takes_the_mark_back_and_leaves_the_run_s_own_best(panel):
    run = _with_checkpoints(panel)
    _select(panel, 2)
    panel.mark_best()

    panel.clear_best()

    assert panel.controller.summary(run).chosen is None
    assert panel.checkpoints_list.item(2).text() == "epoch 3 (run's best, incomplete)"
    assert not panel.unmark_button.isEnabled()


def test_deleting_the_others_never_takes_the_epoch_marked_as_best(panel):
    run = _with_checkpoints(panel)
    _select(panel, 1)
    panel.mark_best()
    _select(panel, 2)
    panel.keep_best_check.setChecked(False)

    panel.delete_others()

    assert "Epoch 1 (marked as best) is kept as well." in panel.answers.asked[-1][1]
    assert sorted(path.name[-12:] for path in run.rglob("*.pth")) == ["epoch001.pth", "epoch002.pth"]


def test_the_run_s_best_goes_with_the_others_only_when_its_box_is_unticked(panel):
    run = _with_checkpoints(panel)
    _select(panel, 1)

    panel.delete_others()
    assert sorted(path.name[-12:] for path in run.rglob("*.pth")) == ["epoch001.pth", "epoch003.pth"]

    panel.keep_best_check.setChecked(False)
    _select(panel, 1)
    panel.delete_others()
    assert sorted(path.name[-12:] for path in run.rglob("*.pth")) == ["epoch001.pth"]


def test_the_curves_mark_the_epoch_marked_as_best(drawn):
    run = drawn.controller.selected_run()
    drawn.controller.choose_best(run.directory, 1)

    drawn.refresh()

    _, metric_axes = drawn.chart._figure.axes
    assert "marked best (1)" in {line.get_label() for line in metric_axes.get_lines()}


def test_the_key_list_says_what_the_stars_beside_a_checkpoint_mean(qt_app):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_TRAIN, _tab_marks

    meanings = " ".join(meaning for _, _, meaning in _tab_marks(TAB_TRAIN))

    assert "the run counts as its best" in meanings and "you marked as best" in meanings


# -- the ticked runs side by side ---------------------------------------------


@pytest.fixture
def tabled(drawn):
    """The panel on its table, the real run listed twice and neither ticked."""
    import shutil

    real = Path(__file__).parent / "data" / "amt_run" / "training_260927_093830"
    shutil.copytree(real, drawn.controller.training_directory / "again" / real.name)
    drawn.refresh()
    drawn.view_control.buttons["table"].click()
    return drawn


def _table_text(panel: TrainingPanel) -> list[str]:
    (axes,) = panel.chart._figure.axes
    return [text.get_text() for text in axes.texts]


def test_the_table_asks_for_runs_until_one_is_ticked(tabled):
    assert tabled.chart.message.text() == "Tick the runs to compare."
    assert tabled.table_row.isVisibleTo(tabled)

    tabled.runs_tree.project_item(0).setCheckState(0, _checked())

    assert "gis" in _table_text(tabled)


def test_the_table_holds_the_ticked_runs_and_the_classes_ticked(tabled):
    tabled._tick_all(True)
    (axes,) = tabled.chart._figure.axes
    assert len(axes.get_yticklabels()) == 2
    assert "background" not in _table_text(tabled)

    tabled.class_boxes["gis"].setChecked(False)

    assert "gis" not in _table_text(tabled)
    assert "crack" in _table_text(tabled)


def test_the_range_of_epochs_is_offered_only_for_a_mean_or_median(tabled):
    tabled._tick_all(True)
    assert not tabled.first_epoch.isVisibleTo(tabled)

    tabled.value_box.setCurrentIndex(tabled.value_box.findData("median"))
    assert tabled.first_epoch.isVisibleTo(tabled)
    assert tabled.controller.table_value == "median"

    tabled.last_epoch.setValue(3)
    tabled.first_epoch.setValue(2)
    assert tabled.controller.epoch_range == (2, 3)
    (axes,) = tabled.chart._figure.axes
    assert axes.get_yticklabels()[0].get_text().endswith("2-3 (2)")


def test_the_table_is_exported_as_a_csv_and_only_from_the_table(tabled, tmp_path: Path):
    tabled._tick_all(True)
    destination = tmp_path / "table.csv"
    tabled._ask_table_destination = lambda: destination

    tabled.export_table()

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "run,epochs,gis iou,lamella iou,crack iou,vacuum iou"
    assert len(lines) == 3
    assert tabled.chart.export_button.isVisibleTo(tabled)

    tabled.view_control.buttons["curves"].click()
    assert not tabled.chart.export_button.isVisibleTo(tabled)
    assert not tabled.table_row.isVisibleTo(tabled)


def test_the_table_is_redrawn_only_when_it_would_differ(tabled, monkeypatch):
    tabled._tick_all(True)
    drawn: list[int] = []
    monkeypatch.setattr(tabled.chart, "show_chart", lambda build, **_: drawn.append(1))

    tabled.refresh()
    assert drawn == []

    tabled.controller.choose_best(tabled._listed[0].directory, 1)
    tabled.value_box.setCurrentIndex(tabled.value_box.findData("marked"))
    assert drawn == [1]


# -- validations --------------------------------------------------------------


@pytest.fixture
def validated(panel, tmp_path: Path, monkeypatch):
    """The panel listing one run, validated on an image of each of two instruments."""
    pytest.importorskip("matplotlib")
    from test_validation_run import _run, scored_validation

    root = tmp_path / "validated"
    root.mkdir()
    scored_validation(root, monkeypatch, _run(root))
    panel.controller.training_directory = root / "runs"
    panel.controller.hidden_classes = set()
    panel.refresh()
    return panel


def _choose(box, value: str) -> None:
    box.setCurrentIndex(box.findData(value))
    box.activated.emit(box.currentIndex())


def test_a_run_s_validations_are_said_and_offered(validated):
    assert "Validations: held finished, 3 of 3 epoch(s) scored." in validated.status_label.text()
    assert [validated.validated_box.itemData(i) for i in range(3)] == ["", "held", "*"]
    assert not validated.split_box.isEnabled()


def test_choosing_a_validation_draws_the_curves_from_it(validated):
    _choose(validated.validated_box, "held")

    _, metric_axes = validated.chart._figure.axes
    assert metric_axes.get_ylabel().endswith("from held")
    assert validated.split_box.isEnabled()

    _choose(validated.split_box, "instrument")
    assert [validated.subset_box.itemData(i) for i in range(2)] == ["Arctis", "Hydra"]
    _choose(validated.subset_box, "Hydra")
    _, metric_axes = validated.chart._figure.axes
    assert metric_axes.get_ylabel().endswith("from held, Hydra")


def test_the_table_gives_each_part_a_column_and_a_box_to_leave_it_out(validated):
    validated._tick_all(True)
    validated.view_control.buttons["table"].click()
    _choose(validated.validated_box, "held")
    _choose(validated.split_box, "instrument")

    assert list(validated.set_boxes) == ["Arctis", "Hydra"]
    assert validated.set_area.isVisibleTo(validated)
    assert not validated.subset_box.isVisibleTo(validated)
    assert {"Arctis", "Hydra"} <= set(_table_text(validated))

    validated.set_boxes["Arctis"].setChecked(False)
    assert "Arctis" not in _table_text(validated)


def test_every_set_puts_the_validation_beside_the_run_s_own(validated):
    validated._tick_all(True)
    validated.view_control.buttons["table"].click()

    _choose(validated.validated_box, "*")

    assert list(validated.set_boxes) == ["validation", "held"]
    assert {"validation", "held"} <= set(_table_text(validated))


def test_validating_asks_what_to_score_on_checks_it_and_asks_before_starting(
    validated, tmp_path: Path
):
    manifest = tmp_path / "validated" / "export" / "set_manifest.yaml"
    started: list[Path] = []
    validated.controller._start = started.append
    validated._ask_validation = lambda run: (manifest, "second", "validation")

    validated.validate()

    title, said = validated.answers.asked[-1]
    assert title == "Start validating" and "1 image(s), the validation list" in said
    assert [path.name for path in started] == ["second"]
    assert "second prepared, 0 of 3 epoch(s) scored" in validated.status_label.text()


def test_declining_or_cancelling_a_validation_writes_nothing(validated, tmp_path: Path):
    run = validated.controller.selected_run()
    manifest = tmp_path / "validated" / "export" / "set_manifest.yaml"
    validated._ask_validation = lambda listed: None
    validated.validate()
    validated._ask_validation = lambda listed: (manifest, "second", "all")
    validated.answers.go = False
    validated.validate()

    assert [v.name for v in validated.controller.summary(run.directory).validations] == ["held"]


def test_a_validation_that_cannot_be_started_gives_its_reasons(validated, tmp_path: Path):
    validated._ask_validation = lambda run: (tmp_path / "no_manifest.yaml", "held", "all")

    validated.validate()

    title, said = validated.answers.asked[-1]
    assert title == "This validation cannot be started" and "already exists" in said


def test_the_dialog_offers_the_export_s_name_and_refuses_one_taken(validated, tmp_path: Path):
    from am_segmentation_manager.gui.widgets.training_panel import ValidationDialog

    run = validated.controller.selected_run()
    dialog = ValidationDialog(validated, run, tmp_path / "exports" / "hydra_manifest.yaml")
    assert dialog.name_edit.text() == "hydra" and dialog.check_button.isEnabled()
    assert dialog.asked()[1:] == ("hydra", "validation")

    dialog.name_edit.setText("held")
    assert not dialog.check_button.isEnabled()
    assert "already has a validation named held" in dialog.problem_label.text()
    dialog.deleteLater()


def test_a_stop_waits_off_the_interface_thread(panel):
    """A process that will not go is waited on for up to a minute."""
    panel.how["training"] = "sleep"
    panel.start()
    panel.refresh()
    ran = []
    inline = panel.run_in_background

    def background(work, done, description):
        ran.append(description)
        inline(work, done, description)

    panel.run_in_background = background
    panel.stop_selected()

    assert ran == [f"Stopping {panel.controller.selected_run().name}"]
    assert panel.controller.selected_run().status.state == "stopped"


def test_starting_beside_a_run_in_training_says_so_before_it_starts(panel):
    panel.how["training"] = "sleep"
    panel.start()
    panel.refresh()
    panel.answers.asked.clear()
    panel.answers.go = False

    panel.start()

    title, said = panel.answers.asked[-1]
    assert title == "Start training" and "may be training now" in said


# -- moving about a chart -----------------------------------------------------


def test_the_wheel_zooms_a_chart_and_reset_view_undoes_it(drawn):
    chart = drawn.chart
    chart._canvas.draw()
    _, metric_axes = chart._figure.axes
    before = metric_axes.get_xlim()
    box = metric_axes.bbox

    chart.wheel((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, 1)
    low, high = metric_axes.get_xlim()
    assert high - low < before[1] - before[0]

    chart.reset_button.click()
    assert metric_axes.get_xlim() == pytest.approx(before)


def test_space_pans_for_as_long_as_it_is_held_and_gives_the_mode_back(drawn, qt_app):
    """As on the Label canvas; a repeat of the held key is not a second press."""
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication

    chart = drawn.chart

    def space(kind, repeat=False):
        event = QKeyEvent(kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", repeat)
        QApplication.sendEvent(chart._canvas, event)

    chart.zoom_button.click()
    space(QEvent.Type.KeyPress)
    space(QEvent.Type.KeyPress, repeat=True)
    assert chart._toolbar.mode == "pan/zoom" and chart.pan_button.isChecked()

    space(QEvent.Type.KeyRelease)
    assert chart._toolbar.mode == "zoom rect" and chart.zoom_button.isChecked()

    chart.zoom_button.click()
    space(QEvent.Type.KeyPress)
    assert chart._toolbar.mode == "pan/zoom"
    space(QEvent.Type.KeyRelease)
    assert not chart._toolbar.mode


def test_space_is_let_go_of_when_the_keyboard_leaves_the_chart(drawn):
    """A release that goes to another window would leave the chart panning."""
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QFocusEvent
    from qtpy.QtWidgets import QApplication

    chart = drawn.chart
    chart.hold_space(True)

    QApplication.sendEvent(chart._canvas, QFocusEvent(QEvent.Type.FocusOut))

    assert not chart._toolbar.mode and not chart.pan_button.isChecked()


def test_a_turn_of_the_wheel_over_the_chart_reaches_it(drawn):
    """Through Qt and matplotlib's own event, which is the wiring a call to `wheel` skips."""
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication

    chart = drawn.chart
    chart._canvas.draw()
    _, metric_axes = chart._figure.axes
    before = metric_axes.get_xlim()
    box = metric_axes.bbox
    ratio = chart._canvas.devicePixelRatioF()
    where = QPointF(
        (box.x0 + box.x1) / 2 / ratio, chart._canvas.height() - (box.y0 + box.y1) / 2 / ratio
    )

    QApplication.sendEvent(chart._canvas, QWheelEvent(
        where, chart._canvas.mapToGlobal(where), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    ))

    low, high = metric_axes.get_xlim()
    assert high - low < before[1] - before[0]

