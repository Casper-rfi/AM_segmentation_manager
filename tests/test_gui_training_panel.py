"""Tests for the Train tab's widgets: what they show of the controller, and what they ask first.

The controller is checked without a window in ``test_training_controller``; here it is
the wiring -- that the form says what is missing, that nothing is written or deleted
until a dialog has been answered, and that the tab reads the runs again only while it
is showing and one is live. Dialogs are answered by the test rather than opened.
"""
from __future__ import annotations

import sys
import time
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
    monkeypatch.setattr(training_run, "label_report", lambda path, classes: (None, None))
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


def _select(panel: TrainingPanel, epoch: int) -> None:
    """Show *epoch* in the row of checkpoints, as choosing it from the list does."""
    index = panel.epoch_box.findData(epoch)
    assert index >= 0, f"epoch {epoch} is not listed"
    panel.epoch_box.setCurrentIndex(index)


def _epochs(panel: TrainingPanel) -> list[str]:
    return [panel.epoch_box.itemText(row) for row in range(panel.epoch_box.count())]


def _epoch_tip(panel: TrainingPanel, epoch: int) -> str:
    return panel.epoch_box.itemData(panel.epoch_box.findData(epoch), 3)  # Qt.ToolTipRole


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
    assert "Choose the weights to refine" in panel.problems_label.text()
    assert not panel.start_button.isEnabled()
    assert panel.max_epochs.value() == 50 and panel.initial_rate.text() == "2e-06"


def test_start_says_on_hover_why_it_is_greyed_and_otherwise_what_it_does(panel):
    assert "Check the training set" in panel.start_button.toolTip()

    panel._mode_chosen("refine")

    assert panel.start_button.toolTip() == panel.problems_label.text() != ""


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
        assert "not installed" in made.start_button.toolTip()
        assert "--extra train" not in made.start_button.toolTip(), "said once, where it can be read"
    finally:
        destroy(made)


def _turn_wheel_over(widget):
    """Send *widget* one turn of the wheel, and return the event as it came back."""
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication

    where = QPointF(widget.rect().center())
    turn = QWheelEvent(
        where, widget.mapToGlobal(where), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, turn)
    return turn


def test_the_wheel_changes_no_setting_of_the_form_it_scrolls(panel, qt_app):
    """A turn meant for the form changed whichever setting the pointer was passing over."""
    from am_segmentation_manager.gui.widgets.controls import leave_the_wheel_to_scrolling

    leave_the_wheel_to_scrolling(qt_app)
    panel.resize(1440, 500)
    panel.show()
    panel.advanced_check.setChecked(True)
    qt_app.processEvents()
    panel.device_box.showPopup()
    panel.device_box.hidePopup()
    before = panel.controller.settings
    device = panel.device_box.currentIndex()
    assert panel.device_box.count() > 1

    for box in (panel.max_epochs, panel.frozen_epochs, panel.batch, panel.validation_batch, panel.seed):
        box.setFocus()
        # Qt offers a turn to the parent only where the control ignored it.
        assert not _turn_wheel_over(box).isAccepted()
    panel.device_box.setFocus()
    _turn_wheel_over(panel.device_box)

    assert panel.controller.settings == before
    assert panel.device_box.currentIndex() == device


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


def test_the_settings_go_back_to_mine_or_to_the_recommended_ones(panel):
    from dataclasses import replace

    from am_segmentation_manager.operations.training_run import NEW_MODEL

    panel.controller._defaults = lambda mode: replace(NEW_MODEL, max_epochs=200)
    panel.max_epochs.setValue(12)

    panel.my_defaults_button.click()
    assert panel.max_epochs.value() == 200

    panel.recommended_button.click()
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


# -- the memory a run needs -------------------------------------------------------


def test_the_form_says_what_the_batch_takes_of_a_gpu_as_it_is_changed(panel):
    assert panel.memory_label.text() == "about 43 GB of GPU memory"

    panel.batch.setValue(2)

    assert panel.memory_label.text() == "about 15 GB of GPU memory"


def test_start_says_where_the_gpu_is_too_small_before_asking(panel):
    from am_segmentation_manager.ml import gpus

    card = gpus.Gpu(0, 0, "GPU-0", "NVIDIA GeForce RTX 5060 Ti", 16 * 1024, 16 * 1024, "0")
    panel.controller._gpus = lambda: gpus.GpuInventory(gpus=(card,))
    panel.answers.go = False

    panel.start()

    title, said = panel.answers.asked[-1]
    assert title == "Start training"
    assert "Batches of 6 need about 43 GB" in said and "largest batch likely to fit is 1" in said


# -- the log --------------------------------------------------------------------


def _log(panel: TrainingPanel) -> str:
    return panel.log_view.toPlainText()


def test_a_run_that_has_just_started_shows_what_its_log_says(panel):
    """Until its first epoch ends there are no curves, and nothing else says it is alive."""
    panel.how["training"] = "sleep"
    panel.start()
    run = panel.controller.selected
    deadline = time.monotonic() + 30
    while "training" not in (run / "train.log").read_text() and time.monotonic() < deadline:
        time.sleep(0.05)

    panel.refresh()

    assert f"training {run.name}" in _log(panel)
    assert "training" in panel.standing_label.text()
    assert "log below" in panel.chart.message.text()


def test_the_log_of_a_live_run_is_read_again_between_the_reads_of_the_runs(panel, qt_app):
    panel.how["training"] = "sleep"
    panel.start()
    panel.refresh()
    panel.show()
    qt_app.processEvents()
    with (panel.controller.selected / "train.log").open("a") as log:
        log.write("Epoch 1 training:  50%|#####     | 5/10 [00:05<00:05,  1.00step/s]\n")

    panel.poll_log()

    assert _log(panel).splitlines()[-1].startswith("Epoch 1 training:  50%")


def test_the_log_of_a_run_that_has_ended_is_not_read_again(panel, qt_app):
    run = _started(panel)
    panel.show()
    qt_app.processEvents()
    before = _log(panel)
    with (run / "train.log").open("a") as log:
        log.write("written by something else\n")

    panel.poll_log()

    assert _log(panel) == before


def test_nothing_is_read_while_the_tab_is_hidden(panel, monkeypatch):
    panel.how["training"] = "sleep"
    panel.start()
    panel.refresh()
    read: list[int] = []
    monkeypatch.setattr(panel.controller, "log_tail", lambda run: read.append(1) or [])

    panel.poll_log()

    assert read == []


def test_a_run_that_failed_says_why_and_shows_where_in_its_log(panel):
    panel.how["training"] = "fail"
    _started(panel)

    assert "failed: RuntimeError('out of GPU memory')" in panel.standing_label.text()
    assert "RuntimeError: out of GPU memory" in _log(panel)
    assert not panel.advice_label.text()


def test_a_run_that_ran_out_of_gpu_memory_is_told_what_to_do(panel):
    panel.how["training"] = "fail"
    run = _started(panel)
    process = run / "am_training_process.yaml"
    process.write_text(
        process.read_text().replace("out of GPU memory", "CUDA out of memory"), encoding="utf-8"
    )

    panel.refresh()

    assert "ran out of memory with batches of 6" in panel.advice_label.text()


def test_the_log_stays_where_it_was_scrolled_to_and_follows_its_end_otherwise(panel, qt_app):
    run = _started(panel)
    panel.resize(1200, 800)
    panel.show()
    (run / "train.log").write_text("".join(f"line {n}\n" for n in range(150)), encoding="utf-8")
    panel.refresh()
    qt_app.processEvents()
    bar = panel.log_view.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()

    bar.setValue(3)
    with (run / "train.log").open("a") as log:
        log.write("one more\n")
    panel.refresh()

    assert bar.value() == 3


def test_a_run_am_seg_did_not_start_has_no_log_to_show_or_open(drawn):
    assert _log(drawn) == ""
    assert not drawn.log_button.isEnabled()
    assert "Not started by am-seg" in drawn.standing_label.text()


def test_the_log_shown_is_a_validation_s_while_one_is_scoring():
    from am_segmentation_manager.gui.training_controller import RunSummary, ValidationSummary
    from am_segmentation_manager.operations.training_process import RunStatus

    run, scoring = Path("runs/a"), Path("runs/a/validations/arctis")
    ended = RunSummary(run, "a", status=RunStatus("finished"))
    validating = RunSummary(
        run, "a", status=RunStatus("finished"),
        validations=(ValidationSummary(scoring, "arctis", status=RunStatus("running")),),
    )

    assert ended.logged_in == run
    assert validating.logged_in == scoring


# -- predictions ----------------------------------------------------------------


def test_predictions_takes_the_chart_s_place_and_looks_at_the_runs_ticked(panel, qt_app):
    run = _with_checkpoints(panel)
    panel.resize(1440, 840)
    panel.show()
    qt_app.processEvents()
    panel._run_ticked(0, True)

    panel.view_control.chosen.emit("predictions")

    assert panel.pages.currentWidget() is panel.predictions
    assert not panel.metric_box.isVisible() and not panel.validated_box.isVisible()
    assert [model.name for model in panel.predictions.controller.models] == [run.name]
    assert panel.checkpoints_row.isVisible(), "the run looked at still has its checkpoints"

    panel.view_control.chosen.emit("curves")

    assert panel.pages.currentWidget() is panel.chart and panel.metric_box.isVisible()


def test_a_run_unticked_leaves_the_predictions(panel, qt_app):
    _with_checkpoints(panel)
    panel._run_ticked(0, True)
    panel.view_control.chosen.emit("predictions")

    panel._run_ticked(0, False)

    assert panel.predictions.controller.models == []
    assert panel.predictions.status.text().replace("\u200b", "") == "Tick the runs to compare."


def test_the_runs_read_again_do_not_rebuild_the_pictures(panel, monkeypatch):
    """A rebuild throws away where every picture was zoomed to, and the runs are read every ten seconds."""
    _with_checkpoints(panel)
    panel._run_ticked(0, True)
    panel.view_control.chosen.emit("predictions")
    built: list[int] = []
    monkeypatch.setattr(panel.predictions, "_build_grid", lambda: built.append(1))

    panel.refresh()
    panel.refresh()

    assert built == []


def test_predictions_are_made_on_the_device_the_form_names(panel):
    panel.controller.device = "cuda:1"

    assert panel.predictions.device() == "cuda:1"


# -- checkpoints --------------------------------------------------------------


def test_checkpoints_are_listed_with_the_best_marked(panel):
    _with_checkpoints(panel)

    listed = [caption.split(" ")[0] for caption in _epochs(panel)]

    assert listed == ["1", "2", "3"]
    assert all(caption.endswith("(incomplete)") for caption in _epochs(panel))
    assert _epoch_tip(panel, 3) == "Epoch 3: run's best, incomplete"


def test_each_epoch_is_listed_with_the_score_the_run_ranks_it_by(panel):
    run = _with_checkpoints(panel)
    written = panel.controller.summary(run).written
    score = written.val[1].score(written.key_metrics)

    assert _epochs(panel)[1] == f"2 · {score:.3f} (incomplete)"


def test_the_row_opens_on_the_run_s_best_and_then_on_the_epoch_marked(panel):
    run = _with_checkpoints(panel)
    assert panel.epoch_box.currentData() == 3

    panel.controller.choose_best(run, 1)
    panel._run_chosen(0)

    assert panel.epoch_box.currentData() == 1


def test_the_epoch_shown_stays_shown_when_the_runs_are_read_again(panel):
    _with_checkpoints(panel)
    _select(panel, 2)

    panel.poll()
    panel.refresh()

    assert panel.epoch_box.currentData() == 2


def test_a_copy_is_offered_for_the_epoch_shown(panel):
    assert not panel.copy_button.isEnabled()

    _with_checkpoints(panel)

    assert panel.copy_button.isEnabled()


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


def test_deleting_the_others_keeps_the_epoch_shown(panel):
    run = _with_checkpoints(panel)
    _select(panel, 1)

    panel.delete_others()

    assert sorted(path.name[-12:] for path in run.rglob("*.pth")) == ["epoch001.pth", "epoch003.pth"]
    assert panel.epoch_box.count() == 2
    assert panel.epoch_box.currentData() == 1


def test_delete_is_not_offered_where_one_checkpoint_is_all_there_is(panel):
    _with_checkpoints(panel, epochs=(2,))

    assert not panel.delete_button.isEnabled()


def test_delete_is_offered_for_a_run_that_has_ended_with_more_than_one(panel):
    _with_checkpoints(panel)

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

    listed = panel.runs_tree.project_item(0)
    assert (listed.text(0), listed.text(1)) == ("wide_seed2", "2/90")
    assert "Not started by am-seg" in listed.toolTip(0)
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

    assert [folder.child(row).text(0) for row in range(2)] == ["fold1", "fold2"]
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
    shown = panel.controller.selected

    panel.runs_tree.project_item(2).setCheckState(0, _checked())
    assert panel.controller.selected == shown

    panel.runs_tree.setCurrentItem(panel.runs_tree.project_item(2))
    assert panel.controller.selected_run().name == "alone"


def test_a_run_s_row_gives_the_path_to_copy_or_reveal(panel):
    _three_runs(panel)

    assert panel.run_path(2) == panel.controller.training_directory / "alone" / "training_260101_000000"
    assert panel.run_path(7) is None


def _checked():
    from qtpy.QtCore import Qt

    return Qt.CheckState.Checked


# -- what a run still holds ---------------------------------------------------


def test_a_run_counts_the_weights_still_on_disk_of_those_it_saved(panel):
    run = _with_checkpoints(panel, epochs=(1, 2, 3, 4))
    weights = panel.runs_tree.columnCount() - 1
    assert panel.runs_tree.headerItem().text(weights) == "Weights"
    assert panel.runs_tree.project_item(0).text(weights) == "4/4"

    _select(panel, 4)
    panel.delete_others()

    assert panel.controller.summary(run).cells()[-1] == "1/4"
    assert panel.runs_tree.project_item(0).text(weights) == "1/4"


def test_a_run_that_saved_no_weights_counts_none(panel):
    _started(panel)
    panel.refresh()

    assert panel.controller.selected_run().cells()[-1] == ""


def test_each_run_says_what_it_holds_in_words_on_hover(panel):
    run = _with_checkpoints(panel)

    item = panel.runs_tree.project_item(0)

    assert item.toolTip(0).startswith(
        f"{run}\n3 of at most 100 epochs trained\n3 of 3 saved weights on disk, 0.0 GB\n"
    )
    assert item.toolTip(0).endswith(panel.controller.selected_run().details())


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


def test_a_box_is_offered_for_each_class_of_the_run_all_but_background_and_vacuum_ticked(drawn):
    assert list(drawn.class_boxes) == ["background", "gis", "lamella", "crack", "vacuum"]
    unticked = [name for name, box in drawn.class_boxes.items() if not box.isChecked()]
    assert unticked == ["background", "vacuum"]
    assert "crack" in _curves(drawn) and not {"background", "vacuum"} & _curves(drawn)

    drawn.class_boxes["background"].setChecked(True)
    assert "background" in _curves(drawn)


def test_unticking_a_class_takes_its_curve_out_and_leaves_the_averages(drawn):
    drawn.class_boxes["crack"].setChecked(False)

    curves = _curves(drawn)

    assert "crack" not in curves and "gis" in curves
    assert "weighted average, validation" in curves


def test_a_class_unticked_is_still_unticked_when_the_runs_are_read_again(drawn):
    drawn.class_boxes["lamella"].setChecked(False)

    drawn.refresh()

    assert not drawn.class_boxes["lamella"].isChecked()
    assert "lamella" not in _curves(drawn)


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

    chart.hold_space(True)
    chart.point_at(x, y)

    assert chart.hovered() == ""


# -- the epoch marked as best -------------------------------------------------


def test_marking_the_selected_epoch_writes_it_beside_the_run_and_says_so(panel):
    from am_segmentation_manager.operations.training_choice import read_chosen_best

    run = _with_checkpoints(panel)
    _select(panel, 2)

    panel.mark_best()

    assert read_chosen_best(run).epoch == 2
    assert panel.epoch_box.currentData() == 2
    assert _epoch_tip(panel, 2) == "Epoch 2: marked best, incomplete"
    assert "Marked as best: epoch 2" in panel.runs_tree.project_item(0).toolTip(0)
    assert panel.runs_tree.project_item(0).text(4) == "2"


def test_the_button_takes_the_mark_back_while_the_epoch_shown_is_the_one_marked(panel):
    run = _with_checkpoints(panel)
    _select(panel, 2)
    assert panel.mark_button.text() == "Mark as best"

    panel.mark_button.click()
    assert panel.mark_button.text() == "Unmark"
    _select(panel, 1)
    assert panel.mark_button.text() == "Mark as best"

    _select(panel, 2)
    panel.mark_button.click()

    assert panel.controller.summary(run).chosen is None
    assert panel.mark_button.text() == "Mark as best"


def test_a_run_whose_weights_are_elsewhere_lists_its_validated_epochs_to_mark(panel):
    run = _started(panel)
    _write_epochs(run, 3, run_id=f"{run.name}_val")
    panel.refresh()

    assert [caption.split(" ")[0] for caption in _epochs(panel)] == ["1", "2", "3"]
    assert all(caption.endswith("(no weights here)") for caption in _epochs(panel))
    assert not panel.copy_button.isEnabled() and not panel.delete_button.isEnabled()

    _select(panel, 2)
    panel.mark_button.click()

    assert panel.controller.summary(run).chosen.epoch == 2


def test_the_epoch_marked_is_listed_once_its_weights_have_gone(panel):
    run = _with_checkpoints(panel)
    panel.controller.choose_best(run, 1)
    next(run.rglob("*_epoch001.pth")).unlink()

    panel.refresh()

    assert _epochs(panel)[0].endswith("(no weights here)")
    _select(panel, 1)
    assert panel.mark_button.text() == "Unmark"
    assert not panel.copy_button.isEnabled()


def test_clearing_takes_the_mark_back_and_leaves_the_run_s_own_best(panel):
    run = _with_checkpoints(panel)
    _select(panel, 2)
    panel.mark_best()

    panel.clear_best()

    assert panel.controller.summary(run).chosen is None
    assert _epoch_tip(panel, 3) == "Epoch 3: run's best, incomplete"
    assert panel.mark_button.text() == "Mark as best"


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


def test_a_run_still_training_has_its_frozen_epochs_shaded_from_what_it_was_asked(panel):
    """Its own files say how long the encoder was frozen only once it has ended."""
    pytest.importorskip("matplotlib")
    panel.controller.change(frozen_epochs=2)
    run = _started(panel)
    _write_epochs(run, 3, run_id=f"{run.name}_val")

    panel.refresh()

    for axes in panel.chart._figure.axes:
        assert [patch.get_width() for patch in axes.patches if patch.get_gid() == "frozen"] == [2]


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
    """Every line written on the table: a set's heading is its name over its images."""
    axes = panel.chart._figure.axes[0]
    return [line for text in axes.texts for line in text.get_text().split("\n")]


def test_the_table_asks_for_runs_until_one_is_ticked(tabled):
    assert tabled.chart.message.text() == "Tick the runs to compare."
    assert tabled.table_row.isVisibleTo(tabled)

    tabled.runs_tree.project_item(0).setCheckState(0, _checked())

    assert "gis" in _table_text(tabled)


def test_the_table_holds_the_ticked_runs_and_the_classes_ticked(tabled):
    tabled._tick_all(True)
    axes = tabled.chart._figure.axes[0]
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
    axes = tabled.chart._figure.axes[0]
    assert axes.get_yticklabels()[0].get_text().endswith("2-3 (2)")


def test_the_table_is_exported_as_a_csv_and_only_from_the_table(tabled, tmp_path: Path):
    tabled._tick_all(True)
    destination = tmp_path / "table.csv"
    tabled._ask_table_destination = lambda: destination

    tabled.export_table()

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "run,epochs,gis iou,lamella iou,crack iou"
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
    assert "Validations: held finished, 3 of 3 epoch(s) scored." in validated.runs_tree.currentItem().toolTip(0)
    assert [validated.validated_box.itemData(i) for i in range(3)] == ["", "held", "*"]
    assert not validated.split_box.isEnabled()


def test_choosing_a_validation_draws_the_curves_from_it(validated):
    _choose(validated.validated_box, "held")

    _, metric_axes = validated.chart._figure.axes
    assert metric_axes.get_ylabel().endswith("from held")
    assert validated.split_box.isEnabled()

    offered = [validated.split_box.itemText(i) for i in range(validated.split_box.count())]
    assert offered == ["nothing", "Training split", "Split", "Other", "Instrument"]
    _choose(validated.split_box, "instrument")
    assert [validated.subset_box.itemData(i) for i in range(2)] == ["Arctis", "Hydra"]
    assert [validated.subset_box.itemText(i) for i in range(2)] == ["Arctis (1)", "Hydra (1)"]
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
    assert {"Arctis", "Hydra", "n=1"} <= set(_table_text(validated))

    validated.set_boxes["Arctis"].setChecked(False)
    assert "Arctis" not in _table_text(validated)


def test_a_split_that_cannot_be_made_says_what_would_allow_it(validated):
    from am_segmentation_manager.gui.widgets.training_panel import SPLIT_HINT

    assert not validated.split_box.isEnabled()
    assert "use Score own list" in validated.split_box.toolTip()

    _choose(validated.validated_box, "*")
    assert validated.split_box.toolTip() == "Choose each run's own list, or one validation, to split it."

    _choose(validated.validated_box, "held")
    assert validated.split_box.isEnabled() and validated.split_box.toolTip() == SPLIT_HINT


def test_scoring_the_own_list_says_why_it_cannot_be_done(validated):
    """The run listed is one AM-model-training wrote, with no record of what it validated on."""
    assert not validated.score_own_button.isEnabled()
    assert validated.score_own_button.toolTip() == "Only a run am-seg prepared says what it validated on."


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
    assert "second prepared, 0 of 3 epoch(s) scored" in validated.runs_tree.currentItem().toolTip(0)


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


def test_the_dialog_calls_the_validation_what_the_export_is_and_asks_no_name(validated, tmp_path: Path):
    from am_segmentation_manager.gui.widgets.training_panel import ValidationDialog

    run = validated.controller.selected_run()
    dialog = ValidationDialog(
        validated, run, tmp_path / "exports" / "hydra_manifest.yaml", lambda manifest: "all"
    )
    assert dialog.check_button.isEnabled()
    assert dialog.asked()[1:] == ("hydra", "all")
    assert dialog.name_label.text() == "It will be called hydra, after the export."
    dialog.deleteLater()

    again = ValidationDialog(validated, run, tmp_path / "exports" / "held_manifest.yaml")
    assert again.asked()[1:] == ("held_2", "validation")
    assert "already has a validation called held" in again.name_label.text()
    again.deleteLater()


def test_the_dialog_waits_for_an_export(validated):
    from am_segmentation_manager.gui.widgets.training_panel import ValidationDialog

    dialog = ValidationDialog(validated, validated.controller.selected_run(), None)
    assert not dialog.check_button.isEnabled()
    assert dialog.name_label.text() == "Choose a training export."
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

    chart.wheel((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, -3)
    assert metric_axes.get_xlim() == pytest.approx(before), "never out beyond what Reset view gives"


def test_space_pans_for_as_long_as_it_is_held(drawn, qt_app):
    """As on the Label canvas; a repeat of the held key is not a second press."""
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication

    chart = drawn.chart

    def space(kind, repeat=False):
        event = QKeyEvent(kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", repeat)
        QApplication.sendEvent(chart._canvas, event)

    space(QEvent.Type.KeyPress)
    space(QEvent.Type.KeyPress, repeat=True)
    assert chart._toolbar.mode == "pan/zoom"

    space(QEvent.Type.KeyRelease)
    assert not chart._toolbar.mode

    space(QEvent.Type.KeyRelease)
    assert not chart._toolbar.mode, "a release with nothing held starts no panning"


def test_space_is_let_go_of_when_the_keyboard_leaves_the_chart(drawn):
    """A release that goes to another window would leave the chart panning."""
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QFocusEvent
    from qtpy.QtWidgets import QApplication

    chart = drawn.chart
    chart.hold_space(True)

    QApplication.sendEvent(chart._canvas, QFocusEvent(QEvent.Type.FocusOut))

    assert not chart._toolbar.mode


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


# -- the table's colours ------------------------------------------------------


def _move_colours(tabled, low: int, high: int) -> None:
    """Move the slider's handles and let them rest, as a drag that has ended."""
    tabled.colour_slider.set_range(low, high)
    tabled._colour_timer.timeout.emit()


def test_the_table_s_colours_follow_it_until_a_value_is_set_by_hand(tabled):
    tabled._tick_all(True)
    controller = tabled.controller
    assert tabled.auto_colours.isChecked()
    followed = controller.colour_range
    assert tuple(value / 100 for value in tabled.colour_slider.values()) == pytest.approx(followed)
    assert (tabled.colour_low_label.text(), tabled.colour_high_label.text()) == (
        f"{followed[0]:.2f}", f"{followed[1]:.2f}",
    )

    moved = []
    for box in tabled.class_boxes.values():
        if box.isChecked():
            box.setChecked(False)
            moved.append(controller.colour_range != followed)
            box.setChecked(True)
    assert any(moved), "a class left out, and the range moved with it"
    assert controller.colour_range == followed

    _move_colours(tabled, 50, 90)
    assert not tabled.auto_colours.isChecked() and not controller.auto_colours
    assert controller.colour_range == (0.5, 0.9)
    assert (tabled.colour_low_label.text(), tabled.colour_high_label.text()) == ("0.50", "0.90")
    _, bar = tabled.chart._figure.axes
    assert bar.get_ylim() == pytest.approx((0.5, 0.9))

    tabled.class_boxes["gis"].setChecked(False)
    assert controller.colour_range == (0.5, 0.9), "set by hand, so it stays"


def test_auto_switched_on_fits_the_colours_and_keeps_them_fitted(tabled):
    tabled._tick_all(True)
    fitted = tabled.controller.colour_range
    _move_colours(tabled, 0, 100)
    assert tabled.controller.colour_range == (0.0, 1.0)

    tabled.auto_colours.setChecked(True)

    assert tabled.controller.colour_range == fitted
    assert tuple(value / 100 for value in tabled.colour_slider.values()) == pytest.approx(fitted)
    _, bar = tabled.chart._figure.axes
    assert bar.get_ylim() == pytest.approx(fitted)


def test_the_table_is_drawn_once_a_handle_rests_and_not_at_each_step_of_a_drag(tabled, monkeypatch):
    """Every table is a figure built anew, and a drag reports each hundredth it crosses."""
    tabled._tick_all(True)
    drawn = []
    monkeypatch.setattr(tabled.chart, "show_chart", lambda build, **_: drawn.append(1))

    for low in range(10, 40):
        tabled.colour_slider.set_range(low, 90)
    assert drawn == [] and tabled.controller.colour_range == (0.39, 0.9)
    assert tabled.colour_low_label.text() == "0.39", "the numbers follow the handle as it moves"

    tabled._colour_timer.timeout.emit()
    assert drawn == [1]


def test_the_slider_between_the_two_values_is_the_table_s_own_ramp(tabled):
    from am_segmentation_manager.plots import comparison_ramp

    slider = tabled.colour_slider

    assert list(slider._ramp) == comparison_ramp(len(slider._ramp)) and len(slider._ramp) > 1


# -- the list of runs in columns ----------------------------------------------


def test_a_run_is_listed_a_column_to_each_thing_said_of_it(panel):
    """One line of them had to be read through to find the one wanted."""
    from am_segmentation_manager.gui.training_controller import RUN_COLUMNS
    from test_training_controller import _bare_run

    _bare_run(panel.controller.training_directory, "wide_seed2")
    panel.refresh()
    tree = panel.runs_tree
    row = tree.project_item(0)

    assert not tree.isHeaderHidden()
    assert [tree.headerItem().text(column) for column in range(tree.columnCount())] == [
        caption for caption, _hint in RUN_COLUMNS
    ]
    assert [row.text(column) for column in range(tree.columnCount())] == [
        "wide_seed2", "2/90", "2", "0.200", "", "0/2",
    ]
    assert "IoU and Dice" in tree.headerItem().toolTip(3), "the number says what it is of"


def _pixels(icon_or_pixmap) -> bytes:
    """Return the pixels of a mark, so two can be told apart."""
    from qtpy.QtGui import QIcon

    from am_segmentation_manager.gui.widgets.training_panel import RUN_MARK

    pixmap = icon_or_pixmap.pixmap(RUN_MARK, RUN_MARK) if isinstance(icon_or_pixmap, QIcon) else icon_or_pixmap
    image = pixmap.toImage()
    return bytes(image.constBits())[: image.sizeInBytes()]


def test_every_way_a_run_can_stand_has_a_mark_of_its_own(qt_app):
    """Two greens on one bar were told apart by shade alone."""
    from typing import get_args

    from am_segmentation_manager.gui.widgets.training_panel import RUN_MARKS, run_mark
    from am_segmentation_manager.operations.training_process import State

    assert set(RUN_MARKS) == set(get_args(State))
    drawn = {state: _pixels(run_mark(state)) for state in RUN_MARKS}
    assert len(set(drawn.values())) == len(drawn)
    shapes = [shape for shape, _colour, _meaning in RUN_MARKS.values()]
    assert len(set(shapes)) == len(shapes), "none is told from another by its colour alone"


def test_a_run_am_seg_did_not_start_is_marked_as_one_it_did(qt_app):
    from am_segmentation_manager.gui.widgets.training_panel import run_mark

    assert _pixels(run_mark("training")) == _pixels(run_mark("running"))
    assert _pixels(run_mark("not finished")) == _pixels(run_mark("interrupted"))
    assert run_mark("").isNull(), "a run that cannot be read has only its problem to say"


def test_a_run_s_mark_says_how_it_stands_and_hover_says_why(panel):
    """A column of words said it before, in one colour but for the run that failed."""
    from am_segmentation_manager.gui.widgets.training_panel import run_mark

    _started(panel)
    panel.how["training"] = "fail"
    _started(panel)
    panel.refresh()

    marks = {
        run.state: (_pixels(item.icon(0)), item.toolTip(0))
        for run, item in zip(panel.controller.runs(), panel.runs_tree.project_items(), strict=True)
    }

    assert set(marks) == {"finished", "failed"}
    assert marks["failed"][0] == _pixels(run_mark("failed"))
    assert marks["finished"][0] == _pixels(run_mark("finished"))
    assert "out of GPU memory" in marks["failed"][1]
    assert "State" not in [
        panel.runs_tree.headerItem().text(column) for column in range(panel.runs_tree.columnCount())
    ]


def test_the_key_list_names_each_mark_in_a_few_words(qt_app):
    """A paragraph to each of five rows was not read."""
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_ROW_MARK, TAB_TRAIN, _tab_marks

    meanings = [meaning for group, _, meaning in _tab_marks(TAB_TRAIN) if group == GROUP_ROW_MARK]

    assert meanings == [
        "Not started", "Training", "Finished", "Stopped by you",
        "Interrupted", "Failed", "Started on another computer",
    ]


def test_a_folder_and_a_run_that_cannot_be_read_are_written_across_the_columns(panel):
    """Held to the name's column, a reason would be cut to its first words."""
    _three_runs(panel)
    broken = panel.controller.training_directory / "broken"
    broken.mkdir()
    (broken / training_run.RUN_RECORD_FILENAME).write_text("version: 99\n", encoding="utf-8")
    panel.refresh()
    tree = panel.runs_tree

    unread = next(item for item in tree.project_items() if "later version" in item.text(0))

    try:
        assert unread.isFirstColumnSpanned()
        assert _folder(panel, "step8").isFirstColumnSpanned()
        assert not _folder(panel, "step8").child(0).isFirstColumnSpanned()
    finally:
        # The panel stops every run when it is done with, and this one cannot be asked.
        (broken / training_run.RUN_RECORD_FILENAME).unlink()
        broken.rmdir()


# -- the sections ---------------------------------------------------------------


def test_each_button_sits_in_the_box_of_what_it_acts_on(panel):
    """Nine in one row under the checkpoints, four of them were about the run."""

    from qtpy.QtWidgets import QWidget

    def inside(box):
        return {
            name for name, widget in vars(panel).items()
            if isinstance(widget, QWidget) and widget is not box and box.isAncestorOf(widget)
        }

    runs, plots, checkpoints = panel.runs_box, panel.plots_box, panel.checkpoints_row

    assert [box.title() for box in (runs, plots)] == ["Runs", "Curves, table and predictions"]
    assert plots.isAncestorOf(checkpoints)
    assert {"runs_tree", "validate_button", "stop_validating_button", "stop_button"} <= inside(runs)
    assert inside(panel.log_box) == {"standing_label", "advice_label", "log_button", "log_view"}
    assert {"chart", "metric_box", "validated_box", "value_box"} <= inside(plots)
    assert inside(checkpoints) == {
        "epoch_box", "copy_button", "mark_button", "delete_button", "keep_best_check",
    }


def test_what_is_done_to_a_run_wraps_and_does_not_widen_the_runs(panel, qt_app):
    """In a row that cannot wrap, one button more took the tab past the smallest screen."""
    before = panel.runs_box.minimumSizeHint().width()
    watching = (panel.score_own_button, panel.validate_button, panel.stop_validating_button)

    panel.validate_button.setText("Validate on another training export altogether...")
    qt_app.processEvents()

    assert panel.runs_box.minimumSizeHint().width() == before
    assert before < sum(button.sizeHint().width() for button in watching)

    panel.runs_box.setFixedWidth(before)
    panel.show()
    qt_app.processEvents()
    area = panel.run_actions

    assert all(area.rect().contains(button.geometry()) for button in watching)
    assert len({button.y() for button in watching}) > 1, "they wrapped"


def test_the_runs_stand_above_the_form_beside_what_they_show(panel, qt_app):
    """Across the top of the right they took the height the curves wanted."""
    panel.resize(1440, 840)
    panel.show()
    qt_app.processEvents()

    def at(widget):
        return widget.mapTo(panel, widget.rect().topLeft())

    directory, runs, form = panel.directory_button, panel.runs_box, panel.start_button
    assert at(directory).y() < at(runs).y() < at(form).y()
    assert not runs.isAncestorOf(directory), "it is where a new run is written as well"
    assert at(runs).x() < at(panel.plots_box).x()
    assert at(panel.chart).y() < at(panel.checkpoints_row).y()
    assert at(panel.plots_box).y() <= at(runs).y()


def test_a_run_s_dated_name_is_whole_at_the_width_the_column_opens_at(panel, qt_app):
    """The name takes what the other columns leave, and is cut short where that is too little."""
    panel.resize(1440, 840)
    panel.show()
    tree = panel.runs_tree
    tree.set_projects(
        ["", "sweeps"], ["20261009_refine_01", "20261009_refine_02"],
        cells=[("100/100", "100", "0.845", "100", "100/100")] * 2,
    )
    tree.expandAll()
    qt_app.processEvents()

    assert tree.header().sectionSize(0) >= tree.sizeHintForColumn(0)
    assert not tree.horizontalScrollBar().isVisible()


def test_a_class_s_switch_is_on_in_the_colour_of_its_curve(panel):
    """A swatch beside a tick box said the same thing in twice the room."""
    from am_segmentation_manager.gui.widgets.controls import ToggleSwitch
    from am_segmentation_manager.plots import class_colour

    _with_checkpoints(panel)

    boxes = list(panel.class_boxes.values())
    assert all(isinstance(box, ToggleSwitch) and box.icon().isNull() for box in boxes)
    assert [box._track_colour() for box in boxes if box.isChecked()] == [
        class_colour(index) for index, box in enumerate(boxes) if box.isChecked()
    ]


def test_the_lists_above_the_chart_are_stacked_in_one_column(panel, qt_app):
    """Each in a row of its own started wherever its caption ended."""
    _with_checkpoints(panel)
    panel.view_control.set_current("table")
    panel._view_chosen("table")
    panel.resize(1400, 900)
    panel.show()
    qt_app.processEvents()
    panel.controller.validated_on = ""
    panel._show_sets()  # refilled, a list sized to its contents takes its own width
    for _ in range(3):
        qt_app.processEvents()

    stacked = [panel.metric_box, panel.validated_box, panel.value_box]

    assert len({(box.x(), box.width()) for box in stacked}) == 1
    assert [box.y() for box in stacked] == sorted(box.y() for box in stacked)
    panel.hide()
