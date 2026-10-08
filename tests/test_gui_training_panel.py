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


def test_published_settings_puts_every_setting_back(panel):
    panel.max_epochs.setValue(12)

    panel._reset_settings()

    assert panel.max_epochs.value() == 100


# -- starting -----------------------------------------------------------------


def test_starting_says_what_will_be_done_and_asks(panel):
    run = _started(panel)

    title, said = panel.answers.asked[0]
    assert title == "Start training"
    assert "1 training and 1 validation pair(s)" in said
    assert "goes on after this window is closed" in said
    assert panel.controller.selected_run().status.state == "finished"
    assert run.name in panel.runs_list.item(0).text()


def test_declining_writes_and_starts_nothing(panel):
    panel.answers.go = False

    panel.start()

    assert not panel.controller.training_directory.exists()
    assert panel.runs_list.count() == 0


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
    monkeypatch.setattr(panel, "_show_runs", lambda: reads.append(1))

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

    assert listed == ["epoch 1 (incomplete)", "epoch 2 (incomplete)", "epoch 3 (best, incomplete)"]


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
    assert "Delete 2 checkpoint(s)" in said and "keeping epoch(s) 2" in said
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
