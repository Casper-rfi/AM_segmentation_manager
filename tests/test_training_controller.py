"""Tests for what the Train tab knows, without a window.

The controller holds no Qt, so everything the tab can show or ask for is checked
here: what stops a run being started, what the list of runs says, and that a curve
does not vanish when a metrics file is read mid-rewrite. Runs are started through
the stand-in child of ``test_training_process``.
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from am_segmentation_manager.gui.training_controller import TrainingController
from am_segmentation_manager.operations import training_process, training_run
from am_segmentation_manager.operations.training_run import NEW_MODEL, REFINE, TrainingRunRefused
from am_segmentation_manager.training_runs import Checkpoint
from test_training_process import CHILD, _settled, export_of_placeholders
from test_training_runs import _epoch


@pytest.fixture
def controller(tmp_path: Path, monkeypatch) -> TrainingController:
    """Return a controller set up to train a new model from a two-pair export."""
    monkeypatch.setattr(training_run, "label_problem", lambda path, classes: None)
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    how = {"training": "finish"}

    def start(directory: Path):
        return training_process.start_training(
            directory, [sys.executable, str(script), how["training"]]
        )

    made = TrainingController(stack_present=lambda: True, start=start)
    made.how = how  # type: ignore[attr-defined]
    made.manifest = export_of_placeholders(tmp_path)
    made.training_directory = tmp_path / "runs"
    made.name = "round01"
    return made


def _write_epochs(run: Path, epochs: int, run_id: str = "round01_val") -> Path:
    """Write what AM-model-training would have by *epochs*, and return its directory."""
    written = run / "training_261008_120000"
    written.mkdir(exist_ok=True)
    entries = [_epoch(n, 0.1 * n, 0.1 * n) for n in range(1, epochs + 1)]
    (written / f"{run_id}_train_metrics.json").write_text(json.dumps(entries))
    (written / f"{run_id}_val_metrics.json").write_text(json.dumps(entries))
    return written


# -- setting a run up ---------------------------------------------------------


def test_a_run_set_up_in_full_has_nothing_against_it(controller: TrainingController):
    assert controller.problems() == []


def test_what_is_missing_is_said_before_anything_is_read(tmp_path: Path):
    blank = TrainingController(stack_present=lambda: False)

    said = "\n".join(blank.problems())

    assert "training stack is not installed" in said
    assert "Choose a training set" in said
    assert "Choose the training directory" in said
    assert "Name the run" in said


def test_refining_asks_for_weights(controller: TrainingController):
    controller.set_mode("refine")

    assert "Choose the model weights to refine" in controller.problems()


def test_a_name_already_in_the_training_directory_is_refused(controller: TrainingController):
    (controller.training_directory / "round01").mkdir(parents=True)

    assert "A run named round01 already exists" in controller.problems()


def test_switching_mode_takes_that_mode_s_published_settings(controller: TrainingController):
    controller.change(max_epochs=7)

    controller.set_mode("refine")
    assert controller.settings == REFINE

    controller.set_mode("new")
    assert controller.settings == NEW_MODEL


def test_choosing_the_mode_already_chosen_keeps_what_was_changed(controller: TrainingController):
    controller.change(max_epochs=7)

    controller.set_mode("new")

    assert controller.settings.max_epochs == 7


def test_settings_can_be_put_back_to_the_published_ones(controller: TrainingController):
    controller.change(max_epochs=7, save_all=True)

    controller.reset_settings()

    assert controller.settings == NEW_MODEL


def test_settings_that_cannot_train_are_said(controller: TrainingController):
    controller.change(frozen_epochs=500)

    assert any("fewer epochs than are trained" in problem for problem in controller.problems())


def test_a_suggested_name_is_one_nothing_there_has(controller: TrainingController):
    today = datetime(2026, 10, 8)
    assert controller.suggest_name(today) == "20261008_new_01"

    (controller.training_directory / "20261008_new_01").mkdir(parents=True)
    assert controller.suggest_name(today) == "20261008_new_02"

    controller.set_mode("refine")
    assert controller.suggest_name(today) == "20261008_refine_01"


def test_the_weights_of_a_refine_are_not_given_to_a_new_model(
    controller: TrainingController, tmp_path: Path
):
    """Chosen while refining, then the mode switched back: a new model takes none."""
    controller.base_weights = tmp_path / "base.pth"
    controller.base_weights.write_bytes(b"weights")

    assert controller.plan().base_weights is None


def test_a_plan_that_cannot_be_made_gives_every_reason(controller: TrainingController):
    controller.name = "has space"
    controller.change(max_epochs=0)

    with pytest.raises(TrainingRunRefused) as refused:
        controller.plan()

    assert len(refused.value.reasons) >= 2


def test_planning_writes_nothing(controller: TrainingController):
    controller.plan()

    assert not controller.training_directory.exists()


# -- starting, listing and stopping -------------------------------------------


def test_starting_writes_the_run_starts_it_and_selects_it(controller: TrainingController):
    started = controller.start(controller.plan(), user="Ada")

    assert controller.selected == controller.training_directory / "round01"
    assert started.record.created_by == "Ada"
    assert _settled(controller.selected).state == "finished"
    assert controller.selected_run().status.state == "finished"


def test_runs_are_listed_newest_first_with_how_they_stand(controller: TrainingController):
    controller.start(controller.plan())
    _settled(controller.selected)
    controller.name = "round02"
    controller.how["training"] = "fail"
    controller.start(controller.plan())
    _settled(controller.selected)

    listed = controller.runs()

    # Prepared within one second of each other, the name orders them the same way.
    assert [run.name for run in listed] == ["round02", "round01"]
    assert {run.name: run.status.state for run in listed} == {
        "round01": "finished", "round02": "failed",
    }


def test_a_running_run_keeps_the_tab_reading_and_a_stop_ends_that(controller: TrainingController):
    controller.how["training"] = "sleep"
    controller.start(controller.plan())

    assert controller.any_live()
    assert "running" in controller.selected_run().line()

    stopped = controller.stop(controller.selected)

    assert stopped.status.state == "stopped"
    assert not controller.any_live()


def test_a_directory_that_is_not_a_run_is_not_listed(controller: TrainingController):
    (controller.training_directory / "notes").mkdir(parents=True)

    assert controller.runs() == []


def test_no_training_directory_is_no_runs():
    assert TrainingController(stack_present=lambda: True).runs() == []


def test_a_record_that_cannot_be_read_is_listed_with_why(controller: TrainingController):
    controller.start(controller.plan())
    _settled(controller.selected)
    (controller.selected / training_run.RUN_RECORD_FILENAME).write_text("version: 99\n")

    (listed,) = controller.runs()

    assert "later version" in listed.problem
    assert "later version" in listed.line()


# -- what a run has written ---------------------------------------------------


def test_a_run_s_line_says_how_far_it_got_and_its_best_epoch(controller: TrainingController):
    controller.start(controller.plan())
    _settled(controller.selected)
    _write_epochs(controller.selected, 3)

    run = controller.selected_run()

    assert run.epochs == 3
    assert run.best.epoch == 3
    assert run.line() == "round01 · finished · epoch 3 of at most 100 · best 3 (0.300)"


def test_a_metrics_file_caught_mid_rewrite_keeps_the_epochs_last_read(
    controller: TrainingController,
):
    """AM-model-training truncates and rewrites it after every epoch; a curve
    read at that moment must not drop to nothing and come back."""
    controller.start(controller.plan())
    _settled(controller.selected)
    written = _write_epochs(controller.selected, 3)
    assert controller.selected_run().epochs == 3

    (written / "round01_val_train_metrics.json").write_text("")
    (written / "round01_val_val_metrics.json").write_text('[{"epoch": 1, "lo')
    caught = controller.selected_run()

    assert caught.epochs == 3 and len(caught.written.val) == 3

    _write_epochs(controller.selected, 4)
    assert controller.selected_run().epochs == 4


def test_a_selected_run_that_has_gone_is_no_run(controller: TrainingController, tmp_path: Path):
    controller.selected = tmp_path / "runs" / "gone"

    assert controller.selected_run() is None


# -- taking a checkpoint out --------------------------------------------------


def _checkpoint(directory: Path, whole: bool = True) -> Checkpoint:
    path = directory / "round01_val_epoch003.pth"
    if whole:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("round01_val_epoch003/data.pkl", b"weights")
    else:
        path.write_bytes(b"PK\x03\x04 cut short")
    return Checkpoint(3, path)


def test_a_checkpoint_is_copied_out_and_the_run_keeps_its_own(
    controller: TrainingController, tmp_path: Path
):
    checkpoint = _checkpoint(tmp_path)

    copied = controller.copy_checkpoint(checkpoint, tmp_path / "models" / "AM_SEM_round01.pth")

    assert copied.read_bytes() == checkpoint.path.read_bytes()
    assert checkpoint.path.exists()


def test_a_checkpoint_is_never_copied_over_a_model(controller: TrainingController, tmp_path: Path):
    existing = tmp_path / "AM_SEM_All_V01.pth"
    existing.write_bytes(b"the published model")

    with pytest.raises(FileExistsError):
        controller.copy_checkpoint(_checkpoint(tmp_path), existing)

    assert existing.read_bytes() == b"the published model"


def test_a_checkpoint_cut_short_is_not_copied_out(controller: TrainingController, tmp_path: Path):
    with pytest.raises(ValueError, match="incomplete"):
        controller.copy_checkpoint(_checkpoint(tmp_path, whole=False), tmp_path / "out.pth")

    assert not (tmp_path / "out.pth").exists()


# -- deleting the checkpoints not wanted --------------------------------------


def _finished_with_checkpoints(controller: TrainingController, epochs=(1, 2, 3)) -> Path:
    """Return a finished run's directory holding a checkpoint for each of *epochs*."""
    controller.start(controller.plan())
    _settled(controller.selected)
    written = _write_epochs(controller.selected, max(epochs))
    for epoch in epochs:
        (written / f"round01_val_epoch{epoch:03}.pth").write_bytes(b"weights")
    return controller.selected


def _checkpoint_names(run: Path) -> list[str]:
    return sorted(path.name for path in run.rglob("*.pth"))


def test_only_the_checkpoints_not_kept_are_deleted(controller: TrainingController):
    run = _finished_with_checkpoints(controller)

    deleted = controller.remove_other_checkpoints(run, keep={2})

    assert sorted(path.name for path in deleted) == [
        "round01_val_epoch001.pth", "round01_val_epoch003.pth",
    ]
    assert _checkpoint_names(run) == ["round01_val_epoch002.pth"]


def test_what_would_be_deleted_can_be_asked_without_deleting(controller: TrainingController):
    run = _finished_with_checkpoints(controller)

    doomed = controller.other_checkpoints(run, keep={1, 3})

    assert [checkpoint.epoch for checkpoint in doomed] == [2]
    assert len(_checkpoint_names(run)) == 3


def test_deleting_checkpoints_leaves_the_rest_of_the_run_as_it_was(controller: TrainingController):
    run = _finished_with_checkpoints(controller)
    before = sorted(path.name for path in run.rglob("*") if path.suffix != ".pth")

    controller.remove_other_checkpoints(run, keep={3})

    assert sorted(path.name for path in run.rglob("*") if path.suffix != ".pth") == before
    assert controller.selected_run().epochs == 3


def test_another_run_s_checkpoints_are_never_touched(controller: TrainingController):
    run = _finished_with_checkpoints(controller)
    other = controller.training_directory / "round00" / "training_261001_000000"
    other.mkdir(parents=True)
    (other / "round00_val_epoch001.pth").write_bytes(b"weights")

    controller.remove_other_checkpoints(run, keep={1})

    assert (other / "round00_val_epoch001.pth").exists()


def test_nothing_is_deleted_without_a_checkpoint_to_keep(controller: TrainingController):
    run = _finished_with_checkpoints(controller)

    with pytest.raises(ValueError, match="at least one"):
        controller.remove_other_checkpoints(run, keep=set())

    assert len(_checkpoint_names(run)) == 3


def test_nothing_is_deleted_when_a_kept_epoch_has_no_checkpoint(controller: TrainingController):
    """A mistyped epoch must not turn into deleting every checkpoint there is."""
    run = _finished_with_checkpoints(controller)

    with pytest.raises(ValueError, match="no checkpoint for epoch"):
        controller.remove_other_checkpoints(run, keep={2, 9})

    assert len(_checkpoint_names(run)) == 3


def test_a_run_still_training_keeps_every_checkpoint(controller: TrainingController):
    controller.how["training"] = "sleep"
    controller.start(controller.plan())
    run = controller.selected
    written = _write_epochs(run, 2)
    (written / "round01_val_epoch001.pth").write_bytes(b"weights")
    (written / "round01_val_epoch002.pth").write_bytes(b"weights")
    try:
        with pytest.raises(ValueError, match="may still be training"):
            controller.remove_other_checkpoints(run, keep={2})
        assert len(_checkpoint_names(run)) == 2
    finally:
        controller.stop(run)


# -- what the window tells it -------------------------------------------------


def test_the_training_set_is_summed_up_in_a_line(controller: TrainingController):
    assert controller.training_set_summary() == "1 training and 1 validation pair(s)"


def test_a_training_set_that_cannot_be_read_says_why(controller: TrainingController, tmp_path: Path):
    controller.manifest = tmp_path / "notes.yaml"
    controller.manifest.write_text("title: notes\n")

    assert "not a training export's manifest" in controller.training_set_summary()


def test_an_export_just_written_is_chosen_only_where_nothing_was(tmp_path: Path):
    blank = TrainingController(stack_present=lambda: True)

    blank.exported(tmp_path / "first_manifest.yaml")
    blank.exported(tmp_path / "second_manifest.yaml")

    assert blank.manifest == tmp_path / "first_manifest.yaml"
    assert blank.last_export == tmp_path / "second_manifest.yaml"


def test_a_run_is_recorded_under_the_window_s_name(controller: TrainingController):
    from am_segmentation_manager.gui.attribution import Attribution
    from am_segmentation_manager.history import Recorder

    controller._attribution = Attribution(Recorder(by="Rosalind"))

    assert controller.start(controller.plan()).record.created_by == "Rosalind"
    _settled(controller.selected)


def test_the_scanned_folder_stands_in_for_a_relative_one(
    controller: TrainingController, tmp_path: Path
):
    """A manifest written by the command line records the folder as typed."""
    text = controller.manifest.read_text(encoding="utf-8")
    recorded = str(tmp_path / "collection")
    assert recorded in text
    controller.manifest.write_text(text.replace(recorded, "."), encoding="utf-8")

    with pytest.raises(TrainingRunRefused, match="does not say where the collection is"):
        controller.plan()

    controller.collection = tmp_path / "collection"
    assert len(controller.plan().train) == 1


def test_the_devices_offered_begin_with_auto_and_end_with_the_cpu(
    controller: TrainingController, monkeypatch
):
    from am_segmentation_manager.ml import gpus

    gpu = gpus.Gpu(0, 0, "GPU-0", "NVIDIA A100 80GB PCIe", 70 * 1024, 80 * 1024, "0:0")
    monkeypatch.setattr(gpus, "gpu_inventory", lambda: gpus.GpuInventory(gpus=(gpu,)))

    offered = controller.devices()

    assert [device for device, _ in offered] == ["auto", "cuda:0", "cpu"]
    assert "A100" in offered[1][1]
