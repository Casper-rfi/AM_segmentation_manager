"""Tests for copying a run's checkpoints out and deleting the ones not wanted.

How a run stands is patched where it is asked: whether a process is really running is
:mod:`test_training_process`'s to test, and what is decided here is what follows from it.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from am_segmentation_manager.training import checkpoints, preparation
from am_segmentation_manager.training.checkpoints import (
    copy_checkpoint,
    other_checkpoints,
    remove_other_checkpoints,
)
from am_segmentation_manager.training.choice import choose_best, chosen_best_or_none
from am_segmentation_manager.training.preparation import (
    NEW_MODEL,
    apply_training_run,
    plan_training_run,
)
from am_segmentation_manager.training.process import RunStatus
from am_segmentation_manager.training.runs import Checkpoint, read_training_run
from test_training_process import export_of_placeholders
from test_training_runs import _epoch


def _checkpoint(directory: Path, whole: bool = True) -> Checkpoint:
    path = directory / "round01_val_epoch003.pth"
    if whole:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("round01_val_epoch003/data.pkl", b"weights")
    else:
        path.write_bytes(b"PK\x03\x04 cut short")
    return Checkpoint(3, path)


def _written(directory: Path, epochs=(1, 2, 3), *, run_id: str = "round01_val",
             finished: bool = True) -> Path:
    """Write what AM-model-training leaves in *directory*, a checkpoint for each of *epochs*."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = [_epoch(n, 0.1 * n, 0.1 * n) for n in range(1, max(epochs) + 1)]
    (directory / f"{run_id}_train_metrics.json").write_text(json.dumps(entries))
    (directory / f"{run_id}_val_metrics.json").write_text(json.dumps(entries))
    for epoch in epochs:
        (directory / f"{run_id}_epoch{epoch:03}.pth").write_bytes(b"weights")
    if finished:
        (directory / "training_parameters.json").write_text(
            json.dumps({"run_id": run_id, "max_epochs": 90, "frozen_epochs": 25})
        )
    return directory


def _prepared(tmp_path: Path, monkeypatch, state: str = "finished", epochs=(1, 2, 3)) -> Path:
    """Return a run am-seg prepared, standing in *state*, holding a checkpoint for each of *epochs*."""
    monkeypatch.setattr(preparation, "label_report", lambda path, classes: (None, None))
    run = apply_training_run(plan_training_run(
        export_of_placeholders(tmp_path), tmp_path / "runs", "round01", NEW_MODEL,
        inspect_label=lambda path, classes: None,
    ))
    monkeypatch.setattr(checkpoints, "run_status", lambda directory: RunStatus(state))
    _written(run / "training_261008_120000", epochs)
    return run


def _checkpoint_names(run: Path) -> list[str]:
    return sorted(path.name for path in run.rglob("*.pth"))


# -- taking a checkpoint out --------------------------------------------------


def test_a_checkpoint_is_copied_out_and_the_run_keeps_its_own(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)

    copied = copy_checkpoint(checkpoint, tmp_path / "models" / "AM_SEM_round01.pth")

    assert copied.read_bytes() == checkpoint.path.read_bytes()
    assert checkpoint.path.exists()


def test_a_checkpoint_is_never_copied_over_a_model(tmp_path: Path):
    existing = tmp_path / "AM_SEM_All_V01.pth"
    existing.write_bytes(b"the published model")

    with pytest.raises(FileExistsError):
        copy_checkpoint(_checkpoint(tmp_path), existing)

    assert existing.read_bytes() == b"the published model"


def test_a_checkpoint_cut_short_is_not_copied_out(tmp_path: Path):
    with pytest.raises(ValueError, match="incomplete"):
        copy_checkpoint(_checkpoint(tmp_path, whole=False), tmp_path / "out.pth")

    assert not (tmp_path / "out.pth").exists()


# -- deleting the checkpoints not wanted --------------------------------------


def test_only_the_checkpoints_not_kept_are_deleted(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)

    deleted = remove_other_checkpoints(run, keep={2})

    assert sorted(path.name for path in deleted) == [
        "round01_val_epoch001.pth", "round01_val_epoch003.pth",
    ]
    assert _checkpoint_names(run) == ["round01_val_epoch002.pth"]


def test_what_would_be_deleted_can_be_asked_without_deleting(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)

    doomed = other_checkpoints(run, keep={1, 3})

    assert [checkpoint.epoch for checkpoint in doomed] == [2]
    assert len(_checkpoint_names(run)) == 3


def test_deleting_checkpoints_leaves_the_rest_of_the_run_as_it_was(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)
    before = sorted(path.name for path in run.rglob("*") if path.suffix != ".pth")

    remove_other_checkpoints(run, keep={3})

    assert sorted(path.name for path in run.rglob("*") if path.suffix != ".pth") == before
    assert len(read_training_run(run / "training_261008_120000").train) == 3


def test_another_run_s_checkpoints_are_never_touched(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)
    other = _written(tmp_path / "runs" / "round00" / "training_261001_000000", (1,), run_id="round00_val")

    remove_other_checkpoints(run, keep={1})

    assert (other / "round00_val_epoch001.pth").exists()


def test_nothing_is_deleted_without_a_checkpoint_to_keep(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="at least one"):
        remove_other_checkpoints(run, keep=set())

    assert len(_checkpoint_names(run)) == 3


def test_nothing_is_deleted_when_a_kept_epoch_has_no_checkpoint(tmp_path: Path, monkeypatch):
    """A mistyped epoch must not turn into deleting every checkpoint there is."""
    run = _prepared(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="no checkpoint for epoch"):
        remove_other_checkpoints(run, keep={2, 9})

    assert len(_checkpoint_names(run)) == 3


@pytest.mark.parametrize("state", ["running", "unknown"])
def test_a_run_still_training_keeps_every_checkpoint(tmp_path: Path, monkeypatch, state: str):
    run = _prepared(tmp_path, monkeypatch, state=state, epochs=(1, 2))

    with pytest.raises(ValueError, match="may still be training"):
        remove_other_checkpoints(run, keep={2})

    assert len(_checkpoint_names(run)) == 2


def test_a_finished_run_s_other_checkpoints_can_be_deleted_whoever_started_it(tmp_path: Path):
    written = _written(tmp_path / "wide_seed2" / "training_261001_120000", (1, 2), run_id="set")

    remove_other_checkpoints(written, keep={2})

    assert sorted(path.name for path in written.glob("*.pth")) == ["set_epoch002.pth"]


def test_a_run_probably_still_training_elsewhere_keeps_its_checkpoints(tmp_path: Path):
    written = _written(
        tmp_path / "cluster" / "training_261001_120000", (1, 2), run_id="set", finished=False
    )

    with pytest.raises(ValueError, match="may still be training"):
        remove_other_checkpoints(written, keep={2})

    assert len(list(written.glob("*.pth"))) == 2


def test_the_epoch_marked_is_kept_whatever_else_is(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)
    choose_best(run, read_training_run(run / "training_261008_120000"), 1)

    remove_other_checkpoints(run, keep={2})

    assert _checkpoint_names(run) == ["round01_val_epoch001.pth", "round01_val_epoch002.pth"]


def test_nothing_is_deleted_while_the_mark_cannot_be_read(tmp_path: Path, monkeypatch):
    run = _prepared(tmp_path, monkeypatch)
    (run / "am_training_choice.yaml").write_text("version: 1\nepoch: best\n", encoding="utf-8")

    with pytest.raises(ValueError, match="names no epoch"):
        remove_other_checkpoints(run, keep={2})

    assert len(_checkpoint_names(run)) == 3
    assert chosen_best_or_none(run) is None


def test_a_run_being_validated_keeps_its_checkpoints(tmp_path: Path, monkeypatch):
    """The run itself has finished, so only its validation still scoring holds the checkpoints."""
    from test_validation_run import _run, scored_validation

    run = scored_validation(tmp_path, monkeypatch, _run(tmp_path))
    (run / "training_parameters.json").write_text(json.dumps({"run_id": "set"}))
    monkeypatch.setattr(checkpoints, "run_status", lambda directory: RunStatus("running"))

    with pytest.raises(ValueError, match="being validated"):
        remove_other_checkpoints(run, keep={1})

    assert len(list(run.glob("*.pth"))) == 3
