"""Tests for ``train``, ``show-training`` on a prepared run, and ``stop-training``.

The process a run starts is replaced by a stand-in, as in ``test_training_process``;
what is pinned here is what the commands say and what they refuse to write.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.cli import training as cli_training
from am_segmentation_manager.io.tables import write_csv
from am_segmentation_manager.ml import environment
from am_segmentation_manager.operations import training_process, training_run
from am_segmentation_manager.operations.training_manifest import write_training_manifest
from am_segmentation_manager.operations.training_run import read_run_record
from test_training_process import CHILD, _settled


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch) -> Path:
    """Return the manifest of a two-pair export of placeholder files, labels taken as fine."""
    collection = tmp_path / "collection"
    collection.mkdir()
    for name in ("a", "b"):
        (collection / f"{name}.tif").write_bytes(b"")
        (collection / f"{name}_label.tif").write_bytes(b"")
    export = tmp_path / "export"
    export.mkdir()
    write_csv(export / "set_train.csv", ("image", "label"), [("a.tif", "a_label.tif")])
    write_csv(export / "set_val.csv", ("image", "label"), [("b.tif", "b_label.tif")])
    monkeypatch.setattr(training_run, "label_problem", lambda path, classes: None)
    return write_training_manifest(
        export / "set_manifest.yaml", source=collection, scope="",
        files={"train": export / "set_train.csv", "validation": export / "set_val.csv"},
        columns=("image", "label"), train_ratio=0.5, seed=1, keep_together="lamella",
        conditions="primary", include_empty_labels=False, counts={},
    )


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch):
    """Make a start run the stand-in child, behaving as its argument says."""
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    real = training_process.start_training

    def use(how: str) -> None:
        monkeypatch.setattr(environment, "training_stack_present", lambda: True)
        monkeypatch.setattr(
            cli_training, "start_training",
            lambda directory: real(directory, [sys.executable, str(script), how]),
        )

    return use


def _train(manifest: Path, *extra: str) -> int:
    runs = manifest.parent.parent / "runs"
    return main(["train", str(manifest), "--training-dir", str(runs), "--name", "round01", *extra])


def test_a_dry_run_says_what_would_be_done_and_writes_nothing(manifest: Path, capsys):
    assert _train(manifest, "--dry-run") == 0

    said = capsys.readouterr().out
    assert "1 training and 1 validation pair(s)" in said
    assert "nothing was written or started" in said
    assert not (manifest.parent.parent / "runs").exists()


def test_a_run_that_cannot_be_prepared_gives_every_reason(manifest: Path, capsys):
    code = main([
        "train", str(manifest), "--training-dir", str(manifest.parent.parent / "runs"),
        "--name", "has space", "--max-epochs", "0",
    ])

    said = capsys.readouterr().out
    assert code == 2
    assert "cannot name a run" in said and "at least 1" in said


def test_without_the_training_stack_nothing_is_written(manifest: Path, monkeypatch, capsys):
    monkeypatch.setattr(environment, "training_stack_present", lambda: False)

    assert _train(manifest) == 2

    assert "--extra train" in capsys.readouterr().out
    assert not (manifest.parent.parent / "runs").exists()


def test_training_starts_a_run_and_says_how_to_follow_it(manifest: Path, stand_in, capsys):
    stand_in("finish")

    assert _train(manifest, "--device", "cpu") == 0

    run = manifest.parent.parent / "runs" / "round01"
    said = capsys.readouterr().out
    assert f"am-seg show-training {run}" in said
    assert f"am-seg stop-training {run}" in said
    assert _settled(run).state == "finished"
    assert read_run_record(run).device == "cpu"


def test_refining_takes_the_published_refining_settings_unless_told(manifest: Path, stand_in):
    stand_in("finish")
    weights = manifest.parent.parent / "base.pth"
    weights.write_bytes(b"weights")

    assert _train(manifest, "--refine", str(weights), "--max-epochs", "20", "--save-all") == 0

    run = manifest.parent.parent / "runs" / "round01"
    _settled(run)
    settings = read_run_record(run).settings
    assert settings.mode == "refine"
    assert (settings.max_epochs, settings.frozen_epochs) == (20, 0)
    assert settings.initial_learning_rate == 2e-6
    assert settings.save_all


def test_show_training_says_how_a_prepared_run_stands(manifest: Path, stand_in, capsys):
    stand_in("fail")
    _train(manifest)
    run = manifest.parent.parent / "runs" / "round01"
    _settled(run)
    capsys.readouterr()

    assert main(["show-training", str(run.parent)]) == 0

    said = capsys.readouterr().out
    assert "round01: failed" in said
    assert "out of GPU memory" in said


def test_stop_training_stops_a_running_run(manifest: Path, stand_in, capsys):
    stand_in("sleep")
    _train(manifest)
    run = manifest.parent.parent / "runs" / "round01"
    capsys.readouterr()

    assert main(["stop-training", str(run)]) == 0
    assert "stopped on request" in capsys.readouterr().out

    assert main(["stop-training", str(run)]) == 0
    assert "Not running: stopped on request" in capsys.readouterr().out


def test_stop_training_refuses_a_directory_that_is_not_a_run(tmp_path: Path):
    assert main(["stop-training", str(tmp_path)]) == 2
