"""Tests for starting a training run in a process of its own, and reading how it stands.

The process is real -- started detached, exactly as a run is -- and what it trains
with is a stand-in, so nothing here needs the training stack. What is pinned is what
a reader of the run directory is told: running while the process lives, and afterwards
which of finished, failed, stopped and interrupted it was.
"""
from __future__ import annotations

import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest
import yaml

from am_segmentation_manager.io.atomic import try_lock
from am_segmentation_manager.io.tables import write_csv
from am_segmentation_manager.operations import training_process
from am_segmentation_manager.operations.training_manifest import write_training_manifest
from am_segmentation_manager.operations.training_process import (
    LOCK_FILENAME,
    LOG_FILENAME,
    PROCESS_FILENAME,
    STOP_FILENAME,
    TrainingNotStarted,
    run_child,
    run_status,
    start_training,
    stop_training,
)
from am_segmentation_manager.operations.training_run import (
    NEW_MODEL,
    REFINE,
    apply_training_run,
    plan_training_run,
)

SRC = Path(__file__).resolve().parent.parent / "src"

#: A child that trains with a stand-in: its first argument says how training goes.
CHILD = f"""
import sys, time
sys.path.insert(0, {str(SRC)!r})
from pathlib import Path
from am_segmentation_manager.operations.training_process import run_child

def trainer(record):
    how = sys.argv[1]
    print("training", record.name, flush=True)
    if how == "fail":
        raise RuntimeError("out of GPU memory")
    if how == "sleep":
        time.sleep(600)
    return "cpu"

if __name__ == "__main__":
    raise SystemExit(run_child(Path(sys.argv[2]), trainer))
"""


def _prepare(tmp_path: Path, settings=NEW_MODEL, base_weights: Path | None = None) -> Path:
    """Return a prepared run directory, made from a two-pair export of placeholder files."""
    collection = tmp_path / "collection"
    collection.mkdir()
    rows = []
    for name in ("a", "b"):
        (collection / f"{name}.tif").write_bytes(b"")
        (collection / f"{name}_label.tif").write_bytes(b"")
        rows.append((f"{name}.tif", f"{name}_label.tif"))
    export = tmp_path / "export"
    export.mkdir()
    write_csv(export / "set_train.csv", ("image", "label"), rows[:1])
    write_csv(export / "set_val.csv", ("image", "label"), rows[1:])
    manifest = write_training_manifest(
        export / "set_manifest.yaml", source=collection, scope="",
        files={"train": export / "set_train.csv", "validation": export / "set_val.csv"},
        columns=("image", "label"), train_ratio=0.5, seed=1, keep_together="lamella",
        conditions="primary", include_empty_labels=False, counts={},
    )
    plan = plan_training_run(
        manifest, tmp_path / "runs", "round01", settings, base_weights=base_weights,
        inspect_label=lambda path, classes: None,
    )
    return apply_training_run(plan)


def _command(tmp_path: Path, how: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    return [sys.executable, str(script), how]


def _settled(run: Path, within: float = 60.0) -> training_process.RunStatus:
    """Wait for the run to stop running, and return how it stands."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        status = run_status(run)
        if status.state != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("the run was still running")


# -- how a run stands ---------------------------------------------------------


def test_a_prepared_run_has_not_started(tmp_path: Path):
    assert run_status(_prepare(tmp_path)).state == "prepared"


def test_a_directory_without_a_record_is_not_a_run(tmp_path: Path):
    with pytest.raises(ValueError, match="not a training run"):
        run_status(tmp_path)


def test_a_run_that_trains_to_the_end_is_finished(tmp_path: Path):
    run = _prepare(tmp_path)

    started = start_training(run, _command(tmp_path, "finish"))
    status = _settled(run)

    assert started.pid is not None and started.host == socket.gethostname()
    assert status.state == "finished"
    assert status.device == "cpu"
    assert status.ended_at
    assert "training round01" in (run / LOG_FILENAME).read_text(encoding="utf-8")


def test_a_run_whose_training_raises_has_failed_and_says_why(tmp_path: Path):
    run = _prepare(tmp_path)

    start_training(run, _command(tmp_path, "fail"))
    status = _settled(run)

    assert status.state == "failed"
    assert "out of GPU memory" in status.error
    assert "out of GPU memory" in status.describe()
    assert "Traceback" in (run / LOG_FILENAME).read_text(encoding="utf-8")


def test_a_run_is_running_while_its_process_lives_and_stopped_on_request(tmp_path: Path):
    run = _prepare(tmp_path)

    started = start_training(run, _command(tmp_path, "sleep"))
    assert started.state == "running"
    assert f"process {started.pid}" in started.describe()

    stopped = stop_training(run, user="Ada")

    assert stopped.state == "stopped"
    request = yaml.safe_load((run / STOP_FILENAME).read_text(encoding="utf-8"))
    assert request["requested_by"] == "Ada"


def test_a_run_killed_without_a_request_was_interrupted(tmp_path: Path):
    """A crash, a kill from outside or a power cut leaves no outcome and no request."""
    run = _prepare(tmp_path)
    started = start_training(run, _command(tmp_path, "sleep"))

    os.kill(started.pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)

    assert _settled(run).state == "interrupted"


def test_a_run_directory_is_started_once(tmp_path: Path):
    run = _prepare(tmp_path)
    start_training(run, _command(tmp_path, "finish"))
    _settled(run)

    with pytest.raises(TrainingNotStarted, match="already started"):
        start_training(run, _command(tmp_path, "finish"))


def test_stopping_a_run_that_is_not_running_changes_nothing(tmp_path: Path):
    run = _prepare(tmp_path)

    assert stop_training(run).state == "prepared"
    assert not (run / STOP_FILENAME).exists()


def test_a_process_that_never_comes_up_is_reported_with_its_log(tmp_path: Path):
    run = _prepare(tmp_path)

    with pytest.raises(TrainingNotStarted, match=LOG_FILENAME):
        start_training(run, [sys.executable, "-c", "raise SystemExit('no such module')"])

    assert run_status(run).state == "prepared"


def test_starting_without_the_training_stack_says_what_to_install(tmp_path: Path, monkeypatch):
    from am_segmentation_manager.ml import environment

    run = _prepare(tmp_path)
    monkeypatch.setattr(environment, "training_stack_present", lambda: False)

    with pytest.raises(TrainingNotStarted, match="--extra train"):
        start_training(run)

    assert not (run / LOG_FILENAME).exists()


# -- another machine, and the lock --------------------------------------------


def _started_elsewhere(run: Path) -> None:
    (run / PROCESS_FILENAME).write_text(
        "pid: 4242\nhost: another-workstation\nstarted_at: '2026-10-08T10:00:00Z'\n",
        encoding="utf-8",
    )


def test_a_run_started_on_another_machine_is_not_guessed_at(tmp_path: Path):
    run = _prepare(tmp_path)
    _started_elsewhere(run)

    status = run_status(run)

    assert status.state == "unknown"
    assert "another-workstation" in status.describe()


def test_a_run_started_on_another_machine_is_not_stopped_from_here(tmp_path: Path):
    """Its process id means another process on this machine."""
    run = _prepare(tmp_path)
    _started_elsewhere(run)

    assert stop_training(run).state == "unknown"
    assert not (run / STOP_FILENAME).exists()


def test_a_held_lock_is_a_living_process_whatever_the_host_is_called(tmp_path: Path):
    """A laptop's host name changes with its network; the lock does not."""
    run = _prepare(tmp_path)
    _started_elsewhere(run)
    descriptor = os.open(run / LOCK_FILENAME, os.O_RDWR | os.O_CREAT)
    try:
        assert try_lock(descriptor) is True
        assert run_status(run).state == "running"
    finally:
        os.close(descriptor)

    assert run_status(run).state == "unknown"


# -- the process's own body ---------------------------------------------------


def test_the_process_records_what_it_ran_with_and_how_it_ended(tmp_path: Path):
    run = _prepare(tmp_path)

    assert run_child(run, lambda record: "cuda:0") == 0

    written = yaml.safe_load((run / PROCESS_FILENAME).read_text(encoding="utf-8"))
    assert written["pid"] == os.getpid()
    assert written["outcome"] == "finished"
    assert written["device"] == "cuda:0"
    assert isinstance(written["versions"], dict)


def test_a_second_process_does_not_train_in_a_directory_being_trained_in(tmp_path: Path):
    run = _prepare(tmp_path)
    descriptor = os.open(run / LOCK_FILENAME, os.O_RDWR | os.O_CREAT)
    try:
        assert try_lock(descriptor) is True

        assert run_child(run, lambda record: "cpu") == 3
    finally:
        os.close(descriptor)
    assert not (run / PROCESS_FILENAME).exists()


def test_the_record_s_settings_reach_the_trainer(tmp_path: Path, monkeypatch):
    from am_segmentation_manager.ml import trainer as training
    from am_segmentation_manager.operations.training_run import read_run_record

    run = _prepare(tmp_path)
    seen = {}

    def train(train_list, validation_list, output, **settings):
        seen.update(settings, lists=(train_list, validation_list), output=output)
        return "cpu"

    monkeypatch.setattr(training, "train", train)

    assert training_process._train(read_run_record(run)) == "cpu"

    assert seen["lists"] == (run / "round01.csv", run / "val.csv")
    assert seen["output"] == run
    assert seen["base_weights"] is None and seen["device"] == "auto"
    assert (seen["max_epochs"], seen["frozen_epochs"]) == (100, 25)
    assert "mode" not in seen


def test_weights_replaced_since_the_run_was_prepared_are_not_refined(tmp_path: Path, monkeypatch):
    from am_segmentation_manager.ml import trainer as training
    from am_segmentation_manager.operations.training_run import read_run_record

    weights = tmp_path / "base.pth"
    weights.write_bytes(b"as prepared")
    run = _prepare(tmp_path, settings=REFINE, base_weights=weights)
    weights.write_bytes(b"something else")
    monkeypatch.setattr(training, "train", lambda *args, **kwargs: pytest.fail("trained"))

    with pytest.raises(RuntimeError, match="contents have changed"):
        training_process._train(read_run_record(run))


# -- the module a start runs --------------------------------------------------


def test_the_module_wants_exactly_one_run_directory(capsys):
    assert training_process.main([]) == 2
    assert "RUN_DIRECTORY" in capsys.readouterr().out


def test_a_start_runs_this_module_with_this_interpreter(tmp_path: Path, monkeypatch):
    """The process is started as a module, so the workers it spawns re-import
    one guarded by ``__name__`` rather than a script that would train again."""
    from am_segmentation_manager.ml import environment

    run = _prepare(tmp_path)
    seen = {}

    class Started:
        pid = 4242

        def poll(self):
            return 0

    def popen(command, **options):
        seen["command"] = command
        seen["options"] = options
        return Started()

    monkeypatch.setattr(environment, "training_stack_present", lambda: True)
    monkeypatch.setattr(training_process.subprocess, "Popen", popen)

    with pytest.raises(TrainingNotStarted):
        start_training(run, wait=0.2)

    assert seen["command"] == [
        sys.executable, "-m", "am_segmentation_manager.operations.training_process", str(run),
    ]
    assert seen["options"]["env"]["MLFLOW_DISABLE_AGENT_HINT"] == "1"
    assert seen["options"].get("start_new_session") or seen["options"].get("creationflags")
