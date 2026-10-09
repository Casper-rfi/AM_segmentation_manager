"""Starting a prepared training run, or a validation of one, in a process of its own, and saying how it stands.

Training takes hours, so the process is detached and outlives whatever started it;
everything known about it afterwards is read from its run directory. **It holds a
lock there for as long as it lives**, which the system releases however it dies, so
a run is running exactly while that lock is held. A stop writes its request before
ending the process, which is what tells a stopped run from one that crashed. **A run
directory is started once.** :func:`run_status` is the one answer to how a run stands.
A validation is prepared, started, stopped and read the same way, in its own directory;
what its process does there is score, an epoch at a time, each kept as it is done.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal

from ..history import resolve_author
from ..io.atomic import try_lock
from ..io.yaml import read_yaml, write_yaml
from .training_run import RUN_RECORD_FILENAME, RunRecord, describe_weights, read_run_record
from .validation_run import (
    VALIDATION_RECORD_FILENAME,
    ValidationRecord,
    read_images,
    read_validation_record,
)

logger = logging.getLogger(__name__)

#: What the process writes when it starts, and again when it ends by itself.
PROCESS_FILENAME: Final[str] = "am_training_process.yaml"

#: Locked by the process for as long as it lives.
LOCK_FILENAME: Final[str] = "am_training.lock"

#: Written by whoever stops a run, before the process is ended.
STOP_FILENAME: Final[str] = "am_training_stop.yaml"

#: Everything the process prints, AM-model-training's progress included.
LOG_FILENAME: Final[str] = "train.log"

#: How long a start waits for the process to say it has started.
START_SECONDS: Final[float] = 60.0

#: How long a stop waits for the process to go before ending it harder.
STOP_GRACE_SECONDS: Final[float] = 30.0

#: The processes this one started, kept so that one which has ended is collected
#: rather than left as a defunct entry for as long as this process runs.
_started: list[subprocess.Popen[bytes]] = []

State = Literal["prepared", "running", "finished", "failed", "stopped", "interrupted", "unknown"]

#: What each state means, for a report.
STATE_MEANINGS: Final[Mapping[str, str]] = {
    "prepared": "prepared, not started",
    "running": "training",
    "finished": "finished",
    "failed": "failed",
    "stopped": "stopped on request",
    "interrupted": "ended without finishing: killed, crashed, or the machine went down",
    "unknown": "started on another machine, which is the only one that can say",
}


class TrainingNotStarted(RuntimeError):
    """A run could not be started, or did not come up."""


@dataclass(frozen=True)
class RunStatus:
    """How a run stands, read from its directory.

    Attributes:
        state: One of ``prepared``, ``running``, ``finished``, ``failed``,
            ``stopped``, ``interrupted`` and ``unknown``.
        pid: The process's id on *host*, once started.
        host: The machine it was started on.
        started_at: When it started, in UTC.
        ended_at: When it ended by itself, in UTC; "" if it did not.
        device: The device it runs on as ``nvidia-smi`` numbers it, once it has chosen one.
        error: Why it failed.
        versions: The am-model-training and torch it ran with.
    """

    state: State
    pid: int | None = None
    host: str = ""
    started_at: str = ""
    ended_at: str = ""
    device: str = ""
    error: str = ""
    versions: Mapping[str, str] | None = None

    def describe(self) -> str:
        """Return the state in words, with what qualifies it."""
        said = STATE_MEANINGS[self.state]
        if self.state == "running":
            where = f" on {self.device}" if self.device else ""
            said += f"{where} (process {self.pid} on {self.host}, since {self.started_at})"
        elif self.state == "failed" and self.error:
            said += f": {self.error}"
        elif self.state == "unknown":
            said = f"started on {self.host}, which is the only machine that can say how it stands"
        return said


def run_status(run_directory: Path) -> RunStatus:
    """Return how the run in *run_directory* stands now.

    Raises:
        ValueError: If the directory holds no run record, or one that cannot be read.
    """
    if read_run_record(run_directory) is None and read_validation_record(run_directory) is None:
        raise ValueError(
            f"{run_directory} holds no {RUN_RECORD_FILENAME} or {VALIDATION_RECORD_FILENAME}: "
            f"not a training run or a validation"
        )
    process = read_yaml(run_directory / PROCESS_FILENAME)
    if not process:
        return RunStatus("prepared")

    pid = process.get("pid")
    host = str(process.get("host") or "")

    def status(state: State) -> RunStatus:
        return RunStatus(
            state,
            pid=pid if isinstance(pid, int) else None,
            host=host,
            started_at=str(process.get("started_at") or ""),
            ended_at=str(process.get("ended_at") or ""),
            device=str(process.get("device") or ""),
            error=str(process.get("error") or ""),
            versions=process.get("versions"),
        )

    outcome = process.get("outcome")
    if outcome == "finished" or outcome == "failed":
        return status(outcome)
    here = host == socket.gethostname()
    # The lock is asked before the host name, which a laptop changes with its
    # network: a lock that is held is a process that is alive, wherever it is.
    if _alive(run_directory, pid if here else None):
        return status("running")
    if not here:
        return status("unknown")
    return status("stopped" if (run_directory / STOP_FILENAME).is_file() else "interrupted")


def start_training(
    run_directory: Path,
    command: Sequence[str] | None = None,
    wait: float = START_SECONDS,
) -> RunStatus:
    """Start the prepared run in *run_directory*, detached, and return how it stands.

    Returns once the process has said it started, which it does before loading
    anything slow.

    Args:
        command: What to run in place of this module, given the run directory
            as its last argument.
        wait: How long to wait for the process to say it has started.

    Raises:
        TrainingNotStarted: If the run was started before, the training stack
            is not installed, or the process ended or stayed silent.
        ValueError: If the directory holds no run record.
    """
    status = run_status(run_directory)
    if status.state != "prepared":
        raise TrainingNotStarted(
            f"{run_directory} was already started ({status.describe()}); a run directory "
            f"is started once, so prepare another run"
        )
    if command is None:
        from ..ml.environment import TRAINING_INSTALL_HINT, training_stack_present

        if not training_stack_present():
            raise TrainingNotStarted(
                f"Training needs AM-model-training, which is not installed. {TRAINING_INSTALL_HINT}"
            )
        command = [sys.executable, "-m", __name__]

    from ..ml.environment import QUIET_MLFLOW

    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.setdefault(*QUIET_MLFLOW)
    log = run_directory / LOG_FILENAME
    with log.open("ab") as output:
        process = subprocess.Popen(  # noqa: S603 - our own interpreter and module
            [*command, str(run_directory)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            env=environment, **_detached(),
        )
    _collect()
    _started.append(process)

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if (run_directory / PROCESS_FILENAME).is_file():
            logger.debug("Started training in %s (process %d)", run_directory, process.pid)
            return run_status(run_directory)
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if (run_directory / PROCESS_FILENAME).is_file():
        return run_status(run_directory)
    raise TrainingNotStarted(
        f"The training process did not start in {run_directory}; {log} says why"
    )


def stop_training(
    run_directory: Path, user: str | None = None, grace: float = STOP_GRACE_SECONDS
) -> RunStatus:
    """Stop the run in *run_directory* if it is running, and return how it stands.

    The epoch in progress is lost, and a checkpoint being written at that moment
    is left incomplete; every earlier one is whole.

    Args:
        user: The name to record the request under, resolved as a history entry's is.
        grace: How long to wait for the process to go before ending it harder.

    A run started on another machine is left alone, and said so.

    Raises:
        ValueError: If the directory holds no run record.
    """
    status = run_status(run_directory)
    if status.state != "running" or status.pid is None:
        return status
    if status.host != socket.gethostname():
        logger.error(
            "The run was started as process %d on %s, and can only be stopped there",
            status.pid, status.host,
        )
        return status

    by, _ = resolve_author(user)
    write_yaml(run_directory / STOP_FILENAME, {"requested_at": _now(), "requested_by": by or ""})
    _end(status.pid, signal.SIGTERM)
    if not _gone(run_directory, status.pid, grace) and sys.platform != "win32":
        logger.warning("Process %d did not stop within %.0f s; killing it", status.pid, grace)
        _end(status.pid, signal.SIGKILL)
        _gone(run_directory, status.pid, grace)
    return run_status(run_directory)


def run_child(
    run_directory: Path,
    trainer: Callable[[RunRecord], str] | None = None,
    scorer: Callable[[ValidationRecord], str] | None = None,
) -> int:
    """Train the run, or score the validation, in *run_directory* in this process, and return its exit status.

    Args:
        trainer: Trains what a run's record asks and returns the device used;
            AM-model-training's where omitted.
        scorer: Scores what a validation's record asks and returns the device used.
    """
    descriptor = os.open(
        run_directory / LOCK_FILENAME, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o666
    )
    try:
        if try_lock(descriptor) is False:
            logger.error("Another process is already training in %s", run_directory)
            return 3
        validation = read_validation_record(run_directory)
        record = read_run_record(run_directory) if validation is None else None
        if record is None and validation is None:
            logger.error("%s holds no %s", run_directory, RUN_RECORD_FILENAME)
            return 2

        started: dict[str, Any] = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _now(),
            "versions": _versions(),
        }
        write_yaml(run_directory / PROCESS_FILENAME, started)

        def chosen(device: str) -> None:
            """Say which device the work is on while it is still going."""
            started["device"] = device
            write_yaml(run_directory / PROCESS_FILENAME, started)

        try:
            if validation is not None:
                device = (scorer or partial(_validate, on_device=chosen))(validation)
            else:
                assert record is not None
                device = (trainer or partial(_train, on_device=chosen))(record)
        # Everything, an interrupt included: the record of how a run ended is
        # what a reader has in place of the process.
        except BaseException as error:  # noqa: BLE001
            logger.exception("Training failed")
            _ended(run_directory, started, outcome="failed", error=repr(error))
            return 1
        _ended(run_directory, started, outcome="finished", device=device)
        logger.info("Training finished")
        return 0
    finally:
        os.close(descriptor)


def _train(record: RunRecord, on_device: Callable[[str], None] | None = None) -> str:
    """Train what *record* asks through AM-model-training, returning the device used.

    Args:
        on_device: Told the device once it is chosen, before training begins.
    """
    base: Path | None = None
    if record.base is not None:
        base = Path(str(record.base["file"]))
        if not base.is_file():
            raise FileNotFoundError(f"The model weights to refine have gone: {base}")
        if describe_weights(base)["sha256"] != record.base.get("sha256"):
            raise RuntimeError(
                f"{base} is not the file this run was prepared with: its contents have changed"
            )

    from ..ml.trainer import train

    settings = {key: value for key, value in asdict(record.settings).items() if key != "mode"}
    return train(
        record.train_list, record.validation_list, record.directory,
        device=record.device, base_weights=base, on_device=on_device, **settings,
    )


def _validate(record: ValidationRecord, on_device: Callable[[str], None] | None = None) -> str:
    """Score each epoch *record* names that has no scores yet, returning the device used.

    Each epoch's scores are written as it is done, so a validation stopped
    part-way keeps what it finished.

    Args:
        on_device: Told the device once it is chosen, before the first epoch is scored.
    """
    used = [record.device]

    def chosen(device: str) -> None:
        used[0] = device
        if on_device is not None:
            on_device(device)

    from ..ml.scoring import score_checkpoints
    from ..training_runs import read_training_run, written_in
    from ..validation_scores import ImageScores, scores_filename, write_scores

    written = written_in(record.run_directory)
    if written is None:
        raise FileNotFoundError(f"The run this validation scores has gone: {record.run_directory}")
    run = read_training_run(written)
    names = tuple(row["image"] for row in read_images(record.images_file))
    todo = [
        epoch for epoch in record.epochs
        if not (record.directory / scores_filename(epoch)).exists()
    ]
    weights = []
    for epoch in todo:
        checkpoint = run.checkpoint_for(epoch)
        if checkpoint is None or not checkpoint.whole():
            raise FileNotFoundError(
                f"The weights of epoch {epoch} have gone, or are incomplete, since this "
                f"validation was prepared"
            )
        weights.append(checkpoint.path)
    scored = score_checkpoints(weights, record.list_file, device=record.device, on_device=chosen)
    for epoch, (matrices, losses) in zip(todo, scored):
        write_scores(
            record.directory / scores_filename(epoch),
            ImageScores(
                epoch, names,
                tuple(tuple(tuple(row) for row in matrix) for matrix in matrices),
                tuple(losses),
            ),
        )
        logger.info("Scored epoch %d", epoch)
    return used[0]


def _ended(run_directory: Path, started: Mapping[str, Any], **outcome: str) -> None:
    """Rewrite the process file with how the run ended."""
    write_yaml(run_directory / PROCESS_FILENAME, {**started, "ended_at": _now(), **outcome})


def _versions() -> dict[str, str]:
    """Return the installed am-model-training and torch, read without importing either."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for distribution in ("am-model-training", "torch"):
        try:
            found[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return found


def _now() -> str:
    """Return the time in UTC, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _detached() -> dict[str, Any]:
    """Return what makes a child outlive this process and its terminal, on this platform."""
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        return {"creationflags": flags}
    # A session of its own: no terminal's hang-up reaches it, and its data
    # loading workers share its process group, so a stop reaches them too.
    return {"start_new_session": True}


def _alive(run_directory: Path, pid: object) -> bool:
    """Return whether the run's process, started on this machine, is still alive."""
    try:
        descriptor = os.open(run_directory / LOCK_FILENAME, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except OSError:
        return False
    try:
        taken = try_lock(descriptor)
    finally:
        # Closing gives back a lock this took, having learned that nobody held it.
        os.close(descriptor)
    if taken is not None:
        return not taken
    # No locks on this filesystem, so the process id is all there is to ask.
    if not isinstance(pid, int) or sys.platform == "win32":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _collect() -> None:
    """Forget each process this one started that has since ended."""
    _started[:] = [process for process in _started if process.poll() is None]


def _end(pid: int, how: int) -> None:
    """Send *how* to the process and, where it leads a group, to its workers."""
    try:
        if sys.platform != "win32" and os.getpgid(pid) == pid:
            os.killpg(pid, how)
        else:
            os.kill(pid, how)
    except (ProcessLookupError, PermissionError) as error:
        logger.debug("Could not signal process %d: %s", pid, error)


def _gone(run_directory: Path, pid: int, within: float) -> bool:
    """Wait up to *within* seconds for the run's process to end, returning whether it did."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _alive(run_directory, pid):
            break
        time.sleep(0.1)
    _collect()
    return not _alive(run_directory, pid)


def main(argv: Sequence[str] | None = None) -> int:
    """Train the run whose directory is the one argument; what a start runs."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) != 1:
        print("usage: python -m am_segmentation_manager.operations.training_process RUN_DIRECTORY")
        return 2
    # In UTC, as every time in the run directory is.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S", stream=sys.stdout,
    )
    return run_child(Path(arguments[0]))


if __name__ == "__main__":
    raise SystemExit(main())
