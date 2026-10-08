"""What the Train tab knows: the run being set up, and the runs in the training directory.

Holds no Qt and decides nothing an operation does not: preparing, starting, stopping
and reading a run are :mod:`..operations.training_run`, :mod:`..operations.training_process`
and :mod:`..training_runs`, as the command line uses them. **Every run shown is read
from its directory**, so a window opened later, or on another day, shows the same.
A metrics file caught mid-rewrite keeps the epochs last read whole, so a curve never
drops to nothing between two polls.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..io.atomic import write_atomically
from ..operations.training_manifest import read_training_manifest
from ..operations.training_process import RunStatus, run_status, start_training, stop_training
from ..operations.training_run import (
    NAME_PATTERN,
    Mode,
    RunRecord,
    TrainingPlan,
    TrainingSettings,
    apply_training_run,
    default_settings,
    find_run_directories,
    plan_training_run,
    read_pairs,
    read_run_record,
)
from ..training_runs import (
    Checkpoint,
    EpochMetrics,
    TrainingRun,
    find_training_runs,
    read_training_run,
)
from .attribution import Attribution

logger = logging.getLogger(__name__)

#: The states in which a run's files can still change, so are worth reading again.
LIVE_STATES = frozenset({"running", "unknown"})


@dataclass(frozen=True)
class RunSummary:
    """One run in the training directory, as the tab lists and draws it.

    Attributes:
        directory: The run's directory.
        name: What it is called.
        record: What it was asked to do, or None where the record cannot be read.
        status: How it stands, or None where that cannot be said.
        written: What AM-model-training has written, or None before it has.
        problem: Why the run cannot be read, or "".
    """

    directory: Path
    name: str
    record: RunRecord | None = None
    status: RunStatus | None = None
    written: TrainingRun | None = None
    problem: str = ""

    @property
    def epochs(self) -> int:
        """Return how many epochs have been trained."""
        return len(self.written.train) if self.written else 0

    @property
    def best(self) -> EpochMetrics | None:
        """Return the validated epoch the run counts as best, if it has one."""
        return self.written.best() if self.written else None

    @property
    def live(self) -> bool:
        """Return whether the run's files may still change."""
        return self.status is not None and self.status.state in LIVE_STATES

    def line(self) -> str:
        """Return the run in one line: its name, how it stands and how far it got."""
        if self.problem:
            return f"{self.name}: {self.problem}"
        parts = [self.name]
        if self.status is not None:
            parts.append(self.status.state)
        if self.record is not None and (self.epochs or self.live):
            parts.append(f"epoch {self.epochs} of at most {self.record.settings.max_epochs}")
        best = self.best
        if best is not None and self.written is not None:
            parts.append(f"best {best.epoch} ({best.score(self.written.key_metrics):.3f})")
        return " · ".join(parts)


class TrainingController:
    """The Train tab's state and the operations it asks for.

    Args:
        attribution: Whose name a run and a stop are recorded under; the window's.
        stack_present: Answers whether the training stack is installed.
        start: Starts a prepared run; :func:`start_training` unless a test says otherwise.

    Attributes:
        collection: The folder the window has scanned, which a manifest that
            recorded a relative folder is resolved against.
        last_export: The manifest of the training export the window last wrote.
    """

    def __init__(
        self,
        attribution: Attribution | None = None,
        stack_present: Callable[[], bool] | None = None,
        start: Callable[[Path], RunStatus] = start_training,
    ) -> None:
        self._attribution = attribution
        self._stack_present = stack_present
        self._start = start
        self.collection: Path | None = None
        self.last_export: Path | None = None
        self.training_directory: Path | None = None
        self.manifest: Path | None = None
        self.name: str = ""
        self.settings: TrainingSettings = default_settings("new")
        self.base_weights: Path | None = None
        self.device: str = "auto"
        self.selected: Path | None = None
        self._whole: dict[Path, TrainingRun] = {}

    # -- what is installed ----------------------------------------------------

    def can_train(self) -> bool:
        """Return whether the training stack is installed, without importing it."""
        if self._stack_present is not None:
            return self._stack_present()
        from ..ml.environment import training_stack_present

        return training_stack_present()

    def install_hint(self) -> str:
        """Return how to install the training stack."""
        from ..ml.environment import TRAINING_INSTALL_HINT

        return TRAINING_INSTALL_HINT

    def devices(self) -> list[tuple[str, str]]:
        """Return where a run can train as ``(device, caption)``: auto, each NVIDIA GPU, the CPU.

        Read from ``nvidia-smi``, never torch, so listing them costs no import.
        """
        from ..ml.gpus import gpu_inventory

        offered = [("auto", "auto")]
        offered.extend((gpu.device, gpu.describe()) for gpu in gpu_inventory().gpus)
        offered.append(("cpu", "cpu"))
        return offered

    # -- the run being set up -------------------------------------------------

    def exported(self, manifest: Path) -> None:
        """Note the training export the window has just written, choosing it if none is chosen."""
        self.last_export = manifest
        if self.manifest is None:
            self.manifest = manifest

    def training_set_summary(self) -> str:
        """Return the chosen training set in a line, or why it cannot be read, or ""."""
        if self.manifest is None:
            return ""
        try:
            manifest = read_training_manifest(self.manifest)
            counts = [
                len(read_pairs(path)) if path is not None else 0
                for path in (manifest.train, manifest.validation)
            ]
        except (OSError, ValueError) as error:
            return str(error)
        return f"{counts[0]} training and {counts[1]} validation pair(s)"

    @property
    def mode(self) -> Mode:
        """Return whether a new model is trained or an existing one refined."""
        return self.settings.mode

    def set_mode(self, mode: Mode) -> None:
        """Switch between a new model and a refine, taking that mode's published settings."""
        if mode != self.settings.mode:
            self.settings = default_settings(mode)

    def change(self, **settings: Any) -> None:
        """Change settings of the run being set up, by :class:`TrainingSettings` field."""
        self.settings = replace(self.settings, **settings)

    def reset_settings(self) -> None:
        """Put every setting back to the published one for the mode."""
        self.settings = default_settings(self.settings.mode)

    def suggest_name(self, today: datetime | None = None) -> str:
        """Return a name no run in the training directory has: the date, the mode, a number."""
        stem = f"{(today or datetime.now()):%Y%m%d}_{self.settings.mode}"
        taken = self._taken_names()
        number = 1
        while f"{stem}_{number:02d}" in taken:
            number += 1
        return f"{stem}_{number:02d}"

    def problems(self) -> list[str]:
        """Return why the run cannot be started yet, from what costs nothing to check.

        What needs the export read and each label opened is :meth:`plan`'s to find.
        """
        found = []
        if not self.can_train():
            found.append("The training stack is not installed")
        if self.manifest is None:
            found.append("Choose a training set: the manifest a training export wrote")
        if self.training_directory is None:
            found.append("Choose the training directory, where runs are kept")
        if not self.name:
            found.append("Name the run")
        elif not NAME_PATTERN.match(self.name):
            found.append("A name holds letters, digits, '.', '_' and '-'")
        elif self.name in self._taken_names():
            found.append(f"A run named {self.name} already exists")
        if self.settings.mode == "refine" and self.base_weights is None:
            found.append("Choose the model weights to refine")
        found.extend(self.settings.problems())
        return found

    def plan(self) -> TrainingPlan:
        """Check the run being set up in full, each label opened, and return it ready to write.

        Raises:
            TrainingRunRefused: Naming every reason; nothing has been written.
            ValueError: If no training set or training directory is chosen.
        """
        if self.manifest is None or self.training_directory is None:
            raise ValueError("Choose a training set and a training directory first")
        return plan_training_run(
            self.manifest, self.training_directory, self.name, self.settings,
            base_weights=self.base_weights if self.settings.mode == "refine" else None,
            device=self.device, start_path=self._start_path(),
        )

    def start(self, plan: TrainingPlan, user: str | None = None) -> RunSummary:
        """Write the run *plan* describes, start it, and return it selected.

        Args:
            user: The name to record in place of the window's.

        Raises:
            FileExistsError: If its directory has appeared since the plan was made.
            TrainingNotStarted: If the process did not come up; the directory stays.
        """
        directory = apply_training_run(plan, user=user or self._user())
        self.selected = directory
        self._start(directory)
        return self.summary(directory)

    # -- the runs there are ---------------------------------------------------

    def runs(self) -> list[RunSummary]:
        """Return every run in the training directory, the newest first."""
        if self.training_directory is None or not self.training_directory.is_dir():
            return []
        summaries = [
            self.summary(directory)
            for directory in find_run_directories(self.training_directory)
        ]
        return sorted(
            summaries,
            key=lambda run: (run.record.created_at if run.record else "", run.name),
            reverse=True,
        )

    def summary(self, directory: Path) -> RunSummary:
        """Return the run in *directory* as its files say it stands now."""
        try:
            record = read_run_record(directory)
            status = run_status(directory)
        except ValueError as error:
            return RunSummary(directory, directory.name, problem=str(error))
        written = find_training_runs(directory, depth=0)
        return RunSummary(
            directory=directory,
            name=record.name if record else directory.name,
            record=record,
            status=status,
            # A run directory is started once, so it holds one of these at most.
            written=self._steady(read_training_run(written[-1])) if written else None,
        )

    def selected_run(self) -> RunSummary | None:
        """Return the run selected, read now, or None if none is or it has gone."""
        if self.selected is None or not self.selected.is_dir():
            return None
        return self.summary(self.selected)

    def any_live(self) -> bool:
        """Return whether any run's files may still change, so the tab should keep reading."""
        return any(run.live for run in self.runs())

    def stop(self, directory: Path, user: str | None = None) -> RunSummary:
        """Stop the run in *directory* if it is running, and return how it stands."""
        stop_training(directory, user=user or self._user())
        return self.summary(directory)

    # -- what a run produced --------------------------------------------------

    def copy_checkpoint(self, checkpoint: Checkpoint, destination: Path) -> Path:
        """Copy a checkpoint out of its run to *destination*, to predict with or share.

        Copied, never moved: the run directory stays as the run left it.

        Raises:
            ValueError: If the checkpoint was cut short by a stop and is not whole.
            FileExistsError: If *destination* exists, or appears meanwhile.
        """
        if not checkpoint.whole():
            raise ValueError(
                f"{checkpoint.path.name} is incomplete: the run was stopped while writing it"
            )
        if destination.exists():
            raise FileExistsError(f"{destination} already exists; nothing was written")
        landed = write_atomically(
            destination,
            lambda temporary: shutil.copyfile(checkpoint.path, temporary),
            commit_if=lambda _temporary: not destination.exists(),
        )
        if not landed:
            raise FileExistsError(
                f"{destination} appeared while it was being written; nothing was written"
            )
        logger.info("Copied %s to %s", checkpoint.path, destination)
        return destination

    def other_checkpoints(self, directory: Path, keep: Collection[int]) -> list[Checkpoint]:
        """Return the checkpoints of the run in *directory* that :meth:`remove_other_checkpoints` would delete.

        Raises:
            ValueError: If the run may still be training, no epoch is kept, or
                a kept epoch has no checkpoint. Nothing is deleted on any of them.
        """
        run = self.summary(directory)
        if run.problem:
            raise ValueError(run.problem)
        if run.live:
            raise ValueError(
                f"{run.name} may still be training; its checkpoints are kept until it has ended"
            )
        kept = set(keep)
        if not kept:
            raise ValueError("Choose at least one checkpoint to keep")
        checkpoints = run.written.checkpoints if run.written else ()
        missing = sorted(kept - {checkpoint.epoch for checkpoint in checkpoints})
        if missing:
            raise ValueError(
                f"{run.name} has no checkpoint for epoch(s) {', '.join(map(str, missing))}"
            )
        return [checkpoint for checkpoint in checkpoints if checkpoint.epoch not in kept]

    def remove_other_checkpoints(self, directory: Path, keep: Collection[int]) -> list[Path]:
        """Delete every checkpoint of the run in *directory* but those of the epochs in *keep*.

        **Never called without the person having been shown what goes and saying yes**:
        a deleted checkpoint cannot be got back. Only that run's are touched, and
        its metrics, lists and records stay, so the run still reads as it did.

        Returns:
            The files deleted.

        Raises:
            ValueError: As :meth:`other_checkpoints`, before anything is deleted.
        """
        doomed = [checkpoint.path for checkpoint in self.other_checkpoints(directory, keep)]
        for path in doomed:
            path.unlink()
            logger.info("Deleted the checkpoint %s", path)
        return doomed

    # -- internals ------------------------------------------------------------

    def _user(self) -> str | None:
        """Return the name the window records under, if it has one."""
        return self._attribution.recorder.by if self._attribution is not None else None

    def _start_path(self) -> Path | None:
        """Return the scanned folder where the manifest did not record an absolute one."""
        if self.manifest is None or self.collection is None:
            return None
        try:
            recorded = read_training_manifest(self.manifest).source
        except ValueError:
            return None
        return None if Path(recorded).is_absolute() else self.collection

    def _taken_names(self) -> set[str]:
        """Return the name of everything in the training directory, run or not."""
        directory = self.training_directory
        if directory is None or not directory.is_dir():
            return set()
        return {child.name for child in directory.iterdir()}

    def _steady(self, run: TrainingRun) -> TrainingRun:
        """Return *run* with any list caught mid-rewrite replaced by the one last read whole."""
        earlier = self._whole.get(run.directory)
        if run.unreadable and earlier is not None:
            run = replace(
                run,
                train=run.train or earlier.train,
                val=run.val or earlier.val,
            )
        if run.train or run.val:
            self._whole[run.directory] = run
        return run
