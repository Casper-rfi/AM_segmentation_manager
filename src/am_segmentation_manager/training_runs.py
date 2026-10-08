"""Reading what an AM-model-training run has written, while it runs or after it ends.

A run is the ``training_<yymmdd_HHMMSS>`` directory AM-model-training creates: metrics it
rewrites after every epoch, a checkpoint for each epoch that improved, and
``training_parameters.json`` once it ends. **Read, never written**, and from those files
alone, so a run reads the same wherever it was started. It lives outside the collection,
which is why this reads directories rather than the ProjectIndex. The format is
AM-model-training's; ``tests/data/amt_run`` holds a real run to it.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

#: The directory AM-model-training makes for each run, named for its start in UTC.
RUN_DIRECTORY_PATTERN: Final = re.compile(r"^training_(?P<started>\d{6}_\d{6})$")

#: What follows the run id in the two metrics files' names.
TRAIN_METRICS_SUFFIX: Final[str] = "_train_metrics.json"
VAL_METRICS_SUFFIX: Final[str] = "_val_metrics.json"

#: Written once, when a run ends without being killed.
PARAMETERS_FILENAME: Final[str] = "training_parameters.json"

#: A checkpoint: the run id, then the epoch it was saved after.
CHECKPOINT_PATTERN: Final = re.compile(r"^(?P<run_id>.+)_epoch(?P<epoch>\d+)\.pth$")

#: The metrics recorded once per class, each also as ``mean_*`` and
#: ``weighted_average_*``.
PER_CLASS_METRICS: Final[tuple[str, ...]] = (
    "iou", "dice", "f1", "accuracy", "precision", "recall",
)

#: What a run ranks epochs by where its parameters do not say: the mean of these.
DEFAULT_KEY_METRICS: Final[tuple[str, ...]] = ("weighted_average_iou", "weighted_average_dice")

#: The classes in label order, where a run's parameters do not say.
DEFAULT_LABEL_NAMES: Final[tuple[str, ...]] = ("background", "gis", "lamella", "crack", "vacuum")

#: How far below the path given :func:`find_training_runs` looks: a training
#: directory, then a directory per run, then AM-model-training's own.
SEARCH_DEPTH: Final[int] = 2


@dataclass(frozen=True)
class EpochMetrics:
    """One epoch's results on one side of the split.

    Attributes:
        epoch: Counted from 1, as the run counts it.
        loss: Mean loss over the epoch's steps; NaN if a step's was.
        scalars: Every single-number metric, ``mean_iou`` and
            ``weighted_average_dice`` among them.
        per_class: Every per-class metric, in label order.
    """

    epoch: int
    loss: float
    scalars: Mapping[str, float] = field(default_factory=dict)
    per_class: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def score(self, keys: Sequence[str]) -> float:
        """Return the mean of the scalars named by *keys*, which is what a run ranks epochs by."""
        return sum(self.scalars[key] for key in keys) / len(keys)


@dataclass(frozen=True)
class Checkpoint:
    """A weights file the run kept, and the epoch it was saved after."""

    epoch: int
    path: Path


@dataclass(frozen=True)
class TrainingRun:
    """Everything a run's directory says about it at the moment it was read.

    Attributes:
        directory: The ``training_<yymmdd_HHMMSS>`` directory.
        run_id: The name its files share, or ``None`` before it has written any.
        started: When it started, from the directory's name.
        train: Each epoch trained so far.
        val: Each epoch validated so far, which may skip epochs.
        checkpoints: The weights kept, by epoch.
        parameters: Its ``training_parameters.json``; empty until it ends.
        unreadable: Files that could not be parsed. A file is rewritten whole
            after every epoch, so one read mid-write is incomplete rather than wrong.
    """

    directory: Path
    run_id: str | None
    started: datetime | None
    train: tuple[EpochMetrics, ...] = ()
    val: tuple[EpochMetrics, ...] = ()
    checkpoints: tuple[Checkpoint, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    unreadable: tuple[Path, ...] = ()

    @property
    def finished(self) -> bool:
        """Return whether the run ended on its own; ``False`` while running, or once killed."""
        return bool(self.parameters)

    @property
    def key_metrics(self) -> tuple[str, ...]:
        """Return the scalars whose mean ranks the validated epochs."""
        return tuple(self.parameters.get("key_val_metrics") or DEFAULT_KEY_METRICS)

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the classes, in the order every per-class metric lists them."""
        return tuple(self.parameters.get("label_names") or DEFAULT_LABEL_NAMES)

    @property
    def frozen_epochs(self) -> int | None:
        """Return for how many epochs the encoder was frozen, once the run has said."""
        frozen = self.parameters.get("frozen_epochs")
        return None if frozen is None else int(frozen)

    @property
    def max_epochs(self) -> int | None:
        """Return the most epochs the run could have trained, once it has said."""
        epochs = self.parameters.get("max_epochs")
        return None if epochs is None else int(epochs)

    def best(self) -> EpochMetrics | None:
        """Return the validated epoch the run counts as best, or ``None`` before one."""
        return best_epoch(self.val, self.key_metrics)

    def checkpoint_for(self, epoch: int) -> Checkpoint | None:
        """Return the weights kept after *epoch*, if the run kept them."""
        return next((kept for kept in self.checkpoints if kept.epoch == epoch), None)


def best_epoch(epochs: Sequence[EpochMetrics], keys: Sequence[str]) -> EpochMetrics | None:
    """Return the epoch scoring highest on *keys*, the earliest of any tie, as the run decides."""
    best: EpochMetrics | None = None
    for epoch in epochs:
        if best is None or epoch.score(keys) > best.score(keys):
            best = epoch
    return best


def parse_epochs(entries: Sequence[Mapping[str, Any]]) -> tuple[EpochMetrics, ...]:
    """Return the epochs one metrics file lists, the confusion matrices left out.

    Raises:
        ValueError: If an entry has no epoch or no loss, which is a file of
            another format rather than one caught mid-write.
    """
    epochs = []
    for entry in entries:
        if "epoch" not in entry or "loss" not in entry:
            raise ValueError(f"An epoch entry without 'epoch' or 'loss': {sorted(entry)}")
        scalars: dict[str, float] = {}
        per_class: dict[str, tuple[float, ...]] = {}
        for key, value in entry.items():
            if key in ("epoch", "loss", "confusion_matrix"):
                continue
            if isinstance(value, list):
                per_class[key] = tuple(float(item) for item in value)
            elif isinstance(value, int | float):
                scalars[key] = float(value)
        epochs.append(
            EpochMetrics(int(entry["epoch"]), float(entry["loss"]), scalars, per_class)
        )
    return tuple(sorted(epochs, key=lambda epoch: epoch.epoch))


def parse_started(name: str) -> datetime | None:
    """Return when a run started, from its directory's name, or ``None`` for another name."""
    match = RUN_DIRECTORY_PATTERN.match(name)
    if match is None:
        return None
    return datetime.strptime(match["started"], "%y%m%d_%H%M%S").replace(tzinfo=UTC)


def is_run_directory(path: Path) -> bool:
    """Return whether *path* is a directory AM-model-training made for a run."""
    return RUN_DIRECTORY_PATTERN.match(path.name) is not None and path.is_dir()


def find_training_runs(path: Path, depth: int = SEARCH_DEPTH) -> list[Path]:
    """Return the run directories at *path* or up to *depth* levels below it, oldest first.

    A run directory is not looked inside for further runs.
    """
    if is_run_directory(path):
        return [path]
    found: list[Path] = []
    level = [path]
    for _ in range(depth + 1):
        below: list[Path] = []
        for directory in level:
            try:
                children = sorted(
                    child for child in directory.iterdir()
                    if child.is_dir() and not child.name.startswith(".")
                )
            except OSError as error:
                logger.warning("Could not look inside %s: %s", directory, error)
                continue
            for child in children:
                (found if is_run_directory(child) else below).append(child)
        level = below
    return sorted(found, key=lambda run: (parse_started(run.name) or datetime.min.replace(tzinfo=UTC), run))


def read_training_run(directory: Path) -> TrainingRun:
    """Return what *directory* says about its run now.

    A file caught mid-write, or of a format this does not know, is listed in
    ``unreadable`` and logged rather than raised, so one bad read never stops a
    report about the rest.
    """
    unreadable: list[Path] = []

    def read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            logger.warning("Could not read %s: %s", path, error)
            unreadable.append(path)
            return None

    def read_epochs(path: Path | None) -> tuple[EpochMetrics, ...]:
        entries = None if path is None else read_json(path)
        if entries is None:
            return ()
        try:
            return parse_epochs(entries)
        except (TypeError, ValueError) as error:
            logger.warning("Could not read %s: %s", path, error)
            unreadable.append(path)  # type: ignore[arg-type]
            return ()

    names = sorted(child.name for child in directory.iterdir() if child.is_file())
    train_file = _ending_with(names, TRAIN_METRICS_SUFFIX)
    val_file = _ending_with(names, VAL_METRICS_SUFFIX)
    saved = [match for name in names if (match := CHECKPOINT_PATTERN.match(name))]
    checkpoints = tuple(sorted(
        (Checkpoint(int(match["epoch"]), directory / match.string) for match in saved),
        key=lambda kept: kept.epoch,
    ))

    parameters = read_json(directory / PARAMETERS_FILENAME) if PARAMETERS_FILENAME in names else None
    if parameters is not None and not isinstance(parameters, dict):
        logger.warning("Could not read %s: not a mapping", directory / PARAMETERS_FILENAME)
        unreadable.append(directory / PARAMETERS_FILENAME)
        parameters = None

    run_id = (
        (parameters or {}).get("run_id")
        or (train_file and train_file.removesuffix(TRAIN_METRICS_SUFFIX))
        or (val_file and val_file.removesuffix(VAL_METRICS_SUFFIX))
        or (saved[0]["run_id"] if saved else None)
    )
    return TrainingRun(
        directory=directory,
        run_id=run_id,
        started=parse_started(directory.name),
        train=read_epochs(directory / train_file if train_file else None),
        val=read_epochs(directory / val_file if val_file else None),
        checkpoints=checkpoints,
        parameters=parameters or {},
        unreadable=tuple(unreadable),
    )


def _ending_with(names: Sequence[str], suffix: str) -> str | None:
    """Return the first of *names* ending with *suffix*, or ``None``."""
    return next((name for name in names if name.endswith(suffix)), None)
