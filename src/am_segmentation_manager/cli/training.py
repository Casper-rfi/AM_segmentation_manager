"""Training a model on an exported set, following the run, and stopping it.

``train`` prepares a run from a training export and starts it in a process of its
own, which goes on after this command returns; ``show-training`` reports how a run
stands and what it has written, and draws its curves; ``stop-training`` ends one.
Only ``train`` needs the training stack. The order is the order they are met in.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..operations.training_process import (
    LOG_FILENAME,
    TrainingNotStarted,
    run_status,
    start_training,
    stop_training,
)
from ..operations.training_run import (
    NEW_MODEL,
    REFINE,
    TrainingRunRefused,
    apply_training_run,
    default_settings,
    find_run_directories,
    plan_training_run,
    read_run_record,
)
from ..plots import plot_training_run
from ..training_runs import (
    PER_CLASS_METRICS,
    SEARCH_DEPTH,
    TrainingRun,
    find_training_runs,
    read_training_run,
)
from .scope import _add_overwrite_argument, _outputs_refused

logger = logging.getLogger(__name__)


#: The heading ``am-seg --help`` files these under.
TITLE = "Train models"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register this group's subcommands on *sub*."""
    train = sub.add_parser(
        "train",
        help="Train a new model, or refine an existing one, on an exported training set",
        description=(
            "Prepare a training run from the manifest a training export wrote, and "
            "start it with AM-model-training in a process of its own, which goes on "
            "after this command returns. The run gets a directory named for it in "
            "the training directory: its two lists, a copy of the manifest, a record "
            "of what was asked, a log, and what AM-model-training writes. Everything "
            "is checked first, each label opened again among it, and an existing "
            "run directory is never reused. The split is the export's own. Without "
            f"--refine a new model is trained, the encoder frozen for the first "
            f"{NEW_MODEL.frozen_epochs} of at most {NEW_MODEL.max_epochs} epochs; with "
            f"it, an existing one is refined for at most {REFINE.max_epochs}. Needs "
            "the train extra."
        ),
    )
    train.add_argument(
        "manifest", help="The <prefix>_manifest.yaml a training export wrote beside its lists"
    )
    train.add_argument(
        "--training-dir", required=True, metavar="DIR",
        help="Where runs are kept, outside the collection. The run gets a directory in it",
    )
    train.add_argument(
        "--name", required=True,
        help=(
            "What to call the run: its directory, and the stem of its checkpoints. "
            "Letters, digits, '.', '_' and '-'"
        ),
    )
    train.add_argument(
        "--refine", default=None, metavar="WEIGHTS",
        help="Refine these model weights (.pth) rather than training a new model",
    )
    train.add_argument(
        "--start-path", default=None, metavar="DIR",
        help=(
            "The folder the export was made from, which its paths are written "
            "against. Needed only where the manifest recorded a relative one"
        ),
    )
    train.add_argument(
        "--max-epochs", type=int, default=None, metavar="N",
        help=(
            "The most epochs to train; a run stops early once it stops improving "
            f"(default: {NEW_MODEL.max_epochs}, or {REFINE.max_epochs} refining)"
        ),
    )
    train.add_argument(
        "--frozen-epochs", type=int, default=None, metavar="N",
        help=(
            "Epochs the encoder is held frozen for before it is trained "
            f"(default: {NEW_MODEL.frozen_epochs}, or {REFINE.frozen_epochs} refining)"
        ),
    )
    train.add_argument(
        "--initial-lr", type=float, default=None, metavar="RATE",
        help=(
            "Where the learning rate starts (default: "
            f"{NEW_MODEL.initial_learning_rate:g}, or {REFINE.initial_learning_rate:g} refining)"
        ),
    )
    train.add_argument(
        "--max-lr", type=float, default=None, metavar="RATE",
        help=(
            "The learning rate's peak (default: "
            f"{NEW_MODEL.max_learning_rate:g}, or {REFINE.max_learning_rate:g} refining)"
        ),
    )
    train.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=(
            "Images per training step; lower it, to 1 if need be, where the GPU runs "
            "out of memory "
            f"(default: {NEW_MODEL.training_batch_size})"
        ),
    )
    train.add_argument(
        "--validation-batch-size", type=int, default=None, metavar="N",
        help=f"Images per validation step (default: {NEW_MODEL.validation_batch_size})",
    )
    train.add_argument(
        "--save-all", action="store_true",
        help=(
            "Keep a checkpoint after every epoch, about 170 MB each, rather than "
            "only after the epochs that improved"
        ),
    )
    train.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help=f"Fixes augmentation and shuffling (default: {NEW_MODEL.seed})",
    )
    train.add_argument(
        "--device", default="auto",
        help=(
            "Where to train: auto, cpu, or a GPU as nvidia-smi numbers it, such as "
            "cuda:1. auto takes the GPU with the most memory free. AM-model-training "
            "trains on a CUDA GPU or the CPU (default: auto)"
        ),
    )
    train.add_argument(
        "--dry-run", action="store_true",
        help="Check everything and say what would be done, writing and starting nothing",
    )

    show = sub.add_parser(
        "show-training",
        help="Report how a training run is going or how it ended, and draw its curves",
        description=(
            "Report every run found at PATH or below it. For a run am-seg prepared: "
            "whether it is training, finished, failed, was stopped, or ended without "
            "finishing. For each training_<date>_<time> directory AM-model-training "
            f"wrote, there or up to {SEARCH_DEPTH} levels down, wherever it was "
            "started: its epochs so far, the validation epoch it counts as best, and "
            "the checkpoints it kept. Reads only, and needs no training stack."
        ),
    )
    show.add_argument(
        "path", nargs="?", default=".",
        help="A run directory, or a directory holding runs (default: current directory)",
    )
    show.add_argument(
        "--metric", default="iou", choices=PER_CLASS_METRICS,
        help="The per-class metric to report and draw (default: iou)",
    )
    show.add_argument(
        "--classes", nargs="+", default=None, metavar="CLASS",
        help="The classes --plot draws a curve for, by name (default: every class)",
    )
    show.add_argument(
        "--plot", default=None, metavar="FILE",
        help=(
            "Draw the run's loss and per-class metric by epoch. The suffix chooses "
            "the format: .png to look at, .svg for a figure. Needs exactly one run "
            "at PATH, and the optional matplotlib extra"
        ),
    )
    _add_overwrite_argument(show)

    stop = sub.add_parser(
        "stop-training",
        help="Stop a training run that am-seg started on this computer",
        description=(
            "End the run's process and record that it was asked to. The epoch in "
            "progress is lost; every checkpoint already written is kept, and one "
            "being written at that moment is left incomplete, which show-training "
            "says. A stopped run is not continued: prepare another, refining from "
            "one of its checkpoints. A run started on another computer is left alone."
        ),
    )
    stop.add_argument("path", help="The run's directory")


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle ``am-seg train``."""
    asked = {
        "max_epochs": args.max_epochs,
        "frozen_epochs": args.frozen_epochs,
        "initial_learning_rate": args.initial_lr,
        "max_learning_rate": args.max_lr,
        "training_batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "seed": args.seed,
    }
    settings = replace(
        default_settings("refine" if args.refine else "new"),
        save_all=args.save_all,
        **{key: value for key, value in asked.items() if value is not None},
    )
    try:
        plan = plan_training_run(
            Path(args.manifest), Path(args.training_dir), args.name, settings,
            base_weights=Path(args.refine) if args.refine else None,
            device=args.device,
            start_path=Path(args.start_path) if args.start_path else None,
        )
    except TrainingRunRefused as refused:
        logger.error("Could not prepare the run:")
        for reason in refused.reasons:
            logger.error("  %s", reason)
        return 2

    for line in plan.describe():
        print(line)
    if args.dry_run:
        print("Dry run: nothing was written or started")
        return 0

    from ..ml.environment import TRAINING_INSTALL_HINT, training_stack_present

    if not training_stack_present():
        logger.error("Training needs AM-model-training, which is not installed. %s",
                     TRAINING_INSTALL_HINT)
        return 2

    directory = apply_training_run(plan, user=getattr(args, "user", None))
    try:
        status = start_training(directory)
    except TrainingNotStarted as error:
        logger.error("%s", error)
        return 2
    print(f"Started: {status.describe()}")
    print(f"  follow it:  am-seg show-training {directory}")
    print(f"  stop it:    am-seg stop-training {directory}")
    print(f"  its log:    {directory / LOG_FILENAME}")
    return 0


def _cmd_show_training(args: argparse.Namespace) -> int:
    """Handle ``am-seg show-training``."""
    if _outputs_refused([args.plot], args.overwrite):
        return 2
    path = Path(args.path)
    if not path.is_dir():
        logger.error("Not a directory: %s", path)
        return 2
    prepared = find_run_directories(path)
    found = find_training_runs(path)
    if not prepared and not found:
        logger.error("No training run found at %s or up to %d levels below it", path, SEARCH_DEPTH)
        return 2

    runs: list[TrainingRun] = []
    for directory in prepared:
        try:
            record = read_run_record(directory)
            status = run_status(directory)
        except ValueError as error:
            logger.error("%s", error)
            continue
        print(f"{directory}  ({record.name if record else directory.name}: {status.describe()})")
        for written in find_training_runs(directory, depth=0):
            runs.append(read_training_run(written))
            _print_run(runs[-1], args.metric)
    inside = {run.directory for run in runs}
    for directory in found:
        if directory not in inside:
            runs.append(read_training_run(directory))
            _print_run(runs[-1], args.metric)

    if args.plot:
        if len(runs) != 1:
            logger.error(
                "%d runs have written results here; name one of them to draw its curves",
                len(runs),
            )
            return 2
        try:
            plot_training_run(runs[0], Path(args.plot), metric=args.metric,
                              classes=args.classes, overwrite=args.overwrite)
        except (RuntimeError, ValueError) as error:
            logger.error("%s", error)
            return 2
    return 0


def _cmd_stop_training(args: argparse.Namespace) -> int:
    """Handle ``am-seg stop-training``."""
    directory = Path(args.path)
    try:
        before = run_status(directory)
        after = stop_training(directory, user=getattr(args, "user", None))
    except ValueError as error:
        logger.error("%s", error)
        return 2
    if before.state != "running":
        print(f"Not running: {before.describe()}")
        return 0
    print(after.describe())
    return 0 if after.state == "stopped" else 1


def _print_run(run: TrainingRun, metric: str) -> None:
    """Print what one run's files say about it."""
    print(f"{run.directory}  ({run.run_id or 'no files written yet'})")

    started = f"started {run.started:%Y-%m-%d %H:%M} UTC, " if run.started else ""
    if run.finished:
        progress = f"finished after {len(run.train)} of at most {run.max_epochs} epochs"
        if run.frozen_epochs:
            progress += f", encoder frozen for the first {run.frozen_epochs}"
    else:
        progress = f"not finished (training, or stopped): {len(run.train)} epoch(s) so far"
    print(f"  {started}{progress}")

    best = run.best()
    if best is not None:
        scores = ", ".join(f"{key} {best.scalars[key]:.3f}" for key in run.key_metrics)
        print(f"  best validation epoch {best.epoch}: {scores}")
        values = best.per_class.get(metric, ())
        if values:
            classes = ", ".join(
                f"{name} {value:.3f}" for name, value in zip(run.label_names, values)
            )
            print(f"    {metric} per class: {classes}")
        kept = run.checkpoint_for(best.epoch)
        print(f"    weights: {kept.path if kept else 'not kept in this directory'}")
    if run.checkpoints:
        epochs = ", ".join(
            f"{kept.epoch}{'' if kept.whole() else ' (incomplete)'}" for kept in run.checkpoints
        )
        print(f"  checkpoints kept after epoch(s): {epochs}")
    for unreadable in run.unreadable:
        print(f"  could not read {unreadable.name}; it may be mid-write, so ask again")


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "train": _cmd_train,
    "show-training": _cmd_show_training,
    "stop-training": _cmd_stop_training,
}
