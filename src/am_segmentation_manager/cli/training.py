"""Following a model's training, and seeing how it ended.

``show-training`` reports what AM-model-training has written for each run it finds,
and draws one run's curves. It only reads, so it works without the training stack
installed, on runs started anywhere.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

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
    show = sub.add_parser(
        "show-training",
        help="Report how a training run is going or how it ended, and draw its curves",
        description=(
            "Read what AM-model-training writes into each training_<date>_<time> "
            "directory -- its metrics after every epoch, the checkpoints it kept, "
            "and its parameters once it ends -- and report every run found at PATH "
            f"or up to {SEARCH_DEPTH} levels below it. A run without its "
            "parameters is still training, or was stopped before it ended. Reads "
            "only, and needs no training stack."
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
        "--plot", default=None, metavar="FILE",
        help=(
            "Draw the run's loss and per-class metric by epoch. The suffix chooses "
            "the format: .png to look at, .svg for a figure. Needs exactly one run "
            "at PATH, and the optional matplotlib extra"
        ),
    )
    _add_overwrite_argument(show)


def _cmd_show_training(args: argparse.Namespace) -> int:
    """Handle ``am-seg show-training``."""
    if _outputs_refused([args.plot], args.overwrite):
        return 2
    path = Path(args.path)
    if not path.is_dir():
        logger.error("Not a directory: %s", path)
        return 2
    found = find_training_runs(path)
    if not found:
        logger.error("No training run found at %s or up to %d levels below it", path, SEARCH_DEPTH)
        return 2

    runs = [read_training_run(directory) for directory in found]
    for run in runs:
        _print_run(run, args.metric)

    if args.plot:
        if len(runs) > 1:
            logger.error("%d runs found; name one of them to draw its curves", len(runs))
            return 2
        try:
            plot_training_run(runs[0], Path(args.plot), metric=args.metric,
                              overwrite=args.overwrite)
        except (RuntimeError, ValueError) as error:
            logger.error("%s", error)
            return 2
    return 0


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
        epochs = ", ".join(str(kept.epoch) for kept in run.checkpoints)
        print(f"  checkpoints kept after epoch(s): {epochs}")
    for unreadable in run.unreadable:
        print(f"  could not read {unreadable.name}; it may be mid-write, so ask again")


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "show-training": _cmd_show_training,
}
