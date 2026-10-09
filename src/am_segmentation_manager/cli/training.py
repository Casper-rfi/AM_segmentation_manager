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

from ..io.tables import write_csv
from ..operations.training_choice import choose_best, clear_chosen_best, read_chosen_best
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
from ..operations.validation_run import (
    ROWS,
    apply_validation,
    find_validations,
    plan_validation,
    read_validation_record,
)
from ..plots import plot_comparison, plot_training_run
from ..training_compare import (
    DEFAULT_EPOCH_RANGE,
    VALUE_CAPTIONS,
    ComparedRun,
    compare_runs,
)
from ..training_runs import (
    PER_CLASS_METRICS,
    SEARCH_DEPTH,
    TrainingRun,
    find_training_runs,
    read_training_run,
    written_in,
)
from ..validation_scores import find_scores, metrics_for, read_scores
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
            "The collection: the folder the export was made from, which its paths "
            "are written against. Needed where the export and the collection have "
            "been moved apart since, the manifest saying only where one was from "
            "the other"
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

    validate = sub.add_parser(
        "validate",
        help="Score a training run's checkpoints on a list it did not train on",
        description=(
            "Prepare a validation of the run at RUN on the rows of a training "
            "export, and start it in a process of its own: each checkpoint is run "
            "over each image once and what it scored on that image is kept, in "
            "RUN/am_validations/NAME. show-training reports it, and a part of the "
            "list -- one instrument, one voltage -- is scored from the same files "
            "afterwards. Everything is checked before anything is written, each "
            "label opened; stop it with stop-training on its directory. A stopped "
            "validation keeps the epochs it finished and is not continued."
        ),
    )
    validate.add_argument(
        "run", metavar="RUN",
        help="The run's directory: one am-seg prepared, or a training_<date>_<time> directory",
    )
    validate.add_argument(
        "manifest", help="The <prefix>_manifest.yaml of the training export to score on",
    )
    validate.add_argument(
        "--name", required=True, metavar="NAME", help="What to call the validation",
    )
    validate.add_argument(
        "--rows", default="validation", choices=ROWS,
        help=(
            "Which rows of the export: its validation list, or every row it names, "
            "for an export made of data held out whole (default: validation)"
        ),
    )
    validate.add_argument(
        "--epochs", nargs="+", type=int, default=None, metavar="N",
        help="The epochs to score (default: every one with whole weights on disk)",
    )
    validate.add_argument(
        "--start-path", default=None, metavar="DIR",
        help=(
            "The collection the export was made from, needed where the two have "
            "been moved apart"
        ),
    )
    validate.add_argument(
        "--device", default="auto", metavar="DEVICE",
        help="Where to score: auto, cuda:N as nvidia-smi numbers them, or cpu (default: auto)",
    )
    validate.add_argument(
        "--dry-run", action="store_true",
        help="Check everything and say what would be done, writing and starting nothing",
    )

    compare = sub.add_parser(
        "compare-training",
        help="Set training runs side by side: one value per run and class",
        description=(
            "Print a table of the runs found at each PATH or below it: a row per "
            "run, a column per class, from each run's validation. A cell is the "
            "metric at the run's best epoch, at the epoch choose-best marked (the "
            "run's own where none is marked), or its mean or median over a range "
            "of epochs. Reads only, and needs no training stack."
        ),
    )
    compare.add_argument(
        "paths", nargs="+", metavar="PATH", help="A run directory, or a directory holding runs",
    )
    compare.add_argument(
        "--metric", default="iou", choices=PER_CLASS_METRICS,
        help="The per-class metric compared (default: iou)",
    )
    compare.add_argument(
        "--value", default="best", choices=tuple(VALUE_CAPTIONS),
        help="What a cell holds (default: best, the run's best epoch)",
    )
    compare.add_argument(
        "--epochs", nargs=2, type=int, default=list(DEFAULT_EPOCH_RANGE), metavar=("FIRST", "LAST"),
        help=(
            "The epochs a mean or median is taken over "
            f"(default: {DEFAULT_EPOCH_RANGE[0]} {DEFAULT_EPOCH_RANGE[1]})"
        ),
    )
    compare.add_argument(
        "--classes", nargs="+", default=None, metavar="CLASS",
        help="The classes compared, by name, in this order (default: every class)",
    )
    compare.add_argument(
        "--output", default=None, metavar="FILE", help="Write the table to this CSV as well",
    )
    compare.add_argument(
        "--plot", default=None, metavar="FILE",
        help=(
            "Draw the table shaded by value. The suffix chooses the format. Needs "
            "the optional matplotlib extra"
        ),
    )
    _add_overwrite_argument(compare)

    choose = sub.add_parser(
        "choose-best",
        help="Mark an epoch as a training run's best, in place of the one the run chose",
        description=(
            "Record the epoch you count as the run's best, or take that back. The "
            "run's own best is the validated epoch scoring highest, which is not "
            "always the model wanted. Written as am_training_choice.yaml in the "
            "run's directory, under the name --user gives; nothing the run wrote is "
            "changed, and show-training reports both."
        ),
    )
    choose.add_argument(
        "path",
        help="The run's directory: one am-seg prepared, or a training_<date>_<time> directory",
    )
    which = choose.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--epoch", type=int, metavar="N", help="The epoch to mark; it must have been validated",
    )
    which.add_argument("--clear", action="store_true", help="Take the mark back")


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
    marked: dict[Path, int | None] = {}
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
            marked[written] = _marked(directory)
            _print_run(runs[-1], args.metric, marked[written])
        _print_validations(directory, args.metric)
    inside = {run.directory for run in runs}
    for directory in found:
        if directory not in inside:
            runs.append(read_training_run(directory))
            marked[directory] = _marked(directory)
            _print_run(runs[-1], args.metric, marked[directory])
            _print_validations(directory, args.metric)

    if args.plot:
        if len(runs) != 1:
            logger.error(
                "%d runs have written results here; name one of them to draw its curves",
                len(runs),
            )
            return 2
        try:
            plot_training_run(runs[0], Path(args.plot), metric=args.metric,
                              classes=args.classes, chosen=marked[runs[0].directory],
                              overwrite=args.overwrite)
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


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle ``am-seg validate``."""
    try:
        plan = plan_validation(
            Path(args.run), Path(args.manifest), args.name, rows=args.rows,
            epochs=args.epochs, device=args.device,
            start_path=Path(args.start_path) if args.start_path else None,
        )
    except TrainingRunRefused as refused:
        for reason in refused.reasons:
            logger.error("%s", reason)
        return 2
    for line in plan.describe():
        print(line)
    if args.dry_run:
        print("Dry run: nothing was written or started")
        return 0
    try:
        directory = apply_validation(plan, user=getattr(args, "user", None))
        start_training(directory)
    except (OSError, TrainingNotStarted) as error:
        logger.error("%s", error)
        return 2
    print(f"Started. Follow it with: am-seg show-training {plan.run_directory}")
    print(f"Stop it with: am-seg stop-training {directory}")
    return 0


def _print_validations(run_directory: Path, metric: str) -> None:
    """Print each validation of a run: how it stands and what its last scored epoch says."""
    for directory in find_validations(run_directory):
        try:
            record = read_validation_record(directory)
            status = run_status(directory)
        except ValueError as error:
            logger.error("%s", error)
            continue
        assert record is not None
        scored = find_scores(directory)
        print(
            f"  validation {record.name}: {status.state}, "
            f"{len(scored)} of {len(record.epochs)} epoch(s) scored"
        )
        if status.state == "failed" and status.error:
            print(f"    {status.error}")
        if scored:
            try:
                metrics = metrics_for(read_scores(scored[-1]))
            except ValueError as error:
                logger.error("%s", error)
                continue
            if metrics is not None:
                values = ", ".join(f"{value:.3f}" for value in metrics.per_class[metric])
                print(f"    epoch {metrics.epoch}, {metric} per class: {values}")


def _cmd_compare_training(args: argparse.Namespace) -> int:
    """Handle ``am-seg compare-training``."""
    if _outputs_refused([args.output, args.plot], args.overwrite):
        return 2
    compared: list[ComparedRun] = []
    for path in map(Path, args.paths):
        if not path.is_dir():
            logger.error("Not a directory: %s", path)
            return 2
        prepared = find_run_directories(path, depth=SEARCH_DEPTH)
        homes = {written: home for home in prepared if (written := written_in(home)) is not None}
        for written in find_training_runs(path):
            homes.setdefault(written, written)
        for written, home in sorted(homes.items()):
            name = home.name if home != written or written == path else written.parent.name
            compared.append(ComparedRun(name, read_training_run(written), _marked(home)))
    if not compared:
        logger.error("No training run has written results at %s", ", ".join(args.paths))
        return 2
    try:
        table = compare_runs(
            compared, metric=args.metric, classes=args.classes, value=args.value,
            epoch_range=(args.epochs[0], args.epochs[1]),
        )
        header, cells = table.header(), table.cells()
        widths = [max(len(str(row[column])) for row in (header, *cells)) for column in range(len(header))]
        for row in (header, *cells):
            print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip())
        if args.output:
            write_csv(Path(args.output), header, cells, overwrite=args.overwrite)
        if args.plot:
            plot_comparison(table, Path(args.plot), overwrite=args.overwrite)
    except (RuntimeError, ValueError, OSError) as error:
        logger.error("%s", error)
        return 2
    return 0


def _cmd_choose_best(args: argparse.Namespace) -> int:
    """Handle ``am-seg choose-best``."""
    directory = Path(args.path)
    if args.clear:
        print("Cleared" if clear_chosen_best(directory) else "No epoch was marked")
        return 0
    written = written_in(directory) if directory.is_dir() else None
    if written is None:
        logger.error("No training run has written results at %s", directory)
        return 2
    try:
        chosen = choose_best(
            directory, read_training_run(written), args.epoch, user=getattr(args, "user", None)
        )
    except ValueError as error:
        logger.error("%s", error)
        return 2
    print(f"Epoch {chosen.epoch} is marked as the best of {directory}")
    return 0


def _marked(directory: Path) -> int | None:
    """Return the epoch marked as best for the run in *directory*, one that cannot be read logged."""
    try:
        chosen = read_chosen_best(directory)
    except ValueError as error:
        logger.error("%s", error)
        return None
    return chosen.epoch if chosen else None


def _print_run(run: TrainingRun, metric: str, marked: int | None = None) -> None:
    """Print what one run's files say about it, and the epoch marked as its best."""
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
    if marked is not None:
        kept = run.checkpoint_for(marked)
        print(f"  marked as best: epoch {marked}")
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
    "choose-best": _cmd_choose_best,
    "validate": _cmd_validate,
    "compare-training": _cmd_compare_training,
}
