"""Copying a run's checkpoints out, and deleting the ones not wanted.

A checkpoint is copied, never moved, and never over a file. **Nothing is deleted
until every check has passed**: never the epoch marked as best, never while the run
or a validation of it may still be writing, never when a kept epoch has no
checkpoint. Only that run's checkpoints are touched; its metrics, lists and records stay.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Collection
from pathlib import Path

from ..io.atomic import write_atomically
from .choice import read_chosen_best
from .listing import prepared_name
from .preparation import read_run_record
from .process import run_status
from .runs import Checkpoint, is_run_directory, read_training_run, written_in, written_recently
from .validation import find_validations

logger = logging.getLogger(__name__)


def copy_checkpoint(checkpoint: Checkpoint, destination: Path) -> Path:
    """Copy a checkpoint out of its run to *destination*, to predict with or share.

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


def other_checkpoints(
    directory: Path, keep: Collection[int], keep_run_best: bool = False
) -> list[Checkpoint]:
    """Return the checkpoints of the run in *directory* that :func:`remove_other_checkpoints` would delete.

    **The epoch marked as best is never among them**, whatever *keep* says: the mark
    is cleared first by whoever means to delete it.

    Args:
        directory: A run am-seg prepared, or a directory AM-model-training wrote.
        keep_run_best: Whether the epoch the run counts as best is kept as well.

    Raises:
        ValueError: If the run cannot be read or may still be writing, no epoch
            is kept, a kept epoch has no checkpoint, or the mark cannot be read.
    """
    chosen = read_chosen_best(directory)
    written_directory = written_in(directory)
    written = read_training_run(written_directory) if written_directory else None
    if is_run_directory(directory):
        name = directory.name
        training = written is not None and not written.finished and written_recently(directory)
    else:
        name = prepared_name(directory, read_run_record(directory))
        training = run_status(directory).live
    if training or any(_validation_live(validation) for validation in find_validations(directory)):
        raise ValueError(
            f"{name} may still be training, or being validated; its checkpoints are "
            f"kept until that has ended"
        )
    kept = set(keep)
    if not kept:
        raise ValueError("Choose at least one checkpoint to keep")
    checkpoints = written.checkpoints if written else ()
    missing = sorted(kept - {checkpoint.epoch for checkpoint in checkpoints})
    if missing:
        raise ValueError(f"{name} has no checkpoint for epoch(s) {', '.join(map(str, missing))}")
    if chosen is not None:
        kept.add(chosen.epoch)
    best = written.best() if written else None
    if keep_run_best and best is not None:
        kept.add(best.epoch)
    return [checkpoint for checkpoint in checkpoints if checkpoint.epoch not in kept]


def remove_other_checkpoints(
    directory: Path, keep: Collection[int], keep_run_best: bool = False
) -> list[Path]:
    """Delete every checkpoint of the run in *directory* but those of the epochs in *keep*.

    **Called only once the person has been shown what goes and said yes**: a
    deleted checkpoint cannot be got back.

    Returns:
        The files deleted.

    Raises:
        ValueError: As :func:`other_checkpoints`, before anything is deleted.
    """
    doomed = [checkpoint.path for checkpoint in other_checkpoints(directory, keep, keep_run_best)]
    for path in doomed:
        path.unlink()
        logger.info("Deleted the checkpoint %s", path)
    return doomed


def _validation_live(directory: Path) -> bool:
    """Return whether the validation in *directory* may still be scoring; one that cannot be read is not."""
    try:
        return run_status(directory).live
    except ValueError:
        return False
