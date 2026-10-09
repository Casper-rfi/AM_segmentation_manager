"""The epoch a person counts as a run's best, where it is not the one the run does.

``am_training_choice.yaml`` in the run's directory holds the epoch, who chose it and
when; a run without the file has no choice made. **Nothing else reads or writes that
file**: :func:`read_chosen_best` is the one answer, :func:`choose_best` and
:func:`clear_chosen_best` the two changes. What AM-model-training wrote is never touched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ..history import resolve_author
from ..io.yaml import read_yaml, version_problem, write_yaml
from ..training_runs import TrainingRun

logger = logging.getLogger(__name__)

#: The choice, in the directory of the run it is about.
CHOICE_FILENAME: Final[str] = "am_training_choice.yaml"

#: Written into every choice, so a reader can tell a layout it does not know.
FILE_VERSION: Final[int] = 1

_HEADER: Final[str] = """\
# The epoch of this training run that a person counts as its best, written by
# am-seg. The run's own best is computed from its metrics; this is kept beside it,
# not in place of it. Delete this file to have no choice made.
"""


@dataclass(frozen=True)
class ChosenBest:
    """The epoch chosen as a run's best, by whom and when."""

    epoch: int
    chosen_by: str = ""
    chosen_at: str = ""


def read_chosen_best(directory: Path) -> ChosenBest | None:
    """Return the epoch chosen as best for the run in *directory*, or None where none is.

    Raises:
        ValueError: If the file cannot be understood, or was written by a later am-seg.
    """
    path = directory / CHOICE_FILENAME
    if not path.is_file():
        return None
    try:
        document = read_yaml(path)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    problem = version_problem(document.get("version"), path, FILE_VERSION, required=True)
    if problem is not None:
        raise ValueError(problem)
    epoch = document.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError(f"{path} names no epoch")
    return ChosenBest(
        epoch, str(document.get("chosen_by") or ""), str(document.get("chosen_at") or "")
    )


def choose_best(
    directory: Path, run: TrainingRun, epoch: int, user: str | None = None
) -> ChosenBest:
    """Record *epoch* as the best of the run in *directory*, in place of any earlier choice.

    Args:
        run: What the run has written, which says the epochs there are to choose from.
        user: The name to record, resolved as a history entry's author is.

    Raises:
        ValueError: If the run did not validate *epoch*; nothing is written.
    """
    validated = [entry.epoch for entry in run.val]
    if epoch not in validated:
        said = f"{validated[0]} to {validated[-1]}" if validated else "none"
        raise ValueError(
            f"Epoch {epoch} was not validated by this run (validated: {said}), "
            f"so it has no score to be best by"
        )
    by, _ = resolve_author(user)
    chosen = ChosenBest(
        epoch, by or "", datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_yaml(
        directory / CHOICE_FILENAME,
        {"version": FILE_VERSION, "epoch": chosen.epoch,
         "chosen_by": chosen.chosen_by, "chosen_at": chosen.chosen_at},
        header=_HEADER,
    )
    logger.info("Chose epoch %d as the best of %s", epoch, directory)
    return chosen


def clear_chosen_best(directory: Path) -> bool:
    """Take back the choice for the run in *directory*, returning whether there was one."""
    path = directory / CHOICE_FILENAME
    if not path.is_file():
        return False
    path.unlink()
    logger.info("Cleared the chosen best of %s", directory)
    return True
