"""The end of a run's log, made readable, and what a failure it records means.

Reads ``train.log`` and writes nothing. A progress bar reaches a file as a line for
every update, two a second, so the log's end is one bar many times over;
:func:`fold_progress` keeps the last state of each, and leaves out the table of
metrics an epoch ends with, which the curves draw. **Only the end of the file is
read**, a run's log growing for as long as it trains.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

from .operations.training_process import LOG_FILENAME

logger = logging.getLogger(__name__)

#: How many lines of a log's end are shown, and how much of the file is read to
#: find them: an epoch of several hundred steps is tens of kilobytes of bar.
TAIL_LINES: Final[int] = 200
TAIL_BYTES: Final[int] = 128 * 1024

#: A terminal's control sequences, which a bar moves the cursor with.
_ESCAPE: Final = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

#: A progress bar as tqdm writes one: what it is of, how far, how fast.
_BAR: Final = re.compile(r"(?P<name>[^|\r\n]*?):\s*\d+%\|[^|]*\|\s*\d+/\d+ \[[^\]]*\]")

#: One line of the table of metrics an epoch ends with.
_METRIC: Final = re.compile(r"\w+: -?\d+(\.\d+)?(e-?\d+)?")


def fold_progress(text: str) -> list[str]:
    """Return the lines of *text* a person would read, each progress bar once, as it last stood.

    Control sequences, blank lines and an epoch's table of metrics are left
    out; whatever was written on the end of a bar's line gets a line of its own.
    """
    folded: list[str] = []
    last_bar = ""
    for raw in _ESCAPE.sub("", text).replace("\r", "\n").split("\n"):
        line = raw.strip()
        while line:
            bar = _BAR.match(line)
            if bar is None:
                if not _METRIC.fullmatch(line):
                    folded.append(line)
                    last_bar = ""
                break
            name = bar["name"]
            if folded and last_bar == name:
                folded[-1] = bar[0]
            else:
                folded.append(bar[0])
            last_bar = name
            line = line[bar.end():].strip()
    return folded


def read_log_tail(
    directory: Path, lines: int = TAIL_LINES, most_bytes: int = TAIL_BYTES
) -> list[str]:
    """Return the end of the log of the run, or validation, in *directory*, folded.

    Empty where there is no log or it cannot be read: a run am-seg did not
    start has none.

    Args:
        lines: The most lines to return.
        most_bytes: How much of the file's end to read.
    """
    path = directory / LOG_FILENAME
    try:
        with path.open("rb") as log:
            size = log.seek(0, 2)
            log.seek(max(0, size - most_bytes))
            read = log.read()
    except OSError as error:
        logger.debug("No log read from %s: %s", path, error)
        return []
    text = read.decode("utf-8", errors="replace")
    if size > most_bytes:
        # The read began part-way through a line.
        text = text.partition("\n")[2]
    return fold_progress(text)[-lines:]


def failure_advice(error: str, training_batch_size: int | None = None) -> str:
    """Return what to do about a run that failed with *error*, or "" where nothing is known.

    Args:
        error: Why it failed, as its process recorded it.
        training_batch_size: The batch it trained with, where that is known.
    """
    if "out of memory" not in error.lower():
        return ""
    batch = f" with batches of {training_batch_size}" if training_batch_size else ""
    if training_batch_size == 1:
        return (
            "The GPU ran out of memory with batches of 1, the smallest there is: "
            "this model needs a GPU with more memory, or one nothing else is using."
        )
    return (
        f"The GPU ran out of memory{batch}. Start a new run with a smaller Training "
        f"batch, under Advanced settings, or on a GPU with more memory free: Start... "
        f"says which batch is likely to fit."
    )
