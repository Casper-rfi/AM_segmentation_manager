"""The log file a command writes when it is asked for one, or is the window.

Separate from the commands because opening it happens before any of them runs:
a crash dump has to have somewhere to go before the work that might crash
starts.
"""
from __future__ import annotations

import faulthandler
import logging
from pathlib import Path
from typing import TextIO

from ..history import CONFIG_DIRECTORY

logger = logging.getLogger(__name__)

#: The open log file, held for the life of the process. ``faulthandler`` keeps
#: only the file descriptor, so a file closed when the function that opened it
#: returned would send the crash dump nowhere.
_LOG_FILE: TextIO | None = None

#: How a line in the log file is written. Richer than the stdout format, which
#: stays bare because that is the command's output: a crash report needs to say
#: when something happened and on which thread, and Qt's worker threads are
#: foreign to Python, so they appear under their ``Dummy-N`` names rather than
#: the object names Qt gives them.
LOG_FILE_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"

#: Where the window logs when nothing else was asked for. Under the user's home
#: rather than beside the data: a log is this package's own record of a session,
#: not part of any collection being reviewed.
#: Where the window's own log goes. Under `CONFIG_DIRECTORY` rather than
#: spelling that path a second time, so the two cannot drift apart.
DEFAULT_GUI_LOG = CONFIG_DIRECTORY / "gui.log"

#: How large a rolled log may grow before the previous run's file is moved
#: aside. Only the window's own log is capped -- a path the user named is left
#: to accumulate, that being what naming it asks for.
LOG_FILE_MAX_BYTES = 5_000_000


def open_log_file(path: Path, *, max_bytes: int | None = None) -> logging.Handler:
    """Return a handler writing to *path*, with crash dumps going there too.

    The Python log and the native stack trace share one file on purpose: a
    stack trace alone does not say what the process was doing, and the log line
    before it usually does. Records are flushed as they are written and
    ``faulthandler`` writes to the descriptor directly, so the two interleave
    in the order they happened.

    Appends, so several runs accumulate rather than the last one erasing the
    crash before it.

    Args:
        max_bytes: Move the file to ``<name>.1`` once it reaches this size,
            replacing any previous ``.1``. Omit to let it grow without limit,
            which is right for a file the user asked for by name.

    Raises:
        OSError: If *path* cannot be opened for writing.
    """
    global _LOG_FILE

    path.parent.mkdir(parents=True, exist_ok=True)
    _roll_if_large(path, max_bytes)
    _LOG_FILE = path.open("a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=_LOG_FILE, all_threads=True)

    handler = logging.StreamHandler(_LOG_FILE)
    handler.setFormatter(logging.Formatter(LOG_FILE_FORMAT))
    return handler


def _roll_if_large(path: Path, max_bytes: int | None) -> None:
    """Move *path* aside when it has grown past *max_bytes*.

    One generation, not a series: what makes a log worth keeping is the crash
    at the end of it, and two files is enough to survive a session that starts
    fresh. A log that cannot be rolled is left to grow rather than refusing to
    open -- losing the session's log is worse than a large file.
    """
    if max_bytes is None:
        return
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        path.replace(path.with_name(path.name + ".1"))
    except OSError as error:
        logger.debug("Could not roll %s aside: %s", path, error)


