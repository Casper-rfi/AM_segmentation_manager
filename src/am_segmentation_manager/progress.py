"""Reporting how far a long operation has got, and asking it to stop.

Expressed without Qt, so the operations layer and the controllers test without a
display. Cancellation is cooperative and lands between whole units of work, never
inside one: a prediction is written atomically and recorded before the next
begins, so a stopped run leaves finished work and an accurate account of it --
which a run stopped by a failure hands back as :class:`StoppedPartway`.
:func:`timed` says how long an operation's parts take, so making one faster is
decided against measurements.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


class OperationCancelled(Exception):
    """Raised inside a running operation when the user has asked it to stop.

    Raised by the progress callback rather than returned, so an operation needs
    no cancellation branch of its own: it reports progress, and stopping is one
    of the things that can happen when it does.
    """


class StoppedPartway(RuntimeError):
    """Raised when an operation stopped on a failure after writing some of its files.

    Carries what was written, so a caller holding records can name those files
    on them rather than leaving the collection's picture of itself behind what
    is on disk.

    Attributes:
        written: Every file written before the failure, in the order written.
    """

    def __init__(self, message: str, written: Iterable[Path] = ()) -> None:
        super().__init__(message)
        self.written: tuple[Path, ...] = tuple(written)


class Cancellation:
    """A request to stop, set on one thread and read on another.

    Wraps :class:`threading.Event` rather than using a plain flag, because the
    two threads involved are the interface and the worker; and rather than a Qt
    signal, because a queued signal is delivered when the receiving thread next
    runs its event loop, which for a worker inside a long call is never.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def requested(self) -> bool:
        """Return whether stopping has been asked for."""
        return self._event.is_set()

    def request(self) -> None:
        """Ask the operation to stop at its next checkpoint."""
        self._event.set()

    def check(self) -> None:
        """Stop the operation if stopping has been asked for.

        Raises:
            OperationCancelled: If :meth:`request` has been called.
        """
        if self._event.is_set():
            raise OperationCancelled()


@contextmanager
def timed(log: logging.Logger, phase: str, level: int = logging.INFO) -> Iterator[None]:
    """Log how long *phase* took, reporting a failed phase as well as a finished one.

    Args:
        log: The module logger of whatever is being timed, so the line is
            attributed where the work is.
        phase: What is being timed, as a noun phrase for the log line.
        level: Kept at INFO for a phase worth seeing without ``-v``.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        log.log(level, "%s: %.2fs", phase, time.perf_counter() - start)
