"""Reporting how far a long operation has got, and asking it to stop.

Every operation here until now was fast enough to run to completion behind a
busy cursor. Inference is not: a run over a project is upwards of an hour, so it
has to say where it is and be stoppable. Both are expressed without Qt, so the
operations layer and the controllers stay testable without a display.

Cancellation is cooperative and lands between whole units of work, never inside
one. That is what makes it safe rather than merely responsive: a prediction is
written atomically and recorded before the next begins, so a stopped run leaves
finished work and an accurate account of it.

:func:`timed` answers the other question about a long operation -- how long its
parts actually take -- so that a decision to make one faster is taken against
measurements rather than against a guess about which part is slow.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class OperationCancelled(Exception):
    """Raised inside a running operation when the user has asked it to stop.

    Raised by the progress callback rather than returned, so an operation needs
    no cancellation branch of its own: it reports progress, and stopping is one
    of the things that can happen when it does.
    """


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
