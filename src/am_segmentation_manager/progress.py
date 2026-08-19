"""Reporting how far a long operation has got, and asking it to stop.

Every operation here until now was fast enough to run to completion behind a
busy cursor. Inference is not: a run over a project is upwards of an hour, so it
has to say where it is and be stoppable. Both are expressed without Qt, so the
operations layer and the controllers stay testable without a display.

Cancellation is cooperative and lands between whole units of work, never inside
one. That is what makes it safe rather than merely responsive: a prediction is
written atomically and recorded before the next begins, so a stopped run leaves
finished work and an accurate account of it.
"""
from __future__ import annotations

import threading


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
