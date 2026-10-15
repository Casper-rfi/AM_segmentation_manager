"""Running one batch operation off the interface thread.

Qt draws nothing while the widgets' thread is busy, so work measured in minutes
runs here, reporting through :attr:`OperationWorker.progressed` and stopped by a
:class:`~am_segmentation_manager.progress.Cancellation` its closure polls. The
worker lives on the thread and the plain object owns it, as in :mod:`.loader`.
**No Python object rides a queued signal in either direction**: the work and its
outcome pass through queues, and the worker lets go of both before signalling.
"""
from __future__ import annotations

import logging
import queue
from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QMetaObject, QObject, Qt, QThread, Signal, Slot

from .loader import keep_alive

logger = logging.getLogger(__name__)

#: How long an operation is given to finish before the window stops waiting for
#: it. Far longer than the loader's, because the work here is measured in
#: minutes and a cancellation is only polled between units.
STOP_TIMEOUT_MS = 30000


class OperationWorker(QObject):
    """Calls a function on a worker thread and reports what came back."""

    #: Emitted once the callable has returned; its result waits in
    #: :meth:`take_outcome`.
    finished = Signal()

    #: Emitted from the worker thread as a long operation advances, with the
    #: units done and the units expected. Connected across threads, so the
    #: window updates on its own thread rather than from inside the work.
    progressed = Signal(int, int)

    #: Emitted once the callable has raised; the exception waits in
    #: :meth:`take_outcome`. Failures come back as a signal rather than
    #: propagating, because an exception raised on a worker thread reaches
    #: nothing that could report it to the user.
    failed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._jobs: queue.SimpleQueue[Callable[[], Any]] = queue.SimpleQueue()
        self._outcomes: queue.SimpleQueue[Any] = queue.SimpleQueue()

    def hand_over(self, work: Callable[[], Any]) -> None:
        """Queue *work* for the next :meth:`run`, from any thread."""
        self._jobs.put(work)

    def take_outcome(self) -> Any:
        """Return what the oldest reported run returned, or the exception it raised.

        One call per :attr:`finished` or :attr:`failed`, on the thread those
        reach.
        """
        return self._outcomes.get_nowait()

    @Slot()
    def run(self) -> None:
        """Call the oldest handed-over work, then emit whether it succeeded."""
        if self._call(self._jobs.get_nowait()):
            self.finished.emit()
        else:
            self.failed.emit()

    def _call(self, work: Callable[[], Any]) -> bool:
        """Call *work*, queue what came of it, and return whether it succeeded.

        A method of its own so that *work* and its result are released when it
        returns -- before :meth:`run` emits, and so before the receiving thread
        can drop the last reference and leave it to be freed here.
        """
        try:
            self._outcomes.put(work())
        except Exception as error:  # reported, not swallowed
            logger.exception("Operation failed")
            self._outcomes.put(error)
            return False
        return True


class BackgroundOperation:
    """Owns the worker and the thread it runs on.

    One thread for the window's lifetime rather than one per operation: only one
    operation runs at a time, and starting a thread per run would leave the
    window's shutdown depending on which of them had finished.
    """

    def __init__(self) -> None:
        self._thread = QThread()
        self._thread.setObjectName("operations")
        self.worker = OperationWorker()
        self.worker.moveToThread(self._thread)
        self._thread.start()

    def submit(self, work: Callable[[], Any]) -> None:
        """Run *work* on the thread; :attr:`OperationWorker.finished` or
        :attr:`OperationWorker.failed` says when its outcome is ready.

        The caller keeps its own reference to *work* until it has taken the
        outcome, so the closure -- and whatever it captured -- is freed on the
        caller's thread.
        """
        self.worker.hand_over(work)
        QMetaObject.invokeMethod(self.worker, "run", Qt.ConnectionType.QueuedConnection)

    def stop(self) -> None:
        """Shut the thread down, waiting for any running operation to finish.

        An operation still running at the deadline is left to finish and this
        object goes with it, for the reason :func:`.loader.keep_alive` gives.
        """
        self._thread.quit()
        if not self._thread.wait(STOP_TIMEOUT_MS):
            keep_alive(self)
