"""Running one batch operation off the interface thread.

Recompressing a project's labels takes minutes, and Qt draws nothing while the
thread that owns the widgets is busy -- so a window that ran it inline would
look crashed for the whole of it.

Most operations report nothing until they finish, which is right for work
measured in seconds. An operation that runs for an hour emits
:attr:`OperationWorker.progressed` as it goes and is stopped through a
:class:`~am_segmentation_manager.progress.Cancellation`, which the work closure
polls; a queued signal would not do, being delivered only when the worker next
returns to its event loop.

The shape follows :mod:`.loader`: the worker lives on the thread, the plain
object owns the thread, and work is submitted by emitting a signal so the call
is delivered on the other side.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import QObject, QThread, Signal, Slot

from .loader import keep_alive

logger = logging.getLogger(__name__)

#: How long an operation is given to finish before the window stops waiting for
#: it. Far longer than the loader's, because the work here is measured in
#: minutes and a cancellation is only polled between units.
STOP_TIMEOUT_MS = 30000


class OperationWorker(QObject):
    """Calls a function on a worker thread and reports what came back."""

    #: Emitted with whatever the callable returned.
    finished = Signal(object)

    #: Emitted from the worker thread as a long operation advances, with the
    #: units done and the units expected. Connected across threads, so the
    #: window updates on its own thread rather than from inside the work.
    progressed = Signal(int, int)

    #: Emitted with the exception when the callable raised. Failures come back
    #: as a signal rather than propagating, because an exception raised on a
    #: worker thread reaches nothing that could report it to the user.
    failed = Signal(object)

    @Slot(object)
    def run(self, work) -> None:
        """Call *work* and emit its result, or the exception it raised."""
        try:
            result = work()
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            logger.exception("Operation failed")
            self.failed.emit(error)
            return
        self.finished.emit(result)


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

    def stop(self) -> None:
        """Shut the thread down, waiting for any running operation to finish.

        An operation still running at the deadline is left to finish and this
        object goes with it, for the reason :func:`.loader.keep_alive` gives.
        """
        self._thread.quit()
        if not self._thread.wait(STOP_TIMEOUT_MS):
            keep_alive(self)
