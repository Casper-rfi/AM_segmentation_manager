"""Background preparation of frames.

Reading an acquisition takes ~200 ms cold and a summary plot ~575 ms, and a
cloud-synced folder can block for far longer. Doing any of that on the thread
that draws the interface would freeze the window, so all of it happens here, on
a worker thread that does nothing else.

The worker is a thin shell: it takes requests, hands them to the cache, and
reports what finished. Deciding *what* to prepare belongs to the controller.
"""
from __future__ import annotations

import logging
import threading

from qtpy.QtCore import QObject, QThread, Signal, Slot

from .cache import FrameCache, LoadRequest

logger = logging.getLogger(__name__)


class FrameLoadWorker(QObject):
    """Prepares frames on a worker thread, newest request first.

    Requests carry a *token*. When the user moves somewhere else, the interface
    raises the token, and any batch still running for an older token stops at
    its next item. Without this, scrubbing a slider would queue up preparation
    work for frames nobody is looking at any more, and the frame actually on
    screen would wait behind it.
    """

    #: Emitted after each request completes, with the token it belonged to.
    loaded = Signal(int, object)

    #: Emitted when a batch runs to completion without being superseded.
    batch_finished = Signal(int)

    def __init__(self, cache: FrameCache) -> None:
        super().__init__()
        self._cache = cache
        self._token = 0
        self._lock = threading.Lock()

    def set_token(self, token: int) -> None:
        """Supersede any batch older than *token*. Called from the UI thread."""
        with self._lock:
            self._token = token

    def _is_current(self, token: int) -> bool:
        with self._lock:
            return token == self._token

    @Slot(list, int)
    def load_batch(self, requests: list[LoadRequest], token: int) -> None:
        """Prepare each request in turn, stopping early if superseded."""
        for request in requests:
            if not self._is_current(token):
                logger.debug("Abandoning superseded batch %d", token)
                return
            try:
                self._cache.ensure(request)
            except Exception:  # noqa: BLE001 - the cache records per-frame errors
                logger.exception("Unexpected failure preparing %s", request)
            self.loaded.emit(token, request)
        self.batch_finished.emit(token)


class BackgroundLoader:
    """Owns the worker and the thread it runs on.

    Keeps the thread's lifecycle in one place so the window does not have to
    manage it, and guarantees the thread is stopped on shutdown.

    Work is submitted by connecting a signal on the window to
    :meth:`FrameLoadWorker.load_batch`. Because the worker lives on another
    thread, Qt delivers that call there and the emit returns immediately.
    """

    def __init__(self, cache: FrameCache) -> None:
        self._thread = QThread()
        self._thread.setObjectName("frame-loader")
        self.worker = FrameLoadWorker(cache)
        self.worker.moveToThread(self._thread)
        self._thread.start()
        self._token = 0

    @property
    def token(self) -> int:
        """Return the token of the most recent batch."""
        return self._token

    def next_token(self) -> int:
        """Supersede any running batch and return the token for the next one."""
        self._token += 1
        self.worker.set_token(self._token)
        return self._token

    def stop(self) -> None:
        """Stop accepting work and shut the thread down."""
        self.next_token()
        self._thread.quit()
        self._thread.wait(5000)
