"""Background preparation of frames.

Reading an acquisition or a summary plot is slow, and a cloud-synced folder can
block for far longer, so none of it happens on the thread that draws the
interface: it happens here, on a worker thread that does nothing else. The
worker takes requests, hands them to the cache, and reports what finished;
deciding *what* to prepare belongs to the controller. Requests reach it through
a queue and its signals carry numbers and text, so no Python object rides a
queued signal; ``gui/worker.py`` says why.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path

from qtpy.QtCore import QMetaObject, QObject, Qt, QThread, Signal, Slot

from ..operations.label_measure import count_specks_in, find_specks
from .cache import FrameCache, LoadRequest
from .label_session import decode_label

logger = logging.getLogger(__name__)

#: How long the loader is given to finish what it is reading before the window
#: stops waiting for it. A read is one call into an image library and cannot be
#: interrupted part-way, so this is a limit on how long closing the window may
#: take, not on the read.
STOP_TIMEOUT_MS = 5000

#: Things kept alive because a thread of theirs is still running. Never read
#: again; holding the reference is the whole point.
_ABANDONED: list[object] = []


def keep_alive(owner: object) -> None:
    """Hold *owner* for the life of the process, its thread having outlived it.

    A ``QThread`` that has not stopped keeps calling into its Python worker.
    Letting the interpreter collect that worker while the thread runs is a use
    after free, and the crash lands on the worker thread with no traceback to
    say why -- which is exactly how it has been seen. Leaking one object per
    abandoned thread is the cheap end of that trade.
    """
    logger.warning("A worker thread outlived its owner; leaving it to finish")
    _ABANDONED.append(owner)


class FrameLoadWorker(QObject):
    """Prepares frames on a worker thread, newest request first.

    Requests carry a *token*. When the user moves somewhere else, the interface
    raises the token, and any batch still running for an older token stops at
    its next item. Without this, scrubbing a slider would queue up preparation
    work for frames nobody is looking at any more, and the frame actually on
    screen would wait behind it.
    """

    #: Emitted after each request completes, with the token it belonged to and
    #: the frame it prepared.
    loaded = Signal(int, int)

    #: Emitted when a batch runs to completion without being superseded.
    batch_finished = Signal(int)

    #: Emitted after each label is counted, with the token, the label's key as
    #: text and how many specks it holds. One signal per label rather than one
    #: per batch, so the marks appear as the sweep goes rather than all at the
    #: end.
    counted = Signal(int, str, int)

    def __init__(self, cache: FrameCache) -> None:
        super().__init__()
        self._cache = cache
        self._token = 0
        self._lock = threading.Lock()
        self._jobs: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()

    def hand_over(self, job: Callable[[], None]) -> None:
        """Queue *job* for the next :meth:`run_next`, from any thread."""
        self._jobs.put(job)

    @Slot()
    def run_next(self) -> None:
        """Run the oldest handed-over job."""
        self._jobs.get_nowait()()

    def set_token(self, token: int) -> None:
        """Supersede any batch older than *token*. Called from the UI thread."""
        with self._lock:
            self._token = token

    def _is_current(self, token: int) -> bool:
        with self._lock:
            return token == self._token

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
            self.loaded.emit(token, request.frame_index)
        self.batch_finished.emit(token)

    def count_specks(self, labels: list, token: int, at_most: int) -> None:
        """Count the specks in each ``(key, path)``, stopping early if superseded.

        Reads the label files and nothing else. Going through the frame cache
        would prepare each acquisition at full resolution to get at the label
        beside it, which is many times the cost of the answer and would evict
        the frames the reviewer is actually working on.

        **A source is a path, or the bytes of a label being edited.** A frame
        with unsaved strokes no longer matches its file, and counting the file
        would report specks the reviewer has already painted out -- so the
        session's own copy comes through instead, decoded here rather than on
        the interface thread.

        A label that will not open is reported as holding none: it is already
        drawn as unreadable on the label line, and a count is not the place to
        raise it a second time.
        """

        for key, source in labels:
            if not self._is_current(token):
                logger.debug("Abandoning superseded speck count %d", token)
                return
            try:
                if isinstance(source, bytes):
                    found = len(find_specks(decode_label(source), at_most=at_most))
                else:
                    found = count_specks_in(source, at_most=at_most)
            except Exception:  # noqa: BLE001 - a bad label must not stop the sweep
                logger.exception("Could not count specks in %s", key)
                found = 0
            self.counted.emit(token, str(key), found)


class BackgroundLoader:
    """Owns the worker and the thread it runs on.

    Keeps the thread's lifecycle in one place so the window does not have to
    manage it, and guarantees the thread is stopped on shutdown. Work is
    submitted through :meth:`load_batch` and :meth:`count_specks`, which return
    at once and are run on the thread in the order they were called.
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

    def load_batch(self, requests: list[LoadRequest], token: int) -> None:
        """Prepare *requests* on the thread; see :meth:`FrameLoadWorker.load_batch`."""
        self._submit(partial(self.worker.load_batch, requests, token))

    def count_specks(
        self, labels: list[tuple[Path, Path | bytes]], token: int, at_most: int
    ) -> None:
        """Count *labels*' specks on the thread; see :meth:`FrameLoadWorker.count_specks`."""
        self._submit(partial(self.worker.count_specks, labels, token, at_most))

    def _submit(self, job: Callable[[], None]) -> None:
        """Queue *job* and ask the thread to run it, carrying nothing in the call."""
        self.worker.hand_over(job)
        QMetaObject.invokeMethod(self.worker, "run_next", Qt.ConnectionType.QueuedConnection)

    def stop(self) -> None:
        """Stop accepting work and shut the thread down.

        Raising the token stops the batch at its next item, but the read
        already running cannot be interrupted: it is a single call into an
        image library, and a file on a cloud-synced volume can hold it for far
        longer than anyone will wait for a window to close. A thread still
        inside one at the deadline is left to finish, and this loader goes with
        it rather than being collected out from under it.
        """
        self.next_token()
        self._thread.quit()
        if not self._thread.wait(STOP_TIMEOUT_MS):
            keep_alive(self)
