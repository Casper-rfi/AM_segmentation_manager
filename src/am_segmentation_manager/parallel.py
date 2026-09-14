"""Running one function over many inputs in worker processes kept for the session.

Supports the layers rather than being one: a read that is mostly Python holds the
GIL, so threads wait on one another and processes do not. The pool starts on first
use with the *spawn* method, a fresh interpreter inheriting nothing Qt has touched,
and lasts until this one exits. **A function sent here is module-level and imports
nothing from ``gui/`` or ``ml/``**; what it raises for one input is raised by the
whole map, as it would be here. What a worker logs at WARNING or above is logged
again here.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: How many worker processes the pool holds, leaving two cores for the window and
#: the disk: more workers than that read headers more slowly, not faster.
POOL_WORKERS = max(1, (os.cpu_count() or 1) - 2)

#: Inputs sent to a worker at a time. Smaller chunks spend the gain on messaging;
#: larger ones are no faster.
CHUNK_SIZE = 64

#: Below this many inputs the work runs in this process, finishing before a pool
#: not yet started would have.
MIN_POOL_ITEMS = 256

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()

#: Set the first time a pool breaks, after which every batch runs in this
#: process: what broke one pool, such as a script without a ``__main__`` guard,
#: breaks the next.
_pool_broken = False


def map_in_processes(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    chunk_size: int | None = None,
    min_items: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Return ``[function(item) for item in items]``, computed in worker processes.

    Order is kept. *chunk_size* and *min_items* default to :data:`CHUNK_SIZE` and
    :data:`MIN_POOL_ITEMS`, which suit an input costing about what a header does.
    *on_progress* is called as ``on_progress(done, total)`` as results come back.
    A script calling this needs the ``if __name__ == "__main__":`` guard that
    spawning requires. If the pool breaks -- a worker killed from outside, or a
    script without that guard -- what is left of the batch, and every batch
    after it, runs in this process under one warning rather than being lost.
    """
    global _pool_broken
    items = list(items)
    results: list[R] = []
    if _pool_broken or len(items) < (MIN_POOL_ITEMS if min_items is None else min_items):
        _map_here(function, items, results, on_progress)
        return results
    size = CHUNK_SIZE if chunk_size is None else chunk_size
    chunks = [items[start:start + size] for start in range(0, len(items), size)]
    try:
        for chunk_results, records in _session_pool().map(
            _run_chunk, [function] * len(chunks), chunks
        ):
            for record in records:
                _log_again(record)
            results.extend(chunk_results)
            if on_progress is not None:
                on_progress(len(results), len(items))
    except BrokenProcessPool:
        logger.warning(
            "The worker processes stopped; finishing %d item(s) in this process, "
            "where every later batch runs too",
            len(items) - len(results),
        )
        _pool_broken = True
        shutdown_pool()
        _map_here(function, items, results, on_progress)
    return results


def _map_here(
    function: Callable[[T], R],
    items: list[T],
    results: list[R],
    on_progress: Callable[[int, int], None] | None,
) -> None:
    """Extend *results* with *function* applied here to the items it lacks."""
    for item in items[len(results):]:
        results.append(function(item))
        if on_progress is not None:
            on_progress(len(results), len(items))


def shutdown_pool() -> None:
    """Stop the worker processes; the next batch large enough starts new ones.

    Unless the pool has broken, in which case none is started again.
    """
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


def _session_pool() -> ProcessPoolExecutor:
    """Return the session's pool, starting it on first use."""
    global _pool
    with _pool_lock:
        if _pool is None:
            logger.info("Starting %d worker process(es)", POOL_WORKERS)
            _pool = ProcessPoolExecutor(
                POOL_WORKERS, mp_context=multiprocessing.get_context("spawn")
            )
        return _pool


def _run_chunk(
    function: Callable[[T], R], chunk: list[T]
) -> tuple[list[R], list[logging.LogRecord]]:
    """Apply *function* to each input in a worker, returning what it logged too."""
    captured = _CapturedRecords()
    root = logging.getLogger()
    root.addHandler(captured)
    try:
        return [function(item) for item in chunk], captured.records
    finally:
        root.removeHandler(captured)


class _CapturedRecords(logging.Handler):
    """Holds a worker's log records in a form that can be sent back."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        # The arguments and the traceback need not pickle; their text does.
        if record.exc_info:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        record.msg, record.args, record.exc_info = record.getMessage(), None, None
        self.records.append(record)


def _log_again(record: logging.LogRecord) -> None:
    """Hand a worker's record to this process's logger of the same name."""
    target = logging.getLogger(record.name)
    if target.isEnabledFor(record.levelno):
        target.handle(record)
