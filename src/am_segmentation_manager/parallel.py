"""Running one function over many inputs in worker processes kept for the session.

Supports the layers rather than being one: a read that is mostly Python holds the
GIL, so threads wait on one another and processes do not. The pool starts on first
use with the *spawn* method, a fresh interpreter inheriting nothing Qt has touched,
and lasts until this one exits. **A function sent here is module-level and imports
nothing from ``gui/`` or ``ml/``**; what it raises for one input is raised by the
whole map, as it would be here. What a worker logs at WARNING or above is logged
again here. How many workers it holds is :func:`worker_count`'s answer, read once.
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import os
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: Names a worker count to use instead of the default, as a whole number.
WORKERS_ENV_VAR = "AM_SEG_WORKERS"

#: The most workers started without being asked. The reference collection's
#: 37,779 headers read in 5.2 s with 4 workers, 4.5 s with 6, and 4.3 s with 8 and
#: with 12: past this, gathering results and the disk set the pace, and a further
#: worker only holds memory the other users of a shared machine want.
MAX_DEFAULT_WORKERS = 8

#: The most workers Windows lets one pool start.
WINDOWS_MAX_WORKERS = 61


def available_cpus() -> int:
    """Return how many CPUs this process may run on, rather than how many the machine has.

    A restriction set by a scheduler or ``taskset`` is honoured wherever the
    platform reports one; a container's CPU quota is not reported, and so not seen.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def worker_count(
    environ: Mapping[str, str] | None = None,
    cpus: int | None = None,
    platform: str | None = None,
) -> int:
    """Return how many worker processes the pool should hold.

    Every CPU this process may use but two, which are left for the window and
    the disk, up to :data:`MAX_DEFAULT_WORKERS` -- unless :data:`WORKERS_ENV_VAR`
    names another whole number. A value that is not one is warned about and the
    default used, so a typing slip slows a scan rather than stopping it. Never
    fewer than one, nor more than :data:`WINDOWS_MAX_WORKERS` on Windows.

    Args:
        environ: The environment to read; this process's by default.
        cpus: The CPUs available; :func:`available_cpus` by default.
        platform: As ``sys.platform``, which it defaults to.
    """
    environ = os.environ if environ is None else environ
    cpus = available_cpus() if cpus is None else cpus
    platform = sys.platform if platform is None else platform

    count = max(1, min(MAX_DEFAULT_WORKERS, cpus - 2))
    requested = environ.get(WORKERS_ENV_VAR, "").strip()
    if requested:
        try:
            asked = int(requested)
        except ValueError:
            asked = 0
        if asked >= 1:
            count = asked
        else:
            logger.warning(
                "%s=%r is not a whole number of 1 or more; using %d worker process(es)",
                WORKERS_ENV_VAR, requested, count,
            )
    if platform == "win32" and count > WINDOWS_MAX_WORKERS:
        logger.warning(
            "Windows allows %d worker processes, not %d; using %d",
            WINDOWS_MAX_WORKERS, count, WINDOWS_MAX_WORKERS,
        )
        count = WINDOWS_MAX_WORKERS
    return count


#: How many worker processes the pool holds. None until the pool first starts,
#: when :func:`worker_count` decides it for the session -- not at import, which
#: every worker repeats, each warning again about a count it was given.
POOL_WORKERS: int | None = None

#: The most inputs sent to a worker at a time. Smaller chunks spend the gain on
#: messaging; larger ones are no faster.
CHUNK_SIZE = 64

#: Chunks each worker is given of a batch, at least, where the batch allows: with
#: one each, a worker given a slow chunk holds up a batch the rest have finished.
CHUNKS_PER_WORKER = 4

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

    Order is kept. *chunk_size* defaults to :func:`chunk_size_for`'s answer and
    *min_items* to :data:`MIN_POOL_ITEMS`, which suit an input costing about what
    a header does.
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
    try:
        pool = _session_pool()
        size = chunk_size if chunk_size is not None else chunk_size_for(len(items), POOL_WORKERS or 1)
        chunks = [items[start:start + size] for start in range(0, len(items), size)]
        for chunk_results, records in pool.map(
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


def chunk_size_for(items: int, workers: int) -> int:
    """Return how many of a batch of *items* to send a worker at a time.

    Small enough that each of *workers* gets :data:`CHUNKS_PER_WORKER` chunks,
    so a batch is shared among all of them, but never more than
    :data:`CHUNK_SIZE` nor fewer than one.
    """
    return max(1, min(CHUNK_SIZE, math.ceil(items / (workers * CHUNKS_PER_WORKER))))


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
    global _pool, POOL_WORKERS
    with _pool_lock:
        if _pool is None:
            if POOL_WORKERS is None:
                POOL_WORKERS = worker_count()
            logger.info(
                "Starting %d worker process(es); %d CPU(s) available, %s changes it",
                POOL_WORKERS, available_cpus(), WORKERS_ENV_VAR,
            )
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
