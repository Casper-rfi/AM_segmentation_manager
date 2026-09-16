"""Running a function over many inputs in worker processes kept for the session."""
from __future__ import annotations

import json
import logging
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from am_segmentation_manager import parallel
from am_segmentation_manager.image_metadata import read_image_metadata, read_records_metadata
from am_segmentation_manager.models import ImageRecord


@pytest.fixture
def pool(monkeypatch):
    """Send even the smallest batch to a pool of two, and take that pool down after."""
    parallel.shutdown_pool()
    monkeypatch.setattr(parallel, "_pool_broken", False)
    monkeypatch.setattr(parallel, "MIN_POOL_ITEMS", 0)
    monkeypatch.setattr(parallel, "POOL_WORKERS", 2)
    monkeypatch.setattr(parallel, "CHUNK_SIZE", 2)
    yield
    parallel.shutdown_pool()


def _window_modules(_item: object) -> list[str]:
    """Return what this process has imported of the window or a Qt binding."""
    return sorted(
        name for name in sys.modules
        if name.startswith(("PySide", "PyQt", "am_segmentation_manager.gui"))
    )


def _images(directory: Path) -> list[Path]:
    """Write two images with a header, and one file that is not an image at all."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")

    paths = []
    for name, beam in (("sem.tif", "ELECTRON"), ("fib.tif", "ION")):
        path = directory / name
        header = {"image": {"beam_type": beam}, "system": {}, "microscope_state": {}}
        tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8), description=json.dumps(header))
        paths.append(path)
    junk = directory / "junk.tif"
    junk.write_bytes(b"not a tiff")
    return [*paths, junk]


def test_headers_read_in_processes_match_those_read_here(tmp_path: Path, pool):
    paths = _images(tmp_path)
    expected = {path: read_image_metadata(path) for path in paths}

    result = read_records_metadata([ImageRecord(image_path=path) for path in paths])

    assert parallel._pool is not None
    assert result == {path: metadata for path, metadata in expected.items() if metadata}
    assert {metadata.beam_type for metadata in result.values()} == {"ELECTRON", "ION"}


def test_order_is_kept_across_chunks(pool):
    assert parallel.map_in_processes(abs, range(-9, 0)) == list(range(9, 0, -1))


def test_progress_is_reported_as_results_come_back(pool):
    seen: list[tuple[int, int]] = []

    parallel.map_in_processes(abs, range(-5, 0), on_progress=lambda *step: seen.append(step))

    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_a_batch_read_here_reports_every_item(monkeypatch):
    seen: list[tuple[int, int]] = []

    parallel.map_in_processes(abs, [-1, -2], on_progress=lambda *step: seen.append(step))

    assert seen == [(1, 2), (2, 2)]


def test_a_small_batch_starts_no_pool(tmp_path: Path):
    parallel.shutdown_pool()

    read_records_metadata([ImageRecord(image_path=path) for path in _images(tmp_path)])

    assert parallel._pool is None


def test_a_worker_imports_nothing_the_window_did(pool):
    """Spawned rather than forked, so a worker carries none of the window's Qt state."""
    import am_segmentation_manager.gui.export_controller  # noqa: F401

    assert "am_segmentation_manager.gui" in sys.modules
    assert parallel.map_in_processes(_window_modules, [None, None]) == [[], []]


def test_a_worker_warning_reaches_this_process(tmp_path: Path, pool, caplog):
    """An unreadable header is warned about in the worker and must reach ``gui.log``."""
    junk = _images(tmp_path)[-1]

    with caplog.at_level(logging.WARNING):
        read_records_metadata([ImageRecord(image_path=junk)] * 3)

    warned = [
        record for record in caplog.records
        if record.name == "am_segmentation_manager.image_metadata"
        and "Could not read header" in record.getMessage()
    ]
    assert len(warned) == 3


def test_a_broken_pool_finishes_the_batch_here(monkeypatch, pool, caplog):
    class Broken:
        def map(self, *_args):
            raise BrokenProcessPool("a worker was killed")

    monkeypatch.setattr(parallel, "_session_pool", Broken)

    with caplog.at_level(logging.WARNING):
        assert parallel.map_in_processes(abs, [-1, -2, -3]) == [1, 2, 3]
    assert "finishing 3 item(s) in this process" in caplog.text


def test_a_pool_that_broke_is_not_started_again(monkeypatch, pool, caplog):
    """What broke one pool -- a script without a ``__main__`` guard -- breaks the
    next, so starting another for every batch only repeats the failure."""
    started: list[int] = []

    class Broken:
        def map(self, *_args):
            raise BrokenProcessPool("a worker was killed")

    def start() -> Broken:
        started.append(1)
        return Broken()

    monkeypatch.setattr(parallel, "_session_pool", start)

    with caplog.at_level(logging.WARNING):
        assert parallel.map_in_processes(abs, [-1, -2]) == [1, 2]
        assert parallel.map_in_processes(abs, [-3, -4]) == [3, 4]
    assert started == [1]
    assert caplog.text.count("worker processes stopped") == 1


def test_a_pool_breaking_part_way_finishes_only_what_is_left(monkeypatch, pool, caplog):
    done_here: list[int] = []

    class BreaksAfterOneChunk:
        def map(self, function, functions, chunks):
            yield function(functions[0], chunks[0])
            raise BrokenProcessPool("a worker was killed")

    def record(item: int) -> int:
        done_here.append(item)
        return abs(item)

    monkeypatch.setattr(parallel, "_session_pool", BreaksAfterOneChunk)

    with caplog.at_level(logging.WARNING):
        assert parallel.map_in_processes(record, [-1, -2, -3, -4, -5]) == [1, 2, 3, 4, 5]
    assert done_here == [-1, -2, -3, -4, -5]
    assert "finishing 3 item(s) in this process" in caplog.text
