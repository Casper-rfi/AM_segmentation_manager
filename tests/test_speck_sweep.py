"""Counting a series' specks off the interface thread.

The worker is driven directly, without a thread: what is asserted here is that
it reads each label once, reports as it goes rather than at the end, and stops
when the reviewer has moved on. Whether Qt delivers the call on another thread
is :mod:`.loader`'s own arrangement and is not re-tested per slot.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("numpy")

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.loader import FrameLoadWorker


def write_label(path, specks: int):
    """Write a label holding an annotation and *specks* single pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((64, 96), dtype=np.uint8)
    array[10:40, 10:80] = 2
    for n in range(specks):
        array[50, 2 + n * 3] = 1
    tifffile.imwrite(str(path), array)
    return path


@pytest.fixture
def worker():
    return FrameLoadWorker(FrameCache(lambda image, label=None: None, lambda plot: None))


def test_each_label_is_reported_as_it_is_counted(worker, tmp_path):
    """One signal per label, not one per batch: the marks fill in as the sweep
    goes, which is the whole reason it is not done in one blocking pass."""
    seen = []
    worker.counted.connect(lambda token, key, count: seen.append((key, count)))
    labels = [
        ("a", write_label(tmp_path / "a.tif", 0)),
        ("b", write_label(tmp_path / "b.tif", 3)),
        ("c", write_label(tmp_path / "c.tif", 1)),
    ]

    worker.count_specks(labels, worker._token, 4)

    assert seen == [("a", 0), ("b", 3), ("c", 1)]


def test_a_superseded_sweep_stops_where_it_is(worker, tmp_path):
    """Moving to another lamella abandons it, which costs the label it was on
    and nothing else -- the sweep asks each time only for what is uncounted."""
    seen = []

    def on_counted(token, key, count):
        seen.append(key)
        worker.set_token(worker._token + 1)      # the reviewer moves on

    worker.counted.connect(on_counted)
    labels = [(name, write_label(tmp_path / f"{name}.tif", 1)) for name in "abcd"]

    worker.count_specks(labels, worker._token, 4)

    assert seen == ["a"], "the sweep should not have gone past the first label"


def test_a_label_that_will_not_open_counts_none(worker, tmp_path):
    """It is already drawn as unreadable on the label line, and a sweep that
    stopped on one would leave every frame after it unmarked."""
    seen = []
    worker.counted.connect(lambda token, key, count: seen.append((key, count)))
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff")
    labels = [("broken", broken), ("good", write_label(tmp_path / "g.tif", 2))]

    worker.count_specks(labels, worker._token, 4)

    assert seen == [("broken", 0), ("good", 2)]
