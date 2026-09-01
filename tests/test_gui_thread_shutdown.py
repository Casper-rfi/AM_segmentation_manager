"""What happens to a worker thread that will not stop when the window closes.

A read is one call into an image library and cannot be interrupted part-way, so
a file on a cloud-synced volume can outlast any deadline worth waiting for. The
property pinned here is what is done about that: the thread is left to finish
and its Python side is held, rather than collected while it is still calling
into it.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import weakref

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.gui import loader as loader_module
    from am_segmentation_manager.gui import worker as worker_module
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.loader import BackgroundLoader
    from am_segmentation_manager.gui.worker import BackgroundOperation


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def parked(monkeypatch) -> list:
    """Watch what gets held, without holding it past the test."""
    held: list = []
    monkeypatch.setattr(loader_module, "_ABANDONED", held)
    return held


def _refuse_to_stop(monkeypatch, owner) -> None:
    """Make the thread look like one still inside a read at the deadline."""
    monkeypatch.setattr(owner._thread, "wait", lambda *_args: False)


def test_a_loader_that_stops_is_not_held(qt_app, parked):
    loader = BackgroundLoader(FrameCache())

    loader.stop()

    assert parked == []


def test_a_loader_that_will_not_stop_is_held(qt_app, parked, monkeypatch):
    loader = BackgroundLoader(FrameCache())
    _refuse_to_stop(monkeypatch, loader)

    loader.stop()

    assert parked == [loader]
    loader._thread.wait(5000)


def test_what_is_held_survives_the_window_letting_go_of_it(
    qt_app, parked, monkeypatch
):
    """The crash this prevents: the worker collected under a running thread."""
    loader = BackgroundLoader(FrameCache())
    _refuse_to_stop(monkeypatch, loader)
    watch = weakref.ref(loader.worker)

    loader.stop()
    thread = loader._thread
    del loader
    gc.collect()

    assert watch() is not None
    thread.wait(5000)


def test_an_operation_that_stops_is_not_held(qt_app, parked):
    operation = BackgroundOperation()

    operation.stop()

    assert parked == []


def test_an_operation_that_will_not_stop_is_held(qt_app, parked, monkeypatch):
    operation = BackgroundOperation()
    _refuse_to_stop(monkeypatch, operation)

    operation.stop()

    assert parked == [operation]
    operation._thread.wait(5000)


def test_both_shutdowns_park_in_the_same_place():
    """One list, so the reason for it is written down once."""
    assert worker_module.keep_alive is loader_module.keep_alive
