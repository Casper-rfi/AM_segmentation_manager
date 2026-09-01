"""What the window does with an exception it did not handle.

The property worth pinning is the one that made a crash unreadable: an
exception escaping a Qt slot must be formatted in Python and logged, never left
to the interpreter's own printer. The dialog is stubbed throughout -- what it
says is not the point, that exactly one is offered per failure is.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtWidgets import QApplication, QListWidget

    from am_segmentation_manager.gui import app as gui_app


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def offered(qt_app, monkeypatch):
    """Install the hooks with the dialog replaced, and collect what it was given.

    The hooks are global, so they are put back afterwards; ``monkeypatch``
    covers the two it replaces and the ``_show`` it would have called.

    Events are drained either side of the test. An offer is deferred by a
    timer, so one left pending by a test that did not drain would otherwise
    fire during the next one and be counted there.
    """
    shown: list[str] = []

    def record(summary: str, detail: str) -> None:
        shown.append(summary)
        gui_app._reporting = False

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(gui_app, "_show", record)
    monkeypatch.setattr(gui_app, "_reporting", False)
    gui_app.install_error_reporting()

    qt_app.processEvents()
    shown.clear()
    yield shown
    qt_app.processEvents()


def _drain(qt_app) -> None:
    """Let the queued dialog request run, the offer being deferred a turn."""
    qt_app.processEvents()


def test_an_exception_escaping_a_slot_is_logged_rather_than_printed(
    qt_app, offered, caplog
):
    widget = QListWidget()
    widget.addItems(["one", "two"])
    widget.currentRowChanged.connect(lambda _row: 1 / 0)

    with caplog.at_level(logging.ERROR, logger="am_segmentation_manager.gui.app"):
        widget.setCurrentRow(1)

    assert "ZeroDivisionError" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


def test_the_failure_is_offered_to_the_reviewer(qt_app, offered):
    widget = QListWidget()
    widget.addItems(["one", "two"])
    widget.currentRowChanged.connect(lambda _row: 1 / 0)

    widget.setCurrentRow(1)
    _drain(qt_app)

    assert offered == ["ZeroDivisionError: division by zero"]


def test_only_one_failure_is_on_screen_at_a_time(qt_app, offered, monkeypatch):
    """A slot raising on every repaint must not open a dialog per repaint."""
    monkeypatch.setattr(gui_app, "_show", lambda summary, detail: offered.append(summary))

    gui_app._offer("first", "detail")
    gui_app._offer("second", "detail")
    _drain(qt_app)

    assert offered == ["first"]


def test_a_thread_python_started_reports_the_same_way(qt_app, offered, caplog):
    def raises() -> None:
        raise RuntimeError("off the main thread")

    with caplog.at_level(logging.ERROR, logger="am_segmentation_manager.gui.app"):
        worker = threading.Thread(target=raises)
        worker.start()
        worker.join()

    assert "off the main thread" in caplog.text
    # Nothing is offered: a dialog cannot be built from a thread that does not
    # own the widgets, and the log is where that failure is read.
    _drain(qt_app)
    assert offered == []


def test_a_thread_leaving_normally_reports_nothing(qt_app, offered, caplog):
    """SystemExit is how a thread is asked to stop, not a failure."""

    def leaves() -> None:
        raise SystemExit

    with caplog.at_level(logging.ERROR, logger="am_segmentation_manager.gui.app"):
        worker = threading.Thread(target=leaves)
        worker.start()
        worker.join()

    assert caplog.text == ""
    _drain(qt_app)
    assert offered == []
