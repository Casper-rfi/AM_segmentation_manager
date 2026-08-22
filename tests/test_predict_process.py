"""Predicting in a process of its own.

The child holds torch and the window holds none of it, which is what the window
crashing repeatedly -- and the command line never once -- is the evidence for.
What is pinned here is the parent's side of that: it pumps the child's
messages, re-emits its log, asks it to stop once, and reports a child that died
without saying anything rather than waiting for it forever.

The child's own logic is exercised directly in ``test_gui_predict.py``: it is a
plain function, and calling it is what lets it be checked without torch.
"""
from __future__ import annotations

import logging
import queue as queue_module

import pytest

from am_segmentation_manager.operations import predict_process
from am_segmentation_manager.operations.predict_process import (
    ProcessOutcome,
    probe_devices,
)
from am_segmentation_manager.progress import Cancellation


class FakeQueue:
    """A queue handing out prepared messages, then staying empty."""

    def __init__(self, messages) -> None:
        self._messages = list(messages)

    def get(self, timeout=None):
        if not self._messages:
            raise queue_module.Empty
        return self._messages.pop(0)


class FakeProcess:
    """A child that is alive until told otherwise, and records being signalled."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.pid = -1
        self.exitcode = None
        self.interrupts = 0

    def is_alive(self) -> bool:
        return self.alive


@pytest.fixture(autouse=True)
def no_real_signals(monkeypatch):
    """Count interrupts rather than sending them to a real process."""
    def interrupt(process) -> None:
        process.interrupts += 1

    monkeypatch.setattr(predict_process, "_interrupt", interrupt)


def test_progress_reaches_the_caller_and_the_outcome_comes_back():
    outcome = ProcessOutcome(written=2, total=2, modality="SEM")
    seen: list[tuple[int, int]] = []

    result = predict_process._await(
        FakeProcess(),
        FakeQueue([("progress", 1, 2), ("progress", 2, 2), ("done", outcome)]),
        lambda done, total: seen.append((done, total)),
        None,
    )

    assert seen == [(1, 2), (2, 2)]
    assert result is outcome


def test_the_childs_log_is_re_emitted_here(caplog):
    """One log file has to hold the whole run, the child's part included --
    which device the model loaded onto is the line that placed the last crash,
    and it is written in the child."""
    record = logging.LogRecord(
        "am_segmentation_manager.ml.model", logging.INFO, __file__, 0,
        "Loaded weights onto mps", (), None,
    )

    with caplog.at_level(logging.INFO):
        predict_process._await(
            FakeProcess(),
            FakeQueue([("log", record), ("done", ProcessOutcome())]),
            None,
            None,
        )

    assert "Loaded weights onto mps" in caplog.text


def test_stopping_asks_the_child_once_and_waits_for_it():
    """Asked between whole images, so what was written is kept. Asking twice
    would interrupt the write the first ask is waiting to finish."""
    process = FakeProcess()
    cancellation = Cancellation()
    cancellation.request()

    predict_process._await(
        process,
        FakeQueue([("done", ProcessOutcome(written=1, total=3, stopped=True))]),
        None,
        cancellation,
    )

    assert process.interrupts == 1


def test_a_child_that_dies_without_reporting_is_not_waited_on_forever():
    """The failure this module exists to survive, seen from the other side."""
    process = FakeProcess(alive=False)
    process.exitcode = -11

    outcome = predict_process._await(process, FakeQueue([]), None, None)

    assert outcome.written == 0
    assert "without reporting" in outcome.error
    assert "-11" in outcome.error


def test_a_cancellation_before_anything_starts_still_only_asks_once():
    process = FakeProcess()
    cancellation = Cancellation()
    cancellation.request()
    messages = [("progress", 1, 3), ("done", ProcessOutcome(stopped=True))]

    predict_process._await(process, FakeQueue(messages), None, cancellation)

    assert process.interrupts == 1


# --- really starting one ----------------------------------------------------

def test_the_device_probe_starts_a_real_process():
    """This process never imports torch, so it cannot answer this itself. A
    machine without the inference stack says nothing rather than raising."""
    devices = probe_devices()

    assert isinstance(devices, tuple)
    assert all(isinstance(name, str) for name in devices)
    assert "torch" not in __import__("sys").modules or devices
