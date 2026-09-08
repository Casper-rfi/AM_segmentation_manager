"""The suite's refusal to run against a Qt binding the window is not written for.

This exists because of a real run: the suite started in conda's `base`, where
qtpy resolves to PyQt5, and reported sixteen failures across seven GUI files.
Every one of them was a difference between the bindings -- the top-line message
was `'sip.voidptr' object has no attribute 'tobytes'` -- and none of them said
so. The result reads as a set of unrelated regressions in the window, which is
the most expensive thing a green package can report.

The rule these pin: name the binding before running, keep an absent qtpy a
skip rather than an error -- the curation environment without the window being a
supported way to work -- and **say what to run, not what to activate**. The
refusal is usually read in a shell where ``conda activate`` is not a function,
so an interpreter that can be run is the only remedy that helps there.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (
    REQUIRED_QT_BINDING,
    SUPPORTED_ENVIRONMENTS,
    installed_qt_binding,
    pytest_configure,
    supported_interpreters,
)


class _Config:
    """Stands in for pytest's own config, which the guard never reaches."""

    def addinivalue_line(self, name: str, value: str) -> None:  # pragma: no cover
        raise AssertionError("the binding guard should not configure markers")


@pytest.fixture
def config(monkeypatch):
    """A configure call with no locale requested, so only the guard runs."""
    monkeypatch.delenv("AM_SEG_TEST_LOCALE", raising=False)
    return _Config()


def test_the_binding_this_suite_is_running_under_is_the_supported_one():
    """Not a tautology: it fails the moment the environment is wrong, which is
    the whole point, and it is the only test here that asks the real machine."""
    assert installed_qt_binding() in (None, REQUIRED_QT_BINDING)


def test_the_supported_binding_starts_the_run(config, monkeypatch):
    monkeypatch.setattr(
        "conftest.installed_qt_binding", lambda: REQUIRED_QT_BINDING
    )

    pytest_configure(config)


def test_no_qt_at_all_starts_the_run(config, monkeypatch):
    """The environment without the window is supported, and the GUI tests skip
    themselves there. Turning that into an error would refuse a real setup."""
    monkeypatch.setattr("conftest.installed_qt_binding", lambda: None)

    pytest_configure(config)


def test_an_unsupported_binding_stops_the_run_before_any_test(config, monkeypatch):
    monkeypatch.setattr("conftest.installed_qt_binding", lambda: "PyQt5")

    with pytest.raises(pytest.UsageError) as excinfo:
        pytest_configure(config)

    assert "PyQt5" in str(excinfo.value)


def test_the_refusal_says_what_to_do_about_it(config, monkeypatch):
    """A guard that only says no sends the reader looking for the cause it
    already knows. It names the binding found, the one wanted, and the two
    environments that have it."""
    monkeypatch.setattr("conftest.installed_qt_binding", lambda: "PyQt6")

    with pytest.raises(pytest.UsageError) as excinfo:
        pytest_configure(config)

    message = str(excinfo.value)
    assert "PyQt6" in message
    assert REQUIRED_QT_BINDING in message
    assert "am_segmentation_manager_ml" in message


def test_the_refusal_names_a_command_that_can_be_run(config, monkeypatch):
    """Naming the environment is not naming a way into it.

    `conda activate` is a shell function an interactive profile loads, so it is
    absent from exactly the shells this refusal is read in. The message
    therefore carries the interpreter itself.
    """
    monkeypatch.setattr("conftest.installed_qt_binding", lambda: "PyQt5")
    monkeypatch.setattr(
        "conftest.supported_interpreters",
        lambda: [Path("/envs/am_segmentation_manager/bin/python")],
    )

    with pytest.raises(pytest.UsageError) as excinfo:
        pytest_configure(config)

    assert "/envs/am_segmentation_manager/bin/python -m pytest" in str(excinfo.value)


def test_the_refusal_still_says_what_to_do_where_no_interpreter_is_found(
    config, monkeypatch
):
    """A machine laid out some other way loses the command, not the advice."""
    monkeypatch.setattr("conftest.installed_qt_binding", lambda: "PyQt5")
    monkeypatch.setattr("conftest.supported_interpreters", lambda: [])

    with pytest.raises(pytest.UsageError) as excinfo:
        pytest_configure(config)

    message = str(excinfo.value)
    assert "Run instead" not in message
    assert all(name in message for name in SUPPORTED_ENVIRONMENTS)


def test_an_interpreter_is_offered_only_where_one_exists():
    """The search answers about this machine, so it may find none -- but what it
    finds is a file, never a path assembled and hoped for."""
    assert all(one.exists() for one in supported_interpreters())
