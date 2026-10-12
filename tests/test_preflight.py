"""Tests for ``scripts/preflight.py``: what it runs, and what it makes of a failure."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import preflight
import pytest

REPOSITORY = Path(__file__).resolve().parent.parent


def test_the_suite_without_extras_runs_in_an_environment_of_its_own():
    """Synced into the main one, it would take the extras out of it."""
    by_name = {check.name: check for check in preflight.checks()}
    core = by_name["the suite with no extra installed"]

    assert core.environment == {"UV_PROJECT_ENVIRONMENT": preflight.CORE_ENVIRONMENT}
    assert all(not check.environment for check in by_name.values() if check is not core)
    assert f"{preflight.CORE_ENVIRONMENT}/" in (REPOSITORY / ".gitignore").read_text().splitlines()


def test_the_whole_suite_runs_only_where_asked_for():
    quick = [check.name for check in preflight.checks()]
    full = [check.name for check in preflight.checks(full=True)]

    assert full == [*quick, "the whole suite"]


def test_a_check_that_fails_does_not_hide_the_ones_after_it(capsys):
    """CI's lint job stops at ruff, and mypy's errors gathered behind one unused import."""
    ran = []

    def run(check):
        ran.append(check.name)
        return check.name != "ruff"

    assert preflight.failed(preflight.checks(), run) == ["ruff"]
    assert ran == [check.name for check in preflight.checks()]


@pytest.mark.parametrize(("failing", "status"), [("mypy", 1), ("nothing", 0)])
def test_any_failure_fails_the_preflight(monkeypatch, capsys, failing, status):
    monkeypatch.setattr(preflight, "run_check", lambda check: check.name != failing)

    assert preflight.main([]) == status
    assert (f"Preflight failed: {failing}." in capsys.readouterr().out) == bool(status)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no executable bit to check")
def test_the_hook_git_runs_before_a_push_is_the_preflight():
    hook = REPOSITORY / ".githooks" / "pre-push"

    assert os.access(hook, os.X_OK)
    assert "scripts/preflight.py" in hook.read_text()
