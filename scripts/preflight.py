#!/usr/bin/env python3
"""Run what a push is checked for and this machine can run, before it is pushed.

    uv run python scripts/preflight.py          # under a minute
    uv run python scripts/preflight.py --full   # and the whole suite

``.githooks/pre-push`` runs the first on every push, once enabled with
``git config core.hooksPath .githooks``. **The checks are listed here and
nowhere else.** Every check runs whatever the one before it found, and the exit
status is non-zero where any failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent

#: The environment the suite runs in with no extra installed, beside ``.venv``
#: so that building it leaves the main environment's extras alone.
CORE_ENVIRONMENT = ".venv-core"


@dataclass(frozen=True)
class Check:
    """One command a push must pass.

    Attributes:
        name: What it is called in the report.
        command: What is run, from the repository root.
        environment: Variables set for it over this process's own.
    """

    name: str
    command: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)


def checks(full: bool = False) -> tuple[Check, ...]:
    """Return the checks in the order they run, the whole suite last where *full*."""
    listed = [
        Check("ruff", ("uv", "run", "--locked", "ruff", "check")),
        Check("mypy", ("uv", "run", "--locked", "mypy")),
        Check(
            "the suite with no extra installed",
            ("uv", "run", "--locked", "pytest", "-q"),
            {"UV_PROJECT_ENVIRONMENT": CORE_ENVIRONMENT},
        ),
    ]
    if full:
        listed.append(Check("the whole suite", ("uv", "run", "--locked", "pytest", "-q")))
    return tuple(listed)


def run_check(check: Check) -> bool:
    """Run *check* from the repository root, its output shown, and return whether it passed."""
    # uv would warn that the environment it was started from is not the one named.
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    environment.update(check.environment)
    return subprocess.run(check.command, cwd=REPOSITORY, env=environment).returncode == 0  # noqa: S603


def failed(listed: Sequence[Check], run: Callable[[Check], bool]) -> list[str]:
    """Run every one of *listed* with *run* and return the names of those that failed."""
    names = []
    for check in listed:
        print(f"-- {check.name}: {' '.join(check.command)}", flush=True)
        if not run(check):
            names.append(check.name)
    return names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--full", action="store_true",
        help="also run the whole suite in the main environment, which takes minutes",
    )
    arguments = parser.parse_args(argv)
    names = failed(checks(arguments.full), run_check)
    if names:
        print(f"Preflight failed: {', '.join(names)}.")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
