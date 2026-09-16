"""The CI workflow held to what ``pyproject.toml`` says the package supports.

Supported Python versions are written in three places -- the classifiers, the
lock's required environments and the workflow's matrix -- and nothing else
notices when one moves without the others.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parent.parent
WORKFLOW = REPOSITORY / ".github" / "workflows" / "tests.yml"

#: The job that resolves the newest dependencies instead of the locked ones.
UNLOCKED_JOB = "newest"


def _pyproject() -> dict:
    return tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _classified_versions() -> set[str]:
    return {
        classifier.rsplit(" :: ", 1)[-1]
        for classifier in _pyproject()["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", classifier)
    }


def test_the_workflow_tests_exactly_the_python_versions_the_package_claims():
    tested = {
        str(entry["python"])
        for job in _jobs().values()
        for entry in job.get("strategy", {}).get("matrix", {}).get("include", [])
        if "python" in entry
    }

    assert tested == _classified_versions()


def test_the_lock_requires_builds_for_exactly_the_claimed_versions():
    """The bound is the version after the newest claimed: a lower one leaves a
    claimed version unchecked, a higher one refuses a lock over a Python nobody
    supports yet."""
    newest = max(_classified_versions(), key=lambda version: int(version.split(".")[1]))
    bound = f"3.{int(newest.split('.')[1]) + 1}"
    environments = _pyproject()["tool"]["uv"]["required-environments"]

    assert environments
    assert all(f"python_full_version < '{bound}'" in marker for marker in environments)


def test_every_job_but_one_installs_what_the_lock_holds():
    """Without --locked, uv re-resolves whenever pyproject.toml has moved ahead
    of uv.lock, and the run tests versions nobody committed."""
    unlocked = [
        f"{name}: {step['run']}"
        for name, job in _jobs().items()
        if name != UNLOCKED_JOB
        for step in job["steps"]
        if "run" in step
        and re.match(r"uv (sync|run)\b", step["run"])
        and "--locked" not in step["run"]
    ]

    assert not unlocked, unlocked
