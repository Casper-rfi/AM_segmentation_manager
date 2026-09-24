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
WORKFLOWS = REPOSITORY / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "tests.yml"

#: The job that resolves the newest dependencies instead of the locked ones.
UNLOCKED_JOB = "newest"


def _pyproject() -> dict:
    return tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _every_job() -> dict:
    """Return the jobs of every workflow, each named for its file."""
    return {
        f"{path.name}:{name}": job
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for name, job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].items()
    }


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


def test_the_two_inference_extras_install_the_same_package_set():
    """`ml-cuda13` exists to change which torch is fetched, nothing else. It is a
    second copy of one list -- uv gives no way to extend an extra whose sources
    conflict -- so the copies are held together here rather than by care."""
    extras = _pyproject()["project"]["optional-dependencies"]

    assert set(extras["ml-cuda13"]) == set(extras["ml"])


def test_the_conflicting_extras_are_declared_so_neither_can_join_the_other():
    """Installed together they would name one torch from two indexes."""
    conflicts = _pyproject()["tool"]["uv"]["conflicts"]

    assert [{"extra": "ml"}, {"extra": "ml-cuda13"}] in conflicts


def test_every_job_but_one_installs_what_the_lock_holds():
    """Without --locked, uv re-resolves whenever pyproject.toml has moved ahead
    of uv.lock, and the run tests versions nobody committed."""
    unlocked = [
        f"{name}: {step['run']}"
        for name, job in _every_job().items()
        if name != f"{WORKFLOW.name}:{UNLOCKED_JOB}"
        for step in job["steps"]
        if "run" in step
        and re.match(r"uv (sync|run)\b", step["run"])
        and "--locked" not in step["run"]
    ]

    assert not unlocked, unlocked


def test_an_action_from_outside_github_is_pinned_to_a_commit():
    """A tag can be moved to other code, and some publishers -- astral-sh among
    them -- publish no major-version tag to float on at all. GitHub's own
    ``actions/`` are the exception, and a local action has no version."""
    loose = [
        step["uses"]
        for job in _every_job().values()
        for step in job["steps"]
        if "uses" in step
        and not step["uses"].startswith(("actions/", "./"))
        and not re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", step["uses"])
    ]

    assert not loose, loose


#: The runner each platform in a ``[tool.uv.sources]`` marker is tested on.
RUNNERS = {"win32": "windows-latest", "linux": "ubuntu-latest"}


def _installed() -> set[tuple[str, str]]:
    """Return each ``(runner, extra)`` some job in the workflow syncs."""
    installed = set()
    for job in _jobs().values():
        entries = job.get("strategy", {}).get("matrix", {}).get("include", [{}])
        for entry in entries:
            def expand(text: str, entry: dict = entry) -> str:
                return re.sub(
                    r"\$\{\{\s*matrix\.(\w+)\s*\}\}", lambda m: str(entry[m.group(1)]), text
                )

            runner = expand(job["runs-on"])
            for step in job["steps"]:
                if "run" in step and step["run"].startswith("uv sync"):
                    for extra in re.findall(r"--extra (\S+)", expand(step["run"])):
                        installed.add((runner, extra))
    return installed


def test_each_torch_build_is_installed_on_every_platform_it_is_chosen_for():
    """Each inference extra fetches its own torch on Windows and Linux; a build
    no job installs is one only a user's machine ever tries."""
    wanted = {
        (RUNNERS[platform], source["extra"])
        for source in _pyproject()["tool"]["uv"]["sources"]["torch"]
        for platform in re.findall(r"sys_platform == '(\w+)'", source["marker"])
    }

    assert wanted
    assert wanted <= _installed(), sorted(wanted - _installed())
