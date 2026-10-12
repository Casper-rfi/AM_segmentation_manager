"""Tests that the suite can be collected where no extra is installed.

CI's ``core`` job installs nothing but PyYAML, and one test module importing
numpy or Qt where it is collected stops that job before a test runs. This
collects the suite with every extra's modules refused, so it fails here too.
"""
from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent

#: What the child runs: refuse every module named on its command line, as an
#: environment without them would, then collect the suite and run nothing.
COLLECT = """
import importlib.abc, importlib.util, sys

import pytest

refused = frozenset(sys.argv[1:])


def is_refused(name):
    return name.partition(".")[0] in refused


class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if is_refused(name):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: (
    None if is_refused(name) else find_spec(name, package)
)
sys.meta_path.insert(0, Refuse())
sys.exit(pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider"]))
"""


def _normalised(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def modules_of_the_extras() -> list[str]:
    """Return the importable name of every package an extra installs by name, where installed."""
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = {
        _normalised(re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0])
        for requirements in project["optional-dependencies"].values()
        for requirement in requirements
    }
    return sorted(
        module
        for module, distributions in importlib.metadata.packages_distributions().items()
        if module.isidentifier() and wanted & {_normalised(one) for one in distributions}
    )


def test_every_test_module_is_collected_where_no_extra_is_installed():
    """A module needing an extra asks for it with ``pytest.importorskip``, or inside its tests."""
    refused = modules_of_the_extras()

    collected = subprocess.run(
        [sys.executable, "-c", COLLECT, *refused],
        cwd=REPOSITORY, capture_output=True, text=True, timeout=300,
    )

    errors = [line for line in collected.stdout.splitlines() if line.startswith("ERROR ")]
    assert not errors, "\n".join(errors)
    assert collected.returncode == 0, collected.stdout[-2000:] + collected.stderr[-2000:]
