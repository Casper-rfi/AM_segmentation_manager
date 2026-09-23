"""The version is stated once, and a file format changes only as a breaking change.

Read from the repository's own files, so these hold whether or not the package
is installed; `decisions.md` § Environment says why each rule is there.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent


def _project() -> dict:
    with (REPOSITORY / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_version_is_a_release_or_the_development_of_one():
    """X.Y.Z for a release, X.Y.Z.devN on the way to one: nothing a reader of
    a record has to guess the meaning of."""
    version = _project()["version"]

    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", version), version


def test_the_lock_records_the_version_pyproject_states():
    """A version edited by hand leaves `uv.lock` behind, and every
    `uv sync --locked` in CI then refuses to run."""
    project = _project()
    with (REPOSITORY / "uv.lock").open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    ours = [p for p in packages if p["name"] == _normalised(project["name"])]

    assert [p["version"] for p in ours] == [project["version"]], (
        "set the version with `uv version <X.Y.Z> --no-sync`, which relocks as well"
    )


def test_the_changelog_describes_the_release_this_is():
    """Work in progress has somewhere to go, and a release says what it holds."""
    version = _project()["version"]
    changelog = (REPOSITORY / "CHANGELOG.md").read_text(encoding="utf-8")

    assert re.search(r"^## \[Unreleased\]$", changelog, re.MULTILINE)
    if ".dev" not in version:
        heading = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
        assert re.search(heading, changelog, re.MULTILINE), (
            f"CHANGELOG.md has no '## [{version}] - <date>' heading"
        )


def test_a_file_format_change_is_a_breaking_change():
    """Collections outlive every version of this package, so a change to what
    it writes beside them is never a detail."""
    from am_segmentation_manager import history, state
    from am_segmentation_manager.operations import export_record, training_manifest

    formats = {
        "AM_state.yaml": state.FILE_VERSION,
        "history": history.FILE_VERSION,
        "training manifest": training_manifest.FILE_VERSION,
        "AM_export.yaml": export_record.FILE_VERSION,
    }

    assert formats == {
        "AM_state.yaml": 1,
        "history": 1,
        "training manifest": 1,
        "AM_export.yaml": 1,
    }, (
        "A file format changed. Name it under Unreleased in CHANGELOG.md as a "
        "breaking change, give files in the old format a way forward, then "
        "update this table."
    )
