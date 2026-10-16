"""The name, version and links are stated once, and a file format changes only as a breaking change.

Read from the repository's own files, so these hold whether or not the package
is installed; `decisions.md` § Environment says why each rule is there.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parent.parent


def _project() -> dict:
    with (REPOSITORY / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


#: A requirement with extras, as a hint quotes it: 'name[gui]'.
INSTALL_HINT = re.compile(r"'([A-Za-z0-9_.-]+)\[[a-z0-9,-]+\]'")


@pytest.mark.parametrize(
    "module",
    ["gui/app.py", "io/image.py", "ml/environment.py", "plots.py"],
)
def test_an_install_hint_names_the_package_pyproject_does(module):
    """The distribution is not called what it is imported as, so a hint
    written from the import name would send a user to install nothing."""
    source = (REPOSITORY / "src" / "am_segmentation_manager" / module).read_text(encoding="utf-8")

    assert set(INSTALL_HINT.findall(source)) == {_project()["name"]}


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


def test_the_citation_describes_the_release_this_is():
    """A citation taken from a release keeps the version it names, so a release
    must not go out still citing the last."""
    project = _project()
    citation = yaml.safe_load((REPOSITORY / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["license"] == project["license"]
    assert citation["repository-code"] == project["urls"]["Homepage"]
    if ".dev" not in project["version"]:
        assert citation["version"] == project["version"], (
            f"CITATION.cff cites {citation['version']}, not {project['version']}"
        )


def test_a_file_format_change_is_a_breaking_change():
    """Collections outlive every version of this package, so a change to what
    it writes beside them is never a detail."""
    from am_segmentation_manager import history, state
    from am_segmentation_manager.operations import export_record, training_manifest
    from am_segmentation_manager.training import preparation

    formats = {
        "AM_state.yaml": state.FILE_VERSION,
        "history": history.FILE_VERSION,
        "training manifest": training_manifest.FILE_VERSION,
        "AM_export.yaml": export_record.FILE_VERSION,
        "am_training_run.yaml": preparation.FILE_VERSION,
    }

    assert formats == {
        "AM_state.yaml": 1,
        "history": 1,
        "training manifest": 1,
        "AM_export.yaml": 1,
        "am_training_run.yaml": 1,
    }, (
        "A file format changed. Name it under Unreleased in CHANGELOG.md as a "
        "breaking change, give files in the old format a way forward, then "
        "update this table."
    )


def test_every_project_link_into_the_repository_names_a_file_it_has():
    """A link on PyPI is fixed with the release it was published in, so a
    renamed guide or changelog must fail here rather than there."""
    urls = _project()["urls"]
    homepage = urls["Homepage"].rstrip("/")
    into_files = {
        label: url.removeprefix(f"{homepage}/blob/main/")
        for label, url in urls.items()
        if url.startswith(f"{homepage}/blob/main/")
    }

    assert into_files, "no link names a file in the repository"
    for label, path in into_files.items():
        assert (REPOSITORY / path).is_file(), f"{label} names {path}, which is not here"


#: A `uv tool install` line in the README: its requirement, and the source after `@`.
TOOL_INSTALL = re.compile(r'uv tool install[^"\n]*"([A-Za-z0-9_.-]+)\[[a-z0-9,-]+\] @ git\+([^"]+)"')


def test_the_readme_installs_the_package_from_the_repository_pyproject_names():
    """Copied into a terminal as written, so a renamed package or a moved
    repository must fail here rather than there."""
    project = _project()
    commands = TOOL_INSTALL.findall((REPOSITORY / "README.md").read_text(encoding="utf-8"))

    assert commands, "the README gives no uv tool install command"
    for name, source in commands:
        assert _normalised(name) == _normalised(project["name"])
        assert source == project["urls"]["Homepage"]

