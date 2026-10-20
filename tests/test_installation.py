"""How this copy says it is updated, from what its installer recorded."""
from __future__ import annotations

import subprocess
import tomllib
from datetime import date, datetime
from pathlib import Path

from am_segmentation_manager import installation
from am_segmentation_manager.identity import REPOSITORY_URL
from am_segmentation_manager.installation import (
    checkout_directory,
    commit_date,
    install_date,
    install_kind,
    install_record,
    update_steps,
    version_date,
)

CHECKOUT = {"url": "file:///home/someone/AM%20manager", "dir_info": {"editable": True}}
REPOSITORY = {"url": REPOSITORY_URL, "vcs_info": {"vcs": "git", "commit_id": "abc"}}


def test_a_checkout_is_told_to_pull_and_sync_where_it_is():
    steps = update_steps(CHECKOUT)

    assert install_kind(CHECKOUT) == "checkout"
    assert checkout_directory(CHECKOUT) == Path("/home/someone/AM manager")
    assert "git pull" in steps[1] and "AM manager" in steps[1]
    assert "uv sync" in steps[2]


def test_a_tool_install_is_told_to_install_again_with_every_extra():
    """The extras name the torch build, and the ones given replace the ones installed."""
    steps = update_steps(REPOSITORY)

    assert install_kind(REPOSITORY) == "repository"
    assert "--upgrade" in steps[1]
    assert "every extra" in steps[2]


def test_an_install_that_recorded_nothing_is_sent_to_the_readme():
    assert install_kind({}) == "other"
    assert "README" in update_steps({})[-1]


def test_every_install_is_told_where_newer_versions_are_listed():
    assert all(REPOSITORY_URL in update_steps(record)[0] for record in (CHECKOUT, REPOSITORY, {}))


def test_a_record_that_cannot_be_read_is_no_record(monkeypatch):
    class Unreadable:
        def read_text(self, _name):
            return "{not json"

    monkeypatch.setattr(installation, "distribution", lambda _name: Unreadable())

    assert install_record() == {}


def test_a_checkout_is_dated_by_its_latest_commit(monkeypatch):
    asked = []
    monkeypatch.setattr(installation, "commit_date", lambda d: asked.append(d) or date(2026, 10, 7))

    assert version_date(CHECKOUT) == "Latest change: 2026-10-07"
    assert asked == [Path("/home/someone/AM manager")]


def test_any_other_install_is_dated_by_when_it_was_installed(monkeypatch):
    monkeypatch.setattr(installation, "commit_date", lambda _directory: 1 / 0)
    monkeypatch.setattr(installation, "install_date", lambda: date(2026, 9, 12))

    assert version_date(REPOSITORY) == "Installed: 2026-09-12"
    assert version_date({}) == "Installed: 2026-09-12"


def test_a_checkout_git_cannot_date_falls_back_to_when_it_was_installed(monkeypatch):
    monkeypatch.setattr(installation, "commit_date", lambda _directory: None)
    monkeypatch.setattr(installation, "install_date", lambda: date(2026, 9, 12))

    assert version_date(CHECKOUT) == "Installed: 2026-09-12"


def test_a_copy_nothing_can_date_says_nothing(monkeypatch):
    monkeypatch.setattr(installation, "install_date", lambda: None)

    assert version_date({}) == ""


def test_the_day_of_a_commit_is_read_from_what_git_answers(monkeypatch, tmp_path):
    def answer(command, **_options):
        assert command[:3] == ["git", "-C", str(tmp_path)]
        return subprocess.CompletedProcess(command, 0, "2026-10-07T23:41:05+02:00\n", "")

    monkeypatch.setattr(installation.subprocess, "run", answer)

    assert commit_date(tmp_path) == date(2026, 10, 7)


def test_a_folder_git_cannot_read_has_no_commit_date(monkeypatch, tmp_path):
    def missing(*_arguments, **_options):
        raise FileNotFoundError("git")

    # A git hook exports its repository, which git would then read in place of this folder.
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable, raising=False)
    assert commit_date(tmp_path) is None, "a folder that is no repository"
    monkeypatch.setattr(installation.subprocess, "run", missing)
    assert commit_date(tmp_path) is None, "a machine without git"


def test_the_install_date_is_the_age_of_what_the_installer_wrote(monkeypatch, tmp_path):
    import os
    from importlib.metadata import PackagePath

    written = tmp_path / "pkg.dist-info" / "METADATA"
    written.parent.mkdir()
    written.write_text("Name: pkg\n", encoding="utf-8")
    stamp = datetime(2026, 9, 12, 12).timestamp()
    os.utime(written, (stamp, stamp))

    class Installed:
        files = [PackagePath("pkg.dist-info/RECORD"), PackagePath("pkg.dist-info/METADATA")]

        def locate_file(self, path):
            return tmp_path / path

    monkeypatch.setattr(installation, "distribution", lambda _name: Installed())
    assert install_date() == date(2026, 9, 12)

    Installed.files = None
    assert install_date() is None, "an install that lists no files"


def test_the_repository_is_the_one_the_package_declares():
    declared = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert declared["project"]["urls"]["Homepage"] == REPOSITORY_URL


def test_updating_reaches_no_network():
    """It says where to look and looks for nothing itself."""
    source = Path(installation.__file__).read_text(encoding="utf-8")

    assert "urlopen" not in source and "socket" not in source and "http.client" not in source
