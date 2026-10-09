"""How this copy says it is updated, from what its installer recorded."""
from __future__ import annotations

import tomllib
from pathlib import Path

from am_segmentation_manager import installation
from am_segmentation_manager.identity import REPOSITORY_URL
from am_segmentation_manager.installation import (
    checkout_directory,
    install_kind,
    install_record,
    update_steps,
)

CHECKOUT = {"url": "file:///home/someone/AM%20manager", "dir_info": {"editable": True}}
REPOSITORY = {"url": REPOSITORY_URL, "vcs_info": {"vcs": "git", "commit_id": "abc"}}


def test_a_checkout_is_told_to_pull_and_sync_where_it_is():
    steps = update_steps(CHECKOUT)

    assert install_kind(CHECKOUT) == "checkout"
    assert checkout_directory(CHECKOUT) == Path("/home/someone/AM manager")
    assert "git pull" in steps[1] and "AM manager" in steps[1]
    assert "uv sync" in steps[2]


def test_a_tool_install_is_told_to_install_again_and_not_to_upgrade():
    """`uv tool upgrade` forgets the torch build, as the README says."""
    steps = update_steps(REPOSITORY)

    assert install_kind(REPOSITORY) == "repository"
    assert "--upgrade" in steps[1]
    assert "uv tool upgrade" in steps[2]


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


def test_the_repository_is_the_one_the_package_declares():
    declared = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert declared["project"]["urls"]["Homepage"] == REPOSITORY_URL


def test_updating_reaches_no_network():
    """It says where to look and looks for nothing itself."""
    source = Path(installation.__file__).read_text(encoding="utf-8")

    assert "urlopen" not in source and "socket" not in source and "http.client" not in source
