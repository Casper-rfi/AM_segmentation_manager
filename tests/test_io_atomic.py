"""Tests for atomic writes and file fingerprints.

These were exercised only through label saving, which skips without the image
stack installed. They are a shared primitive now, so they are pinned directly.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from am_segmentation_manager.io.atomic import (
    file_mode_for,
    fingerprint_of,
    write_atomically,
)


def test_a_write_lands_whole(tmp_path: Path):
    destination = tmp_path / "out.txt"
    write_atomically(destination, lambda temporary: temporary.write_text("written"))
    assert destination.read_text() == "written"


def test_a_failed_write_leaves_the_previous_file(tmp_path: Path):
    destination = tmp_path / "out.txt"
    destination.write_text("original")

    def explode(temporary: Path) -> None:
        temporary.write_text("half of something")
        raise OSError("disk full")

    with pytest.raises(OSError):
        write_atomically(destination, explode)
    assert destination.read_text() == "original"


def test_a_failed_write_leaves_no_temporary_behind(tmp_path: Path):
    destination = tmp_path / "out.txt"
    with pytest.raises(OSError):
        write_atomically(destination, lambda _: (_ for _ in ()).throw(OSError("no")))
    assert list(tmp_path.iterdir()) == []


def test_a_write_creates_the_destination_directory(tmp_path: Path):
    destination = tmp_path / "missing" / "out.txt"
    write_atomically(destination, lambda temporary: temporary.write_text("written"))
    assert destination.read_text() == "written"


def test_a_replaced_file_keeps_its_own_permissions(tmp_path: Path):
    """Replacing a file is not a decision about who may read it."""
    destination = tmp_path / "out.txt"
    destination.write_text("original")
    os.chmod(destination, 0o640)

    write_atomically(destination, lambda temporary: temporary.write_text("replaced"))
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_a_new_file_is_not_owner_only(tmp_path: Path):
    """A file only its author can read is one nobody else can use."""
    destination = tmp_path / "out.txt"
    write_atomically(destination, lambda temporary: temporary.write_text("written"))
    assert stat.S_IMODE(destination.stat().st_mode) == file_mode_for(destination)
    assert stat.S_IMODE(destination.stat().st_mode) & stat.S_IRGRP


def test_a_fingerprint_of_a_missing_file_is_nothing(tmp_path: Path):
    assert fingerprint_of(tmp_path / "absent.txt") is None


def test_a_fingerprint_changes_when_the_contents_do(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("one")
    before = fingerprint_of(path)
    path.write_text("something longer")
    assert fingerprint_of(path) != before


# ---------------------------------------------------------------------------
# Deciding after the write
# ---------------------------------------------------------------------------

def test_a_refused_commit_leaves_the_destination_alone(tmp_path: Path):
    """A caller that can only judge the file once it exists still gets to say no."""
    destination = tmp_path / "out.txt"
    destination.write_text("original")

    replaced = write_atomically(
        destination,
        lambda temporary: temporary.write_text("candidate"),
        commit_if=lambda temporary: False,
    )

    assert replaced is False
    assert destination.read_text() == "original"


def test_a_refused_commit_leaves_no_temporary_behind(tmp_path: Path):
    destination = tmp_path / "out.txt"
    destination.write_text("original")

    write_atomically(
        destination,
        lambda temporary: temporary.write_text("candidate"),
        commit_if=lambda temporary: False,
    )

    assert [path.name for path in tmp_path.iterdir()] == ["out.txt"]


def test_an_accepted_commit_replaces_the_destination(tmp_path: Path):
    destination = tmp_path / "out.txt"
    destination.write_text("original")

    replaced = write_atomically(
        destination,
        lambda temporary: temporary.write_text("candidate"),
        commit_if=lambda temporary: True,
    )

    assert replaced is True
    assert destination.read_text() == "candidate"


def test_the_decision_sees_the_written_file(tmp_path: Path):
    """The point of the hook: judging the result, not predicting it."""
    destination = tmp_path / "out.txt"
    destination.write_text("original contents, long")
    seen: list[int] = []

    write_atomically(
        destination,
        lambda temporary: temporary.write_text("short"),
        commit_if=lambda temporary: seen.append(temporary.stat().st_size) or True,
    )

    assert seen == [len("short")]


def test_a_write_with_no_decision_still_replaces(tmp_path: Path):
    """Every existing caller passes no predicate and must be unaffected."""
    destination = tmp_path / "out.txt"
    destination.write_text("original")

    assert write_atomically(destination, lambda t: t.write_text("new")) is True
    assert destination.read_text() == "new"


def test_a_committed_file_keeps_the_destination_permissions(tmp_path: Path):
    """The mode is settled before the predicate runs, so deciding late does not
    cost the file its permissions."""
    destination = tmp_path / "out.txt"
    destination.write_text("original")
    os.chmod(destination, 0o640)

    write_atomically(
        destination,
        lambda temporary: temporary.write_text("replaced"),
        commit_if=lambda temporary: True,
    )

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
