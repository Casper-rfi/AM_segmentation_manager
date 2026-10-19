"""Tests for atomic writes and file fingerprints.

These were exercised only through label saving, which skips without the image
stack installed. They are a shared primitive now, so they are pinned directly.
"""
from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from am_segmentation_manager.io import atomic
from am_segmentation_manager.io.atomic import (
    append_whole,
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


def test_a_temporary_is_left_out_of_every_listing(tmp_path: Path):
    """A hard kill mid-write leaves it beside the label, where it was listed as a label of no image."""
    from am_segmentation_manager.layout import image_files

    listed: list[Path] = []

    def write(temporary: Path) -> None:
        temporary.write_bytes(b"label")
        listed.extend(image_files(tmp_path))

    write_atomically(tmp_path / "img.tif", write)

    assert listed == []


def test_a_write_creates_the_destination_directory(tmp_path: Path):
    destination = tmp_path / "missing" / "out.txt"
    write_atomically(destination, lambda temporary: temporary.write_text("written"))
    assert destination.read_text() == "written"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
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


def _refusing(monkeypatch, times: int) -> list[int]:
    """Have ``os.replace`` refuse *times* as Windows does a file in use, and count the calls."""
    calls: list[int] = []
    replace = os.replace

    def refuse(source, destination):
        calls.append(1)
        if len(calls) <= times:
            raise PermissionError(errno.EACCES, "in use by another process", str(destination))
        replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", refuse)
    return calls


def test_a_rename_refused_while_the_file_is_read_is_made_again(tmp_path: Path, monkeypatch):
    """A training process recording that it finished lost to the window reading how it stood."""
    destination = tmp_path / "process.yaml"
    destination.write_text("old")
    monkeypatch.setattr(atomic, "_OPEN_FILES_REFUSE", True)
    calls = _refusing(monkeypatch, times=2)

    assert write_atomically(destination, lambda temporary: temporary.write_text("new"))

    assert destination.read_text() == "new"
    assert len(calls) == 3


def test_a_rename_refused_for_longer_than_a_read_takes_is_raised(tmp_path: Path, monkeypatch):
    destination = tmp_path / "process.yaml"
    destination.write_text("old")
    monkeypatch.setattr(atomic, "_OPEN_FILES_REFUSE", True)
    monkeypatch.setattr(atomic, "REFUSAL_WAIT_SECONDS", 0.05)
    calls = _refusing(monkeypatch, times=1_000_000)

    with pytest.raises(PermissionError):
        write_atomically(destination, lambda temporary: temporary.write_text("new"))

    assert destination.read_text() == "old"
    assert len(calls) > 1
    assert list(tmp_path.iterdir()) == [destination]


def test_a_rename_refused_where_open_files_are_shared_is_raised_at_once(
    tmp_path: Path, monkeypatch
):
    """There it is a destination that may not be written, which waiting does not change."""
    destination = tmp_path / "process.yaml"
    monkeypatch.setattr(atomic, "_OPEN_FILES_REFUSE", False)
    calls = _refusing(monkeypatch, times=1)

    with pytest.raises(PermissionError):
        write_atomically(destination, lambda temporary: temporary.write_text("new"))

    assert len(calls) == 1


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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
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


def test_a_file_replaced_by_rename_has_changed_whatever_its_size_and_time(tmp_path: Path):
    """How every write here lands, and how another process's does: the clock
    alone cannot tell two such writes apart within one of its steps."""
    path = tmp_path / "label.tif"
    path.write_bytes(b"12345678")
    before = fingerprint_of(path)
    assert before is not None

    replacement = tmp_path / "replacement.tif"
    replacement.write_bytes(b"87654321")
    os.utime(replacement, ns=(before.modified_ns, before.modified_ns))
    os.replace(replacement, path)
    after = fingerprint_of(path)

    assert after is not None
    assert (after.size, after.modified_ns) == (before.size, before.modified_ns)
    assert after != before


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------

def test_appends_land_in_order_exactly_as_given(tmp_path: Path):
    """Bytes, not text: a line ends in \\n on Windows too."""
    log = tmp_path / "log.jsonl"
    append_whole(log, b"first\n")
    append_whole(log, b"second\n")

    assert log.read_bytes() == b"first\nsecond\n"


def test_a_failed_write_leaves_the_file_free_for_the_next_writer(tmp_path: Path, monkeypatch):
    """A lock left held would stall every later append, in any process."""
    log = tmp_path / "log.jsonl"

    def refuse(descriptor: int, data: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(atomic.os, "write", refuse)
    with pytest.raises(OSError):
        append_whole(log, b"lost\n")
    monkeypatch.undo()

    program = (
        "import sys; from pathlib import Path;"
        "from am_segmentation_manager.io.atomic import append_whole;"
        "append_whole(Path(sys.argv[1]), b'landed\\n')"
    )
    subprocess.run([sys.executable, "-c", program, str(log)], check=True, timeout=60)

    assert log.read_bytes() == b"landed\n"


@pytest.mark.skipif(sys.platform == "win32", reason="flock is not used on Windows")
def test_a_filesystem_without_locks_still_takes_the_append(tmp_path: Path, monkeypatch):
    """Some network shares refuse locks; macOS and Linux keep an append whole
    without one, so recording there goes on as before."""
    import fcntl

    def unsupported(descriptor: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    log = tmp_path / "log.jsonl"
    append_whole(log, b"kept\n")

    assert log.read_bytes() == b"kept\n"


@pytest.mark.skipif(sys.platform == "win32", reason="flock is not used on Windows")
def test_any_other_failure_to_lock_is_raised(tmp_path: Path, monkeypatch):
    import fcntl

    def broken(descriptor: int, operation: int) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(fcntl, "flock", broken)
    with pytest.raises(OSError):
        append_whole(tmp_path / "log.jsonl", b"not written\n")


# -- a lock taken without waiting ---------------------------------------------


def test_a_free_lock_is_taken_and_a_held_one_is_not(tmp_path: Path):
    path = tmp_path / "held.lock"
    first = os.open(path, os.O_RDWR | os.O_CREAT)
    second = os.open(path, os.O_RDWR)
    try:
        assert atomic.try_lock(first) is True
        assert atomic.try_lock(second) is False
    finally:
        os.close(first)
        os.close(second)


def test_a_lock_is_given_back_when_its_file_is_closed(tmp_path: Path):
    """Which is what a process ending does, however it ends."""
    path = tmp_path / "held.lock"
    first = os.open(path, os.O_RDWR | os.O_CREAT)
    assert atomic.try_lock(first) is True
    os.close(first)

    second = os.open(path, os.O_RDWR)
    try:
        assert atomic.try_lock(second) is True
    finally:
        os.close(second)


def test_a_lock_another_process_holds_is_not_taken(tmp_path: Path):
    path = tmp_path / "held.lock"
    path.write_bytes(b"")
    holder = subprocess.Popen(
        [
            sys.executable, "-c",
            "import os, sys, time\n"
            f"sys.path.insert(0, {str(Path(atomic.__file__).parents[2])!r})\n"
            "from am_segmentation_manager.io.atomic import try_lock\n"
            f"d = os.open({str(path)!r}, os.O_RDWR)\n"
            "print(try_lock(d), flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "True"
        descriptor = os.open(path, os.O_RDWR)
        try:
            assert atomic.try_lock(descriptor) is False
        finally:
            os.close(descriptor)
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()

    # Windows gives a killed process's locks back a moment after the kill returns.
    deadline = time.monotonic() + 10.0
    while True:
        descriptor = os.open(path, os.O_RDWR)
        try:
            taken = atomic.try_lock(descriptor)
        finally:
            os.close(descriptor)
        if taken or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert taken is True, "a killed process gives its lock back"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows always offers locks")
def test_a_filesystem_without_locks_is_said_so_rather_than_raised(tmp_path: Path, monkeypatch):
    import fcntl

    def unsupported(descriptor, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    descriptor = os.open(tmp_path / "held.lock", os.O_RDWR | os.O_CREAT)
    try:
        assert atomic.try_lock(descriptor) is None
    finally:
        os.close(descriptor)
