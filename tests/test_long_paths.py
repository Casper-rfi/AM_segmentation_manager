"""Tests for refusing writes Windows cannot reach without long paths.

Whether long paths work is stood in for: a macOS or Linux test machine reaches
every path, and the refusals are what matters here. The one test that asks the
real question runs only on Windows.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from am_segmentation_manager.io import long_paths
from am_segmentation_manager.io.atomic import write_atomically
from am_segmentation_manager.io.long_paths import (
    WINDOWS_MAX_PATH,
    PathTooLong,
    refuse_unreachable,
    unreachable,
    written_length,
)
from am_segmentation_manager.operations.archive import archived_path, projected_archive_path


def deep(base: Path, length: int) -> Path:
    """Return a directory under *base* whose path is at least *length* characters."""
    path = base
    while len(str(path.absolute())) < length:
        path = path / ("d" * 30)
    return path


@pytest.fixture
def no_long_paths(monkeypatch):
    """A process where Windows' old limit applies."""
    monkeypatch.setattr(long_paths, "long_paths_work", lambda: False)


def test_a_write_is_as_long_as_its_temporary_where_that_is_longer(tmp_path: Path):
    """A write lands by rename from a temporary beside it, which a short name
    can outgrow."""
    from am_segmentation_manager.io.atomic import write_atomically

    short = tmp_path / "a.tif"
    temporaries: list[Path] = []
    write_atomically(short, lambda temporary: (temporaries.append(temporary), temporary.touch()))

    assert written_length(short) == len(str(temporaries[0].absolute())) > len(str(short))


def test_nothing_is_unreachable_where_long_paths_work(tmp_path: Path):
    far = deep(tmp_path, 400) / "label.tif"

    assert unreachable([far], works=True) == []


def test_the_unreachable_are_named_longest_first(tmp_path: Path):
    near = tmp_path / "label.tif"
    far = deep(tmp_path, 300) / "label.tif"
    farther = deep(tmp_path, 350) / "label.tif"

    assert unreachable([near, far, farther], works=False) == [farther, far]


def test_a_refusal_says_how_many_the_longest_and_what_to_do(tmp_path: Path):
    far = deep(tmp_path, 300) / "label.tif"

    with pytest.raises(PathTooLong) as refused:
        refuse_unreachable([tmp_path / "a.tif", far], works=False)

    message = str(refused.value)
    assert "1 path(s)" in message
    assert str(far) in message
    assert f"{WINDOWS_MAX_PATH} characters" in message
    assert "shorter folder" in message


def test_a_single_write_that_cannot_be_reached_is_refused_before_anything_is_made(
    tmp_path: Path, no_long_paths
):
    far = deep(tmp_path, 300) / "label.tif"

    with pytest.raises(PathTooLong):
        write_atomically(far, lambda temporary: temporary.write_bytes(b"x"))

    assert not far.parent.exists()


def test_a_projected_archive_path_is_as_long_as_a_real_one(tmp_path: Path):
    """Which digest and second a version gets are unknown before the write, and
    the length must not depend on them."""
    label = tmp_path / "lamella" / "stage" / "sem_labels" / "img_000.tif"
    real = archived_path(
        tmp_path, "lamella/stage/sem_labels/img_000.tif", "9f86d081884c7d65",
        datetime.now(UTC),
    )

    assert len(str(projected_archive_path(tmp_path, label))) == len(str(real))


@pytest.mark.skipif(sys.platform == "win32", reason="the limit applies on Windows")
def test_every_path_is_reachable_away_from_windows():
    assert long_paths.long_paths_work()


@pytest.mark.skipif(sys.platform != "win32", reason="asks Windows itself")
def test_windows_is_asked_by_trying_and_answers():
    """Leaves nothing behind in the temporary directory, whatever it answers."""
    import tempfile

    long_paths.long_paths_work.cache_clear()
    before = set(Path(tempfile.gettempdir()).glob("am-seg-long-path-*"))

    assert isinstance(long_paths.long_paths_work(), bool)
    assert set(Path(tempfile.gettempdir()).glob("am-seg-long-path-*")) == before


@pytest.mark.parametrize("enabled", [0, 1])
def test_every_explanation_offers_a_shorter_folder(monkeypatch, enabled):
    """Whether Windows' setting is off or on but unused, the first thing to try is
    a shorter folder: it needs no administrator, and leaves files every other
    program can still handle. On is found on GitHub's Windows runner."""
    import types

    monkeypatch.setattr(long_paths.sys, "platform", "win32")

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    winreg = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=0,
        OpenKey=lambda root, key: Key(),
        QueryValueEx=lambda key, name: (enabled, 4),
    )
    monkeypatch.setitem(sys.modules, "winreg", winreg)

    assert "shorter folder" in long_paths.explanation()
