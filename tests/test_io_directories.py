"""Tests for the io/directories.py module: one read of a directory per scan."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.io.directories import directory_cache, list_directory


def _reads(monkeypatch) -> list[str]:
    """Return a list every directory read is appended to."""
    import os

    seen: list[str] = []
    original = os.scandir

    def watched(path):
        seen.append(str(path))
        return original(path)

    monkeypatch.setattr("am_segmentation_manager.io.directories.os.scandir", watched)
    return seen


def _names(directory: Path) -> list[str]:
    return [entry.path.name for entry in list_directory(directory)]


def test_a_directory_that_cannot_be_listed_reads_as_empty(tmp_path: Path):
    """A directory that cannot be listed must not abort a whole survey."""
    assert list_directory(tmp_path / "does-not-exist") == []


def test_a_file_where_a_directory_was_expected_lists_as_empty(tmp_path: Path):
    """The other way a listing fails, and the one os.scandir raises for."""
    not_a_directory = tmp_path / "file.tif"
    not_a_directory.write_bytes(b"")

    assert list_directory(not_a_directory) == []


def test_a_directory_is_read_once_inside_the_cache(tmp_path: Path, monkeypatch):
    """The saving: a stage is listed four times over on the way to being
    indexed, and each listing is a read of the whole directory."""
    (tmp_path / "sem").mkdir()
    reads = _reads(monkeypatch)

    with directory_cache():
        for _ in range(4):
            list_directory(tmp_path)

    assert reads.count(str(tmp_path)) == 1


def test_the_cache_reports_what_the_directory_holds(tmp_path: Path):
    """A held listing must answer exactly as an unheld one does."""
    (tmp_path / "sem").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "a.tif").write_bytes(b"")

    with directory_cache():
        cached = list_directory(tmp_path)

    assert cached == list_directory(tmp_path)


def test_nothing_is_held_outside_the_cache(tmp_path: Path, monkeypatch):
    """Held any longer than one traversal, a listing could go stale."""
    (tmp_path / "sem").mkdir()

    with directory_cache():
        list_directory(tmp_path)
    reads = _reads(monkeypatch)
    list_directory(tmp_path)

    assert reads == [str(tmp_path)]


def test_a_directory_created_after_the_block_is_seen(tmp_path: Path):
    """The block is what bounds the staleness, so it has to really end."""
    with directory_cache():
        assert _names(tmp_path) == []

    (tmp_path / "sem").mkdir()
    assert _names(tmp_path) == ["sem"]


def test_nesting_reuses_the_outer_block(tmp_path: Path, monkeypatch):
    """A scan enters the cache and calls a search that enters it again; the
    inner block must not throw away what the outer one holds."""
    (tmp_path / "sem").mkdir()
    reads = _reads(monkeypatch)

    with directory_cache():
        list_directory(tmp_path)
        with directory_cache():
            list_directory(tmp_path)
        list_directory(tmp_path)

    assert reads.count(str(tmp_path)) == 1
