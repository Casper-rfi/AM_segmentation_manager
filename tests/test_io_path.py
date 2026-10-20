"""Tests for the io/path.py module."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from am_segmentation_manager.io import path as io_path
from am_segmentation_manager.io.directories import directory_cache, list_directory
from am_segmentation_manager.io.path import (
    NoRelativePath,
    exported_relative_path,
    make_path_relative,
    resolution_cache,
    resolved,
)

needs_symlinks = pytest.mark.skipif(
    os.name == "nt", reason="creating a symbolic link on Windows needs a privilege"
)


def test_make_path_relative_subpath():
    base = Path("/workspace/data")
    target = Path("/workspace/data/proj1/img.tif")
    rel = make_path_relative(target, base)
    assert rel == "proj1/img.tif"


def test_make_path_relative_outside():
    base = Path("/workspace/data")
    target = Path("/outside/other/img.tif")
    rel = make_path_relative(target, base)
    assert "outside" in rel


def test_a_path_outside_the_base_climbs_to_it_rather_than_being_refused():
    assert exported_relative_path(
        Path("/workspace/other/img.tif"), Path("/workspace/data")
    ) == "../other/img.tif"


def _another_drive(monkeypatch) -> None:
    """Make relpath refuse as Windows' does between two drives."""
    def refuse(path: object, base: object) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(io_path.os.path, "relpath", refuse)


def test_an_export_refuses_a_path_with_no_relative_path(monkeypatch):
    """An absolute path in a training CSV names a file on one machine only."""
    _another_drive(monkeypatch)

    with pytest.raises(NoRelativePath, match="same drive"):
        exported_relative_path(Path("/data/img.tif"), Path("/export"))


def test_showing_a_path_with_no_relative_path_shows_it_whole(monkeypatch):
    """A report that could not be shown would hide the file it is about."""
    _another_drive(monkeypatch)

    shown = make_path_relative(Path("/data/img.tif"), Path("/export"))

    assert shown == Path("/data/img.tif").resolve().as_posix()


@pytest.fixture
def linked_tree(tmp_path: Path) -> Path:
    """A directory of files, a link to it, and a file that is a link to another."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.tif").write_bytes(b"")
    if os.name != "nt":
        (tmp_path / "link").symlink_to(data, target_is_directory=True)
        (data / "b.tif").symlink_to(data / "a.tif")
    return tmp_path


@pytest.mark.parametrize(
    "written",
    [
        "data/a.tif",
        pytest.param("link/a.tif", marks=needs_symlinks),
        pytest.param("data/b.tif", marks=needs_symlinks),
        pytest.param("link/../data/a.tif", marks=needs_symlinks),
        "data/missing.tif",
        "missing/a.tif",
        "data/../data/a.tif",
    ],
    ids=["plain", "through-a-linked-directory", "a-linked-file", "up-out-of-a-link",
         "missing-file", "missing-directory", "dot-dot"],
)
@pytest.mark.parametrize("listed", [False, True], ids=["asked", "listed"])
def test_a_path_resolves_as_path_resolve_resolves_it(
    linked_tree: Path, written: str, listed: bool
):
    """Exports and training runs write these paths; resolving faster must not change one,
    whether the filesystem is asked or a listing answers."""
    path = linked_tree / written

    with directory_cache(), resolution_cache():
        if listed:
            list_directory(path.parent.parent)
            list_directory(path.parent)
        first, again = resolved(path), resolved(path)

    assert first == again == path.resolve()


def test_a_relative_path_resolves_as_path_resolve_resolves_it(linked_tree: Path, monkeypatch):
    monkeypatch.chdir(linked_tree)

    with resolution_cache():
        assert resolved(Path("data/a.tif")) == Path("data/a.tif").resolve()


def _lstats(monkeypatch) -> list[str]:
    """Return a list every lstat is appended to."""
    asked: list[str] = []
    real = os.lstat

    def watched(name, *args, **kwargs):
        asked.append(os.fspath(name))
        return real(name, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", watched)
    return asked


@pytest.mark.skipif(os.name == "nt", reason="Windows resolves every path whole")
def test_no_directory_is_asked_about_twice(tmp_path: Path, monkeypatch):
    """Resolving a path asks about every directory from the root down: on a network
    filesystem a round trip each, for every row of an export."""
    files = [
        tmp_path / directory / f"img_{number}.tif"
        for directory in ("sem", "sem_labels") for number in range(10)
    ]
    asked = _lstats(monkeypatch)

    with resolution_cache():
        for path in files:
            resolved(path)

    assert len(asked) == len(set(asked))
    assert len(asked) <= len(files) + len(tmp_path.parts) + 2


@pytest.mark.skipif(os.name == "nt", reason="Windows resolves every path whole")
def test_a_held_listing_says_which_files_are_links(tmp_path: Path, monkeypatch):
    """A listing names every file in a directory at once; asking each costs one apiece."""
    (tmp_path / "sem").mkdir()
    files = [tmp_path / "sem" / f"img_{number}.tif" for number in range(20)]
    for path in files:
        path.write_bytes(b"")

    with directory_cache(), resolution_cache():
        list_directory(tmp_path / "sem")
        asked = _lstats(monkeypatch)
        for path in files:
            resolved(path)

    assert [name for name in asked if Path(name) in files] == []


def test_nothing_is_remembered_outside_the_block(linked_tree: Path):
    path = linked_tree / "data" / "a.tif"
    with resolution_cache():
        resolved(path)

    assert io_path._resolutions.get() is None
