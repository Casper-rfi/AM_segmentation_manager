"""Tests for the io/path.py module."""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.io import path as io_path
from am_segmentation_manager.io.path import (
    NoRelativePath,
    exported_relative_path,
    make_path_relative,
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
