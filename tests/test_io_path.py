"""Tests for the io/path.py module."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.io.path import make_path_relative


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
