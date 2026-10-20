"""What a scan asks the filesystem, as distinct from what it finds.

On a network filesystem every question is a round trip, so a scan asks only what
the listings it already holds cannot answer.
"""
from __future__ import annotations

import os
from pathlib import Path

from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset


def test_a_scan_asks_about_nothing_that_is_not_there(tmp_path: Path, monkeypatch, sample_layout):
    """Optional directories and files -- predictions, plots, state, metadata --
    are answered from the listing of the directory that would hold them."""
    build_dataset(tmp_path, layout=sample_layout, plots=True, predictions=True)
    asked: list[str] = []
    original_scandir, original_stat = os.scandir, os.stat

    def scandir(path="."):
        asked.append(os.fspath(path))
        return original_scandir(path)

    def stat(path, *args, **kwargs):
        asked.append(os.fspath(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "stat", stat)
    Scanner(tmp_path).build_index()
    monkeypatch.undo()

    inside = [path for path in asked if Path(path).is_relative_to(tmp_path)]
    assert inside
    assert [path for path in inside if not Path(path).exists()] == []
