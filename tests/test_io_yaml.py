"""Tests for reading and writing the package's YAML files.

A write is replaced whole or not at all: the state file, the manifests and the
fork point are read to decide what later operations do, and half of one is
worse than the previous version.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.io.yaml import read_yaml, write_yaml


def test_a_written_file_opens_with_its_header_and_reads_back(tmp_path: Path):
    path = tmp_path / "record.yaml"

    write_yaml(path, {"version": 1, "scope": "--project all"}, header="# Written by am-seg.\n")

    assert path.read_text(encoding="utf-8").startswith("# Written by am-seg.\nversion: 1\n")
    assert read_yaml(path) == {"version": 1, "scope": "--project all"}


def test_keys_keep_their_order_unless_sorting_is_asked_for(tmp_path: Path):
    kept, sorted_ = tmp_path / "kept.yaml", tmp_path / "sorted.yaml"

    write_yaml(kept, {"b": 1, "a": 2})
    write_yaml(sorted_, {"b": 1, "a": 2}, sort_keys=True)

    assert kept.read_text(encoding="utf-8") == "b: 1\na: 2\n"
    assert sorted_.read_text(encoding="utf-8") == "a: 2\nb: 1\n"


def test_text_beyond_ascii_is_written_as_itself(tmp_path: Path):
    path = tmp_path / "author.yaml"

    write_yaml(path, {"user": "Zoë Ångström"})

    assert "Zoë Ångström" in path.read_text(encoding="utf-8")


def test_a_write_that_fails_part_way_leaves_the_previous_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "record.yaml"
    write_yaml(path, {"version": 1})
    before = path.read_text(encoding="utf-8")
    _fail_writes_half_way(monkeypatch)

    with pytest.raises(OSError):
        write_yaml(path, {"version": 2, "projects": {"a": 1}})

    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == before


def _fail_writes_half_way(monkeypatch) -> None:
    """Make every text write stop half way through, as a full disk would."""
    real = Path.write_text

    def half(self, data, *args, **kwargs):
        real(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "write_text", half)
