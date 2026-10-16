"""Tests for reading and writing the package's YAML files.

A write is replaced whole or not at all: the state file, the manifests and the
fork point are read to decide what later operations do, and half of one is
worse than the previous version.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.io import atomic
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.io.yaml import (
    UnreadableYAMLError,
    read_yaml,
    version_problem,
    write_yaml,
    write_yaml_unchanged,
)


def test_a_written_file_opens_with_its_header_and_reads_back(tmp_path: Path):
    path = tmp_path / "record.yaml"

    write_yaml(path, {"version": 1, "scope": "--project all"}, header="# Written by AM-seg.\n")

    assert path.read_text(encoding="utf-8").startswith("# Written by AM-seg.\nversion: 1\n")
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


@pytest.mark.parametrize(
    ("version", "required", "problem"),
    [
        (None, False, None),
        (None, True, "whole number"),
        (1, False, None),
        (2, False, None),
        (3, False, "later version of AM-seg (file version 3, this reads up to 2)"),
        ("2", False, "whole number"),
        (True, False, "whole number"),
    ],
)
def test_a_version_this_am_seg_cannot_read_is_named(version, required, problem):
    found = version_problem(version, Path("f.yaml"), 2, required=required)

    assert found is None if problem is None else problem in found


def _refusing_to_open(monkeypatch, path: Path, times: int) -> list[int]:
    """Have *path* refuse to open *times* as Windows does a file being replaced."""
    calls: list[int] = []
    opened = Path.open

    def refuse(self, *args, **kwargs):
        if self == path:
            calls.append(1)
            if len(calls) <= times:
                raise PermissionError(13, "Permission denied", str(self))
        return opened(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse)
    return calls


def test_a_file_refused_while_it_is_replaced_is_read_once_it_is_there(tmp_path: Path, monkeypatch):
    """Starting a run read how it stood as the process, already finished, wrote that it had."""
    path = tmp_path / "process.yaml"
    write_yaml(path, {"outcome": "finished"})
    monkeypatch.setattr(atomic, "_OPEN_FILES_REFUSE", True)
    calls = _refusing_to_open(monkeypatch, path, times=2)

    assert read_yaml(path) == {"outcome": "finished"}
    assert len(calls) == 3


def test_a_file_refused_for_longer_than_a_rename_takes_is_unreadable(tmp_path: Path, monkeypatch):
    path = tmp_path / "process.yaml"
    write_yaml(path, {"outcome": "finished"})
    monkeypatch.setattr(atomic, "_OPEN_FILES_REFUSE", True)
    monkeypatch.setattr(atomic, "REFUSAL_WAIT_SECONDS", 0.05)
    calls = _refusing_to_open(monkeypatch, path, times=1_000_000)

    with pytest.raises(UnreadableYAMLError, match="Permission denied"):
        read_yaml(path)

    assert len(calls) > 1


def test_a_file_refused_where_open_files_are_shared_is_unreadable_at_once(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "process.yaml"
    write_yaml(path, {"outcome": "finished"})
    monkeypatch.setattr(atomic, "_OPEN_FILES_REFUSE", False)
    calls = _refusing_to_open(monkeypatch, path, times=1)

    with pytest.raises(UnreadableYAMLError):
        read_yaml(path)

    assert len(calls) == 1


def test_a_file_changed_since_it_was_read_is_not_written_over(tmp_path: Path):
    path = tmp_path / "state.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    expected = fingerprint_of(path)
    path.write_text("version: 1\nedited: by hand\n", encoding="utf-8")

    assert not write_yaml_unchanged(path, {"version": 1}, header="", expected=expected)
    assert read_yaml(path) == {"version": 1, "edited": "by hand"}
    assert write_yaml_unchanged(path, {"version": 1}, header="", expected=fingerprint_of(path))
