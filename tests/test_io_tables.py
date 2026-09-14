"""Tests for writing the CSV files the package exports.

An export may be something a person kept, so an existing file is refused unless
the caller says otherwise, and a write lands whole or not at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.io.tables import refuse_existing, write_csv


def test_rows_are_written_beneath_the_header(tmp_path: Path):
    path = tmp_path / "out" / "counts.csv"

    assert write_csv(path, ["modality", "images"], [("SEM", 2), ("FIB", 1)]) == 2

    assert path.read_text(encoding="utf-8").splitlines() == ["modality,images", "SEM,2", "FIB,1"]


def test_an_existing_file_is_refused_and_left_as_it_was(tmp_path: Path):
    path = tmp_path / "counts.csv"
    path.write_text("kept\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="nothing was written"):
        write_csv(path, ["modality", "images"], [("SEM", 2)])

    assert path.read_text(encoding="utf-8") == "kept\n"


def test_overwriting_replaces_an_existing_file(tmp_path: Path):
    path = tmp_path / "counts.csv"
    path.write_text("kept\n", encoding="utf-8")

    write_csv(path, ["modality", "images"], [("SEM", 2)], overwrite=True)

    assert path.read_text(encoding="utf-8").startswith("modality,images")


def test_a_mapping_row_leaves_a_column_it_does_not_name_blank(tmp_path: Path):
    path = tmp_path / "rows.csv"

    write_csv(path, ["image", "label"], [{"image": "a.tif"}])

    assert path.read_text(encoding="utf-8").splitlines()[1] == "a.tif,"


def test_a_mapping_row_naming_a_column_the_header_lacks_writes_nothing(tmp_path: Path):
    path = tmp_path / "rows.csv"

    with pytest.raises(ValueError, match="stray"):
        write_csv(path, ["image"], [{"image": "a.tif", "stray": 1}])

    assert not path.exists()


def test_a_write_that_fails_part_way_leaves_the_previous_file(tmp_path: Path):
    path = tmp_path / "rows.csv"
    path.write_text("kept\n", encoding="utf-8")

    def rows():
        yield ("a.tif",)
        raise OSError("No space left on device")

    with pytest.raises(OSError):
        write_csv(path, ["image"], rows(), overwrite=True)

    assert path.read_text(encoding="utf-8") == "kept\n"


def test_every_file_in_the_way_is_named(tmp_path: Path):
    train, validation = tmp_path / "train.csv", tmp_path / "val.csv"
    train.touch()
    validation.touch()

    with pytest.raises(FileExistsError) as refused:
        refuse_existing([train, validation, None, tmp_path / "absent.csv"])

    assert "train.csv" in str(refused.value) and "val.csv" in str(refused.value)
    refuse_existing([train, validation], overwrite=True)
