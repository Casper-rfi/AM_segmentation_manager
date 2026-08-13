"""Tests for training CSV path checking and repair."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.operations.csv_check import (
    check_csv_paths,
    summarize_results,
    write_fixed_csv,
)


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        writer.writerows(rows)


def test_check_csv_paths_classifies_ok_moved_and_missing(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    # Row 1: still exactly where the CSV says -> ok
    (sem / "img1.tif").write_text("img")
    (labels / "img1.tif").write_text("lbl")

    # Row 2: image was deselected after export -> moved
    (sem / "deselected").mkdir()
    (sem / "deselected" / "img2.tif").write_text("img")
    (labels / "img2.tif").write_text("lbl")

    # Row 3: label no longer exists anywhere -> missing
    (sem / "img3.tif").write_text("img")

    csv_path = tmp_path / "dataset.csv"
    _write_csv(
        csv_path,
        [
            ("proj/L1/stage1/sem/img1.tif", "proj/L1/stage1/labels/img1.tif"),
            ("proj/L1/stage1/sem/img2.tif", "proj/L1/stage1/labels/img2.tif"),
            ("proj/L1/stage1/sem/img3.tif", "proj/L1/stage1/labels/img3.tif"),
        ],
    )

    results = check_csv_paths(csv_path, tmp_path)
    assert [r.image.status for r in results] == ["ok", "moved", "ok"]
    assert [r.label.status for r in results] == ["ok", "ok", "missing"]
    assert results[1].image.resolved_path == sem / "deselected" / "img2.tif"

    lines, problem_count = summarize_results(results, tmp_path)
    assert problem_count == 2
    assert any("moved" in line for line in lines)
    assert any("MISSING" in line for line in lines)


def test_write_fixed_csv_aborts_on_missing_by_default(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (sem / "img1.tif").write_text("img")

    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [("proj/L1/stage1/sem/img1.tif", "proj/L1/stage1/labels/img1.tif")])

    results = check_csv_paths(csv_path, tmp_path)
    with pytest.raises(ValueError, match="missing"):
        write_fixed_csv(results, tmp_path, tmp_path / "out.csv", missing_policy="abort")
    assert not (tmp_path / "out.csv").exists()


def test_write_fixed_csv_updates_moved_rows(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (sem / "deselected").mkdir()
    (sem / "deselected" / "img1.tif").write_text("img")
    (labels / "img1.tif").write_text("lbl")

    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [("proj/L1/stage1/sem/img1.tif", "proj/L1/stage1/labels/img1.tif")])

    results = check_csv_paths(csv_path, tmp_path)
    out_path = write_fixed_csv(results, tmp_path, tmp_path / "out.csv", missing_policy="abort")

    with out_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["image"] == "proj/L1/stage1/sem/deselected/img1.tif"
    assert rows[0]["label"] == "proj/L1/stage1/labels/img1.tif"
    # original CSV is untouched
    assert csv_path.read_text().splitlines()[1].startswith("proj/L1/stage1/sem/img1.tif")


def test_write_fixed_csv_drop_missing_omits_row(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (sem / "img1.tif").write_text("img")
    (labels / "img1.tif").write_text("lbl")
    (sem / "img2.tif").write_text("img")
    # img2's label is missing entirely

    csv_path = tmp_path / "dataset.csv"
    _write_csv(
        csv_path,
        [
            ("proj/L1/stage1/sem/img1.tif", "proj/L1/stage1/labels/img1.tif"),
            ("proj/L1/stage1/sem/img2.tif", "proj/L1/stage1/labels/img2.tif"),
        ],
    )

    results = check_csv_paths(csv_path, tmp_path)
    out_path = write_fixed_csv(results, tmp_path, tmp_path / "out.csv", missing_policy="drop")

    with out_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["image"] == "proj/L1/stage1/sem/img1.tif"


def test_cli_check_training_csv_reports_and_fixes(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (sem / "deselected").mkdir()
    (sem / "deselected" / "img1.tif").write_text("img")
    (labels / "img1.tif").write_text("lbl")

    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [("proj/L1/stage1/sem/img1.tif", "proj/L1/stage1/labels/img1.tif")])

    # Read-only check: reports a problem via non-zero exit code
    rc = main(["check-training-csv", str(csv_path), "--start-path", str(tmp_path)])
    assert rc == 1

    # --fix writes a sibling "_checked" CSV, leaving the original alone
    rc = main(["check-training-csv", str(csv_path), "--start-path", str(tmp_path), "--fix"])
    assert rc == 0
    fixed_path = tmp_path / "dataset_checked.csv"
    assert fixed_path.exists()
    with fixed_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["image"] == "proj/L1/stage1/sem/deselected/img1.tif"


def test_cli_check_training_csv_fix_without_state_flag_still_ok(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (sem / "img1.tif").write_text("img")
    (labels / "img1.tif").write_text("lbl")

    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [("proj/L1/stage1/sem/img1.tif", "proj/L1/stage1/labels/img1.tif")])

    rc = main(["check-training-csv", str(csv_path), "--start-path", str(tmp_path)])
    assert rc == 0
