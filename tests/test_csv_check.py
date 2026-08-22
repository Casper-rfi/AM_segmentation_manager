"""Tests for re-checking an exported training CSV.

Curation cannot move a file any more, so a recorded path either resolves or the
file is gone. What is pinned here is that the check says which, that pruning
keeps the split the CSV records, and that it never rewrites a path.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.operations.csv_check import (
    check_csv_paths,
    summarize_results,
    write_pruned_csv,
)


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        writer.writerows(rows)


def _pair(stage: Path, name: str, *, image: bool = True, label: bool = True) -> tuple[str, str]:
    """Write an image and label pair, optionally leaving one out."""
    (stage / "sem").mkdir(parents=True, exist_ok=True)
    (stage / "labels").mkdir(parents=True, exist_ok=True)
    if image:
        (stage / "sem" / name).write_text("img")
    if label:
        (stage / "labels" / name).write_text("lbl")
    return (f"proj/L1/stage1/sem/{name}", f"proj/L1/stage1/labels/{name}")


def test_a_row_is_either_ok_or_missing(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    rows = [
        _pair(stage, "img1.tif"),
        _pair(stage, "img2.tif", label=False),
        _pair(stage, "img3.tif", image=False),
    ]
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, rows)

    results = check_csv_paths(csv_path, tmp_path)

    assert [r.columns["image"].status for r in results] == ["ok", "ok", "missing"]
    assert [r.columns["label"].status for r in results] == ["ok", "missing", "ok"]
    assert [result.ok for result in results] == [True, False, False]


def test_a_curated_image_still_resolves(tmp_path: Path):
    """The whole reason the repair half of this module went away."""
    from am_segmentation_manager.operations.selection import deselect_records
    from am_segmentation_manager.scanner import Scanner

    stage = tmp_path / "proj" / "L1" / "stage1"
    rows = [_pair(stage, "img1.tif")]
    (tmp_path / "proj" / "experiment.yaml").write_text("name: proj\n", encoding="utf-8")
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, rows)

    project = Scanner(tmp_path).build_index().all_projects()[0]
    deselect_records(project.lamellae[0].stages[0].images)

    assert all(result.ok for result in check_csv_paths(csv_path, tmp_path))


def test_the_summary_names_every_missing_path(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif"), _pair(stage, "img2.tif", label=False)])

    lines, problem_count = summarize_results(check_csv_paths(csv_path, tmp_path), tmp_path)

    assert problem_count == 1
    assert any("MISSING" in line and "img2.tif" in line for line in lines)


def test_pruning_drops_the_rows_that_do_not_resolve(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif"), _pair(stage, "img2.tif", label=False)])
    results = check_csv_paths(csv_path, tmp_path)

    output_path, dropped = write_pruned_csv(results, tmp_path / "out.csv")

    assert dropped == 1
    with output_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [row["image"] for row in rows] == ["proj/L1/stage1/sem/img1.tif"]


def test_pruning_copies_paths_through_untouched(tmp_path: Path):
    """Nothing rewrites a path, because nothing moves a file."""
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif")])
    original = csv_path.read_text(encoding="utf-8")

    output_path, dropped = write_pruned_csv(
        check_csv_paths(csv_path, tmp_path), tmp_path / "out.csv"
    )

    assert dropped == 0
    assert output_path.read_text(encoding="utf-8") == original


def test_the_command_reports_a_missing_file_by_its_exit_code(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif", label=False)])

    assert main(["check-training-csv", str(csv_path), "--start-path", str(tmp_path)]) == 1


def test_the_command_passes_an_intact_export(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif")])

    assert main(["check-training-csv", str(csv_path), "--start-path", str(tmp_path)]) == 0


def test_the_command_prunes_beside_the_original(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif"), _pair(stage, "img2.tif", label=False)])
    original = csv_path.read_text(encoding="utf-8")

    assert main([
        "check-training-csv", str(csv_path), "--start-path", str(tmp_path), "--prune",
    ]) == 0

    with (tmp_path / "dataset_checked.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 1
    assert csv_path.read_text(encoding="utf-8") == original, "the original is left alone"


def test_output_options_need_prune(tmp_path: Path):
    stage = tmp_path / "proj" / "L1" / "stage1"
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, [_pair(stage, "img1.tif")])

    assert main([
        "check-training-csv", str(csv_path), "--start-path", str(tmp_path),
        "--output", str(tmp_path / "out.csv"),
    ]) == 2
    assert not (tmp_path / "out.csv").exists()


# ---------------------------------------------------------------------------
# Checking a paired export
# ---------------------------------------------------------------------------

def test_a_paired_csv_is_checked_by_its_own_header(tmp_path: Path):
    """The columns are read from the file rather than assumed, so one command
    checks a single-beam export, a paired one, and a file written before either
    shape was settled."""
    from am_segmentation_manager.operations.csv_check import path_columns

    stage = tmp_path / "p" / "01" / "s"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / "a.tif").write_bytes(b"x")
    (stage / "sem_labels").mkdir()
    (stage / "sem_labels" / "a.tif").write_bytes(b"x")

    csv_path = tmp_path / "paired_train.csv"
    csv_path.write_text(
        "sem_image,sem_label,fib_image,fib_label\n"
        "p/01/s/sem/a.tif,p/01/s/sem_labels/a.tif,,\n"
        "p/01/s/sem/gone.tif,p/01/s/sem_labels/a.tif,,\n",
        encoding="utf-8",
    )

    results = check_csv_paths(csv_path, tmp_path)

    assert path_columns(
        ["sem_image", "sem_label", "fib_image", "fib_label"]
    ) == ["sem_image", "sem_label", "fib_image", "fib_label"]
    assert [result.ok for result in results] == [True, False]


def test_a_blank_cell_is_not_a_missing_file(tmp_path: Path):
    """A paired export leaves one wherever a beam has no image, or no label
    worth pairing with one. Reporting those as missing would make every
    single-beam acquisition look like a fault."""
    stage = tmp_path / "p" / "01" / "s"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / "a.tif").write_bytes(b"x")

    csv_path = tmp_path / "paired_train.csv"
    csv_path.write_text(
        "sem_image,sem_label,fib_image,fib_label\n"
        "p/01/s/sem/a.tif,,,\n",
        encoding="utf-8",
    )

    (result,) = check_csv_paths(csv_path, tmp_path)

    assert result.ok
    assert result.columns["fib_image"].status == "blank"
    assert result.columns["sem_image"].status == "ok"


def test_a_file_with_no_path_column_is_refused(tmp_path: Path):
    csv_path = tmp_path / "not-an-export.csv"
    csv_path.write_text("lamella,count\n01,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no image or label column"):
        check_csv_paths(csv_path, tmp_path)


def test_pruning_a_paired_csv_writes_a_paired_csv(tmp_path: Path):
    from am_segmentation_manager.operations.csv_check import write_pruned_csv

    stage = tmp_path / "p" / "01" / "s"
    (stage / "sem").mkdir(parents=True)
    (stage / "sem" / "a.tif").write_bytes(b"x")
    csv_path = tmp_path / "paired_train.csv"
    csv_path.write_text(
        "sem_image,sem_label,fib_image,fib_label\n"
        "p/01/s/sem/a.tif,,,\n"
        "p/01/s/sem/gone.tif,,,\n",
        encoding="utf-8",
    )
    results = check_csv_paths(csv_path, tmp_path)

    written, dropped = write_pruned_csv(results, tmp_path / "pruned.csv")

    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sem_image,sem_label,fib_image,fib_label"
    assert dropped == 1
    assert len(lines) == 2
