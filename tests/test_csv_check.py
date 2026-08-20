"""Tests for re-checking an exported training CSV.

Curation cannot move a file any more, so a recorded path either resolves or the
file is gone. What is pinned here is that the check says which, that pruning
keeps the split the CSV records, and that it never rewrites a path.
"""
from __future__ import annotations

import csv
from pathlib import Path

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

    assert [result.image.status for result in results] == ["ok", "ok", "missing"]
    assert [result.label.status for result in results] == ["ok", "missing", "ok"]
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
