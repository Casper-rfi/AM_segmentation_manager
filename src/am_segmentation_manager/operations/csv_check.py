"""Consistency checking and repair for exported training/validation CSVs.

Image and label states can change after a CSV has been exported by
``create-training-csv`` (further select/deselect/correct operations move
files between state folders), which leaves the CSV's recorded paths stale.
This module re-checks each path against the current filesystem state and can
optionally rewrite the CSV with corrected paths.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..io.path import make_path_relative
from .selection import base_path, existing_variant

logger = logging.getLogger(__name__)

PathStatus = Literal["ok", "moved", "missing"]
MissingPolicy = Literal["abort", "drop", "keep"]


@dataclass(frozen=True)
class ColumnCheck:
    """Result of checking a single image/label path from a CSV row."""

    recorded_value: str
    status: PathStatus
    resolved_path: Path | None


@dataclass(frozen=True)
class RowCheck:
    """Result of checking both columns of one CSV data row."""

    row_number: int  # 1-based, counting only data rows (excludes the header)
    image: ColumnCheck
    label: ColumnCheck

    @property
    def ok(self) -> bool:
        """Return whether both columns still point at their recorded location."""
        return self.image.status == "ok" and self.label.status == "ok"


def _check_path(recorded_value: str, start_path: Path) -> ColumnCheck:
    """Classify a single CSV-recorded path as ok, moved, or missing."""
    absolute = start_path / recorded_value
    if absolute.exists():
        return ColumnCheck(recorded_value, "ok", absolute)
    found = existing_variant(base_path(absolute))
    if found is not None:
        return ColumnCheck(recorded_value, "moved", found)
    return ColumnCheck(recorded_value, "missing", None)


def check_csv_paths(csv_path: Path, start_path: Path) -> list[RowCheck]:
    """Check every image/label path recorded in *csv_path* against disk.

    Paths are resolved relative to *start_path* (the scan directory the CSV
    was originally exported against). Returns one :class:`RowCheck` per data
    row, in file order.
    """
    results: list[RowCheck] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_number, row in enumerate(reader, start=1):
            image_check = _check_path(row["image"], start_path)
            label_check = _check_path(row["label"], start_path)
            results.append(RowCheck(row_number, image_check, label_check))
    return results


def summarize_results(results: list[RowCheck], start_path: Path) -> tuple[list[str], int]:
    """Build a human-readable report for *results*.

    Returns ``(lines, problem_count)`` where *problem_count* is the number of
    rows that are not fully "ok".
    """
    counts = {"ok": 0, "moved": 0, "missing": 0}
    lines: list[str] = []
    for result in results:
        for column_name, check in (("image", result.image), ("label", result.label)):
            counts[check.status] += 1
            if check.status == "ok":
                continue
            if check.status == "moved":
                new_rel = make_path_relative(check.resolved_path, start_path)
                lines.append(
                    f"row {result.row_number}: {column_name} moved: "
                    f"{check.recorded_value} -> {new_rel}"
                )
            else:
                lines.append(f"row {result.row_number}: {column_name} MISSING: {check.recorded_value}")

    problem_count = sum(1 for r in results if not r.ok)
    summary = (
        f"Checked {len(results)} row(s): {counts['ok']} ok, "
        f"{counts['moved']} moved, {counts['missing']} missing"
    )
    return [summary, *lines], problem_count


def write_fixed_csv(
    results: list[RowCheck],
    start_path: Path,
    output_path: Path,
    missing_policy: MissingPolicy = "abort",
) -> Path:
    """Write a corrected CSV with paths updated for files that moved state.

    "moved" columns are rewritten to their current relative location. "ok"
    columns are left as recorded. Rows containing a "missing" column are
    handled according to *missing_policy*:

    * ``"abort"`` (default): raise ``ValueError`` without writing anything,
      so a stale CSV is never silently repaired around data that can no
      longer be found.
    * ``"drop"``: omit the row from the output CSV.
    * ``"keep"``: write the row with its unresolved path unchanged.
    """
    missing_rows = [r for r in results if r.image.status == "missing" or r.label.status == "missing"]
    if missing_policy == "abort" and missing_rows:
        row_numbers = ", ".join(str(r.row_number) for r in missing_rows)
        raise ValueError(
            f"{len(missing_rows)} row(s) have paths missing from every workflow "
            f"state (rows: {row_numbers}). Re-run with --drop-missing or "
            "--keep-missing to write the CSV anyway."
        )

    rows: list[tuple[str, str]] = []
    for result in results:
        if missing_policy == "drop" and (result.image.status == "missing" or result.label.status == "missing"):
            continue
        image_value = (
            make_path_relative(result.image.resolved_path, start_path)
            if result.image.status == "moved"
            else result.image.recorded_value
        )
        label_value = (
            make_path_relative(result.label.resolved_path, start_path)
            if result.label.status == "moved"
            else result.label.recorded_value
        )
        rows.append((image_value, label_value))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        writer.writerows(rows)
    logger.info("Wrote %d row(s) to %s", len(rows), output_path)
    return output_path
