"""Re-checking that an exported training CSV still resolves.

A recorded path cannot go stale through curation any more -- workflow state is
declared in a file and nothing moves -- so a path that does not resolve means
the file is genuinely gone: deleted, renamed, or on a drive that is not
mounted. A row is therefore either ``ok`` or ``missing``, and the repair this
module used to perform has nothing left to repair.

Pruning is still worth having. A CSV holds a particular train/validation split,
and losing a few images is not a reason to regenerate the whole thing.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

PathStatus = Literal["ok", "missing"]


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
        """Return whether both of the row's files are still there."""
        return self.image.status == "ok" and self.label.status == "ok"


def _check_path(recorded_value: str, start_path: Path) -> ColumnCheck:
    """Classify a single CSV-recorded path as ok or missing."""
    absolute = start_path / recorded_value
    if absolute.exists():
        return ColumnCheck(recorded_value, "ok", absolute)
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
            results.append(
                RowCheck(
                    row_number,
                    _check_path(row["image"], start_path),
                    _check_path(row["label"], start_path),
                )
            )
    return results


def summarize_results(results: list[RowCheck], start_path: Path) -> tuple[list[str], int]:
    """Build a human-readable report for *results*.

    Returns ``(lines, problem_count)`` where *problem_count* counts the rows
    with a file that could not be found.
    """
    lines: list[str] = []
    missing = 0
    for result in results:
        for column_name, check in (("image", result.image), ("label", result.label)):
            if check.status == "missing":
                missing += 1
                lines.append(
                    f"row {result.row_number}: {column_name} MISSING: "
                    f"{check.recorded_value}"
                )

    problem_count = sum(1 for result in results if not result.ok)
    summary = (
        f"Checked {len(results)} row(s) against {start_path}: "
        f"{len(results) - problem_count} ok, {problem_count} with a missing file "
        f"({missing} path(s))"
    )
    return [summary, *lines], problem_count


def write_pruned_csv(
    results: list[RowCheck], output_path: Path
) -> tuple[Path, int]:
    """Write *results* to *output_path*, without the rows that no longer resolve.

    Paths are copied through exactly as recorded: nothing here rewrites one,
    because curation no longer moves a file.

    Returns:
        The path written, and how many rows were dropped.
    """
    kept = [result for result in results if result.ok]
    dropped = len(results) - len(kept)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        writer.writerows(
            (result.image.recorded_value, result.label.recorded_value)
            for result in kept
        )
    logger.info("Wrote %d row(s) to %s, dropping %d", len(kept), output_path, dropped)
    return output_path, dropped
