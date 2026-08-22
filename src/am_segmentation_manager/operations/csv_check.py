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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

#: What a recorded path turned out to be. ``blank`` is its own answer rather
#: than an absent column: a paired export leaves a cell empty wherever a beam
#: has no image, or no label worth pairing with one, and that is a fact about
#: the acquisition rather than a file that has gone. Keeping it in the row is
#: also what lets a pruned copy come out the same shape as the original.
PathStatus = Literal["ok", "missing", "blank"]

#: What a path column is called, or ends in. ``sem_image`` and ``fib_label``
#: are the paired export's; ``image`` and ``label`` a single-beam one's.
PATH_COLUMNS = frozenset({"image", "label"})


@dataclass(frozen=True)
class ColumnCheck:
    """Result of checking a single image/label path from a CSV row."""

    recorded_value: str
    status: PathStatus
    resolved_path: Path | None


@dataclass(frozen=True)
class RowCheck:
    """Result of checking every path column of one CSV data row.

    Attributes:
        row_number: 1-based, counting data rows only.
        columns: Column name to what was found there, in the file's own order.
            A mapping rather than two fields, because a training CSV covering
            both beams carries four path columns and one covering one carries
            two -- and a file written before either shape was settled still has
            to be checkable. Every path column is present, blanks included.
    """

    row_number: int
    columns: dict[str, ColumnCheck]

    @property
    def ok(self) -> bool:
        """Return whether nothing this row records has gone.

        A blank records nothing, so it cannot be missing.
        """
        return not any(
            check.status == "missing" for check in self.columns.values()
        )


def _check_path(recorded_value: str, start_path: Path) -> ColumnCheck:
    """Classify a single CSV-recorded path as ok, missing or blank."""
    if not recorded_value.strip():
        return ColumnCheck(recorded_value, "blank", None)
    absolute = start_path / recorded_value
    if absolute.exists():
        return ColumnCheck(recorded_value, "ok", absolute)
    return ColumnCheck(recorded_value, "missing", None)


def path_columns(fieldnames: Sequence[str] | None) -> list[str]:
    """Return which of a CSV's columns hold paths.

    Read from the header rather than assumed, so one function checks a
    single-beam export, a paired one, and a file written by an older version.
    A column holds a path if it is called ``image`` or ``label`` or ends in
    ``_image`` or ``_label`` -- which is what every export here names them.
    """
    return [
        name for name in fieldnames or []
        if name in PATH_COLUMNS or name.rpartition("_")[2] in PATH_COLUMNS
    ]


def check_csv_paths(csv_path: Path, start_path: Path) -> list[RowCheck]:
    """Check every path recorded in *csv_path* against disk.

    Paths are resolved relative to *start_path* (the scan directory the CSV was
    originally exported against). A blank cell is not a missing file: a paired
    export leaves one wherever a beam has no image, or no label worth pairing
    with one, so it is skipped rather than reported.

    Returns:
        One :class:`RowCheck` per data row, in file order.

    Raises:
        ValueError: If the file has no column that could hold a path, which
            means it is not one of these exports.
    """
    results: list[RowCheck] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = path_columns(reader.fieldnames)
        if not columns:
            raise ValueError(
                f"{csv_path} has no image or label column: "
                f"{', '.join(reader.fieldnames or ['(no header)'])}"
            )
        for row_number, row in enumerate(reader, start=1):
            results.append(
                RowCheck(
                    row_number,
                    {
                        name: _check_path(row.get(name) or "", start_path)
                        for name in columns
                    },
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
        for column_name, check in result.columns.items():
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
    # Taken from the rows rather than assumed, so a paired CSV is pruned into
    # one of the same shape.
    columns = list(results[0].columns) if results else ["image", "label"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(
            tuple(result.columns[name].recorded_value for name in columns)
            for result in kept
        )
    logger.info("Wrote %d row(s) to %s, dropping %d", len(kept), output_path, dropped)
    return output_path, dropped
