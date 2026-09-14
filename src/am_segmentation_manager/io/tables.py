"""Writing the CSV files the package exports.

**An existing file is refused unless the caller says to replace it**: an export is
something a person may have kept, edited or deposited, and no run replaces one
unasked. :func:`refuse_existing` asks before any work is done; a write asks again
as the file lands, whole, through :func:`.atomic.write_atomically`.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .atomic import write_atomically

logger = logging.getLogger(__name__)

__all__ = ["refuse_existing", "write_csv"]


def refuse_existing(paths: Iterable[Path | None], *, overwrite: bool = False) -> None:
    """Raise if any of *paths* already exists and *overwrite* is False.

    Raises:
        FileExistsError: Naming every path in the way.
    """
    if overwrite:
        return
    existing = [path for path in paths if path is not None and path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        verb = "exists" if len(existing) == 1 else "exist"
        raise FileExistsError(f"{names} already {verb}; nothing was written")


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[object] | Mapping[str, object]],
    *,
    overwrite: bool = False,
) -> int:
    """Write *rows* beneath the header *columns* to *path*, returning how many were written.

    A row is its cells in column order, or a mapping from column to cell in which
    a column it does not name is left blank.

    Raises:
        FileExistsError: If *path* exists, or appears while it is being written,
            and *overwrite* is False.
        ValueError: If a mapping row names a column that is not in *columns*.
    """
    refuse_existing([path], overwrite=overwrite)
    known = set(columns)
    written = 0

    def write(temporary: Path) -> None:
        nonlocal written
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(_cells(row, columns, known))
                written += 1

    landed = write_atomically(
        path,
        write,
        commit_if=None if overwrite else (lambda _temporary: not path.exists()),
    )
    if not landed:
        raise FileExistsError(f"{path} appeared while it was being written; nothing was written")
    logger.debug("Wrote %d row(s) to %s", written, path)
    return written


def _cells(
    row: Sequence[object] | Mapping[str, object], columns: Sequence[str], known: set[str]
) -> Sequence[object]:
    """Return *row* as its cells in column order."""
    if not isinstance(row, Mapping):
        return row
    unknown = [str(key) for key in row if key not in known]
    if unknown:
        raise ValueError(f"A row names columns the header does not: {', '.join(unknown)}")
    return [row.get(column, "") for column in columns]
