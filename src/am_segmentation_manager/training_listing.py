"""Which training runs a folder holds, and what each is called.

Reads the folder and writes nothing: the runs am-seg prepared, then each directory
AM-model-training wrote that belongs to none of them, a run started with it directly.
**The window and the command line both list through** :func:`list_runs`, so a run is
found, and called, the same in each.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .operations.training_run import RunRecord, find_run_directories, read_run_record
from .training_runs import SEARCH_DEPTH, find_training_runs, is_run_directory, written_in


@dataclass(frozen=True)
class ListedRun:
    """One run found under a folder.

    Attributes:
        home: The directory am-seg prepared, where its record, its choice of best
            epoch and its validations are; for a run am-seg did not start, the
            directory AM-model-training wrote.
        written: The directory AM-model-training wrote, or None before it has.
        name: What the run is called.
    """

    home: Path
    written: Path | None
    name: str


def prepared_name(directory: Path, record: RunRecord | None) -> str:
    """Return what a run am-seg prepared is called: what its record says, else its directory."""
    return record.name if record is not None else directory.name


def written_name(written: Path, root: Path | None) -> str:
    """Return what to call a run am-seg did not start: the folder it was written into.

    Its own directory's name is added where that folder holds several runs,
    so that no two are called the same.

    Args:
        root: The folder being listed, which the name is given from; None where there is none.
    """
    parent = written.parent
    if root is None or parent == root:
        return written.name
    try:
        name = parent.relative_to(root).as_posix()
    except ValueError:
        name = parent.name
    alone = sum(is_run_directory(child) for child in parent.iterdir()) == 1
    return name if alone else f"{name}/{written.name}"


def list_runs(root: Path, depth: int = SEARCH_DEPTH) -> list[ListedRun]:
    """Return every run at *root* or up to *depth* levels below it, those am-seg prepared first.

    A record that cannot be read leaves its run listed, under its directory's name.
    """
    prepared = find_run_directories(root, depth=depth)
    listed = []
    for home in prepared:
        try:
            record = read_run_record(home)
        except ValueError:
            record = None
        listed.append(ListedRun(home, written_in(home), prepared_name(home, record)))
    listed.extend(
        ListedRun(written, written, written_name(written, root))
        for written in find_training_runs(root, depth=depth)
        if written.parent not in prepared
    )
    return listed
