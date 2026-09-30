"""Copying a selection of images and labels out into a folder of its own.

The images a selection matches, each one's label and the files describing their
projects are written under the same relative paths, so the copy is a smaller
collection of the same shape that ``am-seg`` scans and proofreading continues in.
A finished copy is given the fork point a merge back needs. **Nothing is
overwritten**: every destination is classified before a byte is written, one
holding the same file is skipped so an interrupted export resumes, and one holding
something else refuses the run. Each file lands by rename.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..families import AM_FAMILIES_FILENAME
from ..history import AM_HISTORY_FILENAME
from ..io.atomic import fingerprint_of, write_atomically
from ..io.long_paths import refuse_unreachable
from ..layout import ARCHIVE_DIRECTORY
from ..metadata import AM_METADATA_FILENAME, EXPERIMENT_FILENAME, PROTOCOL_FILENAME
from ..models import ImageRecord, Project, projects_holding
from ..progress import Cancellation
from ..state import AM_STATE_FILENAME
from .csv import find_matching_label
from .export_record import ExportRecord, build_export_record, write_export_record

logger = logging.getLogger(__name__)

__all__ = [
    "PROJECT_FILES",
    "CopyItem",
    "CopyPlan",
    "CopyReport",
    "DestinationInsideSource",
    "copy_export",
    "plan_copy",
    "projects_of",
    "record_fork_point",
]

#: The files describing a project that a copy takes along, tried in each
#: project directory. ``AM_metadata.yaml`` and ``experiment.yaml`` are what make
#: the copy indexable and carry its metadata; ``protocol.yaml`` records how the
#: milling was set up; ``AM_families.yaml`` which label each image folder
#: shares, where the folder names do not say. A project missing any of them is
#: not a defect: the older datasets predate ``experiment.yaml``, and only this
#: package writes the first and last.
#:
#: ``AM_history.jsonl`` always travels, and is the one that must: a merge back
#: cannot establish a common ancestor without it, and at kilobytes there is
#: nothing to weigh against that. The **archive** does not, unless asked for --
#: see :func:`plan_copy` -- a subset export that dragged every kept version
#: along would balloon, and a deposition would carry intermediates.
PROJECT_FILES: tuple[str, ...] = (
    AM_METADATA_FILENAME,
    EXPERIMENT_FILENAME,
    PROTOCOL_FILENAME,
    AM_FAMILIES_FILENAME,
    AM_HISTORY_FILENAME,
)

#: How far a copy's modification time may lie from its source's and still be
#: the same file, once its contents say so: FAT keeps times to 2 s, exFAT to
#: 10 ms, so an exact match fails on the USB drive an export is often written to.
TIME_TOLERANCE_NS = 2_000_000_000

#: What a planned item would do at its destination.
COPY = "copy"
PRESENT = "present"
CONFLICT = "conflict"


class DestinationInsideSource(ValueError):
    """Raised when the destination sits inside the folder being copied from.

    Not a case worth handling: the copy would be walking into its own output,
    and the resulting tree would be indexed as part of the collection.
    """


@dataclass(frozen=True)
class CopyItem:
    """One file the export would write, and what writing it would do.

    Attributes:
        source: The file as it is now.
        destination: Where it would land, under the same relative path.
        size: The source's size in bytes, which is what the destination costs.
        kind: ``image``, ``label``, ``metadata`` or ``state``. Reported rather
            than acted on: what is copied is decided when the plan is built.
        outcome: :data:`COPY`, :data:`PRESENT` -- the same file is already
            there, so this run would skip it -- or :data:`CONFLICT`, something
            else is there under that name.
    """

    source: Path
    destination: Path
    size: int
    kind: str
    outcome: str


@dataclass(frozen=True)
class CopyPlan:
    """Everything an export would write, resolved before any of it runs.

    Attributes:
        items: Every file considered, in the order they would be copied.
        destination: The directory the copy is written into.
        free_space: Bytes free on the destination's filesystem when the plan
            was made.
    """

    items: tuple[CopyItem, ...] = ()
    destination: Path = Path()
    free_space: int = 0

    @property
    def to_copy(self) -> tuple[CopyItem, ...]:
        """Return the items this run would actually write."""
        return tuple(item for item in self.items if item.outcome == COPY)

    @property
    def already_there(self) -> tuple[CopyItem, ...]:
        """Return the items a previous run of this export already wrote."""
        return tuple(item for item in self.items if item.outcome == PRESENT)

    @property
    def conflicts(self) -> tuple[CopyItem, ...]:
        """Return the destinations holding something other than their source."""
        return tuple(item for item in self.items if item.outcome == CONFLICT)

    @property
    def bytes_to_copy(self) -> int:
        """Return how much this run would write."""
        return sum(item.size for item in self.to_copy)

    @property
    def short_by(self) -> int:
        """Return the bytes by which the destination is too full, or zero."""
        return max(0, self.bytes_to_copy - self.free_space)

    @property
    def is_refused(self) -> bool:
        """Return whether the run must not start.

        Either something would be overwritten or there is not room for it.
        Both are answered for the whole set before anything is written, which
        is the point of resolving the plan first.
        """
        return bool(self.conflicts) or bool(self.short_by)


@dataclass(frozen=True)
class CopyReport:
    """What an export actually did.

    Attributes:
        copied: The files written.
        skipped: The files a previous run had already written.
        stopped: Whether the run was asked to stop before it finished.
    """

    copied: tuple[CopyItem, ...] = ()
    skipped: tuple[CopyItem, ...] = ()
    stopped: bool = False

    @property
    def bytes_copied(self) -> int:
        """Return how much was written."""
        return sum(item.size for item in self.copied)


def plan_copy(
    records: Iterable[ImageRecord],
    destination: Path,
    start_path: Path,
    *,
    projects: Sequence[Project] = (),
    with_archive: bool = False,
) -> CopyPlan:
    """Return everything copying *records* into *destination* would write.

    The hierarchy is mirrored: each file lands under the path it has relative
    to *start_path*, so the copy is the same collection with fewer images in
    it. Alongside the images and their labels go each contributing stage's
    ``AM_state.yaml`` and each contributing project's :data:`PROJECT_FILES`.

    Nothing is read but directory entries: a plan is a question about names and
    sizes, and answering it has to be fast enough to show before the user
    commits to an hour of copying.

    Args:
        records: The images to copy, already filtered.
        destination: The directory to copy into. Created by :func:`copy_export`.
        start_path: The scan root the mirrored paths are relative to.
        projects: The scanned projects, used to find the files describing each.
            An empty sequence copies images and labels alone.
        with_archive: Take each project's kept label versions too. Off by
            default: the log is the record and travels always, while the
            archive is a cache of recoverable pixels and may be partial. Turn
            it on when the copy is for comparing against what a label used to
            hold.

    Raises:
        DestinationInsideSource: If *destination* is inside *start_path*, or
            *start_path* inside *destination*.
    """
    destination = Path(destination).resolve()
    start_path = Path(start_path).resolve()
    _refuse_overlap(destination, start_path)

    sources: list[tuple[Path, str]] = []
    stage_directories: list[Path] = []
    images: list[Path] = []

    for record in records:
        sources.append((record.image_path, "image"))
        images.append(record.image_path)
        label = find_matching_label(record)
        if label is not None:
            sources.append((label, "label"))
        if record.stage_path is not None:
            stage_directories.append(record.stage_path)

    for directory in _unique(stage_directories):
        sources.append((directory / AM_STATE_FILENAME, "state"))
    for directory in projects_of(images, projects):
        sources.extend((directory / name, "metadata") for name in PROJECT_FILES)
        if with_archive:
            sources.extend(
                (path, "archive") for path in _archived_files(directory)
            )

    items = [
        _plan_item(source, kind, destination, start_path)
        for source, kind in _unique_sources(sources)
    ]
    plan = CopyPlan(
        items=tuple(item for item in items if item is not None),
        destination=destination,
        free_space=_free_space(destination),
    )
    logger.debug(
        "Copy plan: %d file(s) to write, %d already there, %d in the way",
        len(plan.to_copy), len(plan.already_there), len(plan.conflicts),
    )
    return plan


def copy_export(
    plan: CopyPlan,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancellation: Cancellation | None = None,
) -> CopyReport:
    """Write everything *plan* says to write.

    Each file is copied through :func:`~..io.atomic.write_atomically`, so an
    interrupted run leaves whole files rather than truncated ones, and keeps
    the source's modification time -- which is what lets a later run recognise
    what it already wrote and carry on from there.

    Stopping is reported rather than raised: a run cancelled after ten of a
    hundred files did copy ten, and they are kept.

    Raises:
        ValueError: If the plan is refused. What it was refused for is the
            caller's to report; running it anyway is not an option.
        PathTooLong: If Windows cannot reach a copy here. Checked before
            anything is copied.
    """
    if plan.is_refused:
        raise ValueError("This export was refused; nothing was copied.")
    refuse_unreachable(item.destination for item in plan.to_copy)

    copied: list[CopyItem] = []
    stopped = False
    total = len(plan.to_copy)
    for done, item in enumerate(plan.to_copy):
        if cancellation is not None and cancellation.requested:
            stopped = True
            break
        _copy_file(item.source, item.destination)
        copied.append(item)
        if on_progress is not None:
            on_progress(done + 1, total)

    # Debug rather than info: the caller reports what happened, and two
    # summaries of one run read as two runs.
    logger.debug(
        "Copied %d file(s) into %s%s",
        len(copied), plan.destination, " (stopped early)" if stopped else "",
    )
    return CopyReport(
        copied=tuple(copied), skipped=plan.already_there, stopped=stopped
    )


def record_fork_point(
    destination: Path,
    records: Iterable[ImageRecord],
    projects: Sequence[Project],
    start_path: Path,
    *,
    stopped: bool = False,
    scope: str = "",
    with_archive: bool = False,
    user: str | None = None,
) -> ExportRecord | None:
    """Record where a finished copy came from, so it can be merged back.

    The last step of an export rather than a separate act: a copy without one
    cannot be merged, and every caller that writes a copy owes it.

    **Written after the copy, and only for a whole one.** A fork point names
    the entry each project's log ended at, which describes the tree the copy
    was meant to be; a stopped run has some of that tree and would be compared
    against an ancestor half of it does not hold.

    Args:
        destination: The copy's root, where the record is written.
        records: The images copied, which say which projects the copy holds.
        projects: The scanned projects those images came from.
        start_path: The scan root, recorded as context for a person.
        stopped: Whether the copy was cut short. Answered here rather than by
            each caller, so that every export applies the rule above.
        scope: What the export was narrowed to, in the caller's own words.
        with_archive: Whether kept label versions came too.
        user: A name to attribute the export to, where somebody stated one.

    Returns:
        The record written, or None where the copy was stopped and none was.
    """
    if stopped:
        logger.warning(
            "The copy was stopped, so no fork point was written; a merge back "
            "needs one. Running the export again finishes it."
        )
        return None

    copied = set(projects_of([record.image_path for record in records], projects))
    record = build_export_record(
        [project for project in projects if project.path in copied],
        start_path,
        scope=scope,
        with_archive=with_archive,
        user=user,
    )
    path = write_export_record(destination, record)
    logger.info("Wrote the fork point to %s", path)
    return record


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _archived_files(project_directory: Path) -> list[Path]:
    """Return every kept label version under *project_directory*.

    Listed rather than resolved from the log, so a version the history does not
    account for still travels: a copy that silently dropped it would make the
    two trees disagree about what exists.
    """
    root = project_directory / ARCHIVE_DIRECTORY
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _refuse_overlap(destination: Path, start_path: Path) -> None:
    """Refuse a destination that sits inside the source tree, or contains it."""
    if destination == start_path or _is_within(destination, start_path):
        raise DestinationInsideSource(
            f"{destination} is inside the folder being copied from"
        )
    if _is_within(start_path, destination):
        raise DestinationInsideSource(
            f"{destination} contains the folder being copied from"
        )


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether *path* is *directory* or sits below it."""
    return directory == path or directory in path.parents


def projects_of(images: Sequence[Path], projects: Sequence[Project]) -> list[Path]:
    """Return the directories of the projects *images* were found in.

    Asked of the images alone -- a label and a state file sit inside the same
    project as the image that named them.
    """
    return [project.path for project in projects_holding(images, projects)]


def _unique(paths: Iterable[Path]) -> list[Path]:
    """Return *paths* without repeats, in the order first seen."""
    seen: set[Path] = set()
    kept: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            kept.append(path)
    return kept


def _unique_sources(
    sources: Iterable[tuple[Path, str]]
) -> list[tuple[Path, str]]:
    """Return the sources without repeats, keeping the first kind seen.

    Several images of one scene share a label, and every image of a stage
    shares its state file, so the same file is named many times over.
    """
    seen: set[Path] = set()
    kept: list[tuple[Path, str]] = []
    for source, kind in sources:
        if source not in seen:
            seen.add(source)
            kept.append((source, kind))
    return kept


def _plan_item(
    source: Path, kind: str, destination: Path, start_path: Path
) -> CopyItem | None:
    """Return what copying *source* would do, or None if it is not there.

    A missing source is not an error: only some projects have a
    ``protocol.yaml``, only a migrated stage has an ``AM_state.yaml``, and a
    record can name a label that has since been removed.
    """
    fingerprint = fingerprint_of(source)
    if fingerprint is None:
        return None

    try:
        relative = source.resolve().relative_to(start_path)
    except ValueError:
        # Outside the scan root entirely. Nothing sensible to mirror it under,
        # and inventing a path would put it somewhere the copy cannot be read.
        logger.warning("Not copying %s: it is outside %s", source, start_path)
        return None

    target = destination / relative
    existing = fingerprint_of(target)
    if existing is None:
        outcome = COPY
    else:
        outcome = PRESENT if _same_copy(source, target) else CONFLICT
    return CopyItem(
        source=source, destination=target, size=fingerprint.size, kind=kind,
        outcome=outcome,
    )


def _same_copy(source: Path, target: Path) -> bool:
    """Whether *target* is a finished copy of *source*.

    A copy is another file, so its size and time are all it can share with the
    source without being read; :func:`_copy_file` keeps the time. Where the
    destination rounded that time, the contents decide.
    """
    copied, original = fingerprint_of(target), fingerprint_of(source)
    if copied is None or original is None or copied.size != original.size:
        return False
    if copied.modified_ns == original.modified_ns:
        return True
    if abs(copied.modified_ns - original.modified_ns) > TIME_TOLERANCE_NS:
        return False
    return _same_contents(source, target)


def _same_contents(first: Path, second: Path, chunk: int = 1 << 20) -> bool:
    """Whether two files hold the same bytes, read afresh every time.

    Not :func:`filecmp.cmp`, which remembers an answer by size and time: the
    two things a drive that rounds times can leave unchanged by a rewrite.
    """
    with first.open("rb") as one, second.open("rb") as other:
        while True:
            block = one.read(chunk)
            if block != other.read(chunk):
                return False
            if not block:
                return True


def _free_space(destination: Path) -> int:
    """Return the bytes free where *destination* would be written.

    Asked of the nearest directory that exists, since the destination itself
    is usually about to be created.
    """
    existing = destination
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        return shutil.disk_usage(existing).free
    except OSError as error:  # pragma: no cover - depends on the filesystem
        logger.warning("Could not measure free space at %s: %s", existing, error)
        return 0


def _copy_file(source: Path, destination: Path) -> None:
    """Copy one file into place whole, keeping its modification time."""
    def write(temporary: Path) -> None:
        shutil.copyfile(source, temporary)
        status = source.stat()
        os.utime(temporary, ns=(status.st_atime_ns, status.st_mtime_ns))

    write_atomically(destination, write)
