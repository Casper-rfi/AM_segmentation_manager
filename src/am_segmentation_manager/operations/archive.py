"""Keeping the labels an operation replaces, so replacing one is reversible.

A version goes to ``<project>/AM_archive/<path relative to the project>/`` under
its own stem with the time and a short digest, as a **whole label file** -- never
a difference against another version. A copy stays readable in any image viewer
and cannot be lost by a broken chain, and at 143 KB against a 211 GB collection
there is nothing to be saved by the alternative.

**What is archived is decided by where the pixels came from, not by what state
claims.** An untouched pseudo-label is machine output the model can make again;
one somebody has edited is human work whatever its group is declared to be. The
digest recorded when the machine wrote it is what tells those apart, so nothing
here asks the state at all. Where no such record exists the provenance is
unknown and the version is kept, that being the safe direction.

Archiving happens only where a project keeps a history, which is the same rule
recording follows: a collection that has not opted in gains no files.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..history import Event, HistoryLog, Recorder, events_for, read_log
from ..layout import ARCHIVE_DIRECTORY

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_DIRECTORY",
    "ArchivedVersion",
    "Archivist",
    "PrunePlan",
    "archive_root",
    "archived_versions",
    "keep_version",
    "apply_prune",
    "restore",
    "plan_prune",
    "version_directory",
    "versions_of",
]

#: How many characters of the digest go in an archived filename. Enough to tell
#: two versions written in the same second apart; the log holds the whole one.
NAME_DIGEST_LENGTH: int = 6

#: The kinds whose written version a later archive must never be pruned past.
#: Losing the copy taken before a machine wrote over a label removes the only
#: answer to "what did I have before the model replaced it".
PROTECTED_PREDECESSORS: frozenset[str] = frozenset({"label-converted", "merged"})


@dataclass(frozen=True)
class ArchivedVersion:
    """One kept version of a label.

    Attributes:
        path: The archived file.
        target: The label it is a version of, relative to the project root.
        event: The entry that recorded putting it there, or None where the log
            names no such version -- reported, never repaired.
        size: The file's size in bytes, or zero where the file is not here.
        present: Whether the file itself is here. A copy exported without
            ``--with-archive`` holds the record of a version without its
            pixels, and a list that hid those rows would be worse than one
            admitting the gap.
    """

    path: Path
    target: str
    event: Event | None
    size: int
    present: bool = True


def keep_version(
    project_root: Path, target: str, source: Path, digest: str
) -> str | None:
    """Copy the version now at *source* into *project_root*'s archive.

    The copying, with the deciding left to the caller. :class:`Archivist` asks
    whether a version being replaced has earned keeping; a savepoint asks
    nothing, because naming a state is itself the reason to keep what it names.

    Returns:
        The archived file's path relative to the project root, ready to go into
        an entry, or None where nothing could be copied.
    """
    destination = archived_path(project_root, target, digest, datetime.now(UTC))
    if destination.exists():
        # The same pixels already kept from the same second. Nothing is gained
        # by a second copy, and overwriting one is never right.
        return destination.relative_to(project_root).as_posix()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError as error:
        logger.warning("Could not archive %s: %s", source, error)
        return None
    logger.info("Archived %s as %s", target, destination.name)
    return destination.relative_to(project_root).as_posix()


def archive_root(project_root: Path) -> Path:
    """Return where *project_root* keeps replaced labels, existing or not."""
    return project_root / ARCHIVE_DIRECTORY


def archived_path(project_root: Path, target: str, digest: str, when: datetime) -> Path:
    """Return where a version of *target* belongs in the archive.

    Named so a plain directory listing sorts chronologically and two versions
    written inside one second stay apart. The suffix is kept, so the file opens
    in whatever an image viewer makes of the original.
    """
    relative = Path(target)
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    name = f"{relative.stem}__{stamp}__{digest[:NAME_DIGEST_LENGTH]}{relative.suffix}"
    return archive_root(project_root) / relative.parent / name


@dataclass
class Archivist:
    """Decides and performs archiving for one run.

    Holds each project's log for the length of the run, a batch touching a
    handful of projects over hundreds of labels.

    Attributes:
        recorder: Says who is acting, and locates the project holding a label.
            A project with no log is one this leaves alone.
    """

    recorder: Recorder
    _logs: dict[Path, HistoryLog] = field(default_factory=dict, repr=False)

    def keep(self, label_path: Path, previous_digest: str, empty: bool) -> str | None:
        """Copy the version now at *label_path* into the archive, if it earns it.

        Called before the file is overwritten, with the digest and emptiness of
        what is there now.

        Returns:
            The archived file's path relative to the project root, ready to go
            into the entry recording the replacement, or None where nothing was
            archived.
        """
        root = self.recorder.project_of(label_path)
        if root is None:
            return None
        try:
            target = label_path.resolve().relative_to(root).as_posix()
        except ValueError:
            return None

        keep, reason = self.decision(root, target, previous_digest, empty)
        if not keep:
            logger.debug("Not archiving %s: %s", target, reason)
            return None
        return keep_version(root, target, label_path, previous_digest)

    def decision(
        self, root: Path, target: str, previous_digest: str, empty: bool
    ) -> tuple[bool, str]:
        """Return whether the version now at *target* is worth keeping, and why.

        Nothing here reads the group's declared state. What matters is whether
        anybody has touched these pixels since a machine produced them, which
        the log answers and the state does not: an untouched pseudo-label and
        one somebody spent an hour proofreading are both ``annotated`` and both
        ``selected``.
        """
        if empty:
            return False, "an empty placeholder holds nothing to keep"
        recorded = self._machine_write(root, target)
        if recorded is not None and recorded == previous_digest:
            return False, "untouched machine output, which the model can make again"
        return True, "edited, or of unknown provenance"

    def _machine_write(self, root: Path, target: str) -> str | None:
        """Return the digest a machine last wrote to *target*, if one did.

        Only the most recent entry counts: a conversion followed by an edit
        leaves the conversion in the log, and treating that as evidence would
        discard the edit.
        """
        recorded = events_for(self._log(root), target)
        if not recorded:
            return None
        last = recorded[-1]
        if last.kind != "label-converted":
            return None
        digest = last.data.get("digest")
        return digest if isinstance(digest, str) else None

    def _log(self, root: Path) -> HistoryLog:
        """Return *root*'s log, read once for the length of this run."""
        if root not in self._logs:
            self._logs[root] = read_log(root)
        return self._logs[root]


# ---------------------------------------------------------------------------
# Reading the archive back
# ---------------------------------------------------------------------------

def archived_versions(project_root: Path) -> list[ArchivedVersion]:
    """Return every version *project_root* holds, oldest first per label.

    Pairs each file with the entry that recorded it. A file the log does not
    name keeps a ``None`` entry rather than being hidden: the two disagreeing
    is a finding, and one this reports rather than repairs.
    """
    root = archive_root(project_root)
    if not root.is_dir():
        return []

    log = read_log(project_root)
    by_path = {
        event.data["archived_as"]: event
        for event in log.events
        if isinstance(event.data.get("archived_as"), str)
    }

    versions = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        event = by_path.get(relative)
        versions.append(
            ArchivedVersion(
                path=path,
                target=event.target if event and event.target else _target_of(path, root),
                event=event,
                size=path.stat().st_size,
            )
        )
    return versions


def version_directory(project_root: Path, target: str) -> Path:
    """Return the one directory holding *target*'s archived versions.

    Every version of a label lands here, by :func:`archived_path`, so this is
    the whole of what has to be listed to find them -- as against walking the
    archive, which is linear in the collection on a path the proofreading view
    takes at every arrow key.
    """
    return archive_root(project_root) / Path(target).parent


def versions_of(
    project_root: Path, target: str, log: HistoryLog | None = None
) -> list[ArchivedVersion]:
    """Return the archived versions of one label present on disk, oldest first.

    Lists the label's own archive directory rather than the whole tree. Names
    there begin with the label's stem and a separator, which is what keeps two
    labels whose stems share a prefix apart.

    Args:
        log: The project's log, where the caller already holds it. Read here
            otherwise, this being the only thing that pairs a file with the
            entry recording it.
    """
    directory = version_directory(project_root, target)
    if not directory.is_dir():
        return []

    known = log if log is not None else read_log(project_root)
    by_path = {
        event.data["archived_as"]: event
        for event in events_for(known, target)
        if isinstance(event.data.get("archived_as"), str)
    }

    # Every label under one image directory archives into one directory, so
    # this listing sees the neighbours too. Names are matched before anything
    # is asked of the filesystem and the size comes from the directory entry
    # already in hand -- a stat for each of a directory's thousands of files is
    # what made stepping through frames slow.
    prefix = f"{Path(target).stem}__"
    found = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.startswith(prefix) or not entry.is_file():
                continue
            path = Path(entry.path)
            found.append(
                ArchivedVersion(
                    path=path,
                    target=target,
                    event=by_path.get(path.relative_to(project_root).as_posix()),
                    size=entry.stat().st_size,
                )
            )
    return sorted(found, key=lambda version: version.path.name)


def label_versions(
    project_root: Path, target: str, log: HistoryLog | None = None
) -> list[ArchivedVersion]:
    """Return every version of one label the project knows of, oldest first.

    The union of what is on disk and what the log says was kept, so a version
    whose file did not travel with a copy is listed as absent rather than
    silently missing. Ordered by when each was recorded, a version with no
    entry falling back to its filename, which sorts the same way.

    Args:
        log: The project's log, where the caller already holds it. Read once
            here otherwise, and shared with :func:`versions_of` rather than
            each reading it for itself.
    """
    known = log if log is not None else read_log(project_root)
    here = {
        version.path: version
        for version in versions_of(project_root, target, known)
    }

    found = list(here.values())
    for event in events_for(known, target):
        recorded = event.data.get("archived_as")
        if not isinstance(recorded, str) or not recorded:
            continue
        path = project_root / recorded
        if path in here:
            continue
        found.append(
            ArchivedVersion(
                path=path, target=target, event=event, size=0, present=False
            )
        )
    return sorted(found, key=_version_order)


def _version_order(version: ArchivedVersion) -> tuple[str, str]:
    """Return what orders a version: when it was recorded, then its name."""
    return ((version.event.at if version.event else ""), version.path.name)


def _target_of(path: Path, root: Path) -> str:
    """Return the label an archived file is a version of, read from its name.

    The name is the fallback, used only for a file the log does not account
    for. It cannot recover the original suffix reliably, so what it returns is
    a description rather than a key.
    """
    relative = path.relative_to(root)
    stem = relative.stem.rsplit("__", 2)[0]
    return (relative.parent / f"{stem}{relative.suffix}").as_posix()


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrunePlan:
    """What pruning would remove, having removed nothing.

    Attributes:
        remove: The versions that would go.
        kept: How many versions would stay.
        protected: Versions a rule selected but a protection kept.
    """

    remove: tuple[ArchivedVersion, ...] = ()
    kept: int = 0
    protected: tuple[ArchivedVersion, ...] = ()

    @property
    def bytes_freed(self) -> int:
        """Return how much disk removing them would return."""
        return sum(version.size for version in self.remove)


def plan_prune(project_root: Path, *, keep_last: int) -> PrunePlan:
    """Return which versions *keep_last* would leave behind.

    Three are never offered up, whatever the rule asks for: a label's oldest
    kept version, which is the furthest back anybody can go; the version taken
    before a machine wrote over the label, which is the only answer to what was
    there before the model; and one taken before a merge took somebody else's
    work. A version the log does not account for is protected too -- what it is
    cannot be established, so it is not this command's to remove.
    """
    if keep_last < 0:
        raise ValueError("keep_last cannot be negative")

    by_target: dict[str, list[ArchivedVersion]] = {}
    for version in archived_versions(project_root):
        by_target.setdefault(version.target, []).append(version)

    remove, protected, kept = [], [], 0
    for versions in by_target.values():
        surplus = versions[: max(0, len(versions) - keep_last)]
        kept += len(versions) - len(surplus)
        for index, version in enumerate(surplus):
            reason = _protection(version, is_oldest=index == 0)
            if reason is None:
                remove.append(version)
            else:
                protected.append(version)
                kept += 1
    return PrunePlan(tuple(remove), kept, tuple(protected))


def _protection(version: ArchivedVersion, *, is_oldest: bool) -> str | None:
    """Return why *version* may not be pruned, or None if it may."""
    if is_oldest:
        return "the oldest version kept for this label"
    if version.event is None:
        return "no entry accounts for it"
    if version.event.kind in PROTECTED_PREDECESSORS:
        return f"taken before a {version.event.kind}"
    return None


def apply_prune(plan: PrunePlan) -> int:
    """Remove the versions *plan* names, returning how many went.

    The only place in this package that deletes a label file, and it does so
    only for versions a person asked for by name and that no protection covers.
    The log is untouched: pruning costs the ability to restore those pixels,
    never the account of what happened.
    """
    removed = 0
    for version in plan.remove:
        try:
            version.path.unlink()
            removed += 1
        except OSError as error:
            logger.warning("Could not remove %s: %s", version.path, error)
    logger.info("Removed %s archived version(s)", removed)
    return removed


def restore(
    version: ArchivedVersion,
    destination: Path,
    *,
    recorder: Recorder,
    corrected: bool = False,
    allow_corrected: bool = False,
) -> Path:
    """Put an archived version back, through the ordinary save.

    **Not a new way to write a label.** It builds the same item an editor would
    and hands it to :func:`~.label_save.save_labels`, so the batch validation,
    the refusal of a file that changed since it was read, and the protection of
    proofread work all apply unchanged -- and the version being replaced is
    itself archived on the way past. Only what is recorded differs.

    Raises:
        RuntimeError: If the archived file cannot be read.
        StaleLabels: If *destination* changed since it was read.
        PermissionError: If it is proofread work and *allow_corrected* is not
            set.
    """
    from ..io.atomic import fingerprint_of
    from ..io.image import read_array
    from .label_io import label_fields
    from .label_save import LabelToSave, WriteProvenance, save_labels

    pixels = read_array(version.path)
    digest = label_fields(pixels)["digest"]
    save_labels(
        [
            LabelToSave(
                destination=destination,
                label=pixels,
                created=not destination.exists(),
                fingerprint=fingerprint_of(destination),
                corrected=corrected,
                provenance=WriteProvenance(
                    "label-restored", {"restored_from": str(digest)}
                ),
            )
        ],
        allow_corrected=allow_corrected,
        recorder=recorder,
    )
    logger.info("Restored %s from %s", destination, version.path.name)
    return destination


def total_size(versions: Iterable[ArchivedVersion]) -> int:
    """Return how many bytes a set of archived versions occupies."""
    return sum(version.size for version in versions)
