"""Naming a state of the data, so a round of work can be compared and undone.

Proofreading produces a gradient: hundreds of small saves, none of them a point
anybody would name. The structure over the top of it is the round --- proofread
and curate, train, predict, convert, proofread again --- and **a savepoint marks
one**. That is what makes it worth having: rounds are rare, so what a savepoint
pins costs little, and they are the boundaries a person already thinks in.

**A savepoint is a set of per-project entries sharing one id.** There is no such
thing as "the collection": the scanner can be pointed at a subset, data can be
split off and added. So a collection-wide savepoint is one ``checkpoint`` entry
in each project in scope, and coverage is answered by looking rather than by a
list somebody stored --- a stored list is wrong the moment a project joins.

**It promises nothing about pixels; it reports what it can reach.** The state it
names is what the *record* knew, and a version it names is recoverable only
while the archive holds it. Which of them are still reachable is a question
asked of the log, cheaply, whenever somebody wants the answer --- at creation,
in a listing, and before anything acts on one.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..history import Event, HistoryLog, Recorder, history_file_path, read_log
from ..models import Project

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVED",
    "AT_RISK",
    "CURRENT",
    "LOST",
    "ProjectSavepoint",
    "Savepoint",
    "SavepointPlan",
    "SavepointRefused",
    "SavepointReport",
    "TargetState",
    "apply_savepoint",
    "close_gap",
    "find_savepoint",
    "list_savepoints",
    "not_recording",
    "open_round",
    "plan_savepoint",
    "reachability",
    "reached_versions",
]

#: A kept version holds these pixels. Recoverable now.
ARCHIVED: Final[str] = "archived"
#: The label still holds them, and replacing it would keep them first.
CURRENT: Final[str] = "current"
#: The label still holds them, but they are untouched machine output: the next
#: conversion replaces it and archives nothing, and the state is gone.
AT_RISK: Final[str] = "at-risk"
#: Nothing here holds them. The savepoint names a state nobody can return to.
LOST: Final[str] = "lost"


class SavepointRefused(RuntimeError):
    """Raised when a savepoint must not be written at all."""


@dataclass(frozen=True)
class TargetState:
    """What one label held at a savepoint, and whether that is still reachable.

    Attributes:
        target: The group's key, relative to the project root.
        digest: What the label held, as the record has it.
        where: One of :data:`ARCHIVED`, :data:`CURRENT`, :data:`AT_RISK` or
            :data:`LOST`.
        archived_as: The kept version holding it, where one does.
    """

    target: str
    digest: str
    where: str
    archived_as: str | None = None


@dataclass(frozen=True)
class ProjectSavepoint:
    """One project's part of a savepoint.

    Attributes:
        project: The project.
        entry: Its checkpoint entry, where it has one.
        targets: What each label held at that point, and whether it is reachable.
    """

    project: Project
    entry: Event | None = None
    targets: tuple[TargetState, ...] = ()

    @property
    def reachable(self) -> int:
        """Return how many of its labels can still be got back."""
        return sum(1 for state in self.targets if state.where != LOST)

    @property
    def at_risk(self) -> tuple[TargetState, ...]:
        """Return the labels a further conversion would make unreachable."""
        return tuple(state for state in self.targets if state.where == AT_RISK)


@dataclass(frozen=True)
class Savepoint:
    """A savepoint as it stands, across whatever is in scope.

    Attributes:
        id: The id its entries share.
        title: What it was called. The first title found, titles being a
            convenience rather than an identity.
        description: Why it was taken.
        at: When the earliest of its entries was written.
        by: Who took it.
        projects: One per project in scope that carries it.
        covered_at_creation: How many projects it was written across, as a
            historical note. **Never a denominator** -- a project joining later
            makes it smaller than what is found, which is not a fault.
    """

    id: str
    title: str = ""
    description: str = ""
    at: str = ""
    by: str = ""
    projects: tuple[ProjectSavepoint, ...] = ()
    covered_at_creation: int = 0

    @property
    def found(self) -> int:
        """Return how many projects carrying it are in scope."""
        return len(self.projects)

    @property
    def targets(self) -> int:
        """Return how many labels it names across those projects."""
        return sum(len(project.targets) for project in self.projects)

    @property
    def reachable(self) -> int:
        """Return how many of those can still be got back."""
        return sum(project.reachable for project in self.projects)

    @property
    def at_risk(self) -> tuple[tuple[Project, TargetState], ...]:
        """Return every label a further conversion would make unreachable."""
        return tuple(
            (project.project, state)
            for project in self.projects
            for state in project.at_risk
        )

    def describe(self) -> str:
        """Return the one line a listing shows."""
        note = f"{self.found} project(s) here"
        if self.covered_at_creation and self.covered_at_creation != self.found:
            note += f", {self.covered_at_creation} at creation"
        return (
            f"{self.title or self.id}  {self.at[:10]}  {self.by}  "
            f"{note}  {self.reachable}/{self.targets} labels reachable"
        )


@dataclass(frozen=True)
class SavepointPlan:
    """What taking a savepoint would write, having written nothing.

    Attributes:
        id: The id the entries would share.
        title: What it would be called.
        description: Why.
        projects: The projects that would gain a checkpoint.
        joining: Whether this adds to a savepoint that already exists.
        states: What each project's labels hold now, keyed by project key.
    """

    id: str
    title: str
    description: str = ""
    projects: tuple[Project, ...] = ()
    joining: bool = False
    states: dict[str, tuple[TargetState, ...]] = field(default_factory=dict)

    @property
    def at_risk(self) -> tuple[tuple[Project, TargetState], ...]:
        """Return what would be unreachable after another conversion."""
        return tuple(
            (project, state)
            for project in self.projects
            for state in self.states.get(project.key, ())
            if state.where == AT_RISK
        )

    def describe(self) -> list[str]:
        """Return one line per project, and what it would pin."""
        lines = []
        for project in self.projects:
            states = self.states.get(project.key, ())
            risky = sum(1 for state in states if state.where == AT_RISK)
            line = f"{project.key}: {len(states)} label(s)"
            if risky:
                line += f", {risky} recoverable only until the next conversion"
            lines.append(line)
        return lines


@dataclass(frozen=True)
class SavepointReport:
    """What taking a savepoint actually wrote."""

    id: str
    entries_written: int = 0
    versions_kept: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# What a savepoint names
# ---------------------------------------------------------------------------

def digests_before(log: HistoryLog, entry: Event | None) -> dict[str, str]:
    """Return what each label held before *entry*, as the record has it.

    Cut by **position** rather than by time: entries written inside one
    millisecond share a timestamp, and a savepoint taken in the middle of them
    would otherwise include or exclude the lot. ``read_log`` orders the file
    once and stably, which is what makes a position mean something.

    Args:
        entry: The checkpoint to read up to, or None for the whole log.
    """
    events = list(log.events)
    if entry is not None:
        for index, candidate in enumerate(events):
            if candidate.id == entry.id:
                events = events[:index]
                break

    digests: dict[str, str] = {}
    for event in events:
        digest = event.data.get("digest")
        if event.target and isinstance(digest, str) and digest:
            digests[event.target] = digest
    return digests


def reachability(
    project_root: Path, log: HistoryLog, entry: Event | None = None
) -> tuple[TargetState, ...]:
    """Return what each label held at *entry*, and whether it can be got back.

    Asked entirely of the log, so it costs no image reads and can be run over a
    collection whenever somebody wants the answer. The consequence is that it
    describes what the record knows: a label edited by the segmentation software
    without recording it reads as unchanged here, which is the same limit every
    other reader of the log has.
    """
    wanted = digests_before(log, entry)
    kept = _kept_by_digest(log)
    last = _last_recorded(log)
    listing: dict[Path, set[str]] = {}

    states = []
    for target, digest in sorted(wanted.items()):
        recorded = kept.get((target, digest))
        archived = (
            recorded
            if recorded is not None and (project_root / recorded).is_file()
            else _archived_file(project_root, target, digest, listing)
        )

        # **Asked of the label first.** A version can be both on the label and
        # in the archive, and answering "archived" then would have a revert put
        # back what is already there -- a write that keeps a copy of nothing.
        latest = last.get(target)
        if latest is not None and latest.data.get("digest") == digest:
            # Whether replacing it would keep it is the archive's own rule:
            # untouched machine output is regenerable and is skipped, so a
            # further conversion would take this state with it.
            at_risk = latest.kind == "label-converted" and archived is None
            states.append(
                TargetState(target, digest, AT_RISK if at_risk else CURRENT, archived)
            )
            continue

        if archived is not None:
            states.append(TargetState(target, digest, ARCHIVED, archived))
            continue
        states.append(TargetState(target, digest, LOST))
    return tuple(states)


def _archived_file(
    project_root: Path, target: str, digest: str, listing: dict[Path, set[str]]
) -> str | None:
    """Return the archived file holding *digest* for *target*, if one is here.

    Asked of the archive rather than of the log, because a version can be kept
    without an entry naming it: closing a savepoint's gap copies a file in and
    records nothing, the copy being a consequence of the savepoint rather than
    a change to the label. Read from the target towards the filename, never the
    other way, so nothing has to be inferred from a name.
    """
    from .archive import NAME_DIGEST_LENGTH, version_directory

    directory = version_directory(project_root, target)
    if directory not in listing:
        try:
            listing[directory] = {
                entry.name for entry in directory.iterdir() if entry.is_file()
            }
        except OSError:
            listing[directory] = set()

    stem, suffix = Path(target).stem, Path(target).suffix
    tail = f"__{digest[:NAME_DIGEST_LENGTH]}{suffix}"
    for name in sorted(listing[directory]):
        if name.startswith(f"{stem}__") and name.endswith(tail):
            return (directory / name).relative_to(project_root).as_posix()
    return None


def _kept_by_digest(log: HistoryLog) -> dict[tuple[str, str], str]:
    """Return where each replaced version went, keyed by target and digest.

    An entry names the version it replaced with ``parent_digest`` and where that
    went with ``archived_as``, so the pair is an exact lookup -- as against
    matching the six digest characters in an archived filename.
    """
    kept: dict[tuple[str, str], str] = {}
    for event in log.events:
        parent = event.data.get("parent_digest")
        archived = event.data.get("archived_as")
        if event.target and isinstance(parent, str) and isinstance(archived, str):
            if parent and archived:
                kept[(event.target, parent)] = archived
    return kept


def _last_recorded(log: HistoryLog) -> dict[str, Event]:
    """Return the last entry about each target."""
    latest: dict[str, Event] = {}
    for event in log.events:
        if event.target:
            latest[event.target] = event
    return latest


# ---------------------------------------------------------------------------
# Reading savepoints back
# ---------------------------------------------------------------------------

def _checkpoints(log: HistoryLog) -> list[Event]:
    """Return the checkpoint entries in *log*, oldest first."""
    return [event for event in log.events if event.kind == "checkpoint"]


def list_savepoints(projects: Iterable[Project]) -> list[Savepoint]:
    """Return every savepoint carried by *projects*, newest first.

    Gathered by looking rather than resolved from anything: a savepoint *is* the
    set of entries sharing its id, so what is in scope is the honest answer to
    what it covers.
    """
    found: dict[str, list[tuple[Project, Event, HistoryLog]]] = {}
    for project in projects:
        log = read_log(project.path)
        for entry in _checkpoints(log):
            identity = entry.data.get("savepoint")
            if isinstance(identity, str) and identity:
                found.setdefault(identity, []).append((project, entry, log))

    savepoints = []
    for identity, carried in found.items():
        first = min(carried, key=lambda item: item[1].at)[1]
        savepoints.append(
            Savepoint(
                id=identity,
                title=str(first.data.get("title") or ""),
                description=str(first.data.get("description") or ""),
                at=first.at,
                by=first.by,
                covered_at_creation=_count(first.data.get("covered_at_creation")),
                projects=tuple(
                    ProjectSavepoint(
                        project=project,
                        entry=entry,
                        targets=reachability(project.path, log, entry),
                    )
                    for project, entry, log in sorted(
                        carried, key=lambda item: item[0].key
                    )
                ),
            )
        )
    return sorted(savepoints, key=lambda point: point.at, reverse=True)


def find_savepoint(projects: Iterable[Project], wanted: str) -> Savepoint:
    """Return the savepoint *wanted* names, by id or by title.

    Raises:
        SavepointRefused: If nothing matches, or a title matches more than one.
            A title is a convenience; an ambiguous one is refused rather than
            picked between, since the two name different states.
    """
    savepoints = list_savepoints(projects)
    by_id = [point for point in savepoints if point.id == wanted]
    if by_id:
        return by_id[0]

    by_title = [point for point in savepoints if point.title == wanted]
    if len(by_title) == 1:
        return by_title[0]
    if len(by_title) > 1:
        raise SavepointRefused(
            f"{wanted!r} names {len(by_title)} savepoints "
            f"({', '.join(point.id for point in by_title)}). Say which by id."
        )
    known = ", ".join(point.title or point.id for point in savepoints[:5])
    raise SavepointRefused(
        f"No savepoint here is called {wanted!r}."
        + (f" There is {known}." if known else " None has been taken.")
    )


def _count(value: object) -> int:
    """Return *value* as a whole number of projects, or zero."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# ---------------------------------------------------------------------------
# Taking one
# ---------------------------------------------------------------------------

def plan_savepoint(
    projects: Sequence[Project],
    title: str,
    *,
    description: str = "",
    joining: Savepoint | None = None,
) -> SavepointPlan:
    """Return what taking a savepoint over *projects* would write.

    Only projects keeping a log take part: a savepoint is an entry in a log, and
    one that has never recorded anything has no state to name.

    Args:
        joining: An existing savepoint to add these projects to, rather than
            starting a new one. Its id and title are used.

    Raises:
        SavepointRefused: If no project in scope keeps a log, or *title* is
            blank on a new savepoint.
    """
    recording = [
        project for project in projects if history_file_path(project.path).is_file()
    ]
    if not recording:
        raise SavepointRefused(
            "No project here keeps a history, so there is no state to name. "
            "Run init-history first."
        )
    if joining is None and not title.strip():
        raise SavepointRefused("A savepoint needs a title: it is what you will use.")

    already = {carried.project.key for carried in (joining.projects if joining else ())}
    wanted = [project for project in recording if project.key not in already]

    states = {}
    for project in wanted:
        log = read_log(project.path)
        states[project.key] = reachability(project.path, log)

    return SavepointPlan(
        id=joining.id if joining else uuid.uuid4().hex[:12],
        title=joining.title if joining else title.strip(),
        description=description.strip(),
        projects=tuple(wanted),
        joining=joining is not None,
        states=states,
    )


def apply_savepoint(
    plan: SavepointPlan, *, recorder: Recorder, close_the_gap: bool = False
) -> SavepointReport:
    """Write *plan*, one checkpoint entry per project.

    Args:
        close_the_gap: Copy into the archive the versions that are on disk but
            that the ordinary rule would not keep --- untouched machine output,
            which the next conversion replaces without archiving. Done here,
            once, where the cost is visible, rather than by a standing rule
            checked at every save for years.
    """
    kept: list[str] = []
    if close_the_gap:
        kept = close_gap(plan)

    written = 0
    for project in plan.projects:
        fields: dict[str, object] = {
            "savepoint": plan.id,
            "title": plan.title,
            "covered_at_creation": len(plan.projects),
        }
        if plan.description:
            fields["description"] = plan.description
        if recorder.record_project(project.path, "checkpoint", **fields) is not None:
            written += 1

    logger.info(
        "Savepoint %s (%r) written across %s project(s)", plan.id, plan.title, written
    )
    return SavepointReport(id=plan.id, entries_written=written, versions_kept=tuple(kept))


def close_gap(plan: SavepointPlan) -> list[str]:
    """Keep the versions a savepoint names that nothing else would keep.

    The archive skips machine output nobody has edited, because the model can
    make it again --- which is right until a savepoint names it, at which point
    the state has to survive the next conversion. Naming it is the reason to
    keep it, so no decision is asked for here.

    Returns:
        The archived paths, relative to their projects.
    """
    from .archive import keep_version

    kept = []
    for project, state in plan.at_risk:
        source = project.path / state.target
        if not source.is_file():
            logger.warning("%s is named by the savepoint but is not here", state.target)
            continue
        archived = keep_version(project.path, state.target, source, state.digest)
        if archived is not None:
            kept.append(f"{project.key}/{archived}")
    return kept


def reached_versions(savepoints: Iterable[Savepoint]) -> dict[Path, set[str]]:
    """Return the archived versions each project's savepoints depend on.

    What pruning must not remove, keyed by project root. Computed here and
    handed to :func:`~.archive.plan_prune` rather than looked up there, so the
    archive stays a thing that keeps and removes files and does not have to know
    what a savepoint is.
    """
    needed: dict[Path, set[str]] = {}
    for point in savepoints:
        for carried in point.projects:
            wanted = needed.setdefault(carried.project.path.resolve(), set())
            for state in carried.targets:
                if state.archived_as:
                    wanted.add(state.archived_as)
    return needed


def not_recording(projects: Iterable[Project]) -> tuple[Project, ...]:
    """Return the projects here that keep no log.

    A project added after a collection opted in has no history until somebody
    runs ``init-history`` on it, and until then it records nothing, joins no
    savepoint and carries no ancestor for a merge -- silently, since recording
    where a log exists is the rule that lets this be opted into at all. The
    silence is the point and also the danger, so the one command that asks what
    the record covers says which projects it does not.
    """
    return tuple(
        project
        for project in projects
        if not history_file_path(project.path).is_file()
    )


def open_round(project_root: Path, log: HistoryLog | None = None) -> set[str]:
    """Return the versions kept since the most recent savepoint of *project_root*.

    The round in hand, which pruning must not collapse. Cut **by position**: a
    savepoint written in the same millisecond as a save cannot be ordered
    against it by timestamp, and `read_log` orders the file once and stably so
    that a position means something.

    A project no savepoint covers has every version in one open round, which is
    what the whole set being returned says.
    """
    entries = list((log if log is not None else read_log(project_root)).events)
    after = 0
    for index, event in enumerate(entries):
        if event.kind == "checkpoint":
            after = index + 1
    return {
        event.data["archived_as"]
        for event in entries[after:]
        if isinstance(event.data.get("archived_as"), str) and event.data["archived_as"]
    }
