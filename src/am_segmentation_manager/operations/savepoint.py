"""Naming a state of the data, so a round of work can be compared and undone.

Proofreading is hundreds of small saves; the round over them -- proofread, train,
predict, convert, proofread again -- is what **a savepoint marks**. **A savepoint
is a set of per-project entries sharing one id**, there being no "collection" to
store: coverage is answered by looking, not by a stored list. **It promises nothing
about pixels; it reports what it can reach** -- the state the record knew, each
version recoverable only while the archive holds it, asked of the log whenever
somebody wants the answer. Workflow states and project metadata are read from the
same log: what each held before the first change recorded after the savepoint.
Family corrections changed since are named, not restored.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..history import Event, HistoryLog, Recorder, history_file_path, read_log
from ..io.long_paths import refuse_unreachable
from ..models import Project
from ..state import STATE_VALUES, RecordState
from .archive import (
    NAME_DIGEST_LENGTH,
    keep_version,
    projected_archive_path,
    version_directory,
)

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
    "families_since",
    "fields_at",
    "find_savepoint",
    "list_savepoints",
    "not_recording",
    "open_round",
    "plan_savepoint",
    "reachability",
    "reached_versions",
    "states_at",
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
        states: The state of each group changed since, as it was then, by target.
        fields: The value of each metadata field changed since, as it was then.
        families: The family corrections changed since, which a return to it
            does not restore.
    """

    project: Project
    entry: Event | None = None
    targets: tuple[TargetState, ...] = ()
    states: Mapping[str, RecordState] = field(default_factory=dict)
    fields: Mapping[str, str | None] = field(default_factory=dict)
    families: tuple[str, ...] = ()

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
    by: str | None = ""
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

    @property
    def states_changed(self) -> int:
        """Return how many groups' states have changed since, across the projects."""
        return sum(len(project.states) for project in self.projects)

    @property
    def fields_changed(self) -> int:
        """Return how many metadata fields have changed since, across the projects."""
        return sum(len(project.fields) for project in self.projects)

    def describe(self) -> str:
        """Return the one line a listing shows."""
        note = f"{self.found} project(s) here"
        if self.covered_at_creation and self.covered_at_creation != self.found:
            note += f", {self.covered_at_creation} at creation"
        return (
            f"{self.title or self.id}  {self.at[:10]}  {self.by or 'unknown'}  "
            f"{note}  {self.reachable}/{self.targets} labels reachable, "
            f"{self.states_changed} state(s) and {self.fields_changed} field(s) changed since"
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
        too_late: Projects left out of a join because the savepoint predates
            everything their record knows.
    """

    id: str
    title: str
    description: str = ""
    projects: tuple[Project, ...] = ()
    joining: bool = False
    states: dict[str, tuple[TargetState, ...]] = field(default_factory=dict)
    too_late: tuple[Project, ...] = ()

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
        lines = [
            f"{project.key}: left out, its history begins after this savepoint "
            f"was taken"
            for project in self.too_late
        ]
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
        if event.target and event.kind == "label-removed":
            digests.pop(event.target, None)
            continue
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
    """Return the last entry about each target's label.

    Only entries that say what the label holds, or that it was removed: a
    group's state is recorded under the same target, and a state changed since
    says nothing about the pixels.
    """
    latest: dict[str, Event] = {}
    for event in log.events:
        if event.target and (
            isinstance(event.data.get("digest"), str) or event.kind == "label-removed"
        ):
            latest[event.target] = event
    return latest


def states_at(log: HistoryLog, entry: Event | None) -> dict[str, RecordState]:
    """Return the state each group changed since *entry* was in at *entry*, by target.

    Read off the first change after it, whose ``old`` is that state. A group no
    change has touched since is in the state it was in then, so it is left out.
    """
    then: dict[str, RecordState] = {}
    for event in _after(log, entry):
        old = event.data.get("old")
        if (
            event.kind == "state-changed" and event.target
            and event.target not in then and old in STATE_VALUES
        ):
            then[event.target] = old
    return then


def fields_at(log: HistoryLog, entry: Event | None) -> dict[str, str | None]:
    """Return the value each metadata field changed since *entry* held at *entry*.

    As :func:`states_at`, from the first edit after it. None is a field that
    held no value. An edit made outside AM-seg is not in the log, so not here.
    """
    then: dict[str, str | None] = {}
    for event in _after(log, entry):
        name, old = event.data.get("field"), event.data.get("old")
        if event.kind == "metadata-changed" and isinstance(name, str) and name not in then:
            then[name] = old if isinstance(old, str) else None
    return then


def families_since(log: HistoryLog, entry: Event | None) -> tuple[str, ...]:
    """Return the family corrections changed since *entry*, each named once, in order.

    A correction is named by the condition it is listed under, after its
    lamella's path where it is that lamella's own.
    """
    named = [
        f"{event.data['lamella']}/{event.data['key']}" if event.data.get("lamella")
        else str(event.data.get("key"))
        for event in _after(log, entry)
        if event.kind == "family-changed"
    ]
    return tuple(dict.fromkeys(named))


def _after(log: HistoryLog, entry: Event | None) -> list[Event]:
    """Return the entries after *entry*, by position, as :func:`digests_before` cuts."""
    events = list(log.events)
    if entry is None:
        return []
    for index, candidate in enumerate(events):
        if candidate.id == entry.id:
            return events[index + 1:]
    return []


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
                        states=states_at(log, entry),
                        fields=fields_at(log, entry),
                        families=families_since(log, entry),
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

    too_late: list[Project] = []
    if joining is not None:
        wanted, too_late = _knew_the_moment(wanted, joining.at)

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
        too_late=tuple(too_late),
    )


def _knew_the_moment(
    projects: Sequence[Project], at: str
) -> tuple[list[Project], list[Project]]:
    """Split *projects* by whether their record reaches back to *at*.

    A project whose history begins after a savepoint was taken **was not in
    that state**, either because it arrived later or because it was recording
    nothing at the time. Letting it join would record a claim about a moment
    the record knows nothing of, which is the one thing a savepoint must not
    do -- and it is what would make joining soften a savepoint into a label
    somebody can attach to anything.

    The honest case is the other one: a machine holding three of twelve
    projects when the round was taken, the other nine recording all along and
    merely out of scope. Those reach back, and may join.

    Compared by time rather than by position, two logs having no shared
    positions. Clocks are the only thing that orders them, and a project that
    was really there predates a savepoint by a long way rather than a
    millisecond.
    """
    joins: list[Project] = []
    refused: list[Project] = []
    for project in projects:
        events = read_log(project.path).events
        (joins if events and events[0].at <= at else refused).append(project)
    return joins, refused


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

    Raises:
        PathTooLong: If Windows cannot reach a version's archived copy here.
            Checked before anything is copied.
        ArchiveFailed: If a version could not be kept, so the savepoint that
            would name it is not taken.
    """
    refuse_unreachable(
        projected
        for project, state in plan.at_risk
        if (projected := projected_archive_path(project.path, project.path / state.target))
    )
    kept = []
    for project, state in plan.at_risk:
        source = project.path / state.target
        if not source.is_file():
            logger.warning("%s is named by the savepoint but is not here", state.target)
            continue
        archived = keep_version(project.path, state.target, source, state.digest)
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

    A project added since the collection was baselined has no history until
    something changes it or ``init-history`` runs, and until then it joins no
    savepoint and carries no ancestor for a merge -- silently, since nothing
    that only reads may start a log. The silence is the point and also the
    danger, so the one command that asks what the record covers says which
    projects it does not.
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
