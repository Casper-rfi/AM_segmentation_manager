"""Bringing a worked-on copy back into the collection it came from.

The copy's ``AM_export.yaml`` proves where it came from; **the common ancestor
is the history the two logs share**, which a copy merged back before holds more
of than its fork point names. **Every label is re-hashed on both sides**, never
read from a log, since anything can edit a file without recording it. **A merge
is additive and never guesses**: a target the copy lacks is not a deletion, no
image moves, a target both sides changed differently is a conflict a person
resolves, and both versions survive in the archive. **Family corrections are
merged first**, per entry, against the fork point's copy of them, and labels are
planned under the families the merge arrives at; a project with an undecided
family is not planned further until it is decided.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from ..families import (
    FAMILIES_SECTION,
    PRIMARY_SECTION,
    FamiliesFileError,
    read_families,
    sections_of,
)
from ..history import (
    Event,
    HistoryLog,
    Recorder,
    append_event,
    declared_project_id,
    read_log,
)
from ..io.atomic import Fingerprint, fingerprint_of
from ..io.image import read_array
from ..layout import FamilyCorrections, FamilyEntries, ProjectFamilies
from ..models import ImageRecord, Project
from ..scanner import Scanner
from ..state import DEFAULT_STATE, RecordState
from .export_record import (
    AM_EXPORT_FILENAME,
    ExportRecord,
    ProjectFork,
    read_export_record,
)
from .families_edit import (
    FamiliesChange,
    FamiliesValidationError,
    apply_families_change,
    plan_corrections,
    refuse_stale_change,
)
from .label_groups import build_groups
from .label_io import label_digest
from .label_save import LabelToSave, WriteProvenance, save_labels
from .metadata_edit import update_project_metadata
from .selection import current_state, plan_transition, transition_records

logger = logging.getLogger(__name__)

__all__ = [
    "CONFLICT",
    "CONVERGED",
    "EVERYTHING",
    "FamilyMerge",
    "KEEP_OURS",
    "OURS",
    "TAKE_THEIRS",
    "THEIRS",
    "UNCHANGED",
    "LabelMerge",
    "MergePlan",
    "MergeRefused",
    "MergeReport",
    "MetadataMerge",
    "ProjectMerge",
    "StateMerge",
    "apply_merge",
    "decisions_for",
    "field_key",
    "label_key",
    "plan_merge",
]

#: Neither side changed this since the fork.
UNCHANGED: Final[str] = "unchanged"
#: Only the copy changed it.
TAKE_THEIRS: Final[str] = "take-theirs"
#: Only this collection changed it.
KEEP_OURS: Final[str] = "keep-ours"
#: Both changed it and arrived at the same thing.
CONVERGED: Final[str] = "converged"
#: Both changed it, differently. A person decides.
CONFLICT: Final[str] = "conflict"

#: What a person's decision about a conflict says, as :func:`plan_merge` reads it.
OURS: Final[str] = "ours"
THEIRS: Final[str] = "theirs"

#: The scope holding every conflict of a merge.
EVERYTHING: Final[str] = "."

#: Outcomes a merge writes something for.
_APPLIED: Final[frozenset[str]] = frozenset({TAKE_THEIRS})


class MergeRefused(RuntimeError):
    """Raised when a merge must not be attempted at all.

    Distinct from a conflict: a conflict is one target a person can decide
    about, while this means the two trees cannot be compared -- no fork point,
    a fork point naming a history this collection does not have, or two
    projects that are not the same project -- or that a label the merge would
    take cannot be read.
    """


@dataclass(frozen=True)
class LabelMerge:
    """What became of one label group.

    Attributes:
        target: The group's key, relative to its project root.
        ancestor: The digest recorded at the fork, or None where the group had
            no label then.
        ours: This collection's digest now, re-hashed rather than read from a
            log. None where there is no label here.
        theirs: The copy's digest now. None where the copy has no label.
        our_path: Where the label belongs here, whether or not it exists.
        their_path: The copy's label.
        outcome: One of the five outcomes above.
        reason: Why, where the outcome is not obvious from the digests.
        source_event: The copy's last entry about this label, so what was taken
            can be traced back to the run that made it.
        corrected: Whether the group this would write over is proofread work.
        our_fingerprint: What our file was when the plan was made, so a merge
            carried out against a tree that has moved on since is refused
            rather than written blind.
        our_image: The acquisition the label belongs to, in this collection.
            Carried so a conflict can be looked at: both trees hold the same
            image, a merge never moving one, and this side is the one less
            likely to have been changed since the copy was taken.
        record: The group's first image record -- this collection's, or the
            copy's where only the copy holds the group. Carried so a conflict
            can be placed in its lamella and in acquisition order.
    """

    target: str
    ancestor: str | None
    ours: str | None
    theirs: str | None
    our_path: Path
    their_path: Path | None
    outcome: str
    reason: str = ""
    source_event: str = ""
    corrected: bool = False
    our_fingerprint: Fingerprint | None = None
    our_image: Path | None = None
    record: ImageRecord | None = None


@dataclass(frozen=True)
class StateMerge:
    """What became of one group's declared workflow state."""

    target: str
    ancestor: RecordState
    ours: RecordState
    theirs: RecordState
    record: ImageRecord | None
    outcome: str
    reason: str = ""


@dataclass(frozen=True)
class MetadataMerge:
    """What became of one project metadata field."""

    field: str
    ancestor: str | None
    ours: str | None
    theirs: str | None
    outcome: str
    reason: str = ""


@dataclass(frozen=True)
class FamilyMerge:
    """What became of one entry of ``AM_families.yaml``.

    Attributes:
        section: ``families`` or ``primary``, as the file names them.
        key: The condition, or the family, the entry is listed under.
        ancestor: Its value at the fork, None where there was no entry.
        ours: Its value here now.
        theirs: Its value in the copy now.
        lamella: The lamella whose section holds it, by its path in the
            project, or None for the project's own.
    """

    section: str
    key: str
    ancestor: str | None
    ours: str | None
    theirs: str | None
    outcome: str
    reason: str = ""
    lamella: str | None = None

    @property
    def field(self) -> str:
        """Return the name a decision about this entry is keyed by, after the project.

        ``family:sem_x``, or ``family:lam_003/sem_x`` for a lamella's: a
        condition's name holds no ``/``, so the two cannot be confused.
        """
        kind = "family" if self.section == FAMILIES_SECTION else "primary"
        return f"{kind}:{self.lamella}/{self.key}" if self.lamella else f"{kind}:{self.key}"

    def describe(self) -> str:
        """Return a one-line description, naming what each side holds."""
        what = "family of" if self.section == FAMILIES_SECTION else "primary of"
        if self.lamella:
            default = "as the project has it" if self.section == FAMILIES_SECTION else "the project's"
        else:
            default = "as named" if self.section == FAMILIES_SECTION else "default"
        where = f" in {self.lamella}" if self.lamella else ""
        return (
            f"{what} {self.key}{where}: {self.ours or default} here, "
            f"{self.theirs or default} in the copy"
        )


@dataclass(frozen=True)
class ProjectMerge:
    """Everything one project's merge would do.

    Attributes:
        key: The project's identity, its path relative to the scan root.
        our_root: Where it lives in this collection.
        their_root: Where it lives in the copy.
        labels: One entry per label group the copy holds.
        states: One entry per group whose declared state either side moved.
        metadata: One entry per metadata field either side edited.
        families: One entry per family correction either side made.
        families_change: What taking the copy's family corrections does here,
            or None where it takes none.
        deferred: Whether labels and states went unplanned, a family conflict
            being undecided: which label a group shares depends on it.
        incoming: The copy's history entries this collection does not have.
        source_project_id: What the copy's log says this project is, carried so
            an entry recording a merge can name where the work came from.
    """

    key: str
    our_root: Path
    their_root: Path
    source_project_id: str = ""
    labels: tuple[LabelMerge, ...] = ()
    states: tuple[StateMerge, ...] = ()
    metadata: tuple[MetadataMerge, ...] = ()
    families: tuple[FamilyMerge, ...] = ()
    families_change: FamiliesChange | None = None
    deferred: bool = False
    incoming: tuple[Event, ...] = ()

    @property
    def conflicts(self) -> tuple[object, ...]:
        """Return everything here a person still has to decide about."""
        return tuple(
            item
            for group in (self.families, self.labels, self.states, self.metadata)
            for item in group
            if item.outcome == CONFLICT
        )


@dataclass(frozen=True)
class MergePlan:
    """What merging a copy back would do, having done none of it."""

    source: Path
    destination: Path
    record: ExportRecord
    projects: tuple[ProjectMerge, ...] = ()

    @property
    def conflicts(self) -> tuple[object, ...]:
        """Return every unresolved conflict, across every project."""
        return tuple(item for project in self.projects for item in project.conflicts)

    @property
    def conflict_keys(self) -> tuple[str, ...]:
        """Return the name of every unresolved conflict, once each, in plan order.

        A label and its state share a name, being one decision.
        """
        keys: dict[str, None] = {}
        for project in self.projects:
            for entry in project.families:
                if entry.outcome == CONFLICT:
                    keys[field_key(project.key, entry.field)] = None
            groups: tuple[LabelMerge | StateMerge, ...] = (*project.labels, *project.states)
            for group in groups:
                if group.outcome == CONFLICT:
                    keys[label_key(project.key, group.target)] = None
            for edit in project.metadata:
                if edit.outcome == CONFLICT:
                    keys[field_key(project.key, edit.field)] = None
        return tuple(keys)

    @property
    def corrected(self) -> tuple[LabelMerge | StateMerge, ...]:
        """Return everything here that would write over, or unmark, proofread work.

        A label taken from the copy replaces work somebody did by hand, and a
        state taken from it can drop a corrected claim. Neither happens without
        being asked for by name -- the same rule ``revert`` follows.
        """
        labels = tuple(
            item
            for project in self.projects
            for item in project.labels
            if item.outcome in _APPLIED and item.corrected
        )
        states = tuple(
            item
            for project in self.projects
            for item in project.states
            if item.outcome in _APPLIED and item.ours == "corrected"
        )
        return labels + states

    @property
    def is_refused(self) -> bool:
        """Return whether anything is still to be decided.

        One conflict stops the whole merge. Applying the rest and leaving the
        conflicts would make "what happened" depend on what somebody had not
        got round to yet.
        """
        return bool(self.conflicts)

    def counted(self, outcome: str) -> int:
        """Return how many entries across the plan came to *outcome*."""
        return sum(
            1
            for project in self.projects
            for group in (project.families, project.labels, project.states, project.metadata)
            for item in group
            if item.outcome == outcome
        )

    def describe(self) -> list[str]:
        """Return one line per entry that would change or needs deciding."""
        lines = []
        for project in self.projects:
            for entry in project.families:
                if entry.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(f"{project.key}: {entry.describe()} ({entry.outcome})")
            if project.deferred:
                lines.append(
                    f"{project.key}: labels and states wait until its families are decided"
                )
            for label in project.labels:
                if label.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(
                        f"{project.key}/{label.target}: label {label.outcome}"
                        + (f" ({label.reason})" if label.reason else "")
                    )
            for state in project.states:
                if state.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(
                        f"{project.key}/{state.target}: state {state.outcome} "
                        f"{state.ours} -> {state.theirs}"
                    )
            for edit in project.metadata:
                if edit.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(
                        f"{project.key}: {edit.field} {edit.outcome} "
                        f"{edit.ours!r} -> {edit.theirs!r}"
                    )
        return lines


@dataclass(frozen=True)
class MergeReport:
    """What a merge actually did."""

    labels_taken: int = 0
    states_taken: int = 0
    fields_taken: int = 0
    families_taken: int = 0
    entries_added: int = 0
    archived: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Naming a decision
# ---------------------------------------------------------------------------

def label_key(project: str, target: str) -> str:
    """Return the name a decision about a label group, and its state, is keyed by."""
    return f"{project}/{target}"


def field_key(project: str, field: str) -> str:
    """Return the name a decision about a project metadata field is keyed by."""
    return f"{project}:{field}"


def decisions_for(plan: MergePlan, chosen: Mapping[str, str]) -> dict[str, str]:
    """Return one decision per conflict of *plan*, from decisions about scopes.

    A scope is a conflict's own name, a folder, project or lamella -- any path
    the names sit under -- or :data:`EVERYTHING`. **The narrowest scope holding
    a conflict decides it**, in whatever order they were given, so a project
    can go one way and a lamella in it the other.

    Raises:
        ValueError: If a side is neither :data:`OURS` nor :data:`THEIRS`, or a
            scope holds no conflict of *plan*: a mistyped path would otherwise
            decide nothing without saying so.
    """
    scopes = {_scope(scope): side for scope, side in chosen.items()}
    wrong = sorted(side for side in scopes.values() if side not in (OURS, THEIRS))
    if wrong:
        raise ValueError(
            f"{', '.join(map(repr, wrong))} is not a side: write {OURS!r} or {THEIRS!r}"
        )
    keys = plan.conflict_keys
    empty = [scope for scope in scopes if not any(_within(key, scope) for key in keys)]
    if empty:
        raise ValueError(
            f"{', '.join(map(repr, empty))} holds no conflict of this merge; "
            f"the conflicts are named '<project>/<label>' or '<project>:<field>'."
        )
    decided = {}
    for key in keys:
        holding = [scope for scope in scopes if _within(key, scope)]
        if holding:
            decided[key] = scopes[max(holding, key=_narrowness)]
    return decided


def _scope(scope: str) -> str:
    """Return *scope* as its conflicts are named: relative, with no trailing slash."""
    scope = scope.strip().removeprefix("./").rstrip("/")
    return scope or EVERYTHING


def _within(key: str, scope: str) -> bool:
    """Return whether the conflict named *key* sits under *scope*, by whole names."""
    if scope == EVERYTHING:
        return True
    return key == scope or key.startswith((f"{scope}/", f"{scope}:"))


def _narrowness(scope: str) -> int:
    """Return how narrow *scope* is: of two holding one conflict, one holds the other."""
    return -1 if scope == EVERYTHING else len(scope)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_merge(
    source: Path,
    destination: Path,
    *,
    resolutions: Mapping[str, str] | None = None,
) -> MergePlan:
    """Return what merging the copy at *source* into *destination* would do.

    Reads both trees and writes nothing.

    Args:
        source: The copy, holding an ``AM_export.yaml``.
        destination: The collection it came from.
        resolutions: Conflicts a person has already decided, keyed
            ``<project>/<target>`` or ``<project>:<field>``, valued
            :data:`THEIRS` or :data:`OURS`.

    Raises:
        MergeRefused: If the copy has no fork point, if a project in it is not
            in *destination*, if the two disagree about a project's identity,
            or if the fork point names an entry *destination* has never held.
    """

    export = read_export_record(source)
    if export is None:
        raise MergeRefused(
            f"{source} holds no fork point, so there is nothing to say what it "
            f"grew out of. Only a copy an export made can be merged back, and "
            f"the fork point is the {AM_EXPORT_FILENAME} at the copy's own "
            f"root -- choose that folder rather than one inside it."
        )

    ours = _projects_by_key(Scanner(destination).build_index().all_projects())
    theirs = _projects_by_key(Scanner(source).build_index().all_projects())
    decided = dict(resolutions or {})

    pairs: list[tuple[Project, Project, ProjectFork]] = []
    for key, fork in sorted(export.projects.items()):
        if key not in theirs:
            logger.info("%s is named by the fork point but not in the copy", key)
            continue
        pairs.append((_matching_project(key, fork, ours, destination), theirs[key], fork))

    # Families first, from the files alone: which label a group shares depends
    # on them. A tree is scanned again only where they differ from its own.
    families = {
        our.key: _plan_families(our.path, their.path, fork, our.key, decided)
        for our, their, fork in pairs
    }
    our_given: dict[Path, ProjectFamilies] = {}
    their_given: dict[Path, ProjectFamilies] = {}
    for our, their, _ in pairs:
        merged = families[our.key][1]
        if merged is None:
            continue
        if merged != _families_of(our.path, "here"):
            our_given[our.path] = merged
        if merged != _families_of(their.path, "in the copy"):
            their_given[their.path] = merged
    if our_given:
        ours = _projects_by_key(Scanner(destination, our_given).build_index().all_projects())
    if their_given:
        theirs = _projects_by_key(Scanner(source, their_given).build_index().all_projects())

    planned = []
    for our, their, fork in pairs:
        entries, merged = families[our.key]
        our_project, their_project = ours[our.key], theirs[their.key]
        change = None if merged is None else _check_families(our_project, their_project, merged)
        planned.append(
            _plan_project(
                our_project, their_project, fork, decided,
                families=entries, families_change=change, deferred=merged is None,
            )
        )
    return MergePlan(
        source=source, destination=destination, record=export, projects=tuple(planned)
    )


def _matching_project(
    key: str, fork: ProjectFork, ours: Mapping[str, Project], destination: Path
) -> Project:
    """Return the project in this collection the copy's *key* refers to.

    Matched by relative path, which is a project's identity everywhere here,
    then **verified by the id its log declares**. A path can be reused; an id
    that disagrees means these are two different projects wearing one name, and
    merging them would write one's labels over the other's.
    """
    if key not in ours:
        raise MergeRefused(
            f"The copy holds {key}, which is not in {destination}. A merge adds "
            "work to projects that are already there; it does not create them."
        )
    project = ours[key]
    if not fork.has_history:
        raise MergeRefused(
            f"{key} kept no history when the copy was taken, so there is no "
            "common ancestor to compare against. Run init-history in the "
            "collection and export again."
        )

    log = read_log(project.path)
    declared = declared_project_id(log.events)
    if fork.project_id and declared and fork.project_id != declared:
        raise MergeRefused(
            f"{key} in the copy says it is project {fork.project_id}, and here "
            f"it says {declared}. These are not the same project."
        )
    if not any(event.id == fork.tip for event in log.events):
        raise MergeRefused(
            f"The fork point says {key} was taken at entry {fork.tip}, which is "
            f"not in this collection's history for it. The copy did not come "
            f"from here, or came from a copy of it."
        )
    return project


def _plan_project(
    our_project: Project,
    their_project: Project,
    fork: ProjectFork,
    decided: Mapping[str, str],
    *,
    families: tuple[FamilyMerge, ...] = (),
    families_change: FamiliesChange | None = None,
    deferred: bool = False,
) -> ProjectMerge:
    """Return what one project's merge would do.

    Args:
        families: What became of each family correction.
        families_change: What taking the copy's corrections does here.
        deferred: Plan no labels or states, a family being undecided.
    """
    our_log = read_log(our_project.path)
    their_log = read_log(their_project.path)
    shared, ours_since, theirs_since = _common_history(our_log, their_log)

    our_groups = {} if deferred else _groups_of(our_project)
    their_groups = {} if deferred else _groups_of(their_project)
    ancestors = _digests_at(shared)

    their_tips = _last_by_target(theirs_since)
    labels = tuple(
        _plan_label(
            target, ancestors.get(target), our_groups.get(target), their_group,
            our_project.path, decided.get(label_key(our_project.key, target)),
            their_tips.get(target, ""),
        )
        for target, their_group in sorted(their_groups.items())
    )
    states = tuple(
        _plan_state(
            target, _state_at(shared, target), our_groups.get(target), their_group,
            decided.get(label_key(our_project.key, target)),
        )
        for target, their_group in sorted(their_groups.items())
    )
    return ProjectMerge(
        key=our_project.key,
        our_root=our_project.path,
        their_root=their_project.path,
        source_project_id=declared_project_id(their_log.events) or "",
        labels=tuple(item for item in labels if item.outcome != UNCHANGED),
        states=tuple(item for item in states if item.outcome != UNCHANGED),
        metadata=_plan_metadata(ours_since, theirs_since, our_project.key, decided),
        families=families,
        families_change=families_change,
        deferred=deferred,
        incoming=tuple(
            event
            for event in theirs_since
            if event.id not in {existing.id for existing in our_log.events}
        ),
    )


def _plan_label(
    target: str,
    ancestor: str | None,
    ours: _Group | None,
    theirs: _Group,
    our_root: Path,
    decision: str | None,
    source_event: str,
) -> LabelMerge:
    """Return what becomes of one label, on the digests read from disk now."""

    our_digest = ours.digest if ours else None
    their_digest = theirs.digest
    our_path = ours.label_path if ours and ours.label_path else our_root / target

    outcome, reason = _resolve(ancestor, our_digest, their_digest, decision)
    return LabelMerge(
        target=target, ancestor=ancestor, ours=our_digest, theirs=their_digest,
        our_path=our_path, their_path=theirs.label_path,
        outcome=outcome, reason=reason, source_event=source_event,
        corrected=bool(ours and ours.state == "corrected"),
        our_fingerprint=fingerprint_of(our_path),
        our_image=ours.record.image_path if ours else None,
        record=ours.record if ours else theirs.record,
    )


def _plan_state(
    target: str,
    ancestor: RecordState,
    ours: _Group | None,
    theirs: _Group,
    decision: str | None,
) -> StateMerge:
    """Return what becomes of one group's declared state."""
    our_state = ours.state if ours else DEFAULT_STATE
    outcome, reason = _resolve(ancestor, our_state, theirs.state, decision)
    return StateMerge(
        target=target, ancestor=ancestor, ours=our_state, theirs=theirs.state,
        record=ours.record if ours else None, outcome=outcome, reason=reason,
    )


def _families_of(root: Path, side: str) -> ProjectFamilies:
    """Return a project's family corrections, refusing a file that will not read."""
    try:
        return read_families(root)
    except FamiliesFileError as error:
        raise MergeRefused(f"{error} ({side}); correct it before merging.") from error


def _plan_families(
    our_root: Path,
    their_root: Path,
    fork: ProjectFork,
    key: str,
    decided: Mapping[str, str],
) -> tuple[tuple[FamilyMerge, ...], ProjectFamilies | None]:
    """Return what becomes of each family correction, and the corrections merged.

    Three-way per entry, as a metadata field is, but **the ancestor is the fork
    point's copy of the file** rather than one recovered from the log, and each
    side is read from its file, so an edit by hand is seen. Names compare
    whatever their case. Each lamella's section merges as the project's does,
    its entries its own (``test_a_lamella_entry_changed_only_in_the_copy_is_taken``).

    Returns:
        The entries either side changed, and the corrections the merge arrives
        at -- None while an entry is in conflict and undecided.

    Raises:
        MergeRefused: If either file will not read, or the entries taken
            together do not make valid corrections.
    """
    ours = _families_of(our_root, "here")
    theirs = _families_of(their_root, "in the copy")
    ancestor = fork.families
    entries: list[FamilyMerge] = []
    project_section = _merge_section(
        (ancestor.project, ours.project, theirs.project), None, key, decided, entries
    )
    empty = FamilyEntries()
    lamellae: dict[str, FamilyEntries] = {}
    for lamella in sorted(set(ancestor.lamellae) | set(ours.lamellae) | set(theirs.lamellae)):
        sections = tuple(
            side.lamellae.get(lamella, empty) for side in (ancestor, ours, theirs)
        )
        families, primary = _merge_section(sections, lamella, key, decided, entries)
        if families or primary:
            lamellae[lamella] = FamilyEntries(families=families, primary=primary)

    if any(entry.outcome == CONFLICT for entry in entries):
        return tuple(entries), None
    try:
        corrections = ProjectFamilies(
            project=FamilyCorrections(families=project_section[0], primary=project_section[1]),
            lamellae=lamellae,
        )
    except ValueError as error:
        raise MergeRefused(
            f"{key}: the family corrections the two sides make do not fit together: {error}"
        ) from error
    return tuple(entries), corrections


def _merge_section(
    sides: tuple[FamilyCorrections | FamilyEntries, ...],
    lamella: str | None,
    key: str,
    decided: Mapping[str, str],
    entries: list[FamilyMerge],
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge one section, the project's or a lamella's, entry by entry.

    Args:
        sides: The section at the fork, here and in the copy, in that order.
        entries: Appended with each entry either side changed.

    Returns:
        The families and primary mappings the merge arrives at, ours where
        an entry is in conflict.
    """
    ancestor_side, our_side, their_side = (sections_of(side) for side in sides)
    merged = {section: dict(entries) for section, entries in our_side}
    for (section, ancestor_map), (_, our_map), (_, their_map) in zip(
        ancestor_side, our_side, their_side, strict=True
    ):
        maps = [
            {name.lower(): (name, value) for name, value in mapping.items()}
            for mapping in (ancestor_map, our_map, their_map)
        ]
        for lowered in sorted(set().union(*maps)):
            spelled = next(m[lowered][0] for m in (maps[1], maps[2], maps[0]) if lowered in m)
            ancestor, mine, other = (m.get(lowered, (None, None))[1] for m in maps)
            entry = FamilyMerge(
                section, spelled, ancestor, mine, other, UNCHANGED, lamella=lamella
            )
            outcome, reason = _resolve(
                _lowered(ancestor), _lowered(mine), _lowered(other),
                decided.get(field_key(key, entry.field)),
            )
            if outcome == UNCHANGED:
                continue
            entries.append(replace(entry, outcome=outcome, reason=reason))
            if outcome == TAKE_THEIRS:
                taken = {n: v for n, v in merged[section].items() if n.lower() != lowered}
                if other is not None:
                    taken[maps[2][lowered][0]] = other
                merged[section] = taken
    return merged[FAMILIES_SECTION], merged[PRIMARY_SECTION]


def _lowered(value: str | None) -> str | None:
    return value.lower() if value is not None else None


def _check_families(
    our_project: Project, their_project: Project, merged: ProjectFamilies
) -> FamiliesChange | None:
    """Return what the merged corrections do here, having checked both trees.

    Held to every check ``edit-families`` makes, on both sides: here, so no
    label of this collection is left unread or replaced; in the copy, so none
    of its work is.

    Raises:
        MergeRefused: Naming what either tree refuses.
    """
    checks = (
        (our_project, "cannot be applied here"),
        (their_project, "would leave work in the copy unread"),
    )
    changes = []
    for project, what in checks:
        try:
            changes.append(plan_corrections(project, merged, allow_leaving_corrected=True))
        except FamiliesValidationError as error:
            raise MergeRefused(
                f"{our_project.key}: the families this merge arrives at {what}: "
                + "; ".join(error.problems)
            ) from error
    return None if changes[0].is_empty else changes[0]


def _plan_metadata(
    ours_since: Sequence[Event],
    theirs_since: Sequence[Event],
    key: str,
    decided: Mapping[str, str],
) -> tuple[MetadataMerge, ...]:
    """Return what becomes of each metadata field either side edited.

    The ancestor is not read from a file: nothing baselines metadata, so what
    each field held at the fork is recovered from the *first* edit either side
    made to it, which records the value it replaced. A field neither side
    touched needs no ancestor, because nothing has happened to it.
    """
    ours_edits = _edits_by_field(ours_since)
    theirs_edits = _edits_by_field(theirs_since)

    merged = []
    for name in sorted(set(ours_edits) | set(theirs_edits)):
        mine, other = ours_edits.get(name, []), theirs_edits.get(name, [])
        ancestor = (mine or other)[0].data.get("old")
        our_value = mine[-1].data.get("new") if mine else ancestor
        their_value = other[-1].data.get("new") if other else ancestor
        outcome, reason = _resolve(
            ancestor, our_value, their_value, decided.get(field_key(key, name))
        )
        if outcome != UNCHANGED:
            merged.append(
                MetadataMerge(
                    field=name, ancestor=ancestor, ours=our_value,
                    theirs=their_value, outcome=outcome, reason=reason,
                )
            )
    return tuple(merged)


def _resolve(
    ancestor: object, ours: object, theirs: object, decision: str | None
) -> tuple[str, str]:
    """Return the outcome of a three-way comparison, and why.

    The whole resolution rule, in one place so a label, a state and a metadata
    field cannot come to different answers about the same shape of change.
    **Timestamps decide nothing here**, clocks differing between machines and a
    later one routinely being a recompression sitting on top of real work; nor
    does any rule about which tree is authoritative, that being a way of
    discarding somebody's proofreading quietly.
    """
    if ours == theirs:
        return (UNCHANGED, "") if ours == ancestor else (CONVERGED, "both arrived here")
    if ours == ancestor:
        return TAKE_THEIRS, ""
    if theirs == ancestor:
        return KEEP_OURS, ""
    if decision == THEIRS:
        return TAKE_THEIRS, "resolved by hand"
    if decision == OURS:
        return KEEP_OURS, "resolved by hand"
    return CONFLICT, "both changed this since the copy was taken"


# ---------------------------------------------------------------------------
# Reading the two sides
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Group:
    """One label group as it stands in a tree, read from disk."""

    target: str
    label_path: Path | None
    digest: str | None
    state: RecordState
    record: ImageRecord


def _groups_of(project: Project) -> dict[str, _Group]:
    """Return every label group of *project*, keyed by its history target.

    **The digest is taken from the file now**, not from the log: the log says
    what this package did, and a merge has to know what is actually there.
    """

    groups = {}
    # Grouped under the families the project was scanned with, which in a merge
    # can be ones on neither disk yet.
    families = {
        stage.path: project.families_at(stage.path)
        for lamella in project.lamellae for stage in lamella.stages
    }
    for group in build_groups(project.image_records, families):
        identity = group.label_path or group.identity
        if identity is None:
            continue
        try:
            target = identity.resolve().relative_to(project.path.resolve()).as_posix()
        except ValueError:
            continue
        groups[target] = _Group(
            target=target,
            label_path=group.label_path,
            digest=label_digest(group.label_path),
            state=current_state(group.records[0]),
            record=group.records[0],
        )
    return groups


def _projects_by_key(projects: Sequence[Project]) -> dict[str, Project]:
    """Return *projects* keyed by identity."""
    return {project.key: project for project in projects}


def _common_history(
    ours: HistoryLog, theirs: HistoryLog
) -> tuple[list[Event], list[Event], list[Event]]:
    """Return the entries both logs hold, and what each has that the other has not.

    The common ancestor, and the two divergences from it. Read as a set rather
    than as a prefix, because a log is not a line: a previous merge added the
    copy's entries to this collection, and a later export from either side puts
    entries below them. An id is a hash of an entry's whole body, so an entry
    the two share is one that physically travelled between them.
    """
    our_ids = {event.id for event in ours.events}
    their_ids = {event.id for event in theirs.events}
    return (
        [event for event in ours.events if event.id in their_ids],
        [event for event in ours.events if event.id not in their_ids],
        [event for event in theirs.events if event.id not in our_ids],
    )


def _digests_at(events: Sequence[Event]) -> dict[str, str | None]:
    """Return what each target's label held at the end of *events*."""
    digests: dict[str, str | None] = {}
    for event in events:
        if event.target and event.kind == "label-removed":
            digests[event.target] = None
        elif event.target and "digest" in event.data:
            value = event.data["digest"]
            digests[event.target] = value if isinstance(value, str) else None
    return digests


def _state_at(events: Sequence[Event], target: str) -> RecordState:
    """Return the state *target* was declared to be in at the fork."""
    state: RecordState = DEFAULT_STATE
    for event in events:
        if event.target != target:
            continue
        if event.kind == "state-changed" and event.data.get("new") in _STATES:
            state = event.data["new"]
        elif event.kind == "baseline" and event.data.get("state") in _STATES:
            state = event.data["state"]
    return state


_STATES: Final[frozenset[str]] = frozenset({"selected", "deselected", "corrected"})


def _last_by_target(events: Sequence[Event]) -> dict[str, str]:
    """Return the id of the last entry about each target."""
    return {event.target: event.id for event in events if event.target}


def _edits_by_field(events: Sequence[Event]) -> dict[str, list[Event]]:
    """Return the metadata edits in *events*, grouped by field and in order."""
    edits: dict[str, list[Event]] = {}
    for event in events:
        if event.kind != "metadata-changed":
            continue
        name = event.data.get("field")
        if isinstance(name, str):
            edits.setdefault(name, []).append(event)
    return edits


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def apply_merge(
    plan: MergePlan, *, recorder: Recorder, allow_corrected: bool = False
) -> MergeReport:
    """Carry out *plan*, or none of it.

    Refuses while any conflict stands: applying the rest would make what
    happened depend on what somebody had not got round to deciding. Every label
    to take is read, and every state change rehearsed, before the first write.

    **Every label goes through the ordinary save, in one batch**, so a merge is
    validated whole before it writes anything and is held to every guard a
    proofreading save is held to -- the staleness refusal, the protection of
    proofread work, and the archiving of what it replaces. The copy's history
    entries are added whole, an id being a hash of its content, so adding them
    twice adds them once.

    Args:
        allow_corrected: Confirm replacing labels, and dropping corrected
            claims, that are somebody's proofreading.

    Raises:
        MergeRefused: If any conflict is unresolved, or a label to take from
            the copy cannot be read. Nothing is written.
        PermissionError: If it would touch proofread work and *allow_corrected*
            was not set. Checked over the whole merge before the first write.
        StaleLabels: If a label here changed between the plan and now.
        SaveStopped: If the version a label replaces could not be kept; the
            labels after it are left as they were.
    """
    if plan.is_refused:
        raise MergeRefused(
            f"{len(plan.conflicts)} conflict(s) are undecided; nothing was merged."
        )
    if plan.corrected and not allow_corrected:
        names = ", ".join(item.target for item in plan.corrected[:3])
        raise PermissionError(
            f"Refusing to merge over {len(plan.corrected)} corrected item(s) "
            f"without explicit confirmation ({names}). The version here is "
            f"kept either way, so this can be undone -- but not unasked."
        )
    changes = [project.families_change for project in plan.projects if project.families_change]
    leaving = [
        effect for change in changes for effect in change.states.effects
        if effect.old == "corrected"
    ]
    if leaving and not allow_corrected:
        raise PermissionError(
            f"Refusing to take the copy's families without explicit confirmation: "
            f"{len(leaving)} image(s) here would leave a corrected group, starting "
            f"with {leaving[0].image}."
        )

    taking = [
        (project, item)
        for project in plan.projects
        for item in project.labels
        if item.outcome in _APPLIED and item.their_path is not None
    ]
    to_save, unreadable = [], []
    for project, item in taking:
        try:
            to_save.append(_label_to_save(item, project))
        except Exception as error:  # noqa: BLE001 - every label that will not read is named
            unreadable.append(f"{project.key}/{item.target} ({error})")
    if unreadable:
        raise MergeRefused(
            f"{len(unreadable)} of the copy's label(s) could not be read; nothing "
            f"was merged: {', '.join(unreadable[:3])}"
        )

    # Every state change is rehearsed before the first label is written, so a
    # state that cannot be declared stops the merge with nothing changed.
    by_state: dict[RecordState, list[ImageRecord]] = {}
    for project in plan.projects:
        for state in project.states:
            if state.outcome in _APPLIED and state.record is not None:
                by_state.setdefault(state.theirs, []).append(state.record)
    for declared, records in by_state.items():
        plan_transition(records, declared, allow_leaving_corrected=allow_corrected)
    for change in changes:
        refuse_stale_change(change)

    # Written before the labels, which are placed where the merged families put them.
    for change in changes:
        apply_families_change(change, recorder=recorder)

    report = save_labels(
        to_save, allow_corrected=allow_corrected, recorder=recorder
    )

    for declared, records in by_state.items():
        transition_records(
            records, declared, allow_leaving_corrected=allow_corrected, recorder=recorder
        )
    states = sum(len(records) for records in by_state.values())
    fields = entries = 0
    for project in plan.projects:
        for edit in project.metadata:
            if edit.outcome in _APPLIED:
                _take_field(project, edit, recorder)
                fields += 1
        entries += _add_entries(project)

    families = sum(
        1 for project in plan.projects for entry in project.families
        if entry.outcome in _APPLIED
    )
    logger.info(
        "Merged %s famil(y/ies), %s label(s), %s state(s) and %s field(s); "
        "added %s history entr(ies)",
        families, report.count, states, fields, entries,
    )
    return MergeReport(
        labels_taken=report.count, states_taken=states, fields_taken=fields,
        families_taken=families, entries_added=entries, archived=report.archived,
    )


def _label_to_save(item: LabelMerge, project: ProjectMerge) -> LabelToSave:
    """Return the copy's label as something the ordinary save can write.

    The pixels are read rather than the file copied, so what lands here is
    written the way every other label is. Nothing is written here: the save
    creates the label's directory where it has none.

    Raises:
        ValueError: If the copy holds no label for the item.
        Exception: Whatever reading the copy's label raised.
    """

    if item.their_path is None:
        raise ValueError("the copy holds no label")
    pixels = read_array(item.their_path)

    fields: dict[str, object] = {
        "resolution": item.outcome,
        "source_project_id": project.source_project_id,
        "source_event": item.source_event,
    }
    if item.ancestor:
        fields["parent_digest"] = item.ancestor
    return LabelToSave(
        destination=item.our_path,
        label=pixels,
        created=not item.our_path.is_file(),
        fingerprint=item.our_fingerprint,
        corrected=item.corrected,
        provenance=WriteProvenance("merged", fields),
    )


def _take_field(project: ProjectMerge, item: MetadataMerge, recorder: Recorder) -> None:
    """Write the copy's value for one metadata field."""

    update_project_metadata(
        project.our_root, {item.field: item.theirs}, recorder=recorder
    )


def _add_entries(project: ProjectMerge) -> int:
    """Append the copy's history entries this collection does not have.

    Union rather than reconciliation: an id is a hash of its content, so the
    same entry from both sides is one entry, and doing this twice changes
    nothing the second time.
    """
    present = {event.id for event in read_log(project.our_root).events}
    added = 0
    for event in project.incoming:
        if event.id in present:
            continue
        append_event(project.our_root, event)
        present.add(event.id)
        added += 1
    return added
