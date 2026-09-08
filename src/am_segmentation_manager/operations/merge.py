"""Bringing a worked-on copy back into the collection it came from.

A copy carries its projects' histories and an ``AM_export.yaml`` naming the
entry each log ended at. **That entry proves where the copy came from; the
common ancestor is the history the two logs share.** Everything here is a
three-way comparison against that: what this collection has done since, what
the copy has done since, and whether those can both be true.

Taking the recorded tip as the ancestor instead would be right exactly once. A
merge adds the copy's entries to this collection, so a copy that is worked on
and brought back a second time shares more history than its fork point names --
and comparing against the fork point would call every label taken the first time
a conflict, each side having "changed" it since.

**Nothing is trusted about current content.** Every label in scope is re-hashed
on both sides rather than read out of a log, because anything can have edited a
file without recording it -- the external segmentation software, a manual copy,
a restore from backup -- and a merge that believed the log would keep the wrong
version while reporting success.

**A merge is additive and it never guesses.** A target the copy does not hold is
not a deletion, a copy holding a fraction of a collection being the ordinary
case; no image is ever moved; and a target both sides changed differently is a
conflict a person resolves by name, never by a timestamp or a rule about which
tree wins. Both versions survive in the archive whichever way it goes.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..history import Event, HistoryLog, Recorder, append_event, read_log
from ..models import ImageRecord, Project
from ..state import DEFAULT_STATE, RecordState
from .export_record import (
    AM_EXPORT_FILENAME,
    ExportRecord,
    ProjectFork,
    read_export_record,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONFLICT",
    "CONVERGED",
    "KEEP_OURS",
    "TAKE_THEIRS",
    "UNCHANGED",
    "LabelMerge",
    "MergePlan",
    "MergeRefused",
    "MergeReport",
    "MetadataMerge",
    "ProjectMerge",
    "StateMerge",
    "apply_merge",
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

#: Outcomes a merge writes something for.
_APPLIED: Final[frozenset[str]] = frozenset({TAKE_THEIRS})


class MergeRefused(RuntimeError):
    """Raised when a merge must not be attempted at all.

    Distinct from a conflict: a conflict is one target a person can decide
    about, while this means the two trees cannot be compared -- no fork point,
    a fork point naming a history this collection does not have, or two
    projects that are not the same project.
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
    our_fingerprint: tuple[int, int] | None = None


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
class ProjectMerge:
    """Everything one project's merge would do.

    Attributes:
        key: The project's identity, its path relative to the scan root.
        our_root: Where it lives in this collection.
        their_root: Where it lives in the copy.
        labels: One entry per label group the copy holds.
        states: One entry per group whose declared state either side moved.
        metadata: One entry per metadata field either side edited.
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
    incoming: tuple[Event, ...] = ()

    @property
    def conflicts(self) -> tuple[object, ...]:
        """Return everything here a person still has to decide about."""
        return tuple(
            item
            for group in (self.labels, self.states, self.metadata)
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
    def corrected(self) -> tuple[object, ...]:
        """Return everything here that would write over, or unmark, proofread work.

        A label taken from the copy replaces work somebody did by hand, and a
        state taken from it can drop a corrected claim. Neither happens without
        being asked for by name -- the same rule ``revert`` follows, and the
        one thing a merge did not used to check.
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
            for group in (project.labels, project.states, project.metadata)
            for item in group
            if item.outcome == outcome
        )

    def describe(self) -> list[str]:
        """Return one line per entry that would change or needs deciding."""
        lines = []
        for project in self.projects:
            for item in project.labels:
                if item.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(
                        f"{project.key}/{item.target}: label {item.outcome}"
                        + (f" ({item.reason})" if item.reason else "")
                    )
            for item in project.states:
                if item.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(
                        f"{project.key}/{item.target}: state {item.outcome} "
                        f"{item.ours} -> {item.theirs}"
                    )
            for item in project.metadata:
                if item.outcome in (TAKE_THEIRS, CONFLICT):
                    lines.append(
                        f"{project.key}: {item.field} {item.outcome} "
                        f"{item.ours!r} -> {item.theirs!r}"
                    )
        return lines


@dataclass(frozen=True)
class MergeReport:
    """What a merge actually did."""

    labels_taken: int = 0
    states_taken: int = 0
    fields_taken: int = 0
    entries_added: int = 0
    archived: tuple[str, ...] = ()


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
            ``<project>/<target>`` or ``<project>:<field>``, valued ``theirs``
            or ``ours``.

    Raises:
        MergeRefused: If the copy has no fork point, if a project in it is not
            in *destination*, if the two disagree about a project's identity,
            or if the fork point names an entry *destination* has never held.
    """
    from ..scanner import Scanner

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

    merged = []
    for key, fork in sorted(export.projects.items()):
        if key not in theirs:
            logger.info("%s is named by the fork point but not in the copy", key)
            continue
        our_project = _matching_project(key, fork, ours, destination)
        merged.append(
            _plan_project(our_project, theirs[key], fork, decided)
        )
    return MergePlan(
        source=source, destination=destination, record=export, projects=tuple(merged)
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
    declared = _declared_id(log.events)
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
) -> ProjectMerge:
    """Return what one project's merge would do."""
    our_log = read_log(our_project.path)
    their_log = read_log(their_project.path)
    shared, ours_since, theirs_since = _common_history(our_log, their_log)

    our_groups = _groups_of(our_project)
    their_groups = _groups_of(their_project)
    ancestors = _digests_at(shared)

    their_tips = _last_by_target(theirs_since)
    labels = tuple(
        _plan_label(
            target, ancestors.get(target), our_groups.get(target), their_group,
            our_project.path, decided.get(f"{our_project.key}/{target}"),
            their_tips.get(target, ""),
        )
        for target, their_group in sorted(their_groups.items())
    )
    states = tuple(
        _plan_state(
            target, _state_at(shared, target), our_groups.get(target), their_group,
            decided.get(f"{our_project.key}/{target}"),
        )
        for target, their_group in sorted(their_groups.items())
    )
    return ProjectMerge(
        key=our_project.key,
        our_root=our_project.path,
        their_root=their_project.path,
        source_project_id=_declared_id(their_log.events) or "",
        labels=tuple(item for item in labels if item.outcome != UNCHANGED),
        states=tuple(item for item in states if item.outcome != UNCHANGED),
        metadata=_plan_metadata(ours_since, theirs_since, our_project.key, decided),
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
    from ..io.atomic import fingerprint_of

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
        mine, other = ours_edits.get(name), theirs_edits.get(name)
        ancestor = (mine or other)[0].data.get("old")
        our_value = mine[-1].data.get("new") if mine else ancestor
        their_value = other[-1].data.get("new") if other else ancestor
        outcome, reason = _resolve(
            ancestor, our_value, their_value, decided.get(f"{key}:{name}")
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
    if decision == "theirs":
        return TAKE_THEIRS, "resolved by hand"
    if decision == "ours":
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
    from .label_groups import build_groups
    from .label_io import label_fields
    from .selection import current_state

    groups = {}
    for group in build_groups(_records_of(project)):
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
            digest=_digest_of(group.label_path, label_fields),
            state=current_state(group.records[0]),
            record=group.records[0],
        )
    return groups


def _digest_of(path: Path | None, label_fields) -> str | None:
    """Return what a label holds now, or None where it has nothing to say."""
    if path is None or not path.is_file():
        return None
    try:
        from ..io.image import read_array

        return str(label_fields(read_array(path))["digest"])
    except Exception as error:
        logger.warning("Could not read %s: %s", path, error)
        return None


def _records_of(project: Project) -> list[ImageRecord]:
    """Return every image record of *project*."""
    return [record for lamella in project.lamellae for record in lamella.image_records]


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
        if event.target and "digest" in event.data:
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


def _declared_id(events: Sequence[Event]) -> str | None:
    """Return the identity a log declares, if it declares one."""
    for event in events:
        if event.kind == "log-created":
            declared = event.data.get("project_id")
            if isinstance(declared, str) and declared:
                return declared
    return None


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def apply_merge(
    plan: MergePlan, *, recorder: Recorder, allow_corrected: bool = False
) -> MergeReport:
    """Carry out *plan*, or none of it.

    Refuses while any conflict stands: applying the rest would make what
    happened depend on what somebody had not got round to deciding.

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
        MergeRefused: If any conflict is unresolved.
        PermissionError: If it would touch proofread work and *allow_corrected*
            was not set. Checked over the whole merge before the first write.
        StaleLabels: If a label here changed between the plan and now.
    """
    if plan.is_refused:
        raise MergeRefused(
            f"{len(plan.conflicts)} conflict(s) are undecided; nothing was merged."
        )
    if plan.corrected and not allow_corrected:
        names = ", ".join(item.target for item in plan.corrected[:3])
        raise PermissionError(
            f"Refusing to merge over {len(plan.corrected)} piece(s) of proofread "
            f"work without explicit confirmation ({names}). The version here is "
            f"kept either way, so this can be undone -- but not unasked."
        )

    from .label_save import save_labels

    taking = [
        (project, item)
        for project in plan.projects
        for item in project.labels
        if item.outcome in _APPLIED and item.their_path is not None
    ]
    to_save = []
    for project, item in taking:
        prepared = _label_to_save(item, project)
        if prepared is not None:
            to_save.append(prepared)

    report = save_labels(
        to_save, allow_corrected=allow_corrected, recorder=recorder
    )

    states = fields = entries = 0
    for project in plan.projects:
        for item in project.states:
            if item.outcome in _APPLIED and item.record is not None:
                _take_state(item, recorder, allow_corrected=allow_corrected)
                states += 1
        for item in project.metadata:
            if item.outcome in _APPLIED:
                _take_field(project, item, recorder)
                fields += 1
        entries += _add_entries(project)

    logger.info(
        "Merged %s label(s), %s state(s) and %s field(s); added %s history entr(ies)",
        report.count, states, fields, entries,
    )
    return MergeReport(
        labels_taken=report.count, states_taken=states, fields_taken=fields,
        entries_added=entries, archived=report.archived,
    )


def _label_to_save(item: LabelMerge, project: ProjectMerge):
    """Return the copy's label as something the ordinary save can write.

    The pixels are read rather than the file copied, so what lands here is
    written the way every other label is -- and so that a label that will not
    open stops this one item rather than the whole merge partway through.
    """
    from ..io.image import read_array
    from .label_save import LabelToSave, WriteProvenance

    try:
        pixels = read_array(item.their_path)
    except Exception as error:
        logger.warning("Not taking %s: %s", item.target, error)
        return None

    fields: dict[str, object] = {
        "resolution": item.outcome,
        "source_project_id": project.source_project_id,
        "source_event": item.source_event,
    }
    if item.ancestor:
        fields["parent_digest"] = item.ancestor
    item.our_path.parent.mkdir(parents=True, exist_ok=True)
    return LabelToSave(
        destination=item.our_path,
        label=pixels,
        created=not item.our_path.is_file(),
        fingerprint=item.our_fingerprint,
        corrected=item.corrected,
        provenance=WriteProvenance("merged", fields),
    )


def _take_state(
    item: StateMerge, recorder: Recorder, *, allow_corrected: bool = False
) -> None:
    """Declare the copy's state here, through the one operation that may.

    Leaving corrected is permitted only where the caller said so. Granting it
    whenever it happened to be needed made the guard describe the situation
    rather than protect against it.
    """
    from .selection import transition_records

    transition_records(
        [item.record],
        item.theirs,
        allow_leaving_corrected=allow_corrected,
        recorder=recorder,
    )


def _take_field(project: ProjectMerge, item: MetadataMerge, recorder: Recorder) -> None:
    """Write the copy's value for one metadata field."""
    from .metadata_edit import update_project_metadata

    update_project_metadata(
        project.our_root, {item.field: item.theirs}, recorder=recorder
    )


def _add_entries(project: ProjectMerge) -> int:
    """Append the copy's history entries this collection does not have.

    Union rather than reconciliation: an id is a hash of its content, so the
    same entry from both sides is one entry, and doing this twice changes
    nothing the second time.
    """
    added = 0
    for event in project.incoming:
        append_event(project.our_root, event)
        added += 1
    return added
