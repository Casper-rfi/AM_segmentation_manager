"""Putting a selection of labels back to what they held before an operation.

Convert a project's predictions, find the model not good enough, and put the whole
project back in one command. **Only what an operation replaced can be put back**:
the history names the version each label held before the run, and the archive holds
it. A label whose version was never kept, or whose pixels are not here, is reported
and left alone, guessing at one being the failure this exists to prevent. Every
write is the ordinary save, which archives what it replaces, so undoing a revert is
another revert. **Returning to a savepoint restores a group's state with its label**,
only where the label ends up holding what it held then, and metadata where asked.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from ..history import Recorder, containing_project, events_for, read_log
from ..models import ImageRecord
from ..state import RecordState
from .archive import ArchivedVersion, label_versions
from .label_groups import LabelGroup, build_groups, group_identity
from .label_save import restore
from .metadata_edit import MetadataChange, plan_metadata_update, update_project_metadata
from .savepoint import ARCHIVED, LOST, TargetState
from .selection import current_state, plan_transition, transition_records

logger = logging.getLogger(__name__)

__all__ = [
    "UNDOABLE",
    "plan_revert_to",
    "RevertItem",
    "RevertPlan",
    "RevertReport",
    "StateItem",
    "apply_revert",
    "plan_revert",
]

#: The operations this can undo, and what each is called on the command line.
#: A conversion is the one that matters -- it is how a new model's guess gets
#: over proofread work -- and a merge is the same shape of change.
UNDOABLE: Final[dict[str, str]] = {
    "conversion": "label-converted",
    "merge": "merged",
}

#: Entries that rewrite a label's pixels: one after the operation being undone
#: is newer work, which putting the old version back would discard.
REWRITES: Final[frozenset[str]] = frozenset({
    "label-saved", "label-restored", "label-converted", "merged", "label-removed",
})

#: Why a label cannot be put back.
NO_ENTRY: Final[str] = "no such operation is recorded for it"
NOT_NAMED: Final[str] = "the savepoint says nothing about it"
UNREACHABLE: Final[str] = "no version holds what it had then"
ALREADY_THERE: Final[str] = "it already holds what it had then"
NOT_ARCHIVED: Final[str] = "the operation kept no earlier version"
NOT_HERE: Final[str] = "the earlier version's pixels are not in this collection"
SUPERSEDED: Final[str] = "it has been edited since, so there is newer work to lose"

#: Why a group's state is not restored with the rest.
LABEL_STAYS: Final[str] = "its label is not restored, so its state is kept with it"
NO_LABEL: Final[str] = "it has no label now to be marked corrected"

#: What becomes of a family correction changed since the savepoint.
FAMILY_KEPT: Final[str] = (
    "family correction changed since, not restored; use Families on Curate or edit-families"
)


@dataclass(frozen=True)
class StateItem:
    """One group whose state a return to a savepoint would restore, or could not.

    Attributes:
        target: Its key, relative to the project root, as its label's is.
        records: The images sharing it.
        now: The state it is in.
        then: The state it was in at the savepoint.
        blocked: Why it is left as it is, or empty where it is restored.
    """

    target: str
    records: tuple[ImageRecord, ...]
    now: RecordState
    then: RecordState
    blocked: str = ""


@dataclass(frozen=True)
class RevertItem:
    """One label a revert would put back, or could not.

    Attributes:
        label_path: The label as it is now.
        target: Its key, relative to the project root.
        version: The version to put back, or None where there is none.
        corrected: Whether its group is declared corrected.
        blocked: Why it cannot be put back, or empty where it can.
    """

    label_path: Path
    target: str
    version: ArchivedVersion | None
    corrected: bool
    blocked: str = ""

    @property
    def can_revert(self) -> bool:
        """Return whether this one would be put back."""
        return self.version is not None and not self.blocked


@dataclass(frozen=True)
class RevertPlan:
    """What putting a selection back would do, having done none of it.

    Attributes:
        undoing: The entry kind being undone.
        items: Every label considered, in a stable order.
    """

    undoing: str
    items: tuple[RevertItem, ...] = ()
    states: tuple[StateItem, ...] = ()
    fields: tuple[MetadataChange, ...] = ()
    families: tuple[str, ...] = ()

    @property
    def revertible(self) -> tuple[RevertItem, ...]:
        """Return the labels that would be put back."""
        return tuple(item for item in self.items if item.can_revert)

    @property
    def restorable_states(self) -> tuple[StateItem, ...]:
        """Return the groups whose state would be restored."""
        return tuple(item for item in self.states if not item.blocked)

    @property
    def leaving_corrected(self) -> tuple[StateItem, ...]:
        """Return the groups whose corrected state would be given up."""
        return tuple(item for item in self.restorable_states if item.now == "corrected")

    @property
    def corrected_targets(self) -> tuple[str, ...]:
        """Return the groups whose proofread label or corrected state this would replace."""
        return tuple(dict.fromkeys(
            [*(item.target for item in self.corrected),
             *(item.target for item in self.leaving_corrected)]
        ))

    @property
    def anything(self) -> bool:
        """Return whether applying would change anything."""
        return bool(self.revertible or self.restorable_states or self.fields)

    @property
    def blocked(self) -> tuple[RevertItem, ...]:
        """Return the labels that would be left alone, and why."""
        return tuple(item for item in self.items if item.blocked)

    @property
    def corrected(self) -> tuple[RevertItem, ...]:
        """Return the revertible labels whose group is proofread work."""
        return tuple(item for item in self.revertible if item.corrected)

    def describe(self) -> list[str]:
        """Return one line per label, saying what would happen to it."""
        lines = []
        for item in self.items:
            if item.can_revert and item.version is not None:
                when = item.version.event.at if item.version.event else "unrecorded"
                lines.append(f"{item.target}: back to the version from {when}")
            else:
                lines.append(f"{item.target}: left alone, {item.blocked}")
        for state in self.states:
            if state.blocked:
                lines.append(f"{state.target}: kept {state.now}, {state.blocked}")
            else:
                lines.append(f"{state.target}: {state.now} -> {state.then}")
        for change in self.fields:
            lines.extend(f"{change.project}: {line}" for line in change.describe())
        lines.extend(f"{named}: {FAMILY_KEPT}" for named in self.families)
        return lines


@dataclass(frozen=True)
class RevertReport:
    """What a revert actually put back."""

    restored: tuple[Path, ...] = ()
    failed: tuple[tuple[Path, str], ...] = ()
    states: int = 0
    fields: int = 0


def plan_revert(
    records: Iterable[ImageRecord], undoing: str, *, allow_superseded: bool = False
) -> RevertPlan:
    """Return what putting *records* back before *undoing* would do.

    Args:
        records: The images whose labels to consider. Several conditions
            sharing one label contribute it once.
        undoing: A key of :data:`UNDOABLE`.
        allow_superseded: Put a label back even where somebody has edited it
            since the operation being undone, which discards that editing.

    Raises:
        ValueError: If *undoing* is not something this can undo.
    """
    kind = UNDOABLE.get(undoing)
    if kind is None:
        raise ValueError(
            f"{undoing!r} is not something this can undo; try "
            f"{' or '.join(sorted(UNDOABLE))}"
        )


    items: list[RevertItem] = []
    logs: dict[Path, object] = {}
    for group in build_groups(list(records)):
        label_path = group.label_path
        if label_path is None:
            continue
        project = containing_project(label_path)
        if project is None:
            continue
        try:
            target = label_path.resolve().relative_to(project).as_posix()
        except ValueError:
            continue
        if project not in logs:
            logs[project] = read_log(project)

        version, blocked = _version_before(
            logs[project], project, target, kind, allow_superseded=allow_superseded
        )
        items.append(
            RevertItem(
                label_path=label_path,
                target=target,
                version=version,
                corrected=current_state(group.records[0]) == "corrected",
                blocked=blocked,
            )
        )
    items.sort(key=lambda item: item.target)
    logger.debug("Revert plan: %s of %s label(s) can go back",
                 sum(1 for item in items if item.can_revert), len(items))
    return RevertPlan(undoing=kind, items=tuple(items))


def _version_before(
    log, project: Path, target: str, kind: str, *, allow_superseded: bool
) -> tuple[ArchivedVersion | None, str]:
    """Return the version *kind* replaced for *target*, or why there is none."""
    entries = events_for(log, target)
    latest = None
    for index, event in enumerate(entries):
        if event.kind == kind:
            latest = index
    if latest is None:
        return None, NO_ENTRY

    # Anything after it changed the label again, so putting the old one back
    # would discard that too. Said rather than done.
    if not allow_superseded and any(
        event.kind in REWRITES for event in entries[latest + 1:]
    ):
        return None, SUPERSEDED

    archived = entries[latest].data.get("archived_as")
    if not isinstance(archived, str) or not archived:
        return None, NOT_ARCHIVED

    for version in label_versions(project, target):
        if version.path == project / archived:
            if not version.present:
                return None, NOT_HERE
            return version, ""
    return None, NOT_HERE


def apply_revert(
    plan: RevertPlan, *, recorder: Recorder, allow_corrected: bool = False
) -> RevertReport:
    """Put back every label, state and field *plan* can, or none of them.

    Raises:
        PermissionError: If any label or state is proofread work and *allow_corrected*
            was not set. Checked over the whole batch before the first write,
            so the answer is "nothing happened" rather than "some of it did".
    """
    if plan.corrected_targets and not allow_corrected:
        names = ", ".join(plan.corrected_targets[:3])
        raise PermissionError(
            f"Refusing to put back {len(plan.corrected_targets)} corrected label(s) "
            f"without explicit confirmation ({names})"
        )
    # Rehearsed whole before the first label is written, so a state or a field
    # that cannot be declared stops the return with nothing changed.
    by_state = _records_by_state(plan.restorable_states)
    for declared, records in by_state.items():
        plan_transition(records, declared, allow_leaving_corrected=allow_corrected)
    for change in plan.fields:
        plan_metadata_update(change.path.parent, _values(change))

    restored: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for item in plan.revertible:
        assert item.version is not None, "a revertible item has a version"
        try:
            restore(
                item.version, item.label_path,
                recorder=recorder,
                corrected=item.corrected,
                allow_corrected=allow_corrected,
            )
            restored.append(item.label_path)
        except Exception as error:  # noqa: BLE001 - one label failing must not stop the batch
            logger.warning("Could not restore %s: %s", item.target, error)
            failed.append((item.label_path, str(error)))

    if failed:
        unwritten = {path.resolve() for path, _ in failed}
        kept = [
            item for item in plan.restorable_states
            if _label_of(item) not in unwritten
        ]
        by_state = _records_by_state(kept)
    states = 0
    for declared, records in by_state.items():
        transition_records(
            records, declared, allow_leaving_corrected=allow_corrected, recorder=recorder
        )
        states += len({group_identity(record) for record in records})
    for change in plan.fields:
        update_project_metadata(change.path.parent, _values(change), recorder=recorder)

    logger.info(
        "Put %s label(s), %s state(s) and %s project(s)' metadata back",
        len(restored), states, len(plan.fields),
    )
    return RevertReport(
        restored=tuple(restored), failed=tuple(failed), states=states, fields=len(plan.fields)
    )


def _label_of(item: StateItem) -> Path | None:
    """Return the label *item*'s group reads now, resolved, or None where it has none."""
    label = next((record.labels[0] for record in item.records if record.labels), None)
    return label.resolve() if label is not None else None


def _records_by_state(items: Iterable[StateItem]) -> dict[RecordState, list[ImageRecord]]:
    """Return the images of *items*, by the state each is to be declared in."""
    by_state: dict[RecordState, list[ImageRecord]] = {}
    for item in items:
        by_state.setdefault(item.then, []).extend(item.records)
    return by_state


def _values(change: MetadataChange) -> dict[str, str | None]:
    """Return the edits that make *change*, as :func:`plan_metadata_update` takes them."""
    return {edit.field: edit.new for edit in change.changes}


# ---------------------------------------------------------------------------
# Back to a named state
# ---------------------------------------------------------------------------

def plan_revert_to(
    records: Iterable[ImageRecord], savepoint, *, metadata: bool = False
) -> RevertPlan:
    """Return what putting *records* back to *savepoint* would do.

    The counterpart of undoing one operation: a round of work is undone by name
    rather than by finding the run that started it. Which version each label
    needs is read from the savepoint's own reachability, so this asks nothing
    the listing has not already answered -- including of the labels it cannot
    reach, which are reported here rather than discovered partway through.

    A label already holding what it held then is left alone, not rewritten:
    there is nothing to put back, and a write would archive a copy of nothing.

    A group's state goes back where its label holds what it held then, put
    back now or never changed; a label left as it is keeps its state with it,
    as a merge keeps a label and its state one decision.

    Args:
        savepoint: A :class:`~.savepoint.Savepoint`.
        metadata: Also restore the project metadata of the projects *records*
            reach.
    """

    named: dict[Path, dict[str, TargetState]] = {}
    then: dict[Path, Mapping[str, RecordState]] = {}
    for carried in savepoint.projects:
        root = carried.project.path.resolve()
        named[root] = {state.target: state for state in carried.targets}
        then[root] = carried.states

    items: list[RevertItem] = []
    # Keyed by project as well: a target is relative to its project, and two
    # projects laid out alike name their labels alike.
    states: list[tuple[Path, StateItem]] = []
    reached: set[Path] = set()
    for group in build_groups(list(records)):
        project = containing_project(group.identity) if group.identity else None
        if project is None or project not in named:
            continue
        reached.add(project)
        state_item = _state_item(group, project, then[project])
        if state_item is not None:
            states.append((project, state_item))
        label_path = group.label_path
        if label_path is None:
            continue
        try:
            target = label_path.resolve().relative_to(project).as_posix()
        except ValueError:
            continue

        corrected = current_state(group.records[0]) == "corrected"
        state = named[project].get(target)
        if state is None:
            items.append(RevertItem(label_path, target, None, corrected, NOT_NAMED))
            continue
        if state.where == LOST:
            items.append(RevertItem(label_path, target, None, corrected, UNREACHABLE))
            continue
        if state.where != ARCHIVED:
            items.append(RevertItem(label_path, target, None, corrected, ALREADY_THERE))
            continue

        wanted = project / str(state.archived_as)
        version = next(
            (
                candidate
                for candidate in label_versions(project, target)
                if candidate.path == wanted and candidate.present
            ),
            None,
        )
        items.append(
            RevertItem(
                label_path=label_path, target=target, version=version,
                corrected=corrected,
                blocked="" if version is not None else NOT_HERE,
            )
        )

    labels = {item.label_path.resolve(): item for item in items}
    restoring = sorted(
        (
            _with_its_label(state, labels.get((project / state.target).resolve()))
            for project, state in states
        ),
        key=lambda state: state.target,
    )
    items.sort(key=lambda item: item.target)
    fields = tuple(
        change
        for carried in savepoint.projects
        if metadata and carried.fields and carried.project.path.resolve() in reached
        if not (
            change := plan_metadata_update(carried.project.path, dict(carried.fields))
        ).is_empty
    )
    logger.debug(
        "Revert to %s: %s of %s label(s) and %s state(s) can go back",
        savepoint.id, sum(1 for item in items if item.can_revert), len(items),
        sum(1 for state in restoring if not state.blocked),
    )
    families = tuple(
        f"{carried.project.key}: {named}"
        for carried in savepoint.projects
        if carried.project.path.resolve() in reached
        for named in carried.families
    )
    return RevertPlan(
        undoing=f"savepoint {savepoint.title or savepoint.id}",
        items=tuple(items), states=tuple(restoring), fields=fields, families=families,
    )


def _state_item(
    group: LabelGroup, project: Path, then: Mapping[str, RecordState]
) -> StateItem | None:
    """Return the state *group* would go back to, or None where it has not changed since."""
    assert group.identity is not None, "a group in a project has an identity"
    try:
        target = group.identity.resolve().relative_to(project).as_posix()
    except ValueError:
        return None
    was = then.get(target)
    now = current_state(group.records[0])
    if was is None or was == now:
        return None
    blocked = NO_LABEL if was == "corrected" and group.label_path is None else ""
    return StateItem(target, group.records, now, was, blocked)


def _with_its_label(state: StateItem, label: RevertItem | None) -> StateItem:
    """Return *state*, blocked where its label will not hold what it held then."""
    if state.blocked or label is None:
        return state
    if label.can_revert or label.blocked == ALREADY_THERE:
        return state
    return replace(state, blocked=LABEL_STAYS)
