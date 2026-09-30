"""Workflow state changes for selected, deselected and corrected records.

State is declared in a stage's ``AM_state.yaml``, so a change edits that file and
moves nothing. **Validate the whole batch, then act**: every stage file is re-checked
against what the plan read before the first write. **A corrected group is never left
without** ``allow_leaving_corrected``, and **a write that could replace its label asks
the file, not the record** (:func:`declared_states`). **An unmigrated stage is
refused** until ``scripts/migrate_state.py`` transcribes its folders.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..families import families_for
from ..history import Recorder
from ..io.atomic import fingerprint_of
from ..layout import (
    classify_image_directory,
    classify_loose_image,
    condition_of,
    image_files,
    resolve_labels_directory,
    resolve_loose_labels_directory,
    resolve_loose_predictions_directory,
    resolve_predictions_directory,
    state_directories,
    subdirectories_by_name,
)
from ..models import ImageRecord
from ..state import (
    DEFAULT_STATE,
    STATE_VALUES,
    RecordState,
    StageState,
    StaleState,
    StateFileError,
    read_state,
    state_key,
    write_state,
)
from .label_groups import expand_to_groups, group_identity

logger = logging.getLogger(__name__)

__all__ = [
    "RecordState",
    "Regrouping",
    "RegroupingPlan",
    "StaleState",
    "StateEffect",
    "StateFileError",
    "TransitionPlan",
    "UnmigratedStage",
    "apply_plan",
    "apply_regrouping",
    "plan_regrouping",
    "refuse_stale_regrouping",
    "current_state",
    "declared_states",
    "plan_transition",
    "transition_records",
    "deselect_records",
    "select_records",
    "mark_corrected_records",
    "record_from_image_path",
]


class UnmigratedStage(ValueError):
    """Raised when a stage's folders still hold its state.

    A :class:`ValueError` so that callers already handling one keep working:
    from their point of view this is a request that cannot be carried out.
    """


@dataclass(frozen=True)
class StateChange:
    """One label group moving from one state to another.

    Attributes:
        stage_directory: Where the state file declaring it lives.
        key: The group's entry in that file.
        old: The state it is in now.
        new: The state it is moving to.
    """

    stage_directory: Path
    key: str
    old: RecordState
    new: RecordState

    def describe(self) -> str:
        """Return a one-line description, for a preview or a log."""
        return f"{self.key}: {self.old} -> {self.new}"


@dataclass(frozen=True)
class _StageWrite:
    """Everything one stage's file should declare after the change.

    Attributes:
        previous: The file as it was read, holding the fingerprint the write
            refuses to go ahead without.
        states: What the file should declare afterwards.
        records: The records this stage's write accounts for, so each one is
            brought into step as its own stage lands rather than after the
            whole batch.
    """

    previous: StageState
    states: dict[str, RecordState]
    records: tuple[ImageRecord, ...] = ()


def current_state(record: ImageRecord) -> RecordState:
    """Return the workflow state *record* is in.

    Read from the record rather than from the filesystem: the scanner sets it
    from the stage's declared state, and a successful transition updates it.
    The one place the package asks this question of a record, so how a record
    carries its state stays a detail of :class:`~..models.ImageRecord`; a write
    about to replace a label asks :func:`declared_states` instead.
    """
    return record.state


@dataclass(frozen=True)
class TransitionPlan:
    """A validated, not-yet-performed state change.

    Attributes:
        state: The state everything moves into.
        records: Every record the change applies to, after expanding across
            shared labels. This can be larger than what the caller asked for.
        added: The records that expansion added, so a caller can say how far
            the change actually reaches before performing it.
        changes: One entry per group whose state actually changes. A record
            already in the target state contributes none.
    """

    state: RecordState
    records: tuple[ImageRecord, ...]
    added: tuple[ImageRecord, ...]
    changes: tuple[StateChange, ...]
    #: What to write, per stage, computed before anything is written.
    writes: tuple[_StageWrite, ...] = field(default_factory=tuple)

    @property
    def reaches_beyond_request(self) -> bool:
        """Return whether a shared label pulled in images the caller did not name."""
        return bool(self.added)

    def describe(self) -> list[str]:
        """Return one line per changed group, for a preview or a log."""
        return [change.describe() for change in self.changes]


def _stage_and_key(record: ImageRecord) -> tuple[Path, str]:
    """Return the state file and entry naming *record*'s group.

    Raises:
        ValueError: If the image sits in no recognised image directory, and so
            belongs to no stage whose state could be declared.
    """
    identity = group_identity(record)
    if identity is None:
        raise ValueError(
            f"{record.image_path} is not in a recognised image directory, "
            "so its workflow state cannot be declared"
        )
    return _stage_and_key_of_label(identity)


def _stage_and_key_of_label(label_path: Path) -> tuple[Path, str]:
    """Return the stage declaring the group whose label is *label_path*, and its key.

    A label directory sits directly in its stage, whether its images fill a
    directory beside it or sit loose in the stage itself.
    """
    stage_directory = label_path.parent.parent
    return stage_directory, state_key(label_path, stage_directory)


def declared_states(label_paths: Iterable[Path]) -> dict[Path, RecordState]:
    """Return the state each label's group is declared in now, read from disk.

    :func:`current_state` answers from the record, which holds what the file
    said at the last scan; this is the answer for a write about to replace a
    label, which must not trust that. Each stage's file is read once.

    Args:
        label_paths: Labels as :func:`group_identity` names them -- where each
            group's label is or would be.

    Raises:
        StateFileError: If a stage's file cannot be read. A write that cannot
            tell whether a label is corrected must not go ahead.
    """
    stages: dict[Path, StageState] = {}
    declared: dict[Path, RecordState] = {}
    for label_path in label_paths:
        stage_directory, key = _stage_and_key_of_label(label_path)
        if stage_directory not in stages:
            stages[stage_directory] = read_state(stage_directory)
        declared[label_path] = stages[stage_directory].state_of(key)
    return declared


def _has_nested_files(stage_directory: Path) -> bool:
    """Return whether any file still sits in one of *stage_directory*'s state folders."""
    for directory in subdirectories_by_name(stage_directory).values():
        for candidate in state_directories(directory)[1:]:
            if image_files(candidate):
                return True
    return False


def _stage_state(stage_directory: Path) -> StageState:
    """Return a stage's declared state, refusing one that has not been migrated.

    Raises:
        UnmigratedStage: If the stage has no state file but does have files in
            its state folders, which are then the only record of what was
            decided about them.
    """
    declared = read_state(stage_directory)
    if declared.fingerprint is None and _has_nested_files(stage_directory):
        raise UnmigratedStage(
            f"{stage_directory} still keeps its workflow state in folders; "
            "run scripts/migrate_state.py on it first"
        )
    return declared


def plan_transition(
    records: Iterable[ImageRecord],
    state: RecordState,
    *,
    allow_leaving_corrected: bool = False,
) -> TransitionPlan:
    """Validate a state change without performing it.

    The whole batch is checked before anything is written: every image is
    located, every stage is checked for having been migrated, and corrected
    data is refused unless explicitly confirmed. The same validation runs
    whether or not the plan is carried out, so a dry run is a genuine rehearsal.

    Records are expanded across shared labels first. Declaring a state for a
    shared label without covering the images that share it would silently
    change their state too.

    Raises:
        ValueError: If *state* is not a workflow state, or a record belongs to
            no recognised image directory.
        UnmigratedStage: If a stage still keeps its state in folders.
        FileNotFoundError: If a record's image is not on disk.
        PermissionError: If corrected data would be changed without confirmation.
    """
    if state not in STATE_VALUES:
        raise ValueError(f"Unknown workflow state: {state}")

    requested = list(records)
    expanded = expand_to_groups(requested)
    requested_paths = {str(record.image_path) for record in requested}
    added = tuple(
        record for record in expanded
        if str(record.image_path) not in requested_paths
    )

    stage_states: dict[Path, StageState] = {}
    pending: dict[Path, dict[str, RecordState]] = {}
    stage_records: dict[Path, list[ImageRecord]] = {}
    changes: list[StateChange] = []
    seen_keys: set[tuple[Path, str]] = set()

    for record in expanded:
        if not record.image_path.exists():
            raise FileNotFoundError(record.image_path)

        stage_directory, key = _stage_and_key(record)
        if stage_directory not in stage_states:
            stage_states[stage_directory] = _stage_state(stage_directory)
            pending[stage_directory] = dict(stage_states[stage_directory].states)
        stage_records.setdefault(stage_directory, []).append(record)

        if (stage_directory, key) in seen_keys:
            continue
        seen_keys.add((stage_directory, key))

        old = stage_states[stage_directory].state_of(key)
        if old == state:
            continue
        pending[stage_directory][key] = state
        changes.append(StateChange(stage_directory, key, old, state))

    if not allow_leaving_corrected:
        leaving_corrected = [
            change.key for change in changes if change.old == "corrected"
        ]
        if leaving_corrected:
            raise PermissionError(
                "Refusing to change corrected artifacts without explicit "
                f"confirmation: {leaving_corrected}"
            )

    changed_stages = {change.stage_directory for change in changes}
    writes = tuple(
        _StageWrite(
            previous=stage_states[stage_directory],
            states=states,
            records=tuple(stage_records.get(stage_directory, ())),
        )
        for stage_directory, states in pending.items()
        if stage_directory in changed_stages
    )
    return TransitionPlan(
        state=state,
        records=tuple(expanded),
        added=added,
        changes=tuple(changes),
        writes=writes,
    )


def transition_records(
    records: Iterable[ImageRecord],
    state: RecordState,
    *,
    allow_leaving_corrected: bool = False,
    dry_run: bool = False,
    recorder: Recorder | None = None,
) -> int:
    """Declare *records* to be in ``state`` after validating the whole batch.

    The states are mutually exclusive, so this replaces whatever each group was
    in. Selecting can restore a corrected group, but requires
    ``allow_leaving_corrected=True`` to protect proofread labels.

    Records sharing a label change together: the label is the unit of
    proofreading, so a state change applies to every image of the same scene.
    The return value counts the expanded set, which can exceed what was asked
    for; use :func:`plan_transition` when the difference needs reporting first.

    On success each record is updated, so callers that keep records alive across
    several operations -- a long-running UI, for example -- stay in step. No
    file moves, so the paths a record holds stay valid either way.

    Args:
        records: The records to change.
        state: The state to declare.
        allow_leaving_corrected: Confirm taking data out of the corrected
            state.
        dry_run: Validate everything and report the count without writing.
        recorder: Where to record the changes, or None to record nothing.
    """
    plan = plan_transition(
        records, state, allow_leaving_corrected=allow_leaving_corrected
    )
    if dry_run:
        for line in plan.describe():
            logger.info("Would declare %s", line)
        return len(plan.records)
    return apply_plan(plan, recorder=recorder)


def apply_plan(plan: TransitionPlan, *, recorder: Recorder | None = None) -> int:
    """Carry out a validated plan, or none of it.

    Every stage file is compared against what the plan read before any of them
    is written, so a collection being curated somewhere else at the same time
    stops the batch with nothing changed. That check is what lets a plan be
    shown to somebody, waited on, and then carried out: the decisions are the
    ones that were previewed, while the safety check is taken fresh.

    This is not a transaction. Between the check and the writes there is a
    window of microseconds, against the seconds or minutes a preview takes, and
    a write failing for a reason no check can anticipate -- a full disk -- can
    still stop the batch partway. Each stage's records are brought into step as
    that stage lands rather than at the end, so what is in memory describes what
    is on disk either way.

    Args:
        plan: A plan from :func:`plan_transition`.
        recorder: Where to record the changes, or None to record nothing. A
            stage is recorded as it lands, so a run stopped by a full disk has
            recorded what reached the disk and nothing more.

    Returns:
        How many records were declared, counting the whole expanded set.

    Raises:
        StaleState: If any stage's file changed since the plan read it.
    """
    _refuse_if_stale(plan.writes)

    by_stage: dict[Path, list[StateChange]] = {}
    if recorder is not None:
        for change in plan.changes:
            by_stage.setdefault(change.stage_directory, []).append(change)

    for write in plan.writes:
        write_state(write.previous, write.states)
        for record in write.records:
            record.state = plan.state
        if recorder is not None:
            for change in by_stage.get(write.previous.path.parent, ()):
                recorder.record(
                    change.stage_directory / change.key,
                    "state-changed",
                    old=change.old,
                    new=change.new,
                )
    # Records whose group was already in the target state belong to no write.
    for record in plan.records:
        record.state = plan.state
    return len(plan.records)


def _refuse_if_stale(writes: Iterable[_StageWrite]) -> None:
    """Raise unless every stage a plan writes is as the plan read it.

    Checked over the whole batch first, so the answer is "nothing happened"
    rather than "some of it happened"; :func:`~..state.write_state` checks the
    same thing again per file, which is what closes the remaining window.
    """
    changed = [
        write.previous.path
        for write in writes
        if fingerprint_of(write.previous.path) != write.previous.fingerprint
    ]
    if changed:
        listed = ", ".join(str(path) for path in changed)
        raise StaleState(
            f"{len(changed)} state file(s) changed since this change was worked "
            f"out, so none of it was applied ({listed}). Something else is "
            "curating this collection; look again and repeat the change."
        )


# ---------------------------------------------------------------------------
# Regrouping: images whose label group changes with their family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Regrouping:
    """One condition's images in one stage leaving a label group for another.

    Attributes:
        image_directory: The condition's directory, directly inside its stage.
        image_names: Its images, by filename.
        old_labels: The label directory the images read before.
        new_labels: The label directory they read after.
        old_group_remains: Whether another condition still reads *old_labels*.
        new_group_exists: Whether another condition already read *new_labels*.
    """

    image_directory: Path
    image_names: tuple[str, ...]
    old_labels: Path
    new_labels: Path
    old_group_remains: bool
    new_group_exists: bool


@dataclass(frozen=True)
class StateEffect:
    """One image whose workflow state a regrouping changes."""

    image: Path
    old: RecordState
    new: RecordState


@dataclass(frozen=True)
class RegroupingPlan:
    """The state entries a regrouping writes, validated and not yet written.

    Attributes:
        changes: One entry per state-file key written or dropped.
        effects: One entry per image whose state changes.
    """

    changes: tuple[StateChange, ...] = ()
    effects: tuple[StateEffect, ...] = ()
    writes: tuple[_StageWrite, ...] = field(default_factory=tuple)


def plan_regrouping(regroupings: Iterable[Regrouping]) -> RegroupingPlan:
    """Return the state entries that keep images' curation through a regrouping.

    **An image joining a group other images already share takes that group's
    state**, a group being in exactly one. **Images forming a new group carry
    ``deselected`` but not ``corrected``**: the first is a decision about the
    images, the second a claim about a label they no longer share; where the
    images carried in disagree, ``deselected`` wins. An entry nothing will read
    any more is dropped rather than left behind.

    Raises:
        UnmigratedStage: If a stage still keeps its state in folders.
        StateFileError: If a stage's file cannot be read.
    """
    stage_states: dict[Path, StageState] = {}
    #: Per image: its stage, the key it reads after, and the state it had.
    moved: list[tuple[Path, Path, str, RecordState]] = []
    carried: dict[tuple[Path, str], RecordState] = {}
    dropped: dict[tuple[Path, str], None] = {}

    for regrouping in regroupings:
        stage = regrouping.image_directory.parent
        if stage not in stage_states:
            stage_states[stage] = _stage_state(stage)
        declared = stage_states[stage]
        for name in regrouping.image_names:
            old_key = state_key(regrouping.old_labels / name, stage)
            new_key = state_key(regrouping.new_labels / name, stage)
            old = declared.state_of(old_key)
            moved.append((stage, regrouping.image_directory / name, new_key, old))
            if not regrouping.old_group_remains:
                dropped[(stage, old_key)] = None
            if regrouping.new_group_exists:
                continue
            if old == "deselected" or (stage, new_key) not in carried:
                carried[(stage, new_key)] = "deselected" if old == "deselected" else DEFAULT_STATE

    declared_after = {stage: dict(state.states) for stage, state in stage_states.items()}
    for stage, key in dropped:
        declared_after[stage].pop(key, None)
    for (stage, key), state in carried.items():
        if state == DEFAULT_STATE:
            declared_after[stage].pop(key, None)
        else:
            declared_after[stage][key] = state

    changes = [
        StateChange(stage, key, stage_states[stage].state_of(key), after.get(key, DEFAULT_STATE))
        for stage, after in declared_after.items()
        for key in sorted(set(stage_states[stage].states) | set(after))
        if stage_states[stage].state_of(key) != after.get(key, DEFAULT_STATE)
    ]
    effects = [
        StateEffect(image, old, declared_after[stage].get(new_key, DEFAULT_STATE))
        for stage, image, new_key, old in moved
        if declared_after[stage].get(new_key, DEFAULT_STATE) != old
    ]
    changed = {change.stage_directory for change in changes}
    return RegroupingPlan(
        changes=tuple(changes),
        effects=tuple(effects),
        writes=tuple(
            _StageWrite(previous=stage_states[stage], states=declared_after[stage])
            for stage in stage_states
            if stage in changed
        ),
    )


def refuse_stale_regrouping(plan: RegroupingPlan) -> None:
    """Raise unless every stage *plan* writes is as it was read.

    Raises:
        StaleState: If any stage's file changed since the plan read it.
    """
    _refuse_if_stale(plan.writes)


def apply_regrouping(plan: RegroupingPlan, *, recorder: Recorder | None = None) -> int:
    """Write the state entries of a validated regrouping, or none of them.

    Returns:
        How many entries were written or dropped.

    Raises:
        StaleState: If any stage's file changed since the plan read it.
    """
    _refuse_if_stale(plan.writes)
    for write in plan.writes:
        write_state(write.previous, write.states)
    if recorder is not None:
        for change in plan.changes:
            recorder.record(
                change.stage_directory / change.key, "state-changed",
                old=change.old, new=change.new,
            )
    return len(plan.changes)


def deselect_records(
    records: Iterable[ImageRecord],
    *,
    allow_leaving_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into the deselected state."""
    return transition_records(
        records, "deselected",
        allow_leaving_corrected=allow_leaving_corrected, dry_run=dry_run,
    )


def select_records(
    records: Iterable[ImageRecord],
    *,
    allow_leaving_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into the normal selected state."""
    return transition_records(
        records, "selected",
        allow_leaving_corrected=allow_leaving_corrected, dry_run=dry_run,
    )


def mark_corrected_records(
    records: Iterable[ImageRecord],
    *,
    allow_leaving_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into the corrected state."""
    return transition_records(
        records, "corrected",
        allow_leaving_corrected=allow_leaving_corrected, dry_run=dry_run,
    )


def record_from_image_path(path: Path) -> ImageRecord:
    """Build a record for direct-path use, including an associated label.

    The image directory is found by walking up from the file. Resolution goes
    through :mod:`am_segmentation_manager.layout`, so a path in any supported
    layout -- including a variant directory such as ``sem_5kv/`` -- finds the
    right label directory, under the family its project's corrections give. An
    image sitting loose in its stage has no such directory, and is read from its
    filename instead.

    Raises:
        FamiliesFileError: If the project's families file will not read.
    """
    image_path = path
    image_directory: Path | None = None
    classification = None
    for parent in image_path.parents:
        classification = classify_image_directory(parent.name)
        if classification is not None:
            image_directory = parent
            break

    if image_directory is None or classification is None:
        return _loose_record(image_path)

    corrections = families_for(image_directory)
    classification = condition_of(image_directory.name, corrections) or classification
    siblings = subdirectories_by_name(image_directory.parent)
    labels_directory, _, _ = resolve_labels_directory(
        image_directory, classification.modality, siblings, corrections=corrections
    )
    predictions_directory, _, _ = resolve_predictions_directory(
        image_directory, classification.modality, siblings, corrections=corrections
    )
    labels: list[Path] = []
    label_path = labels_directory / image_path.name
    if label_path.exists():
        labels.append(label_path)
    return ImageRecord(
        image_path=image_path,
        labels=labels,
        modality=classification.modality,
        variant=classification.variant,
        family=classification.family,
        labels_directory=labels_directory,
        predictions_directory=predictions_directory,
    )


def _loose_record(image_path: Path) -> ImageRecord:
    """Return a record for an image sitting loose in its stage directory.

    Reference, alignment and the prototype's Adaptive Polishing images have no
    image directory to take a modality or a label directory from; the filename
    states both, and :func:`~am_segmentation_manager.layout.classify_loose_image`
    reads it. An image whose name says nothing gets a bare record, as before:
    nothing here would be a fact rather than a guess.
    """
    classified = classify_loose_image(image_path.name)
    if classified is None or classified.modality is None:
        return ImageRecord(image_path=image_path)

    siblings = subdirectories_by_name(image_path.parent)
    labels_directory, _ = resolve_loose_labels_directory(
        image_path.parent, classified.series, classified.modality, siblings
    )
    predictions_directory, _ = resolve_loose_predictions_directory(
        image_path.parent, classified.series, classified.modality, siblings
    )
    label_path = labels_directory / image_path.name
    return ImageRecord(
        image_path=image_path,
        labels=[label_path] if label_path.exists() else [],
        modality=classified.modality,
        series=classified.series,
        labels_directory=labels_directory,
        predictions_directory=predictions_directory,
        stage_path=image_path.parent,
    )
