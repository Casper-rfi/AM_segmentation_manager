"""Workflow state changes for selected, deselected and corrected records.

State is declared in a stage's ``AM_state.yaml``, so changing it means editing
that file rather than moving anything. Three rules govern every function here.

**Validate the whole batch, then act.** Everything is located, checked and
resolved into per-stage writes before the first one happens, and every stage
file is re-checked against what the plan read immediately before writing. A
collection curated in two places at once therefore fails with nothing written
rather than half of it.

**A corrected group is never left without confirmation**: an operation that
would take one out of the corrected state refuses unless the caller passes
``allow_leaving_corrected``.

**A stage that has not been migrated is refused.** Declaring one group's state
there would leave every other group unlisted, and unlisted reads as selected --
promoting deselected images into a training set. Such a stage keeps its folders
until ``migrate-state`` transcribes them.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..io.atomic import fingerprint_of
from ..layout import (
    classify_image_directory,
    classify_loose_image,
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
    STATE_VALUES,
    RecordState,
    StageState,
    StaleState,
    read_state,
    state_key,
    write_state,
)
from .label_groups import expand_to_groups, group_identity

logger = logging.getLogger(__name__)

__all__ = [
    "RecordState",
    "StaleState",
    "TransitionPlan",
    "UnmigratedStage",
    "apply_plan",
    "current_state",
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
    The one place the package asks this question, so how a record carries its
    state stays a detail of :class:`~..models.ImageRecord`.
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
    The one place the package asks this question, so how a record carries its
    state stays a detail of :class:`~..models.ImageRecord`.
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
    stage_directory = identity.parent.parent
    return stage_directory, state_key(identity, stage_directory)


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
            "run migrate-state on it first"
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
    """
    plan = plan_transition(
        records, state, allow_leaving_corrected=allow_leaving_corrected
    )
    if dry_run:
        for line in plan.describe():
            logger.info("Would declare %s", line)
        return len(plan.records)
    return apply_plan(plan)


def apply_plan(plan: TransitionPlan) -> int:
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

    Returns:
        How many records were declared, counting the whole expanded set.

    Raises:
        StaleState: If any stage's file changed since the plan read it.
    """
    _refuse_if_stale(plan)

    for write in plan.writes:
        write_state(write.previous, write.states)
        for record in write.records:
            record.state = plan.state
    # Records whose group was already in the target state belong to no write.
    for record in plan.records:
        record.state = plan.state
    return len(plan.records)


def _refuse_if_stale(plan: TransitionPlan) -> None:
    """Raise unless every stage the plan writes is as the plan read it.

    Checked over the whole batch first, so the answer is "nothing happened"
    rather than "some of it happened"; :func:`~..state.write_state` checks the
    same thing again per file, which is what closes the remaining window.
    """
    changed = [
        write.previous.path
        for write in plan.writes
        if fingerprint_of(write.previous.path) != write.previous.fingerprint
    ]
    if changed:
        listed = ", ".join(str(path) for path in changed)
        raise StaleState(
            f"{len(changed)} state file(s) changed since this change was worked "
            f"out, so none of it was applied ({listed}). Something else is "
            "curating this collection; look again and repeat the change."
        )


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
    right label directory. An image sitting loose in its stage has no such
    directory, and is read from its filename instead.
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

    siblings = subdirectories_by_name(image_directory.parent)
    labels_directory, _, _ = resolve_labels_directory(
        image_directory, classification.modality, siblings
    )
    predictions_directory, _, _ = resolve_predictions_directory(
        image_directory, classification.modality, siblings
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
