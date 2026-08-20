"""Workflow state changes for selected, deselected and corrected records.

State is declared in a stage's ``AM_state.yaml``, so changing it means editing
that file rather than moving anything. Three rules govern every function here.

**Validate the whole batch, then act**, so a batch cannot fail partway and leave
a collection in a state neither the caller nor the files describe.

**A corrected group is never left without confirmation**: an operation that
would take one out of the corrected state refuses unless the caller passes
``allow_move_from_corrected``.

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

from ..layout import (
    RecordState,
    base_path,
    classify_image_directory,
    image_files,
    resolve_labels_directory,
    state_directories,
    state_of_path,
    state_path,
    state_variants,
    subdirectories_by_name,
)
from ..models import ImageRecord
from ..state import (
    STATE_VALUES,
    StageState,
    read_state,
    state_key,
    write_state,
)
from .label_groups import (
    LabelGroup,
    describe_group,
    expand_to_groups,
    find_inconsistent_groups,
    group_identity,
)
from .labels import move_to_state

logger = logging.getLogger(__name__)

__all__ = [
    "RecordState",
    "TransitionPlan",
    "UnmigratedStage",
    "base_path",
    "current_state",
    "existing_variant",
    "plan_transition",
    "repair_group_states",
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
    """Everything one stage's file should declare after the change."""

    previous: StageState
    states: dict[str, RecordState]


def current_state(record: ImageRecord) -> RecordState:
    """Return the workflow state *record* is in.

    Read from the record rather than from the filesystem: the scanner sets it
    from the stage's declared state, and a successful transition updates it.
    """
    if record.corrected:
        return "corrected"
    return "selected" if record.selected else "deselected"


def existing_variant(path: Path) -> Path | None:
    """Find a file's sole on-disk state variant, rejecting ambiguous copies."""
    existing = [candidate for candidate in state_variants(path) if candidate.exists()]
    if len(existing) > 1:
        raise FileExistsError(f"Multiple state copies found for {base_path(path)}: {existing}")
    return existing[0] if existing else None


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
    allow_move_from_corrected: bool = False,
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
        FileNotFoundError: If a record's image cannot be found in any state.
        FileExistsError: If a file sits in more than one state folder at once.
        PermissionError: If corrected data would be changed without confirmation.
    """
    if state not in STATE_VALUES:
        raise ValueError(f"Unknown workflow state: {state}")

    requested = list(records)
    expanded = expand_to_groups(requested)
    requested_paths = {str(base_path(record.image_path)) for record in requested}
    added = tuple(
        record for record in expanded
        if str(base_path(record.image_path)) not in requested_paths
    )

    stage_states: dict[Path, StageState] = {}
    pending: dict[Path, dict[str, RecordState]] = {}
    changes: list[StateChange] = []
    seen_keys: set[tuple[Path, str]] = set()

    for record in expanded:
        if existing_variant(record.image_path) is None:
            raise FileNotFoundError(record.image_path)

        stage_directory, key = _stage_and_key(record)
        if stage_directory not in stage_states:
            stage_states[stage_directory] = _stage_state(stage_directory)
            pending[stage_directory] = dict(stage_states[stage_directory].states)

        if (stage_directory, key) in seen_keys:
            continue
        seen_keys.add((stage_directory, key))

        old = stage_states[stage_directory].state_of(key)
        if old == state:
            continue
        pending[stage_directory][key] = state
        changes.append(StateChange(stage_directory, key, old, state))

    if not allow_move_from_corrected:
        leaving_corrected = [
            change.key for change in changes if change.old == "corrected"
        ]
        if leaving_corrected:
            raise PermissionError(
                "Refusing to change corrected artifacts without explicit "
                f"confirmation: {leaving_corrected}"
            )

    writes = tuple(
        _StageWrite(previous=stage_states[stage_directory], states=states)
        for stage_directory, states in pending.items()
        if any(change.stage_directory == stage_directory for change in changes)
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
    allow_move_from_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Declare *records* to be in ``state`` after validating the whole batch.

    The states are mutually exclusive, so this replaces whatever each group was
    in. Selecting can restore a corrected group, but requires
    ``allow_move_from_corrected=True`` to protect proofread labels.

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
        allow_move_from_corrected: Confirm taking data out of the corrected
            state.
        dry_run: Validate everything and report the count without writing.
    """
    plan = plan_transition(
        records, state, allow_move_from_corrected=allow_move_from_corrected
    )
    if dry_run:
        for line in plan.describe():
            logger.info("Would declare %s", line)
        return len(plan.records)

    for write in plan.writes:
        write_state(write.previous, write.states)
    for record in plan.records:
        record.selected = state == "selected"
        record.corrected = state == "corrected"
    return len(plan.records)


def deselect_records(
    records: Iterable[ImageRecord],
    *,
    allow_move_from_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into the deselected state."""
    return transition_records(
        records, "deselected",
        allow_move_from_corrected=allow_move_from_corrected, dry_run=dry_run,
    )


def select_records(
    records: Iterable[ImageRecord],
    *,
    allow_move_from_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into the normal selected state."""
    return transition_records(
        records, "selected",
        allow_move_from_corrected=allow_move_from_corrected, dry_run=dry_run,
    )


def mark_corrected_records(
    records: Iterable[ImageRecord],
    *,
    allow_move_from_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into the corrected state."""
    return transition_records(
        records, "corrected",
        allow_move_from_corrected=allow_move_from_corrected, dry_run=dry_run,
    )


def repair_group_states(
    records: Iterable[ImageRecord],
    *,
    allow_move_from_corrected: bool = False,
    dry_run: bool = False,
) -> list[tuple[LabelGroup, int]]:
    """Move images to agree with the state of the label they share.

    This is about where files sit, not about declared state: it exists to make
    a stage's folders self-consistent so ``migrate-state`` can transcribe them.
    The label is authoritative, so repair moves the *images* to match it and
    never the other way round -- the label holds the proofreading work.

    A group becomes inconsistent when files are moved by something other than
    this package, or when an operation is interrupted partway.

    Args:
        records: Records to check. Their groups are read from disk, so passing
            one image of a group is enough.
        allow_move_from_corrected: Confirm moving images out of a corrected
            folder, which is needed when a group's label is *not* corrected but
            one of its images sits in ``corrected/``.
        dry_run: Report what would move without moving anything.

    Returns:
        One ``(group, moved_count)`` pair per repaired group, in the order the
        groups were found. Empty when everything already agrees.
    """
    repaired: list[tuple[LabelGroup, int]] = []
    for group in find_inconsistent_groups(records):
        state = group.state()
        disagreeing = group.inconsistent_records()
        logger.info(
            "Group %s is %s; moving %d image(s) to match",
            describe_group(group), state, len(disagreeing),
        )
        for record in disagreeing:
            source = existing_variant(record.image_path)
            if source is None:
                raise FileNotFoundError(record.image_path)
            # Checked here as well as in move_to_state, so a dry run refuses
            # what the real run would refuse rather than previewing it.
            if (
                state_of_path(source) == "corrected"
                and state != "corrected"
                and not allow_move_from_corrected
            ):
                raise PermissionError(
                    "Refusing to move corrected artifact without explicit "
                    f"confirmation: {source}"
                )
            destination = state_path(source, state)
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {destination}"
                )
            if not dry_run:
                move_to_state(
                    source, state,
                    allow_move_from_corrected=allow_move_from_corrected,
                )
                record.image_path = destination
        repaired.append((group, len(disagreeing)))
    return repaired


def record_from_image_path(path: Path) -> ImageRecord:
    """Build a record for direct-path use, including an associated label.

    The image directory is found by walking up from the file, so a path inside
    a state subfolder resolves the same as one beside it. Resolution goes
    through :mod:`am_segmentation_manager.layout`, so a path in any supported
    layout -- including a variant directory such as ``sem_5kv/`` -- finds the
    right label directory.
    """
    image_path = base_path(path)
    image_directory: Path | None = None
    classification = None
    for parent in image_path.parents:
        classification = classify_image_directory(parent.name)
        if classification is not None:
            image_directory = parent
            break

    if image_directory is None or classification is None:
        return ImageRecord(image_path=image_path)

    labels_directory, _, _ = resolve_labels_directory(
        image_directory,
        classification.modality,
        subdirectories_by_name(image_directory.parent),
    )
    labels: list[Path] = []
    label_path = labels_directory / image_path.name
    if existing_variant(label_path) is not None:
        labels.append(label_path)
    return ImageRecord(
        image_path=image_path,
        labels=labels,
        modality=classification.modality,
        variant=classification.variant,
    )
