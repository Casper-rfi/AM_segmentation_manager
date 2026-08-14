"""Filesystem transitions for selected, deselected, and corrected records."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .label_groups import (
    LabelGroup,
    describe_group,
    expand_to_groups,
    find_inconsistent_groups,
)
from .labels import STATE_DIRECTORIES, move_to_state
from ..layout import (
    RecordState,
    base_path,
    classify_image_directory,
    resolve_labels_directory,
    state_of_path,
    state_path,
    state_variants,
    subdirectories_by_name,
)
from ..models import ImageRecord

logger = logging.getLogger(__name__)

__all__ = [
    "RecordState",
    "TransitionPlan",
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


@dataclass(frozen=True)
class _Move:
    source: Path
    destination: Path


@dataclass(frozen=True)
class _RecordPlan:
    """The validated result of transitioning one record to a new state.

    Holds both the moves to perform and the paths the record's files will
    occupy afterwards, so the record can be brought up to date once every move
    in the batch has succeeded.
    """

    record: ImageRecord
    moves: tuple[_Move, ...]
    image_path: Path
    labels: tuple[Path, ...]


def current_state(record: ImageRecord) -> RecordState:
    """Return the workflow state implied by a record's image path.

    Note that this reflects the path the record currently holds. After a
    successful :func:`transition_records` that path is up to date; if files have
    been moved by something else since the record was built, re-scan first.
    """
    return state_of_path(record.image_path)


def existing_variant(path: Path) -> Path | None:
    """Find a file's sole on-disk state variant, rejecting ambiguous copies."""
    existing = [candidate for candidate in state_variants(path) if candidate.exists()]
    if len(existing) > 1:
        raise FileExistsError(f"Multiple state copies found for {base_path(path)}: {existing}")
    return existing[0] if existing else None


def _plan_record_transition(record: ImageRecord, state: RecordState) -> _RecordPlan:
    """Work out the moves and the resulting paths for one record.

    A destination is recorded for every file that exists on disk, whether or not
    it needs moving: a file may already be in the target state while the record
    still refers to it by a path in a different state folder, and that stale
    path should be corrected too. Labels that cannot be found anywhere keep the
    path the record already had, since nothing about them has changed.
    """
    moves: list[_Move] = []
    resolved: list[Path] = []
    for index, path in enumerate([record.image_path, *record.labels]):
        source = existing_variant(path)
        if source is None:
            if index == 0:
                raise FileNotFoundError(path)
            logger.warning("Associated label not found: %s", path)
            resolved.append(path)
            continue
        destination = state_path(path, state)
        if source != destination:
            moves.append(_Move(source, destination))
        resolved.append(destination)
    return _RecordPlan(
        record=record,
        moves=tuple(moves),
        image_path=resolved[0],
        labels=tuple(resolved[1:]),
    )


@dataclass(frozen=True)
class TransitionPlan:
    """A validated, not-yet-performed state change.

    Attributes:
        state: The state everything moves into.
        records: Every record the change applies to, after expanding across
            shared labels. This can be larger than what the caller asked for.
        added: The records that expansion added, so a caller can say how far
            the change actually reaches before performing it.
        moves: The file moves to perform, deduplicated.
    """

    state: RecordState
    records: tuple[ImageRecord, ...]
    added: tuple[ImageRecord, ...]
    moves: tuple[_Move, ...]
    #: Per-record outcomes, computed before anything moved, so the records can
    #: be brought up to date afterwards without re-reading the filesystem.
    record_plans: tuple[_RecordPlan, ...] = ()

    @property
    def reaches_beyond_request(self) -> bool:
        """Return whether a shared label pulled in images the caller did not name."""
        return bool(self.added)

    def describe(self) -> list[str]:
        """Return one line per move, for a preview or a log."""
        return [f"{move.source} -> {move.destination}" for move in self.moves]


def plan_transition(
    records: Iterable[ImageRecord],
    state: RecordState,
    *,
    allow_move_from_corrected: bool = False,
) -> TransitionPlan:
    """Validate a state change without performing it.

    The whole batch is checked before anything moves: every file is located,
    destinations are checked for collisions and for existing files, and
    corrected data is refused unless explicitly confirmed. The same validation
    runs whether or not the plan is executed, so a dry run is a genuine
    rehearsal rather than a guess.

    Records are expanded across shared labels first. Moving a shared label
    without moving the images that share it would silently change their state.

    Raises:
        ValueError: If *state* is not a workflow state.
        FileNotFoundError: If a record's image cannot be found in any state.
        FileExistsError: If two files would land on the same destination, or a
            destination is already occupied.
        PermissionError: If corrected data would be moved without confirmation.
    """
    if state not in STATE_DIRECTORIES:
        raise ValueError(f"Unknown workflow state: {state}")

    requested = list(records)
    expanded = expand_to_groups(requested)
    requested_paths = {str(base_path(record.image_path)) for record in requested}
    added = tuple(
        record for record in expanded
        if str(base_path(record.image_path)) not in requested_paths
    )

    plans = [_plan_record_transition(record, state) for record in expanded]

    # A shared label is referenced by every image that shares it, so the same
    # move is planned several times. Identical moves collapse; two *different*
    # files heading for one destination remains an error.
    moves: list[_Move] = []
    by_destination: dict[Path, Path] = {}
    for move in (move for plan in plans for move in plan.moves):
        existing = by_destination.get(move.destination)
        if existing is not None:
            if existing == move.source:
                continue
            raise FileExistsError(f"Multiple files would be moved to {move.destination}")
        by_destination[move.destination] = move.source
        moves.append(move)

    if not allow_move_from_corrected:
        corrected_sources = [
            move.source for move in moves if state_of_path(move.source) == "corrected"
        ]
        if corrected_sources:
            raise PermissionError(
                "Refusing to move corrected artifacts without explicit confirmation: "
                f"{corrected_sources}"
            )

    for move in moves:
        if move.destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {move.destination}")

    return TransitionPlan(
        state=state,
        records=tuple(expanded),
        added=added,
        moves=tuple(moves),
        record_plans=tuple(plans),
    )


def transition_records(
    records: Iterable[ImageRecord],
    state: RecordState,
    *,
    allow_move_from_corrected: bool = False,
    dry_run: bool = False,
) -> int:
    """Move records into ``state`` after validating the whole batch.

    The state locations are mutually exclusive. Selecting can restore a
    corrected record, but requires ``allow_move_from_corrected=True`` to
    protect manually corrected labels.

    Records sharing a label move together: the label is the unit of
    proofreading, so a state change applies to every image of the same scene.
    The return value counts the expanded set, which can exceed what was asked
    for; use :func:`plan_transition` when the difference needs reporting first.

    On success each record is updated to refer to its new location, so callers
    that keep records alive across several operations -- a long-running UI, for
    example -- continue to hold valid paths.

    Args:
        records: The records to move.
        state: The state to move into.
        allow_move_from_corrected: Confirm moving data out of the corrected
            state.
        dry_run: Validate everything and report the count without moving any
            file.
    """
    plan = plan_transition(
        records, state, allow_move_from_corrected=allow_move_from_corrected
    )
    if dry_run:
        for line in plan.describe():
            logger.info("Would move %s", line)
        return len(plan.records)

    for move in plan.moves:
        move_to_state(
            move.source,
            state,
            allow_move_from_corrected=allow_move_from_corrected,
        )
    for record_plan in plan.record_plans:
        _apply_plan(record_plan, state)
    return len(plan.records)


def _apply_plan(plan: _RecordPlan, state: RecordState) -> None:
    """Update a record to reflect where its files now live and their state."""
    plan.record.image_path = plan.image_path
    plan.record.labels = list(plan.labels)
    plan.record.selected = state == "selected"
    plan.record.corrected = state == "corrected"


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

    A group becomes inconsistent when files are moved by something other than
    this package, or when an operation is interrupted partway. The label is
    authoritative, so repair moves the *images* to match it and never the other
    way round -- the label holds the proofreading work, and its location is the
    record of what was decided about the scene.

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
        logger.info(
            "Group %s is %s; moving %d image(s) to match",
            describe_group(group), state, len(group.inconsistent_records()),
        )
        moved = transition_records(
            group.records,
            state,
            allow_move_from_corrected=allow_move_from_corrected,
            dry_run=dry_run,
        )
        repaired.append((group, moved))
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
