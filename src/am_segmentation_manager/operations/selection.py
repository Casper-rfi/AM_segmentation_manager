"""Filesystem transitions for selected, deselected, and corrected records."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .labels import STATE_DIRECTORIES, move_to_state
from ..models import ImageRecord

logger = logging.getLogger(__name__)

RecordState = Literal["selected", "deselected", "corrected"]
_STATE_FOLDERS = {folder for folder in STATE_DIRECTORIES.values() if folder is not None}
_FOLDER_STATES: dict[str, RecordState] = {
    folder: state for state, folder in STATE_DIRECTORIES.items() if folder is not None
}


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

    The state folder a file sits in *is* its state -- there is no separate
    store to consult -- so this reads the state straight off the recorded path.

    Note that this reflects the path the record currently holds. After a
    successful :func:`transition_records` that path is up to date; if files have
    been moved by something else since the record was built, re-scan first.
    """
    return _FOLDER_STATES.get(record.image_path.parent.name, "selected")


def base_path(path: Path) -> Path:
    """Return *path* with any state subfolder (deselected/corrected) stripped.

    For example ``sem/deselected/img.tif`` and ``sem/img.tif`` both resolve
    to ``sem/img.tif`` — the shared identity used to look up a file's other
    possible state locations.
    """
    if path.parent.name in _STATE_FOLDERS:
        return path.parent.parent / path.name
    return path


def existing_variant(path: Path) -> Path | None:
    """Find a file's sole on-disk state variant, rejecting ambiguous copies."""
    base = base_path(path)
    variants = [base, *(base.parent / folder / base.name for folder in _STATE_FOLDERS)]
    existing = [candidate for candidate in variants if candidate.exists()]
    if len(existing) > 1:
        raise FileExistsError(f"Multiple state copies found for {base}: {existing}")
    return existing[0] if existing else None


def _destination(path: Path, state: RecordState) -> Path:
    base = base_path(path)
    folder = STATE_DIRECTORIES[state]
    return base if folder is None else base.parent / folder / base.name


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
        destination = _destination(path, state)
        if source != destination:
            moves.append(_Move(source, destination))
        resolved.append(destination)
    return _RecordPlan(
        record=record,
        moves=tuple(moves),
        image_path=resolved[0],
        labels=tuple(resolved[1:]),
    )


def transition_records(
    records: Iterable[ImageRecord],
    state: RecordState,
    *,
    allow_move_from_corrected: bool = False,
) -> int:
    """Move records into ``state`` after validating the whole batch.

    The state locations are mutually exclusive. Selecting can restore a
    corrected record, but requires ``allow_move_from_corrected=True`` to
    protect manually corrected labels.

    On success each record is updated to refer to its new location, so callers
    that keep records alive across several operations -- a long-running UI, for
    example -- continue to hold valid paths.
    """
    if state not in STATE_DIRECTORIES:
        raise ValueError(f"Unknown workflow state: {state}")
    record_list = list(records)
    plans = [_plan_record_transition(record, state) for record in record_list]
    moves = [move for plan in plans for move in plan.moves]
    if not allow_move_from_corrected:
        corrected_sources = [move.source for move in moves if move.source.parent.name == "corrected"]
        if corrected_sources:
            raise PermissionError(
                "Refusing to move corrected artifacts without explicit confirmation: "
                f"{corrected_sources}"
            )
    destinations: set[Path] = set()
    for move in moves:
        if move.destination in destinations:
            raise FileExistsError(f"Multiple files would be moved to {move.destination}")
        destinations.add(move.destination)
        if move.destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {move.destination}")
    for move in moves:
        move_to_state(
            move.source,
            state,
            allow_move_from_corrected=allow_move_from_corrected,
        )
    for plan in plans:
        _apply_plan(plan, state)
    return len(record_list)


def _apply_plan(plan: _RecordPlan, state: RecordState) -> None:
    """Update a record to reflect where its files now live and their state."""
    plan.record.image_path = plan.image_path
    plan.record.labels = list(plan.labels)
    plan.record.selected = state == "selected"
    plan.record.corrected = state == "corrected"


def deselect_records(
    records: Iterable[ImageRecord], *, allow_move_from_corrected: bool = False
) -> int:
    """Move records into the deselected state."""
    return transition_records(records, "deselected", allow_move_from_corrected=allow_move_from_corrected)


def select_records(
    records: Iterable[ImageRecord], *, allow_move_from_corrected: bool = False
) -> int:
    """Move records into the normal selected state."""
    return transition_records(records, "selected", allow_move_from_corrected=allow_move_from_corrected)


def mark_corrected_records(
    records: Iterable[ImageRecord], *, allow_move_from_corrected: bool = False
) -> int:
    """Move records into the corrected state."""
    return transition_records(records, "corrected", allow_move_from_corrected=allow_move_from_corrected)


def record_from_image_path(path: Path) -> ImageRecord:
    """Build a record for direct-path use, including an associated label."""
    image_path = base_path(path)
    modality_directory = next((parent for parent in image_path.parents if parent.name in {"sem", "fib"}), None)
    labels: list[Path] = []
    if modality_directory is not None:
        label_directory = modality_directory.parent / ("labels" if modality_directory.name == "sem" else "labels_fib")
        label_path = label_directory / image_path.name
        if existing_variant(label_path) is not None:
            labels.append(label_path)
    modality = modality_directory.name.upper() if modality_directory is not None else None
    return ImageRecord(image_path=image_path, labels=labels, modality=modality)
