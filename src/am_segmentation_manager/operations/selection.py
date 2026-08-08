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


@dataclass(frozen=True)
class _Move:
    source: Path
    destination: Path


def _base_path(path: Path) -> Path:
    if path.parent.name in _STATE_FOLDERS:
        return path.parent.parent / path.name
    return path


def _existing_variant(path: Path) -> Path | None:
    """Find a file's sole on-disk state variant, rejecting ambiguous copies."""
    base = _base_path(path)
    variants = [base, *(base.parent / folder / base.name for folder in _STATE_FOLDERS)]
    existing = [candidate for candidate in variants if candidate.exists()]
    if len(existing) > 1:
        raise FileExistsError(f"Multiple state copies found for {base}: {existing}")
    return existing[0] if existing else None


def _destination(path: Path, state: RecordState) -> Path:
    base = _base_path(path)
    folder = STATE_DIRECTORIES[state]
    return base if folder is None else base.parent / folder / base.name


def _plan_record_transition(record: ImageRecord, state: RecordState) -> list[_Move]:
    moves: list[_Move] = []
    for index, path in enumerate([record.image_path, *record.labels]):
        source = _existing_variant(path)
        if source is None:
            if index == 0:
                raise FileNotFoundError(path)
            logger.warning("Associated label not found: %s", path)
            continue
        destination = _destination(path, state)
        if source != destination:
            moves.append(_Move(source, destination))
    return moves


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
    """
    if state not in STATE_DIRECTORIES:
        raise ValueError(f"Unknown workflow state: {state}")
    record_list = list(records)
    plans = [(record, _plan_record_transition(record, state)) for record in record_list]
    moves = [move for _, record_moves in plans for move in record_moves]
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
    for record, _ in plans:
        record.selected = state == "selected"
        record.corrected = state == "corrected"
    return len(record_list)


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
    image_path = _base_path(path)
    modality_directory = next((parent for parent in image_path.parents if parent.name in {"sem", "fib"}), None)
    labels: list[Path] = []
    if modality_directory is not None:
        label_directory = modality_directory.parent / ("labels" if modality_directory.name == "sem" else "labels_fib")
        label_path = label_directory / image_path.name
        if _existing_variant(label_path) is not None:
            labels.append(label_path)
    modality = modality_directory.name.upper() if modality_directory is not None else None
    return ImageRecord(image_path=image_path, labels=labels, modality=modality)
