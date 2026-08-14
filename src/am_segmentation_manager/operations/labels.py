"""Low-level, conservative filesystem moves for workflow state folders."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..layout import STATE_DIRECTORIES, state_of_path, state_path

logger = logging.getLogger(__name__)

__all__ = [
    "STATE_DIRECTORIES",
    "move_to_state",
]


def move_to_state(path: Path, state: str, *, allow_move_from_corrected: bool = False) -> Path:
    """Move a file to ``selected``, ``deselected``, or ``corrected``.

    State folders are siblings, so transitioning between states never nests
    ``corrected`` inside ``deselected`` (or vice versa). Existing files are
    never overwritten. Moving a corrected artifact requires explicit consent.
    """
    if state not in STATE_DIRECTORIES:
        raise ValueError(f"Unknown workflow state: {state}")
    if state_of_path(path) == "corrected" and state != "corrected" and not allow_move_from_corrected:
        raise PermissionError(
            f"Refusing to move corrected artifact without explicit confirmation: {path}"
        )
    destination = state_path(path, state)
    if destination == path:
        return destination
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(exist_ok=True)
    shutil.move(str(path), str(destination))
    logger.info("Moved %s -> %s", path, destination)
    return destination
