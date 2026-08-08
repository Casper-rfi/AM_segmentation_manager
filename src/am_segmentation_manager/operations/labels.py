"""Low-level, conservative filesystem moves for workflow state folders."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_DIRECTORIES = {"selected": None, "deselected": "deselected", "corrected": "corrected"}


def move_to_state(path: Path, state: str, *, allow_move_from_corrected: bool = False) -> Path:
    """Move a file to ``selected``, ``deselected``, or ``corrected``.

    State folders are siblings, so transitioning between states never nests
    ``corrected`` inside ``deselected`` (or vice versa). Existing files are
    never overwritten. Moving a corrected artifact requires explicit consent.
    """
    if state not in STATE_DIRECTORIES:
        raise ValueError(f"Unknown workflow state: {state}")
    if path.parent.name == "corrected" and state != "corrected" and not allow_move_from_corrected:
        raise PermissionError(
            f"Refusing to move corrected artifact without explicit confirmation: {path}"
        )
    base_directory = path.parent.parent if path.parent.name in {"deselected", "corrected"} else path.parent
    folder = STATE_DIRECTORIES[state]
    destination = base_directory / path.name if folder is None else base_directory / folder / path.name
    if destination == path:
        return destination
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(exist_ok=True)
    shutil.move(str(path), str(destination))
    logger.info("Moved %s -> %s", path, destination)
    return destination


def move_to_deselected(path: Path) -> Path:
    """Move a file into its deselected state directory."""
    return move_to_state(path, "deselected")


def move_to_selected(path: Path, *, allow_move_from_corrected: bool = False) -> Path:
    """Move a deselected or corrected file into the normal state directory."""
    if path.parent.name not in {"deselected", "corrected"}:
        raise ValueError("Path is not inside a workflow state folder: %s" % path)
    return move_to_state(path, "selected", allow_move_from_corrected=allow_move_from_corrected)


def move_to_corrected(path: Path) -> Path:
    """Move a file into its corrected state directory."""
    return move_to_state(path, "corrected")
