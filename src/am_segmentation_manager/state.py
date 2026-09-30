"""The declared workflow state of a stage's label groups.

``AM_state.yaml`` in each directory holding image and label directories names the
groups that are not selected, keyed by the group's label path relative to it, so an
acquisition's conditions share one entry. No entry means ``selected``, so images
added to a stage arrive selected. The file is hand-editable and nothing read from
it is trusted: a key must be a path inside the stage, a value a state, and a
duplicated key is an error. A write refuses a file that changed since it was read.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args

from .io.atomic import Fingerprint, fingerprint_of
from .io.path import checked_relative_key
from .io.yaml import DuplicateKeyError, load_unique_keys, write_yaml

logger = logging.getLogger(__name__)

__all__ = [
    "AM_STATE_FILENAME",
    "RecordState",
    "DEFAULT_STATE",
    "STATE_VALUES",
    "StageState",
    "StaleState",
    "StateFileError",
    "read_state",
    "state_file_path",
    "state_key",
    "write_state",
]

#: A label group's workflow state. The three are mutually exclusive.
RecordState = Literal["selected", "deselected", "corrected"]

#: The file declaring workflow state, one per directory holding image and label
#: directories.
AM_STATE_FILENAME: Final[str] = "AM_state.yaml"

#: The state of a group with no entry. Not written out.
DEFAULT_STATE: Final[RecordState] = "selected"

#: The states an entry may name, taken from the type so there is one vocabulary.
STATE_VALUES: Final[tuple[RecordState, ...]] = get_args(RecordState)

#: Written into every file. A reader refuses a version it does not know, so a
#: file written by a later am-seg is a refusal rather than a misreading.
FILE_VERSION: Final[int] = 1

_VERSION_KEY: Final[str] = "version"
_STATE_KEY: Final[str] = "state"

_HEADER: Final[str] = """\
# Workflow state for this stage, written by am-seg.
#
# One entry per label group: selected | deselected | corrected. A path names
# the group's label, relative to this file, and may not exist yet. Anything not
# listed is selected.
#
# Safe to edit by hand: one state per path, and no path may leave this
# directory. Renaming an image or label directory changes a group's key, which
# am-seg reports rather than guesses at.
"""


class StateFileError(RuntimeError):
    """Raised when a state file cannot be read, or holds something invalid."""


class StaleState(RuntimeError):
    """Raised when a state file changed between being read and written.

    Two windows on one collection, or a window and a command line, would
    otherwise discard each other's curation with no error.
    """


@dataclass(frozen=True)
class StageState:
    """The states declared for one stage, as read from its file.

    Attributes:
        path: The state file, whether or not it exists.
        states: Declared states by key. Groups in the default state are absent.
        fingerprint: What the file looked like when read, or None if it did not
            exist. :func:`write_state` refuses to write over anything else.
    """

    path: Path
    states: Mapping[str, RecordState]
    fingerprint: Fingerprint | None

    def state_of(self, key: str) -> RecordState:
        """Return the state declared for *key*, or the default if none is."""
        return self.states.get(key, DEFAULT_STATE)


def state_file_path(stage_directory: Path) -> Path:
    """Return where *stage_directory*'s state file lives, existing or not."""
    return stage_directory / AM_STATE_FILENAME


def state_key(label_path: Path, stage_directory: Path) -> str:
    """Return the key naming the group whose label is *label_path*.

    Raises:
        StateFileError: If *label_path* is not inside *stage_directory*.
    """
    try:
        relative = label_path.relative_to(stage_directory)
    except ValueError as exc:
        raise StateFileError(
            f"{label_path} is not inside the stage directory {stage_directory}"
        ) from exc
    return relative.as_posix()


def read_state(stage_directory: Path) -> StageState:
    """Read *stage_directory*'s declared states.

    A missing file reads as no declarations, which is a stage where everything
    is selected -- the same answer, so a caller need not ask which it was.

    Raises:
        StateFileError: If the file cannot be parsed, was written by a later
            version, duplicates a key, or holds a key or state that is not
            valid.
    """
    path = state_file_path(stage_directory)
    fingerprint = fingerprint_of(path)
    if fingerprint is None:
        return StageState(path=path, states={}, fingerprint=None)

    try:
        with path.open("r", encoding="utf-8") as handle:
            document = _load(handle)
    except StateFileError:
        raise
    except Exception as exc:
        raise StateFileError(f"Failed to parse state file {path}: {exc}") from exc

    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise StateFileError(f"Expected a mapping at the top level of {path}")

    _check_version(document.get(_VERSION_KEY), path)
    states = _checked_states(document.get(_STATE_KEY), path, stage_directory)
    logger.debug("Read %s declared states from %s", len(states), path)
    return StageState(path=path, states=states, fingerprint=fingerprint)


def write_state(previous: StageState, states: Mapping[str, str]) -> StageState:
    """Write *states* over the file *previous* was read from.

    Entries in the default state are dropped: a state file lists what is not
    selected. A stage with nothing left to declare keeps an empty file rather
    than having one deleted out from under it.

    Returns:
        The state as written, ready to be written over again.

    Raises:
        StaleState: If the file changed since *previous* was read.
        StateFileError: If a key or state is not valid.
    """
    path = previous.path
    stale = StaleState(
        f"{path} changed since it was read; re-read it and apply the change again"
    )
    if fingerprint_of(path) != previous.fingerprint:
        raise stale

    declared = _checked_states(dict(states), path, path.parent)
    kept = {key: value for key, value in declared.items() if value != DEFAULT_STATE}
    # Sorted, so two runs producing the same states produce the same file.
    # Checked again just before the rename, which is as late as it can be asked.
    if not write_yaml(
        path,
        {_VERSION_KEY: FILE_VERSION, _STATE_KEY: dict(sorted(kept.items()))},
        header=_HEADER,
        commit_if=lambda _temporary: fingerprint_of(path) == previous.fingerprint,
    ):
        raise stale
    logger.debug("Wrote %s declared states to %s", len(kept), path)
    return StageState(path=path, states=kept, fingerprint=fingerprint_of(path))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _load(handle: Any) -> Any:
    """Parse a state file, refusing a duplicated key rather than picking one."""
    try:
        return load_unique_keys(handle)
    except DuplicateKeyError as error:
        raise StateFileError(str(error)) from error


def _check_version(version: Any, path: Path) -> None:
    """Reject a file written by a later am-seg; accept one that omits it."""
    if version is None:
        return
    if not isinstance(version, int) or isinstance(version, bool):
        raise StateFileError(f"{path}: {_VERSION_KEY} must be a whole number")
    if version > FILE_VERSION:
        raise StateFileError(
            f"{path} was written by a later version of am-seg "
            f"(file version {version}, this reads up to {FILE_VERSION})"
        )


def _checked_states(
    block: Any, path: Path, stage_directory: Path
) -> dict[str, RecordState]:
    """Return the validated entries of a state block."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise StateFileError(f"{path}: {_STATE_KEY} must be a mapping of path to state")

    checked: dict[str, RecordState] = {}
    for key, value in block.items():
        if not isinstance(key, str):
            raise StateFileError(f"{path}: {key!r} is not a path")
        if value not in STATE_VALUES:
            raise StateFileError(
                f"{path}: {key} is {value!r}, which is not one of "
                f"{', '.join(STATE_VALUES)}"
            )
        checked[_checked_key(key, stage_directory)] = value
    return checked


def _checked_key(key: str, stage_directory: Path) -> str:
    """Return *key* if it names a path inside the stage, else raise.

    The rule itself is :func:`~am_segmentation_manager.io.path.checked_relative_key`,
    shared with the history log so the two cannot disagree about what a key is;
    only the exception type is this module's.
    """
    try:
        return checked_relative_key(key, stage_directory)
    except ValueError as error:
        raise StateFileError(str(error)) from error
