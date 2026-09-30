"""Reading a project's ``AM_families.yaml``, its corrections to family names.

An image directory's name gives the family whose label it shares; this file,
at a project root, moves a condition into another family and may name a
family's primary condition. It is read into a :class:`~.layout.FamilyCorrections`
that resolution takes as plain data, and a project without it has none. It is
hand-editable, so nothing read from it is trusted: a file that will not read is
an error, never read as no corrections, which would resolve labels elsewhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

from .discovery import project_markers
from .io.yaml import DuplicateKeyError, load_unique_keys
from .layout import NO_CORRECTIONS, FamilyCorrections

logger = logging.getLogger(__name__)

__all__ = [
    "AM_FAMILIES_FILENAME",
    "FamiliesFileError",
    "families_file_path",
    "families_for",
    "read_families",
]

AM_FAMILIES_FILENAME: Final[str] = "AM_families.yaml"

#: A reader refuses a version it does not know, so a file written by a later
#: am-seg is a refusal rather than a misreading.
FILE_VERSION: Final[int] = 1

_VERSION_KEY: Final[str] = "version"
_FAMILIES_KEY: Final[str] = "families"
_PRIMARY_KEY: Final[str] = "primary"
_KEYS: Final[frozenset[str]] = frozenset({_VERSION_KEY, _FAMILIES_KEY, _PRIMARY_KEY})


class FamiliesFileError(RuntimeError):
    """Raised when a project's families file cannot be read, or holds something invalid."""


def families_file_path(project_root: Path) -> Path:
    """Return where *project_root*'s families file lives, existing or not."""
    return project_root / AM_FAMILIES_FILENAME


def read_families(project_root: Path) -> FamilyCorrections:
    """Return the family corrections *project_root* declares.

    Raises:
        FamiliesFileError: If the file does not parse, was written by a later
            version, holds an unknown section or repeats a key, or holds an
            entry :func:`~.layout.correction_problems` refuses.
    """
    path = families_file_path(project_root)
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = load_unique_keys(handle)
    except FileNotFoundError:
        return NO_CORRECTIONS
    except DuplicateKeyError as error:
        raise FamiliesFileError(f"{path}: {error}") from error
    except Exception as error:
        raise FamiliesFileError(f"Failed to parse {path}: {error}") from error

    if document is None:
        return NO_CORRECTIONS
    if not isinstance(document, dict):
        raise FamiliesFileError(f"Expected a mapping at the top level of {path}")
    unknown = sorted(str(key) for key in document if key not in _KEYS)
    if unknown:
        raise FamiliesFileError(
            f"{path}: unknown section {', '.join(unknown)}; a families file holds "
            f"{', '.join(sorted(_KEYS))}"
        )
    _check_version(document.get(_VERSION_KEY), path)
    families = _mapping(document.get(_FAMILIES_KEY), _FAMILIES_KEY, path)
    primary = _mapping(document.get(_PRIMARY_KEY), _PRIMARY_KEY, path)
    try:
        corrections = FamilyCorrections(families=families, primary=primary)
    except ValueError as error:
        raise FamiliesFileError(f"{path}: {error}") from error
    logger.debug(
        "Read %d family correction(s) and %d primary condition(s) from %s",
        len(families), len(primary), path,
    )
    return corrections


def families_for(
    path: Path, cache: dict[Path, FamilyCorrections] | None = None
) -> FamilyCorrections:
    """Return the corrections of the project holding *path*.

    Lets an image path alone find its label. The project is the nearest
    directory at or above *path* marked as one; outside any project there are
    none.

    Args:
        cache: Reused across a batch, keyed by the directory asked about, so a
            batch reads each project's file once. Hold it no longer than the
            batch: nothing notices the file changing.

    Raises:
        FamiliesFileError: As :func:`read_families`.
    """
    start = path if path.is_dir() else path.parent
    if cache is not None and start in cache:
        return cache[start]
    corrections = NO_CORRECTIONS
    for directory in (start, *start.parents):
        if project_markers(directory):
            corrections = read_families(directory)
            break
    if cache is not None:
        cache[start] = corrections
    return corrections


def _check_version(version: Any, path: Path) -> None:
    """Reject a file written by a later am-seg; accept one that omits it."""
    if version is None:
        return
    if not isinstance(version, int) or isinstance(version, bool):
        raise FamiliesFileError(f"{path}: {_VERSION_KEY} must be a whole number")
    if version > FILE_VERSION:
        raise FamiliesFileError(
            f"{path} was written by a later version of am-seg "
            f"(file version {version}, this reads up to {FILE_VERSION})"
        )


def _mapping(block: Any, name: str, path: Path) -> dict[str, str]:
    """Return a section of names to names, refusing anything else."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise FamiliesFileError(f"{path}: {name} must be a mapping of name to name")
    for key, value in block.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise FamiliesFileError(
                f"{path}: {name}: {key!r}: {value!r} is not a name mapped to a name"
            )
    return dict(block)
