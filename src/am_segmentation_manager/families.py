"""Reading a project's ``AM_families.yaml``, its corrections to family names.

An image directory's name gives the family whose label it shares; this file,
at a project root, moves a condition into another family and may name a
family's primary condition, for the whole project and for one lamella over it.
It is read into a :class:`~.layout.ProjectFamilies`, and a project without it
has none. It is hand-editable, so nothing read from it is trusted: a file that
will not read is an error, never read as no corrections, which would resolve
labels elsewhere. Only :mod:`.operations.families_edit` writes it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

from .discovery import project_markers
from .io.atomic import Fingerprint, fingerprint_of
from .io.yaml import DuplicateKeyError, load_unique_keys, write_yaml
from .layout import (
    NO_PROJECT_FAMILIES,
    FamilyCorrections,
    FamilyEntries,
    ProjectFamilies,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AM_FAMILIES_FILENAME",
    "FamiliesFileError",
    "StaleFamilies",
    "families_file_path",
    "families_for",
    "read_families",
    "write_families",
]

AM_FAMILIES_FILENAME: Final[str] = "AM_families.yaml"

#: A reader refuses a version it does not know, so a file written by a later
#: am-seg is a refusal rather than a misreading. A file with no lamella
#: sections is written as version 1, which an am-seg without them still reads
#: (``test_a_file_without_lamella_sections_stays_version_1``).
FILE_VERSION: Final[int] = 2
_FIRST_VERSION: Final[int] = 1

_VERSION_KEY: Final[str] = "version"
_FAMILIES_KEY: Final[str] = "families"
_PRIMARY_KEY: Final[str] = "primary"
_LAMELLAE_KEY: Final[str] = "lamellae"
_KEYS: Final[frozenset[str]] = frozenset(
    {_VERSION_KEY, _FAMILIES_KEY, _PRIMARY_KEY, _LAMELLAE_KEY}
)
_SECTION_KEYS: Final[frozenset[str]] = frozenset({_FAMILIES_KEY, _PRIMARY_KEY})

_HEADER: Final[str] = """\
# Families of imaging conditions in this project, written by am-seg.
#
# An image folder is named <family>_<variant>, and the conditions of one family
# share one label folder, <family>_labels. Under families:, a condition belongs
# to the family given instead of the one its name gives, e.g. sem2kVb: sem-2kV.
# Under primary:, a family names the condition it is predicted from. Under
# lamellae:, a lamella's path holds families: and primary: of its own, each
# entry replacing the project's in that lamella alone.
#
# Change it with 'am-seg edit-families', which refuses a change that would leave
# a label unread or replace one, and records each change in AM_history.jsonl.
"""


class FamiliesFileError(RuntimeError):
    """Raised when a project's families file cannot be read, or holds something invalid."""


class StaleFamilies(RuntimeError):
    """Raised when a families file changed between being read and written."""


def families_file_path(project_root: Path) -> Path:
    """Return where *project_root*'s families file lives, existing or not."""
    return project_root / AM_FAMILIES_FILENAME


def read_families(project_root: Path) -> ProjectFamilies:
    """Return the family corrections *project_root* declares, its own and each lamella's.

    Raises:
        FamiliesFileError: If the file does not parse, was written by a later
            version, holds an unknown section or repeats a key, or holds an
            entry :func:`~.layout.correction_problems` refuses, alone or over
            the project's in a lamella.
    """
    path = families_file_path(project_root)
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = load_unique_keys(handle)
    except FileNotFoundError:
        return NO_PROJECT_FAMILIES
    except DuplicateKeyError as error:
        raise FamiliesFileError(f"{path}: {error}") from error
    except Exception as error:
        raise FamiliesFileError(f"Failed to parse {path}: {error}") from error

    if document is None:
        return NO_PROJECT_FAMILIES
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
    lamellae = _lamellae(document.get(_LAMELLAE_KEY), path)
    try:
        corrections = ProjectFamilies(
            project=FamilyCorrections(families=families, primary=primary), lamellae=lamellae
        )
    except ValueError as error:
        raise FamiliesFileError(f"{path}: {error}") from error
    logger.debug(
        "Read %d family correction(s), %d primary condition(s) and %d lamella "
        "section(s) from %s", len(families), len(primary), len(lamellae), path,
    )
    return corrections


def write_families(
    project_root: Path, corrections: ProjectFamilies, expected: Fingerprint | None
) -> Path:
    """Write *corrections* as *project_root*'s families file, and return its path.

    A project left with no corrections keeps an empty file rather than having
    one deleted.

    Args:
        expected: The file's fingerprint when the change was planned, None if
            it did not exist.

    Raises:
        StaleFamilies: If the file changed since *expected* was taken.
    """
    path = families_file_path(project_root)
    stale = StaleFamilies(
        f"{path} changed since this change was worked out; look again and repeat it"
    )
    if fingerprint_of(path) != expected:
        raise stale
    sections = {
        key: entries for key, entries in sorted(corrections.lamellae.items())
        if entries.families or entries.primary
    }
    body: dict[str, Any] = {
        _VERSION_KEY: FILE_VERSION if sections else _FIRST_VERSION,
        _FAMILIES_KEY: dict(sorted(corrections.project.families.items(), key=_lowered)),
        _PRIMARY_KEY: dict(sorted(corrections.project.primary.items(), key=_lowered)),
    }
    if sections:
        body[_LAMELLAE_KEY] = {
            key: {
                _FAMILIES_KEY: dict(sorted(entries.families.items(), key=_lowered)),
                _PRIMARY_KEY: dict(sorted(entries.primary.items(), key=_lowered)),
            }
            for key, entries in sections.items()
        }
    if not write_yaml(
        path, body, header=_HEADER,
        commit_if=lambda _temporary: fingerprint_of(path) == expected,
    ):
        raise stale
    return path


def _lowered(item: tuple[str, str]) -> str:
    return item[0].lower()


def families_for(
    path: Path, cache: dict[Path, FamilyCorrections] | None = None
) -> FamilyCorrections:
    """Return the corrections in force at *path*: its project's, and its lamella's over them.

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
    corrections = NO_PROJECT_FAMILIES.project
    for directory in (start, *start.parents):
        if project_markers(directory):
            corrections = read_families(directory).at(start.relative_to(directory).as_posix())
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


def _lamellae(block: Any, path: Path) -> dict[str, FamilyEntries]:
    """Return each lamella's section, refusing anything else."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise FamiliesFileError(f"{path}: {_LAMELLAE_KEY} must be a mapping of lamella to sections")
    sections: dict[str, FamilyEntries] = {}
    for key, section in block.items():
        where = f"{_LAMELLAE_KEY}: {key}"
        if not isinstance(key, str) or not isinstance(section, dict):
            raise FamiliesFileError(f"{path}: {where} must be a lamella's path holding sections")
        unknown = sorted(str(name) for name in section if name not in _SECTION_KEYS)
        if unknown:
            raise FamiliesFileError(
                f"{path}: {where}: unknown section {', '.join(unknown)}; a lamella holds "
                f"{', '.join(sorted(_SECTION_KEYS))}"
            )
        sections[key] = FamilyEntries(
            families=_mapping(section.get(_FAMILIES_KEY), f"{where}: {_FAMILIES_KEY}", path),
            primary=_mapping(section.get(_PRIMARY_KEY), f"{where}: {_PRIMARY_KEY}", path),
        )
    return sections


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
