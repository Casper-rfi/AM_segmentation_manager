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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from .discovery import project_markers
from .io.atomic import Fingerprint
from .io.yaml import (
    DuplicateKeyError,
    load_unique_keys,
    version_problem,
    write_yaml_unchanged,
)
from .layout import (
    NO_PROJECT_FAMILIES,
    FamilyCorrections,
    FamilyEntries,
    ProjectFamilies,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AM_FAMILIES_FILENAME",
    "FAMILIES_SECTION",
    "PRIMARY_SECTION",
    "FamiliesFileError",
    "FamiliesLookup",
    "StaleFamilies",
    "document_of",
    "families_file_path",
    "families_for",
    "families_from_document",
    "read_families",
    "sections_of",
    "write_families",
]

AM_FAMILIES_FILENAME: Final[str] = "AM_families.yaml"

#: A reader refuses a version it does not know, so a file written by a later
#: am-seg is a refusal rather than a misreading. A file with no lamella
#: sections is written as version 1, which an am-seg without them still reads
#: (``test_a_file_without_lamella_sections_stays_version_1``).
FILE_VERSION: Final[int] = 2
_FIRST_VERSION: Final[int] = 1

#: The two sections of a project's or a lamella's corrections, as the file
#: names them: condition to family, and family to its primary condition.
FAMILIES_SECTION: Final[str] = "families"
PRIMARY_SECTION: Final[str] = "primary"

_VERSION_KEY: Final[str] = "version"
_LAMELLAE_KEY: Final[str] = "lamellae"
_SECTION_KEYS: Final[frozenset[str]] = frozenset({FAMILIES_SECTION, PRIMARY_SECTION})
_DOCUMENT_KEYS: Final[frozenset[str]] = _SECTION_KEYS | {_LAMELLAE_KEY}

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

    if isinstance(document, dict):
        _check_version(document.get(_VERSION_KEY), path)
        document = {key: value for key, value in document.items() if key != _VERSION_KEY}
    corrections = families_from_document(document, str(path))
    logger.debug(
        "Read %d family correction(s), %d primary condition(s) and %d lamella "
        "section(s) from %s", len(corrections.project.families),
        len(corrections.project.primary), len(corrections.lamellae), path,
    )
    return corrections


def document_of(corrections: ProjectFamilies) -> dict[str, Any]:
    """Return *corrections* as the mapping the file holds below its version.

    What an export record copies too, so the two are written alike. A lamella
    section with no entries is left out.
    """
    document: dict[str, Any] = {
        section: dict(sorted(entries.items(), key=_lowered))
        for section, entries in sections_of(corrections.project)
    }
    lamellae = {
        key: {
            section: dict(sorted(entries.items(), key=_lowered))
            for section, entries in sections_of(own)
        }
        for key, own in sorted(corrections.lamellae.items())
        if own.families or own.primary
    }
    if lamellae:
        document[_LAMELLAE_KEY] = lamellae
    return document


def families_from_document(document: Any, where: str) -> ProjectFamilies:
    """Return the corrections :func:`document_of` wrote, refusing anything else.

    None reads as no corrections.

    Raises:
        FamiliesFileError: Prefixed with *where*, if *document* holds an unknown
            section or anything that is not a name mapped to a name, or an
            entry :func:`~.layout.correction_problems` refuses.
    """
    if document is None:
        return NO_PROJECT_FAMILIES
    if not isinstance(document, dict):
        raise FamiliesFileError(f"Expected a mapping at the top level of {where}")
    unknown = sorted(str(key) for key in document if key not in _DOCUMENT_KEYS)
    if unknown:
        raise FamiliesFileError(
            f"{where}: unknown section {', '.join(unknown)}; a families file holds "
            f"{', '.join(sorted({*_DOCUMENT_KEYS, _VERSION_KEY}))}"
        )
    families = _mapping(document.get(FAMILIES_SECTION), FAMILIES_SECTION, where)
    primary = _mapping(document.get(PRIMARY_SECTION), PRIMARY_SECTION, where)
    lamellae = _lamellae(document.get(_LAMELLAE_KEY), where)
    try:
        return ProjectFamilies(
            project=FamilyCorrections(families=families, primary=primary), lamellae=lamellae
        )
    except ValueError as error:
        raise FamiliesFileError(f"{where}: {error}") from error


def sections_of(
    corrections: FamilyCorrections | FamilyEntries,
) -> tuple[tuple[str, Mapping[str, str]], ...]:
    """Return a project's or a lamella's two sections, each named as the file names it."""
    return (
        (FAMILIES_SECTION, corrections.families),
        (PRIMARY_SECTION, corrections.primary),
    )


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
    document = document_of(corrections)
    version = FILE_VERSION if _LAMELLAE_KEY in document else _FIRST_VERSION
    if not write_yaml_unchanged(
        path, {_VERSION_KEY: version, **document}, header=_HEADER, expected=expected
    ):
        raise stale
    return path


def _lowered(item: tuple[str, str]) -> str:
    return item[0].lower()


def families_for(path: Path) -> FamilyCorrections:
    """Return the corrections in force at *path*: its project's, and its lamella's over them.

    Lets an image path alone find its label. The project is the nearest
    directory at or above *path* marked as one; outside any project there are
    none. A batch asks a :class:`FamiliesLookup` instead.

    Raises:
        FamiliesFileError: As :func:`read_families`.
    """
    return FamiliesLookup().at(path)


class FamiliesLookup:
    """The corrections in force anywhere, each project's file read once.

    Hold one no longer than a batch: nothing notices a file changing.

    Args:
        given: Corrections to use for a project in place of its file, keyed by
            its root, as a merge plans a tree under families on neither disk
            yet. Honoured whichever directory below the root is asked about
            (``test_a_project_given_corrections_is_never_read``).
    """

    def __init__(self, given: Mapping[Path, ProjectFamilies] | None = None) -> None:
        self._projects: dict[Path, ProjectFamilies] = dict(given or {})
        self._roots: dict[Path, Path | None] = {}

    def at(self, path: Path) -> FamilyCorrections:
        """Return the corrections in force at *path*, as :func:`families_for` does.

        Raises:
            FamiliesFileError: As :func:`read_families`.
        """
        start = path if path.is_dir() else path.parent
        root = self._root_of(start)
        if root is None:
            return NO_PROJECT_FAMILIES.project
        if root not in self._projects:
            self._projects[root] = read_families(root)
        return self._projects[root].at(start.relative_to(root).as_posix())

    def _root_of(self, start: Path) -> Path | None:
        """Return the project root at or above *start*, or None outside any project."""
        if start not in self._roots:
            self._roots[start] = next(
                (
                    directory for directory in (start, *start.parents)
                    if directory in self._projects or project_markers(directory)
                ),
                None,
            )
        return self._roots[start]


def _check_version(version: Any, path: Path) -> None:
    """Reject a file written by a later am-seg; accept one that omits it."""
    problem = version_problem(version, path, FILE_VERSION)
    if problem is not None:
        raise FamiliesFileError(problem)


def _lamellae(block: Any, where: str) -> dict[str, FamilyEntries]:
    """Return each lamella's section, refusing anything else."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise FamiliesFileError(f"{where}: {_LAMELLAE_KEY} must be a mapping of lamella to sections")
    sections: dict[str, FamilyEntries] = {}
    for key, section in block.items():
        section_name = f"{_LAMELLAE_KEY}: {key}"
        if not isinstance(key, str) or not isinstance(section, dict):
            raise FamiliesFileError(f"{where}: {section_name} must be a lamella's path holding sections")
        unknown = sorted(str(name) for name in section if name not in _SECTION_KEYS)
        if unknown:
            raise FamiliesFileError(
                f"{where}: {section_name}: unknown section {', '.join(unknown)}; a lamella holds "
                f"{', '.join(sorted(_SECTION_KEYS))}"
            )
        sections[key] = FamilyEntries(
            families=_mapping(
                section.get(FAMILIES_SECTION), f"{section_name}: {FAMILIES_SECTION}", where
            ),
            primary=_mapping(
                section.get(PRIMARY_SECTION), f"{section_name}: {PRIMARY_SECTION}", where
            ),
        )
    return sections


def _mapping(block: Any, name: str, where: str) -> dict[str, str]:
    """Return a section of names to names, refusing anything else."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise FamiliesFileError(f"{where}: {name} must be a mapping of name to name")
    for key, value in block.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise FamiliesFileError(
                f"{where}: {name}: {key!r}: {value!r} is not a name mapped to a name"
            )
    return dict(block)
