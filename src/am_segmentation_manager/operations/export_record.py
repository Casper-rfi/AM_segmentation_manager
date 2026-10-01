"""The fork point an export writes, so a merge back is not a guess.

``AM_export.yaml`` at a copy's root names, per project, the last history entry it
held when copied: the common ancestor a three-way merge needs. **The tip is an
entry id, and a merge verifies it**: one missing from the collection's log means
the fork point does not describe these two trees, and the merge is refused. The
entry count lets a mismatch say which way the two have drifted. **The project's
family corrections are copied in as they stood**, being the ancestor a merge of
``AM_families.yaml`` compares against. Nothing here reads or writes a label.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .. import __version__
from ..families import read_families
from ..history import declared_project_id, read_log, resolve_author
from ..io.yaml import read_yaml, write_yaml
from ..layout import NO_PROJECT_FAMILIES, FamilyCorrections, FamilyEntries, ProjectFamilies
from ..models import Project

logger = logging.getLogger(__name__)

__all__ = [
    "AM_EXPORT_FILENAME",
    "FILE_VERSION",
    "ExportRecord",
    "ProjectFork",
    "build_export_record",
    "export_record_path",
    "read_export_record",
    "write_export_record",
]

#: The file naming the fork point, at the root of a copy.
AM_EXPORT_FILENAME: Final[str] = "AM_export.yaml"

#: Written into every file. A reader refuses a version it does not know: unlike
#: a history log, which is appended to and tolerates what it cannot read, this
#: is read once to decide how a merge behaves, and a half-understood fork point
#: is worse than none.
FILE_VERSION: Final[int] = 1

_HEADER: Final[str] = """\
# Where this copy came from, written by am-seg when the copy was made.
#
# `tip` is the last history entry each project held at that moment: the common
# ancestor a merge back compares against. Do not edit -- a merge verifies these
# ids against the collection they name, and one that does not match is refused
# rather than guessed at.
"""


@dataclass(frozen=True)
class ProjectFork:
    """Where one project's history stood when the copy was taken.

    Attributes:
        key: The project's path relative to the scan root, which is its
            identity here as everywhere else.
        project_id: What its log declares itself to be, or None where it keeps
            no log or has never been baselined.
        tip: The id of its last entry, or None where it has none.
        entries: How many entries it held, so a mismatch can say which way the
            two trees have drifted.
        families: Its ``AM_families.yaml`` as it stood, lamella sections
            included. A fork point written before these were recorded holds
            none, which is exact: nothing could write that file then.
    """

    key: str
    project_id: str | None = None
    tip: str | None = None
    entries: int = 0
    families: ProjectFamilies = NO_PROJECT_FAMILIES

    @property
    def has_history(self) -> bool:
        """Return whether a merge has an ancestor to work from for this project."""
        return self.tip is not None


@dataclass(frozen=True)
class ExportRecord:
    """What a copy says about where it came from.

    Attributes:
        exported_at: When the copy was taken, UTC.
        exported_by: Who took it, or empty where nobody stated a name.
        source: The scan root copied from, as it was spelt then. Context for a
            person; never an identity, a collection being movable.
        scope: What the export was asked for, in the words the caller used.
        with_archive: Whether kept label versions came too.
        projects: One entry per project in the copy, keyed by relative path.
        am_seg_version: What wrote it.
    """

    exported_at: str
    exported_by: str
    source: str
    scope: str = ""
    with_archive: bool = False
    projects: Mapping[str, ProjectFork] = field(default_factory=dict)
    am_seg_version: str = __version__

    @property
    def without_history(self) -> tuple[str, ...]:
        """Return the projects a merge would have no ancestor for."""
        return tuple(
            key for key, fork in sorted(self.projects.items()) if not fork.has_history
        )


def export_record_path(destination: Path) -> Path:
    """Return where a copy's fork point lives, existing or not."""
    return destination / AM_EXPORT_FILENAME


def build_export_record(
    projects: Iterable[Project],
    source: Path,
    *,
    scope: str = "",
    with_archive: bool = False,
    user: str | None = None,
) -> ExportRecord:
    """Return the fork point for a copy of *projects*, reading their logs.

    A project keeping no log contributes an entry saying so rather than being
    left out: a merge must be able to tell a project it has no ancestor for
    from one the copy never held. **Pass only the projects the copy actually
    holds** -- a fork point naming one that is not there has a merge look for
    an ancestor in a tree that does not exist.

    Raises:
        FamiliesFileError: If a project's families file will not read, since a
            copy whose ancestor is unknown could not be merged back exactly.
    """
    from datetime import UTC, datetime

    by, _ = resolve_author(user)
    forks = {}
    for project in projects:
        log = read_log(project.path)
        tip = log.events[-1].id if log.events else None
        forks[project.key] = ProjectFork(
            key=project.key,
            project_id=declared_project_id(log.events),
            tip=tip,
            entries=len(log.events),
            families=read_families(project.path),
        )
    return ExportRecord(
        exported_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        exported_by=by or "",
        source=str(source),
        scope=scope,
        with_archive=with_archive,
        projects=forks,
    )


def write_export_record(destination: Path, record: ExportRecord) -> Path:
    """Write *record* at the root of the copy, and return where it went."""
    body = {
        "version": FILE_VERSION,
        "am_seg_version": record.am_seg_version,
        "exported_at": record.exported_at,
        "exported_by": record.exported_by,
        "source": record.source,
        "scope": record.scope,
        "with_archive": record.with_archive,
        "projects": {
            key: {
                "project_id": fork.project_id,
                "tip": fork.tip,
                "entries": fork.entries,
                **(
                    {"families": _families_body(fork.families)}
                    if fork.families != NO_PROJECT_FAMILIES
                    else {}
                ),
            }
            for key, fork in sorted(record.projects.items())
        },
    }
    path = export_record_path(destination)
    write_yaml(path, body, header=_HEADER)
    logger.debug("Wrote the fork point to %s", path)
    return path


def read_export_record(destination: Path) -> ExportRecord | None:
    """Return the fork point at the root of *destination*, or None if there is none.

    Raises:
        ValueError: If the file cannot be understood, or was written by a later
            am-seg. A fork point half-read would have a merge compare against
            the wrong ancestor, which is worse than having none.
    """
    path = export_record_path(destination)
    if not path.is_file():
        return None

    try:
        document = read_yaml(path)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    if not isinstance(document, dict):
        raise ValueError(f"{path} does not hold a mapping")

    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{path}: version must be a whole number")
    if version > FILE_VERSION:
        raise ValueError(
            f"{path} was written by a later version of am-seg "
            f"(file version {version}, this reads up to {FILE_VERSION})"
        )

    return ExportRecord(
        exported_at=_text(document.get("exported_at")),
        exported_by=_text(document.get("exported_by")),
        source=_text(document.get("source")),
        scope=_text(document.get("scope")),
        with_archive=bool(document.get("with_archive")),
        projects=_forks(document.get("projects"), path),
        am_seg_version=_text(document.get("am_seg_version")),
    )


def _forks(block: Any, path: Path) -> dict[str, ProjectFork]:
    """Return the per-project entries of a fork point, validated."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{path}: projects must be a mapping")

    forks = {}
    for key, entry in block.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: {key!r} is not a project")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {key} does not hold a mapping")
        entries = entry.get("entries")
        forks[key] = ProjectFork(
            key=key,
            project_id=_optional_text(entry.get("project_id")),
            tip=_optional_text(entry.get("tip")),
            entries=entries if isinstance(entries, int) else 0,
            families=_families(entry.get("families"), f"{path}: {key}"),
        )
    return forks


def _families_body(families: ProjectFamilies) -> dict[str, Any]:
    """Return a project's family corrections as the fork point records them."""
    body: dict[str, Any] = {
        "families": dict(families.project.families),
        "primary": dict(families.project.primary),
    }
    if families.lamellae:
        body["lamellae"] = {
            key: {"families": dict(entries.families), "primary": dict(entries.primary)}
            for key, entries in sorted(families.lamellae.items())
        }
    return body


def _families(block: Any, where: str) -> ProjectFamilies:
    """Return a project's recorded family corrections, validated."""
    if block is None:
        return NO_PROJECT_FAMILIES
    try:
        families, primary = _sections(block)
        lamellae_block = block.get("lamellae") or {}
        if not isinstance(lamellae_block, dict):
            raise ValueError("lamellae must map a lamella to its sections")
        lamellae = {}
        for key, section in lamellae_block.items():
            lamella_families, lamella_primary = _sections(section)
            lamellae[str(key)] = FamilyEntries(families=lamella_families, primary=lamella_primary)
        return ProjectFamilies(
            project=FamilyCorrections(families=families, primary=primary), lamellae=lamellae
        )
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


def _sections(block: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Return the families and primary mappings of one recorded section."""
    if not isinstance(block, dict) or not all(
        isinstance(block.get(name) or {}, dict) for name in ("families", "primary")
    ):
        raise ValueError("families must hold a families and a primary mapping")
    return dict(block.get("families") or {}), dict(block.get("primary") or {})


def _text(value: Any) -> str:
    """Return *value* as a string, or empty where it is not one."""
    return value if isinstance(value, str) else ""


def _optional_text(value: Any) -> str | None:
    """Return *value* as a string, or None where it is not a usable one."""
    return value if isinstance(value, str) and value else None
