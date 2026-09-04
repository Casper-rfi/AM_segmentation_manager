"""The fork point an export writes, so a merge back is not a guess.

``AM_export.yaml`` sits at the root of a copy and names, per project, the last
history entry that project held when the copy was taken. That is the common
ancestor a three-way merge needs: with it, a merge can tell "only the copy
changed this" from "both did" without inferring anything.

**The tip is an entry id, and a merge verifies it.** A copy that has itself been
exported from, or been reorganised, still carries a plausible-looking file; an
id that is not in the main collection's log means the fork point does not
describe these two trees, and refusing is the only safe answer. Naming the
count as well as the id lets a mismatch say which way the two have drifted.

Nothing here reads or writes a label. It records what was true of the logs.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .. import __version__
from ..history import read_log, resolve_author
from ..io.atomic import write_atomically
from ..io.yaml import read_yaml
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
    """

    key: str
    project_id: str | None = None
    tip: str | None = None
    entries: int = 0

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
    """
    from datetime import UTC, datetime

    by, _ = resolve_author(user)
    forks = {}
    for project in projects:
        log = read_log(project.path)
        tip = log.events[-1].id if log.events else None
        forks[project.key] = ProjectFork(
            key=project.key,
            project_id=_declared_id(log.events),
            tip=tip,
            entries=len(log.events),
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
    import yaml

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
            }
            for key, fork in sorted(record.projects.items())
        },
    }
    path = export_record_path(destination)
    text = _HEADER + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    write_atomically(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
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
        forks[key] = ProjectFork(
            key=key,
            project_id=_optional_text(entry.get("project_id")),
            tip=_optional_text(entry.get("tip")),
            entries=entry.get("entries") if isinstance(entry.get("entries"), int) else 0,
        )
    return forks


def _declared_id(events: Iterable) -> str | None:
    """Return the identity a log declares, if it declares one."""
    for event in events:
        if event.kind == "log-created":
            declared = event.data.get("project_id")
            if isinstance(declared, str) and declared:
                return declared
    return None


def _text(value: Any) -> str:
    """Return *value* as a string, or empty where it is not one."""
    return value if isinstance(value, str) else ""


def _optional_text(value: Any) -> str | None:
    """Return *value* as a string, or None where it is not a usable one."""
    return value if isinstance(value, str) and value else None
