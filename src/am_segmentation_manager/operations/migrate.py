"""Renaming pre-convention label directories to the current naming.

Older datasets name their label directories ``labels`` and ``labels_fib`` (or
``labels_FIB``). The current convention is ``<image-directory>_labels`` beside
the images, which makes the image-to-label mapping a pure string function in
both directions rather than a lookup table.

Both names are read indefinitely, so migrating is optional. It is offered
because a collection using one convention is easier to reason about than one
using two, and because every newly written label already uses the current name
-- leaving the old directories in place means a project slowly acquires both.

The operation is a directory rename and nothing else. No file is read, written,
copied or deleted, and a rename that would collide with something already on
disk aborts the whole batch before anything moves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..discovery import discover_stages, find_project_roots
from ..layout import (
    LEGACY_LABEL_DIRECTORIES,
    build_image_sets,
    shared_labels_directory_name,
    subdirectories_by_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelRename:
    """One label directory to rename.

    Attributes:
        source: The directory as it is named now.
        destination: The name it would take under the current convention.
        modality: The modality its labels belong to.
        image_sets: Names of the image directories that resolve to it, so a
            preview can show what the rename affects.
    """

    source: Path
    destination: Path
    modality: str
    image_sets: tuple[str, ...] = ()

    def describe(self, root: Path | None = None) -> str:
        """Return a one-line description for a preview or a log."""
        from ..io.path import make_path_relative

        location = make_path_relative(self.source, root) if root else self.source
        used_by = ", ".join(self.image_sets) if self.image_sets else "no image set"
        return f"{location} -> {self.destination.name}/  ({self.modality}: {used_by})"


class MigrationConflict(RuntimeError):
    """Raised when a rename cannot proceed safely.

    Carries every conflict found, not just the first, so one run reports all
    the work needed rather than surfacing problems one at a time.
    """

    def __init__(self, conflicts: list[str]) -> None:
        self.conflicts = conflicts
        super().__init__(
            f"{len(conflicts)} label directory rename(s) cannot proceed:\n  "
            + "\n  ".join(conflicts)
        )


def plan_label_migration(root: Path) -> list[LabelRename]:
    """Return the label directory renames needed under *root*.

    Every project below *root* is inspected. A directory already using the
    current convention is left alone, so running this twice is safe and the
    second run finds nothing.

    Raises:
        MigrationConflict: If any rename would collide with a directory that
            already exists. Reported for the whole batch before anything moves,
            because a partly migrated project is harder to reason about than an
            unmigrated one.
    """
    renames: list[LabelRename] = []
    conflicts: list[str] = []

    for project_root in find_project_roots(root):
        for stage in discover_stages(project_root):
            renames.extend(_stage_renames(stage.path, conflicts))

    _check_for_duplicate_destinations(renames, conflicts)
    if conflicts:
        raise MigrationConflict(conflicts)
    logger.info("Planned %d label directory rename(s) under %s", len(renames), root)
    return renames


def _stage_renames(stage_path: Path, conflicts: list[str]) -> list[LabelRename]:
    """Return the renames for one stage, appending any conflicts."""
    siblings = subdirectories_by_name(stage_path)
    image_sets = build_image_sets(stage_path)

    renames: list[LabelRename] = []
    for legacy_name, modality in LEGACY_LABEL_DIRECTORIES.items():
        source = siblings.get(legacy_name)
        if source is None:
            continue
        destination = stage_path / shared_labels_directory_name(modality)
        if destination.exists():
            conflicts.append(
                f"{source}: cannot rename to {destination.name}/ because that "
                f"directory already exists; merge them by hand and re-run"
            )
            continue
        renames.append(
            LabelRename(
                source=source,
                destination=destination,
                modality=modality,
                image_sets=tuple(
                    image_set.name
                    for image_set in image_sets
                    if image_set.labels_directory == source
                ),
            )
        )
    return renames


def _check_for_duplicate_destinations(
    renames: Iterable[LabelRename], conflicts: list[str]
) -> None:
    """Reject two directories that would end up with the same name.

    ``labels_fib`` and ``labels_FIB`` in one stage both map to ``fib_labels``.
    A case-insensitive filesystem cannot hold both, but a case-sensitive one
    can, and silently renaming one over the other would lose data.
    """
    seen: dict[Path, Path] = {}
    for rename in renames:
        first = seen.get(rename.destination)
        if first is not None:
            conflicts.append(
                f"{first} and {rename.source} would both become "
                f"{rename.destination.name}/; merge them by hand and re-run"
            )
            continue
        seen[rename.destination] = rename.source


def apply_label_migration(renames: Iterable[LabelRename]) -> int:
    """Perform the planned renames.

    Validation belongs to :func:`plan_label_migration`; this performs what that
    approved. Each rename is a directory rename, so nothing is copied and no
    file content is touched.

    Returns:
        The number of directories renamed.
    """
    count = 0
    for rename in renames:
        rename.source.rename(rename.destination)
        logger.info("Renamed %s -> %s", rename.source, rename.destination)
        count += 1
    return count
