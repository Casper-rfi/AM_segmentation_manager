"""Putting a selection of labels back to what they held before an operation.

The counterpart to trying a new model over labels somebody has already
proofread: convert a project's predictions, look at a few, decide the model is
not good enough, and put the whole project back in one command rather than a
label at a time.

**Only what an operation replaced can be put back**, so this reads the history
for the version each label held before the run being undone, and finds it in
the archive. A label whose earlier version was never kept, or whose pixels are
not here, is reported and left alone -- there is nothing to put back, and
guessing at one is the failure this whole feature exists to prevent.

Every write goes through :func:`~.archive.restore` and so through the ordinary
save, which archives what it replaces on the way past. Undoing a revert is
therefore another revert.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..history import Recorder, containing_project, events_for, read_log
from ..models import ImageRecord
from .archive import ArchivedVersion, label_versions, restore

logger = logging.getLogger(__name__)

__all__ = [
    "UNDOABLE",
    "RevertItem",
    "RevertPlan",
    "RevertReport",
    "apply_revert",
    "plan_revert",
]

#: The operations this can undo, and what each is called on the command line.
#: A conversion is the one that matters -- it is how a new model's guess gets
#: over proofread work -- and a merge is the same shape of change.
UNDOABLE: Final[dict[str, str]] = {
    "conversion": "label-converted",
    "merge": "merged",
}

#: Why a label cannot be put back.
NO_ENTRY: Final[str] = "no such operation is recorded for it"
NOT_ARCHIVED: Final[str] = "the operation kept no earlier version"
NOT_HERE: Final[str] = "the earlier version's pixels are not in this collection"
SUPERSEDED: Final[str] = "it has been edited since, so there is newer work to lose"


@dataclass(frozen=True)
class RevertItem:
    """One label a revert would put back, or could not.

    Attributes:
        label_path: The label as it is now.
        target: Its key, relative to the project root.
        version: The version to put back, or None where there is none.
        corrected: Whether its group is declared corrected.
        blocked: Why it cannot be put back, or empty where it can.
    """

    label_path: Path
    target: str
    version: ArchivedVersion | None
    corrected: bool
    blocked: str = ""

    @property
    def can_revert(self) -> bool:
        """Return whether this one would be put back."""
        return self.version is not None and not self.blocked


@dataclass(frozen=True)
class RevertPlan:
    """What putting a selection back would do, having done none of it.

    Attributes:
        undoing: The entry kind being undone.
        items: Every label considered, in a stable order.
    """

    undoing: str
    items: tuple[RevertItem, ...] = ()

    @property
    def revertible(self) -> tuple[RevertItem, ...]:
        """Return the labels that would be put back."""
        return tuple(item for item in self.items if item.can_revert)

    @property
    def blocked(self) -> tuple[RevertItem, ...]:
        """Return the labels that would be left alone, and why."""
        return tuple(item for item in self.items if item.blocked)

    @property
    def corrected(self) -> tuple[RevertItem, ...]:
        """Return the revertible labels whose group is proofread work."""
        return tuple(item for item in self.revertible if item.corrected)

    def describe(self) -> list[str]:
        """Return one line per label, saying what would happen to it."""
        lines = []
        for item in self.items:
            if item.can_revert:
                when = item.version.event.at if item.version.event else "unrecorded"
                lines.append(f"{item.target}: back to the version from {when}")
            else:
                lines.append(f"{item.target}: left alone, {item.blocked}")
        return lines


@dataclass(frozen=True)
class RevertReport:
    """What a revert actually put back."""

    restored: tuple[Path, ...] = ()
    failed: tuple[tuple[Path, str], ...] = ()


def plan_revert(
    records: Iterable[ImageRecord], undoing: str, *, allow_superseded: bool = False
) -> RevertPlan:
    """Return what putting *records* back before *undoing* would do.

    Args:
        records: The images whose labels to consider. Several conditions
            sharing one label contribute it once.
        undoing: A key of :data:`UNDOABLE`.
        allow_superseded: Put a label back even where somebody has edited it
            since the operation being undone, which discards that editing.

    Raises:
        ValueError: If *undoing* is not something this can undo.
    """
    kind = UNDOABLE.get(undoing)
    if kind is None:
        raise ValueError(
            f"{undoing!r} is not something this can undo; try "
            f"{' or '.join(sorted(UNDOABLE))}"
        )

    from .label_groups import build_groups
    from .selection import current_state

    items: list[RevertItem] = []
    logs: dict[Path, object] = {}
    for group in build_groups(list(records)):
        label_path = group.label_path
        if label_path is None:
            continue
        project = containing_project(label_path)
        if project is None:
            continue
        try:
            target = label_path.resolve().relative_to(project).as_posix()
        except ValueError:
            continue
        if project not in logs:
            logs[project] = read_log(project)

        version, blocked = _version_before(
            logs[project], project, target, kind, allow_superseded=allow_superseded
        )
        items.append(
            RevertItem(
                label_path=label_path,
                target=target,
                version=version,
                corrected=current_state(group.records[0]) == "corrected",
                blocked=blocked,
            )
        )
    items.sort(key=lambda item: item.target)
    logger.debug("Revert plan: %s of %s label(s) can go back",
                 sum(1 for item in items if item.can_revert), len(items))
    return RevertPlan(undoing=kind, items=tuple(items))


def _version_before(
    log, project: Path, target: str, kind: str, *, allow_superseded: bool
) -> tuple[ArchivedVersion | None, str]:
    """Return the version *kind* replaced for *target*, or why there is none."""
    entries = events_for(log, target)
    latest = None
    for index, event in enumerate(entries):
        if event.kind == kind:
            latest = index
    if latest is None:
        return None, NO_ENTRY

    # Anything after it changed the label again, so putting the old one back
    # would discard that too. Said rather than done.
    if not allow_superseded and any(
        event.kind in ("label-saved", "label-restored")
        for event in entries[latest + 1:]
    ):
        return None, SUPERSEDED

    archived = entries[latest].data.get("archived_as")
    if not isinstance(archived, str) or not archived:
        return None, NOT_ARCHIVED

    for version in label_versions(project, target):
        if version.path == project / archived:
            if not version.present:
                return None, NOT_HERE
            return version, ""
    return None, NOT_HERE


def apply_revert(
    plan: RevertPlan, *, recorder: Recorder, allow_corrected: bool = False
) -> RevertReport:
    """Put back every label *plan* can, or none of them.

    Raises:
        PermissionError: If any label is proofread work and *allow_corrected*
            was not set. Checked over the whole batch before the first write,
            so the answer is "nothing happened" rather than "some of it did".
    """
    if plan.corrected and not allow_corrected:
        names = ", ".join(item.target for item in plan.corrected[:3])
        raise PermissionError(
            f"Refusing to put back {len(plan.corrected)} proofread label(s) "
            f"without explicit confirmation ({names})"
        )

    restored: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for item in plan.revertible:
        try:
            restore(
                item.version, item.label_path,
                recorder=recorder,
                corrected=item.corrected,
                allow_corrected=allow_corrected,
            )
            restored.append(item.label_path)
        except Exception as error:
            logger.warning("Could not put %s back: %s", item.target, error)
            failed.append((item.label_path, str(error)))

    logger.info("Put %s label(s) back", len(restored))
    return RevertReport(restored=tuple(restored), failed=tuple(failed))
