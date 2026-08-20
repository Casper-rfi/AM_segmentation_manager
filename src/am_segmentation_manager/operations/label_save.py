"""Writing proofread labels back to disk.

The whole batch is validated before any of it is written, so a save cannot stop
partway leaving the reviewer unable to say which labels reached the disk. Each
file is then written through a temporary and renamed into place, so an
interrupted run leaves the label that was there untouched rather than truncated.

Two things stop a write on their own. A label whose file has changed on disk
since it was read would be overwritten blind, and a label already in
``corrected/`` is proofread work, which nothing here touches without the
reviewer saying so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.atomic import fingerprint_of, write_atomically
from .label_io import save_label_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "LabelToSave",
    "SavePlan",
    "SaveReport",
    "StaleLabels",
    "plan_saves",
    "save_labels",
]


class StaleLabels(RuntimeError):
    """Raised when a label's file changed after the reviewer opened it.

    Nothing is written: which of the two versions is wanted is not a question
    this package can answer, and overwriting is the answer that cannot be
    undone.
    """


@dataclass(frozen=True)
class LabelToSave:
    """One edited label, and what was true of its file when it was opened.

    Attributes:
        destination: Where to write it.
        label: The edited array.
        created: Whether no file existed when the reviewer started, so saving
            writes a new one rather than replacing.
        fingerprint: The destination's size and modification time when it was
            read, or None if it did not exist. Compared before writing, so a
            file something else has changed since is not overwritten blind.
        corrected: Whether the group owning this label is declared corrected.
            Carried on the item because the destination path no longer says.
    """

    destination: Path
    label: np.ndarray
    created: bool = False
    fingerprint: tuple[int, int] | None = None
    corrected: bool = False


@dataclass(frozen=True)
class SavePlan:
    """What a save would write, and what is holding it up.

    Attributes:
        writes: Labels that can be written as things stand.
        corrected: Labels whose destination is already proofread. Written only
            when the reviewer confirms replacing that work.
        stale: Labels whose file changed since it was read. Never written.
    """

    writes: tuple[LabelToSave, ...] = ()
    corrected: tuple[LabelToSave, ...] = ()
    stale: tuple[LabelToSave, ...] = ()

    @property
    def count(self) -> int:
        """Return how many labels would be written, given confirmation."""
        return len(self.writes) + len(self.corrected)

    @property
    def created(self) -> int:
        """Return how many of them are labels that did not exist before."""
        return sum(
            1 for item in (*self.writes, *self.corrected) if item.created
        )

    def summary(self) -> str:
        """Return a one-line description of what the save would do."""
        parts = [f"{self.count} label{'s' if self.count != 1 else ''}"]
        if self.created:
            parts.append(f"{self.created} new")
        if self.corrected:
            parts.append(f"{len(self.corrected)} already corrected")
        if self.stale:
            parts.append(f"{len(self.stale)} changed on disk")
        return ", ".join(parts)


@dataclass(frozen=True)
class SaveReport:
    """What a save actually did.

    Attributes:
        written: Every destination written.
        created: Those that did not exist before.
    """

    written: tuple[Path, ...] = ()
    created: tuple[Path, ...] = ()

    @property
    def count(self) -> int:
        """Return how many labels were written."""
        return len(self.written)

    def headline(self) -> str:
        """Return the line for the window's log."""
        if not self.written:
            return "nothing to save"
        new = f", {len(self.created)} new" if self.created else ""
        return f"wrote {len(self.written)} label{'s' if self.count != 1 else ''}{new}"


def plan_saves(items) -> SavePlan:
    """Sort *items* into what can be written, what needs consent, and what cannot.

    Args:
        items: The :class:`LabelToSave` entries a save would cover.
    """
    writes: list[LabelToSave] = []
    corrected: list[LabelToSave] = []
    stale: list[LabelToSave] = []

    for item in items:
        if fingerprint_of(item.destination) != item.fingerprint:
            logger.warning(
                "Label %s changed on disk since it was opened", item.destination
            )
            stale.append(item)
        elif item.corrected:
            corrected.append(item)
        else:
            writes.append(item)

    return SavePlan(tuple(writes), tuple(corrected), tuple(stale))


def save_labels(items, *, allow_corrected: bool = False) -> SaveReport:
    """Write every label in *items*, or none of them.

    Args:
        items: The :class:`LabelToSave` entries to write.
        allow_corrected: Confirm replacing labels that are already proofread.

    Returns:
        What was written.

    Raises:
        StaleLabels: If any label's file changed after it was opened.
        PermissionError: If any destination is already corrected and
            *allow_corrected* was not set.
    """
    plan = plan_saves(items)

    if plan.stale:
        names = ", ".join(item.destination.name for item in plan.stale)
        raise StaleLabels(
            f"Refusing to save: {len(plan.stale)} label(s) changed on disk "
            f"since they were opened ({names}). Re-open the lamella to see "
            f"what is there now."
        )
    if plan.corrected and not allow_corrected:
        names = ", ".join(item.destination.name for item in plan.corrected)
        raise PermissionError(
            f"Refusing to replace {len(plan.corrected)} proofread label(s) "
            f"without explicit confirmation ({names})"
        )

    written: list[Path] = []
    created: list[Path] = []
    for item in (*plan.writes, *plan.corrected):
        write_atomically(
            item.destination,
            lambda temporary, item=item: save_label_image(
                item.label, temporary, item.label.dtype
            ),
        )
        written.append(item.destination)
        if item.created:
            created.append(item.destination)
        logger.info("Wrote label %s", item.destination)

    return SaveReport(tuple(written), tuple(created))
