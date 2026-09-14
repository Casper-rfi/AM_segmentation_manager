"""Writing proofread labels back to disk.

The whole batch is validated before any of it is written, so a save cannot stop
partway leaving the reviewer unable to say which labels reached the disk. Each
file is then written through a temporary and renamed into place, so an
interrupted run leaves the label that was there untouched rather than truncated.

Two things stop a write on their own. A label whose file has changed on disk
since it was read would be overwritten blind, and a label whose group is
declared corrected is proofread work, which nothing here touches without the
reviewer saying so.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..io.atomic import fingerprint_of, write_atomically
from .archive import Archivist, archive_replaced
from .label_io import label_fields, read_label, replacement_fields, save_label_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ..history import Recorder

logger = logging.getLogger(__name__)

__all__ = [
    "LabelToSave",
    "WriteProvenance",
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
class WriteProvenance:
    """How a write is to be recorded, where it is not an ordinary save.

    Restoring a version and taking one from a merge are the *same write* as
    proofreading one -- same validation, same staleness refusal, same
    protection of proofread work, same archiving of what is replaced. Only the
    account of them differs, so that difference is carried here rather than
    inferred from which optional field happens to be set.

    Attributes:
        kind: The entry kind to record. One of :data:`~..history.KINDS`.
        fields: What that kind carries beyond what the written pixels say.
    """

    kind: str
    fields: Mapping[str, Any] = field(default_factory=dict)


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
        provenance: How to record this write, or None for an ordinary save.
    """

    destination: Path
    label: np.ndarray
    created: bool = False
    fingerprint: tuple[int, int] | None = None
    corrected: bool = False
    provenance: WriteProvenance | None = None


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
        archived: The versions kept on the way past, relative to their project.
    """

    written: tuple[Path, ...] = ()
    created: tuple[Path, ...] = ()
    archived: tuple[str, ...] = ()

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


def save_labels(
    items, *, allow_corrected: bool = False, recorder: Recorder | None = None
) -> SaveReport:
    """Write every label in *items*, or none of them.

    Args:
        items: The :class:`LabelToSave` entries to write.
        allow_corrected: Confirm replacing labels that are already proofread.
        recorder: Where to record what was written, or None to record nothing.
            The version being replaced is read before it is overwritten, so
            that an entry can name what this one grew out of -- work done only
            when there is somewhere to record it.

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
            f"Refusing to replace {len(plan.corrected)} corrected label(s) "
            f"without explicit confirmation ({names})"
        )

    archivist = None
    if recorder is not None:

        archivist = Archivist(recorder)

    written: list[Path] = []
    created: list[Path] = []
    kept: list[str] = []
    for item in (*plan.writes, *plan.corrected):
        previous = _previous_pixels(item) if recorder is not None else None
        archived = archive_replaced(
            archivist, item.destination, previous, replacement=item.label
        )
        if archived is not None:
            kept.append(archived)
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
        if recorder is not None:
            _record_save(recorder, item, previous, archived)

    return SaveReport(tuple(written), tuple(created), tuple(kept))


def _previous_pixels(item: LabelToSave) -> np.ndarray | None:
    """Return the pixels *item* is about to replace, or None where it replaces nothing.

    A label that will not open is not a reason to refuse the save: the entry
    then says what was written without claiming what it grew out of.
    """
    if item.created or not item.destination.is_file():
        return None
    return read_label(item.destination)


def _record_save(
    recorder: Recorder, item: LabelToSave, previous, archived: str | None
) -> None:
    """Record one written label, what it replaced, and where that went."""
    try:
        fields = label_fields(item.label)
    except ValueError as error:
        logger.warning("Not recording %s: %s", item.destination, error)
        return
    fields.update(replacement_fields(item.destination, previous, item.label, archived))

    if item.provenance is None:
        recorder.record(item.destination, "label-saved", **fields)
        return

    fields.update(item.provenance.fields)
    if item.provenance.kind == "label-restored":
        # Its kind requires both, and neither is the caller's to supply: what
        # was replaced is read here, on the way past.
        fields.setdefault("parent_digest", fields["digest"])
        fields.setdefault("archived_as", archived or "")
    recorder.record(item.destination, item.provenance.kind, **fields)
