"""Writing proofread labels back to disk.

The whole batch is validated before any of it is written, and each file goes
through a temporary renamed into place, so no label is left truncated. A write
failing partway raises :class:`SaveStopped` carrying what was written before
it, so a caller can settle those. Two things stop a write on their own: a label whose
file changed on disk since it was read, which would be overwritten blind, and
one whose group is declared corrected -- in its state file now, not only when it
was opened -- which nothing here touches without the reviewer saying so.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..io.atomic import Fingerprint, fingerprint_of, write_atomically
from ..io.image import read_array
from ..io.long_paths import refuse_unreachable
from ..progress import StoppedPartway
from .archive import ArchivedVersion, Archivist, archive_paths_for, archive_replaced
from .label_io import label_fields, read_label, replacement_fields, save_label_image
from .selection import declared_states

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ..history import Recorder

logger = logging.getLogger(__name__)

__all__ = [
    "LabelToSave",
    "WriteProvenance",
    "SavePlan",
    "SaveReport",
    "SaveStopped",
    "StaleLabels",
    "plan_saves",
    "save_labels",
    "restore",
]


class SaveStopped(StoppedPartway):
    """Raised when a save stopped partway, having written what it reports and nothing after.

    Attributes:
        report: What was written before it stopped, which a caller holding
            those labels has to bring into step as after any save.
    """

    def __init__(self, message: str, report: SaveReport) -> None:
        super().__init__(message, report.written)
        self.report = report


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
        fingerprint: The destination's fingerprint when it was read, or None
            if it did not exist. Compared before writing, so a
            file something else has changed since is not overwritten blind.
        corrected: Whether the group owning this label is declared corrected.
            Carried on the item because the destination path no longer says.
        provenance: How to record this write, or None for an ordinary save.
    """

    destination: Path
    label: np.ndarray
    created: bool = False
    fingerprint: Fingerprint | None = None
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

    A label is corrected if its item says so or its stage's state file does
    now: the item carries what was known when the label was opened, and a
    group marked corrected since must still be asked about.

    Args:
        items: The :class:`LabelToSave` entries a save would cover.

    Raises:
        StateFileError: If a stage's state file cannot be read, so whether a
            label there is proofread cannot be told.
    """
    items = list(items)
    declared = declared_states(item.destination for item in items)
    writes: list[LabelToSave] = []
    corrected: list[LabelToSave] = []
    stale: list[LabelToSave] = []

    for item in items:
        if fingerprint_of(item.destination) != item.fingerprint:
            logger.warning(
                "Label %s changed on disk since it was opened", item.destination
            )
            stale.append(item)
        elif item.corrected or declared[item.destination] == "corrected":
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
        StateFileError: If a stage's state file cannot be read. Nothing is
            written.
        PathTooLong: If Windows cannot reach a label or the archived copy of
            what it replaces here. Checked before anything is written.
        SaveStopped: If a label could not be written, or the version it
            replaces could not be kept. That label and every one after it are
            left as they were; the error carries what was written before it.
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

    destinations = [item.destination for item in (*plan.writes, *plan.corrected)]
    refuse_unreachable([*destinations, *archive_paths_for(recorder, destinations)])

    archivist = None
    if recorder is not None:

        archivist = Archivist(recorder)

    written: list[Path] = []
    created: list[Path] = []
    kept: list[str] = []
    batch = (*plan.writes, *plan.corrected)
    for item in batch:
        previous = _previous_pixels(item) if recorder is not None else None

        def write(temporary: Path, item: LabelToSave = item) -> None:
            save_label_image(item.label, temporary, item.label.dtype)

        def unchanged(temporary: Path, item: LabelToSave = item) -> bool:
            return fingerprint_of(item.destination) == item.fingerprint

        # Past the first write, a failure of any kind must say what was written:
        # a caller still holding those labels as unsaved would next find them
        # "changed on disk" and be told to discard the edits it has not saved.
        try:
            archived = archive_replaced(
                archivist, item.destination, previous, replacement=item.label
            )
            if archived is not None:
                kept.append(archived)
            # Asked again just before the rename: the plan's check is as old as
            # the batch, and a label rewritten since is not to be overwritten.
            if not write_atomically(item.destination, write, commit_if=unchanged):
                raise StaleLabels(
                    f"{item.destination.name} changed on disk while the save was "
                    f"running, so it was not written over"
                )
        except Exception as error:
            raise SaveStopped(
                f"Saved {len(written)} of {len(batch)} label(s), then stopped: {error}",
                SaveReport(tuple(written), tuple(created), tuple(kept)),
            ) from error
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


def restore(
    version: ArchivedVersion,
    destination: Path,
    *,
    recorder: Recorder,
    corrected: bool = False,
    allow_corrected: bool = False,
) -> Path:
    """Put an archived version back, through the ordinary save.

    **Not a new way to write a label.** It builds the same item an editor would
    and hands it to :func:`save_labels`, so the batch validation,
    the refusal of a file that changed since it was read, and the protection of
    proofread work all apply unchanged -- and the version being replaced is
    itself archived on the way past. Only what is recorded differs.

    Raises:
        RuntimeError: If the archived file cannot be read.
        StaleLabels: If *destination* changed since it was read.
        PermissionError: If it is proofread work and *allow_corrected* is not
            set.
    """
    pixels = read_array(version.path)
    digest = label_fields(pixels)["digest"]
    save_labels(
        [
            LabelToSave(
                destination=destination,
                label=pixels,
                created=not destination.exists(),
                fingerprint=fingerprint_of(destination),
                corrected=corrected,
                provenance=WriteProvenance(
                    "label-restored", {"restored_from": str(digest)}
                ),
            )
        ],
        allow_corrected=allow_corrected,
        recorder=recorder,
    )
    logger.info("Restored %s from %s", destination, version.path.name)
    return destination
