"""Label edits made in the window and not yet written to disk.

Proofreading is done a lamella at a time, so the session holds every label
touched since the last save, not just the one on screen. Only the label being
worked on is held as an array; the rest are PNG-encoded in memory, which costs
around 14 kB for a label that is 6 MB decoded and keeps a whole lamella of
pending work well under a megabyte.

Nothing here touches the filesystem. A label is identified by the path it will
be written to, which is what makes several imaging conditions of one scene share
one pending edit, as they share one label.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.image import to_2d
from ..operations.label_edit import DEFAULT_UNDO_DEPTH, EditPatch, UndoStack

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: The encoding pending edits are held in. PNG is lossless, carries 16-bit
#: single-channel images, and is decoded by the stack this package already
#: depends on.
ENCODING = ".png"


def encode_label(label: np.ndarray) -> bytes:
    """Return *label* losslessly compressed, for holding in memory.

    Raises:
        RuntimeError: If the array could not be encoded, which would mean a
            pending edit silently disappearing.
    """
    import cv2

    encoded, buffer = cv2.imencode(ENCODING, label)
    if not encoded:
        raise RuntimeError(f"Could not encode a {label.dtype} label of {label.shape}")
    return buffer.tobytes()


def decode_label(data: bytes) -> np.ndarray:
    """Return the label held in *data*, at its original dtype."""
    import cv2
    import numpy as np


    label = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError("Could not decode a pending label edit")
    return to_2d(label)


@dataclass
class PendingLabel:
    """What has happened to one label since it was last read from disk.

    Attributes:
        key: Where the label will be written. Its identity here.
        created: Whether no label file existed, so saving writes a new one.
        fingerprint: What the label's file looked like when it was first
            opened, for the save to check nothing else has rewritten it since.
            Held rather than taken here, so this module stays off the disk.
        edited: Whether a stroke has changed it.
        reviewed: Whether the reviewer has passed it as correct. Set by editing
            it, and settable on its own for a label that needed no change --
            without which the only way to approve a good label would be to make
            a token mark on it.
    """

    key: Path
    created: bool = False
    fingerprint: tuple[int, int] | None = None
    edited: bool = False
    reviewed: bool = False
    undo: UndoStack = field(default_factory=UndoStack)

    @property
    def pending(self) -> bool:
        """Return whether this label has anything waiting to be saved."""
        return self.edited or self.reviewed


class LabelSession:
    """Every label edited in the current lamella, and the one in hand.

    One label is *open* at a time -- the array a brush writes into. Opening
    another puts the previous one away compressed, so the amount held does not
    grow with how much of the lamella has been worked through.
    """

    def __init__(self, *, undo_depth: int = DEFAULT_UNDO_DEPTH) -> None:
        """Create an empty session whose labels each keep *undo_depth* edits."""
        self.undo_depth = undo_depth
        self._pending: dict[Path, PendingLabel] = {}
        self._stored: dict[Path, bytes] = {}
        self._open_key: Path | None = None
        self._open_label: np.ndarray | None = None

    # -- the label in hand --------------------------------------------------

    def open(
        self,
        key: Path,
        source: np.ndarray,
        *,
        created: bool = False,
        fingerprint: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return the working array for *key*, and make it the open one.

        A label with edits pending is restored from them; one without is a copy
        of *source*, so that painting never writes through to the cached frame
        the source came from.

        Args:
            key: Where the label will be written.
            source: The label as it is on disk, or a blank one where no file
                exists yet.
            created: Whether *source* stands in for a file that is not there.
            fingerprint: What the file looked like when it was read. Recorded
                the first time a label is opened and not revised after, so the
                save compares against the state the reviewer started from.
        """
        if key == self._open_key and self._open_label is not None:
            return self._open_label

        self.release()
        entry = self._pending.get(key)
        if entry is None:
            entry = PendingLabel(
                key=key,
                created=created,
                fingerprint=fingerprint,
                undo=UndoStack(self.undo_depth),
            )
            self._pending[key] = entry

        stored = self._stored.pop(key, None)
        self._open_label = decode_label(stored) if stored is not None else source.copy()
        self._open_key = key
        return self._open_label

    def release(self) -> None:
        """Put the open label away, compressed, keeping its pending state.

        A label with nothing pending is dropped instead of stored: holding an
        untouched copy of every frame the reviewer looked at would defeat the
        point of holding anything.
        """
        if self._open_key is None or self._open_label is None:
            return
        entry = self._pending.get(self._open_key)
        if entry is not None and entry.pending:
            self._stored[self._open_key] = encode_label(self._open_label)
        elif entry is not None:
            del self._pending[self._open_key]
        self._open_key = None
        self._open_label = None

    @property
    def open_key(self) -> Path | None:
        """Return the key of the label in hand, or None."""
        return self._open_key

    @property
    def open_label(self) -> np.ndarray | None:
        """Return the array being edited, or None."""
        return self._open_label

    # -- editing ------------------------------------------------------------

    def record(self, patch: EditPatch | None) -> None:
        """Record *patch* as an edit to the open label, ignoring None."""
        if patch is None or self._open_key is None:
            return
        entry = self._pending[self._open_key]
        entry.undo.push(patch)
        entry.edited = True
        entry.reviewed = True

    def undo(self) -> bool:
        """Reverse the open label's most recent edit; return whether one was."""
        return self._step(redo=False)

    def redo(self) -> bool:
        """Reapply the open label's most recently undone edit."""
        return self._step(redo=True)

    def _step(self, *, redo: bool) -> bool:
        if self._open_key is None or self._open_label is None:
            return False
        entry = self._pending[self._open_key]
        moved = (
            entry.undo.redo(self._open_label) if redo
            else entry.undo.undo(self._open_label)
        )
        if moved is None:
            return False
        # Undoing back past every edit leaves the label as it was read, which is
        # not something to save -- but it stays reviewed, because looking at it
        # and deciding it was already right is a judgement, not an edit.
        entry.edited = entry.undo.can_undo
        return True

    @property
    def can_undo(self) -> bool:
        """Return whether the open label has an edit to undo."""
        entry = self._pending.get(self._open_key) if self._open_key else None
        return bool(entry and entry.undo.can_undo)

    @property
    def can_redo(self) -> bool:
        """Return whether the open label has an undone edit to reapply."""
        entry = self._pending.get(self._open_key) if self._open_key else None
        return bool(entry and entry.undo.can_redo)

    # -- what is waiting ----------------------------------------------------

    def mark_reviewed(self, key: Path, *, reviewed: bool = True) -> None:
        """Record that *key* has been proofread, whether or not it was changed."""
        entry = self._pending.get(key)
        if entry is None:
            if not reviewed:
                return
            entry = PendingLabel(key=key, undo=UndoStack(self.undo_depth))
            self._pending[key] = entry
        entry.reviewed = reviewed

    def state_of(self, key: Path) -> PendingLabel | None:
        """Return what is pending for *key*, or None if nothing is."""
        return self._pending.get(key)

    def is_edited(self, key: Path) -> bool:
        """Return whether *key* has unsaved changes."""
        entry = self._pending.get(key)
        return bool(entry and entry.edited)

    def edited_keys(self) -> set[Path]:
        """Return the labels holding unsaved changes."""
        return {key for key, entry in self._pending.items() if entry.edited}

    def stored_label(self, key: Path) -> bytes | None:
        """Return the put-away array for *key*, encoded, or None.

        Encoded rather than decoded: anything asking is going to work on it
        somewhere else, and this is how the session holds it -- decoding here
        would build a full array only to hand it over.

        The label currently open is absent: it is the one on screen, whose
        answers are asked of it directly.
        """
        return self._stored.get(key)

    def reviewed_keys(self) -> set[Path]:
        """Return the labels marked proofread, changed or not."""
        return {key for key, entry in self._pending.items() if entry.reviewed}

    def __len__(self) -> int:
        """Return how many labels have anything pending."""
        return sum(1 for entry in self._pending.values() if entry.pending)

    def __bool__(self) -> bool:
        """Return whether anything at all is waiting to be saved."""
        return len(self) > 0

    def labels(self) -> Iterator[tuple[PendingLabel, np.ndarray]]:
        """Yield each pending label with its array, decoding as needed.

        What a save walks. The open label is yielded from memory rather than
        from its stored copy, so an edit made a moment ago is included.
        """
        for key, entry in sorted(self._pending.items()):
            if not entry.pending:
                continue
            if key == self._open_key and self._open_label is not None:
                yield entry, self._open_label
                continue
            stored = self._stored.get(key)
            if stored is None:
                logger.warning("No stored array for pending label %s", key)
                continue
            yield entry, decode_label(stored)

    def discard(self) -> None:
        """Forget every pending edit, saved or not."""
        self._pending.clear()
        self._stored.clear()
        self._open_key = None
        self._open_label = None
