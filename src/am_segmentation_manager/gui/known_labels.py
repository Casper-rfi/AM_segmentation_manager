"""What each label holds, opened once and shared by every view that asks.

A label's state is :mod:`..operations.label_io`'s answer -- annotated, empty,
missing or unreadable -- and answering it opens the file, so it is remembered.
**One instance serves the Curate, Label and Export views**: a label one of them has
read costs the others nothing, and one any of them writes is read afresh by all
three. A scan clears it; a write forgets what it wrote. A view asking about many
labels at once prefetches them, so they are read together rather than in turn.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..models import ImageRecord
from ..operations import label_io
from ..operations.label_groups import find_matching_label
from ..operations.label_io import classify_label, read_emptiness

#: How a label's state is drawn on the timeline. A display decision, and the
#: only place the two vocabularies meet: :mod:`.operations.label_io` says what
#: a label *is*, and the timeline has three marks for it. UNREADABLE draws as
#: annotated because the file does exist -- what is unknown is what is inside
#: it, and drawing it as a placeholder would assert the one thing nobody could
#: check.
LABEL_DISPLAY: dict[str, str] = {
    label_io.ANNOTATED: "present",
    label_io.UNREADABLE: "present",
    label_io.EMPTY: "empty",
    label_io.MISSING: "missing",
}


class KnownLabels:
    """What each record's label holds, read at most once until it is forgotten."""

    def __init__(self) -> None:
        #: Whether each label holds anything, **keyed by the label**: one entry
        #: per file however many imaging conditions share it, None where it
        #: could not be read.
        self._empty: dict[Path, bool | None] = {}
        #: Each image's state and the label it was worked out from, **keyed by
        #: the image**, so forgetting a label reaches every image sharing it.
        self._states: dict[Path, tuple[Path | None, str]] = {}

    def drawn(self, record: ImageRecord | None) -> str | None:
        """Return *record*'s label state as the timeline draws it, None for no record."""
        return LABEL_DISPLAY[self.state(record)] if record is not None else None

    def state(self, record: ImageRecord) -> str:
        """Return what *record*'s label holds, opening it only if nothing has."""
        self.prefetch([record])
        return self._states[record.image_path][1]

    def emptiness(self, label: Path) -> bool | None:
        """Return whether *label* holds only background, or None where it will not read."""
        self._read([label])
        return self._empty[label]

    def prefetch(self, records: Iterable[ImageRecord | None]) -> None:
        """Open together every label of *records* whose state is not yet known.

        Asking :meth:`state` of each of them afterwards opens nothing.
        """
        unknown = {
            record.image_path: record
            for record in records
            if record is not None and record.image_path not in self._states
        }
        labels = {image: find_matching_label(record) for image, record in unknown.items()}
        self._read(label for label in labels.values() if label is not None)
        for image, label in labels.items():
            empty = self._empty[label] if label is not None else None
            self._states[image] = (label, classify_label(label, empty))

    def _read(self, labels: Iterable[Path]) -> None:
        """Open those of *labels* not yet opened, together."""
        unread = [label for label in dict.fromkeys(labels) if label not in self._empty]
        if unread:
            self._empty.update(read_emptiness(unread))

    def knows(self, label: Path) -> bool:
        """Return whether *label* has been opened since it was last forgotten."""
        return label in self._empty

    def remember(self, image: Path, label: Path | None, empty: bool | None) -> None:
        """Take what a pass that has already opened *label* found, for *image*."""
        if label is not None:
            self._empty[label] = empty
        self._states[image] = (label, classify_label(label, empty))

    def forget(
        self, *, labels: Iterable[Path] = (), records: Iterable[ImageRecord] = ()
    ) -> int:
        """Drop what was read from *labels* and about *records*, so both are read again.

        A label is forgotten for every image whose state was worked out from it.
        A record is named where it has just gained a label its state predates.

        Returns:
            How many of *labels* had been opened.
        """
        wanted = set(labels)
        forgotten = 0
        for label in wanted:
            if label in self._empty:
                del self._empty[label]
                forgotten += 1
        images = {record.image_path for record in records}
        if wanted or images:
            self._states = {
                image: entry
                for image, entry in self._states.items()
                if image not in images and entry[0] not in wanted
            }
        return forgotten

    def clear(self) -> None:
        """Forget everything, as a scan does: any file may have changed since."""
        self._empty.clear()
        self._states.clear()
