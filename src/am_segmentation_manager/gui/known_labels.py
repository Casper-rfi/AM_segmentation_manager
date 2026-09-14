"""What each label holds, opened once and shared by every view that asks.

A label's state is :mod:`..operations.label_io`'s answer -- annotated, empty,
missing or unreadable -- and answering it opens the file, so it is remembered.
**One instance serves the Curate, Label and Export views**: a label one of them has
read costs the others nothing, and one any of them writes is read afresh by all
three. A scan clears it; a write forgets what it wrote.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..models import ImageRecord
from ..operations.csv import find_matching_label
from ..operations.label_io import classify_label, is_label_file_empty


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

    def state(self, record: ImageRecord) -> str:
        """Return what *record*'s label holds, opening it only if nothing has."""
        remembered = self._states.get(record.image_path)
        if remembered is not None:
            return remembered[1]
        label = find_matching_label(record)
        state = classify_label(label, self.emptiness(label) if label is not None else None)
        self._states[record.image_path] = (label, state)
        return state

    def emptiness(self, label: Path) -> bool | None:
        """Return whether *label* holds only background, or None where it will not read."""
        if label not in self._empty:
            self._empty[label] = is_label_file_empty(label)
        return self._empty[label]

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
