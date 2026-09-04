"""The versions kept of the label on screen, and comparing against one.

What the proofreading view needs to answer three questions about a label: what
it used to hold, how far each version is from the one before it, and whether a
new model's guess is better than what is there now. Holds no Qt, so the listing,
the comparison and the restore are all testable without a display.

**A row is legible without opening anything.** Each carries who, when, what
produced it and the share of pixels that changed, which was computed when the
version was kept and cannot be had cheaply afterwards -- ten rows where nine say
0.1% and one says 34% tell a reader which one matters.

A label with no kept versions has nothing here, which is the ordinary case.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..history import Recorder, containing_project
from ..operations.archive import ArchivedVersion, label_versions, restore

logger = logging.getLogger(__name__)

__all__ = ["VersionController", "VersionRow"]

#: What a row says produced a version, by the entry kind that recorded it.
_PRODUCED_BY: dict[str, str] = {
    "label-saved": "proofreading save",
    "label-converted": "converted",
    "label-restored": "restored",
    "label-compressed": "recompressed",
    "merged": "merged",
    "baseline": "already there",
}


@dataclass(frozen=True)
class VersionRow:
    """One kept version, as a reader sees it.

    Attributes:
        version: What it is and where it is.
        when: When it was kept, or empty where nothing recorded it.
        who: Who kept it, or "unknown".
        produced_by: What operation replaced it.
        changed: The share of pixels that differed from the version before,
            or None where that was never computed -- across a resize, and for
            a version nothing recorded.
        available: Whether the pixels are here to be looked at.
    """

    version: ArchivedVersion
    when: str
    who: str
    produced_by: str
    changed: float | None
    available: bool

    def describe(self) -> str:
        """Return the one line a list shows for this version."""
        share = "unknown" if self.changed is None else f"{self.changed * 100:.1f}%"
        line = f"{self.when or '(not recorded)'}  {self.who}  {self.produced_by}"
        line += f"  {share} changed"
        return line if self.available else f"{line}  (pixels not here)"


class VersionController:
    """The kept versions of one label, and the one being compared against.

    Attributes:
        recorder: Who a restore is attributed to.
    """

    def __init__(self, recorder: Recorder | None = None) -> None:
        """Create a controller attributing restores to *recorder*."""
        self.recorder = recorder if recorder is not None else Recorder.for_run()
        self._key: Path | None = None
        self._project: Path | None = None
        self._target: str = ""
        self._rows: tuple[VersionRow, ...] = ()
        self._selected: int | None = None
        self._pixels: dict[Path, object] = {}

    # -- what is being looked at -------------------------------------------

    def set_label(self, key: Path | None) -> None:
        """Point the controller at the label on screen.

        Reads the archive and the log. Selecting nothing, because moving to
        another frame should not leave a version from the last one on display.
        """
        self._selected = None
        self._pixels.clear()
        if key is None or key == self._key:
            if key is None:
                self._key, self._project, self._target, self._rows = None, None, "", ()
            return
        self._key = key
        self._project = containing_project(key)
        if self._project is None:
            self._target, self._rows = "", ()
            return
        try:
            self._target = key.resolve().relative_to(self._project).as_posix()
        except ValueError:
            self._target, self._rows = "", ()
            return
        self._rows = tuple(
            _row(version) for version in label_versions(self._project, self._target)
        )

    def refresh(self) -> None:
        """Read the versions again, after something changed the label."""
        key, self._key = self._key, None
        self.set_label(key)

    @property
    def rows(self) -> tuple[VersionRow, ...]:
        """Return the kept versions, newest first."""
        return tuple(reversed(self._rows))

    @property
    def has_versions(self) -> bool:
        """Return whether this label has any kept version at all."""
        return bool(self._rows)

    @property
    def target(self) -> str:
        """Return the label's key, relative to its project."""
        return self._target

    # -- comparing ---------------------------------------------------------

    @property
    def selected(self) -> VersionRow | None:
        """Return the version being compared against, or None."""
        rows = self.rows
        if self._selected is None or not 0 <= self._selected < len(rows):
            return None
        return rows[self._selected]

    def select(self, index: int | None) -> None:
        """Compare against the version at *index* of :attr:`rows`, or none."""
        if index is None:
            self._selected = None
            return
        if 0 <= index < len(self.rows):
            self._selected = index

    def comparison_pixels(self):
        """Return the selected version's array, or None.

        Read once and kept, because flicking between this and the label on
        screen is the point and re-reading a file each time would make it
        stutter.
        """
        row = self.selected
        if row is None or not row.available:
            return None
        path = row.version.path
        if path not in self._pixels:
            self._pixels[path] = _read(path)
        return self._pixels[path]

    def difference(self, current) -> object | None:
        """Return where the label on screen and the selected version disagree.

        Returns:
            A boolean array, True where they differ, or None where there is
            nothing to compare or the two are not the same size -- a
            conversion writes at model resolution, and a mask across a resize
            would mark every pixel.
        """
        other = self.comparison_pixels()
        if other is None or current is None:
            return None
        from ..io.image import to_2d

        mine, theirs = to_2d(current), to_2d(other)
        if mine.shape != theirs.shape:
            return None
        return mine != theirs

    # -- putting one back --------------------------------------------------

    def restore_selected(self, *, corrected: bool = False, confirmed: bool = False):
        """Put the selected version back, through the ordinary save.

        Every guard protecting a label protects this: the batch validation, the
        refusal of a file changed since it was read, and the protection of
        proofread work. The version being replaced is archived on the way past,
        so a restore is itself reversible.

        Raises:
            ValueError: If no version is selected, or its pixels are not here.
            PermissionError: If the group is corrected and *confirmed* is not
                set.
        """
        row = self.selected
        if row is None:
            raise ValueError("No version is selected")
        if not row.available:
            raise ValueError(
                "This collection holds the record of that version but not its "
                "pixels. It was kept somewhere else -- a copy that was exported "
                "without --with-archive, whose history came back on a merge."
            )
        if self._key is None:
            raise ValueError("No label is on screen")

        restored = restore(
            row.version, self._key,
            recorder=self.recorder.for_batch(),
            corrected=corrected,
            allow_corrected=confirmed,
        )
        self.refresh()
        return restored


def _row(version: ArchivedVersion) -> VersionRow:
    """Return how one kept version reads."""
    event = version.event
    changed = event.data.get("changed_fraction") if event else None
    return VersionRow(
        version=version,
        when=event.at if event else "",
        who=event.by if event else "unknown",
        produced_by=_PRODUCED_BY.get(event.kind, event.kind) if event else "unrecorded",
        changed=changed if isinstance(changed, (int, float)) else None,
        available=version.present and version.path.is_file(),
    )


def _read(path: Path):
    """Return what an archived label holds, or None where it will not open."""
    try:
        from ..io.image import read_array

        return read_array(path)
    except Exception as error:
        logger.warning("Could not read %s: %s", path, error)
        return None


def rows_for(project_root: Path, target: str) -> Sequence[VersionRow]:
    """Return the rows for one label without holding a controller.

    For a report that lists versions and does not compare against one.
    """
    return [_row(version) for version in label_versions(project_root, target)]
