"""The versions of the label on screen, and comparing against one.

Answers what a label held before, what wrote each version, and whether a model's
guess beats what is there, with no Qt. **A version is anything that is a version of
this segmentation**: a kept label can be put back, a prediction has to be converted.
**A row is legible without opening anything** -- who wrote it, when, by what, and
for a kept version how much the next one changed it. Everything runs when the frame
changes, so nothing may be linear in the collection: the archive is read one
label's directory at a time and the log is cached against its fingerprint.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..history import (
    HistoryLog,
    Recorder,
    containing_project,
    events_for,
    history_file_path,
    read_log,
)
from ..io.atomic import fingerprint_of
from ..io.image import read_array, to_2d
from ..layout import is_convertible_prediction, prediction_precedence
from ..operations.archive import ArchivedVersion, label_versions, restore, written_by
from ..operations.predict import read_provenance
from ..operations.predictions import as_converted
from .attribution import Attribution

logger = logging.getLogger(__name__)

__all__ = ["KEPT", "PREDICTION", "SAVED", "VersionController", "VersionRow", "comparison_badge"]

#: A version kept when something replaced the label. Can be put back.
KEPT: str = "kept"

#: A prediction beside the image. Compared against, never restored -- putting
#: one in place is a conversion, which is a different act with its own
#: confirmation.
PREDICTION: str = "prediction"

#: The label as it is on disk, listed while the window holds unsaved edits to
#: it. Compared against, never restored -- undoing the edits is how to go back.
SAVED: str = "saved"

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
    """One version of the segmentation on screen, as a reader sees it.

    Attributes:
        path: Where the pixels are.
        source: :data:`KEPT` or :data:`PREDICTION`.
        when: When it was written or predicted, or empty where nothing says.
        who: Who wrote it, empty where nobody stated a name, or the model that
            predicted it.
        produced_by: What operation wrote it.
        changed: The share of pixels the next version changed, or None where
            that was never computed -- across a resize, for a version nothing
            recorded, and for every prediction, which is not measured against
            the label until it is looked at.
        available: Whether the pixels are here to be looked at.
        version: The archived version, for a kept row. None for a prediction.
    """

    path: Path
    source: str
    when: str
    who: str
    produced_by: str
    changed: float | None
    available: bool
    version: ArchivedVersion | None = None

    @property
    def restorable(self) -> bool:
        """Return whether this version can be put back where the label is."""
        return self.source == KEPT and self.version is not None and self.available

    @property
    def not_restorable_because(self) -> str:
        """Return why this version is not put back, or empty where it can be."""
        if self.source == PREDICTION:
            return (
                "Predictions cannot be restored, only converted. Use Convert predictions on the "
                "Curate tab."
            )
        if self.source == SAVED:
            return "This is the saved label. Undo your edits to go back to it."
        return "" if self.restorable else "The data for this version is missing."

    def describe(self) -> str:
        """Return the one line a list shows for this version.

        **Every row carries its date in the same place**, whatever produced it,
        so the list can be read down its second column. A row that says what it
        is before saying when defeats the ordering it is listed in.
        """
        kind = {PREDICTION: "predict", SAVED: SAVED}.get(self.source, "kept")
        when = "on disk now" if self.source == SAVED else _shown(self.when) or "(not recorded)"
        # The first two columns are fixed width so the date can be read down
        # the list; a row with nobody's name simply has one fewer column, most
        # rows having no name at all.
        columns = [f"{kind:<{KIND_WIDTH}}", f"{when:<{WHEN_WIDTH}}"]
        if self.who:
            columns.append(self.who)
        columns.append(_short(self.produced_by))
        line = "  ".join(columns)
        if self.source == KEPT:
            line += f"  {_share(self.changed)}"
        if self.available:
            return line
        return f"{line}  ({'will not open' if self.source == PREDICTION else 'pixels not here'})"


class VersionController:
    """The versions of one label, and the one being compared against.

    Attributes:
        attribution: Who a restore is attributed to, shared with every other
            controller so all of them credit the same person.
    """

    def __init__(self, attribution: Attribution | None = None) -> None:
        """Create a controller attributing restores through *attribution*."""
        #: Who changes are recorded as, shared with every other view.
        self.attribution = attribution if attribution is not None else Attribution()
        self._key: Path | None = None
        self._project: Path | None = None
        self._target: str = ""
        self._rows: tuple[VersionRow, ...] = ()
        self._selected: int | None = None
        self._pixels: dict[tuple[Path, tuple[int, int] | None], object] = {}
        self._log: tuple[object, HistoryLog] | None = None
        self._provenance: dict[Path, dict] = {}
        self._saved = None

    @property
    def recorder(self) -> Recorder:
        """Where this view's changes are recorded.

        Shared with every other view, so a name adopted in Settings reaches
        this one without anybody remembering to pass it on.
        """
        return self.attribution.recorder


    # -- what is being looked at -------------------------------------------

    def set_label(
        self, key: Path | None, predictions: Sequence[Path] = (), saved=None
    ) -> None:
        """Point the controller at the label on screen and its predictions.

        Reads the label's own archive directory and the project's log; the
        predictions are paths the scan already found, so nothing is searched
        for. Selecting nothing, because moving to another frame should not
        leave a version from the last one on display.

        Args:
            key: Where the label is, or would be written.
            predictions: The prediction files for the frame on screen.
            saved: The label as it is on disk, where the window holds unsaved
                edits to it; listed first, as what those edits started from.
        """
        self._selected = None
        self._pixels.clear()
        self._saved = saved
        if key is None:
            self._key, self._project, self._target, self._rows = None, None, "", ()
            return

        self._key = key
        self._project = containing_project(key)
        self._target = ""
        if self._project is not None:
            try:
                self._target = key.resolve().relative_to(self._project).as_posix()
            except ValueError:
                self._target = ""

        saved_rows = [_saved_row(key)] if saved is not None else []
        self._rows = tuple(
            [*saved_rows, *self._kept_rows(), *self._prediction_rows(predictions)]
        )

    def refresh(self, predictions: Sequence[Path] = ()) -> None:
        """Read the versions again, after something changed the label."""
        key, self._key = self._key, None
        self.set_label(key, predictions)

    def _kept_rows(self) -> list[VersionRow]:
        """Return a row per kept version, oldest first.

        Empty where the project keeps no log, which is one nothing has changed
        yet.
        """
        if self._project is None or not self._target:
            return []
        log = self._project_log()
        return [
            _kept_row(version, log)
            for version in label_versions(self._project, self._target, log)
        ]

    def _prediction_rows(self, predictions: Sequence[Path]) -> list[VersionRow]:
        """Return a row per prediction that could stand in for the label.

        Ordered am-seg's first, the order :attr:`rows` keeps between two
        predicted at the same moment or with no time recorded. The post-processed kind
        is left out: its values are no longer class indices, so drawing it over
        the image would show something that is not a segmentation.
        """

        usable = [path for path in predictions if is_convertible_prediction(path)]
        usable.sort(key=lambda path: (prediction_precedence(path), path.name))
        return [self._prediction_row(path) for path in usable]

    def _prediction_row(self, path: Path) -> VersionRow:
        """Return how one prediction reads, from its directory's provenance."""
        entry = self._provenance_for(path.parent).get(path.name)
        entry = entry if isinstance(entry, dict) else {}
        model = entry.get("model")
        run = entry.get("run")
        when = entry.get("predicted_at")
        return VersionRow(
            path=path,
            source=PREDICTION,
            # Kept whole, offset included: a prediction's provenance writes
            # local time and the log writes UTC, and truncating the offset away
            # would place it against a kept version by the wrong clock.
            when=when if isinstance(when, str) else "",
            who=str(model) if isinstance(model, str) and model else "unrecorded model",
            produced_by=f"run {run}" if isinstance(run, str) and run else path.parent.name,
            changed=None,
            available=path.is_file(),
        )

    def _provenance_for(self, directory: Path) -> dict:
        """Return what a predictions directory records, read once per frame set."""
        if directory not in self._provenance:

            self._provenance[directory] = read_provenance(directory)
        return self._provenance[directory]

    def _project_log(self) -> HistoryLog:
        """Return the project's log, re-read only when the file has changed.

        The log is appended to and never rewritten, so its fingerprint settles
        whether a held copy is still current. Without this every
        arrow key re-reads every entry the project has.
        """

        assert self._project is not None
        stamp = fingerprint_of(history_file_path(self._project))
        if self._log is None or self._log[0] != stamp:
            self._log = (stamp, read_log(self._project))
        return self._log[1]

    @property
    def rows(self) -> tuple[VersionRow, ...]:
        """Return every version, newest first, whatever produced it.

        One order rather than two groups: a prediction made this morning
        belongs above a version kept last week, and what each row *is* the
        first word of the row already says. A row whose time nothing recorded
        sorts last, having nothing to place it by; the saved label sorts first,
        being newer than anything else listed.
        """
        return tuple(
            sorted(
                self._rows,
                key=lambda row: (
                    row.source == SAVED,
                    _moment(row.when) is not None,
                    _moment(row.when),
                ),
                reverse=True,
            )
        )

    @property
    def default_index(self) -> int:
        """Return the row of :attr:`rows` a comparison starts on, or -1 for none.

        **The newest version of the label that can be shown, before any
        prediction**, however recent the prediction: what the label held is
        what a reviewer looks back at, and a newer model's guess is one
        choice away. The saved label first, where there are unsaved edits; a
        prediction only where no version of the label can be shown.
        """
        rows = self.rows
        for source in (SAVED, KEPT, PREDICTION):
            for index, row in enumerate(rows):
                if row.source == source and row.available:
                    return index
        return 0 if rows else -1

    @property
    def saved(self):
        """Return the label as saved, where it is listed, or None."""
        return self._saved

    @property
    def has_versions(self) -> bool:
        """Return whether there is anything to compare the label against."""
        return bool(self._rows)

    @property
    def count(self) -> int:
        """Return how many versions there are, for a control that is closed."""
        return len(self._rows)

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

    def comparison_pixels(self, shape: tuple[int, int] | None = None):
        """Return the selected version's array, or None.

        **A prediction is returned as the label converting it would write**,
        resized to *shape*, the label on screen's; a kept version is returned
        as it is, its size being part of what is compared. Read once and kept,
        because flicking between this and the label on screen is the point and
        re-reading a file each time would make it stutter.
        """
        row = self.selected
        if row is None or not row.available:
            return None
        if row.source == SAVED:
            return self._saved
        key = (row.path, shape if row.source == PREDICTION else None)
        if key not in self._pixels:
            pixels = _read(row.path)
            if pixels is not None and key[1] is not None:
                pixels = _as_converted(pixels, key[1], row.path)
            self._pixels[key] = pixels
        return self._pixels[key]

    def difference(self, current) -> object | None:
        """Return where the label on screen and the selected version disagree.

        Returns:
            A boolean array, True where they differ, or None where there is
            nothing to compare or the two are not the same size -- a kept
            version of another size has no pixel to pair with.
        """
        if current is None:
            return None
        mine = to_2d(current)
        other = self.comparison_pixels(mine.shape)
        if other is None:
            return None

        theirs = to_2d(other)
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
            ValueError: If no version is selected, its pixels are not here, it
                is a prediction -- which is put in place by converting it --
                or it is the saved label, which undoing the edits goes back to.
            PermissionError: If the group is corrected and *confirmed* is not
                set.
        """
        row = self.selected
        if row is None:
            raise ValueError("No version is selected")
        if row.source == PREDICTION:
            raise ValueError(
                "A prediction is put in place by converting it, which writes a "
                "label from it and records which model made it. Restoring is "
                "for versions this label already held."
            )
        if row.source == SAVED:
            raise ValueError(
                "This is the label as it is saved; undoing the edits goes back "
                "to it, and restoring would write over nothing."
            )
        if not row.available or row.version is None:
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


def comparison_badge(row: VersionRow) -> str:
    """Return what the canvas says while *row* is shown in place of the label.

    A prediction says it is shown as converted: it is drawn at the label's size,
    not the model's, and a reviewer who did not know would misread its edges.
    """
    if row.source == PREDICTION:
        when = _shown(row.when)
        return f"showing the prediction as converted{f', from {when}' if when else ''}"
    if row.source == SAVED:
        return "showing the label as saved, without your edits"
    return f"showing the version from {row.when[:16]}"


def _moment(stamp: str):
    """Return *stamp* as an instant, or None where it is not one.

    The log writes UTC and a prediction's provenance writes local time with its
    offset, so the two are only comparable once parsed. Sorting on the strings
    would put an offset stamp in the wrong place against a ``Z`` one.
    """
    from datetime import UTC, datetime

    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _shown(stamp: str) -> str:
    """Return *stamp* as the minute it names, in UTC as the log records it."""
    moment = _moment(stamp)
    return moment.strftime("%Y-%m-%d %H:%M") if moment else ""


def _share(changed: float | None) -> str:
    """Return how much of the label differed, said so it cannot mislead.

    Said as what the next version did to this one, which is what the number
    is. A correction made in passing is a few hundred pixels of a few million, and
    rounding that to ``0.0%`` reads as *nothing changed* -- which is the one
    thing it does not mean. Below what one decimal can show, the row says so.
    """
    if changed is None:
        return "next change unrecorded"
    if 0 < changed < 0.0005:
        return "then <0.1% changed"
    return f"then {changed * 100:.1f}% changed"


#: The columns a row is laid out in, so the date sits at the same offset on
#: every one of them and the list can be read down it.
KIND_WIDTH: int = 7
WHEN_WIDTH: int = 16

#: How much of a name a row shows before eliding its middle. A training run is
#: named by what it was trained on and which epoch it stopped at, and the
#: middle of that is the part two runs share.
MAX_NAME: int = 34


def _short(name: str) -> str:
    """Return *name* short enough to read, keeping both ends."""
    if len(name) <= MAX_NAME:
        return name
    head = (MAX_NAME - 1) // 2
    return f"{name[:head]}…{name[-(MAX_NAME - head - 1):]}"


def _kept_row(version: ArchivedVersion, log: HistoryLog) -> VersionRow:
    """Return how one kept version reads.

    **Described by the entry that wrote it**, not the one that replaced it:
    a converted label saved over is looked for as "converted", and the save
    is the next version's story. Only the share changed comes from the
    replacement, being how far the next version moved from this one.
    """
    writer = written_by(version, log)
    replaced = version.event
    changed = replaced.data.get("changed_fraction") if replaced else None
    produced_by = "origin unrecorded"
    if writer is not None:
        produced_by = _PRODUCED_BY.get(writer.kind, writer.kind)
        if _marked_corrected(log, version.target, writer):
            produced_by += ", corrected"
    return VersionRow(
        path=version.path,
        source=KEPT,
        when=writer.at if writer else "",
        who=(writer.by or "") if writer else "",
        produced_by=produced_by,
        changed=changed if isinstance(changed, (int, float)) else None,
        available=version.present and version.path.is_file(),
        version=version,
    )


def _marked_corrected(log: HistoryLog, target: str, writer) -> bool:
    """Return whether the run that wrote a version also marked it corrected."""
    if writer.batch is None:
        return False
    return any(
        event.kind == "state-changed"
        and event.batch == writer.batch
        and event.data.get("new") == "corrected"
        for event in events_for(log, target)
    )


def _saved_row(key: Path) -> VersionRow:
    """Return the row for the label as it is on disk."""
    return VersionRow(
        path=key, source=SAVED, when="", who="",
        produced_by="before your unsaved edits", changed=None, available=True,
    )


def _as_converted(pixels, shape: tuple[int, int], path: Path):
    """Return a prediction as converting it would write it, or as read.

    As read where its values are not class indices: converting refuses it, so
    there is no converted label to show, and what is in the file is the truth.
    """
    try:
        return as_converted(pixels, shape, path)
    except ValueError as error:
        logger.warning("Showing %s as read: %s", path, error)
        return pixels


def _read(path: Path):
    """Return what a label or prediction holds, or None where it will not open."""
    try:
        return read_array(path)
    except Exception as error:  # noqa: BLE001 - a damaged file raises whatever its decoder does
        logger.warning("Could not read %s: %s", path, error)
        return None


def rows_for(project_root: Path, target: str) -> Sequence[VersionRow]:
    """Return the kept-version rows for one label without holding a controller.

    For a report that lists versions and does not compare against one.
    """
    log = read_log(project_root)
    return [
        _kept_row(version, log)
        for version in label_versions(project_root, target, log)
    ]
