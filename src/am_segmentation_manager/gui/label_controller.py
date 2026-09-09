"""What the Label tab is looking at: one lamella, one modality, one frame.

Holds the proofreading view's navigation, its own full-resolution frame cache
and the edits waiting to be written, with no dependency on a GUI toolkit.
Nothing here draws; this layer decides what is on screen and what a save would
do with it.

Two things separate it from the curation view. Frames are read at full
resolution, because proofreading a boundary means seeing the pixels it runs
through, so the cache holds a handful of frames rather than a lamella. And only
the frames worth proofreading are visited -- a deselected image is one that was
decided against, and stepping through it wastes the reviewer's attention.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..config import SegmentationConfig
from ..display import prepare_display_image
from ..frames import (
    LamellaFrame,
    build_lamella_frames,
    resolve_series,
    series_present,
)
from ..history import Recorder
from ..io.atomic import fingerprint_of
from ..layout import MODALITIES, SERIES
from ..models import Lamella, Project, projects_holding
from ..operations import label_io
from ..operations.csv import find_matching_label
from ..operations.init_history import writing_recorder
from ..operations.label_edit import EditPatch, new_label
from ..operations.label_measure import DEFAULT_SPECK_PIXELS, Speck, find_specks
from ..operations.label_save import (
    LabelToSave,
    SavePlan,
    plan_saves,
    save_labels,
)
from ..operations.selection import RecordState, current_state, transition_records
from .attribution import Attribution
from .cache import FrameCache, FrameImage, LoadRequest, prefetch_order
from .controller import DEFAULT_SERIES, LABEL_DISPLAY
from .label_session import LabelSession

logger = logging.getLogger(__name__)

#: Passed as ``max_edge`` to read an acquisition without reducing it.
FULL_RESOLUTION = 0

#: Prepared full-resolution frames to keep. Each costs the image plus its label
#: at native size -- around 12 MB for a 2048x3072 acquisition -- so this is a
#: working neighbourhood rather than the whole lamella the curation view holds.
DEFAULT_IMAGE_LIMIT = 6

#: How far either side of the current frame to prepare ahead. Smaller than the
#: curation view's for the same reason the cache is: reading every pixel of a
#: frame is several times the work of reading a reduced one.
DEFAULT_RADIUS = 1

#: The states worth opening in the proofreading view. ``corrected`` is included
#: so finished work can be revisited; changing one still needs the confirmation
#: every corrected label is guarded by.
PROOFREADABLE: tuple[RecordState, ...] = ("selected", "corrected")


def load_full_resolution(image_path, label_path) -> object:
    """Read an acquisition and its label without reducing either."""
    return prepare_display_image(image_path, label_path, max_edge=FULL_RESOLUTION)


class LabelController:
    """The proofreading view's model: a lamella, a modality and a position.

    The lamella is handed in rather than chosen here, so that the collection has
    one navigation cursor rather than two that can disagree about where the user
    is. Positions are indices into the frames worth proofreading, not into the
    lamella, so stepping never lands on an image that was deselected.
    """

    def __init__(
        self,
        config: SegmentationConfig,
        *,
        cache: FrameCache | None = None,
        attribution: Attribution | None = None,
    ) -> None:
        """Create a controller drawing classes from *config*.

        Args:
            config: Supplies the class colours and names the overlay is drawn
                with. Shared with the curation view, so both answer "which
                class is this" the same way.
            cache: Where prepared frames are kept. Defaults to a small
                full-resolution cache; tests pass one with a stub loader.
        """
        #: Who changes are recorded as, shared with every other view.
        self.attribution = attribution if attribution is not None else Attribution()
        #: The scanned projects, so a save can bring the one it writes into
        #: into the record first. Empty until the window shares a scan, which
        #: is also every state in which there is nothing to save.
        self.projects: Sequence[Project] = ()
        self.config = config
        self.cache = cache if cache is not None else FrameCache(
            load_image=load_full_resolution, image_limit=DEFAULT_IMAGE_LIMIT
        )
        self.show_overlay = True
        #: Whether the acquisition is drawn under the label.
        self.show_image = True
        #: Whether components of a few pixels are being pointed out.
        self.finding_specks = False
        self._speck_size = DEFAULT_SPECK_PIXELS
        self._specks: dict[Path, tuple[Speck, ...]] = {}
        #: How many specks each label holds, keyed as :attr:`_specks` is. The
        #: timeline's marks come from here rather than from :attr:`_specks`,
        #: which holds *where* they are and is only ever filled for the frame
        #: on screen: counting a series means reading its label files, and
        #: locating every speck in all of them would be work nobody asked for.
        self._speck_counts: dict[Path, int] = {}
        #: Labels whose count is present but known to be wrong, the label
        #: having changed since it was taken. **Distinct from having no count
        #: at all**, which the timeline draws as blank: a frame the reviewer
        #: has just drawn on should go on showing what it last held rather than
        #: flickering empty until the recount lands, and a count that is merely
        #: absent is one the sweep would pick up anyway.
        self._stale_specks: set[Path] = set()
        self._lamella: Lamella | None = None
        self._frames: tuple[LamellaFrame, ...] = ()
        self._positions: tuple[int, ...] = ()
        self._cursor = 0
        self._modality = MODALITIES[0]
        #: The beam asked for, which a series holding nothing to proofread in
        #: it does not overwrite; see :meth:`_settled_modality`.
        self._wanted_modality = MODALITIES[0]
        #: Which series is being proofread, and which was asked for. One at a
        #: time for the reason the curation timeline is: annotating a lamella
        #: means annotating one series of it, and a list mixing an alignment
        #: run with polishing frames is not one anybody works through. Its own
        #: choice rather than the Curate tab's, as the modality already is.
        self._series = DEFAULT_SERIES
        self._wanted_series = DEFAULT_SERIES
        #: The imaging condition on screen, per modality. Purely a view choice:
        #: the conditions of one acquisition share a label, so which one is
        #: being looked at never decides what an edit or a state change touches.
        self._variants: dict[str, str | None] = {}
        self._label_states: dict[object, str] = {}
        self.session = LabelSession()
        self._keys: dict[Path, Path] = {}

    @property
    def recorder(self) -> Recorder:
        """Where this view's changes are recorded.

        Shared with every other view, so a name adopted in Settings reaches
        this one without anybody remembering to pass it on.
        """
        return self.attribution.recorder


    # -- what is being reviewed --------------------------------------------

    def set_projects(self, projects: Sequence[Project]) -> None:
        """Note the scanned projects, so a save knows whose history to start."""
        self.projects = tuple(projects)

    def set_lamella(self, lamella: Lamella | None) -> None:
        """Point the view at *lamella*, keeping the position if it is the same one.

        Re-pointing at the same lamella happens every time the tab comes forward,
        and resetting to its first frame then would throw away the place the
        reviewer had got to. The frames are rebuilt regardless, because the
        records may have changed state since -- and what is rebuilt is a pure
        transformation of the scan, costing nothing.

        Prepared frames are discarded either way, so returning to the tab
        re-reads the frame on screen. Keeping them would mean holding pixels
        against frame indices that a rescan may have renumbered.
        """
        same = (
            lamella is not None
            and self._lamella is not None
            and lamella.path == self._lamella.path
        )
        keep_frame = self.frame_index if same else None

        self._lamella = lamella
        if not same:
            self._label_states.clear()
            self._keys.clear()
            self._specks.clear()
            self._speck_counts.clear()
            self._stale_specks.clear()
            self.session.discard()
        self._rebuild_frames(keep_frame=keep_frame)
        logger.debug(
            "Label view now on %s: %d %s frames, %d proofreadable %s",
            lamella.name if lamella else "nothing",
            len(self._frames), self._series, len(self._positions), self._modality,
        )

    def _rebuild_frames(self, *, keep_frame: int | None) -> None:
        """Rebuild the frames of the series being proofread.

        **What is shown is settled here**, as in the curation view: the series
        falls back to one this lamella holds and the beam to one this series
        has work in, without either overwriting what was asked for.
        """
        self._series = resolve_series(self._wanted_series, self.available_series())
        self._frames = (
            tuple(build_lamella_frames(self._lamella, series=self._series))
            if self._lamella else ()
        )
        self.cache.set_frames(self._frames)
        self._refresh_positions(keep_frame=keep_frame)

    def available_series(self) -> list[str]:
        """Return the series this lamella holds images of, in display order.

        What it *holds*, not what it has work to do in: a series whose images
        are all deselected is still worth being able to look at, and the tab
        says so rather than leaving the choice out.
        """
        return series_present(self._lamella)

    @property
    def series(self) -> str:
        """Return the series being proofread."""
        return self._series

    def set_series(self, series: str) -> None:
        """Proofread *series* instead, rebuilding the frames to review.

        The position goes back to the start: a frame index means nothing
        outside the list it was taken in. Edits waiting to be saved are kept --
        they belong to the lamella, which has not changed.

        Raises:
            ValueError: If *series* is not one this package indexes.
        """
        if series not in SERIES:
            raise ValueError(f"Unknown series: {series!r}; expected one of {list(SERIES)}")
        self._wanted_series = series
        if series == self._series:
            return
        self._rebuild_frames(keep_frame=None)

    @property
    def lamella(self) -> Lamella | None:
        """Return the lamella being proofread."""
        return self._lamella

    @property
    def modality(self) -> str:
        """Return the modality on screen, ``"SEM"`` or ``"FIB"``."""
        return self._modality

    def set_modality(self, modality: str) -> None:
        """Show *modality*, staying on the current acquisition where it exists.

        Switching SEM to FIB is a request to look at the same moment of the
        milling from the other beam, so the position follows the frame rather
        than the index.
        """
        wanted = modality.upper()
        if wanted not in MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")
        self._wanted_modality = wanted
        if wanted == self._modality:
            return
        self._refresh_positions(keep_frame=self.frame_index)

    def available_modalities(self) -> list[str]:
        """Return the modalities with anything to proofread in this series."""
        return [
            modality for modality in MODALITIES
            if any(self._is_proofreadable(frame, modality) for frame in self._frames)
        ]

    def _settled_modality(self) -> str:
        """Return the beam to proofread: the one asked for where it has work.

        An alignment run is imaged with the ion beam alone, so opening one with
        SEM chosen would offer nothing to review and say only that. The choice
        is untouched, so a series with SEM work in it opens on SEM again.
        """
        available = self.available_modalities()
        if not available or self._wanted_modality in available:
            return self._wanted_modality
        return available[0]

    def _is_proofreadable(self, frame: LamellaFrame, modality: str) -> bool:
        """Return whether *frame* has an image of *modality* worth opening."""
        record = frame.record(modality)
        return record is not None and current_state(record) in PROOFREADABLE

    def _refresh_positions(self, *, keep_frame: int | None) -> None:
        """Recompute the visitable frames, staying as near *keep_frame* as possible."""
        self._modality = self._settled_modality()
        self._positions = tuple(
            index for index, frame in enumerate(self._frames)
            if self._is_proofreadable(frame, self._modality)
        )
        if not self._positions:
            self._cursor = 0
            return
        if keep_frame is None:
            self._cursor = 0
            return
        # The nearest visitable frame, so a modality switch on an acquisition
        # that only one beam imaged lands beside it rather than at the start.
        self._cursor = min(
            range(len(self._positions)),
            key=lambda index: abs(self._positions[index] - keep_frame),
        )

    # -- position -----------------------------------------------------------

    @property
    def count(self) -> int:
        """Return how many frames there are to proofread."""
        return len(self._positions)

    @property
    def position(self) -> int:
        """Return the cursor's index among the proofreadable frames."""
        return self._cursor

    @property
    def frame_index(self) -> int | None:
        """Return the current frame's index within the lamella, or None."""
        if not self._positions:
            return None
        return self._positions[self._cursor]

    @property
    def frame(self) -> LamellaFrame | None:
        """Return the frame on screen, or None when there is nothing to show."""
        index = self.frame_index
        return self._frames[index] if index is not None else None

    @property
    def record(self):
        """Return the image record on screen, or None.

        The frame's base image where this frame was not captured under the
        chosen condition, on the same reasoning as the curation view: stepping
        through a lamella where only some acquisitions have a condition should
        keep showing something rather than blanking the pane.
        """
        frame = self.frame
        if frame is None:
            return None
        record = frame.record(self._modality, self.variant)
        return record if record is not None else frame.record(self._modality)

    @property
    def variant(self) -> str | None:
        """Return the imaging condition on screen, or None for the base image."""
        return self._variants.get(self._modality)

    def set_variant(self, variant: str | None) -> None:
        """Show *variant* of the modality on screen.

        Which frames are worth proofreading is unchanged: the conditions of one
        acquisition share a label and therefore share its state, so the set is
        read from the base image whichever condition is being looked at.
        """
        self._variants[self._modality] = variant or None
        logger.debug("Proofreading %s variant %s", self._modality, variant or "base")

    def available_variants(self) -> list[str | None]:
        """Return the conditions the frame on screen was captured under."""
        frame = self.frame
        return frame.variants(self._modality) if frame is not None else []

    def select(self, position: int) -> None:
        """Move to *position* among the proofreadable frames, clamped."""
        if not self._positions:
            self._cursor = 0
            return
        self._cursor = max(0, min(position, len(self._positions) - 1))

    def step(self, delta: int) -> None:
        """Move *delta* frames, stopping at either end."""
        self.select(self._cursor + delta)

    # -- what to draw -------------------------------------------------------

    def current_image(self) -> FrameImage | None:
        """Return the prepared frame, reading it if it is not already held."""
        index = self.frame_index
        if index is None:
            return None
        return self.cache.image(index, self._modality, self.variant)

    def cached_image(self) -> FrameImage | None:
        """Return the prepared frame only if it is already held.

        Lets the panel say *loading* instead of blocking the interface on a
        full-resolution read.
        """
        index = self.frame_index
        if index is None:
            return None
        return self.cache.cached_image(index, self._modality, self.variant)

    def pending_loads(self, *, radius: int | None = DEFAULT_RADIUS) -> list[LoadRequest]:
        """Return the frames to prepare, current first then outwards.

        Ordered over the proofreadable frames rather than the lamella, so the
        read-ahead follows where stepping will actually go.
        """
        if not self._positions:
            return []
        requests = []
        for cursor in prefetch_order(self._cursor, len(self._positions), radius):
            index = self._positions[cursor]
            if self.cache.holds(index, self._modality, self.variant):
                continue
            requests.append(LoadRequest(index, self._modality, self.variant))
        return requests

    # -- editing ------------------------------------------------------------

    def key_for(self, record) -> Path | None:
        """Return where *record*'s label is, or would be written.

        Cached per record: resolving a destination for an image with no label
        yet reads the directory it would go in, which is not something to do on
        every redraw.
        """
        if record is None:
            return None
        cached = self._keys.get(record.image_path)
        if cached is None:
            cached = label_io.label_destination(record)
            self._keys[record.image_path] = cached
        return cached

    @property
    def current_key(self) -> Path | None:
        """Return the label the frame on screen is proofread through."""
        return self.key_for(self.record)

    def _is_corrected(self, key: Path) -> bool:
        """Return whether the group owning the label at *key* is corrected.

        Read from the records rather than from the path, which no longer says.
        Several conditions share one label, so any of them answering is enough.
        """
        return any(
            current_state(record) == "corrected"
            for frame in self._frames
            for record in frame.records
            if self.key_for(record) == key
        )

    @property
    def editable(self) -> bool:
        """Return whether the frame on screen may be painted on.

        A label whose dimensions disagree with its image is refused: it was
        resampled to be displayed, and saving that back would rewrite the
        annotation at the wrong scale.
        """
        prepared = self.cached_image()
        if prepared is None or prepared.failed or prepared.display is None:
            return False
        return not prepared.display.label_resized

    def working_label(self):
        """Return the array to draw for the frame on screen.

        While the frame is editable this is the session's own copy, which the
        canvas paints into; otherwise it is the prepared frame's label, which
        must not be written to. Where no label file exists, an all-background
        one is opened, so an unlabelled image can be annotated from nothing.
        """
        prepared = self.cached_image()
        if prepared is None or prepared.failed or prepared.display is None:
            return None
        display = prepared.display
        if not self.editable:
            return display.label

        key = self.current_key
        if key is None:
            return display.label
        source = (
            display.label if display.label is not None
            else new_label(display.image.shape)
        )
        return self.session.open(
            key,
            source,
            created=display.label is None,
            fingerprint=fingerprint_of(key),
        )

    def record_edit(self, patch: EditPatch | None) -> None:
        """Record a completed stroke against the label on screen."""
        self.session.record(patch)
        self.note_label_changed()

    def note_label_changed(self) -> None:
        """Say that the label on screen no longer matches what was counted.

        Called for anything that changes it -- a stroke, an undo, a redo. The
        ringed specks are dropped, recounting them belonging to whoever decides
        that the reviewer has stopped drawing; the *count* is kept and marked
        stale instead, so the frame goes on showing what it last held until a
        better answer arrives, and so the sweep knows to ask again. Without the
        mark it would not: a frame that already has a count is one the sweep
        passes over, which is how a recount lost to moving on stayed lost.
        """
        key = self.current_key
        if key is None:
            return
        self._specks.pop(key, None)
        self._stale_specks.add(key)

    @property
    def speck_size(self) -> int:
        """Return the largest component still pointed out, in pixels."""
        return self._speck_size

    def set_speck_size(self, pixels: int) -> None:
        """Point out components of at most *pixels*, forgetting the old count."""
        pixels = max(1, int(pixels))
        if pixels == self._speck_size:
            return
        self._speck_size = pixels
        self._specks.clear()
        self._speck_counts.clear()
        self._stale_specks.clear()

    def specks(self) -> tuple[Speck, ...]:
        """Return the components of a few pixels in the label on screen.

        Measured on the array being edited, unsaved strokes included, so a
        speck painted out stops being reported and one just painted starts.
        Held per label until a stroke changes it, so stepping back and forth
        across a lamella measures each frame once.
        """
        key = self.current_key
        if key is None:
            return ()
        held = self._specks.get(key)
        if held is not None:
            return held
        label = self.working_label()
        found = () if label is None else find_specks(label, at_most=self._speck_size)
        self._specks[key] = found
        # The count goes in beside it, so the frame being edited keeps its mark
        # on the timeline after the reviewer moves on: what is on screen has
        # unsaved strokes in it, and the file on disk no longer says the same.
        # This is the freshest answer there is, so it settles any staleness.
        self._speck_counts[key] = len(found)
        self._stale_specks.discard(key)
        return found

    def mark_reviewed(self, reviewed: bool = True) -> None:
        """Record that the label on screen has been proofread, edited or not."""
        key = self.current_key
        if key is not None:
            self.session.mark_reviewed(key, reviewed=reviewed)

    def is_reviewed(self) -> bool:
        """Return whether the label on screen is marked proofread."""
        key = self.current_key
        entry = self.session.state_of(key) if key is not None else None
        return bool(entry and entry.reviewed)

    def edited_positions(self) -> set[int]:
        """Return the timeline positions whose label has unsaved changes."""
        edited = self.session.edited_keys()
        if not edited:
            return set()
        return {
            position
            for position, index in enumerate(self._positions)
            if self.key_for(self._frames[index].record(self._modality)) in edited
        }

    def pending_count(self) -> int:
        """Return how many labels are waiting to be saved."""
        return len(self.session)

    # -- saving ---------------------------------------------------------------

    def records_for(self, keys) -> list:
        """Return every record in the lamella proofread through one of *keys*.

        Every record of every frame is considered: pending edits are not
        confined to the modality on screen, and imaging conditions of one scene
        share a label and so share its state.
        """
        wanted = set(keys)
        found = []
        for frame in self._frames:
            for record in frame.records:
                if self.key_for(record) in wanted:
                    found.append(record)
        return found

    def pending_saves(self) -> list[LabelToSave]:
        """Return what a save would write, newest edits included."""
        return [
            LabelToSave(
                destination=entry.key,
                label=label,
                created=entry.created,
                fingerprint=entry.fingerprint,
                corrected=self._is_corrected(entry.key),
            )
            for entry, label in self.session.labels()
            if entry.edited
        ]

    def pending_modalities(self) -> dict[str, int]:
        """Return how many labels are waiting, per modality.

        What a save is about to write is not confined to the modality on
        screen: proofreading holds a whole lamella's edits, so someone working
        through the SEM images can have FIB edits waiting from earlier without
        the dialog saying so. A label whose images span both modalities counts
        under each, which overstates the total rather than hiding half of it.
        """
        counts: dict[str, int] = {}
        for entry, _ in self.session.labels():
            if not entry.edited:
                continue
            for modality in {
                record.modality for record in self.records_for([entry.key])
                if record.modality
            }:
                counts[modality] = counts.get(modality, 0) + 1
        return counts

    def save_plan(self) -> SavePlan:
        """Return what a save would write, and what is holding it up."""
        return plan_saves(self.pending_saves())

    def approved_records(self) -> list:
        """Return the records whose label the reviewer has passed.

        What "mark corrected" would cover. Edited and approved-unchanged alike:
        both are a judgement that the label is right.
        """
        return self.records_for(self.session.reviewed_keys())

    def save(self, *, mark_corrected: bool = False, allow_corrected: bool = False) -> str:
        """Write every edited label, and optionally mark the reviewed set corrected.

        Leaves the view in step with what was written, so nothing has to scan
        the collection again to catch up: the state change declares itself in a
        file without moving anything, and brings the records it touched with it.

        Args:
            mark_corrected: Also declare the reviewed labels, and the images
                sharing them, corrected.
            allow_corrected: Confirm replacing labels that are already
                proofread.

        Returns:
            A line describing what was done, for the window's log.

        Raises:
            StaleLabels: If a label's file changed after it was opened.
            PermissionError: If a proofread label would be replaced without
                *allow_corrected*.
        """
        pending = self.pending_saves()
        batch = writing_recorder(
            projects_holding(
                (item.destination for item in pending), self.projects
            ),
            self.recorder,
        )
        report = save_labels(
            pending, allow_corrected=allow_corrected, recorder=batch
        )
        self._attach_created_labels(report.created)
        headline = report.headline()

        if mark_corrected:
            records = self.approved_records()
            if records:
                declared = transition_records(records, "corrected", recorder=batch)
                headline += f", marked {declared} image(s) corrected"

        # Before the session is discarded, and before anything redraws: a
        # prepared frame holds the pixels its label had when it was read, so
        # one just written would go on drawing what it replaced. The session's
        # own copy is what hid this until now.
        self.cache.relabel({item.destination: item.label for item in pending})
        self.session.discard()
        self._label_states.clear()
        # The frames worth visiting are recomputed here rather than by the
        # rescan that used to follow a save: marking corrected moves the state
        # this reads, and a label written where there was none changes what a
        # frame's line reports. Both are known here, so nothing has to re-read
        # the collection to find them out.
        self._refresh_positions(keep_frame=self.frame_index)
        return headline

    def _attach_created_labels(self, created: Iterable[Path]) -> None:
        """Name each newly written label on the records proofread through it.

        A group is keyed by its label, and a record the scan built before that
        label existed carries no label path -- so it would be keyed by where a
        label *would* go. That resolves to the same place, but only while the
        record is asked afresh; naming the label now keeps the key stable for
        the state change that follows the save.
        """
        wanted = set(created)
        if not wanted:
            return
        for record in self.records_for(wanted):
            if not record.labels:
                record.labels = [self.key_for(record)]

    def discard_edits(self) -> None:
        """Throw away every pending edit in this lamella."""
        if self.session:
            logger.info("Discarding %d pending label edit(s)", len(self.session))
        self.session.discard()

    def timeline_states(self) -> dict[str, list[RecordState | None]]:
        """Return the current modality's state for every proofreadable frame.

        Only the one row: the other modality's proofreadable frames are a
        different set, and drawing them along the same axis would put two
        different sequences of acquisitions above one another.
        """
        return {
            self._modality: [
                current_state(self._frames[index].record(self._modality))
                for index in self._positions
            ]
        }

    def timeline_labels(self) -> dict[str, list[str | None]]:
        """Return each proofreadable frame's label state, for the timeline."""
        return {
            self._modality: [
                self._label_state(self._frames[index].record(self._modality))
                for index in self._positions
            ]
        }

    def timeline_specks(self) -> dict[str, list[int | None]]:
        """Return each proofreadable frame's speck count, for the timeline.

        Empty while the filter is off: the marks exist because the reviewer
        asked for them, and a series that quietly kept them would be answering
        a question nobody had put. An entry of None is a frame nothing has
        counted yet, which the timeline leaves blank -- not counted must not
        read as none.
        """
        if not self.finding_specks:
            return {}
        return {
            self._modality: [
                self._speck_counts.get(self._key_at(index))
                for index in self._positions
            ]
        }

    def uncounted_labels(self) -> list[tuple[Path, Path | bytes]]:
        """Return ``(key, source)`` for frames whose specks are not counted.

        What a background sweep is given. A source is normally the label file,
        but **a frame holding unsaved strokes gives the session's own copy of
        it instead**: its file no longer says what the reviewer is looking at,
        and counting it reports specks that have already been painted out. That
        is not a rare case -- a size change invalidates every count at once, so
        the sweep that follows one re-reads every frame including the edited
        ones.

        The frame on screen is left out wherever it is already counted, which
        it is as soon as the canvas has drawn its rings.
        """
        if not self.finding_specks:
            return []
        edited = self.session.edited_keys()
        wanted: list[tuple[Path, Path | bytes]] = []
        seen: set[Path] = set()
        for index in self._positions:
            key = self._key_at(index)
            if key is None or key in seen:
                continue
            if key in self._speck_counts and key not in self._stale_specks:
                continue
            seen.add(key)
            if key in edited:
                stored = self.session.stored_label(key)
                if stored is not None:
                    wanted.append((key, stored))
                # No stored copy means it is the label open on screen, which
                # `specks` answers for directly.
                continue
            record = self._frames[index].record(self._modality)
            label = None if record is None else find_matching_label(record)
            if label is not None:
                wanted.append((key, label))
        return wanted

    def record_speck_count(self, key: Path, count: int) -> None:
        """Take one label's count from the sweep.

        A count already held is kept, the sweep being the slower of the two
        answers and often the older -- except where the label has changed since
        that count was taken, which is the one case the sweep was asked to
        settle.
        """
        if key in self._stale_specks:
            self._speck_counts[key] = count
            self._stale_specks.discard(key)
            return
        self._speck_counts.setdefault(key, count)

    def _key_at(self, index: int) -> Path | None:
        """Return the label key of the frame at *index* in the current view."""
        record = self._frames[index].record(self._modality)
        return None if record is None else self.key_for(record)

    def record_label_state(self, record) -> str:
        """Return what *record*'s label is, reading the file at most once.

        Cached for the controller's life, so stepping back and forth across a
        lamella re-reads nothing.
        """
        path = record.image_path
        if path not in self._label_states:
            self._label_states[path] = label_io.label_state(record)
        return self._label_states[path]

    def forget_label_state(self, record) -> None:
        """Drop what was known about *record*'s label, so it is read again."""
        self._label_states.pop(record.image_path, None)

    def _label_state(self, record) -> str | None:
        """Return one record's label state as the timeline draws it."""
        if record is None:
            return None
        return LABEL_DISPLAY[self.record_label_state(record)]

    def status_text(self) -> str:
        """Return the line describing where the reviewer is."""
        if self._lamella is None:
            return "No lamella selected"
        if not self._positions:
            return (
                f"{self._lamella.display_name}: no {self._modality} images to "
                f"proofread. Select them in the Curate tab."
            )
        frame = self.frame
        parts = [
            self._lamella.display_name,
            f"{self._modality} {self._cursor + 1} / {len(self._positions)}",
        ]
        if frame is not None:
            parts.append(f"stage {frame.stage}")
            if frame.image_number is not None:
                parts.append(f"image {frame.image_number}")
        record = self.record
        if record is not None:
            parts.append(current_state(record))
        return " · ".join(parts)
