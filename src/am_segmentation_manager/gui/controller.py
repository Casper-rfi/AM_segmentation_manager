"""The curation view model: what is selected, what to show, what to change.

This holds everything the curation window knows, with no dependency on a GUI
toolkit. Widgets read from it to decide what to draw and call into it when the
user does something; none of the logic below needs a display to run, which is
what makes it testable.

Keeping this layer free of Qt is the main structural rule of the UI. Widgets
render and forward events; decisions live here.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config import SegmentationConfig, load_segmentation_config
from ..frames import (
    LamellaFrame,
    build_lamella_frames,
    resolve_series,
    series_present,
)
from ..layout import LABELS_EXPECTED, MODALITIES, SERIES, SERIES_LABELS
from ..models import Lamella, Project
from ..operations import label_io
from ..operations.label_groups import expand_to_groups
from ..operations.label_io import LabelReadiness, classify_labels
from ..operations.selection import RecordState, current_state, transition_records
from ..scanner import ProjectIndex, Scanner
from .cache import (
    DEFAULT_RADIUS,
    FrameCache,
    FrameImage,
    FramePlot,
    LoadRequest,
)
from .selection import (
    LEVEL_CURRENT_FRAME,
    LEVEL_LAMELLAE,
    LEVEL_PROJECTS,
    Selection,
    SelectionSnapshot,
    counted,
    selection_key,
)

logger = logging.getLogger(__name__)

#: Which view of a frame is on screen.
DisplayMode = Literal["sem", "fib", "plot"]

#: What a state change applies to.
EditTarget = Literal["SEM", "FIB", "both"]

#: The state of a frame whose two acquisitions disagree.
MIXED = "mixed"

MODE_MODALITIES: dict[DisplayMode, str] = {"sem": "SEM", "fib": "FIB"}

#: The same pairing read the other way, for asking which view shows a beam.
MODALITY_MODES: dict[str, DisplayMode] = {
    modality: mode for mode, modality in MODE_MODALITIES.items()
}

#: Which series the Curate tab opens on. Adaptive Polishing, so the tab shows
#: what it always showed; the other two are one click away. Unlike the Export
#: tab this is a single choice rather than a set, because a timeline mixing
#: alignment runs with polishing frames is not something anyone proofreads.
DEFAULT_SERIES: str = "adaptive"

#: Reports how many projects have been indexed, and how many there are.
ScanProgress = Callable[[int, int], None]

#: Scans a directory and returns the projects found in it, reporting as it
#: goes where it is given somewhere to report to.
ProjectScanner = Callable[[Path, "ScanProgress | None"], ProjectIndex]


@dataclass(frozen=True)
class ViewPosition:
    """Where the view is, in terms that survive a rescan.

    Projects and lamellae are named by path because a rescan rebuilds every
    object and can renumber them: a project added or removed on disk shifts
    every index after it. The frame keeps its index, which is all a frame has.

    Attributes:
        project: Path of the selected project, or None if there was none.
        lamella: Path of the selected lamella, or None.
        frame: Index of the frame within the lamella.
        folder: The folder whose lamellae were listed, or None when a single
            project was. Restoring a folder puts the whole list back, where
            restoring its project alone would silently narrow the scope.
    """

    project: Path | None
    lamella: Path | None
    frame: int
    folder: str | None = None


@dataclass(frozen=True)
class LamellaEntry:
    """One row of the lamella list: a lamella, and the project it is in.

    The project travels with the lamella because the list can be drawn from
    several at once, and a lamella cannot say which one it came from --
    :class:`~...models.Lamella` has no back-reference, the scanner building
    projects out of lamellae rather than the other way round.
    """

    project: Project
    lamella: Lamella


def _index_of(items: Sequence, path: Path | None) -> int | None:
    """Return the position of the item at *path*, or None if it is not there."""
    if path is None:
        return None
    return next(
        (index for index, item in enumerate(items) if item.path == path), None
    )


#: How far curation has progressed for one modality, in one lamella or project.
#:
#: Derived from the records every time it is asked for; nothing is stored. A
#: stored "reviewed" flag would be a second source of truth that could disagree
#: with the files, and this cannot: it *is* the per-image states, summarised.
CurationState = Literal["empty", "not started", "in progress", "done"]

#: A modality with no images at all. Kept distinct from "not started": many
#: older datasets have no FIB whatsoever, and showing those as 0% complete
#: forever would be wrong.
EMPTY: CurationState = "empty"


@dataclass(frozen=True)
class CurationStatus:
    """How far curation has progressed, per series and per beam.

    SEM and FIB are separate because finishing one and not having started the
    other is normal and important, and a single combined mark cannot say it.

    **The series are separate for a stronger reason: combining them was
    wrong.** Reference and alignment images are not labelled by anyone, so they
    sit at the default state for ever, and "selected beside corrected" combines
    to "in progress" -- which reported twenty of the reference collection's 123
    projects as part-done when the only work anyone intends to do in them was
    finished.
    """

    cells: Mapping[tuple[str, str], CurationState]

    def of(self, series: str, modality: str) -> CurationState:
        """Return the status of one series' images of one beam."""
        return self.cells.get((series, modality.upper()), EMPTY)

    def holds(self, series: str) -> bool:
        """Return whether there are images of *series* at all, either beam."""
        return any(self.of(series, modality) != EMPTY for modality in MODALITIES)

    @property
    def summary(self) -> str:
        """Return the long form shown as a tooltip.

        Only a series anyone labels gets its progress read out. For the rest
        the useful fact is that they are there, a state nobody drives being no
        more informative spelled out than drawn.
        """
        lines = []
        for series in SERIES:
            if not self.holds(series):
                continue
            if series in LABELS_EXPECTED:
                lines.append(
                    f"{SERIES_LABELS[series]} — SEM: {self.of(series, 'SEM')}"
                    f" · FIB: {self.of(series, 'FIB')}"
                )
            else:
                lines.append(f"Also holds {SERIES_LABELS[series]} images")
        if not any(self.holds(series) for series in LABELS_EXPECTED):
            lines.insert(0, "No Adaptive Polishing images: nothing to proofread")
        return "\n".join(lines) or "No images"


def _modality_status(records: list) -> CurationState:
    """Return the curation status of one modality's records.

    Every record is considered, not just the base image of each scene. The
    conditions of one scene share a state entry, so they cannot disagree.
    """
    if not records:
        return EMPTY
    states = {current_state(record) for record in records}
    if states <= {"deselected", "corrected"}:
        return "done"
    if states == {"selected"}:
        return "not started"
    return "in progress"


def status_cells(records: Iterable) -> dict[tuple[str, str], CurationState]:
    """Return the state of each series-and-beam pair among *records*.

    Every pair is present in the answer, so a caller reads a cell by position
    rather than asking whether it is there; a pair with no images is
    :data:`EMPTY`.
    """
    grouped: dict[tuple[str, str], list] = {
        (series, modality): [] for series in SERIES for modality in MODALITIES
    }
    for record in records:
        key = (record.series, (record.modality or "").upper())
        if key in grouped:
            grouped[key].append(record)
    return {key: _modality_status(got) for key, got in grouped.items()}


def lamella_status(lamella: Lamella) -> CurationStatus:
    """Return how far curation has progressed in *lamella*, per series and beam.

    Reads the records directly rather than building frames: the chronological
    pairing that :func:`~am_segmentation_manager.frames.build_lamella_frames`
    does is not needed to count states, and skipping it keeps this cheap enough
    to compute for every lamella of a project on demand.
    """
    return CurationStatus(status_cells(lamella.image_records))


def project_status(project: Project) -> CurationStatus:
    """Return how far curation has progressed across *project*, per series and beam.

    Taken from the project's records rather than combined from its lamellae.
    Combining lost which series each answer was about, which is the whole of
    what made the old one wrong; and the weakest-wins rule it used is what a
    cell already does across the images inside it.
    """
    return CurationStatus(status_cells(project.image_records))


@dataclass(frozen=True)
class FolderSummary:
    """What is under one folder of the scan root, counted rather than opened.

    With the folders closed by default the tree is a list of sessions, and the
    question a reader has of each is whether it is worth opening. Counting
    answers it without their having to.

    Attributes:
        folder: The folder this describes, relative to the scan root.
        projects: Autolamella projects filed under it.
        lamellae: Lamellae across those projects.
        images: Indexed images across those projects, every series.
        status: How far curation has got, per series and beam, over the whole
            folder -- read the same way a project's own chip is.
    """

    folder: str
    projects: int
    lamellae: int
    images: int
    status: CurationStatus


@dataclass(frozen=True)
class StateChangeReach:
    """How far a state change would reach, counted before it is made.

    Attributes:
        images: Images the change would cover, after expanding across shared
            labels.
        labels: Distinct label groups those images fall into. Nothing is done
            to the label files themselves; the count says how many separate
            entries the change writes.
        shared: How many of *images* were pulled in by a shared label rather
            than named by the current scope.
        lamellae: Distinct lamellae those images belong to.
        modalities: ``(modality, count)`` over *images*, in
            :data:`~am_segmentation_manager.layout.MODALITIES` order. Reported
            per modality because the edit target moves one or both, and a
            single total cannot say which.
    """

    images: int
    labels: int
    shared: int
    lamellae: int
    modalities: tuple[tuple[str, int], ...] = ()

    def summary(self) -> str:
        """Return the description shown above the state controls."""
        if not self.images:
            return "Nothing selected"
        parts = [
            self._image_part(),
            counted(self.labels, "label", "labels") if self.labels else "no label yet",
            counted(self.lamellae, "lamella", "lamellae"),
        ]
        return "Covers " + ", ".join(parts)

    def _image_part(self) -> str:
        """Return the image count, split by modality where both are present."""
        present = [(name, count) for name, count in self.modalities if count]
        if len(present) < 2:
            return counted(self.images, "image", "images")
        split = " + ".join(f"{count} {name}" for name, count in present)
        return f"{split} images"

    def shared_note(self) -> str:
        """Return the second line, or an empty string when nothing is shared."""
        if not self.shared:
            return ""
        if self.shared == 1:
            return "1 more shares a label with this one"
        return f"{self.shared} more share a label with this one"


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


class EditingDisabled(RuntimeError):
    """Raised when a state change is attempted while editing is switched off.

    The window disables the state controls too; this is the backstop that keeps
    the guard meaningful no matter how the change was triggered.
    """


class CurationController:
    """Drives the curation window.

    Holds the scanned projects, the position within them, how frames are being
    displayed, and the guard around changing state on disk.
    """

    def __init__(
        self,
        *,
        cache: FrameCache | None = None,
        config: SegmentationConfig | None = None,
        scan: ProjectScanner | None = None,
    ) -> None:
        """Create a controller.

        Args:
            cache: Where prepared display data is held. A default cache using
                the real display pipeline is created if not supplied.
            config: Class names and colours. Loaded from the default location
                if not supplied.
            scan: How to discover projects. Defaults to the package scanner;
                tests inject a stub to avoid touching the filesystem.
        """
        self.cache = cache if cache is not None else FrameCache()
        self.config = config if config is not None else load_segmentation_config()
        self._scan: ProjectScanner = scan or (
            lambda path, on_progress=None: Scanner(path).build_index(on_progress)
        )

        self.start_path: Path | None = None
        self.index: ProjectIndex | None = None
        self._projects: list[Project] = []
        #: The lamella list, and the projects its rows came from. Filled from
        #: one project or from every project under a folder; nothing
        #: downstream of it can tell the difference, which is the point.
        self._entries: list[LamellaEntry] = []
        #: The folder whose lamellae are listed, or None when a single project
        #: is. Only the summary line and :meth:`position` ask.
        self._scope_folder: str | None = None
        self._frames: list[LamellaFrame] = []

        #: What each record's label turned out to be, keyed by image path, in
        #: :mod:`.operations.label_io`'s vocabulary. Cached because reading a
        #: label costs a few milliseconds and the timeline asks per frame;
        #: cleared by :meth:`scan`, which is the user saying the filesystem has
        #: moved on.
        self._label_states: dict[Path, str] = {}

        self._project_position = 0
        self._lamella_position = 0
        self._frame_position = 0

        self.mode: DisplayMode = "sem"
        #: The view the user asked for. Kept apart from :attr:`mode` for the
        #: reason :attr:`wanted_series` is kept apart from :attr:`series`: a
        #: timeline imaged with one beam shows the other, and the choice comes
        #: back on a timeline that holds it.
        self.wanted_mode: DisplayMode = "sem"
        #: Which series the timeline shows, one at a time. A 2026 lamella holds
        #: its reference images, its alignment runs and its Adaptive Polishing
        #: stages -- three unrelated things across up to 21 stages -- and a
        #: timeline mixing them is not something anyone proofreads. Starts on
        #: Adaptive Polishing, so the tab opens on what it always showed.
        #:
        #: Read it, never assign it: it is what the current lamella can
        #: actually show of :attr:`wanted_series`, and :meth:`_rebuild_frames`
        #: settles the two.
        self.series: str = DEFAULT_SERIES
        #: The series the user asked for, which a lamella holding none of it
        #: does not overwrite. See :func:`~..frames.resolve_series`.
        self.wanted_series: str = DEFAULT_SERIES
        self.show_overlay = True
        #: Whether the label is drawn over the acquisition or over black.
        self.show_image = True
        #: Which imaging condition is on screen for each modality. None means
        #: the base image. This is display state only: a state change follows
        #: the label, so which condition is being viewed never decides what
        #: moves.
        self.variant: dict[str, str | None] = {modality: None for modality in MODALITIES}
        self.editing_enabled = False
        # Starts matching the initial view, for the same reason the target
        # follows the view thereafter: the modality on screen is the one a
        # state change is meant for.
        self.target: EditTarget = "SEM"
        self.follow_mode = True
        #: What a state change applies to; see :mod:`.selection`.
        self.selection = Selection()

    # -- scanning ----------------------------------------------------------

    def scan(
        self, start_path: Path, on_progress: ScanProgress | None = None
    ) -> list[Project]:
        """Discover projects under *start_path* and select the first one.

        Args:
            on_progress: Passed to the scanner, which calls it once per
                project. The scan holds whichever thread it is on, so this is
                the only chance anything has to say how far it has got.

        Returns:
            The projects found, in scan order.
        """
        self.start_path = Path(start_path)
        # Proofreading happens in another application while this one is open,
        # so a rescan must not answer from labels read before it.
        self._label_states.clear()
        self.index = self._scan(self.start_path, on_progress)
        self._projects = self.index.all_projects()
        logger.info("Found %d project(s) under %s", len(self._projects), self.start_path)
        self.select_project(0)
        return self._projects

    def rescan(self, on_progress: ScanProgress | None = None) -> list[Project]:
        """Read the same root again, keeping the view and what is ticked.

        What follows an operation that changed the collection. The index is
        derived from the filesystem rather than patched, because the filesystem
        is what the operation changed -- but neither where the reviewer had got
        to nor what they had picked out is a fact about the filesystem, and
        losing either after every operation makes a long session unworkable.
        The selection matters most between the two steps of preparing labels:
        predicting and converting act on the same set, and a scan between them
        that emptied it would have the second ask for the choice again.

        Raises:
            RuntimeError: If nothing has been scanned yet.
        """
        if self.start_path is None:
            raise RuntimeError("Nothing has been scanned yet")
        position = self.position()
        picked = self._snapshot_selection()
        projects = self.scan(self.start_path, on_progress)
        self.restore(position)
        self._restore_selection(picked)
        return projects

    def _snapshot_selection(self) -> SelectionSnapshot:
        """Return what is ticked, named so the coming scan can put it back."""
        return SelectionSnapshot(
            projects=frozenset(self.selection.projects),
            lamellae=frozenset(self.selection.lamellae),
            frames=frozenset(
                self._frames[position].key
                for position in self.selection.frames
                if position < len(self._frames)
            ),
        )

    def _restore_selection(self, picked: SelectionSnapshot) -> None:
        """Put back what was ticked, dropping whatever the scan no longer holds.

        A directory that has gone is dropped rather than kept as a tick nobody
        can see: what is ticked on screen has to be the whole of what a change
        will touch. The lamellae are matched against the project now selected,
        since that is the only one whose ticks are displayed -- the same rule
        :meth:`select_project` applies.
        """
        positions = {frame.key: index for index, frame in enumerate(self._frames)}
        self.selection.restore(
            projects=picked.projects
            & {selection_key(project.path) for project in self._projects},
            lamellae=picked.lamellae
            & {selection_key(entry.lamella.path) for entry in self._entries},
            frames={positions[key] for key in picked.frames if key in positions},
        )

    def position(self) -> ViewPosition:
        """Return where the view is, named by path rather than by index."""
        project, lamella = self.project, self.lamella
        return ViewPosition(
            project=project.path if project is not None else None,
            lamella=lamella.path if lamella is not None else None,
            frame=self._frame_position,
            folder=self._scope_folder,
        )

    def restore(self, position: ViewPosition) -> None:
        """Move back to *position*, as nearly as what was found allows.

        Anything that has gone since is dropped rather than searched for: a
        lamella that is no longer there is one the reviewer has to be told
        about by seeing where they now are.
        """
        if position.folder is not None:
            self.select_folder(position.folder)
        else:
            project = _index_of(self._projects, position.project)
            if project is None:
                return
            self.select_project(project)
        lamella = _index_of(
            [entry.lamella for entry in self._entries], position.lamella
        )
        if lamella is None:
            return
        self.select_lamella(lamella)
        self.select_frame(position.frame)

    # -- selection ---------------------------------------------------------

    @property
    def projects(self) -> list[Project]:
        """Return the discovered projects."""
        return self._projects

    @property
    def lamellae(self) -> list[Lamella]:
        """Return the lamellae listed: one project's, or a whole folder's."""
        return [entry.lamella for entry in self._entries]

    @property
    def scope_folder(self) -> str | None:
        """Return the folder whose lamellae are listed, or None for a project."""
        return self._scope_folder

    @property
    def frames(self) -> list[LamellaFrame]:
        """Return the frames of the selected lamella, in acquisition order."""
        return self._frames

    @property
    def project(self) -> Project | None:
        """Return the project the lamella in view belongs to.

        The current *lamella's* project, not the row last clicked in the tree.
        With a folder listed the two differ, and everything that asks -- the
        status line, the variant names a project gives its own directories, the
        path to copy -- is asking about what is on screen. A project holding no
        lamellae has none to answer for, so the tree's own row stands in and it
        still names itself.
        """
        entry = self._at(self._entries, self._lamella_position)
        if entry is not None:
            return entry.project
        return self._at(self._projects, self._project_position)

    @property
    def lamella(self) -> Lamella | None:
        """Return the selected lamella, if any."""
        entry = self._at(self._entries, self._lamella_position)
        return entry.lamella if entry is not None else None

    @property
    def frame(self) -> LamellaFrame | None:
        """Return the frame being viewed, if any."""
        return self._at(self._frames, self._frame_position)

    @property
    def project_position(self) -> int:
        """Return the index of the selected project."""
        return self._project_position

    @property
    def lamella_position(self) -> int:
        """Return the index of the selected lamella within the list."""
        return self._lamella_position

    @property
    def frame_position(self) -> int:
        """Return the index of the frame being viewed within the lamella."""
        return self._frame_position

    def select_project(self, position: int) -> None:
        """List one project's lamellae, and move into them."""
        self._project_position = _clamp(position, len(self._projects))
        project = self._at(self._projects, self._project_position)
        self._list_lamellae(
            [project] if project is not None else [], folder=None
        )

    def select_folder(self, folder: str) -> None:
        """List the lamellae of every project under *folder*, in project order.

        An Autolamella run restarts for reasons that have nothing to do with
        the specimen -- a crash, a change of parameters, an overnight queue --
        so one session's lamellae are spread over a dozen projects holding one
        or two each. Listing them together is how they are actually worked
        through; stepping with Up and Down then crosses a project boundary
        without stopping, because the boundary is an artifact.

        The tree's current row stays on the folder. Moving it onto whichever
        project the lamella belongs to would re-enter this list's own signal
        and snap the scope back to that one project.
        """
        self._scope_folder = folder
        self._list_lamellae(self.projects_in(folder), folder=folder)

    def projects_in(self, folder: str) -> list[Project]:
        """Return the projects filed under *folder*, in scan order."""
        return [project for project in self._projects if project.folder == folder]

    def _list_lamellae(self, projects: list[Project], *, folder: str | None) -> None:
        """Replace the lamella list with the lamellae of *projects*.

        **Where you are is kept if it is still on the list.** Widening from a
        project to the folder holding it, or narrowing back, then moves nothing
        -- which is what makes choosing a folder a question about what to look
        at rather than a jump to somewhere else. A different project never
        holds the current lamella, so this changes nothing there.

        The ticked lamellae are cleared: they belonged to the list that was on
        screen, and a tick nobody can see is one that decides what an operation
        touches without saying so. The ticked projects stay, the tree still
        showing them.
        """
        self._scope_folder = folder
        was = self.lamella
        self._entries = [
            LamellaEntry(project, lamella)
            for project in projects
            for lamella in project.lamellae
        ]
        self.selection.lamellae.clear()
        self.select_lamella(
            _index_of([entry.lamella for entry in self._entries],
                      was.path if was is not None else None) or 0
        )

    def select_lamella(self, position: int) -> None:
        """Select a lamella by index and rebuild its frame stack.

        Everything prepared for the previous lamella is discarded: at a lamella
        boundary none of it is worth keeping. The frame selection goes with it,
        since a frame index only means anything within one lamella.
        """
        self._lamella_position = _clamp(position, len(self._entries))
        self._rebuild_frames()

    def _rebuild_frames(self) -> None:
        """Rebuild the frame stack for the current lamella and series.

        Everything prepared for the previous stack is discarded: at a lamella
        boundary none of it is worth keeping, and a change of series replaces
        the timeline outright. The frame selection goes with it, since a frame
        index only means anything within one stack.

        **What is shown is settled here**, and nowhere else: this is the one
        path both a change of lamella and a change of series pass through, so
        neither can leave the window on an empty timeline or on a beam this
        one holds no image of.
        """
        lamella = self.lamella
        self.series = resolve_series(self.wanted_series, self.available_series())
        self._frames = (
            build_lamella_frames(lamella, series=self.series)
            if lamella is not None
            else []
        )
        self.cache.set_frames(self._frames)
        self._frame_position = 0
        self.selection.clear_frames()
        self._settle_mode()
        logger.debug(
            "Selected lamella %s with %d %s frame(s)",
            lamella.name if lamella is not None else "(none)",
            len(self._frames),
            self.series,
        )

    def select_frame(self, position: int) -> None:
        """Move to a frame by index, clamped to the lamella's range."""
        self._frame_position = _clamp(position, len(self._frames))

    def step_frame(self, delta: int) -> None:
        """Move *delta* frames, stopping at the ends of the lamella."""
        self.select_frame(self._frame_position + delta)

    def step_lamella(self, delta: int) -> None:
        """Move *delta* lamellae, stopping at the ends of the list.

        The list, not the project: with a folder listed this steps from the
        last lamella of one project into the first of the next.
        """
        self.select_lamella(self._lamella_position + delta)

    # -- display -----------------------------------------------------------

    def set_mode(self, mode: DisplayMode) -> None:
        """Switch between the SEM, FIB and plot views.

        Unless :attr:`follow_mode` is off, switching to a single-modality view
        also points the edit target at that modality. Changing the state of the
        image you are *not* looking at is far more likely to be a mistake than
        an intention, so the target follows the eye. The plot view leaves the
        target alone, since it shows both modalities' context at once.
        """
        if mode not in ("sem", "fib", "plot"):
            raise ValueError(f"Unknown display mode: {mode}")
        self.wanted_mode = mode
        self._show_mode(mode)

    def _show_mode(self, mode: DisplayMode) -> None:
        """Put *mode* on screen without recording it as what was asked for.

        What :meth:`_settle_mode` calls when it has to show the other beam.
        """
        self.mode = mode
        if self.follow_mode:
            modality = MODE_MODALITIES.get(mode)
            if modality is not None:
                self.target = modality

    def _modalities_present(self) -> list[str]:
        """Return the beams this timeline holds an image of, in display order.

        Asked of the whole stack rather than of one frame: an acquisition
        captured with one beam is ordinary, and the view should not follow it.
        """
        return [
            modality for modality in MODALITIES
            if any(frame.modality_records(modality) for frame in self._frames)
        ]

    def _settle_mode(self) -> None:
        """Show a beam this timeline actually holds.

        An alignment run is imaged with the ion beam alone, so opening one with
        the SEM view selected showed "no image for this modality" on every
        frame of it, and the reviewer had to notice and switch. The choice
        itself is untouched, so a timeline that does hold it opens on it again.

        The plot view is left alone: it says what is available for itself, and
        a figure is not an image of either beam.
        """
        if self.wanted_mode == "plot":
            return
        present = self._modalities_present()
        if not present:
            return
        wanted = MODE_MODALITIES[self.wanted_mode]
        mode = self.wanted_mode if wanted in present else MODALITY_MODES[present[0]]
        if mode != self.mode:
            logger.debug("Showing %s: this timeline holds %s", mode, present)
            self._show_mode(mode)

    def available_modes(self, frame: LamellaFrame | None = None) -> list[DisplayMode]:
        """Return the views that have something to show for a frame.

        A frame may have only one modality, or no plot, and the window marks
        those views unavailable rather than showing an empty panel.
        """
        frame = frame if frame is not None else self.frame
        if frame is None:
            return []
        modes: list[DisplayMode] = []
        if frame.sem:
            modes.append("sem")
        if frame.fib:
            modes.append("fib")
        if frame.displayed_plot is not None:
            modes.append("plot")
        return modes

    def available_series(self) -> list[str]:
        """Return the series the current lamella actually holds, in order.

        A pre-Autolamella lamella has only Adaptive Polishing images and gets
        no choice to make -- but the control still shows what it is on, rather
        than disappearing.
        """
        return series_present(self.lamella)

    def set_series(self, series: str) -> None:
        """Show one series' timeline, rebuilding the frame stack.

        Unlike the variant, this is not only a view choice: it decides which
        frames exist, so what is ticked and where the view sits go with it.

        Remembered whether or not this lamella can show it, so that moving
        through lamellae that hold different series does not quietly change
        what the reviewer asked to look at.
        """
        if series not in SERIES:
            raise ValueError(f"Unknown series: {series!r}; expected one of {list(SERIES)}")
        self.wanted_series = series
        if series == self.series:
            return
        self._rebuild_frames()

    def available_variants(self, modality: str | None = None) -> list[str | None]:
        """Return the imaging conditions the current frame has for a modality.

        Base first. A list of one -- the usual case -- means there is nothing
        to choose between, and the window hides the control.
        """
        frame = self.frame
        target = modality or MODE_MODALITIES.get(self.mode)
        if frame is None or target is None:
            return []
        return frame.variants(target)

    def set_variant(self, variant: str | None, modality: str | None = None) -> None:
        """Choose which imaging condition to display for a modality.

        Purely a view choice. State changes follow the label, so the condition
        on screen never decides what a state change moves.
        """
        target = modality or MODE_MODALITIES.get(self.mode)
        if target is None:
            return
        self.variant[target] = variant
        logger.debug("Viewing %s variant %s", target, variant or "base")

    def current_variant(self, modality: str | None = None) -> str | None:
        """Return the imaging condition on screen for a modality."""
        target = modality or MODE_MODALITIES.get(self.mode)
        if target is None:
            return None
        return self.variant.get(target)

    # -- curation progress, for the project and lamella lists --------------

    def project_statuses(self) -> list[CurationStatus]:
        """Return each discovered project's curation status, in list order."""
        return [project_status(project) for project in self._projects]

    def lamella_statuses(self) -> list[CurationStatus]:
        """Return each lamella's curation status for the selected project."""
        return [lamella_status(entry.lamella) for entry in self._entries]

    def folder_summary(self, folder: str) -> FolderSummary:
        """Return what is under *folder*, counted across the projects in it.

        A folder holding no projects -- one that has just been renamed out from
        under the window -- counts zero rather than raising: this answers a
        question about the tree on screen, and the tree is redrawn by the
        rescan that follows, not by this.
        """
        projects = [
            project for project in self.projects_in(folder)
        ]
        records = [
            record for project in projects for record in project.image_records
        ]
        return FolderSummary(
            folder=folder,
            projects=len(projects),
            lamellae=sum(len(project.lamellae) for project in projects),
            images=len(records),
            status=CurationStatus(status_cells(records)),
        )

    def project_path(self, position: int | None = None) -> Path | None:
        """Return the absolute path of a project, or of the selected one."""
        project = self._at(
            self._projects, self._project_position if position is None else position
        )
        return project.path.resolve() if project is not None else None

    def lamella_path(self, position: int | None = None) -> Path | None:
        """Return the absolute path of a lamella, or of the selected one.

        A synthetic lamella -- one the dataset has no directory for, where the
        project directory stands in -- returns the directory that stood in, so
        the path always names something that exists on disk.
        """
        lamella = self._at(
            self.lamellae, self._lamella_position if position is None else position
        )
        return lamella.path.resolve() if lamella is not None else None

    def current_image_path(self) -> Path | None:
        """Return the absolute path of the image currently on screen.

        Follows the mode and the variant dropdown, so this is the file the user
        is actually looking at rather than the frame's base image. Returns None
        in plot mode, which has no single acquisition behind it.
        """
        modality = MODE_MODALITIES.get(self.mode)
        frame = self.frame
        if modality is None or frame is None:
            return None
        record = frame.record(modality, self.variant.get(modality))
        return record.image_path.resolve() if record is not None else None

    def current_plot_path(self) -> Path | None:
        """Return the absolute path of the summary plot on screen, if any."""
        frame = self.frame
        if frame is None or frame.displayed_plot is None:
            return None
        return frame.displayed_plot.resolve()

    def current_image(self) -> FrameImage | None:
        """Return the prepared acquisition for the current mode and frame.

        Returns None in plot mode, or when the frame has no such acquisition.
        """
        modality = MODE_MODALITIES.get(self.mode)
        if modality is None:
            return None
        return self.cache.image(self._frame_position, modality, self.variant.get(modality))

    def current_plot(self) -> FramePlot | None:
        """Return the prepared plot for the current frame, if it has one."""
        return self.cache.plot(self._frame_position)

    def pending_loads(self, *, radius: int | None = DEFAULT_RADIUS) -> list[LoadRequest]:
        """Return preparation work for the current position, most useful first.

        The mode in view is prepared first so the visible panel fills before
        anything else. Plots are only included while the plot view is on,
        because they cost several times more than an acquisition.
        """
        modality = MODE_MODALITIES.get(self.mode)
        ordered = [modality, *(m for m in ("SEM", "FIB") if m != modality)] if modality else ["SEM", "FIB"]
        showing_plots = self.mode == "plot"
        return self.cache.pending(
            self._frame_position,
            [m for m in ordered if m is not None],
            variants=dict(self.variant),
            # A plot takes ~180 ms to read, which is slower than a held-down
            # arrow key, so preparing only the immediate neighbours guarantees
            # the user outruns the loader. In the plot view everything the
            # cache can keep is prepared instead, working outwards from where
            # they are -- and no more than that, or the far end of a long
            # lamella evicts the near end as it goes.
            radius=self.cache.plot_radius if showing_plots else radius,
            include_plots=showing_plots,
        )

    # -- workflow state ----------------------------------------------------

    def state_of(self, modality: str, frame: LamellaFrame | None = None) -> RecordState | None:
        """Return a frame's on-disk state for one modality, or None if absent."""
        frame = frame if frame is not None else self.frame
        if frame is None:
            return None
        record = frame.record(modality)
        return current_state(record) if record is not None else None

    def combined_state(
        self, frame: LamellaFrame | None = None
    ) -> RecordState | Literal["mixed"] | None:
        """Return a frame's state across both modalities.

        Returns :data:`MIXED` when the two acquisitions are in different states,
        which the window shows as no selection rather than picking a side. This
        is a normal situation: curation deselects one modality at a time.

        Read from the base image of each modality, which is what the window
        displays. Variants are not consulted here; once state changes expand
        across shared labels, a variant's state will follow its group.
        """
        frame = frame if frame is not None else self.frame
        if frame is None:
            return None
        states = {current_state(record) for record in frame.base_records}
        if not states:
            return None
        if len(states) > 1:
            return MIXED
        return states.pop()

    def target_state(self) -> RecordState | Literal["mixed"] | None:
        """Return the state that the current edit target is in."""
        if self.target == "both":
            return self.combined_state()
        return self.state_of(self.target)

    def frame_states(self) -> list[RecordState | Literal["mixed"] | None]:
        """Return the combined state of every frame in the lamella.

        Lets the window show curation progress across the whole lamella at a
        glance, instead of requiring the user to scrub through it.
        """
        return [self.combined_state(frame) for frame in self._frames]

    def modality_states(self) -> dict[str, list[RecordState | None]]:
        """Return each modality's state for every frame in the lamella.

        Reported per modality rather than combined because a state change
        commonly targets one of them: a frame with its SEM deselected and its
        FIB still selected is a normal intermediate step, and collapsing that
        to "mixed" hides which modality is in which state.
        """
        return {
            modality: [self.state_of(modality, frame) for frame in self._frames]
            for modality in ("SEM", "FIB")
        }

    def label_states(self) -> dict[str, list[str | None]]:
        """Return each modality's label state for every frame in the lamella.

        One of ``"missing"``, ``"empty"``, ``"present"``, or None when the
        frame has no image of that modality at all.

        The distinction that matters is ``"empty"``: the segmentation software
        writes a label for every image it is pointed at, so a label file is not
        evidence of annotation, and progress read from file existence alone
        would overstate how much work is done.

        Reading a label costs a few milliseconds, so results are cached by path
        for as long as the controller lives -- navigating back and forth across
        a lamella re-reads nothing.
        """
        return {
            modality: [
                self._label_state(frame.record(modality)) for frame in self._frames
            ]
            for modality in MODALITIES
        }

    def record_label_state(self, record) -> str:
        """Return what *record*'s label turned out to be, reading it once.

        One of :mod:`.operations.label_io`'s four states. The timeline and the
        guard on marking corrected both ask this, so they cannot disagree
        about a label between them.
        """
        path = record.image_path
        if path not in self._label_states:
            self._label_states[path] = label_io.label_state(record)
        return self._label_states[path]

    def _label_state(self, record) -> str | None:
        """Return one record's label state as the timeline draws it."""
        if record is None:
            return None
        return LABEL_DISPLAY[self.record_label_state(record)]

    def set_target(self, target: EditTarget) -> None:
        """Choose whether state changes apply to SEM, FIB, or both."""
        if target not in ("SEM", "FIB", "both"):
            raise ValueError(f"Unknown edit target: {target}")
        self.target = target

    # -- what a state change applies to ------------------------------------
    #
    # The window works in row numbers; the selection works in resolved paths,
    # because a row means a different lamella as soon as the project changes.
    # Translating between the two is this layer's job.

    def set_project_checked(self, position: int, checked: bool) -> None:
        """Check or uncheck the project at a row of the project list."""
        project = self._at(self._projects, position)
        if project is not None:
            self.selection.check(LEVEL_PROJECTS, selection_key(project.path), checked)

    def set_lamella_checked(self, position: int, checked: bool) -> None:
        """Check or uncheck the lamella at a row of the lamella list."""
        lamella = self._at(self.lamellae, position)
        if lamella is not None:
            self.selection.check(LEVEL_LAMELLAE, selection_key(lamella.path), checked)

    def check_all_projects(self, checked: bool) -> None:
        """Check or uncheck every discovered project."""
        self.selection.check_all(
            LEVEL_PROJECTS,
            [selection_key(project.path) for project in self._projects],
            checked,
        )

    def check_all_lamellae(self, checked: bool) -> None:
        """Check or uncheck every lamella of the selected project.

        Lamellae checked in other projects are left alone: the list only shows
        this project's, and un-checking what is not on screen would be a change
        nobody could see.
        """
        self.selection.check_all(
            LEVEL_LAMELLAE,
            [selection_key(entry.lamella.path) for entry in self._entries],
            checked,
        )

    def project_checks(self) -> list[bool]:
        """Return whether each project is checked, in list order."""
        return [
            self.selection.is_checked(LEVEL_PROJECTS, selection_key(project.path))
            for project in self._projects
        ]

    def lamella_checks(self) -> list[bool]:
        """Return whether each listed lamella is checked."""
        return [
            self.selection.is_checked(
                LEVEL_LAMELLAE, selection_key(entry.lamella.path)
            )
            for entry in self._entries
        ]

    def selected_frames(self) -> set[int]:
        """Return the frames picked out on the timeline."""
        return set(self.selection.frames)

    def describe_selection(self) -> str:
        """Return a phrase naming what a state change would apply to."""
        return self.selection.describe()

    def target_records(self, frame: LamellaFrame | None = None) -> list:
        """Return the image records a state change would apply to in one frame.

        The base image of each targeted modality. A frame with several imaging
        conditions reaches its variants through their shared label rather than
        through this list.
        """
        frame = frame if frame is not None else self.frame
        if frame is None:
            return []
        if self.target == "both":
            return frame.base_records
        record = frame.record(self.target)
        return [record] if record is not None else []

    def scoped_records(self) -> list:
        """Return every record a state change would apply to.

        Only one level is ever picked out at a time -- see :mod:`.selection` --
        so this reads whichever one holds anything. Everything downstream
        (:meth:`expanded_records`, :meth:`reach`, :meth:`apply_state`) reads
        this one method, so label-group expansion and the corrected-data guard
        apply to a checked selection unchanged.

        Whole lamellae are read straight from the scanned stages rather than
        through frames, since the chronological pairing only matters for display.
        """
        level = self.selection.level
        if level == LEVEL_CURRENT_FRAME:
            return self.target_records()
        if level == LEVEL_PROJECTS:
            return self._records_of(self._checked_lamellae())
        if level == LEVEL_LAMELLAE:
            return self._records_of(self._checked_lamellae())
        return [
            record
            for position in sorted(self.selection.frames)
            if 0 <= position < len(self._frames)
            for record in self.target_records(self._frames[position])
        ]

    def _checked_lamellae(self) -> list[Lamella]:
        """Return the lamellae the selection covers.

        Ticked lamellae are the selected project's own: they are cleared when
        the project changes, so this searches only what the list is showing.
        Ticked projects are looked up across the whole scan, which is what the
        project list shows.
        """
        if self.selection.lamellae:
            wanted = self.selection.lamellae
            return [
                lamella
                for lamella in self.lamellae
                if selection_key(lamella.path) in wanted
            ]
        wanted = self.selection.projects
        return [
            lamella
            for project in self._projects
            if selection_key(project.path) in wanted
            for lamella in project.lamellae
        ]

    def _records_of(self, lamellae: list[Lamella]) -> list:
        """Return every record of *lamellae* in the edit target and the series shown.

        **Both narrowings, not only the modality.** A 2026 lamella holds its
        reference images and its alignment runs beside its Adaptive Polishing
        stages, and the tab shows one series at a time; a ticked lamella that
        reached every series would predict, convert or restate images the
        timeline was not showing. Frame-level selection was already scoped this
        way, being built from the frames themselves.

        Each lamella is asked what it can show of the chosen series, exactly as
        navigating to it would -- see :func:`~..frames.resolve_series` -- so a
        lamella holding none of it contributes the series it does hold rather
        than nothing at all.
        """
        modalities = ("SEM", "FIB") if self.target == "both" else (self.target,)
        records: list = []
        for lamella in lamellae:
            series = resolve_series(self.wanted_series, series_present(lamella))
            records.extend(
                record
                for record in lamella.image_records
                if record.modality in modalities and record.series == series
            )
        return records

    def corrected_records(self, records: list | None = None) -> list:
        """Return those records currently in the corrected state.

        The window needs this to name only the files that actually hold
        proofread work when it asks for confirmation.
        """
        candidates = records if records is not None else self.scoped_records()
        return [record for record in candidates if current_state(record) == "corrected"]

    def expanded_records(self, records: list | None = None) -> list:
        """Return every record a state change would actually move.

        Wider than :meth:`scoped_records` whenever a label is shared: a scene
        captured under several imaging conditions shares one label, and the
        change follows the label to every image of that scene. Anything
        reporting how much is about to move must count *this*, not the scope.
        """
        candidates = records if records is not None else self.scoped_records()
        if not candidates:
            return []
        return expand_to_groups(candidates)

    def shared_label_records(self, records: list | None = None) -> list:
        """Return images a state change would reach through a shared label.

        These are not in the current selection, so the window must be able to
        say how far the change actually goes before making it. Empty in the
        common case where nothing is shared.
        """
        candidates = records if records is not None else self.scoped_records()
        if not candidates:
            return []
        requested = {str(record.image_path) for record in candidates}
        return [
            record for record in self.expanded_records(candidates)
            if str(record.image_path) not in requested
        ]

    def label_readiness(self, records: list | None = None) -> LabelReadiness:
        """Return the images of a selection that hold no annotation.

        Asked before marking a set corrected, which is the claim that somebody
        proofread these labels. Counted over the *expanded* set, because a
        shared label is marked corrected by any image of its scene.

        Reads each label at most once per scan, through
        :meth:`record_label_state`, so the timeline has usually paid for it
        already.
        """
        candidates = records if records is not None else self.scoped_records()
        if not candidates:
            return LabelReadiness()
        return classify_labels(
            self.expanded_records(candidates), self.record_label_state
        )

    def reach(self) -> StateChangeReach:
        """Return how far a state change would go, for display before it is made.

        Counted from the same expansion the operation performs, so the number
        shown cannot drift from the number of records declared.
        """
        scoped = self.scoped_records()
        if not scoped:
            return StateChangeReach(images=0, labels=0, shared=0, lamellae=0)
        expanded = self.expanded_records(scoped)
        labels = {
            str(label)
            for record in expanded
            for label in record.labels
        }
        # Qualified by project because lamella names repeat between projects,
        # and a change over ticked projects spans several.
        lamellae = {(record.project, record.lamella) for record in expanded}
        return StateChangeReach(
            images=len(expanded),
            labels=len(labels),
            shared=len(expanded) - len(scoped),
            lamellae=len(lamellae),
            modalities=tuple(
                (modality, sum(1 for r in expanded if r.modality == modality))
                for modality in MODALITIES
            ),
        )

    def apply_state(self, state: RecordState, *, confirm_corrected: bool = False) -> int:
        """Declare the current frame's targeted records to be in *state*.

        Delegates to :func:`transition_records`, which validates the whole set
        before writing anything and refuses to take data out of the corrected
        state unless *confirm_corrected* is set. Both behaviours are
        deliberately left intact: this method adds the editing guard and
        nothing else.

        Args:
            state: The workflow state to declare.
            confirm_corrected: Pass on the user's explicit confirmation to take
                data out of the corrected state.

        Returns:
            The number of records declared.

        Raises:
            EditingDisabled: If editing has not been switched on.
            PermissionError: If corrected data would be changed without
                confirmation.
            StaleState: If the collection was curated somewhere else since this
                window read it. Nothing is written.
            UnmigratedStage, FileNotFoundError: Propagated unchanged so the
                window can report exactly what went wrong.
        """
        if not self.editing_enabled:
            raise EditingDisabled(
                "Switch on editing before changing the state of any image"
            )
        records = self.scoped_records()
        if not records:
            logger.info(
                "Nothing to change: no %s image in %s",
                self.target, self.selection.describe(),
            )
            return 0

        changed = transition_records(
            records, state, allow_leaving_corrected=confirm_corrected
        )
        logger.info(
            "Declared %d record(s) %s across %s",
            changed, state, self.selection.describe(),
        )
        return changed

    def set_editing_enabled(self, enabled: bool) -> None:
        """Switch the guard on state changes on or off."""
        self.editing_enabled = bool(enabled)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _at(items: Sequence, position: int):
        """Return ``items[position]``, or None when the sequence is empty."""
        if not items or not 0 <= position < len(items):
            return None
        return items[position]


def _clamp(position: int, length: int) -> int:
    """Return *position* limited to a valid index, or 0 when there is nothing."""
    if length <= 0:
        return 0
    return max(0, min(position, length - 1))
