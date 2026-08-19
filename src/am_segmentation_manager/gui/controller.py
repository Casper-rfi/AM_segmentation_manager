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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config import SegmentationConfig, load_segmentation_config
from ..frames import LamellaFrame, build_lamella_frames
from ..layout import MODALITIES, base_path
from ..models import Lamella, Project
from ..operations import label_io
from ..operations.label_groups import build_groups, expand_to_groups
from ..operations.label_io import LabelReadiness, classify_labels
from ..operations.selection import RecordState, current_state, transition_records
from ..scanner import ProjectIndex, Scanner
from .cache import (
    DEFAULT_RADIUS,
    PLOT_RADIUS,
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

#: Scans a directory and returns the projects found in it.
ProjectScanner = Callable[[Path], ProjectIndex]


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
    """

    project: Path | None
    lamella: Path | None
    frame: int


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
CurationState = Literal["empty", "not started", "in progress", "done", "needs repair"]

#: A modality with no images at all. Kept distinct from "not started": many
#: older datasets have no FIB whatsoever, and showing those as 0% complete
#: forever would be wrong.
EMPTY: CurationState = "empty"

#: Images disagree with the label they share. Overrides every other status,
#: because until ``am-seg repair-state`` runs, the state shown for the group is
#: the image's rather than the label's -- so "done" would be a claim the files
#: do not support.
NEEDS_REPAIR: CurationState = "needs repair"


@dataclass(frozen=True)
class CurationStatus:
    """How far curation has progressed, reported per modality.

    SEM and FIB are separate because finishing one and not having started the
    other is a normal and important situation that a single combined mark
    cannot express.
    """

    sem: CurationState
    fib: CurationState

    def of(self, modality: str) -> CurationState:
        """Return the status for ``'SEM'`` or ``'FIB'``."""
        normalised = modality.upper()
        if normalised == "SEM":
            return self.sem
        if normalised == "FIB":
            return self.fib
        raise ValueError(f"Unknown modality: {modality}")

    @property
    def summary(self) -> str:
        """Return the long form shown as a tooltip."""
        return f"SEM: {self.sem} · FIB: {self.fib}"


def _modality_status(records: list) -> CurationState:
    """Return the curation status of one modality's records.

    Every record is considered, not just the base image of each scene. Variants
    share a label and so should share its state; where they do not, the group is
    inconsistent and reported as needing repair rather than resolved silently.
    """
    if not records:
        return EMPTY
    if any(not group.is_consistent for group in build_groups(records)):
        return NEEDS_REPAIR
    states = {current_state(record) for record in records}
    if states <= {"deselected", "corrected"}:
        return "done"
    if states == {"selected"}:
        return "not started"
    return "in progress"


def lamella_status(lamella: Lamella) -> CurationStatus:
    """Return how far curation has progressed in *lamella*, per modality.

    Reads the records directly rather than building frames: the chronological
    pairing that :func:`~am_segmentation_manager.frames.build_lamella_frames`
    does is not needed to count states, and skipping it keeps this cheap enough
    to compute for every lamella of a project on demand.
    """
    by_modality: dict[str, list] = {modality: [] for modality in MODALITIES}
    for stage in lamella.stages:
        for record in stage.images:
            if record.modality in by_modality:
                by_modality[record.modality].append(record)
    return CurationStatus(
        sem=_modality_status(by_modality["SEM"]),
        fib=_modality_status(by_modality["FIB"]),
    )


def _combine(statuses: list[CurationState]) -> CurationState:
    """Return the status of a whole from its parts, for one modality.

    The weakest wins rather than an average: "90% done" is not something you can
    act on, whereas "one lamella left" is. Lamellae with no images of the
    modality contribute nothing -- a project is not part-finished because half
    its lamellae never had a FIB image.
    """
    if NEEDS_REPAIR in statuses:
        return NEEDS_REPAIR
    present = [status for status in statuses if status != EMPTY]
    if not present:
        return EMPTY
    if len(set(present)) == 1:
        return present[0]
    return "in progress"


def project_status(project: Project) -> CurationStatus:
    """Return how far curation has progressed across *project*, per modality."""
    statuses = [lamella_status(lamella) for lamella in project.lamellae]
    if not statuses:
        return CurationStatus(sem=EMPTY, fib=EMPTY)
    return CurationStatus(
        sem=_combine([status.sem for status in statuses]),
        fib=_combine([status.fib for status in statuses]),
    )


@dataclass(frozen=True)
class StateChangeReach:
    """How much a state change would move, counted before it is made.

    Attributes:
        images: Images that would move, after expanding across shared labels.
        labels: Distinct label files that would move with them.
        shared: How many of *images* were pulled in by a shared label rather
            than named by the current scope.
    """

    images: int
    labels: int
    shared: int

    def summary(self) -> str:
        """Return the one-line description shown above the state controls."""
        if not self.images:
            return "Nothing selected"
        label_part = (
            f"{self.labels} label" if self.labels == 1 else f"{self.labels} labels"
        )
        image_part = (
            f"{self.images} image" if self.images == 1 else f"{self.images} images"
        )
        if not self.labels:
            return f"Moves {image_part}, no label yet"
        return f"Moves {image_part} + {label_part}"

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
        self._scan: ProjectScanner = scan or (lambda path: Scanner(path).build_index())

        self.start_path: Path | None = None
        self.index: ProjectIndex | None = None
        self._projects: list[Project] = []
        self._lamellae: list[Lamella] = []
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

    def scan(self, start_path: Path) -> list[Project]:
        """Discover projects under *start_path* and select the first one.

        Returns:
            The projects found, in scan order.
        """
        self.start_path = Path(start_path)
        # Proofreading happens in another application while this one is open,
        # so a rescan must not answer from labels read before it.
        self._label_states.clear()
        self.index = self._scan(self.start_path)
        self._projects = self.index.all_projects()
        logger.info("Found %d project(s) under %s", len(self._projects), self.start_path)
        self.select_project(0)
        return self._projects

    def rescan(self) -> list[Project]:
        """Read the same root again, staying where the view is.

        What follows an operation that moved files. The index is derived from
        the filesystem rather than patched, because the filesystem is what the
        operation changed -- but where the reviewer had got to is not a fact
        about the filesystem, and starting them at the first project again
        after every save makes a long session unworkable.

        Raises:
            RuntimeError: If nothing has been scanned yet.
        """
        if self.start_path is None:
            raise RuntimeError("Nothing has been scanned yet")
        position = self.position()
        projects = self.scan(self.start_path)
        self.restore(position)
        return projects

    def position(self) -> ViewPosition:
        """Return where the view is, named by path rather than by index."""
        project, lamella = self.project, self.lamella
        return ViewPosition(
            project=project.path if project is not None else None,
            lamella=lamella.path if lamella is not None else None,
            frame=self._frame_position,
        )

    def restore(self, position: ViewPosition) -> None:
        """Move back to *position*, as nearly as what was found allows.

        Anything that has gone since is dropped rather than searched for: a
        lamella that is no longer there is one the reviewer has to be told
        about by seeing where they now are.
        """
        project = _index_of(self._projects, position.project)
        if project is None:
            return
        self.select_project(project)
        lamella = _index_of(self._lamellae, position.lamella)
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
        """Return the lamellae of the selected project."""
        return self._lamellae

    @property
    def frames(self) -> list[LamellaFrame]:
        """Return the frames of the selected lamella, in acquisition order."""
        return self._frames

    @property
    def project(self) -> Project | None:
        """Return the selected project, if any."""
        return self._at(self._projects, self._project_position)

    @property
    def lamella(self) -> Lamella | None:
        """Return the selected lamella, if any."""
        return self._at(self._lamellae, self._lamella_position)

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
        """Return the index of the selected lamella within the project."""
        return self._lamella_position

    @property
    def frame_position(self) -> int:
        """Return the index of the frame being viewed within the lamella."""
        return self._frame_position

    def select_project(self, position: int) -> None:
        """Select a project by index and move to its first lamella."""
        self._project_position = _clamp(position, len(self._projects))
        project = self.project
        self._lamellae = list(project.lamellae) if project is not None else []
        self.select_lamella(0)

    def select_lamella(self, position: int) -> None:
        """Select a lamella by index and rebuild its frame stack.

        Everything prepared for the previous lamella is discarded: at a lamella
        boundary none of it is worth keeping. The frame selection goes with it,
        since a frame index only means anything within one lamella.
        """
        self._lamella_position = _clamp(position, len(self._lamellae))
        lamella = self.lamella
        self._frames = build_lamella_frames(lamella) if lamella is not None else []
        self.cache.set_frames(self._frames)
        self._frame_position = 0
        self.selection.clear_frames()
        logger.debug(
            "Selected lamella %s with %d frames",
            lamella.name if lamella is not None else "(none)", len(self._frames),
        )

    def select_frame(self, position: int) -> None:
        """Move to a frame by index, clamped to the lamella's range."""
        self._frame_position = _clamp(position, len(self._frames))

    def step_frame(self, delta: int) -> None:
        """Move *delta* frames, stopping at the ends of the lamella."""
        self.select_frame(self._frame_position + delta)

    def step_lamella(self, delta: int) -> None:
        """Move *delta* lamellae, stopping at the ends of the project."""
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
        self.mode = mode
        if self.follow_mode:
            modality = MODE_MODALITIES.get(mode)
            if modality is not None:
                self.target = modality

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
        if frame.plot_path is not None:
            modes.append("plot")
        return modes

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
        return [lamella_status(lamella) for lamella in self._lamellae]

    # -- paths, for copying out of the window ------------------------------
    #
    # Absolute paths throughout. The relative path is a project's identity
    # within an index, but a path pasted into a terminal or a file manager has
    # to be absolute to be any use, and that is the whole point of copying one.

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
            self._lamellae, self._lamella_position if position is None else position
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
        if frame is None or frame.plot_path is None:
            return None
        return frame.plot_path.resolve()

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
            # A plot takes ~200 ms to decode, which is slower than a held-down
            # arrow key. Preparing only the immediate neighbours guarantees the
            # user outruns the loader, so in the plot view the whole lamella is
            # prepared instead, working outwards from where they are.
            radius=PLOT_RADIUS if showing_plots else radius,
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

    def mismatched_frames(self) -> set[int]:
        """Return frames whose images disagree with the label they share.

        The timeline draws the base image's state, so a frame with a
        disagreement is showing something that is not the whole truth. Marking
        it is all the GUI does: repair happens through ``am-seg repair-state``,
        which previews the moves before making them.
        """
        mismatched: set[int] = set()
        for position, frame in enumerate(self._frames):
            records = frame.records
            if not records:
                continue
            if any(not group.is_consistent for group in build_groups(records)):
                mismatched.add(position)
        return mismatched

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
        lamella = self._at(self._lamellae, position)
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
            [selection_key(lamella.path) for lamella in self._lamellae],
            checked,
        )

    def project_checks(self) -> list[bool]:
        """Return whether each project is checked, in list order."""
        return [
            self.selection.is_checked(LEVEL_PROJECTS, selection_key(project.path))
            for project in self._projects
        ]

    def lamella_checks(self) -> list[bool]:
        """Return whether each lamella of the selected project is checked."""
        return [
            self.selection.is_checked(LEVEL_LAMELLAE, selection_key(lamella.path))
            for lamella in self._lamellae
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

        The finest level the user has picked out wins, and nothing coarser is
        consulted -- see :mod:`.selection` for why that order and not the other.
        Everything downstream (:meth:`expanded_records`, :meth:`reach`,
        :meth:`apply_state`) reads this one method, so label-group expansion and
        the corrected-data guard apply to a checked selection unchanged.

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
        """Return the lamellae the selection covers, across every project.

        Checks persist by path, so a lamella checked before the project was
        changed is still checked -- and still has to be found, which is why this
        searches every project rather than only the one on screen.
        """
        if self.selection.lamellae:
            wanted = self.selection.lamellae
            return [
                lamella
                for project in self._projects
                for lamella in project.lamellae
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
        """Return every record of *lamellae* belonging to the edit target."""
        modalities = ("SEM", "FIB") if self.target == "both" else (self.target,)
        return [
            record
            for lamella in lamellae
            for stage in lamella.stages
            for record in stage.images
            if record.modality in modalities
        ]

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
        requested = {str(base_path(record.image_path)) for record in candidates}
        return [
            record for record in self.expanded_records(candidates)
            if str(base_path(record.image_path)) not in requested
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
        shown cannot drift from the number of files moved.
        """
        scoped = self.scoped_records()
        if not scoped:
            return StateChangeReach(images=0, labels=0, shared=0)
        expanded = self.expanded_records(scoped)
        labels = {
            str(base_path(label))
            for record in expanded
            for label in record.labels
        }
        return StateChangeReach(
            images=len(expanded),
            labels=len(labels),
            shared=len(expanded) - len(scoped),
        )

    def apply_state(self, state: RecordState, *, confirm_corrected: bool = False) -> int:
        """Move the current frame's targeted records into *state*.

        Delegates to :func:`transition_records`, which validates the whole set
        before moving anything and refuses to move corrected data unless
        *confirm_corrected* is set. Both behaviours are deliberately left
        intact: this method adds the editing guard and nothing else.

        Args:
            state: The workflow state to move into.
            confirm_corrected: Pass on the user's explicit confirmation to move
                data out of the corrected state.

        Returns:
            The number of records moved.

        Raises:
            EditingDisabled: If editing has not been switched on.
            PermissionError: If corrected data would be moved without
                confirmation.
            FileExistsError, FileNotFoundError: Propagated unchanged so the
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
            records, state, allow_move_from_corrected=confirm_corrected
        )
        logger.info(
            "Moved %d record(s) to %s across %s",
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
