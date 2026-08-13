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
from pathlib import Path
from typing import Callable, Literal, Sequence

from ..config import SegmentationConfig, load_segmentation_config
from ..frames import LamellaFrame, build_lamella_frames
from ..models import Lamella, Project
from ..operations.selection import RecordState, current_state, transition_records
from ..scanner import ProjectIndex, Scanner
from .cache import DEFAULT_RADIUS, FrameCache, FrameImage, FramePlot, LoadRequest

logger = logging.getLogger(__name__)

#: Which view of a frame is on screen.
DisplayMode = Literal["sem", "fib", "plot"]

#: What a state change applies to.
EditTarget = Literal["SEM", "FIB", "both"]

#: How far a state change reaches. "frame" is the default: the wider scopes
#: move many files at once and the window confirms them before proceeding.
EditScope = Literal["frame", "lamella", "project"]

#: The state of a frame whose two acquisitions disagree.
MIXED = "mixed"

MODE_MODALITIES: dict[DisplayMode, str] = {"sem": "SEM", "fib": "FIB"}

#: Scans a directory and returns the projects found in it.
ProjectScanner = Callable[[Path], ProjectIndex]


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

        self._project_position = 0
        self._lamella_position = 0
        self._frame_position = 0

        self.mode: DisplayMode = "sem"
        self.show_overlay = True
        self.editing_enabled = False
        # Starts matching the initial view, for the same reason the target
        # follows the view thereafter: the modality on screen is the one a
        # state change is meant for.
        self.target: EditTarget = "SEM"
        self.scope: EditScope = "frame"
        self.follow_mode = True

    # -- scanning ----------------------------------------------------------

    def scan(self, start_path: Path) -> list[Project]:
        """Discover projects under *start_path* and select the first one.

        Returns:
            The projects found, in scan order.
        """
        self.start_path = Path(start_path)
        self.index = self._scan(self.start_path)
        self._projects = self.index.all_projects()
        logger.info("Found %d project(s) under %s", len(self._projects), self.start_path)
        self.select_project(0)
        return self._projects

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
        boundary none of it is worth keeping.
        """
        self._lamella_position = _clamp(position, len(self._lamellae))
        lamella = self.lamella
        self._frames = build_lamella_frames(lamella) if lamella is not None else []
        self.cache.set_frames(self._frames)
        self._frame_position = 0
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
        if frame.sem is not None:
            modes.append("sem")
        if frame.fib is not None:
            modes.append("fib")
        if frame.plot_path is not None:
            modes.append("plot")
        return modes

    def current_image(self) -> FrameImage | None:
        """Return the prepared acquisition for the current mode and frame.

        Returns None in plot mode, or when the frame has no such acquisition.
        """
        modality = MODE_MODALITIES.get(self.mode)
        if modality is None:
            return None
        return self.cache.image(self._frame_position, modality)

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
        return self.cache.pending(
            self._frame_position,
            [m for m in ordered if m is not None],
            radius=radius,
            include_plots=self.mode == "plot",
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
        """
        frame = frame if frame is not None else self.frame
        if frame is None:
            return None
        states = {current_state(record) for record in frame.records}
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

    def set_target(self, target: EditTarget) -> None:
        """Choose whether state changes apply to SEM, FIB, or both."""
        if target not in ("SEM", "FIB", "both"):
            raise ValueError(f"Unknown edit target: {target}")
        self.target = target

    def set_scope(self, scope: EditScope) -> None:
        """Choose whether a state change covers one frame, a lamella, or all."""
        if scope not in ("frame", "lamella", "project"):
            raise ValueError(f"Unknown edit scope: {scope}")
        self.scope = scope

    def target_records(self, frame: LamellaFrame | None = None) -> list:
        """Return the image records a state change would apply to in one frame."""
        frame = frame if frame is not None else self.frame
        if frame is None:
            return []
        if self.target == "both":
            return frame.records
        record = frame.record(self.target)
        return [record] if record is not None else []

    def scoped_records(self) -> list:
        """Return every record a state change would apply to, honouring scope.

        Wider scopes read the records straight from the scanned stages rather
        than building frames, since the frame grouping only matters for display.
        """
        if self.scope == "frame":
            return self.target_records()

        lamellae = [self.lamella] if self.scope == "lamella" else self._lamellae
        modalities = ("SEM", "FIB") if self.target == "both" else (self.target,)
        return [
            record
            for lamella in lamellae
            if lamella is not None
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
            logger.info("Nothing to change: no %s image in this %s", self.target, self.scope)
            return 0

        changed = transition_records(
            records, state, allow_move_from_corrected=confirm_corrected
        )
        logger.info("Moved %d record(s) to %s across this %s", changed, state, self.scope)
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
