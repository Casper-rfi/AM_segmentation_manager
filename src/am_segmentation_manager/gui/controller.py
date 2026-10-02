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
from typing import Final, Literal

from ..config import SegmentationConfig, load_segmentation_config
from ..frames import (
    ConditionChoice,
    LamellaFrame,
    build_lamella_frames,
    primary_among,
    resolve_series,
    series_present,
)
from ..history import Recorder
from ..layout import (
    LABELLED_SERIES,
    LABELS_EXPECTED,
    MODALITIES,
    NO_CORRECTIONS,
    SERIES,
    SERIES_LABELS,
    condition_of,
    family_key,
)
from ..models import ImageRecord, Lamella, Project, projects_holding
from ..operations import label_io
from ..operations.init_history import writing_recorder
from ..operations.label_groups import expand_to_groups, group_key
from ..operations.label_io import LabelReadiness, classify_labels
from ..operations.predict import prediction_destination
from ..operations.selection import RecordState, current_state, transition_records
from ..scanner import ProjectIndex, Scanner
from .attribution import Attribution
from .cache import (
    DEFAULT_RADIUS,
    FrameCache,
    FrameImage,
    FramePlot,
    LoadRequest,
)
from .grid_controller import GridController
from .known_labels import KnownLabels
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

#: Which view is on screen. The first three are views of the *frame*; ``grid``
#: is the project's own overview, which every frame of it shows alike.
DisplayMode = Literal["sem", "fib", "plot", "grid"]

#: What a state change applies to.
EditTarget = Literal["SEM", "FIB", "both"]

#: Whether a state change reaches the family on screen or every family of a
#: beam; see :meth:`CurationController.set_family_scope`.
FamilyScope = Literal["this", "every"]

#: The state of a frame whose two acquisitions disagree.
MIXED: Final = "mixed"

MODE_MODALITIES: dict[DisplayMode, Literal["SEM", "FIB"]] = {"sem": "SEM", "fib": "FIB"}

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


def _add_written(record, wanted: set[Path]) -> bool:
    """Name any of *wanted* that belongs to *record*, and say whether one did.

    Asked of the record rather than derived from its image path: where a
    record's label and predictions live depends on the shape of its image set,
    which is why the scanner resolves both once and the record carries them.
    A record that cannot say raises from `prediction_destination`, so it is
    asked only where the record knows.
    """
    added = False
    if record.predictions_directory is not None:
        prediction = prediction_destination(record)
        if prediction in wanted and prediction not in record.predictions:
            record.predictions.append(prediction)
            added = True
    if record.labels_directory is not None:
        label = label_io.label_destination(record)
        if label in wanted and label not in record.labels:
            record.labels.append(label)
            added = True
    return added


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

    Taken from the project's records rather than combined from its lamellae,
    which would lose which series each answer was about; a cell already applies
    the weakest-wins rule across the images inside it.
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
        shared_images: The images counted in *shared*, as ``directory/file``:
            conditions sharing a label often share a file name, so the
            directory is what tells them apart.
        family_choice: Whether the scope holds several families of a
            targeted beam, and so whether there is a family to choose.
        families: The families on screen the change is confined to, named
            only where there was that choice.
        every_family: Whether there was that choice and the change reaches
            every family.
    """

    images: int
    labels: int
    shared: int
    lamellae: int
    modalities: tuple[tuple[str, int], ...] = ()
    shared_images: tuple[str, ...] = ()
    family_choice: bool = False
    families: tuple[str, ...] = ()
    every_family: bool = False

    def summary(self) -> str:
        """Return the description shown above the state controls."""
        if not self.images:
            return "Nothing selected"
        parts = ", ".join([
            self._image_part(),
            counted(self.labels, "label", "labels") if self.labels else "no label yet",
            counted(self.lamellae, "lamella", "lamellae"),
        ])
        family = self._family_part()
        return f"{family}: covers {parts}" if family else f"Covers {parts}"

    def _family_part(self) -> str:
        """Return which families the change reaches, or empty where there was no choice."""
        if self.every_family:
            return "Every family"
        if len(self.families) == 1:
            return f"{self.families[0]} family"
        if self.families:
            return " + ".join(self.families) + " families"
        return ""

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
        attribution: Attribution | None = None,
        known_labels: KnownLabels | None = None,
    ) -> None:
        """Create a controller.

        Args:
            cache: Where prepared display data is held. A default cache using
                the real display pipeline is created if not supplied.
            config: Class names and colours. Loaded from the default location
                if not supplied.
            scan: How to discover projects. Defaults to the package scanner;
                tests inject a stub to avoid touching the filesystem.
            attribution: Who changes are recorded as. The window hands the
                same one to every controller; a controller on its own
                resolves the author itself.
            known_labels: What each label holds. The window hands this
                controller's to its other views; one on its own keeps its own.
        """
        self.cache = cache if cache is not None else FrameCache()
        #: The project's grid overview and where its lamellae fall on it, held
        #: while that project is on screen. Kept up to date by
        #: :meth:`_settle_grid`, which every path that changes project reaches.
        self.grid = GridController()
        self.config = config if config is not None else load_segmentation_config()

        def scan_collection(
            path: Path, on_progress: ScanProgress | None = None
        ) -> ProjectIndex:
            return Scanner(path).build_index(on_progress)

        self._scan: ProjectScanner = scan or scan_collection

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
        #: The series the lists and the series menu cover; see
        #: :meth:`set_shown_series`. Every one until the window says otherwise.
        self._shown_series: tuple[str, ...] = SERIES
        #: The projects the lamella list is drawn from, kept so a change of
        #: :attr:`shown_series` can list them again.
        self._listed_projects: list[Project] = []
        #: Lamellae of those projects the list leaves out.
        self._hidden_lamellae = 0
        self._frames: list[LamellaFrame] = []

        #: What each label holds, shared with the window's other views. Cleared
        #: by :meth:`scan`, which is the user saying the filesystem has moved on.
        self.known_labels = known_labels if known_labels is not None else KnownLabels()

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
        #: Whether the label is drawn over the acquisition or over a flat ground.
        self.show_image = True
        #: Which imaging condition is on screen for each modality, by its
        #: directory's name. None means the frame's primary. Its *family* is
        #: what a state change acts on; which of the family's conditions is
        #: viewed never decides what moves, the change following the label.
        self.condition: dict[str, str | None] = {modality: None for modality in MODALITIES}
        #: Whether a state change reaches the family on screen or every one.
        self.family_scope: FamilyScope = "this"
        self.editing_enabled = False
        # Starts matching the initial view, for the same reason the target
        # follows the view thereafter: the modality on screen is the one a
        # state change is meant for.
        self.target: EditTarget = "SEM"
        self.follow_mode = True
        #: What a state change applies to; see :mod:`.selection`.
        self.selection = Selection()
        #: Who changes are recorded as, shared with every other view.
        self.attribution = attribution if attribution is not None else Attribution()

    @property
    def recorder(self) -> Recorder:
        """Where this view's changes are recorded.

        Shared with every other view, so a name adopted in Settings reaches
        this one without anybody remembering to pass it on.
        """
        return self.attribution.recorder

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
        self.known_labels.clear()
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
        status line, the path to copy -- is asking about what is on screen. A project holding no
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

        **A lamella holding only series switched off is left out**, and
        counted in :attr:`hidden_lamellae` -- see :meth:`set_shown_series`.
        """
        self._scope_folder = folder
        self._listed_projects = list(projects)
        was = self.lamella
        every = [
            LamellaEntry(project, lamella)
            for project in projects
            for lamella in project.lamellae
        ]
        self._entries = [entry for entry in every if self._is_listed(entry.lamella)]
        self._hidden_lamellae = len(every) - len(self._entries)
        self.selection.lamellae.clear()
        self.select_lamella(
            _index_of([entry.lamella for entry in self._entries],
                      was.path if was is not None else None) or 0
        )

    def _is_listed(self, lamella: Lamella) -> bool:
        """Return whether *lamella* belongs on the list under :attr:`shown_series`.

        One holding no images at all stays: switching a series off says
        nothing about it, and hiding it would leave it nowhere to be seen.
        """
        present = series_present(lamella)
        return not present or any(name in self._shown_series for name in present)

    @property
    def shown_series(self) -> tuple[str, ...]:
        """Return the series the lists and the series menu cover, in `SERIES` order."""
        return self._shown_series

    @property
    def hidden_lamellae(self) -> int:
        """Return how many lamellae of the listed projects the list leaves out."""
        return self._hidden_lamellae

    def set_shown_series(self, series: Iterable[str]) -> None:
        """Cover only *series*: in the lamella list, the chips and the series menu.

        **Hiding is never silent.** A lamella holding only series switched off
        leaves the list and is counted in :attr:`hidden_lamellae`, and a ticked
        project's hidden lamellae contribute nothing to a change -- what is
        ticked on screen has to be the whole of what a change touches. The
        grid overview belongs to the project, and stays.

        The list is drawn again, clearing its ticks for the reason
        :meth:`_list_lamellae` gives and staying on the lamella in view where
        it is still listed.

        Raises:
            ValueError: If *series* names nothing, or a series this package
                does not index.
        """
        wanted = set(series)
        unknown = wanted - set(SERIES)
        if unknown:
            raise ValueError(
                f"Unknown series: {sorted(unknown)}; expected some of {list(SERIES)}"
            )
        chosen = tuple(name for name in SERIES if name in wanted)
        if not chosen:
            raise ValueError("At least one series must be shown")
        if chosen == self._shown_series:
            return
        self._shown_series = chosen
        self._list_lamellae(self._listed_projects, folder=self._scope_folder)

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
        # Before the mode settles, which asks whether the grid view has
        # anything to show. Cheap where the project has not changed.
        self.grid.set_project(self.project)
        self.series = resolve_series(self.wanted_series, self.available_series())
        self._frames = (
            build_lamella_frames(
                lamella, series=self.series
            )
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
        """Switch between the SEM, FIB, plot and grid views.

        Unless :attr:`follow_mode` is off, switching to a single-modality view
        also points the edit target at that modality. Changing the state of the
        image you are *not* looking at is far more likely to be a mistake than
        an intention, so the target follows the eye. The plot and grid views
        leave the target alone: one shows both modalities' context at once and
        the other is a picture of the whole grid.
        """
        if mode not in ("sem", "fib", "plot", "grid"):
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

        The plot and grid views are left alone: each says what is available for
        itself, and neither a figure nor the grid overview is an image of
        either beam.
        """
        if self.wanted_mode in ("plot", "grid"):
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
        """Return the views that have something to show.

        A frame may have only one modality, or no plot, and the window marks
        those views unavailable rather than showing an empty panel. ``grid`` is
        asked of the *project* rather than the frame -- an overview covers the
        whole grid, and every lamella on it shows the same picture -- so it
        stays available while stepping through a timeline, and is absent for
        the projects that have no overview at all.
        """
        frame = frame if frame is not None else self.frame
        modes: list[DisplayMode] = []
        if frame is not None:
            if frame.sem:
                modes.append("sem")
            if frame.fib:
                modes.append("fib")
            if frame.displayed_plot is not None:
                modes.append("plot")
        if self.grid.current is not None:
            modes.append("grid")
        return modes

    def available_series(self) -> list[str]:
        """Return the series the current lamella actually holds, in order.

        A pre-Autolamella lamella has only Adaptive Polishing images and gets
        no choice to make -- but the control still shows what it is on, rather
        than disappearing. Only series switched on are offered; see
        :meth:`set_shown_series`.
        """
        return [
            name for name in series_present(self.lamella) if name in self._shown_series
        ]

    def set_series(self, series: str) -> None:
        """Show one series' timeline, rebuilding the frame stack.

        More than a view choice: it decides which
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

    def condition_choices(self, modality: str | None = None) -> list[ConditionChoice]:
        """Return each condition the current frame has, of *modality* or the beam on screen.

        In condition order. A list of one -- the usual case -- means there is
        nothing to choose between, and the window greys the control.
        """
        frame = self.frame
        target = modality or MODE_MODALITIES.get(self.mode)
        if frame is None or target is None:
            return []
        return frame.condition_choices(target)

    def set_condition(self, condition: str | None, modality: str | None = None) -> None:
        """Choose which imaging condition to display for a modality, by name.

        Choosing one of another family also chooses what a state change acts
        on, a family being one label; between the conditions of one family it
        is a view choice only.
        """
        target = modality or MODE_MODALITIES.get(self.mode)
        if target is None:
            return
        self.condition[target] = condition
        logger.debug("Viewing %s condition %s", target, condition or "(primary)")

    def current_condition(self, modality: str | None = None) -> str | None:
        """Return the imaging condition on screen for a modality, as its frame spells it.

        Where the frame was not captured under the one chosen, what
        :meth:`~..frames.LamellaFrame.shown_record` falls back to.
        """
        record = self._shown_record(modality or MODE_MODALITIES.get(self.mode))
        return record.condition if record is not None else None

    def _shown_record(self, modality: str | None) -> ImageRecord | None:
        """Return the record on screen for *modality*, as the cache chooses it."""
        frame = self.frame
        if frame is None or modality is None:
            return None
        return frame.shown_record(modality, self.condition.get(modality))

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

        Follows the mode and the condition dropdown, so this is the file the user
        is actually looking at rather than the frame's primary. Returns None
        in plot mode, which has no single acquisition behind it.
        """
        modality = MODE_MODALITIES.get(self.mode)
        frame = self.frame
        if modality is None or frame is None:
            return None
        record = self._shown_record(modality)
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
        return self.cache.image(self._frame_position, modality, self.condition.get(modality))

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
            conditions=dict(self.condition),
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
        """Return the state of the image a view shows for one modality, or None if absent.

        The family on screen, where the frame has it, and whatever the frame
        would show in its place where it does not: the timeline previews what
        stepping onto a frame shows.
        """
        record = self._shown_in(frame if frame is not None else self.frame, modality)
        return current_state(record) if record is not None else None

    def _shown_in(self, frame: LamellaFrame | None, modality: str) -> ImageRecord | None:
        """Return the record a view shows for *modality* in *frame*."""
        if frame is None:
            return None
        return frame.shown_record(modality, self.condition.get(modality))

    def combined_state(
        self, frame: LamellaFrame | None = None
    ) -> RecordState | Literal["mixed"] | None:
        """Return a frame's state across both modalities.

        Returns :data:`MIXED` when the two acquisitions are in different states,
        which the window shows as no selection rather than picking a side. This
        is a normal situation: curation deselects one modality at a time.

        Read from the image a view shows for each modality, as
        :meth:`state_of` is.
        """
        frame = frame if frame is not None else self.frame
        return _combined(
            record
            for record in (self._shown_in(frame, modality) for modality in MODALITIES)
            if record is not None
        )

    def target_state(self) -> RecordState | Literal["mixed"] | None:
        """Return the state the current frame's share of a change is in.

        Over the images a change on this frame would declare, so under *Every
        family* two families in different states read as :data:`MIXED`.
        """
        frame = self.frame
        if frame is None:
            return None
        return _combined(self._narrowed(self._frame_candidates(frame)))

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
                self.known_labels.drawn(self._shown_in(frame, modality)) for frame in self._frames
            ]
            for modality in MODALITIES
        }

    def set_target(self, target: EditTarget) -> None:
        """Choose whether state changes apply to SEM, FIB, or both."""
        if target not in ("SEM", "FIB", "both"):
            raise ValueError(f"Unknown edit target: {target}")
        self.target = target

    def set_family_scope(self, scope: FamilyScope) -> None:
        """Choose whether state changes reach the family on screen or every family.

        Consulted only where the scope holds several families of a targeted
        beam: with one there is nothing to choose between.
        """
        if scope not in ("this", "every"):
            raise ValueError(f"Unknown family scope: {scope}")
        self.family_scope = scope

    def viewed_family(self, modality: str) -> str | None:
        """Return the family on screen for *modality*, as its records spell it.

        The chosen condition's where the frame on screen has no image of the
        beam, and None where nothing says which family is meant.
        """
        record = self._shown_record(modality)
        if record is not None:
            return record.family or None
        chosen = self.condition.get(modality)
        lamella = self.lamella
        corrections = lamella.families if lamella is not None else NO_CORRECTIONS
        parsed = condition_of(chosen, corrections) if chosen else None
        return parsed.family if parsed is not None else None

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

    def _targeted(self) -> tuple[str, ...]:
        """Return the modalities the edit target names."""
        return MODALITIES if self.target == "both" else (self.target,)

    def _frame_candidates(self, frame: LamellaFrame) -> list[ImageRecord]:
        """Return the primary of each label group, for each targeted beam in *frame*.

        One per label rather than every condition: the others are reached
        through the label they share, which is what lets the reach line say
        how many were pulled in that way, and keeps which of a family's
        conditions is on screen from changing what a change covers. A family
        is one label except while its labels are split across directories,
        when a change reaches each of them
        (``test_a_state_change_reaches_every_label_of_a_family_in_conflict``).
        """
        candidates: list[ImageRecord] = []
        for modality in self._targeted():
            groups: dict[tuple[str, str], list[ImageRecord]] = {}
            for record in frame.modality_records(modality):
                key = (family_key(record.family), group_key(record))
                groups.setdefault(key, []).append(record)
            candidates.extend(
                primary_among(members, frame.corrections) for members in groups.values()
            )
        return candidates

    def _narrowed(self, records: list) -> list:
        """Return *records* confined to the family on screen, where that is asked.

        Only beams holding several families are narrowed, and each to the
        family on screen by name, so a frame without it contributes nothing
        of that beam. Deciding one family never decides another: they are
        separate segmentations. An image of no family -- a reference or
        alignment image, which no directory names -- is never narrowed away,
        a ticked lamella holding only those contributing them as it did.
        """
        if self.family_scope == "every":
            return records
        several = _beams_with_several_families(records)
        if not several:
            return records
        wanted = {modality: self.viewed_family(modality) for modality in several}
        return [
            record
            for record in records
            if record.modality not in several
            or not record.family
            or (
                wanted[record.modality] is not None
                and family_key(record.family) == family_key(wanted[record.modality])
            )
        ]

    def attach_written(self, written: Iterable[Path]) -> int:
        """Name newly written predictions and labels on the records they belong to.

        What a scan after predicting or converting would have discovered, done
        from what the run reported instead. Both writes change exactly two
        fields of :class:`~...models.ImageRecord` -- ``predictions`` and
        ``labels`` -- and everything drawn from them (the timeline's label line,
        the status chips, what the Export tab counts) is computed from the
        records themselves, which the other tabs hold by reference rather than
        copying. So there is nothing left for a walk of the collection to find.

        A path is matched against where each record says its own prediction and
        label go, so a label shared by several imaging conditions is named on
        all of them: the group is what a conversion writes for, and a condition
        left without it would keep drawing as unlabelled.

        **Two memoised layers sit between a record and the screen, and both are
        dropped here.** :attr:`known_labels` opens a label once and remembers
        the answer, and the frame cache holds each frame as a composed image
        *and its label*, keyed by position rather than by what it was made
        from. A record named without clearing those keeps drawing exactly as it
        did before the file was written.

        Returns:
            How many records gained a path. Zero is normal for a preview.
        """
        wanted = {Path(path) for path in written}
        if not wanted:
            return 0
        gained = [
            record
            for project in self.projects
            for record in project.image_records
            if _add_written(record, wanted)
        ]
        # A replaced label is forgotten as well as a new one: its record already
        # named it, so only the path says that what it holds has changed.
        self.known_labels.forget(labels=wanted, records=gained)
        touched = len(gained)
        if touched:
            # Cleared whole rather than per frame: the cache is keyed by frame
            # position, so finding the entries made from these records means
            # walking the frames anyway, and what it costs is re-reading one
            # lamella lazily.
            self.cache.clear()
        logger.info(
            "Named %d written file(s) on %d record(s) without rescanning",
            len(wanted), touched,
        )
        return touched

    @property
    def lamella_holds_curated_images(self) -> bool:
        """Return whether the lamella on screen holds a series anyone curates.

        False for a directory holding nothing but ``other`` -- a folder of
        overview tiles is indexed as a lamella, because it sits below the
        project and holds images. Nothing about such a directory is a lamella
        Autolamella milled, so nothing may be claimed about the stage position
        it was never going to have.
        """
        lamella = self.lamella
        if lamella is None:
            return False
        return any(
            record.series in LABELLED_SERIES for record in lamella.image_records
        )

    @property
    def series_is_editable(self) -> bool:
        """Return whether the series on screen is one an operation may act on.

        The window's half of the rule the command line states as
        :data:`~..layout.LABELLED_SERIES`: an ``other`` record has no label
        directory and no predictions directory, so every operation could only
        refuse it. This is asked of the series actually being shown, not of the
        one the reviewer chose -- :func:`~..frames.resolve_series` lands on
        ``other`` by itself for a lamella that holds nothing else.
        """
        return self.series in LABELLED_SERIES

    def scoped_records(self) -> list:
        """Return every record a state change would apply to.

        What the scope holds, confined to the family on screen unless every
        family was asked for; see :meth:`set_family_scope`.

        Only one level is ever picked out at a time -- see :mod:`.selection` --
        so this reads whichever one holds anything. Everything downstream
        (:meth:`expanded_records`, :meth:`reach`, :meth:`apply_state`) reads
        this one method, so label-group expansion and the corrected-data guard
        apply to a checked selection unchanged.

        Whole lamellae are read straight from the scanned stages rather than
        through frames, since the chronological pairing only matters for display.

        **A series no operation can act on contributes nothing**, whichever
        level is picked out. The window disables those controls as well, and
        this is the backstop that holds however a change was triggered: it is
        the one method every operation reads, so filtering here covers all of
        them rather than each guarding itself. Ticked lamellae need it even
        when the tab is showing a labelled series, one that holds no other
        resolving to ``other`` on its own.
        """
        return self._narrowed(self._candidate_records())

    def _candidate_records(self) -> list:
        """Return what the scope holds of the edit target, in every family."""
        level = self.selection.level
        if level == LEVEL_CURRENT_FRAME:
            frame = self.frame
            chosen = self._frame_candidates(frame) if frame is not None else []
        elif level in (LEVEL_PROJECTS, LEVEL_LAMELLAE):
            chosen = self._records_of(self._checked_lamellae())
        else:
            chosen = [
                record
                for position in sorted(self.selection.frames)
                if 0 <= position < len(self._frames)
                for record in self._frame_candidates(self._frames[position])
            ]
        return [record for record in chosen if record.series in LABELLED_SERIES]

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
        than nothing at all. One holding only series switched off contributes
        nothing, not being on the list -- see :meth:`set_shown_series`.
        """
        modalities = self._targeted()
        records: list = []
        for lamella in lamellae:
            present = [
                name for name in series_present(lamella) if name in self._shown_series
            ]
            if not present:
                continue
            series = resolve_series(self.wanted_series, present)
            records.extend(
                record
                for record in lamella.image_records
                if record.modality in modalities and record.series == series
            )
        return records

    def _held_records(self) -> list[ImageRecord]:
        """Return every record of the lamellae a state change can reach.

        The scope holds one record per label group; the conditions sharing it
        are these, and a change must update them rather than copies of them
        (``test_a_state_change_shows_on_every_condition_without_a_rescan``).
        """
        if self.selection.level in (LEVEL_PROJECTS, LEVEL_LAMELLAE):
            lamellae = self._checked_lamellae()
        else:
            lamellae = [self.lamella] if self.lamella is not None else []
        return [record for lamella in lamellae for record in lamella.image_records]

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
        :attr:`known_labels`, so the timeline has usually paid for it
        already.
        """
        candidates = records if records is not None else self.scoped_records()
        if not candidates:
            return LabelReadiness()
        return classify_labels(
            self.expanded_records(candidates), self.known_labels.state
        )

    def reach(self) -> StateChangeReach:
        """Return how far a state change would go, for display before it is made.

        Counted from the same expansion the operation performs, so the number
        shown cannot drift from the number of records declared.
        """
        candidates = self._candidate_records()
        scoped = self._narrowed(candidates)
        several = _beams_with_several_families(candidates)
        if not scoped:
            return StateChangeReach(
                images=0, labels=0, shared=0, lamellae=0, family_choice=bool(several)
            )
        confined = bool(several) and self.family_scope == "this"
        families = (
            tuple(dict.fromkeys(
                family
                for modality in MODALITIES
                if modality in several and (family := self.viewed_family(modality))
            ))
            if confined else ()
        )
        expanded = self.expanded_records(scoped)
        requested = {str(record.image_path) for record in scoped}
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
            shared_images=tuple(
                f"{record.image_path.parent.name}/{record.image_path.name}"
                for record in expanded
                if str(record.image_path) not in requested
            ),
            family_choice=bool(several),
            families=families,
            every_family=bool(several) and self.family_scope == "every",
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

        batch = writing_recorder(
            projects_holding((record.image_path for record in records), self._projects),
            self.recorder,
        )
        changed = transition_records(
            records, state, allow_leaving_corrected=confirm_corrected, recorder=batch,
            known=self._held_records(),
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


def _combined(records: Iterable[ImageRecord]) -> RecordState | Literal["mixed"] | None:
    """Return the state *records* share, :data:`MIXED` where they differ, None if none."""
    states = {current_state(record) for record in records}
    if not states:
        return None
    if len(states) > 1:
        return MIXED
    return states.pop()


def _beams_with_several_families(records: Iterable[ImageRecord]) -> set[str]:
    """Return the modalities whose images among *records* span more than one family.

    Images of no family are not counted as one: there is nothing to choose
    them by.
    """
    families: dict[str, set[str]] = {}
    for record in records:
        if record.modality is not None and record.family:
            families.setdefault(record.modality, set()).add(family_key(record.family))
    return {modality for modality, names in families.items() if len(names) > 1}
