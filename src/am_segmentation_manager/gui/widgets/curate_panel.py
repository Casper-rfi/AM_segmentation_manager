"""The Curate tab: the project lists, the frame on screen, its timeline, and the column that changes it.

Built around :class:`~am_segmentation_manager.gui.controller.CurationController`, as
each other tab is around its own controller. **What the tab shows follows the
controller**: every move ends in :meth:`CuratePanel.refresh`, which redraws from it.
What reaches beyond the tab -- copying a path, the remembered model, the series the
Label tab covers, scanning -- leaves as a signal for the window to act on.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from ..controller import MODE_MODALITIES, CurationController, DisplayMode
from ..families_controller import FamiliesController
from ..loader import BackgroundLoader
from ..operations_controller import OperationsController
from .controls import GripSplitter
from .families_panel import FamiliesPanel
from .frame_strip import CURATE_BAND, FrameTimeline
from .image_panel import ImagePanel
from .operations_panel import OperationsPanel
from .project_panel import ProjectPanel
from .state_panel import StatePanel

logger = logging.getLogger(__name__)


class CuratePanel(QWidget):
    """The lists, the frame on screen with its timeline, and the state, families and operations."""

    #: Emitted with a path to copy and what it is: "Project", "Lamella" or "Image".
    path_requested = Signal(object, str)
    #: Emitted with a path whose folder to open.
    reveal_requested = Signal(object)
    #: Emitted on every redraw, so the window can point the model box at the
    #: series and beam now shown.
    view_changed = Signal()
    #: Emitted with the series the lists now cover, which the Label tab covers too.
    series_shown = Signal(tuple)

    def __init__(
        self,
        controller: CurationController,
        families_controller: FamiliesController,
        operations_controller: OperationsController,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.families_controller = families_controller
        self.operations_controller = operations_controller
        #: What the scan box holds, shown where nothing has been read yet; the
        #: window's to set, the box being in its corner.
        self.typed_folder: Callable[[], str] = lambda: ""
        self.project_panel = ProjectPanel(checkable=not controller.read_only)
        self.image_panel = ImagePanel(controller.config)
        self.state_panel = StatePanel()
        self.operations_panel = OperationsPanel()
        self.families_panel = FamiliesPanel(families_controller)
        #: The project and lamella the Families tab was last given, by identity:
        #: a rescan makes new ones, and nothing else needs the tree rebuilt.
        self._families_shown: tuple[object, object] = (None, None)
        self._build_layout()
        self._connect()
        self.loader = BackgroundLoader(controller.cache)
        self.loader.worker.loaded.connect(self._on_loaded)

    # -- construction ------------------------------------------------------

    def _build_layout(self) -> None:
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        # The bottom margin has to clear the timeline's FIB label line, which is
        # the widget's bottom-most 3px by design: the SEM band is
        # label-line-then-row and the FIB band is row-then-label-line, so the two
        # label lines sit on the outside edges. Two pixels left that line almost
        # touching the window edge and hard to see.
        centre_layout.setContentsMargins(0, 0, 0, 8)
        centre_layout.setSpacing(3)
        centre_layout.addWidget(self.image_panel, stretch=1)
        centre_layout.addLayout(self._build_timeline())

        # The state controls take the top of the right column and the operations
        # the bottom, with the free space between them: both act on the same
        # selection, and separating them puts the buttons that change files
        # where nothing else is going to be clicked by accident. The families
        # share the top, a tab apart, under the one switch that unlocks both;
        # the operations stay in view under either, predicting being what a
        # family's primary is chosen for.
        self.side_tabs = QTabBar()
        self.side_tabs.addTab("State")
        self.side_tabs.addTab("Families")
        self.side_tabs.setToolTip(
            "State: what to keep and what was proofread. Families: which imaging "
            "conditions share a label, and which one is predicted from."
        )
        self.side_pages = QStackedWidget()
        self.side_pages.addWidget(self.state_panel)
        self.side_pages.addWidget(self.families_panel)
        self.side_tabs.currentChanged.connect(self.side_pages.setCurrentIndex)
        guard = QWidget()
        guard_row = QHBoxLayout(guard)
        panel_layout = self.state_panel.layout()
        assert panel_layout is not None, "the state panel lays itself out"
        margins = panel_layout.contentsMargins()
        guard_row.setContentsMargins(margins.left(), margins.top(), margins.right(), 0)
        guard_row.addWidget(self.state_panel.enable_check)
        right = QWidget()
        right_column = QVBoxLayout(right)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.addWidget(self.side_tabs)
        right_column.addWidget(guard)
        right_column.addWidget(self.side_pages, stretch=1)
        right_column.addWidget(self.operations_panel)

        self.columns = GripSplitter(Qt.Orientation.Horizontal)
        self.columns.addWidget(self.project_panel)
        self.columns.addWidget(centre)
        # A viewer has nothing to change and nothing to run, which is all the
        # right column holds; the timeline already shows each frame's state.
        # Built all the same and held hidden, so nothing that refreshes it has
        # to know about a viewer.
        if self.controller.read_only:
            right.setParent(self)
            right.hide()
        else:
            self.columns.addWidget(right)
        self.columns.setStretchFactor(1, 1)
        # The left column opens wide enough for the project box's four
        # buttons to sit on one line; narrower and they wrap, which is the
        # reader's call rather than the default.
        self.columns.setSizes([330, 1050, 260][: self.columns.count()])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.columns)

    def _build_timeline(self) -> QHBoxLayout:
        """Build the frame timeline row.

        One widget carries both the position and the per-modality curation
        states, so the two cannot drift out of alignment. There is no lamella
        slider: the Lamellae list and Up/Down cover that, and the slider only
        took space from the images.
        """
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(6)

        self.timeline = FrameTimeline(band=None if self.controller.read_only else CURATE_BAND)
        self.timeline.frame_selected.connect(self.seek_frame)
        row.addWidget(self.timeline, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(70)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.frame_label.setToolTip("Frame position within this lamella")
        row.addWidget(self.frame_label)

        return row

    def _connect(self) -> None:
        self.project_panel.project_selected.connect(self._select_project)
        self.project_panel.folder_selected.connect(self._select_folder)
        self.project_panel.lamella_selected.connect(self._select_lamella)
        self.project_panel.project_path_requested.connect(
            lambda row: self.path_requested.emit(self.controller.project_path(row), "Project")
        )
        self.project_panel.lamella_path_requested.connect(
            lambda row: self.path_requested.emit(self.controller.lamella_path(row), "Lamella")
        )
        self.project_panel.project_reveal_requested.connect(
            lambda row: self.reveal_requested.emit(self.controller.project_path(row))
        )
        self.project_panel.lamella_reveal_requested.connect(
            lambda row: self.reveal_requested.emit(self.controller.lamella_path(row))
        )
        self.project_panel.shown_series_changed.connect(self._show_series)

        self.image_panel.mode_changed.connect(self.set_mode)
        self.image_panel.condition_changed.connect(self._set_condition)
        self.image_panel.overview_chosen.connect(self._choose_overview)
        self.image_panel.series_changed.connect(self._set_series)
        self.image_panel.overlay_toggled.connect(self._set_overlay)
        self.image_panel.image_toggled.connect(self._set_image_shown)
        self.image_panel.alpha_changed.connect(lambda _alpha: self._render())
        self.image_panel.intensity_range_changed.connect(
            lambda _low, _high: self._render()
        )
        self.image_panel.path_requested.connect(
            lambda: self.path_requested.emit(self._displayed_path(), "Image")
        )
        self.image_panel.reveal_requested.connect(
            lambda: self.reveal_requested.emit(self._displayed_path())
        )

        self.state_panel.editing_toggled.connect(self.controller.set_editing_enabled)
        self.state_panel.editing_toggled.connect(self.families_panel.set_editing_enabled)
        self.operations_panel.origin_changed.connect(
            lambda _source: self._refresh_prediction_summary()
        )
        self.families_panel.condition_chosen.connect(self._show_family_condition)

    def take_focus(self) -> None:
        """Put the keyboard on the project list, not the scan box.

        The scan box is the first focusable widget in the tab, and a focused
        ``QLineEdit`` claims Left and Right as shortcut overrides -- so the
        keys that step through frames never reach the window while it holds
        focus, and the tab opens unable to be driven by the keys it documents.
        The project list claims no arrow key of its own, so the window's own
        bindings all still fire with it focused.
        """
        self.project_panel.project_list.setFocus()

    def came_forward(self) -> None:
        """Take the keyboard, and redraw only if the label on screen was rewritten while away."""
        self.take_focus()
        if self.controller.drop_rewritten_frame():
            self._schedule_loads()
            self._render()

    # -- moving through the collection ---------------------------------------

    def _select_folder(self, folder: str) -> None:
        """List every lamella under a folder, and say what is in it.

        A session's lamellae are spread over however many projects Autolamella
        happened to start, most of them holding one; choosing the folder puts
        them in one list to be worked through. Nothing is moved that does not
        have to be: the lamella in view stays if it is under the folder, which
        it is whenever the folder being chosen is the one already being worked
        in.

        **The project list is not rebuilt here**, for the reason
        :meth:`_select_project` gives -- this runs from that list's own signal.
        """
        self.controller.select_folder(folder)
        self._refresh_lamellae()
        self.refresh()

    def _select_project(self, row: int) -> None:
        """Move to a project, rebuilding its lamellae and nothing else.

        **The project list must not be rebuilt here.** This runs from that
        list's own ``currentRowChanged``, delivered while Qt is still part-way
        through the mouse press that caused it -- and rebuilding a list from
        inside its own signal empties the rows Qt is holding a position in. It
        also threw the reviewer back to the top of the list on every click,
        which on an archive of a hundred projects loses their place entirely.
        """
        self.controller.select_project(row)
        self._refresh_lamellae()
        self.refresh()

    def _select_lamella(self, row: int) -> None:
        self.controller.select_lamella(row)
        self.refresh()

    def _show_series(self, series: tuple) -> None:
        """Cover only *series*, in both lists and every chip, and say so.

        Nothing is redrawn before the first scan: there is nothing listed, and
        the switches first announce themselves while the window is still being
        built.
        """
        self.controller.set_shown_series(series)
        self.project_panel.set_chip_series(series)
        if self.controller.projects:
            self.refresh(reload_lists=True)
        self.series_shown.emit(series)

    def seek_frame(self, value: int) -> None:
        if value != self.controller.frame_position:
            self.controller.select_frame(value)
            self.refresh()

    def step_frame(self, delta: int) -> None:
        self.controller.step_frame(delta)
        self.refresh()

    def step_lamella(self, delta: int) -> None:
        """Move within the project on screen, which the project list outlives."""
        self.controller.step_lamella(delta)
        self._refresh_lamellae()
        self.refresh()

    def set_mode(self, mode: DisplayMode, from_user: bool = False) -> None:
        if from_user and mode not in self.controller.available_modes():
            return
        self.controller.set_mode(mode)
        self.image_panel.set_mode(mode)
        self.refresh()

    def _set_overlay(self, show: bool) -> None:
        self.controller.show_overlay = show
        self._render()

    def _set_image_shown(self, show: bool) -> None:
        self.controller.show_image = show
        self._render()

    def toggle_overlay(self) -> None:
        self.image_panel.overlay_check.toggle()

    def _set_series(self, series: str) -> None:
        """Show a different series' timeline.

        More than a view change: it decides which frames exist, so the whole
        window is brought back in step rather than only redrawn.
        """
        self.controller.set_series(series)
        self.refresh()

    def _set_condition(self, condition: str) -> None:
        """Show a different imaging condition.

        One of another family also moves what a state change acts on, and the
        timeline's chips with it; see :meth:`CurationController.set_condition`.
        """
        self.controller.set_condition(condition or None)
        self.refresh()

    def _show_family_condition(self, modality: str, condition: str) -> None:
        """Show a condition chosen in the Families tab, on its own beam."""
        if MODE_MODALITIES.get(self.controller.mode) != modality:
            self.set_mode("sem" if modality == "SEM" else "fib")
        self.controller.set_condition(condition, modality)
        self.refresh()

    def _choose_overview(self, index: int) -> None:
        """Show another of the project's grid overviews."""
        self.controller.grid.choose(index)
        self._render()

    def _displayed_path(self) -> Path | None:
        """Return the file currently on screen, image or plot."""
        if self.controller.mode == "plot":
            return self.controller.current_plot_path()
        return self.controller.current_image_path()

    # -- drawing ----------------------------------------------------------

    def refresh(self, *, reload_lists: bool = False) -> None:
        """Bring the tab back in step with the controller."""
        if reload_lists:
            self._refresh_lists()
        # Outside the reload: the summary describes the lamella being looked
        # at, so it has to follow the cursor rather than only a rescan.
        self.project_panel.show_summary(self._summary_text())
        self.view_changed.emit()
        self.refresh_timeline()
        # Loads before the render: a frame an older batch finishes in between
        # is then either held when the render looks, or asked for again by the
        # new batch, whose signal is the only one not discarded as stale.
        self._schedule_loads()
        self._render()
        self.refresh_state_panel()
        self._refresh_families()

    def _refresh_families(self) -> None:
        """Point the Families tab at the project in view, and at the condition shown."""
        project, lamella = self.controller.project, self.controller.lamella
        shown_project, shown_lamella = self._families_shown
        if project is not shown_project or lamella is not shown_lamella:
            self._families_shown = (project, lamella)
            self.families_panel.show_scope([project] if project is not None else [], lamella)
        if self.families_controller.editing_enabled != self.controller.editing_enabled:
            self.families_panel.set_editing_enabled(self.controller.editing_enabled)
        modality = MODE_MODALITIES.get(self.controller.mode)
        if modality is not None:
            self.families_panel.select_condition(self.controller.current_condition(modality))

    def _refresh_lists(self) -> None:
        """Rebuild both lists. What a scan, or a change of state, needs.

        Moving *within* a project rebuilds only the lamellae -- see
        :meth:`_refresh_lamellae`.
        """
        self._refresh_projects()
        self._refresh_lamellae()

    def _refresh_projects(self) -> None:
        """Rebuild the project tree: its rows, its folders, chips and ticks.

        The name shown is the project's own rather than its whole relative
        path, the folder above it saying the rest -- which is what the tree is
        for and why the two are asked for separately. Which series a project
        holds is not asked for at all: its status chip is a cell per series, so
        an empty column already says it.
        """
        projects = self.controller.projects
        # The cursor goes on the folder while a folder's lamellae are listed.
        # Putting it on the project a lamella happens to belong to would
        # re-enter that list's own signal and snap the scope back to that one
        # project -- and would say the wrong thing about what is being shown.
        folder = self.controller.scope_folder
        self.project_panel.show_projects(
            [project.own_name for project in projects],
            self.controller.project_position,
            self.controller.project_statuses(),
            self.controller.project_checks(),
            folders=[project.folder for project in projects],
            current_folder=folder,
        )

    def _refresh_lamellae(self) -> None:
        """Rebuild the lamella list alone, leaving the project list standing.

        The two are separable because nothing about a project row changes when
        a project is picked: its label, its progress chip and its tick are the
        same before and after. Only the lamellae are the chosen project's own.
        """
        self.project_panel.show_lamellae(
            [lamella.display_name for lamella in self.controller.lamellae],
            self.controller.lamella_position,
            self.controller.lamella_statuses(),
            self.controller.lamella_checks(),
            hidden=self.controller.hidden_lamellae,
        )

    def _summary_text(self) -> str:
        """Describe whatever the tree is on: a folder, or the lamella in view.

        Counted per modality to match the timeline. A combined count would have
        to report frames whose modalities differ as "mixed", which says nothing
        about how much work is left in either.

        Held as state rather than written straight to the label, so that a
        refresh for any other reason does not replace a folder's count with the
        frame's while the folder is still the row selected.
        """
        folder = self.controller.scope_folder
        if folder is not None:
            return self._folder_text(folder)
        if self.controller.project is None:
            return "No project loaded"

        lines = [f"{len(self.controller.frames)} frames"]
        for modality, states in self.controller.modality_states().items():
            counts: dict[str, int] = {}
            for state in states:
                if state is not None:
                    counts[state] = counts.get(state, 0) + 1
            if not counts:
                lines.append(f"{modality}: none")
                continue
            breakdown = ", ".join(
                f"{counts[state]} {state}"
                for state in ("deselected", "selected", "corrected")
                if state in counts
            )
            lines.append(f"{modality}: {breakdown}")
        return "\n".join(lines)

    def _folder_text(self, folder: str) -> str:
        """Describe what is under a folder, so a closed one can be judged shut.

        Progress is read out only for a series anyone labels, the same rule the
        rows follow: a folder of untouched reference images is not work waiting
        to be done.
        """
        summary = self.controller.folder_summary(folder)
        if not summary.projects:
            return f"{folder}\nNothing under this folder"
        counted = (
            f"{summary.projects} project{'s' if summary.projects != 1 else ''} · "
            f"{summary.lamellae} lamella{'e' if summary.lamellae != 1 else ''} · "
            f"{summary.images} image{'s' if summary.images != 1 else ''}"
        )
        return f"{folder}\n{counted}\n{summary.status.summary}"

    def refresh_timeline(self) -> None:
        total = len(self.controller.frames)
        selected = self.controller.selected_frames()
        self.timeline.set_states(
            self.controller.modality_states(),
            self.controller.frame_position,
            self.controller.label_states(),
            selected,
        )
        position = f"{self.controller.frame_position + 1} / {total}" if total else "0 / 0"
        self.frame_label.setText(
            f"{position}\n{len(selected)} picked" if selected else position
        )

    def _render(self) -> None:
        """Draw the current frame, or say why it cannot be drawn yet."""
        # Before the empty check: a lamella can hold no frames of the series
        # being shown while holding plenty of another, and the control is the
        # only way out of that.
        self.image_panel.set_series(
            self.controller.available_series(), self.controller.series
        )

        # Before the frame, and before the empty check: the overview belongs to
        # the project rather than to any frame, so it is there to look at even
        # where the lamella on screen holds nothing of the series being shown.
        if self.controller.mode == "grid":
            self.image_panel.set_available_modes(self.controller.available_modes())
            self.image_panel.set_mode("grid")
            self.image_panel.set_status(self._status_text())
            self._render_grid()
            return

        frame = self.controller.frame
        if frame is None:
            # Emptied rather than left as it was: the conditions on offer
            # belonged to a frame that is no longer on screen.
            self.image_panel.set_conditions([])
            if self.controller.start_path is None:
                self.image_panel.show_nothing_read(self.typed_folder().strip())
            else:
                self.image_panel.show_empty()
            return

        self.image_panel.set_available_modes(self.controller.available_modes())
        # Read back rather than only set on the way in: a timeline holding no
        # image of the beam on screen puts the view on the other one, and the
        # buttons have to say so.
        self.image_panel.set_mode(self.controller.mode)
        self.image_panel.set_conditions(
            self.controller.condition_choices(),
            self.controller.current_condition(),
        )
        self.image_panel.set_status(self._status_text())

        if self.controller.mode == "plot":
            if frame.displayed_plot is None:
                self.image_panel.show_plot(None)
            elif self.controller.cache.cached_plot(self.controller.frame_position) is None:
                self.image_panel.show_loading()
            else:
                self._render_plot()
            self.image_panel.set_plot_scope(frame.plot_is_shared)
            return

        modality = "SEM" if self.controller.mode == "sem" else "FIB"
        condition = self.controller.current_condition(modality)
        if frame.record(modality) is None:
            self.image_panel.show_image(
                None, self.controller.show_overlay, self.controller.show_image
            )
            return
        # Only read what is already prepared: a miss means the loader is still
        # working, and blocking here would freeze the window on a slow read.
        cached = self.controller.cache.cached_image(
            self.controller.frame_position, modality, condition
        )
        if cached is None:
            self.image_panel.show_loading()
        else:
            self.image_panel.show_image(
                cached, self.controller.show_overlay, self.controller.show_image,
                labelled=self.controller.series_is_editable,
            )

    def _render_grid(self) -> None:
        """Draw the project's overview with a mark on each of its lamellae.

        The picture is read on the interface thread, once per project: it is
        one file of a few megabytes, where the frame loader exists for a
        timeline being stepped through several times a second. The overview
        drawn is the one the picker names; one that places nothing is still
        drawn, with the reason in place of the marks.
        """
        grid = self.controller.grid
        choice = grid.current
        if choice is None:
            self.image_panel.show_grid(None)
            return
        self.image_panel.set_overviews(
            [entry.caption for entry in grid.choices], grid.chosen
        )
        placement = choice.placement
        prepared = grid.image()
        if prepared is None or prepared.failed:
            self.image_panel.show_grid(
                None, error=prepared.error if prepared is not None else None
            )
            return
        lamella = self.controller.lamella
        # A synthetic lamella is the project root standing in for one, which is
        # where the overviews themselves are indexed: it is not a lamella and
        # is not missing a position, so nothing is claimed about it. Nor is a
        # directory of overview tiles -- see `lamella_holds_curated_images`.
        expects_a_position = (
            lamella is not None
            and not lamella.synthetic
            and self.controller.lamella_holds_curated_images
        )
        name = lamella.name if lamella is not None else None
        self.image_panel.show_grid(
            prepared.image,
            placement.markers if placement is not None else (),
            name,
            unplaced=placement is not None
            and expects_a_position
            and grid.marker_named(name) is None,
            reason=choice.reason,
        )

    def _render_plot(self) -> None:
        """Hand the panel both copies of the current figure, where it has both.

        Which one is drawn is the panel's to decide, because it depends on how
        far the figure is zoomed and that changes without anything reaching
        this far.
        """
        position = self.controller.frame_position
        self.image_panel.show_plot(
            self.controller.current_plot(),
            self.controller.cache.cached_plot_detail(position),
        )

    def _status_text(self) -> str:
        frame = self.controller.frame
        project = self.controller.project
        lamella = self.controller.lamella
        if frame is None or project is None or lamella is None:
            return ""
        lamella_position = (
            f"{self.controller.lamella_position + 1}/{len(self.controller.lamellae)}"
        )
        return (
            f"{project.display_name}  |  {lamella.display_name} "
            f"({lamella_position})  |  {frame.display_name}  |  {frame.stage}"
        )

    def _refresh_prediction_summary(self) -> None:
        """Say what converting the selection would find, from the source chosen."""
        self.operations_panel.show_predictions(
            self.operations_controller.describe_predictions(
                self.controller.scoped_records(),
                self.operations_panel.prediction_origin(),
            )
        )

    def refresh_state_panel(self) -> None:
        self.state_panel.show_state(
            self.controller.target_state(),
            self.controller.state_of("SEM"),
            self.controller.state_of("FIB"),
        )
        reach = self.controller.reach()
        self.state_panel.show_reach(
            reach.summary(), reach.shared_note(), reach.shared_images
        )
        self.state_panel.show_selection(self.controller.describe_selection())
        self.operations_panel.show_selection(self.controller.describe_selection())
        self._refresh_prediction_summary()
        self.state_panel.set_target(self.controller.target)
        self.state_panel.show_family_choice(reach.family_choice, self.controller.family_scope)
        self.state_panel.set_editing_enabled(self.controller.editing_enabled)
        # Asked of the series being shown rather than the one chosen: a lamella
        # holding only unclassified images resolves to `other` on its own.
        editable = self.controller.series_is_editable
        self.state_panel.set_series_editable(editable)
        self.operations_panel.set_series_editable(editable)

    def _schedule_loads(self) -> None:
        """Ask the loader thread to prepare the frames around this one.

        Taking a new token first abandons any batch still running for a
        position the user has already left.
        """
        token = self.loader.next_token()
        requests = self.controller.pending_loads()
        if requests:
            self.loader.load_batch(requests, token)

    def _on_loaded(self, token: int, frame_index: int) -> None:
        """Redraw when the frame on screen finishes preparing."""
        if token != self.loader.token:
            return
        if frame_index == self.controller.frame_position:
            self._render()
