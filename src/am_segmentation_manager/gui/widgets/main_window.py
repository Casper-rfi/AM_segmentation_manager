"""The curation window: layout, wiring and keyboard shortcuts.

This module connects the three panels and the frame timeline to the controller.
It makes no decisions of its own -- it asks the controller what to show, and
tells the controller what the user did.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QByteArray, QRect, QSettings, Qt, QUrl, Signal
from qtpy.QtGui import QCursor, QDesktopServices
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ...operations.selection import RecordState, current_state
from ..controller import CurationController, EditingDisabled
from ..export_controller import ExportController
from ..loader import BackgroundLoader
from ..metadata_controller import MetadataController
from .changes_log import ChangesLog
from .export_panel import ExportPanel
from .frame_strip import FrameTimeline
from .image_panel import ImagePanel
from .metadata_panel import MetadataPanel
from .project_panel import ProjectPanel
from .state_panel import StatePanel

logger = logging.getLogger(__name__)

#: Where the window's own preferences live. QSettings writes to the platform's
#: user configuration area, never into the data folders being reviewed.
SETTINGS_ORGANISATION = "am_segmentation_manager"
SETTINGS_APPLICATION = "curation"
SETTING_SCAN_ROOT = "scan_root"
SETTING_GEOMETRY = "geometry"
SETTING_SPLITTER = "splitter"

#: How many filenames a confirmation lists before summarising the rest.
MAX_LISTED_FILES = 12

try:  # Qt 6 moved QShortcut out of QtWidgets.
    from qtpy.QtGui import QKeySequence, QShortcut
except ImportError:  # pragma: no cover - depends on the installed binding
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import QShortcut

#: Which tab a key belongs to. A key is live only while its tab is showing --
#: without that, Up/Down step the lamella behind whichever tab you are actually
#: looking at, and the lists on the other tabs never see an arrow key at all.
#:
#: Each is named after its tab, and the tabs after the step of the workflow they
#: cover. The widget behind "Add data" is the metadata panel, because describing
#: a project is what adding one to the collection currently consists of.
TAB_CURATE = "curate"
TAB_ADD_DATA = "add data"
TAB_EXPORT = "export"

#: A key that works everywhere, whatever is showing.
ANY_TAB = "any"


@dataclass(frozen=True)
class Binding:
    """One keyboard shortcut: what it does, where it applies, and what it says.

    Attributes:
        key: The key sequence, as :class:`QKeySequence` accepts it.
        description: The line shown in the F1 list. Every binding has one.
        action: What pressing the key does.
        tab: The tab this key belongs to, or :data:`ANY_TAB`.
        scope: Restrict the key to this widget and its children instead of the
            whole window. Needed where a tab holds editors with arrow-key
            behaviour of their own, which a window-wide key would take first.
    """

    key: str
    description: str
    action: Callable[[], None]
    tab: str
    scope: QWidget | None = None


class CurationWindow(QMainWindow):
    """Main window for reviewing and curating adaptive milling data."""

    #: Emitted to the background loader thread; see :class:`BackgroundLoader`.
    request_load = Signal(list, int)

    def __init__(
        self, controller: CurationController, settings: QSettings | None = None
    ) -> None:
        """Build the window.

        Args:
            controller: The view model the window reads from and reports to.
            settings: Where the window's own preferences are kept. Defaults to
                the user's configuration area; tests pass a throwaway file so a
                test run cannot overwrite a real window's remembered layout.
        """
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Adaptive milling segmentation curation")
        self.resize(1600, 950)

        self.settings = settings if settings is not None else QSettings(
            SETTINGS_ORGANISATION, SETTINGS_APPLICATION
        )
        self.project_panel = ProjectPanel()
        self.image_panel = ImagePanel(controller.config)
        self.state_panel = StatePanel()
        self.changes_log = ChangesLog()

        # The other tabs get their own Qt-free controllers, and are handed the
        # projects from the curation scan rather than scanning again: a second
        # scan would double the slowest thing the application does.
        self.export_controller = ExportController()
        self.metadata_controller = MetadataController()
        self.export_panel = ExportPanel(self.export_controller)
        self.metadata_panel = MetadataPanel(self.metadata_controller)

        self._build_layout()
        self._connect()
        self._add_shortcuts()
        self._restore_layout()

        self.loader = BackgroundLoader(controller.cache)
        self.request_load.connect(self.loader.worker.load_batch)
        self.loader.worker.loaded.connect(self._on_loaded)

        remembered = self.settings.value(SETTING_SCAN_ROOT, "", type=str)
        if remembered:
            self.project_panel.set_path(remembered)

        self.refresh()

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

        self.splitter = splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.addWidget(self.project_panel)
        splitter.addWidget(centre)
        splitter.addWidget(self.state_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1080, 220])

        # The log spans the window under everything else, and can be dragged
        # shut. See changes_log.py for why it is not in the right-hand column.
        self.curate = QSplitter(Qt.Vertical)
        self.curate.setHandleWidth(3)
        self.curate.addWidget(splitter)
        self.curate.addWidget(self.changes_log)
        self.curate.setStretchFactor(0, 1)
        self.curate.setSizes([800, 120])

        # Left to right in workflow order: describe the data, curate it, export
        # it. Curate opens, because it is where the repeated work happens and
        # the other two are visited once per collection.
        self.tabs = QTabWidget()
        self.tabs.addTab(self.metadata_panel, "Add data")
        self.tabs.addTab(self.curate, "Curate")
        self.tabs.addTab(self.export_panel, "Export")
        # Chosen before the signal is connected: _tab_changed switches the
        # keyboard shortcuts, which do not exist until _add_shortcuts runs.
        self.tabs.setCurrentWidget(self.curate)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

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

        self.timeline = FrameTimeline()
        self.timeline.frame_selected.connect(self._seek_frame)
        self.timeline.frame_toggled.connect(self._toggle_frame)
        self.timeline.extend_started.connect(self._begin_extend)
        self.timeline.extend_reached.connect(self._extend_frames)
        row.addWidget(self.timeline, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(70)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.frame_label.setToolTip("Frame position within this lamella")
        row.addWidget(self.frame_label)

        return row

    def _connect(self) -> None:
        self.project_panel.scan_requested.connect(self.scan)
        self.project_panel.project_selected.connect(self._select_project)
        self.project_panel.lamella_selected.connect(self._select_lamella)
        self.project_panel.project_path_requested.connect(
            lambda row: self._copy_path(self.controller.project_path(row), "Project")
        )
        self.project_panel.lamella_path_requested.connect(
            lambda row: self._copy_path(self.controller.lamella_path(row), "Lamella")
        )
        self.project_panel.project_reveal_requested.connect(
            lambda row: self._reveal_path(self.controller.project_path(row))
        )
        self.project_panel.lamella_reveal_requested.connect(
            lambda row: self._reveal_path(self.controller.lamella_path(row))
        )
        self.project_panel.project_checked.connect(self._check_project)
        self.project_panel.lamella_checked.connect(self._check_lamella)
        self.project_panel.projects_check_all.connect(self._check_all_projects)
        self.project_panel.lamellae_check_all.connect(self._check_all_lamellae)
        self.project_panel.target_changed.connect(self._set_target)

        self.image_panel.mode_changed.connect(self._set_mode)
        self.image_panel.variant_changed.connect(self._set_variant)
        self.image_panel.overlay_toggled.connect(self._set_overlay)
        self.image_panel.alpha_changed.connect(lambda _alpha: self._render())
        self.image_panel.path_requested.connect(
            lambda: self._copy_path(self._displayed_path(), "Image")
        )
        self.image_panel.reveal_requested.connect(
            lambda: self._reveal_path(self._displayed_path())
        )

        self.state_panel.editing_toggled.connect(self.controller.set_editing_enabled)
        self.state_panel.state_requested.connect(self.apply_state)

        self.export_panel.training_export_requested.connect(self._export_training)
        self.export_panel.image_metadata_export_requested.connect(self._export_image_metadata)
        self.export_panel.project_metadata_export_requested.connect(self._export_project_metadata)
        # Editing metadata changes what the export view groups by, so it has to
        # be told rather than left showing the previous sample name.
        self.metadata_panel.metadata_changed.connect(self.export_panel.refresh)
        self.metadata_panel.project_path_requested.connect(
            lambda row: self._copy_path(self.metadata_controller.project_path(row), "Project")
        )
        self.metadata_panel.project_reveal_requested.connect(
            lambda row: self._reveal_path(self.metadata_controller.project_path(row))
        )

    # -- remembering the layout --------------------------------------------

    def _splitters(self) -> dict[str, QSplitter]:
        """Return the resizable divisions, keyed by the name they persist under.

        Keyed by name rather than by position so that adding a splitter, or
        moving one, cannot silently restore the wrong sizes into it.
        """
        return {
            "curate": self.splitter,
            "changes": self.curate,
            "lists": self.project_panel.lists,
            "add_data": self.metadata_panel.panes,
        }

    def _restore_layout(self) -> None:
        """Put the window back the size and shape it was left in."""
        geometry = self.settings.value(SETTING_GEOMETRY, None, type=QByteArray)
        if geometry:
            self.restoreGeometry(geometry)
        for name, splitter in self._splitters().items():
            state = self.settings.value(
                f"{SETTING_SPLITTER}/{name}", None, type=QByteArray
            )
            if state:
                splitter.restoreState(state)

    def _save_layout(self) -> None:
        """Record the window's size and its splitter positions."""
        self.settings.setValue(SETTING_GEOMETRY, self.saveGeometry())
        for name, splitter in self._splitters().items():
            self.settings.setValue(f"{SETTING_SPLITTER}/{name}", splitter.saveState())

    # -- the other tabs ----------------------------------------------------

    def _tab_changed(self, index: int) -> None:
        """Refresh a tab when it comes forward, and hand it its keys.

        Refreshing only on the way in keeps the cost off the curation view,
        which is where the keyboard-driven work happens.
        """
        widget = self.tabs.widget(index)
        if widget is self.export_panel:
            self.export_panel.refresh()
        elif widget is self.metadata_panel:
            self.metadata_panel.refresh()
            # The arrow keys here are scoped to the project list, so they only
            # work once something in it has focus. Giving it focus on arrival
            # is what makes them work without a click first.
            self.metadata_panel.projects.setFocus()
        self._apply_tab_shortcuts(self._current_tab())

    def _share_scan(self) -> None:
        """Hand the scanned projects to the other tabs."""
        projects = self.controller.projects
        self.export_controller.set_projects(projects, self.controller.start_path)
        self.metadata_controller.set_projects(projects)
        self.metadata_panel.set_projects(self.metadata_controller.project_summaries())

    def _export_training(self, directory: str) -> None:
        from ...operations.csv import export_training_csv

        records = self.export_controller.selected_records()
        if not records:
            self._flash("Nothing matches these filters")
            return
        filters = self.export_controller.filters
        try:
            train, validation = export_training_csv(
                records=records,
                start_path=self.controller.start_path or Path(directory),
                output_dir=Path(directory),
                include_empty_labels=filters.include_empty_labels,
                conditions=filters.conditions,
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        written = ", ".join(str(path.name) for path in (train, validation) if path)
        self._flash(f"Wrote {written}", 4000)
        logger.info("Exported training CSV to %s", directory)

    def _export_image_metadata(self, path: str) -> None:
        from ...aggregate import project_lookup
        from ...operations.metadata_csv import export_image_metadata_csv

        records = self.export_controller.selected_records()
        if not records:
            self._flash("Nothing matches these filters")
            return
        try:
            export_image_metadata_csv(
                records,
                self.controller.start_path or Path(path).parent,
                Path(path),
                projects=project_lookup(self.export_controller.projects),
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self._flash(f"Wrote {Path(path).name}", 4000)

    def _export_project_metadata(self, path: str) -> None:
        from ...operations.metadata_csv import export_project_metadata_csv

        try:
            export_project_metadata_csv(
                self.export_controller.projects,
                self.controller.start_path or Path(path).parent,
                Path(path),
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self._flash(f"Wrote {Path(path).name}", 4000)

    # -- copying paths out of the window -----------------------------------

    def _displayed_path(self) -> Path | None:
        """Return the file currently on screen, image or plot."""
        if self.controller.mode == "plot":
            return self.controller.current_plot_path()
        return self.controller.current_image_path()

    def _copy_path(self, path: Path | None, noun: str) -> None:
        """Put *path* on the clipboard and confirm it beside the cursor.

        Confirmed with a transient tooltip rather than a status bar. A status
        bar costs ~25px of window height permanently, and the image panes are
        downscaled to fit whatever is left -- so permanent chrome added for
        occasional feedback changes the resampling of every image on screen.
        Vertical space is the scarce resource in this window.
        """
        if path is None:
            self._flash(f"No {noun.lower()} to copy a path for")
            return
        QApplication.clipboard().setText(str(path))
        self._flash(f"{noun} path copied")
        logger.info("Copied %s path to clipboard: %s", noun.lower(), path)

    def _flash(self, message: str, milliseconds: int = 2000) -> None:
        """Show a short confirmation at the pointer, taking no layout space."""
        QToolTip.showText(QCursor.pos(), message, self, QRect(), milliseconds)

    def _reveal_path(self, path: Path | None) -> None:
        """Open *path*'s folder in the desktop file manager.

        The containing directory is opened rather than the file itself: opening
        a .tif would hand it to whatever application claims that extension,
        which is not what "reveal" means.
        """
        if path is None:
            self._flash("Nothing to reveal")
            return
        folder = path if path.is_dir() else path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        logger.info("Revealed %s", folder)

    def _add_shortcuts(self) -> None:
        """Bind the keys, each tagged with the tab it belongs to.

        A binding scoped to a widget uses ``WidgetWithChildrenShortcut`` and so
        only fires with focus inside it; every other binding belongs to the
        window and is switched on and off as tabs change.
        """
        curate = TAB_CURATE
        self._bindings = [
            Binding("Right", "Next frame", lambda: self._step_frame(1), curate),
            Binding("Left", "Previous frame", lambda: self._step_frame(-1), curate),
            Binding("Down", "Next lamella", lambda: self._step_lamella(1), curate),
            Binding("Up", "Previous lamella", lambda: self._step_lamella(-1), curate),
            Binding("1", "SEM view", lambda: self._set_mode("sem", from_user=True), curate),
            Binding("2", "FIB view", lambda: self._set_mode("fib", from_user=True), curate),
            Binding("3", "Plot view", lambda: self._set_mode("plot", from_user=True), curate),
            Binding("Space", "Show / hide the label overlay", self._toggle_overlay, curate),
            Binding("D", "Set state: deselected", lambda: self.apply_state("deselected"), curate),
            Binding("S", "Set state: selected", lambda: self.apply_state("selected"), curate),
            Binding("C", "Set state: corrected", lambda: self.apply_state("corrected"), curate),
            Binding("F", "Fit the images to the panes", self.image_panel.pair.fit, curate),
            Binding(
                "Esc", "Clear the picked-out frames", self._clear_frame_selection, curate
            ),
            # Scoped to the project list: the fields beside it include editable
            # combo boxes, whose own Up/Down a window-wide key would swallow.
            Binding(
                "Down", "Next project", lambda: self.metadata_panel.step_project(1),
                TAB_ADD_DATA, scope=self.metadata_panel.projects,
            ),
            Binding(
                "Up", "Previous project", lambda: self.metadata_panel.step_project(-1),
                TAB_ADD_DATA, scope=self.metadata_panel.projects,
            ),
            Binding("F1", "Show this list", self.show_shortcuts, ANY_TAB),
        ]
        self._shortcuts = []
        for binding in self._bindings:
            shortcut = QShortcut(
                QKeySequence(binding.key), binding.scope or self, activated=binding.action
            )
            if binding.scope is not None:
                shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            self._shortcuts.append(shortcut)
        self._apply_tab_shortcuts(self._current_tab())

    def _apply_tab_shortcuts(self, tab: str) -> None:
        """Switch on the keys belonging to *tab*, and switch off the rest."""
        for binding, shortcut in zip(self._bindings, self._shortcuts):
            shortcut.setEnabled(binding.tab in (tab, ANY_TAB))

    def _current_tab(self) -> str:
        """Return the key of the tab now showing."""
        widget = self.tabs.currentWidget()
        for panel, tab in (
            (self.metadata_panel, TAB_ADD_DATA), (self.export_panel, TAB_EXPORT)
        ):
            if widget is panel:
                return tab
        return TAB_CURATE

    def shortcut_help(self, tab: str | None = None) -> str:
        """Return the shortcut table for a tab, as the help dialog's HTML.

        Separate from showing it so what the dialog says can be tested without
        opening a modal window.
        """
        tab = tab if tab is not None else self._current_tab()
        applicable = [b for b in self._bindings if b.tab in (tab, ANY_TAB)]
        rows = "".join(
            f"<tr><td style='padding-right: 14px;'><b>{binding.key}</b></td>"
            f"<td>{binding.description}</td></tr>"
            for binding in applicable
        )
        help_text = f"<table>{rows}</table>"
        if not any(binding.tab != ANY_TAB for binding in applicable):
            help_text += (
                "<p style='margin-top: 10px;'>This tab has no keys of its own.</p>"
            )
        if tab == TAB_CURATE:
            help_text += (
                "<p style='margin-top: 10px;'>State changes need <i>Enable editing</i> "
                "switched on.</p>"
            )
        return help_text

    def show_shortcuts(self) -> None:
        """List the keys that work on the tab now showing.

        Tooltips only help someone who already suspects a control exists; this
        is the one place that answers "what can I press?". Only the active tab's
        keys are listed, because the others are switched off and pressing one
        does nothing.
        """
        QMessageBox.information(
            self,
            f"Keyboard shortcuts: {self.tabs.tabText(self.tabs.currentIndex())}",
            self.shortcut_help(),
        )

    # -- user actions ------------------------------------------------------

    def scan(self, path: str) -> None:
        """Scan *path* for projects and show what was found.

        Under a wait cursor throughout: scanning a collection is the slowest
        thing the window does, and until it finishes there is nothing on screen
        to say anything is happening.
        """
        if not path:
            return
        directory = Path(path)
        if not directory.is_dir():
            QMessageBox.warning(self, "Scan", f"Not a folder: {directory}")
            return

        failure: Exception | None = None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.controller.scan(directory)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.exception("Scan failed")
            failure = exc
        else:
            self.project_panel.set_path(str(directory))
            self.settings.setValue(SETTING_SCAN_ROOT, str(directory))
            self._share_scan()
            self.refresh(reload_lists=True)
        finally:
            QApplication.restoreOverrideCursor()

        # Reported after the cursor is back: a modal dialog under an hourglass
        # looks like the window is still working on something.
        if failure is not None:
            QMessageBox.critical(self, "Scan failed", str(failure))

    def _select_project(self, row: int) -> None:
        self.controller.select_project(row)
        self.refresh(reload_lists=True)

    def _select_lamella(self, row: int) -> None:
        self.controller.select_lamella(row)
        self.refresh()

    def _seek_frame(self, value: int) -> None:
        if value != self.controller.frame_position:
            self.controller.select_frame(value)
            self.refresh()

    def _step_frame(self, delta: int) -> None:
        self.controller.step_frame(delta)
        self.refresh()

    def _step_lamella(self, delta: int) -> None:
        self.controller.step_lamella(delta)
        self.refresh(reload_lists=True)

    def _set_mode(self, mode: str, from_user: bool = False) -> None:
        if from_user and mode not in self.controller.available_modes():
            return
        self.controller.set_mode(mode)
        self.image_panel.set_mode(mode)
        # set_mode may have pointed the edit target at the new modality.
        self.project_panel.set_target(self.controller.target)
        self.refresh()

    def _set_overlay(self, show: bool) -> None:
        self.controller.show_overlay = show
        self._render()

    def _toggle_overlay(self) -> None:
        self.image_panel.overlay_check.toggle()

    def _set_target(self, target: str) -> None:
        self.controller.set_target(target)
        self._refresh_state_panel()

    # -- choosing what a change applies to ---------------------------------
    #
    # Every one of these ends in a refresh of the state panel, because the
    # reach line above the state buttons is the number that has to agree with
    # whatever was just ticked or picked out.

    def _check_project(self, row: int, checked: bool) -> None:
        self.controller.set_project_checked(row, checked)
        self._refresh_list_titles()
        self._refresh_state_panel()

    def _check_lamella(self, row: int, checked: bool) -> None:
        self.controller.set_lamella_checked(row, checked)
        self._refresh_list_titles()
        self._refresh_state_panel()

    def _check_all_projects(self, checked: bool) -> None:
        self.controller.check_all_projects(checked)
        self._refresh_lists()
        self._refresh_state_panel()

    def _check_all_lamellae(self, checked: bool) -> None:
        self.controller.check_all_lamellae(checked)
        self._refresh_lists()
        self._refresh_state_panel()

    def _toggle_frame(self, position: int) -> None:
        """Add a frame to the selection, or take it out again."""
        self.controller.selection.toggle(position)
        self._refresh_after_selecting()

    def _begin_extend(self, position: int) -> None:
        self.controller.selection.begin_extend(position)

    def _extend_frames(self, position: int) -> None:
        self.controller.selection.extend_to(position)
        self._refresh_after_selecting()

    def _clear_frame_selection(self) -> None:
        self.controller.selection.clear_frames()
        self._refresh_after_selecting()

    def _refresh_after_selecting(self) -> None:
        """Redraw what a frame selection changes, and nothing else.

        Deliberately not the whole window: picking frames out does not move the
        view, so nothing needs re-rendering or re-loading, and a drag along the
        timeline emits one of these per pixel.
        """
        self._refresh_timeline()
        self._refresh_state_panel()

    def _refresh_list_titles(self) -> None:
        """Update the counts of ticked rows without rebuilding the lists.

        Rebuilding would reset the row the user is on, and ticking a row is not
        a change of what is being looked at.
        """
        self.project_panel.show_check_counts(
            self.controller.project_checks(), self.controller.lamella_checks()
        )

    def _set_variant(self, variant: str) -> None:
        """Show a different imaging condition.

        A view change only: which condition is on screen never decides what a
        state change moves, because the change follows the label.
        """
        self.controller.set_variant(variant or None)
        self.refresh()

    def apply_state(self, state: RecordState) -> None:
        """Move the targeted records into *state*, confirming where required.

        Two confirmations can apply, and they are asked in order of how much is
        at stake: first whether the user really means to change more than the
        current frame, then whether proofread work may be moved.
        """
        if self.controller.editing_enabled and not self.controller.selection.is_empty:
            if not self._confirm_bulk(state):
                return
        try:
            changed = self.controller.apply_state(state)
        except EditingDisabled:
            QMessageBox.information(
                self, "Editing is off",
                "Tick 'Enable editing' before changing the state of an image.",
            )
            return
        except PermissionError:
            if not self._confirm_corrected(state):
                self._refresh_state_panel()
                return
            try:
                changed = self.controller.apply_state(state, confirm_corrected=True)
            except OSError as exc:
                self._report_failure(exc)
                return
        except OSError as exc:
            self._report_failure(exc)
            return

        if changed:
            frame = self.controller.frame
            where = self.controller.describe_selection()
            if self.controller.selection.is_empty and frame is not None:
                where = frame.display_name
            self.changes_log.append(
                f"{where}: {changed} file(s) -> {state} ({self.controller.target})"
            )
        # Rebuild the lists too: a state change is exactly what moves a lamella
        # or project between "not started", "in progress" and "done", so leaving
        # the status chips alone here would leave them stale at the one moment
        # they are meant to report on. Recomputing every status costs ~60ms on a
        # 21-project collection, against a change that has just moved files.
        self.refresh(reload_lists=True)

    def _confirm_bulk(self, state: str) -> bool:
        """Ask before changing more than the frame on screen.

        Counts the *expanded* set, not the selection: images sharing a label
        move with it, so counting what was ticked alone would promise to move
        fewer files than the operation actually does.
        """
        scoped = self.controller.scoped_records()
        if not scoped:
            return True
        records = self.controller.expanded_records(scoped)
        shared = len(records) - len(scoped)

        where = self.controller.describe_selection()
        counts = Counter(record.modality for record in records)
        breakdown = ", ".join(f"{count} {modality}" for modality, count in sorted(counts.items()))
        unchanged = sum(1 for r in records if current_state(r) == state)

        message = (
            f"Set <b>{len(records)}</b> images ({breakdown}) in {where} "
            f"to <b>{state}</b>?"
        )
        if shared:
            message += (
                f"<p>{shared} of them share a label with your selection and move "
                f"with it.</p>"
            )
        if unchanged:
            message += f"<p>{unchanged} of them are already {state}.</p>"

        answer = QMessageBox.warning(
            self, "Change many images?", message,
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

    def _confirm_corrected(self, state: str) -> bool:
        """Ask before moving proofread data out of the corrected state.

        Only the files actually holding proofread work are named. Listing
        everything targeted would misreport which data is at risk -- a frame
        whose SEM is corrected and whose FIB is merely deselected puts only the
        SEM at stake.
        """
        corrected = self.controller.corrected_records()
        if not corrected:
            return True

        names = [record.image_path.name for record in corrected]
        shown = "\n".join(names[:MAX_LISTED_FILES])
        if len(names) > MAX_LISTED_FILES:
            shown += f"\n... and {len(names) - MAX_LISTED_FILES} more"

        answer = QMessageBox.warning(
            self,
            "Move corrected data?",
            f"{len(names)} of these files are in the corrected state, which "
            f"holds manually proofread work:\n\n{shown}\n\nMove them to '{state}'?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

    def _report_failure(self, exc: Exception) -> None:
        logger.warning("State change refused: %s", exc)
        QMessageBox.critical(self, "Could not change state", str(exc))

    # -- rendering ---------------------------------------------------------

    def refresh(self, *, reload_lists: bool = False) -> None:
        """Bring the whole window back in step with the controller."""
        if reload_lists:
            self._refresh_lists()
        self._refresh_timeline()
        self._render()
        self._refresh_state_panel()
        self._schedule_loads()

    def _refresh_lists(self) -> None:
        self.project_panel.show_projects(
            [project.display_name for project in self.controller.projects],
            self.controller.project_position,
            self.controller.project_statuses(),
            self.controller.project_checks(),
        )
        self.project_panel.show_lamellae(
            [lamella.display_name for lamella in self.controller.lamellae],
            self.controller.lamella_position,
            self.controller.lamella_statuses(),
            self.controller.lamella_checks(),
        )
        self.project_panel.show_summary(self._summary_text())

    def _summary_text(self) -> str:
        """Describe the lamella's curation progress, per modality.

        Counted per modality to match the timeline. A combined count would have
        to report frames whose modalities differ as "mixed", which says nothing
        about how much work is left in either.
        """
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

    def _refresh_timeline(self) -> None:
        total = len(self.controller.frames)
        selected = self.controller.selected_frames()
        self.timeline.set_states(
            self.controller.modality_states(),
            self.controller.frame_position,
            self.controller.mismatched_frames(),
            self.controller.label_states(),
            selected,
        )
        position = f"{self.controller.frame_position + 1} / {total}" if total else "0 / 0"
        self.frame_label.setText(
            f"{position}\n{len(selected)} picked" if selected else position
        )

    def _render(self) -> None:
        """Draw the current frame, or say why it cannot be drawn yet."""
        frame = self.controller.frame
        if frame is None:
            self.image_panel.show_empty()
            return

        self.image_panel.set_available_modes(self.controller.available_modes())
        self.image_panel.set_variants(
            self.controller.available_variants(),
            self.controller.current_variant(),
            self._variant_descriptions(),
        )
        self.image_panel.set_status(self._status_text())

        if self.controller.mode == "plot":
            self._match_plot_size_to_pane()
            if frame.plot_path is None:
                self.image_panel.show_plot(None)
            elif self.controller.cache.cached_plot(self.controller.frame_position) is None:
                self.image_panel.show_loading()
            else:
                self.image_panel.show_plot(self.controller.current_plot())
            return

        modality = "SEM" if self.controller.mode == "sem" else "FIB"
        variant = self.controller.current_variant(modality)
        if frame.record(modality, variant) is None and frame.record(modality) is None:
            self.image_panel.show_image(None, self.controller.show_overlay)
            return
        # Only read what is already prepared: a miss means the loader is still
        # working, and blocking here would freeze the window on a slow read.
        cached = self.controller.cache.cached_image(
            self.controller.frame_position, modality, variant
        )
        if cached is None:
            self.image_panel.show_loading()
        else:
            self.image_panel.show_image(cached, self.controller.show_overlay)

    def _variant_descriptions(self) -> dict[str, str]:
        """Return human-readable names for the project's imaging conditions.

        A variant is an opaque string taken from a directory name, so ``5kv``
        is all the filesystem offers. A project may name them in its
        ``AM_metadata.yaml``; where it does not, the raw name is shown.
        """
        project = self.controller.project
        metadata = getattr(project, "experiment_metadata", None) if project else None
        if metadata is None:
            return {}
        return dict(metadata.variants)

    def _match_plot_size_to_pane(self) -> None:
        """Prepare plots at the size they are actually shown at.

        Reported in *physical* pixels. On a high-resolution display a widget
        1070 logical pixels wide occupies 2140 real ones, and a plot prepared
        to the logical size would be magnified to fill it -- which no choice of
        filter can undo.
        """
        view = self.image_panel.plot_view
        viewport = view.viewport().rect()
        if viewport.isEmpty():
            return
        ratio = view.devicePixelRatioF()
        size = (round(viewport.width() * ratio), round(viewport.height() * ratio))
        if self.controller.cache.set_plot_max_size(size):
            # The prepared plots were the wrong size and have been discarded.
            self._schedule_loads()

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

    def _refresh_state_panel(self) -> None:
        self.state_panel.show_state(
            self.controller.target_state(),
            self.controller.state_of("SEM"),
            self.controller.state_of("FIB"),
        )
        reach = self.controller.reach()
        self.state_panel.show_reach(reach.summary(), reach.shared_note())
        self.state_panel.show_selection(self.controller.describe_selection())
        self.project_panel.set_target(self.controller.target)
        self.state_panel.set_editing_enabled(self.controller.editing_enabled)

    def _schedule_loads(self) -> None:
        """Ask the loader thread to prepare the frames around this one.

        Taking a new token first abandons any batch still running for a
        position the user has already left.
        """
        token = self.loader.next_token()
        requests = self.controller.pending_loads()
        if requests:
            self.request_load.emit(requests, token)

    def _on_loaded(self, token: int, request) -> None:
        """Redraw when the frame on screen finishes preparing."""
        if token != self.loader.token:
            return
        if request.frame_index == self.controller.frame_position:
            self._render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._save_layout()
        self.loader.stop()
        super().closeEvent(event)
