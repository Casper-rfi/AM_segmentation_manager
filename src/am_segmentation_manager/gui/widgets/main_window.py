"""The window: four tabs, the controllers behind them, and what joins them.

Each tab has a controller holding no Qt; this module decides nothing, asking one
what to show and telling it what the user did, and builds them and the window.
**Over a read-only controller it is a viewer**: Curate alone, with nothing that
writes. The Curate tab is :mod:`.curate_panel`; :mod:`.operation_runner`,
:mod:`.operation_actions` and :mod:`.label_tab` are mixins beside it, and the keys
are :mod:`.shortcuts`, held rather than inherited. ``docs/display.md`` holds why,
and every Qt trap in it.
"""
from __future__ import annotations

import html
import logging
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from qtpy import API_NAME, QT_VERSION
from qtpy.QtCore import QByteArray, QEvent, QRect, QSettings, QSize, Qt, QUrl
from qtpy.QtGui import QAction, QCursor, QDesktopServices, QIcon, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ... import __version__
from ...history import clear_author, set_author
from ...identity import PRODUCT_NAME, REPOSITORY_URL, VIEWER_NAME
from ...installation import update_steps, version_date
from ...launcher import VIEWER, WINDOW, LauncherRefused, menu_label, this_computer
from ...layout import MODALITIES
from ...operations.predict_process import DeviceProbe
from ...operations.selection import (
    RecordState,
    StaleState,
    UnmigratedStage,
    current_state,
)
from .. import preferences
from ..batch_scope import counted
from ..controller import (
    MODE_MODALITIES,
    CurationController,
    EditingDisabled,
    EditTarget,
    FamilyScope,
    ScanProgress,
)
from ..export_controller import ExportController
from ..families_controller import FamiliesController
from ..history_controller import HistoryController
from ..label_controller import LabelController
from ..loader import BackgroundLoader
from ..metadata_controller import MetadataController
from ..operations_controller import OperationsController
from ..training_controller import TrainingController
from ..version_controller import VersionController
from ..worker import BackgroundOperation
from .branding import logo_pixmap
from .busy_dialog import BusyDialog
from .common import MAX_LISTED_FILES, vertical_rule
from .controls import wheel_left_to_scrolling_once_built
from .curate_panel import CuratePanel
from .export_panel import ExportPanel
from .icons import settings_icon
from .label_panel import LabelPanel
from .label_tab import LabelTab
from .launcher_dialog import LauncherDialog
from .metadata_panel import MetadataPanel
from .operation_actions import OperationActions
from .operation_runner import OperationRunner
from .scan_banner import ScanBanner
from .scan_bar import ScanBar
from .settings_dialog import SettingsDialog, author_summary
from .shortcuts import (
    TAB_ADD_DATA,
    TAB_CURATE,
    TAB_EXPORT,
    TAB_LABEL,
    TAB_TRAIN,
    Shortcuts,
)
from .theme import ACCENTS, COMPACT_BUTTON, MARK_BUTTON, MUTED, apply_accent
from .training_panel import TrainingPanel

logger = logging.getLogger(__name__)

#: Where the window's own preferences live. QSettings writes to the platform's
#: user configuration area, never into the data folders being reviewed.
SETTINGS_ORGANISATION = "am_segmentation_manager"
SETTINGS_APPLICATION = "curation"
SETTING_SCAN_ROOT = "scan_root"
#: Where the model to predict with is remembered, one entry per series and
#: beam: ``model/<series>/<modality>``. The key below it is what a single box
#: wrote before there were six, and is read where a slot has never been filled.
SETTING_MODEL = "model"
SETTING_LEGACY_MODEL = "model_directory"
#: The device last predicted on, remembered like the model: one machine
#: usually has one answer, and re-choosing it every session is noise.
SETTING_DEVICE = "predict_device"
#: What the Train tab was last pointed at: where runs are kept, the device they
#: train on and the weights last refined. Remembered like the model, and for
#: the same reason; none is a preference, each being a place on this machine.
SETTING_TRAINING_DIRECTORY = "training/directory"
SETTING_TRAINING_DEVICE = "training/device"
SETTING_TRAINING_WEIGHTS = "training/weights"
SETTING_GEOMETRY = "geometry"
SETTING_SPLITTER = "splitter"
#: The group a viewer keeps its own geometry and splitters under.
SETTING_VIEWER = "viewer"

AUTHOR_TITLE = "Settings"

#: The title of what is said when this computer has nowhere to add the window to.
LAUNCHER_TITLE = "Add to this computer"

#: The em dash between the product name and the folder in the window title.
#: Two windows on two collections are told apart by the title and by nothing
#: else, the taskbar and the window switcher showing only that.
TITLE_SEPARATOR = " — "

#: What the window opens at where nothing is remembered, in logical pixels.
#: Chosen for a desktop monitor and **not** a promise that the screen is that
#: big: :meth:`CurationWindow.fit_to_screen` pulls it back to whatever is
#: actually there.
DEFAULT_SIZE = (1600, 950)

#: Left clear of the screen edges when the window is pulled back onto it, in
#: logical pixels. Enough that it still reads as a window that can be moved
#: rather than something wedged into the corner, as the shortcut list keeps.
SCREEN_MARGIN = 40

def user_settings() -> QSettings:
    """Return the store the window keeps its own preferences in, for this user."""
    return QSettings(SETTINGS_ORGANISATION, SETTINGS_APPLICATION)


def _setting_text(settings: QSettings, key: str) -> str:
    """Return the text stored under *key*, or an empty string where none is."""
    value = settings.value(key, "", type=str)
    return value if isinstance(value, str) else ""


def _setting_bytes(settings: QSettings, key: str) -> QByteArray | None:
    """Return the byte array stored under *key*, or None where none is."""
    value = settings.value(key, None, type=QByteArray)
    return value if isinstance(value, QByteArray) else None


class CornerBar(QWidget):
    """The tab row's corner, naming the widgets in it the window reaches back into."""

    mark: QToolButton
    author: QLabel
    settings_button: QToolButton
    help_button: QToolButton


def _corner(scan_bar: QWidget, on_help, on_settings, on_about) -> CornerBar:
    """Return the tab row's corner: the folder, who is recording, and the mark.

    **Three runs with a rule between each**, read from what is being worked on
    to whose window this is. The folder first, nearest the tabs it serves; then
    this machine's own state and the two buttons for what nothing else on
    screen mentions, F2 and F1; then the mark at the far end. **The mark
    without the words**: the title bar carries the product name on Windows and
    Linux and the menu bar does on macOS, and printed here as well it read as
    the window saying its own name twice.

    A widget rather than a `QLabel`, a label showing either a pixmap or text
    and this needing both. **The mark is a button that opens About**, which
    is where a reader asking whose window this is finds its version.

    Args:
        scan_bar: The path, Browse and Scan, which the window builds and owns.
        on_help: Called when the help button is pressed.
        on_settings: Called when the settings button is pressed.
        on_about: Called when the mark is pressed.
    """
    row = CornerBar()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 10, 0)
    layout.setSpacing(6)

    def rule() -> None:
        layout.addSpacing(4)
        layout.addWidget(vertical_rule(CORNER_LOGO))
        layout.addSpacing(4)

    layout.addWidget(scan_bar)
    rule()

    # Always rather than at the moment of a write: a name reaches every entry
    # written from anywhere in the window, so there is no one tab to put it on
    # and far too many actions to warn at. Muted, because it is a standing fact
    # and not a message -- and next to the button that changes it, which is
    # what answers "why is this here" for a reader who has just noticed it.
    author = QLabel()
    author.setObjectName("recordingAs")
    author.setStyleSheet(f"color: {MUTED};")
    layout.addWidget(author)

    settings_button = QToolButton()
    settings_button.setObjectName(COMPACT_BUTTON)
    settings_button.setIcon(settings_icon())
    settings_button.setToolTip("Settings (F2)")
    settings_button.clicked.connect(on_settings)
    layout.addWidget(settings_button)

    # A one-character caption, which the stylesheet's button padding would
    # otherwise clip blank; see `theme.COMPACT_BUTTON`.
    help_button = QToolButton()
    help_button.setObjectName(COMPACT_BUTTON)
    help_button.setText("?")
    help_button.setToolTip("Show keyboard shortcuts for this tab (F1)")
    help_button.clicked.connect(on_help)
    layout.addWidget(help_button)
    rule()

    mark = QToolButton()
    mark.setObjectName(MARK_BUTTON)
    mark.setIcon(QIcon(logo_pixmap(CORNER_LOGO)))
    mark.setIconSize(QSize(CORNER_LOGO, CORNER_LOGO))
    mark.setCursor(Qt.CursorShape.PointingHandCursor)
    mark.setToolTip(f"About {PRODUCT_NAME}")
    mark.clicked.connect(on_about)
    layout.addWidget(mark)

    row.mark = mark
    row.author = author
    row.settings_button = settings_button
    row.help_button = help_button
    return row


#: The mark's side beside the tabs, in logical pixels. Sized to the tab text
#: rather than to the mark: it sits on that row and must not set its height.
CORNER_LOGO = 16

#: The mark's side in the About dialog, where it is the one place in the window
#: it is shown at a size worth looking at.
ABOUT_LOGO = 64

#: How wide the recording-as line beside the mark may get, in logical pixels.
#: The corner shares its row with the tab bar, so a long name has to elide
#: rather than push the tabs along; the whole of it stays in the tooltip.
AUTHOR_WIDTH = 220


def _log_file_path() -> str | None:
    """Return the file the root logger is writing to, or None.

    Asked of the handlers rather than assumed, so this reports where the log is
    going rather than where it usually goes.
    """
    for handler in logging.getLogger().handlers:
        filename = getattr(handler, "baseFilename", None)
        # os.devnull is what a test runner attaches when nothing asked for a
        # log, and reporting it would send someone to read an empty file.
        if filename and Path(filename) != Path(os.devnull):
            return str(filename)
    return None


#: How a heading is set apart from the rows under it.
#:
class CurationWindow(OperationRunner, OperationActions, LabelTab, QMainWindow):
    """Main window for reviewing and curating adaptive milling data.

    The three bases before ``QMainWindow`` each hold one subject's methods and
    no state of their own; everything they reach is built here. Their order
    does not matter -- no name is defined twice, which
    ``test_no_mixin_shadows_another`` is what keeps true.
    """

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
        self.setWindowTitle(self._name)
        self.resize(*DEFAULT_SIZE)
        application = QApplication.instance()
        assert application is not None, "a window exists only inside an application"
        with wheel_left_to_scrolling_once_built(application):
            self.settings = settings if settings is not None else user_settings()
            #: Whether the screen the window is on is already being watched. Set
            #: once: a signal connected again on every show would be delivered
            #: once per connection.
            self._watching_screen = False
            self.scan_bar = ScanBar()
            # The other tabs get their own Qt-free controllers, and are handed the
            # projects from the curation scan rather than scanning again: a second
            # scan would double the slowest thing the application does.
            # Handed the curation controller's config rather than loading its own:
            # which class is a crack has to be the same question the overlays
            # answer, and resolving it twice invites two answers.
            # One attribution, shared by everything that records. The curation
            # controller was built before the window and brought its own, so that
            # is the one the rest join rather than a sixth being made here.
            self._attribution = controller.attribution
            self.operations_controller = OperationsController(self._attribution)
            self.families_controller = FamiliesController(self._attribution)
            self.curate_panel = CuratePanel(
                controller, self.families_controller, self.operations_controller
            )
            self.curate_panel.typed_folder = self.scan_bar.path_edit.text
            # Located, not imported: importing torch to grey out a button would add
            # seconds to every start of a window that may never predict anything.
            # Nothing deeper is checked here or on the press -- a missing or broken
            # stack is reported by the child process that has to import it.
            from ...ml.environment import ml_stack_present

            self.curate_panel.operations_panel.set_prediction_available(ml_stack_present())
            #: Which remembered model the box is currently showing, as
            #: ``(series, modality)``. Held so that what is typed can be put back
            #: where it came from when the view moves on; see
            #: :meth:`_follow_model_context`.
            self._model_context = (
                controller.series, MODE_MODALITIES.get(controller.mode, MODALITIES[0])
            )
            self.curate_panel.operations_panel.set_model(
                self._remembered_model(self._model_context)
            )
            self.curate_panel.operations_panel.set_device(
                _setting_text(self.settings, SETTING_DEVICE)
            )

            # One answer to what each label holds, so a label one tab has read or
            # written is known the same way by the others.
            self.export_controller = ExportController(
                config=controller.config, known_labels=controller.known_labels
            )
            self.label_controller = LabelController(
                config=controller.config,
                attribution=self._attribution,
                known_labels=controller.known_labels,
            )
            self.metadata_controller = MetadataController(self._attribution)
            self.history_controller = HistoryController(self._attribution)
            self.export_panel = ExportPanel(self.export_controller)
            self.training_controller = TrainingController(
                self._attribution,
                # Read when asked, so a default changed in F2 is what the next reset gives.
                defaults=lambda mode: preferences.training_settings_from(
                    preferences.read(self.settings), mode
                ),
            )
            self._restore_training()
            self.training_panel = TrainingPanel(self.training_controller, config=controller.config)
            self.training_panel.run_in_background = lambda work, done, description: self._run_operation(
                description, work, done
            )
            self.training_panel.run_stoppable = self._run_stoppable
            self.training_panel.predictions.run_stoppable = self._run_stoppable
            self.training_panel.remembered.connect(self._remember_training)
            self.training_panel.run_path_requested.connect(
                lambda row: self._copy_path(self.training_panel.run_path(row), "Run")
            )
            self.training_panel.run_reveal_requested.connect(
                lambda row: self._reveal_path(self.training_panel.run_path(row))
            )
            self.label_panel = LabelPanel(
                self.label_controller, VersionController(self._attribution)
            )
            self.metadata_panel = MetadataPanel(self.metadata_controller)

            self._build_layout()
            self._connect()
            #: The folder the tree is on, when it is on one rather than a
            #: project. What the summary line describes; None means the frame.
            #: The shortcut list, built the first time F1 asks for it.
            self._keys = Shortcuts(self)
            self._apply_defaults()
            self._restore_layout()
            # Says which name is in effect from the moment the window opens, not
            # from the first time somebody changes it.
            self._adopt_author()

            # The Label tab has a loader of its own because the two prepare frames
            # at different sizes: proofreading needs every pixel, and reviewing
            # deliberately does not. Its speck counts queue behind its frame
            # preparation and share its token, so the frame being looked at is read
            # first and moving away abandons both.
            self.label_loader = BackgroundLoader(self.label_controller.cache)
            self.label_loader.worker.loaded.connect(self._on_label_loaded)
            self.label_loader.worker.counted.connect(self._on_speck_counted)

            # Connected once and left alone. Disconnecting a slot from inside the
            # emission of the signal it is handling is a connection list mutated
            # mid-emit, which the two bindings do not treat alike; what is pending
            # is tracked in a plain attribute instead.
            self.operations = BackgroundOperation()
            self.operations.worker.finished.connect(self._operation_finished)
            self.operations.worker.failed.connect(self._operation_failed)
            self.operations.worker.progressed.connect(self._operation_progressed)
            self._pending_operation: tuple | None = None
            self._busy: BusyDialog | None = None
            #: What a child process reported of the devices, asked once per session.
            #: None until asked; no devices is a machine that could not say.
            self._devices: DeviceProbe | None = None
            #: Whether the archive has been measured for the folder now open. One
            #: measurement per scan: the number cannot change without something
            #: writing, and everything that writes rescans.
            self._archive_measured = False

            remembered = _setting_text(self.settings, SETTING_SCAN_ROOT)
            if remembered:
                self.scan_bar.set_path(remembered)
                self._show_folder_in_title(Path(remembered))

            self.curate_panel.refresh()
            # Curate either way, and _tab_changed does not fire for the tab that is
            # already showing. A first run once opened on Add data, Curate having
            # nothing to curate and no cue about what comes first; it now offers
            # the folder to read where the frames would be, which is the cue that
            # was missing and the one thing to do next.
            self.curate_panel.take_focus()

    # -- construction ------------------------------------------------------

    def _build_layout(self) -> None:
        # Left to right in workflow order: describe the data, curate it,
        # proofread its labels, export it. Curate opens, because it is where the
        # repeated work starts and the outer two are visited once per collection.
        # A viewer holds Curate alone: every other tab exists to change files.
        self.tabs = QTabWidget()
        if not self.read_only:
            # Before its tabs are added: under a stylesheet, a widget taken in
            # has its whole subtree styled again, each tab's as well as its own.
            self.setCentralWidget(self.tabs)
            self.tabs.addTab(self.metadata_panel, "Add data")
        self.tabs.addTab(self.curate_panel, "Curate")
        if not self.read_only:
            self.tabs.addTab(self.label_panel, "Label")
            self.tabs.addTab(self.export_panel, "Export")
            self.tabs.addTab(self.training_panel, "Train")
        # Chosen before the signal is connected: _tab_changed switches the
        # keyboard shortcuts, which do not exist until _add_shortcuts runs.
        self.tabs.setCurrentWidget(self.curate_panel)
        self.tabs.currentChanged.connect(self._tab_changed)
        # In the tab row rather than in Curate: a scan serves every tab, and
        # this is the one place all four of them show. The mark ends the row,
        # and appears only here and on the banner -- the orange and green in it
        # are segmentation class colours, so a use beyond the logo and the
        # accent on controls would be read as data.
        self._corner = _corner(
            self.scan_bar, self.show_shortcuts, self.show_settings, self._show_about
        )
        if self.read_only:
            self.setCentralWidget(self._viewer_page())
            # Built all the same, so nothing that refreshes them has to know
            # about a viewer; held by a page never shown, so each has an owner
            # and none becomes a window of its own.
            unshown = QWidget(self)
            unshown.hide()
            for widget in (
                self.metadata_panel, self.label_panel, self.export_panel, self.training_panel,
            ):
                widget.setParent(unshown)
        else:
            self.tabs.setCornerWidget(self._corner, Qt.Corner.TopRightCorner)
        self._build_menus()

        # Parented to the window rather than to a tab: a scan can be asked for
        # from any of them, and what it holds up is the whole window.
        self.banner = ScanBanner(self)

    def _viewer_page(self) -> QWidget:
        """Return the viewer's one page: the corner's row above Curate.

        **The corner cannot stay a corner widget once the tab bar is hidden**:
        `QTabWidget` bounds a corner to the bar's height, and a hidden bar has
        none. The tabs stay, holding the one page, so nothing asking which tab
        is in front needs to know there is a viewer. Nothing records in a
        viewer, so the corner does not say who would.
        """
        self.tabs.tabBar().hide()
        self.tabs.setDocumentMode(True)
        self._corner.author.hide()
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._corner)
        column.addLayout(row)
        column.addWidget(self.tabs, stretch=1)
        return page

    @property
    def read_only(self) -> bool:
        """Whether this window is a viewer, as its controller was built."""
        return self.controller.read_only

    @property
    def _name(self) -> str:
        """What the window calls itself, which says whether it can edit."""
        return VIEWER_NAME if self.read_only else PRODUCT_NAME

    def _build_menus(self) -> None:
        """Build the menu bar.

        Deliberately short. Every action here is reachable in the window
        already, so this is not a second way to drive the application: it is
        where someone looks for the things a window is expected to have and
        this one had nowhere to put -- what the folder is, what version this
        is, where the log went, that the shortcut list exists at all, and
        adding the window to this computer's applications.

        macOS moves About and Quit into the application menu by their text, so
        neither is spelled to read well under File.

        **Every menu and action is kept in an attribute.** `addMenu` does not
        hand PySide6 ownership of what it returns, so a menu nothing holds is
        collected and its C++ object deleted underneath the bar that is still
        showing it -- the same shape as the connect-per-call trap in
        `docs/display.md`.
        """
        menus = self.menuBar()

        self._menus = {}
        self._menu_actions: dict[str, QAction] = {}
        file_menu = self._menus["file"] = menus.addMenu("&File")
        self._add_action(
            file_menu, "Open folder...", self.scan_bar.browse,
            QKeySequence.StandardKey.Open,
        )
        self._add_action(
            file_menu, "Rescan", self.rescan, QKeySequence.StandardKey.Refresh,
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu, "Settings...", self.show_settings, QKeySequence("F2")
        )
        label = menu_label()
        if label is not None:
            self._add_action(file_menu, label, self.show_launcher_dialog)
        file_menu.addSeparator()
        self._add_action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        help_menu = self._menus["help"] = menus.addMenu("&Help")
        # No shortcut on this one, deliberately: F1 is already an application
        # shortcut, and a second binding of one key is Qt's "ambiguous
        # shortcut", where neither fires reliably. The list teaches its own
        # key -- "F1  Show this list" is a row in it.
        self._add_action(help_menu, "Keyboard shortcuts", self.show_shortcuts)
        self._add_action(help_menu, f"About {PRODUCT_NAME}", self._show_about)

    def menu_actions(self) -> dict[str, QAction]:
        """Return each menu item by the words it shows, for F1 to read its key from.

        The list explains what the window offers, and the menu is part of that;
        read here rather than copied there, so a key changed in one place
        cannot leave the other teaching the old one.
        """
        return dict(self._menu_actions)

    def _add_action(self, menu, text: str, slot, shortcut=None) -> None:
        """Add one menu item, keeping the reference that stops it being freed."""
        action = menu.addAction(text)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        self._menu_actions[text] = action

    def _connect(self) -> None:
        self.label_panel.loads_needed.connect(self._schedule_label_loads)
        self.label_panel.speck_counts_needed.connect(self._sweep_specks)
        self.label_panel.save_requested.connect(self.save_labels)
        self.label_panel.restore_requested.connect(self.restore_label_version)
        self.label_panel.notice.connect(self._flash)
        self.scan_bar.scan_requested.connect(self.scan)
        curate = self.curate_panel
        curate.path_requested.connect(self._copy_path)
        curate.reveal_requested.connect(self._reveal_path)
        curate.view_changed.connect(self._follow_model_context)
        curate.series_shown.connect(self._cover_series)
        curate.image_panel.scan_requested.connect(self.scan)
        curate.image_panel.browse_requested.connect(self.scan_bar.browse)

        curate.project_panel.project_checked.connect(self._check_project)
        curate.project_panel.lamella_checked.connect(self._check_lamella)
        curate.project_panel.projects_check_all.connect(self._check_all_projects)
        curate.project_panel.lamellae_check_all.connect(self._check_all_lamellae)
        # Frames are picked out only for a change to act on, which a viewer
        # cannot make; unconnected, a Ctrl-click there picks nothing.
        if not self.read_only:
            curate.timeline.frame_toggled.connect(self._toggle_frame)
            curate.timeline.extend_started.connect(self._begin_extend)
            curate.timeline.extend_reached.connect(self._extend_frames)
        curate.state_panel.state_requested.connect(self.apply_state)
        curate.state_panel.target_changed.connect(self._set_target)
        curate.state_panel.family_scope_changed.connect(self._set_family_scope)

        curate.operations_panel.predict_requested.connect(self._predict)
        curate.operations_panel.devices_requested.connect(self._offer_devices)
        curate.operations_panel.convert_requested.connect(self._convert_predictions)
        curate.operations_panel.revert_requested.connect(self._revert)
        curate.operations_panel.specks_requested.connect(self._find_specks)

        self.metadata_panel.survey_requested.connect(self._inspect_layout)
        self.metadata_panel.init_requested.connect(self._add_projects)
        self.metadata_panel.compress_requested.connect(self._compress_labels)
        self.metadata_panel.savepoint_requested.connect(self._take_savepoint)
        self.metadata_panel.reclaim_requested.connect(self._reclaim_archive)
        self.metadata_panel.restore_requested.connect(self._restore_savepoint)
        self.export_panel.csv_check_requested.connect(self._check_csv)
        self.export_panel.measure_requested.connect(self._measure_labels)
        self.export_panel.labels_requested.connect(self._read_labels_for_view)

        self.export_panel.files_export_requested.connect(self._copy_files)
        self.export_panel.merge_requested.connect(self._merge_copy)
        self.export_panel.training_export_requested.connect(self._export_training)
        self.export_panel.image_metadata_export_requested.connect(self._export_image_metadata)
        self.export_panel.project_metadata_export_requested.connect(self._export_project_metadata)
        self.export_panel.breakdown_export_requested.connect(self._export_breakdown)
        # Editing metadata changes what the export view groups by, so it has to
        # be told rather than left showing the previous sample name.
        self.metadata_panel.metadata_changed.connect(self.export_panel.refresh)
        # A family change regroups images, which only a scan can show.
        curate.families_panel.families_changed.connect(self.rescan)
        curate.families_panel.may_change_families = self._may_change_families
        self.metadata_panel.project_path_requested.connect(
            lambda row: self._copy_path(self.metadata_controller.project_path(row), "Project")
        )
        self.metadata_panel.project_reveal_requested.connect(
            lambda row: self._reveal_path(self.metadata_controller.project_path(row))
        )

    def _show_folder_in_title(self, directory: Path) -> None:
        """Name the open collection in the window title.

        The folder rather than the whole path: a collection is recognised by
        what it is called, and a title long enough to be elided in the window
        switcher says nothing at all there.
        """
        self.setWindowTitle(f"{self._name}{TITLE_SEPARATOR}{directory.name}")

    def _show_about(self) -> None:
        """Say what this is, which version and how recent, where its log is going, and how to update it.

        The version and the log path are the two things anybody reporting a
        problem is asked for first, and neither was anywhere on screen. The log
        is read back from the handlers rather than from the path the command
        line would have chosen, so a run started with ``--log-file`` reports
        where its output actually went. The repository is a link the reader
        follows in their own browser; the window asks it nothing.
        """

        lines = [
            f"<b>{PRODUCT_NAME}</b>",
            f"Version {__version__}",
            f"Qt binding: {API_NAME} {QT_VERSION}",
        ]
        dated = version_date()
        if dated:
            lines.insert(2, dated)
        log = _log_file_path()
        lines.append(f"Log: {log}" if log else "Not logging to a file")
        link = f'<a href="{REPOSITORY_URL}">{REPOSITORY_URL}</a>'
        updating = [html.escape(step).replace(REPOSITORY_URL, link) for step in update_steps()]

        about = QMessageBox(self)
        about.setWindowTitle(f"About {PRODUCT_NAME}")
        about.setIconPixmap(logo_pixmap(ABOUT_LOGO))
        about.setText("<br>".join(lines))
        about.setInformativeText("<br><br>".join(updating))
        about.setTextFormat(Qt.TextFormat.RichText)
        about.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        about.exec_()

    # -- being the window in front -----------------------------------------

    def bring_to_front(self) -> None:
        """Raise this window and give it the keyboard.

        A window that is merely open is not one the user can drive. macOS keeps
        the launching application in front, and a window that is not in front
        receives no key presses and no pointer motion at all -- so F1 does
        nothing, no tooltip ever appears, and the brush outline sits where it
        was last drawn. Three faults that look unrelated, one cause, and it
        clears the moment anything is clicked, which is what made it look
        occasional rather than constant.
        """
        self.raise_()
        self.activateWindow()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Watch for the window being moved to another display.

        Undocking a laptop is exactly the case this is for: the window is put
        onto a screen smaller than the one it was sized for, with nothing else
        asking it to fit. The connection is made here rather than in the
        constructor because a window has no ``windowHandle`` until it is
        shown, and once rather than per show -- reconnecting would call
        :meth:`fit_to_screen` once per show of the window's whole life.
        """
        super().showEvent(event)
        handle = self.windowHandle()
        if handle is not None and not self._watching_screen:
            self._watching_screen = True
            handle.screenChanged.connect(lambda _screen: self.fit_to_screen())

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Record gaining and losing the keyboard, at debug level.

        Whether the window was in front is not otherwise recoverable after the
        fact, and it is the first thing to establish about any report that the
        keyboard or the pointer stopped working.
        """
        if event.type() == QEvent.Type.ActivationChange:
            logger.debug(
                "Window %s", "activated" if self.isActiveWindow() else "deactivated"
            )
        super().changeEvent(event)

    # -- remembering the layout --------------------------------------------

    def _splitters(self) -> dict[str, QSplitter]:
        """Return the resizable divisions, keyed by the name they persist under.

        Keyed by name rather than by position so that adding a splitter, or
        moving one, cannot silently restore the wrong sizes into it. A viewer
        has only the two it shows.
        """
        shown: dict[str, QSplitter] = {
            "curate": self.curate_panel.columns, "lists": self.curate_panel.project_panel.lists
        }
        if self.read_only:
            return shown
        return shown | {
            "add_data": self.metadata_panel.panes,
            "export": self.export_panel.columns,
            "export_filters": self.export_panel.filters,
            "train": self.training_panel.columns,
            "train_lists": self.training_panel.lists,
        }

    def _layout_key(self, name: str) -> str:
        """Return where the layout setting *name* is kept.

        **A viewer keeps its own layout**: its Curate splitter holds two panes
        to the full window's three, and either restoring the other's would
        put a divider somewhere nobody left it.
        """
        return f"{SETTING_VIEWER}/{name}" if self.read_only else name

    def _apply_defaults(self) -> None:
        """Open every control at what the settings say it should open at.

        Once, here, rather than in each panel's constructor: the store is the
        window's and the panels are built without one, which is also what lets
        a test build a panel with no settings at all and get the packaged
        answers. Read once and handed round, so eight controls cannot disagree
        about what was stored.
        """
        values = preferences.read(self.settings)
        for panel in (
            self.curate_panel.image_panel,
            self.curate_panel.operations_panel,
            self.label_panel,
            self.export_panel,
            # Last: its switches redraw the lists, which the others feed.
            self.curate_panel.project_panel,
        ):
            panel.apply_defaults(values)

    def _restore_layout(self) -> None:
        """Put the window back the size and shape it was left in, on this screen."""
        geometry = _setting_bytes(self.settings, self._layout_key(SETTING_GEOMETRY))
        if geometry:
            self.restoreGeometry(geometry)
        for name, splitter in self._splitters().items():
            state = _setting_bytes(
                self.settings, self._layout_key(f"{SETTING_SPLITTER}/{name}")
            )
            if state:
                splitter.restoreState(state)
        self.fit_to_screen()

    def fit_to_screen(self) -> None:
        """Pull the window inside the screen it is on.

        Two ordinary things put it outside one, and neither announces itself:
        :data:`DEFAULT_SIZE` is chosen for a desktop monitor, and
        ``restoreGeometry`` puts back a size and place recorded on whatever
        display the last session used without asking whether this one is the
        same. Either way the far edge lands off the screen, where there is no
        way to drag it back.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            self._fit_within(screen.availableGeometry())

    def _fit_within(self, room: QRect) -> None:
        """Resize and move the window to sit inside *room*.

        Split from :meth:`fit_to_screen` so the arithmetic can be tested
        against a screen this machine does not have: the fault being fixed
        only appears on a display smaller than the developer's.

        Qt clamps a resize to the window's own minimum, so a window that
        cannot be made to fit stays too large rather than being cut down to
        something unusable -- which is the case
        `test_no_tab_needs_more_room_than_the_smallest_screen_supported`
        exists to stop arising.
        """
        self.resize(
            min(self.width(), room.width() - SCREEN_MARGIN),
            min(self.height(), room.height() - SCREEN_MARGIN),
        )
        frame = self.frameGeometry()
        left = max(room.left(), min(frame.left(), room.right() - frame.width() + 1))
        top = max(room.top(), min(frame.top(), room.bottom() - frame.height() + 1))
        if (left, top) != (frame.left(), frame.top()):
            self.move(left, top)

    def _save_layout(self) -> None:
        """Record the window's size, its splitter positions and its model box.

        A viewer shows no model box, so it leaves the remembered model alone.
        """
        self.settings.setValue(self._layout_key(SETTING_GEOMETRY), self.saveGeometry())
        for name, splitter in self._splitters().items():
            self.settings.setValue(
                self._layout_key(f"{SETTING_SPLITTER}/{name}"), splitter.saveState()
            )
        if not self.read_only:
            self._remember_model()

    # -- the model to predict with -----------------------------------------

    def _model_key(self, context: tuple[str, str]) -> str:
        """Return the settings key holding the model for a series and beam."""
        series, modality = context
        return f"{SETTING_MODEL}/{series}/{modality}"

    def _remembered_model(self, context: tuple[str, str]) -> str:
        """Return the model last used for a series and beam.

        A slot nothing has written falls back to the single path kept before
        there were six, so an existing window opens on the model it always
        did rather than on an empty box.
        """
        key = self._model_key(context)
        if self.settings.contains(key):
            return _setting_text(self.settings, key)
        return _setting_text(self.settings, SETTING_LEGACY_MODEL)

    def _remember_model(self) -> None:
        """Store what is in the model box against the view it belongs to."""
        self.settings.setValue(
            self._model_key(self._model_context), self.curate_panel.operations_panel.model()
        )

    def _remember_device(self) -> str:
        """Store what is in the device box, and return it.

        One machine usually has one answer, so re-choosing it every session is
        noise. Beside :meth:`_remember_model` because both are the window's
        memory of what a prediction was last asked for.
        """
        device = self.curate_panel.operations_panel.device()
        self.settings.setValue(SETTING_DEVICE, device)
        return device

    def _restore_training(self) -> None:
        """Point the Train tab at what it was last pointed at, where that is still there."""
        directory = _setting_text(self.settings, SETTING_TRAINING_DIRECTORY)
        if directory and Path(directory).is_dir():
            self.training_controller.training_directory = Path(directory)
        weights = _setting_text(self.settings, SETTING_TRAINING_WEIGHTS)
        if weights and Path(weights).is_file():
            self.training_controller.base_weights = Path(weights)
        self.training_controller.device = (
            _setting_text(self.settings, SETTING_TRAINING_DEVICE) or "auto"
        )

    def _remember_training(self) -> None:
        """Store where the Train tab is pointed, for the next session."""
        controller = self.training_controller
        for key, value in (
            (SETTING_TRAINING_DIRECTORY, controller.training_directory),
            (SETTING_TRAINING_WEIGHTS, controller.base_weights),
        ):
            if value is not None:
                self.settings.setValue(key, str(value))
        self.settings.setValue(SETTING_TRAINING_DEVICE, controller.device)

    def _follow_model_context(self) -> None:
        """Point the model box at the series and beam now on screen.

        Called on every redraw of the Curate tab rather than from the controls,
        because the beam can change without anyone touching one: a timeline
        holding no image of it shows the other.

        The plot view leaves the box alone. It names no beam, so there is no
        slot to put what is typed into, and emptying the box on the way past a
        figure would lose a path the user had just chosen.
        """
        modality = MODE_MODALITIES.get(self.controller.mode)
        if modality is None:
            return
        context = (self.controller.series, modality)
        if context == self._model_context:
            return
        self._remember_model()
        self._model_context = context
        self.curate_panel.operations_panel.set_model(self._remembered_model(context))

    # -- the other tabs ----------------------------------------------------

    def _tab_changed(self, index: int) -> None:
        """Refresh a tab when it comes forward, and hand it its keys.

        Refreshing only on the way in keeps the cost off the curation view,
        which is where the keyboard-driven work happens.
        """
        widget = self.tabs.widget(index)
        if widget is self.curate_panel:
            self.curate_panel.came_forward()
        elif widget is self.export_panel:
            self._open_export()
        elif widget is self.training_panel:
            self.training_panel.refresh()
        elif widget is self.label_panel:
            self._point_label_tab()
            # Space pans, and it is read by the canvas as a key press, so the
            # canvas has to hold the keyboard before the pointer arrives.
            self.label_panel.focus_canvas()
        elif widget is self.metadata_panel:
            self.metadata_panel.refresh()
            self._measure_archive()
            # The arrow keys here are scoped to the project list, so they only
            # work once something in it has focus. Giving it focus on arrival
            # is what makes them work without a click first.
            self.metadata_panel.projects.setFocus()
        self._keys.follow_tab(self._current_tab())

    def _cover_series(self, series: tuple) -> None:
        """Cover in the Label tab the series Curate now covers.

        Pointed again if it is in front, the lamella it was on having possibly
        left the list.
        """
        self.label_controller.set_shown_series(series)
        if self.controller.projects and self.tabs.currentWidget() is self.label_panel:
            self._point_label_tab()

    def _open_export(self) -> None:
        """Read the image headers before the tab first draws, and nothing else.

        A few seconds over a real collection, read in worker processes, and it
        is what the acquisition filters and breakdowns are made of, so
        paying for it on arrival is what lets them simply work rather than sit
        there unusable with no way to tell why.

        The labels are *not* read here. Opening every one is ten times the cost
        and answers a different question, so it stays behind Measure labels
        where the user asks for it against the selection they care about.
        """
        if not self.export_controller.records or self.export_controller.metadata_loaded():
            self.export_panel.refresh()
            return
        images = self.export_controller.headers_to_read()
        progressed = self.operations.worker.progressed
        self._run_operation(
            f"Reading {images} image header(s)",
            lambda: self.export_controller.load_metadata(progressed.emit),
            lambda _read: self.export_panel.refresh(),
        )

    def _share_scan(self) -> None:
        """Hand the scanned projects to the other tabs."""
        projects = self.controller.projects
        index = self.controller.index
        self.export_controller.set_projects(
            projects,
            self.controller.start_path,
            index.headers if index is not None else None,
        )
        self.metadata_controller.set_projects(projects)
        self.label_controller.set_projects(projects)
        self.operations_controller.set_start_path(self.controller.start_path)
        self.history_controller.set_start_path(self.controller.start_path)
        self.training_controller.collection = self.controller.start_path
        self.operations_controller.set_projects(projects)
        self.metadata_panel.set_projects(self.metadata_controller.project_summaries())
        self.metadata_panel.set_root(self.controller.start_path)
        if self.controller.start_path is not None:
            self.metadata_panel.set_unindexed(
                *self.operations_controller.unindexed_summary(self.controller.start_path)
            )
        # Re-pointed rather than left holding records from the previous scan,
        # whose paths may no longer be where the files are. Redrawn if in
        # front, re-pointing having put away the label the canvas holds.
        self.label_controller.set_lamella(self.controller.lamella)
        if self.tabs.currentWidget() is self.label_panel:
            self.label_panel.refresh()
        # The archive can only have changed if something wrote, and everything
        # that writes lands here; so this is where a stale number is dropped.
        self._archive_measured = False

    # -- copying paths out of the window -----------------------------------

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

    def show_shortcuts(self) -> None:
        """List the keys for the tab now showing, or put the list away."""
        self._keys.toggle()

    def current_tab_name(self) -> str:
        """Return what the tab in front is called, as its label reads."""
        return self.tabs.tabText(self.tabs.currentIndex())

    def _current_tab(self) -> str:
        """Return the key of the tab now showing."""
        widget = self.tabs.currentWidget()
        for panel, tab in (
            (self.metadata_panel, TAB_ADD_DATA),
            (self.label_panel, TAB_LABEL),
            (self.export_panel, TAB_EXPORT),
            (self.training_panel, TAB_TRAIN),
        ):
            if widget is panel:
                return tab
        return TAB_CURATE

    # -- user actions ------------------------------------------------------

    def scan(self, path: str) -> None:
        """Scan *path* for projects, starting at the first one found.

        For a folder the reviewer chose. Re-reading the folder already open
        goes through :meth:`rescan`, which keeps their place.
        """
        if not path:
            return
        directory = Path(path)
        if not directory.is_dir():
            QMessageBox.warning(self, "Scan", f"Not a folder: {directory}")
            return

        if self._reading(
            lambda on_progress: self.controller.scan(directory, on_progress),
            f"Scanning {directory.name} for projects...",
            phase=f"Reading {directory.name}",
        ):
            self.scan_bar.set_path(str(directory))
            self.scan_bar.show_as_next_step(False)
            self.settings.setValue(SETTING_SCAN_ROOT, str(directory))
            self._show_folder_in_title(directory)

    def rescan(self) -> None:
        """Read the folder again, staying on the lamella being reviewed.

        What follows anything that moved files. Scanning again is how the
        window learns what an operation did; going back to where the reviewer
        was is what stops that costing them their place.

        Before anything has been read, this reads the folder the box holds --
        the one a previous session left there. The key is called Rescan and a
        key that does nothing is worse than one that is not there.
        """
        if self.controller.start_path is None:
            remembered = self.scan_bar.path_edit.text().strip()
            if remembered:
                self.scan(remembered)
            else:
                self._flash("No folder to read yet. Choose one with Browse.")
            return
        self._reading(
            lambda on_progress: self.controller.rescan(on_progress),
            "Reading the folder again...",
            phase=f"Reading {self.controller.start_path.name} again",
        )

    def _reading(
        self,
        read: Callable[[ScanProgress], object],
        announcement: str = "",
        phase: str = "",
    ) -> bool:
        """Run *read* behind the scan banner and redraw; return whether it worked.

        The read holds the interface thread for as long as it takes -- several
        seconds on a large collection -- so nothing else can paint, answer or
        report while it runs. The banner covers the window for the whole of it
        and is repainted from inside the read, turning the event loop once per
        update so the frame reaches the screen. It swallows the window's input
        and refuses to let it close meanwhile, so nothing a person does
        re-enters the read; timers and queued signals still run. The wait
        cursor stays, for the moment before the banner is up and for whatever
        is outside it.

        Args:
            read: Called with the progress callback to report through.
            announcement: Shown where the project summary goes while the read
                runs. Empty leaves whatever is there.
            phase: What the banner says is happening. Empty puts up no banner,
                for a read too short to be worth covering the window for.
        """
        failure: Exception | None = None
        if announcement:
            self.curate_panel.project_panel.announce(announcement)
        if phase:
            self.banner.begin(phase)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            read(self.banner.set_progress)
        except Exception as exc:  # reported, not swallowed
            logger.exception("Scan failed")
            failure = exc
        else:
            # Down before the lists are rebuilt: the banner says the reading is
            # happening, and what follows it is the window catching up.
            self.banner.end()
            self._share_scan()
            projects = self.controller.projects
            lamellae = sum(len(project.lamellae) for project in projects)
            self.scan_bar.show_summary(
                f"{counted(len(projects), 'project', 'projects')} · "
                f"{counted(lamellae, 'lamella', 'lamellae')}"
            )
            self.curate_panel.refresh(reload_lists=True)
        finally:
            self.banner.end()
            QApplication.restoreOverrideCursor()

        # Reported after the cursor is back: a modal dialog under an hourglass
        # looks like the window is still working on something.
        if failure is not None:
            QMessageBox.critical(self, "Scan failed", str(failure))
        return failure is None

    def _set_target(self, target: EditTarget) -> None:
        self.controller.set_target(target)
        self.curate_panel.refresh_state_panel()

    def _set_family_scope(self, scope: FamilyScope) -> None:
        self.controller.set_family_scope(scope)
        self.curate_panel.refresh_state_panel()

    # -- choosing what a change applies to ---------------------------------
    #
    # Every one of these ends in a refresh of the state panel, because the
    # reach line above the state buttons is the number that has to agree with
    # whatever was just ticked or picked out.

    def _check_project(self, row: int, checked: bool) -> None:
        self.controller.set_project_checked(row, checked)
        self._refresh_checks()
        self.curate_panel.refresh_state_panel()

    def _check_lamella(self, row: int, checked: bool) -> None:
        self.controller.set_lamella_checked(row, checked)
        self._refresh_checks()
        self.curate_panel.refresh_state_panel()

    def _check_all_projects(self, checked: bool) -> None:
        self.controller.check_all_projects(checked)
        self._refresh_checks()
        self.curate_panel.refresh_state_panel()

    def _check_all_lamellae(self, checked: bool) -> None:
        self.controller.check_all_lamellae(checked)
        self._refresh_checks()
        self.curate_panel.refresh_state_panel()

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
        timeline emits one of these per pixel. The lists are included because
        picking a frame out clears their ticks.
        """
        self.curate_panel.refresh_timeline()
        self._refresh_checks()
        self.curate_panel.refresh_state_panel()

    def _refresh_checks(self) -> None:
        """Put both lists' ticks back without rebuilding them.

        Rebuilding would reset the row the user is on, and ticking a row is not
        a change of what is being looked at. Both lists every time: a tick at
        one level clears the other, and that has to be visible.
        """
        self.curate_panel.project_panel.show_checks(
            self.controller.project_checks(), self.controller.lamella_checks()
        )

    def apply_state(self, state: RecordState) -> None:
        """Declare the targeted records to be in *state*, confirming where required.

        Two confirmations can apply, and they are asked in order of how much is
        at stake: first whether the user really means to change more than the
        current frame, then whether proofread work may leave the corrected
        state.
        """
        if self.controller.editing_enabled and not self.controller.selection.is_empty:
            if not self._confirm_bulk(state):
                return
        if self.controller.editing_enabled and state == "corrected":
            if not self._confirm_unannotated():
                return
        try:
            changed = self.controller.apply_state(state)
        except EditingDisabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing before changing the state of an image.",
            )
            return
        except PermissionError:
            if not self._confirm_corrected(state):
                self.curate_panel.refresh_state_panel()
                return
            try:
                changed = self.controller.apply_state(state, confirm_corrected=True)
            except StaleState as exc:
                self._report_stale(exc)
                return
            except (UnmigratedStage, OSError) as exc:
                self._report_failure(exc)
                return
        except StaleState as exc:
            self._report_stale(exc)
            return
        except (UnmigratedStage, OSError) as exc:
            self._report_failure(exc)
            return

        if changed:
            frame = self.controller.frame
            where = self.controller.describe_selection()
            if self.controller.selection.is_empty and frame is not None:
                where = frame.display_name
            self.curate_panel.state_panel.show_last_change(
                f"{where}: {changed} image(s) -> {state} ({self.controller.target})"
            )
        # Rebuild the lists too: a state change is exactly what moves a lamella
        # or project between "not started", "in progress" and "done", so leaving
        # the status chips alone here would leave them stale at the one moment
        # they are meant to report on. Recomputing every status costs ~60ms on a
        # 21-project collection, against a change that has just been written.
        self.curate_panel.refresh(reload_lists=True)

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
                f"<p>{shared} of them share a label with your selection and "
                f"change with it.</p>"
            )
        if unchanged:
            message += f"<p>{unchanged} of them are already {state}.</p>"

        answer = QMessageBox.warning(
            self, "Change many images?", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_unannotated(self) -> bool:
        """Ask before calling images corrected when their label holds nothing.

        Corrected is the claim that somebody proofread these labels, and it is
        the claim a training export trusts. An image that reaches it without an
        annotation is skipped by every export in silence, so the place to catch
        it is where the claim is made rather than months later in a CSV that is
        quietly short.

        Named by project and lamella, not by filename: the point of the dialog
        is to be able to go and fix them, and image filenames repeat across
        lamellae.
        """
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Reads each label once per scan; the timeline has usually paid for
            # the lamella on screen already.
            readiness = self.controller.label_readiness()
        finally:
            QApplication.restoreOverrideCursor()
        if not readiness:
            return True

        rows = [
            f"{record.project or '?'} / {record.lamella or '?'} / {record.image_path.name}"
            for record in readiness.records
        ]
        shown = "\n".join(rows[:MAX_LISTED_FILES])
        if len(rows) > MAX_LISTED_FILES:
            shown += f"\n... and {len(rows) - MAX_LISTED_FILES} more"

        answer = QMessageBox.warning(
            self,
            "Mark as corrected without an annotation?",
            f"{readiness.summary()}. A training export skips images whose label is empty:\n"
            "\n"
            f"{shown}\n"
            "\n"
            "Mark them corrected anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

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
            "Change the state of corrected images?",
            f"{len(names)} of these images are marked corrected, which means they were proofread:\n"
            "\n"
            f"{shown}\n"
            "\n"
            f"Set them to '{state}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _report_failure(self, exc: Exception) -> None:
        logger.warning("State change refused: %s", exc)
        QMessageBox.critical(self, "Could not change state", str(exc))

    def _report_stale(self, exc: Exception) -> None:
        """Say that the collection moved underneath the window, and re-read it.

        Nothing was written, so there is nothing to undo -- but what is on
        screen is now describing a collection that has changed, and the next
        thing the reviewer does would be decided from it. Scanning again is
        the whole remedy, so it happens rather than being offered.
        """
        logger.warning("State change refused: %s", exc)
        QMessageBox.warning(
            self, "This collection was changed elsewhere",
            f"{exc}\n\nNothing was changed. Reading the folder again.",
        )
        self.rescan()

    # -- settings, launchers and who is recording ----------------------------

    def show_launcher_dialog(self) -> None:
        """Offer to add the window or the viewer to this computer's applications.

        Opens on the one this window is, which is how someone at a viewer makes
        its launcher without a terminal. It writes into the user's own menus,
        never into a collection.
        """
        try:
            computer = this_computer()
        except LauncherRefused as error:
            QMessageBox.warning(self, LAUNCHER_TITLE, str(error))
            return
        LauncherDialog(self, computer, VIEWER if self.read_only else WINDOW).exec_()

    def show_settings(self) -> None:
        """Open the settings that belong to this machine rather than the data.

        Kept under the user's home directory: who is at this keyboard is a
        property of the machine, and a name written into a collection travels
        with it. **A viewer's leaves the name alone**, asking none.
        """
        stored = preferences.read(self.settings)
        dialog = SettingsDialog(
            self, stored, preview_accent=self.show_accent, viewer=self.read_only
        )
        if dialog.exec_() != QDialog.DialogCode.Accepted:
            return
        wanted = dialog.author
        if wanted is None:
            preferences.write(self.settings, dialog.values)
            self._flash("Settings kept")
            return
        try:
            if wanted:
                set_author(wanted)
            else:
                clear_author()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, AUTHOR_TITLE, str(error))
            # Nothing was kept, so neither is the accent the dialog previewed.
            self.show_accent(preferences.accent_from(stored))
            return
        # After the name, and only if it was written: a failure there is
        # reported and returns, and preferences saved past it would leave the
        # dialog half applied with nothing saying which half.
        preferences.write(self.settings, dialog.values)
        self._adopt_author()
        self._flash(author_summary(wanted or None))

    def show_accent(self, key: str) -> None:
        """Draw the window in the accent *key*, without storing it.

        The palette and stylesheet reach every widget; the grid overview's
        marks are scene items holding the colour they were drawn in, so they
        are drawn again.
        """
        application = QApplication.instance()
        assert application is not None, "a window exists only inside an application"
        apply_accent(application, ACCENTS[key])
        self.curate_panel.image_panel.grid_view.refresh_accent()

    def _adopt_author(self) -> None:
        """Have everything that records pick the new name up at once.

        One assignment, because every controller records through the same
        :class:`~am_segmentation_manager.gui.attribution.Attribution`. There is
        no list of who to tell, and so nothing for a new controller to fall off.
        """
        recorder = self._attribution.adopt()
        line = author_summary(recorder.by)
        # Elided against a fixed budget rather than by `ElidingLabel`, which
        # sizes itself to whatever it has been given: in the corner nothing
        # else decides its width, so it would take its hint from the text it
        # had already shortened and shrink on every pass. The end of the line
        # is what goes, the name starting where it is most worth reading.
        self._corner.author.setText(
            self._corner.author.fontMetrics().elidedText(
                line, Qt.TextElideMode.ElideRight, AUTHOR_WIDTH
            )
        )
        self._corner.author.setToolTip(
            author_summary(recorder.by, asked=True) + " Press F2 to change it."
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._may_discard_label_edits():
            event.ignore()
            return
        self._save_layout()
        self.curate_panel.loader.stop()
        self.label_loader.stop()
        self.operations.stop()
        super().closeEvent(event)
