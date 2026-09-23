"""The window: four tabs, the controllers behind them, and what joins them.

*Add data*, *Curate*, *Label* and *Export* each have a controller holding no Qt;
this module holds no decisions of its own, asking a controller what to show and
telling it what the user did. It builds and owns the controllers, the Curate tab
and the window itself. Three subjects are mixins beside it --
:mod:`.operation_runner`, :mod:`.operation_actions` and :mod:`.label_tab` -- and
the keys are :mod:`.shortcuts`, held rather than inherited. Why the window looks
and behaves as it does, and every Qt trap in it, is ``docs/display.md``.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from qtpy import API_NAME, QT_VERSION
from qtpy.QtCore import QByteArray, QEvent, QRect, QSettings, Qt, QUrl, Signal
from qtpy.QtGui import QAction, QCursor, QDesktopServices, QKeySequence
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
from ...identity import PRODUCT_NAME
from ...layout import MODALITIES
from ...operations.predict_process import DeviceProbe
from ...operations.selection import (
    RecordState,
    StaleState,
    UnmigratedStage,
    current_state,
)
from .. import preferences
from ..controller import (
    MODE_MODALITIES,
    CurationController,
    DisplayMode,
    EditingDisabled,
    EditTarget,
    ScanProgress,
)
from ..export_controller import ExportController
from ..history_controller import HistoryController
from ..label_controller import LabelController
from ..loader import BackgroundLoader
from ..metadata_controller import MetadataController
from ..operations_controller import OperationsController
from ..selection import counted
from ..version_controller import VersionController
from ..worker import BackgroundOperation
from .branding import logo_pixmap
from .busy_dialog import BusyDialog
from .common import MAX_LISTED_FILES, vertical_rule
from .controls import GripSplitter
from .export_panel import ExportPanel
from .frame_strip import CURATE_BAND, FrameTimeline
from .icons import settings_icon
from .image_panel import ImagePanel
from .label_panel import LabelPanel
from .label_tab import LabelTab
from .metadata_panel import MetadataPanel
from .operation_actions import OperationActions
from .operation_runner import OperationRunner
from .operations_panel import OperationsPanel
from .project_panel import ProjectPanel
from .scan_banner import ScanBanner
from .scan_bar import ScanBar
from .settings_dialog import SettingsDialog, author_summary
from .shortcuts import (
    TAB_ADD_DATA,
    TAB_CURATE,
    TAB_EXPORT,
    TAB_LABEL,
    Shortcuts,
)
from .state_panel import StatePanel
from .theme import ACCENTS, COMPACT_BUTTON, MUTED, apply_accent

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
SETTING_GEOMETRY = "geometry"
SETTING_SPLITTER = "splitter"

AUTHOR_TITLE = "Settings"

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

    mark: QLabel
    author: QLabel
    settings_button: QToolButton
    help_button: QToolButton


def _corner(scan_bar: QWidget, on_help, on_settings) -> CornerBar:
    """Return the tab row's corner: the folder, who is recording, and the mark.

    **Three runs with a rule between each**, read from what is being worked on
    to whose window this is. The folder first, nearest the tabs it serves; then
    this machine's own state and the two buttons for what nothing else on
    screen mentions, F2 and F1; then the mark at the far end. **The mark
    without the words**: the title bar carries the product name on Windows and
    Linux and the menu bar does on macOS, and printed here as well it read as
    the window saying its own name twice.

    A widget rather than a `QLabel`, a label showing either a pixmap or text
    and this needing both. **The mark and the name are not clickable**: they
    say whose window this is and answer no question.

    Args:
        scan_bar: The path, Browse and Scan, which the window builds and owns.
        on_help: Called when the help button is pressed.
        on_settings: Called when the settings button is pressed.
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
    settings_button.setToolTip("Settings for this machine, including that name (F2)")
    settings_button.clicked.connect(on_settings)
    layout.addWidget(settings_button)

    # A one-character caption, which the stylesheet's button padding would
    # otherwise clip blank; see `theme.COMPACT_BUTTON`.
    help_button = QToolButton()
    help_button.setObjectName(COMPACT_BUTTON)
    help_button.setText("?")
    help_button.setToolTip("Show the keyboard shortcuts for this tab (F1)")
    help_button.clicked.connect(on_help)
    layout.addWidget(help_button)
    rule()

    mark = QLabel()
    mark.setPixmap(logo_pixmap(CORNER_LOGO))
    mark.setToolTip(PRODUCT_NAME)
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

    #: Emitted to the background loader thread; see :class:`BackgroundLoader`.
    request_load = Signal(list, int)

    #: Emitted to the Label tab's own loader thread. Separate from
    #: :attr:`request_load` because the two prepare frames at different sizes:
    #: proofreading needs every pixel, and reviewing deliberately does not.
    request_label_load = Signal(list, int)

    #: Emitted to that same thread with the labels whose specks are still
    #: uncounted. Behind the frame preparation and sharing its token, so the
    #: frame being looked at is read first and moving away abandons both.
    request_speck_counts = Signal(list, int, int)

    #: Emitted to the operations thread with the callable to run; see
    #: :class:`~am_segmentation_manager.gui.worker.BackgroundOperation`.
    request_operation = Signal(object)

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
        self.setWindowTitle(PRODUCT_NAME)
        self.resize(*DEFAULT_SIZE)

        self.settings = settings if settings is not None else user_settings()
        #: Whether the screen the window is on is already being watched. Set
        #: once: a signal connected again on every show would be delivered
        #: once per connection.
        self._watching_screen = False
        self.project_panel = ProjectPanel()
        self.scan_bar = ScanBar()
        self.image_panel = ImagePanel(controller.config)
        self.state_panel = StatePanel()
        self.operations_panel = OperationsPanel()
        # Located, not imported: importing torch to grey out a button would add
        # seconds to every start of a window that may never predict anything.
        # Nothing deeper is checked here or on the press -- a missing or broken
        # stack is reported by the child process that has to import it.
        from ...ml.environment import ml_stack_present

        self.operations_panel.set_prediction_available(ml_stack_present())
        #: Which remembered model the box is currently showing, as
        #: ``(series, modality)``. Held so that what is typed can be put back
        #: where it came from when the view moves on; see
        #: :meth:`_follow_model_context`.
        self._model_context = (
            controller.series, MODE_MODALITIES.get(controller.mode, MODALITIES[0])
        )
        self.operations_panel.set_model(self._remembered_model(self._model_context))
        self.operations_panel.set_device(
            _setting_text(self.settings, SETTING_DEVICE)
        )

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
        self.operations_controller = OperationsController(self._attribution)
        self.history_controller = HistoryController(self._attribution)
        self.export_panel = ExportPanel(self.export_controller)
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

        self.loader = BackgroundLoader(controller.cache)
        self.request_load.connect(self.loader.worker.load_batch)
        self.loader.worker.loaded.connect(self._on_loaded)

        self.label_loader = BackgroundLoader(self.label_controller.cache)
        self.request_label_load.connect(self.label_loader.worker.load_batch)
        self.label_loader.worker.loaded.connect(self._on_label_loaded)
        self.request_speck_counts.connect(self.label_loader.worker.count_specks)
        self.label_loader.worker.counted.connect(self._on_speck_counted)

        # Connected once and left alone. Disconnecting a slot from inside the
        # emission of the signal it is handling is a connection list mutated
        # mid-emit, which the two bindings do not treat alike; what is pending
        # is tracked in a plain attribute instead.
        self.operations = BackgroundOperation()
        self.request_operation.connect(self.operations.worker.run)
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

        self.refresh()
        # Curate either way, and _tab_changed does not fire for the tab that is
        # already showing. A first run once opened on Add data, Curate having
        # nothing to curate and no cue about what comes first; it now offers
        # the folder to read where the frames would be, which is the cue that
        # was missing and the one thing to do next.
        self._focus_curate()

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
        # where nothing else is going to be clicked by accident.
        right = QWidget()
        right_column = QVBoxLayout(right)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.addWidget(self.state_panel, stretch=1)
        right_column.addWidget(self.operations_panel)

        self.curate = GripSplitter(Qt.Orientation.Horizontal)
        self.curate.addWidget(self.project_panel)
        self.curate.addWidget(centre)
        self.curate.addWidget(right)
        self.curate.setStretchFactor(1, 1)
        # The left column opens wide enough for the project box's four
        # buttons to sit on one line; narrower and they wrap, which is the
        # reader's call rather than the default.
        self.curate.setSizes([330, 1050, 260])

        # Left to right in workflow order: describe the data, curate it,
        # proofread its labels, export it. Curate opens, because it is where the
        # repeated work starts and the outer two are visited once per collection.
        self.tabs = QTabWidget()
        self.tabs.addTab(self.metadata_panel, "Add data")
        self.tabs.addTab(self.curate, "Curate")
        self.tabs.addTab(self.label_panel, "Label")
        self.tabs.addTab(self.export_panel, "Export")
        # Chosen before the signal is connected: _tab_changed switches the
        # keyboard shortcuts, which do not exist until _add_shortcuts runs.
        self.tabs.setCurrentWidget(self.curate)
        self.tabs.currentChanged.connect(self._tab_changed)
        # In the tab row rather than in Curate: a scan serves every tab, and
        # this is the one place all four of them show. The mark ends the row,
        # and appears only here and on the banner -- the orange and green in it
        # are segmentation class colours, so a use beyond the logo and the
        # accent on controls would be read as data.
        self._corner = _corner(self.scan_bar, self.show_shortcuts, self.show_settings)
        self.tabs.setCornerWidget(self._corner, Qt.Corner.TopRightCorner)
        self.setCentralWidget(self.tabs)
        self._build_menus()

        # Parented to the window rather than to a tab: a scan can be asked for
        # from any of them, and what it holds up is the whole window.
        self.banner = ScanBanner(self)

    def _build_menus(self) -> None:
        """Build the menu bar.

        Deliberately short. Every action here is reachable in the window
        already, so this is not a second way to drive the application: it is
        where someone looks for the things a window is expected to have and
        this one had nowhere to put -- what the folder is, what version this
        is, where the log went, and that the shortcut list exists at all.

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

        self.timeline = FrameTimeline(band=CURATE_BAND)
        self.timeline.frame_selected.connect(self._seek_frame)
        self.timeline.frame_toggled.connect(self._toggle_frame)
        self.timeline.extend_started.connect(self._begin_extend)
        self.timeline.extend_reached.connect(self._extend_frames)
        row.addWidget(self.timeline, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(70)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.frame_label.setToolTip("Frame position within this lamella")
        row.addWidget(self.frame_label)

        return row

    def _connect(self) -> None:
        self.label_panel.loads_needed.connect(self._schedule_label_loads)
        self.label_panel.speck_counts_needed.connect(self._sweep_specks)
        self.label_panel.save_requested.connect(self.save_labels)
        self.label_panel.restore_requested.connect(self.restore_label_version)
        self.scan_bar.scan_requested.connect(self.scan)
        self.project_panel.project_selected.connect(self._select_project)
        self.project_panel.folder_selected.connect(self._select_folder)
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
        self.project_panel.shown_series_changed.connect(self._show_series)

        self.image_panel.scan_requested.connect(self.scan)
        self.image_panel.browse_requested.connect(self.scan_bar.browse)
        self.image_panel.mode_changed.connect(self._set_mode)
        self.image_panel.variant_changed.connect(self._set_variant)
        self.image_panel.overview_chosen.connect(self._choose_overview)
        self.image_panel.series_changed.connect(self._set_series)
        self.image_panel.overlay_toggled.connect(self._set_overlay)
        self.image_panel.image_toggled.connect(self._set_image_shown)
        self.image_panel.alpha_changed.connect(lambda _alpha: self._render())
        self.image_panel.intensity_range_changed.connect(
            lambda _low, _high: self._render()
        )
        self.image_panel.path_requested.connect(
            lambda: self._copy_path(self._displayed_path(), "Image")
        )
        self.image_panel.reveal_requested.connect(
            lambda: self._reveal_path(self._displayed_path())
        )

        self.state_panel.editing_toggled.connect(self.controller.set_editing_enabled)
        self.state_panel.state_requested.connect(self.apply_state)
        self.state_panel.target_changed.connect(self._set_target)

        self.operations_panel.predict_requested.connect(self._predict)
        self.operations_panel.devices_requested.connect(self._offer_devices)
        self.operations_panel.convert_requested.connect(self._convert_predictions)
        self.operations_panel.origin_changed.connect(
            lambda _source: self._refresh_prediction_summary()
        )
        self.operations_panel.revert_requested.connect(self._revert)
        self.operations_panel.specks_requested.connect(self._find_specks)

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
        self.setWindowTitle(f"{PRODUCT_NAME}{TITLE_SEPARATOR}{directory.name}")

    def _show_about(self) -> None:
        """Say what this is, which version, and where its log is going.

        The version and the log path are the two things anybody reporting a
        problem is asked for first, and neither was anywhere on screen. The log
        is read back from the handlers rather than from the path the command
        line would have chosen, so a run started with ``--log-file`` reports
        where its output actually went.
        """

        lines = [
            f"<b>{PRODUCT_NAME}</b>",
            f"Version {__version__}",
            f"Qt binding: {API_NAME} {QT_VERSION}",
        ]
        log = _log_file_path()
        lines.append(f"Log: {log}" if log else "Not logging to a file")

        about = QMessageBox(self)
        about.setWindowTitle(f"About {PRODUCT_NAME}")
        about.setIconPixmap(logo_pixmap(ABOUT_LOGO))
        about.setText("<br>".join(lines))
        about.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        moving one, cannot silently restore the wrong sizes into it.
        """
        return {
            "curate": self.curate,
            "lists": self.project_panel.lists,
            "add_data": self.metadata_panel.panes,
            "export": self.export_panel.columns,
            "export_filters": self.export_panel.filters,
        }

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
            self.image_panel,
            self.operations_panel,
            self.label_panel,
            self.export_panel,
            # Last: its switches redraw the lists, which the others feed.
            self.project_panel,
        ):
            panel.apply_defaults(values)

    def _restore_layout(self) -> None:
        """Put the window back the size and shape it was left in, on this screen."""
        geometry = _setting_bytes(self.settings, SETTING_GEOMETRY)
        if geometry:
            self.restoreGeometry(geometry)
        for name, splitter in self._splitters().items():
            state = _setting_bytes(self.settings, f"{SETTING_SPLITTER}/{name}")
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
        """Record the window's size, its splitter positions and its model box."""
        self.settings.setValue(SETTING_GEOMETRY, self.saveGeometry())
        for name, splitter in self._splitters().items():
            self.settings.setValue(f"{SETTING_SPLITTER}/{name}", splitter.saveState())
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
            self._model_key(self._model_context), self.operations_panel.model()
        )

    def _remember_device(self) -> str:
        """Store what is in the device box, and return it.

        One machine usually has one answer, so re-choosing it every session is
        noise. Beside :meth:`_remember_model` because both are the window's
        memory of what a prediction was last asked for.
        """
        device = self.operations_panel.device()
        self.settings.setValue(SETTING_DEVICE, device)
        return device

    def _follow_model_context(self) -> None:
        """Point the model box at the series and beam now on screen.

        Called from :meth:`refresh` rather than from the controls, because the
        beam can change without anyone touching one: a timeline holding no
        image of it shows the other.

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
        self.operations_panel.set_model(self._remembered_model(context))

    # -- the other tabs ----------------------------------------------------

    def _tab_changed(self, index: int) -> None:
        """Refresh a tab when it comes forward, and hand it its keys.

        Refreshing only on the way in keeps the cost off the curation view,
        which is where the keyboard-driven work happens.
        """
        widget = self.tabs.widget(index)
        if widget is self.curate:
            self._focus_curate()
        elif widget is self.export_panel:
            self._open_export()
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
        self._run_operation(
            f"Reading {images} image header(s)",
            self.export_controller.load_metadata,
            lambda _read: self.export_panel.refresh(),
        )

    def _focus_curate(self) -> None:
        """Put the keyboard on the project list, not the scan box.

        The scan box is the first focusable widget in the tab, and a focused
        ``QLineEdit`` claims Left and Right as shortcut overrides -- so the
        keys that step through frames never reach the window while it holds
        focus, and the tab opens unable to be driven by the keys it documents.
        The project list claims no arrow key of its own, so the window's own
        bindings all still fire with it focused.
        """
        self.project_panel.project_list.setFocus()

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
        self.operations_controller.set_projects(projects)
        self.metadata_panel.set_projects(self.metadata_controller.project_summaries())
        self.metadata_panel.set_root(self.controller.start_path)
        if self.controller.start_path is not None:
            self.metadata_panel.set_unindexed(
                *self.operations_controller.unindexed_summary(self.controller.start_path)
            )
        # Re-pointed rather than left holding records from the previous scan,
        # whose paths may no longer be where the files are.
        self.label_controller.set_lamella(self.controller.lamella)
        # The archive can only have changed if something wrote, and everything
        # that writes lands here; so this is where a stale number is dropped.
        self._archive_measured = False

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
                self._flash("No folder to read yet -- choose one with Browse")
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
        update so the frame reaches the screen; it swallows every input event
        meanwhile, which is what stops anything being re-entered on the way.
        The wait cursor stays, for the
        moment before the banner is up and for whatever is outside it.

        Args:
            read: Called with the progress callback to report through.
            announcement: Shown where the project summary goes while the read
                runs. Empty leaves whatever is there.
            phase: What the banner says is happening. Empty puts up no banner,
                for a read too short to be worth covering the window for.
        """
        failure: Exception | None = None
        if announcement:
            self.project_panel.announce(announcement)
        if phase:
            self.banner.begin(phase)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            read(self.banner.set_progress)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
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
            self.refresh(reload_lists=True)
        finally:
            self.banner.end()
            QApplication.restoreOverrideCursor()

        # Reported after the cursor is back: a modal dialog under an hourglass
        # looks like the window is still working on something.
        if failure is not None:
            QMessageBox.critical(self, "Scan failed", str(failure))
        return failure is None

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
        """Cover only *series*, in both tabs, both lists and every chip.

        Nothing is redrawn before the first scan: there is nothing listed, and
        the switches first announce themselves while the window is still being
        built, before the frame loader a redraw schedules exists. The Label tab
        is pointed again if it is in front, the lamella it was on having
        possibly left the list.
        """
        self.controller.set_shown_series(series)
        self.label_controller.set_shown_series(series)
        self.project_panel.set_chip_series(series)
        if not self.controller.projects:
            return
        self.refresh(reload_lists=True)
        if self.tabs.currentWidget() is self.label_panel:
            self._point_label_tab()

    def _seek_frame(self, value: int) -> None:
        if value != self.controller.frame_position:
            self.controller.select_frame(value)
            self.refresh()

    def _step_frame(self, delta: int) -> None:
        self.controller.step_frame(delta)
        self.refresh()

    def _step_lamella(self, delta: int) -> None:
        """Move within the project on screen, which the project list outlives."""
        self.controller.step_lamella(delta)
        self._refresh_lamellae()
        self.refresh()

    def _set_mode(self, mode: DisplayMode, from_user: bool = False) -> None:
        if from_user and mode not in self.controller.available_modes():
            return
        self.controller.set_mode(mode)
        self.image_panel.set_mode(mode)
        # set_mode may have pointed the edit target at the new modality.
        self.state_panel.set_target(self.controller.target)
        self.refresh()

    def _set_overlay(self, show: bool) -> None:
        self.controller.show_overlay = show
        self._render()

    def _set_image_shown(self, show: bool) -> None:
        self.controller.show_image = show
        self._render()

    def _toggle_overlay(self) -> None:
        self.image_panel.overlay_check.toggle()

    def _set_target(self, target: EditTarget) -> None:
        self.controller.set_target(target)
        self._refresh_state_panel()

    # -- choosing what a change applies to ---------------------------------
    #
    # Every one of these ends in a refresh of the state panel, because the
    # reach line above the state buttons is the number that has to agree with
    # whatever was just ticked or picked out.

    def _check_project(self, row: int, checked: bool) -> None:
        self.controller.set_project_checked(row, checked)
        self._refresh_checks()
        self._refresh_state_panel()

    def _check_lamella(self, row: int, checked: bool) -> None:
        self.controller.set_lamella_checked(row, checked)
        self._refresh_checks()
        self._refresh_state_panel()

    def _check_all_projects(self, checked: bool) -> None:
        self.controller.check_all_projects(checked)
        self._refresh_checks()
        self._refresh_state_panel()

    def _check_all_lamellae(self, checked: bool) -> None:
        self.controller.check_all_lamellae(checked)
        self._refresh_checks()
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
        timeline emits one of these per pixel. The lists are included because
        picking a frame out clears their ticks.
        """
        self._refresh_timeline()
        self._refresh_checks()
        self._refresh_state_panel()

    def _refresh_checks(self) -> None:
        """Put both lists' ticks back without rebuilding them.

        Rebuilding would reset the row the user is on, and ticking a row is not
        a change of what is being looked at. Both lists every time: a tick at
        one level clears the other, and that has to be visible.
        """
        self.project_panel.show_checks(
            self.controller.project_checks(), self.controller.lamella_checks()
        )

    def _set_series(self, series: str) -> None:
        """Show a different series' timeline.

        More than a view change: it decides which frames exist, so the whole
        window is brought back in step rather than only redrawn.
        """
        self.controller.set_series(series)
        self.refresh()

    def _set_variant(self, variant: str) -> None:
        """Show a different imaging condition.

        A view change only: which condition is on screen never decides what a
        state change moves, because the change follows the label.
        """
        self.controller.set_variant(variant or None)
        self.refresh()

    def _choose_overview(self, index: int) -> None:
        """Show another of the project's grid overviews."""
        self.controller.grid.choose(index)
        self._render()

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
                "Tick 'Enable editing' before changing the state of an image.",
            )
            return
        except PermissionError:
            if not self._confirm_corrected(state):
                self._refresh_state_panel()
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
            self.state_panel.show_last_change(
                f"{where}: {changed} image(s) -> {state} ({self.controller.target})"
            )
        # Rebuild the lists too: a state change is exactly what moves a lamella
        # or project between "not started", "in progress" and "done", so leaving
        # the status chips alone here would leave them stale at the one moment
        # they are meant to report on. Recomputing every status costs ~60ms on a
        # 21-project collection, against a change that has just been written.
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
            f"{readiness.summary()}. Corrected means the label was proofread, "
            f"and a training export skips an image whose label holds "
            f"nothing:\n\n{shown}\n\nMark them corrected anyway?",
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
            "Move corrected data?",
            f"{len(names)} of these files are in the corrected state, which "
            f"holds manually proofread work:\n\n{shown}\n\nMove them to '{state}'?",
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
            self, "Somebody else changed this collection",
            f"{exc}\n\nNothing was changed. Reading the folder again now.",
        )
        self.rescan()

    # -- rendering ---------------------------------------------------------

    def refresh(self, *, reload_lists: bool = False) -> None:
        """Bring the whole window back in step with the controller."""
        if reload_lists:
            self._refresh_lists()
        # Outside the reload: the summary describes the lamella being looked
        # at, so it has to follow the cursor rather than only a rescan.
        self.project_panel.show_summary(self._summary_text())
        self._follow_model_context()
        self._refresh_timeline()
        self._render()
        self._refresh_state_panel()
        self._schedule_loads()

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

    def _refresh_timeline(self) -> None:
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
            self.image_panel.set_variants([])
            if self.controller.start_path is None:
                self.image_panel.show_nothing_read(self.scan_bar.path_edit.text().strip())
            else:
                self.image_panel.show_empty()
            return

        self.image_panel.set_available_modes(self.controller.available_modes())
        # Read back rather than only set on the way in: a timeline holding no
        # image of the beam on screen puts the view on the other one, and the
        # buttons have to say so.
        self.image_panel.set_mode(self.controller.mode)
        self.image_panel.set_variants(
            self.controller.available_variants(),
            self.controller.current_variant(),
            self._variant_descriptions(),
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
        variant = self.controller.current_variant(modality)
        if frame.record(modality, variant) is None and frame.record(modality) is None:
            self.image_panel.show_image(
                None, self.controller.show_overlay, self.controller.show_image
            )
            return
        # Only read what is already prepared: a miss means the loader is still
        # working, and blocking here would freeze the window on a slow read.
        cached = self.controller.cache.cached_image(
            self.controller.frame_position, modality, variant
        )
        if cached is None:
            self.image_panel.show_loading()
        else:
            self.image_panel.show_image(
                cached, self.controller.show_overlay, self.controller.show_image,
                labelled=self.controller.series_is_editable,
            )

    def _match_plot_size_to_pane(self) -> None:
        """Prepare the stepping copies at the size they are actually shown at.

        Reported in *physical* pixels. On a high-resolution display a widget
        1070 logical pixels wide occupies 2140 real ones, and a plot prepared
        to the logical size would be magnified to fill it. The copies read
        whole are not affected: what the pane is does not change what the file
        holds.
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

    def _refresh_state_panel(self) -> None:
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
        self.state_panel.set_editing_enabled(self.controller.editing_enabled)
        # Asked of the series being shown rather than the one chosen: a lamella
        # holding only unclassified images resolves to `other` on its own.
        editable = self.controller.series_is_editable
        self.state_panel.set_series_editable(editable)
        self.operations_panel.set_series_editable(editable)

    def show_settings(self) -> None:
        """Open the settings that belong to this machine rather than the data.

        Kept under the user's home directory: who is at this keyboard is a
        property of the machine, and a name written into a collection travels
        with it.
        """
        stored = preferences.read(self.settings)
        dialog = SettingsDialog(self, stored, preview_accent=self.show_accent)
        if dialog.exec_() != QDialog.DialogCode.Accepted:
            return
        wanted = dialog.author
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
        self.image_panel.grid_view.refresh_accent()

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
        if not self._may_discard_label_edits():
            event.ignore()
            return
        self._save_layout()
        self.loader.stop()
        self.label_loader.stop()
        self.operations.stop()
        super().closeEvent(event)
