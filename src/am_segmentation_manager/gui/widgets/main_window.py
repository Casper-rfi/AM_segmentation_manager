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

from qtpy.QtCore import QByteArray, QEvent, QRect, QSettings, Qt, QUrl, Signal
from qtpy.QtGui import QCursor, QDesktopServices, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ...layout import MODALITIES
from ...operations.selection import (
    RecordState,
    StaleState,
    UnmigratedStage,
    current_state,
)
from ...progress import Cancellation
from ..controller import (
    MODE_MODALITIES,
    CurationController,
    EditingDisabled,
    ScanProgress,
)
from ..export_controller import ExportController
from ..label_controller import LabelController
from ..loader import BackgroundLoader
from ..metadata_controller import MetadataController
from ..operations_controller import OperationReport, OperationsController
from ..worker import BackgroundOperation
from .branding import PRODUCT_NAME, logo_pixmap
from .busy_dialog import BusyDialog
from .common import GripSplitter, keyboard_diagram, mark_image
from .export_panel import ExportPanel
from .frame_strip import FrameTimeline, timeline_legend
from .image_panel import ImagePanel
from .label_panel import CLASS_KEYS, LabelPanel
from .metadata_panel import MetadataPanel, completeness_legend
from .operations_panel import OperationsPanel
from .project_dialog import AddProjectsDialog
from .project_panel import ProjectPanel, status_legend
from .report_dialog import ReportDialog
from .scan_banner import ScanBanner
from .state_panel import StatePanel
from .theme import BORDER_LIGHT, HEADING, MUTED

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

#: How many filenames a confirmation lists before summarising the rest.
MAX_LISTED_FILES = 12

#: Names the save in every dialog and log line it produces.
SAVE_TITLE = "Save labels"

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
TAB_LABEL = "label"
TAB_EXPORT = "export"

#: A key that works everywhere, whatever is showing.
ANY_TAB = "any"

#: What the confirm button says for each thing an operation can be blocked on.
#: The generic wording is right for including corrected files in a batch;
#: replacing a label needs the destructive word, because it is one.
BLOCKED_ACTIONS: dict[str, str] = {
    "corrected": "Include them",
    "existing": "Replace them",
}

#: The headings the F1 list groups its rows under.
#:
#: A row belongs to exactly one, and the *order* the headings appear in is the
#: order the rows below declare them -- so the list is arranged by moving a
#: binding, not by keeping a second table of headings in step with it.
#: :data:`GROUP_HELP` is the exception, and is always last.
GROUP_VIEW = "Change the view"
GROUP_CHOOSE = "Choose what changes"
GROUP_STATE = "Set the segmentation state"
GROUP_TOOLS = "Choose a tool"
GROUP_CLASSES = "Choose a class"
GROUP_SPECKS = "Find specks"
GROUP_EDITS = "Save and undo"
GROUP_ELSEWHERE = "Elsewhere in the window"
#: The two legend headings. Separate rather than one "what the marks mean",
#: because a reader arrives asking about a particular mark and the heading is
#: what tells them which half to read.
GROUP_ROW_MARK = "The mark beside each row"
GROUP_STRIP_MARK = "The bar under the images"
GROUP_HELP = "This list"

#: Mouse gestures worth listing beside the keys, per tab, as
#: ``(heading, what to do, what it does)``.
#:
#: Not bindings: nothing registers these, the widgets read the modifiers
#: themselves. They are here because F1 is the one place that answers "what can
#: I do here?", and a gesture nobody can discover is a gesture nobody uses.
#: They carry a heading for the same reason the keys do -- picking frames out
#: is one idea whether it is done with Esc or with Ctrl-click, and splitting it
#: into a keys half and a mouse half hid that.
TAB_GESTURES: dict[str, tuple[tuple[str, str, str], ...]] = {
    TAB_CURATE: (
        (GROUP_VIEW, "Click or drag the timeline", "Move to a frame"),
        (
            GROUP_CHOOSE, "Ctrl-click a frame",
            "Pick it out for a state change, or drop it again",
        ),
        (
            GROUP_CHOOSE, "Shift-click a frame",
            "Extend the picked run from the last frame you pointed at",
        ),
        (GROUP_CHOOSE, "Ctrl or Shift + drag", "Sweep a run of frames"),
        (
            GROUP_CHOOSE, "Tick a project or lamella",
            "Apply a state change to the whole of it",
        ),
        (
            GROUP_ELSEWHERE, "Right-click a row or an image",
            "Copy its path, or open its folder",
        ),
        (
            GROUP_ELSEWHERE, "Operations",
            "Act on the same selection; each previews before it applies",
        ),
    ),
    TAB_ADD_DATA: (
        (
            GROUP_ELSEWHERE, "Right-click a project",
            "Copy its path, or open its folder",
        ),
    ),
    TAB_LABEL: (
        # First of the tool gestures, so it lands directly under [ and ]: all
        # three resize the brush, and a reader looking for how to do that
        # should not have to step over two other rows to find the third way.
        # The mouse rows stay together either way, this one leading them.
        (GROUP_TOOLS, "Shift + scroll wheel", "Make the brush bigger or smaller"),
        (GROUP_TOOLS, "Drag with the left button", "Paint with the brush"),
        (
            GROUP_TOOLS, "Click with the fill tool",
            "Flood the area of one class you clicked in",
        ),
        (
            GROUP_VIEW, "Hold Space, or drag with the middle button",
            "Pan instead of painting",
        ),
        (GROUP_VIEW, "Scroll wheel", "Zoom about the pointer"),
        (GROUP_VIEW, "Click or drag the timeline", "Move to a frame"),
    ),
}


def _brand_mark() -> QWidget:
    """Return the mark and the product name, for the corner of the tab row.

    A widget rather than a `QLabel`, a label showing either a pixmap or text
    and this needing both. Nothing is clickable: it says whose window this is
    and answers no question, so giving it behaviour would only invite pressing.
    """
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 10, 0)
    layout.setSpacing(6)

    mark = QLabel()
    mark.setPixmap(logo_pixmap(CORNER_LOGO))
    layout.addWidget(mark)

    name = QLabel(PRODUCT_NAME)
    name.setStyleSheet(f"color: {MUTED};")
    layout.addWidget(name)
    return row


#: The mark's side beside the tabs, in logical pixels. Sized to the tab text
#: rather than to the mark: it sits on that row and must not set its height.
CORNER_LOGO = 16


#: How a heading is set apart from the rows under it.
#:
#: Every left-hand cell of the tables is bold, being the column a reader scans,
#: so bold alone left the headings looking like one more key. They take the
#: colour and the weight a `QGroupBox` title takes over the captions inside its
#: card, plus a size and a rule -- the same idea as the cards, so the list
#: agrees with the window rather than inventing a second hierarchy.
HEADING_STYLE = (
    f"margin-top: 16px; margin-bottom: 2px; color: {HEADING}; "
    f"font-size: 12pt; font-weight: bold; "
    f"border-bottom: 1px solid {BORDER_LIGHT};"
)


def _table(rows) -> str:
    """Return an HTML table of ``(what to do, what it does)`` pairs.

    The left cell also carries the legend's pictures, which is why it is
    middle-aligned: a sample taller than its line otherwise sits on the text
    baseline and drags the row apart.
    """
    body = "".join(
        f"<tr><td style='padding-right: 14px; vertical-align: middle;'>"
        f"<b>{action}</b></td>"
        f"<td style='vertical-align: middle;'>{description}</td></tr>"
        for action, description in rows
    )
    return f"<table>{body}</table>"


def _keys_of(bindings: list) -> set[str]:
    """Return every key the diagram should pick out, chords broken apart.

    ``Ctrl+Shift+Z`` lights Ctrl, Shift and Z: the modifiers are keys somebody
    has to find too, and a diagram that showed only the letter would be saying
    the shortcut is one key.
    """
    return {
        part.strip()
        for binding in bindings
        for part in binding.key.split("+")
        if part.strip()
    }


@dataclass(frozen=True)
class Binding:
    """One keyboard shortcut: what it does, where it applies, and what it says.

    Attributes:
        key: The key sequence, as :class:`QKeySequence` accepts it.
        description: The line shown in the F1 list. Every binding has one.
        action: What pressing the key does.
        tab: The tab this key belongs to, or :data:`ANY_TAB`.
        group: The heading the F1 list files this key under. Required, because
            a key with nowhere to go is one the reader has to place themselves.
        scope: Restrict the key to this widget and its children instead of the
            whole window. Needed where a tab holds editors with arrow-key
            behaviour of their own, which a window-wide key would take first.
        anywhere: Let the key fire from any of the application's windows rather
            than only the one it is bound to. A window-scoped shortcut is dead
            while a second top-level window holds the keyboard, and a dialog --
            even a parented one -- is a second top-level window.
    """

    key: str
    description: str
    action: Callable[[], None]
    tab: str
    group: str
    scope: QWidget | None = None
    anywhere: bool = False


#: Kept clear of the screen edges when the list is capped to fit, so it still
#: reads as a window that can be moved rather than something wedged into the
#: corner.
SCREEN_MARGIN = 64


class ShortcutHelp(QDialog):
    """The keys for the tab in front, listed beside the window.

    Deliberately not modal, and closed by the key that opened it. A reference
    you have to dismiss before you can try what it just told you is a reference
    read once and then guessed at.
    """

    def __init__(self, parent) -> None:
        """Build the dialog against the window it belongs to."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._text = QLabel()
        self._text.setTextFormat(Qt.RichText)
        # Two tabs run past 800 pixels now the legend is in the list, so it
        # scrolls rather than being cut short: every row answers a question
        # somebody has had. A dialog merely sized past the screen is worse than
        # either, having no scrollbar the user can reach.
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._text)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self._scroll)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._buttons.rejected.connect(self.close)
        layout.addWidget(self._buttons)

    def fit_to_screen(self) -> None:
        """Size to the whole list, or to the screen where that is smaller.

        Not ``adjustSize``: a scroll area reports a small default hint rather
        than the size of what is inside it, so the dialog would come up a stub
        whatever the list said.
        """
        self._text.adjustSize()
        wanted = self._text.sizeHint()
        margins = self.layout().contentsMargins()
        around = margins.top() + margins.bottom() + self.layout().spacing()
        tall = wanted.height() + around + self._buttons.sizeHint().height()
        wide = wanted.width() + margins.left() + margins.right()

        screen = self.screen() or QApplication.primaryScreen()
        room = screen.availableGeometry()
        height = min(tall, room.height() - SCREEN_MARGIN)
        if height < tall:
            # Room for the scrollbar the cap has just made necessary, or the
            # last column of the table goes under it.
            wide += self._scroll.verticalScrollBar().sizeHint().width()
        self.resize(min(wide, room.width() - SCREEN_MARGIN), height)

    def is_in_front(self) -> bool:
        """Return whether this list is the window a key press is aimed at.

        Not ``isVisible()``: a list left open behind the window is still
        visible, and treating that as "in front" is what made F1 dismiss,
        unseen, the thing it was pressed to see.
        """
        return self.isActiveWindow()

    def show_for(self, tab_name: str, help_text: str) -> None:
        """List *tab_name*'s keys, replacing whatever was listed before."""
        self.setWindowTitle(f"Keyboard shortcuts: {tab_name}")
        self._text.setText(help_text)
        self.fit_to_screen()
        self.show()
        self.raise_()
        # Not just raised: a window that is on top but not activated still
        # leaves the keyboard where it was, and on macOS receives no pointer
        # motion, so the list would not even show its own tooltips.
        self.activateWindow()


class SaveLabelsDialog(QDialog):
    """The one dialog that writes proofread work, laid out on purpose.

    Not a message box. Fusion reports the Windows button order wherever it runs,
    so a message box ends the row with *Cancel* -- the button that changes
    nothing sitting where the eye lands last, past both ways of saving. The
    order here is the shape of the choice: leave, save, save and finish. The
    last is the default, and therefore the accented one, because proofreading a
    label and not marking it is the exception.

    Overriding the style hint that decides the order would fix every dialog at
    once, and was tried: ``QProxyStyle.styleHint`` segfaults PySide6 while a
    message box is being built.
    """

    def __init__(
        self, parent, message: str, note: str, *, offer_corrected: bool
    ) -> None:
        """Ask how to save.

        Args:
            parent: The window the save belongs to.
            message: What the save would do.
            note: A second line, or empty. Says how to reach what is not on
                offer when *offer_corrected* is False.
            offer_corrected: Whether marking the reviewed set corrected is
                available, which follows the editing guard.
        """
        super().__init__(parent)
        self.setWindowTitle(SAVE_TITLE)
        self.mark_corrected = False

        layout = QVBoxLayout(self)
        said = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(
            self.style().standardIcon(QStyle.SP_MessageBoxQuestion).pixmap(40, 40)
        )
        icon.setAlignment(Qt.AlignTop)
        said.addWidget(icon)
        said.addSpacing(8)

        text = QLabel(message if not note else f"{message}\n\n{note}")
        text.setWordWrap(True)
        said.addWidget(text, stretch=1)
        layout.addLayout(said)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        buttons.addStretch()

        # "Save only" only means something against a second way of saving.
        save_only = QPushButton("Save only" if offer_corrected else "Save")
        save_only.clicked.connect(self.accept)
        buttons.addWidget(save_only)

        if offer_corrected:
            mark = QPushButton("Save and mark corrected")
            mark.clicked.connect(self._mark_and_accept)
            mark.setDefault(True)
            buttons.addWidget(mark)
        else:
            save_only.setDefault(True)
        layout.addLayout(buttons)

    def _mark_and_accept(self) -> None:
        """Accept, having chosen to mark the reviewed set corrected."""
        self.mark_corrected = True
        self.accept()


class CurationWindow(QMainWindow):
    """Main window for reviewing and curating adaptive milling data."""

    #: Emitted to the background loader thread; see :class:`BackgroundLoader`.
    request_load = Signal(list, int)

    #: Emitted to the Label tab's own loader thread. Separate from
    #: :attr:`request_load` because the two prepare frames at different sizes:
    #: proofreading needs every pixel, and reviewing deliberately does not.
    request_label_load = Signal(list, int)

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
        self.setWindowTitle("Adaptive milling segmentation curation")
        self.resize(1600, 950)

        self.settings = settings if settings is not None else QSettings(
            SETTINGS_ORGANISATION, SETTINGS_APPLICATION
        )
        self.project_panel = ProjectPanel()
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
            self.settings.value(SETTING_DEVICE, "", type=str)
        )

        # The other tabs get their own Qt-free controllers, and are handed the
        # projects from the curation scan rather than scanning again: a second
        # scan would double the slowest thing the application does.
        # Handed the curation controller's config rather than loading its own:
        # which class is a crack has to be the same question the overlays
        # answer, and resolving it twice invites two answers.
        self.export_controller = ExportController(config=controller.config)
        self.label_controller = LabelController(config=controller.config)
        self.metadata_controller = MetadataController()
        self.operations_controller = OperationsController()
        self.export_panel = ExportPanel(self.export_controller)
        self.label_panel = LabelPanel(self.label_controller)
        self.metadata_panel = MetadataPanel(self.metadata_controller)

        self._build_layout()
        self._connect()
        #: The folder the tree is on, when it is on one rather than a
        #: project. What the summary line describes; None means the frame.
        #: The shortcut list, built the first time F1 asks for it.
        self._help: ShortcutHelp | None = None
        self._add_shortcuts()
        self._restore_layout()

        self.loader = BackgroundLoader(controller.cache)
        self.request_load.connect(self.loader.worker.load_batch)
        self.loader.worker.loaded.connect(self._on_loaded)

        self.label_loader = BackgroundLoader(self.label_controller.cache)
        self.request_label_load.connect(self.label_loader.worker.load_batch)
        self.label_loader.worker.loaded.connect(self._on_label_loaded)

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
        #: The devices a child process reported, asked for once per session.
        #: None until asked; an empty tuple is a machine that could not say.
        self._devices: tuple[str, ...] | None = None

        remembered = self.settings.value(SETTING_SCAN_ROOT, "", type=str)
        if remembered:
            self.project_panel.set_path(remembered)

        self.refresh()
        # The window opens on Curate, and _tab_changed does not fire for the
        # tab that was already showing.
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

        self.curate = GripSplitter(Qt.Horizontal)
        self.curate.addWidget(self.project_panel)
        self.curate.addWidget(centre)
        self.curate.addWidget(right)
        self.curate.setStretchFactor(1, 1)
        self.curate.setSizes([300, 1080, 260])

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
        # The mark, quietly, at the end of the tab row: the one place the
        # window has spare that a reader already looks at. Only here and on the
        # banner -- the orange and green in it are segmentation class colours
        # everywhere else, so a third use would be read as data.
        self.tabs.setCornerWidget(_brand_mark(), Qt.TopRightCorner)
        self.setCentralWidget(self.tabs)

        # Parented to the window rather than to a tab: a scan can be asked for
        # from any of them, and what it holds up is the whole window.
        self.banner = ScanBanner(self)

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
        self.label_panel.loads_needed.connect(self._schedule_label_loads)
        self.label_panel.save_requested.connect(self.save_labels)
        self.project_panel.scan_requested.connect(self.scan)
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

        self.image_panel.mode_changed.connect(self._set_mode)
        self.image_panel.variant_changed.connect(self._set_variant)
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

        self.metadata_panel.survey_requested.connect(self._inspect_layout)
        self.metadata_panel.init_requested.connect(self._add_projects)
        self.metadata_panel.compress_requested.connect(self._compress_labels)
        self.export_panel.csv_check_requested.connect(self._check_csv)
        self.export_panel.measure_requested.connect(self._measure_labels)

        self.export_panel.files_export_requested.connect(self._copy_files)
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

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Record gaining and losing the keyboard, at debug level.

        Whether the window was in front is not otherwise recoverable after the
        fact, and it is the first thing to establish about any report that the
        keyboard or the pointer stopped working.
        """
        if event.type() == QEvent.ActivationChange:
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
            return self.settings.value(key, "", type=str)
        return self.settings.value(SETTING_LEGACY_MODEL, "", type=str)

    def _remember_model(self) -> None:
        """Store what is in the model box against the view it belongs to."""
        self.settings.setValue(
            self._model_key(self._model_context), self.operations_panel.model()
        )

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
            # The arrow keys here are scoped to the project list, so they only
            # work once something in it has focus. Giving it focus on arrival
            # is what makes them work without a click first.
            self.metadata_panel.projects.setFocus()
        self._apply_tab_shortcuts(self._current_tab())
        if self._help is not None and self._help.isVisible():
            self._show_help()

    def _open_export(self) -> None:
        """Read the image headers before the tab first draws, and nothing else.

        Half a millisecond an image -- a second or two over a real collection --
        and it is what the acquisition filters and breakdowns are made of, so
        paying for it on arrival is what lets them simply work rather than sit
        there unusable with no way to tell why.

        The labels are *not* read here. Opening every one is ten times the cost
        and answers a different question, so it stays behind Measure labels
        where the user asks for it against the selection they care about.
        """
        if not self.export_controller.records or self.export_controller.metadata_loaded():
            self.export_panel.refresh()
            return
        images = len(self.export_controller.records)
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
        self.export_controller.set_projects(projects, self.controller.start_path)
        self.metadata_controller.set_projects(projects)
        self.operations_controller.set_start_path(self.controller.start_path)
        self.metadata_panel.set_projects(self.metadata_controller.project_summaries())
        self.metadata_panel.set_root(self.controller.start_path)
        if self.controller.start_path is not None:
            self.metadata_panel.set_unindexed(
                *self.operations_controller.unindexed_summary(self.controller.start_path)
            )
        # Re-pointed rather than left holding records from the previous scan,
        # whose paths may no longer be where the files are.
        self.label_controller.set_lamella(self.controller.lamella)

    # -- operations that change files --------------------------------------

    def _run_operation(
        self, description: str, work, on_done, *, cancellation: Cancellation | None = None
    ) -> None:
        """Run *work* off the interface thread, then hand its result to *on_done*.

        The one seam through which every operation reaches its thread, so a test
        can substitute an inline call and still exercise the rest of the flow.

        Args:
            cancellation: Pass one to give the dialog a Stop button. The bar is
                a separate question: the dialog opens indeterminate and turns
                into a count as soon as the work reports progress, so an
                operation can say how far it has got without being stoppable.
        """
        self._busy = BusyDialog(self, description, stoppable=cancellation is not None)
        if cancellation is not None:
            self._busy.stop_requested.connect(cancellation.request)
        self._busy.show()
        self.operations_panel.set_busy(True)

        self._pending_operation = (description, on_done)
        self.request_operation.emit(work)

    def _operation_progressed(self, done: int, total: int) -> None:
        """Move the busy dialog's bar, on the thread that owns it.

        A report can arrive after the work has returned -- both are queued
        signals from the same thread -- so a closed dialog is a normal outcome
        rather than a fault.
        """
        if self._busy is None:
            return
        self._busy.set_progress(done, total)

    def _operation_finished(self, result) -> None:
        """Hand a finished operation's result to whatever asked for it."""
        pending = self._take_pending_operation()
        if pending is None:  # pragma: no cover - nothing was submitted
            return
        _description, on_done = pending
        on_done(result)

    def _operation_failed(self, error) -> None:
        """Report an operation that raised, on the thread that can show it."""
        logger.warning("Operation failed: %s", error)
        pending = self._take_pending_operation()
        description = pending[0] if pending else "Operation failed"
        QMessageBox.critical(self, description, str(error))

    def _take_pending_operation(self) -> tuple | None:
        """Close the busy dialog and return what was waiting on it, once.

        Cleared before the callback runs, so a callback that starts the next
        operation -- which is how preview leads to apply -- is not overwritten
        by the one that just finished.
        """
        pending, self._pending_operation = self._pending_operation, None
        # Detached before it is closed, so anything that runs while it closes
        # finds no dialog rather than one that is being taken apart.
        busy, self._busy = self._busy, None
        if busy is not None:
            busy.close()
            busy.deleteLater()
        self.operations_panel.set_busy(False)
        return pending

    def _operate(
        self, title: str, run, *, rescan: bool = False,
        cancellation: Cancellation | None = None,
    ) -> None:
        """Preview an operation, confirm it, then apply what was previewed.

        Args:
            title: Names the operation in every dialog it opens.
            run: Called as ``run(allow, dry_run)``, returning an
                :class:`OperationReport`. ``allow`` answers whatever the report
                was blocked on. It runs on the worker thread, so it must touch
                nothing belonging to the window.
            rescan: Set where the operation creates files a scan would have
                indexed -- a new label, a new project. A refresh recounts what
                is already in memory, which cannot show a file that was not
                there when the scan ran.
            cancellation: Makes the applying run stoppable. The previews are
                not: they open no file and answer in milliseconds, so a Stop
                button on one is a button with nothing to stop.

        Preview and apply are two runs of the same function rather than one run
        whose result is replayed: a preview held across a dialog describes a
        filesystem the user has had time to change.
        """

        def previewed(report: OperationReport, allowed: bool = False) -> None:
            if report.blocked and not allowed:
                action = BLOCKED_ACTIONS.get(report.blocked, "Include them")
                if not self._confirm_report(title, report, action):
                    return
                self._run_operation(
                    title, lambda: run(True, True), lambda r: previewed(r, True)
                )
                return
            if not report.count:
                self._show_report(title, report)
                return
            if not self._confirm_report(title, report, "Apply"):
                return
            self._run_operation(
                title,
                lambda: run(allowed, False),
                lambda applied: self._operation_applied(title, applied, rescan),
                cancellation=cancellation,
            )

        self._run_operation(title, lambda: run(False, True), previewed)

    def _operation_applied(
        self, title: str, report: OperationReport, rescan: bool = False
    ) -> None:
        """Record what an operation did and bring the window back in step."""
        self.state_panel.show_last_change(f"{title}: {report.headline}")
        logger.info("%s: %s", title, report.headline)
        self._show_report(title, report)
        # Marked on both sides, as the steps of a prediction run are: the last
        # thing an operation does is read the whole collection again, and a
        # window that disappears once the work is finished has to be placed
        # either side of that read.
        logger.info("%s: %s", title, "rescanning" if rescan else "refreshing")
        if rescan:
            self.rescan()
        else:
            self.refresh(reload_lists=True)
        logger.info("%s: the window is back in step", title)

    def _confirm_report(self, title: str, report: OperationReport, action: str) -> bool:
        """Show a report and ask whether to go ahead."""
        return ReportDialog(self, title, report.text(), action).exec() == QDialog.Accepted

    def _show_report(self, title: str, report: OperationReport) -> None:
        """Show a report with nothing to accept."""
        ReportDialog(self, title, report.text()).exec()

    def _editable_records(self) -> list | None:
        """Return what an operation would act on, or None with a reason shown.

        The same guard and the same selection the state buttons use: an
        operation that writes files is no less of a change for being called
        something else.
        """
        if not self.controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Tick 'Enable editing' before running an operation.",
            )
            return None
        records = self.controller.scoped_records()
        if not records:
            self._flash("Nothing selected")
            return None
        return records

    def _predict(self, model: str) -> None:
        """Predict the selection with the model named in the panel.

        Guarded in the order the answers are needed: whether this window may
        write at all, then whether it can predict, then whether it was told
        what with. Asking for a model first would ask for a file before saying
        the operation was never going to run.
        """
        records = self._editable_records()
        if records is None:
            return

        # No check that imports torch. The window predicts in a child process
        # precisely so that torch is never loaded here, and asking whether it
        # would load is asking by loading it. A stack that is missing or broken
        # is reported by the child, which is the process that has to import it.
        if not model:
            self._flash("Choose a model to predict with")
            return
        weights = Path(model)
        # Remembered against the view it was chosen for, so the next session
        # opens with the same model in the box -- one model, many runs.
        self._remember_model()
        device = self.operations_panel.device()
        self.settings.setValue(SETTING_DEVICE, device)

        # Created here rather than in _operate so the work closure and the
        # dialog share one: the closure polls it on the worker thread, the
        # dialog's Stop button sets it on this one.
        cancellation = Cancellation()
        progressed = self.operations.worker.progressed

        # Rescanned afterwards: a prediction that did not exist when the scan
        # ran is not in the index, so nothing would offer to convert it.
        self._operate(
            "Predict labels",
            lambda allow, dry: self.operations_controller.predict(
                records, weights, device=device, dry_run=dry,
                cancellation=None if dry else cancellation,
                on_progress=None if dry else progressed.emit,
            ),
            rescan=True,
            cancellation=cancellation,
        )

    def _offer_devices(self) -> None:
        """Fill the device box with what the machine actually has.

        Asked of a separate process, because this one never imports torch --
        that is what predicting in a child is for. It costs an interpreter
        start and a torch import, so it is done once per session, when the box
        is first opened: by then the user has said they want to choose one.
        """
        if self._devices is not None:
            return
        from ...operations.predict_process import probe_devices

        self.project_panel.announce("Asking which devices are available...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._devices = probe_devices()
        finally:
            QApplication.restoreOverrideCursor()
        logger.info("Devices available for prediction: %s", self._devices or "none")
        self.operations_panel.show_devices(self._devices)
        self.refresh()

    def _convert_predictions(self) -> None:
        records = self._editable_records()
        if records is None:
            return
        # Rescanned afterwards: the new labels do not exist in the scanned
        # index, so the timeline would keep drawing these frames as unlabelled.
        self._operate(
            "Convert predictions",
            lambda allow, dry: self.operations_controller.convert_predictions(
                records, allow_overwrite=allow, dry_run=dry
            ),
            rescan=True,
        )

    def _compress_labels(self) -> None:
        """Recompress every label under the scanned folder.

        The whole collection rather than a selection: recompression preserves
        the pixels exactly, so there is no label it should be withheld from,
        and choosing which ones to shrink is not a decision worth a control.
        The corrected-label confirmation still applies -- rewriting the file is
        what needs the yes, not changing what is in it.
        """
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Tick 'Enable editing' before recompressing labels.",
            )
            return
        records = [
            record
            for project in self.controller.projects
            for record in project.image_records
        ]
        if not records:
            self._flash("Scan a folder first")
            return
        # Only the applying run counts: the preview reads the file sizes and
        # answers at once, where re-encoding a whole collection is minutes with
        # nothing else on screen to say it is working.
        cancellation = Cancellation()
        progressed = self.operations.worker.progressed
        self._operate(
            "Compress labels",
            lambda allow, dry: self.operations_controller.compress_labels(
                records, allow_corrected=allow, dry_run=dry,
                on_progress=None if dry else progressed.emit,
                cancellation=None if dry else cancellation,
            ),
            cancellation=cancellation,
        )

    def _inspect_layout(self) -> None:
        """Report how the scanned directory is interpreted. Read-only."""
        root = self.controller.start_path
        if root is None:
            self._flash("Scan a folder first")
            return
        self._run_operation(
            "Inspect layout",
            lambda: self.operations_controller.survey(root),
            lambda report: self._show_report("Inspect layout", report),
        )

    def _add_projects(self) -> None:
        """Ask which directories become projects, then write their files.

        The dialog *is* the preview: every row already says what it would
        become, so this goes straight to writing rather than through the
        confirm-a-report pair the other operations use. Showing the same
        information a second time would not be read.
        """
        root = self.controller.start_path
        if root is None:
            self._flash("Scan a folder first")
            return
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Tick 'Enable editing' before writing metadata files.",
            )
            return
        dialog = AddProjectsDialog(self, self.operations_controller, root)
        if not dialog.exec_():
            return
        chosen = dialog.chosen()
        if not chosen:
            return
        # Rescanned afterwards: making an unindexed dataset into a project is
        # the point of this, and only a scan discovers the projects it created.
        self._run_operation(
            "Add projects",
            lambda: self.operations_controller.create_projects(chosen, dry_run=False),
            lambda report: self._operation_applied("Add projects", report, True),
        )

    def _check_csv(self, path: str) -> None:
        """Check that every path an exported CSV records still exists."""
        root = self.controller.start_path
        if root is None:
            self._flash("Scan a folder first")
            return
        self._run_operation(
            "Check exported CSV",
            lambda: self.operations_controller.check_csv(Path(path), root),
            lambda report: self._show_report("Check exported CSV", report),
        )

    def _measure_labels(self) -> None:
        """Measure every label off the interface thread, then redraw.

        The first thing in the Export tab slow enough to need the worker: it
        opens each label and labels its components, tens of milliseconds each
        against half a millisecond for a header. Long enough that the dialog
        has to say it is advancing rather than only that it is busy.
        """
        labels = self.export_controller.measurable_labels()
        progressed = self.operations.worker.progressed
        self._run_operation(
            f"Measuring {labels} label(s)",
            lambda: self.export_controller.load_measurements(progressed.emit),
            lambda measured: self._labels_measured(measured),
        )

    def _labels_measured(self, measured: int) -> None:
        """Redraw with the measurements the worker read."""
        logger.info("Measured labels for %d image(s)", measured)
        self.export_panel.refresh()

    def _with_labels(self, then: Callable[[], None]) -> None:
        """Read the labels this selection needs, then do the thing needing them.

        An export has to know what is inside each label -- a blank mask is
        never paired with an image -- and that is the one slow read in this
        tab. Doing it here rather than inside the write keeps it off the
        interface thread and behind a dialog that counts, and doing it only
        when it has not already happened means measuring first costs the
        export nothing.
        """
        if self.export_controller.labels_known():
            then()
            return
        labels = self.export_controller.measurable_labels()
        progressed = self.operations.worker.progressed
        self._run_operation(
            f"Reading {labels} label(s)",
            lambda: self.export_controller.warm_labels(progressed.emit),
            lambda _read: (self.export_panel.refresh(), then()),
        )

    def _export_training(self, directory: str) -> None:
        if not self.export_controller.selected_records():
            self._flash("Nothing matches these filters")
            return
        self._with_labels(lambda: self._write_training(directory))

    def _write_training(self, directory: str) -> None:
        if not self._confirm_dropping_corrected():
            return
        try:
            train, validation = self.export_controller.export_training(Path(directory))
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        written = ", ".join(str(path.name) for path in (train, validation) if path)
        self._flash(f"Wrote {written}", 4000)
        logger.info("Exported training CSV to %s", directory)

    def _confirm_dropping_corrected(self) -> bool:
        """Ask before writing a training set that silently omits proofread work.

        The export skips an image whose label is missing or blank, which is
        right -- pairing an image with a blank mask teaches a model that the
        structure is background. What is wrong is doing it quietly: a corrected
        image being dropped means the proofreading or the label has gone
        missing, and the CSV that results is short with nothing saying so.
        """
        readiness = self.export_controller.unexportable_corrected()
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
            "Export without proofread images?",
            f"These images are marked corrected, but hold no annotation to "
            f"pair them with, so the export will leave them out "
            f"({readiness.summary()}):\n\n{shown}\n\nWrite the files anyway?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

    def _copy_files(self, directory: str) -> None:
        """Copy the matching images, labels and project files into *directory*.

        On the same preview-then-apply path as the operations that write into
        the collection, and stoppable for the same reason: this one writes
        gigabytes, and a copy that lands by rename leaves whole files wherever
        it is stopped.

        No rescan afterwards. Nothing in the scanned collection has changed --
        the copy is somewhere else, and refuses to be anywhere else.
        """
        if not self.export_controller.selected_records():
            self._flash("Nothing matches these filters")
            return

        cancellation = Cancellation()
        progressed = self.operations.worker.progressed
        self._operate(
            "Export images and labels",
            lambda allow, dry: self.operations_controller.copy_files(
                self.export_controller.selected_records(),
                Path(directory),
                projects=self.export_controller.projects,
                start_path=self.export_controller.start_path,
                dry_run=dry,
                cancellation=None if dry else cancellation,
                on_progress=None if dry else progressed.emit,
            ),
            cancellation=cancellation,
        )

    def _export_image_metadata(self, path: str) -> None:
        if not self.export_controller.selected_records():
            self._flash("Nothing matches these filters")
            return
        try:
            self.export_controller.export_image_metadata(Path(path))
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
        """Bind the keys, each tagged with the tab and the heading it belongs to.

        A binding scoped to a widget uses ``WidgetWithChildrenShortcut`` and so
        only fires with focus inside it; every other binding belongs to the
        window and is switched on and off as tabs change.

        **The order below is the order F1 reads in.** Headings appear in the
        order the bindings first name them, so grouping the list is a matter of
        moving a line here rather than of maintaining a table of headings
        elsewhere that can fall out of step with it.
        """
        curate = TAB_CURATE
        paint_keys = [
            Binding(
                key,
                f"Paint {name}",
                lambda position=position: self.label_panel.select_class_at(position),
                TAB_LABEL,
                GROUP_CLASSES,
            )
            for position, (key, name) in enumerate(
                zip(CLASS_KEYS, self.label_panel.class_names())
            )
        ]
        self._bindings = [
            # -- curate --------------------------------------------------
            Binding("Right", "Next frame", lambda: self._step_frame(1), curate, GROUP_VIEW),
            Binding("Left", "Previous frame", lambda: self._step_frame(-1), curate, GROUP_VIEW),
            Binding("Down", "Next lamella", lambda: self._step_lamella(1), curate, GROUP_VIEW),
            Binding("Up", "Previous lamella", lambda: self._step_lamella(-1), curate, GROUP_VIEW),
            Binding(
                "1", "SEM view", lambda: self._set_mode("sem", from_user=True),
                curate, GROUP_VIEW,
            ),
            Binding(
                "2", "FIB view", lambda: self._set_mode("fib", from_user=True),
                curate, GROUP_VIEW,
            ),
            Binding(
                "3", "Plot view", lambda: self._set_mode("plot", from_user=True),
                curate, GROUP_VIEW,
            ),
            Binding(
                "V", "Show / hide the label overlay", self._toggle_overlay,
                curate, GROUP_VIEW,
            ),
            Binding(
                "G", "Fit the view to the pane", self.image_panel.fit, curate, GROUP_VIEW
            ),
            Binding(
                "Esc", "Clear the picked-out frames", self._clear_frame_selection,
                curate, GROUP_CHOOSE,
            ),
            Binding(
                "D", "Set state: deselected", lambda: self.apply_state("deselected"),
                curate, GROUP_STATE,
            ),
            Binding(
                "S", "Set state: selected", lambda: self.apply_state("selected"),
                curate, GROUP_STATE,
            ),
            Binding(
                "C", "Set state: corrected", lambda: self.apply_state("corrected"),
                curate, GROUP_STATE,
            ),
            # -- add data ------------------------------------------------
            # Scoped to the project list: the fields beside it include editable
            # combo boxes, whose own Up/Down a window-wide key would swallow.
            Binding(
                "Down", "Next project", lambda: self.metadata_panel.step_project(1),
                TAB_ADD_DATA, GROUP_VIEW, scope=self.metadata_panel.projects,
            ),
            Binding(
                "Up", "Previous project", lambda: self.metadata_panel.step_project(-1),
                TAB_ADD_DATA, GROUP_VIEW, scope=self.metadata_panel.projects,
            ),
            # -- label ---------------------------------------------------
            Binding(
                "Right", "Next frame", lambda: self.label_panel.step_frame(1),
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "Left", "Previous frame", lambda: self.label_panel.step_frame(-1),
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "Down", "Next lamella", lambda: self._step_label_lamella(1),
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "Up", "Previous lamella", lambda: self._step_label_lamella(-1),
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "1", "SEM images", lambda: self.label_panel.set_modality("SEM"),
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "2", "FIB images", lambda: self.label_panel.set_modality("FIB"),
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "V", "Show / hide the label", self.label_panel.toggle_label,
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "G", "Fit the image to the pane", self.label_panel.canvas_fit,
                TAB_LABEL, GROUP_VIEW,
            ),
            Binding(
                "D", "Brush tool", lambda: self.label_panel.set_tool("brush"),
                TAB_LABEL, GROUP_TOOLS,
            ),
            Binding(
                "F", "Fill tool", lambda: self.label_panel.set_tool("fill"),
                TAB_LABEL, GROUP_TOOLS,
            ),
            Binding(
                "[", "Smaller brush", lambda: self.label_panel.step_brush(-1),
                TAB_LABEL, GROUP_TOOLS,
            ),
            Binding(
                "]", "Larger brush", lambda: self.label_panel.step_brush(1),
                TAB_LABEL, GROUP_TOOLS,
            ),
            *paint_keys,
            Binding(
                "A", "Lock or unlock every class",
                self.label_panel.toggle_all_locks, TAB_LABEL, GROUP_CLASSES,
            ),
            Binding(
                "X", "Find specks: ring the small components",
                self.label_panel.toggle_specks, TAB_LABEL, GROUP_SPECKS,
            ),
            Binding(
                "N", "Go to the next speck",
                lambda: self.label_panel.step_speck(1), TAB_LABEL, GROUP_SPECKS,
            ),
            Binding(
                "Ctrl+S", "Save this lamella's labels", self.save_labels,
                TAB_LABEL, GROUP_EDITS,
            ),
            Binding(
                "Ctrl+Z", "Undo the last stroke", self.label_panel.undo,
                TAB_LABEL, GROUP_EDITS,
            ),
            Binding(
                "Ctrl+Shift+Z", "Redo the last stroke", self.label_panel.redo,
                TAB_LABEL, GROUP_EDITS,
            ),
            # The one key that has to work while the list it opens holds the
            # keyboard, since closing it again is what it is for.
            Binding(
                "F1", "Show this list", self.show_shortcuts,
                ANY_TAB, GROUP_HELP, anywhere=True,
            ),
        ]
        self._shortcuts = []
        for binding in self._bindings:
            shortcut = QShortcut(
                QKeySequence(binding.key), binding.scope or self, activated=binding.action
            )
            if binding.scope is not None:
                shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            elif binding.anywhere:
                shortcut.setContext(Qt.ApplicationShortcut)
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
            (self.metadata_panel, TAB_ADD_DATA),
            (self.label_panel, TAB_LABEL),
            (self.export_panel, TAB_EXPORT),
        ):
            if widget is panel:
                return tab
        return TAB_CURATE

    def shortcut_help(self, tab: str | None = None) -> str:
        """Return the shortcut table for a tab, as the help dialog's HTML.

        Under a heading per group, keys and mouse gestures together: what a
        reader is looking for is "how do I pick frames out", and answering that
        across a list of keys and a separate list of gestures makes them do the
        grouping the list should have done.

        Separate from showing it so what the dialog says can be tested without
        opening a modal window.
        """
        tab = tab if tab is not None else self._current_tab()
        applicable = [b for b in self._bindings if b.tab in (tab, ANY_TAB)]
        grouped = self._grouped_rows(
            applicable, TAB_GESTURES.get(tab, ()), self._tab_marks(tab)
        )
        help_text = "".join(
            f"<p style='{HEADING_STYLE}'>{group}</p>{_table(rows)}"
            for group, rows in grouped.items()
        )
        own_keys = [b for b in applicable if b.tab != ANY_TAB]
        if own_keys:
            # Above the tables: which of the keys under your hands do anything
            # here is a question a picture answers faster than a list, and what
            # each one does is the list's own job directly below.
            help_text = (
                f"<p>{mark_image(keyboard_diagram(_keys_of(applicable)))}</p>"
                + help_text
            )
        else:
            help_text += (
                "<p style='margin-top: 10px;'>This tab has no keys of its own.</p>"
            )
        if tab == TAB_CURATE:
            help_text += (
                "<p style='margin-top: 10px;'>State changes need <i>Enable editing</i> "
                "switched on.</p>"
            )
        if tab == TAB_LABEL:
            help_text += (
                "<p style='margin-top: 10px;'>Edits are held in this window until "
                "you save them. Marking the saved set corrected needs "
                "<i>Enable editing</i> switched on.</p>"
            )
        return help_text

    @staticmethod
    def _grouped_rows(
        bindings: list[Binding],
        gestures: tuple[tuple[str, str, str], ...],
        marks: tuple[tuple[str, str, str], ...] = (),
    ) -> dict[str, list[tuple[str, str]]]:
        """Return the rows of each heading, in the order the headings arise.

        Keys first, then the gestures that belong beside them, so a heading
        reads as its keys and then the pointer equivalents; the legend last,
        under headings of its own that nothing else names. :data:`GROUP_HELP`
        is moved to the end: the key that opens this list is the one the reader
        has already found.
        """
        grouped: dict[str, list[tuple[str, str]]] = {}
        for binding in bindings:
            grouped.setdefault(binding.group, []).append(
                (binding.key, binding.description)
            )
        for group, action, description in (*gestures, *marks):
            grouped.setdefault(group, []).append((action, description))
        if GROUP_HELP in grouped:
            grouped[GROUP_HELP] = grouped.pop(GROUP_HELP)
        return grouped

    @staticmethod
    def _tab_marks(tab: str) -> tuple[tuple[str, str, str], ...]:
        """Return the legend for *tab*, as ``(heading, picture, meaning)``.

        Every picture is drawn by the module that draws the real mark, so the
        list cannot explain one thing and the window show another. Only the
        marks on the tab are listed: the white band means frames picked out
        while curating and unsaved edits while proofreading, which is one mark
        with two meanings and no way to say both at once.
        """
        rows: list[tuple[str, list[tuple[QPixmap, str]]]] = []
        if tab == TAB_CURATE:
            rows = [
                (GROUP_ROW_MARK, status_legend()),
                (
                    GROUP_STRIP_MARK,
                    timeline_legend("Picked out for a state change"),
                ),
            ]
        elif tab == TAB_LABEL:
            rows = [
                (GROUP_STRIP_MARK, timeline_legend("This label has unsaved edits")),
            ]
        elif tab == TAB_ADD_DATA:
            rows = [(GROUP_ROW_MARK, completeness_legend())]
        return tuple(
            (group, mark_image(sample), meaning)
            for group, legend in rows
            for sample, meaning in legend
        )

    def show_shortcuts(self) -> None:
        """List the keys for the tab now showing, or put the list away.

        Tooltips only help someone who already suspects a control exists; this
        is the one place that answers "what can I press?". Only the active tab's
        keys are listed, because the others are switched off and pressing one
        does nothing.

        F1 puts the list away only when the list is the window in front. One
        left open behind the main window is still visible, so closing on that
        alone made F1 answer "what can I press?" by dismissing, unseen, the one
        thing that says -- and the fix for a key that appears to do nothing is
        to press it again, which reopened it and made the whole thing look
        intermittent.

        Nothing happens while an operation is running behind its modal dialog.
        The key reaches here from any window now, and a list opened over a
        modal one cannot be read, moved or closed until the operation ends.
        """
        if QApplication.activeModalWidget() is not None:
            return
        if self._help is not None and self._help.is_in_front():
            self._help.close()
            return
        if self._help is None:
            self._help = ShortcutHelp(self)
        self._show_help()

    def _show_help(self) -> None:
        """Fill the shortcut list with the keys of the tab in front."""
        self._help.show_for(
            self.tabs.tabText(self.tabs.currentIndex()), self.shortcut_help()
        )

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
            self.project_panel.set_path(str(directory))
            self.settings.setValue(SETTING_SCAN_ROOT, str(directory))

    def rescan(self) -> None:
        """Read the same folder again, staying on the lamella being reviewed.

        What follows anything that moved files. Scanning again is how the
        window learns what an operation did; going back to where the reviewer
        was is what stops that costing them their place.
        """
        if self.controller.start_path is None:  # pragma: no cover - always set by here
            self.refresh(reload_lists=True)
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

        The read holds the interface thread for as long as it takes -- fifteen
        seconds on a large collection -- so nothing else can paint, answer or
        report while it runs. The banner covers the window for the whole of it
        and is repainted from inside the read, which is the only thing still
        running; each repaint is synchronous and processes no events, so
        nothing can be re-entered on the way. The wait cursor stays, for the
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
        QApplication.setOverrideCursor(Qt.WaitCursor)
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

    def _set_mode(self, mode: str, from_user: bool = False) -> None:
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
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

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
        QApplication.setOverrideCursor(Qt.WaitCursor)
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
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
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

        frame = self.controller.frame
        if frame is None:
            # Emptied rather than left as it was: the conditions on offer
            # belonged to a frame that is no longer on screen.
            self.image_panel.set_variants([])
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
                cached, self.controller.show_overlay, self.controller.show_image
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

    def _refresh_state_panel(self) -> None:
        self.state_panel.show_state(
            self.controller.target_state(),
            self.controller.state_of("SEM"),
            self.controller.state_of("FIB"),
        )
        reach = self.controller.reach()
        self.state_panel.show_reach(reach.summary(), reach.shared_note())
        self.state_panel.show_selection(self.controller.describe_selection())
        self.operations_panel.show_selection(self.controller.describe_selection())
        self.operations_panel.show_predictions(
            self.operations_controller.describe_predictions(self.controller.scoped_records())
        )
        self.state_panel.set_target(self.controller.target)
        self.state_panel.set_editing_enabled(self.controller.editing_enabled)

    def _point_label_tab(self) -> None:
        """Show the lamella the curation view is on, and draw it.

        The Label tab has no lamella list of its own: proofreading follows the
        lamella being curated, so there is one cursor over the collection rather
        than two that can disagree. Where that would throw away unsaved edits,
        the tab stays where it is instead.
        """
        wanted = self.controller.lamella
        current = self.label_controller.lamella
        changing = wanted is not None and (
            current is None or wanted.path != current.path
        )
        if changing and not self._may_discard_label_edits():
            self.label_panel.refresh()
            return
        self.label_controller.set_lamella(wanted)
        self.label_panel.set_variant_descriptions(self._variant_descriptions())
        self.label_panel.refresh()

    def _step_label_lamella(self, delta: int) -> None:
        """Move both views to another lamella, from the Label tab."""
        if not self._may_discard_label_edits():
            return
        self._step_lamella(delta)
        self._point_label_tab()

    def _may_discard_label_edits(self) -> bool:
        """Ask before throwing away unsaved label edits; return whether to go on.

        Deliberately not offered as "save these first": saving runs off the
        interface thread and asks its own questions, and threading a navigation
        through the far side of that is how a half-finished save ends up moving
        files under a lamella nobody is looking at any more.
        """
        pending = self.label_controller.pending_count()
        if not pending:
            return True
        lamella = self.label_controller.lamella
        name = lamella.display_name if lamella is not None else "this lamella"
        answer = QMessageBox.warning(
            self,
            "Unsaved label edits",
            f"{pending} label{'s' if pending != 1 else ''} edited in {name} "
            f"{'have' if pending != 1 else 'has'} not been saved.\n\n"
            "Save them with Ctrl+S first, or discard them to continue.",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Discard:
            return False
        self.label_controller.discard_edits()
        return True

    def save_labels(self) -> None:
        """Write this lamella's edited labels, after asking what to do with them."""
        if not self.label_controller.pending_count():
            self._flash("No label edits to save")
            return

        plan = self.label_controller.save_plan()
        if plan.stale:
            QMessageBox.critical(
                self,
                SAVE_TITLE,
                f"{len(plan.stale)} label(s) changed on disk since you opened "
                f"them, so nothing has been saved:\n\n"
                + "\n".join(item.destination.name for item in plan.stale)
                + "\n\nRe-scan to see what is there now. Re-opening the "
                "lamella will discard the edits made here.",
            )
            return

        choice = self._ask_how_to_save(plan)
        if choice is None:
            return
        mark_corrected = choice
        allow_corrected = bool(plan.corrected)
        self._run_operation(
            SAVE_TITLE,
            lambda: self.label_controller.save(
                mark_corrected=mark_corrected, allow_corrected=allow_corrected
            ),
            self._labels_saved,
        )

    def _ask_how_to_save(self, plan) -> bool | None:
        """Ask whether to mark the saved labels corrected; None cancels.

        Marking corrected is a state change, so it is offered only while editing
        is switched on -- the same guard every other state change is behind.
        With it switched off the dialog says nothing about what marking would
        move: describing an option that is not on offer is what made this ask
        three questions at once.
        """
        editing = self.controller.editing_enabled
        lines = [f"Save {plan.summary()} in this lamella?"]
        spread = self._modality_note()
        if spread:
            lines.append(spread)
        if plan.corrected:
            lines.append(
                f"{len(plan.corrected)} of them replace a label already marked "
                "corrected."
            )
        approved = len(self.label_controller.approved_records()) if editing else 0
        if approved:
            lines.append(
                f"Marking corrected would cover {approved} image(s), and the "
                "labels they share."
            )

        note = "" if editing else (
            "To set their state to corrected as well, switch on Enable editing "
            "in the Curate tab."
        )
        dialog = SaveLabelsDialog(
            self, "\n\n".join(lines), note, offer_corrected=editing
        )
        if dialog.exec_() != QDialog.Accepted:
            return None
        return dialog.mark_corrected

    def _modality_note(self) -> str:
        """Return the line naming what is waiting per modality, or nothing.

        Only when more than one is involved. A save writes the whole lamella's
        edits, so the reviewer working through the SEM images has to be told
        that the FIB labels they touched an hour ago are going out too.
        """
        counts = self.label_controller.pending_modalities()
        if len(counts) < 2:
            return ""
        return "They are " + " and ".join(
            f"{count} {modality}" for modality, count in sorted(counts.items())
        ) + "."

    def _labels_saved(self, headline: str) -> None:
        """Record what a save did and bring the window back in step.

        Refreshed rather than rescanned. A save is the one write whose whole
        effect is already in memory when it finishes: the labels it created are
        named on their records, and declaring a group corrected writes a file
        and updates those records without moving anything. Reading the
        collection again to learn what is already known cost twenty seconds of
        the action taken most often, and threw away the prepared frames of the
        lamella being proofread on the way.
        """
        self.state_panel.show_last_change(f"{SAVE_TITLE}: {headline}")
        logger.info("%s: %s", SAVE_TITLE, headline)
        self._flash(headline, 4000)
        self.refresh(reload_lists=True)
        self.label_panel.refresh()

    def _schedule_label_loads(self) -> None:
        """Ask the Label tab's loader to prepare the frames around this one."""
        token = self.label_loader.next_token()
        requests = self.label_controller.pending_loads()
        if requests:
            self.request_label_load.emit(requests, token)

    def _on_label_loaded(self, token: int, request) -> None:
        """Redraw the canvas when the frame on screen finishes preparing."""
        if token != self.label_loader.token:
            return
        if request.frame_index == self.label_controller.frame_index:
            self.label_panel.draw_current()

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
