"""Left panel: choosing what to review, and what to change.

Holds the scan root, the projects found under it, and the lamellae of the
selected project. Scanning happens on the caller's terms -- this panel only
reports that the user asked for it. Which *rows* a change covers is chosen
here; which modality, and which state, are chosen beside the buttons that act.

The lists do two jobs at once: the *current* row is what is being looked at,
the *checked* rows are what a state change applies to, and the chip beside them
reports curation progress. Three separate questions, three separate marks --
see `docs/decisions.md` § Workflow state.

Only one of the two lists carries ticks at a time, and the ticked lamellae are
the selected project's own -- so everything a change will touch is a tick
visible on screen.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..controller import CurationStatus
from .common import CheckList, add_path_menu, chip_icon, choose_directory, ticked_caption
from .frame_strip import STATE_COLOURS
from .theme import (
    SWATCH_BAND,
    SWATCH_GAP,
    SWATCH_HEIGHT,
    SWATCH_OUTLINE,
    SWATCH_WIDTH,
)


class ProjectPanel(QWidget):
    """Scan root, project list and lamella list."""

    #: Emitted with the directory to scan.
    scan_requested = Signal(str)
    #: Emitted with the index of the project the user picked.
    project_selected = Signal(int)
    #: Emitted with the index of the lamella the user picked.
    lamella_selected = Signal(int)
    #: Emitted with the row the user right-clicked, to have its path copied.
    #: The panel holds display strings, not model objects, so it asks for the
    #: path by row rather than resolving one itself -- that stays the window's
    #: job, via the controller.
    project_path_requested = Signal(int)
    lamella_path_requested = Signal(int)
    #: As above, but to open the folder in the desktop file manager.
    project_reveal_requested = Signal(int)
    lamella_reveal_requested = Signal(int)
    #: Emitted with a row and whether it is now ticked.
    project_checked = Signal(int, bool)
    lamella_checked = Signal(int, bool)
    #: Emitted with whether to tick or clear every row of that list.
    projects_check_all = Signal(bool)
    lamellae_check_all = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._project_labels: list[str] = []
        self._lamella_labels: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 2, 4)
        layout.setSpacing(4)

        # -- scan root -----------------------------------------------------
        # No group box. A "Scan root" title said the same thing as the field's
        # placeholder and the Scan button beside it -- one fact stated three
        # times. The placeholder carries the explanation where it is useful
        # (visible while the field is empty, gone once a path is in it), and the
        # tooltips carry what neither a label nor a button caption can say.
        #
        # The path keeps a row to itself. Putting it beside both buttons was
        # tried and left roughly 140px for the field in a 270px panel, showing
        # the tail of a path and nothing else -- and reading the path is the
        # entire reason the field is there.
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder to scan for Autolamella projects")
        self.path_edit.setToolTip(
            "Folder searched recursively for Autolamella projects "
            "(any folder containing an experiment.yaml or AM_metadata.yaml)"
        )
        self.path_edit.returnPressed.connect(
            lambda: self.scan_requested.emit(self.path_edit.text())
        )
        layout.addWidget(self.path_edit)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addStretch(1)

        browse = QPushButton("Browse...")
        browse.setToolTip("Choose a folder to scan")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)

        # Not full width any more: Scan is not the window's primary action, and
        # a full-width button implied that it was.
        self.scan_button = QPushButton("Scan")
        self.scan_button.setToolTip(
            "Re-read the folder from disk. Use this after changing files "
            "outside the window."
        )
        self.scan_button.clicked.connect(
            lambda: self.scan_requested.emit(self.path_edit.text())
        )
        row.addWidget(self.scan_button)

        layout.addLayout(row)

        # -- projects and lamellae -----------------------------------------
        # In a splitter so that an archive with many projects but few lamellae
        # (or the reverse) can be given the space where it is needed.
        self.project_box, self.project_filter, self.project_list = self._build_list(
            "Projects", "Filter projects", self._project_changed,
            self.project_checked, self.projects_check_all,
        )
        self.project_filter.textChanged.connect(
            lambda text: self._apply_filter(self.project_list, self._project_labels, text)
        )
        add_path_menu(
            self.project_list, "project",
            self.project_path_requested, self.project_reveal_requested,
        )
        self.lamella_box, self.lamella_filter, self.lamella_list = self._build_list(
            "Lamellae", "Filter lamellae", self._lamella_changed,
            self.lamella_checked, self.lamellae_check_all,
        )
        self.lamella_filter.textChanged.connect(
            lambda text: self._apply_filter(self.lamella_list, self._lamella_labels, text)
        )
        add_path_menu(
            self.lamella_list, "lamella",
            self.lamella_path_requested, self.lamella_reveal_requested,
        )

        self.lists = QSplitter(Qt.Vertical)
        self.lists.addWidget(self.project_box)
        self.lists.addWidget(self.lamella_box)
        self.lists.setSizes([200, 300])
        layout.addWidget(self.lists, stretch=1)

        self.summary = QLabel("No project loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def _build_list(
        self, title: str, placeholder: str, on_change, checked_signal, check_all_signal
    ):
        """Build a titled, filterable list box with tick-all controls."""
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 4, 6, 6)
        box_layout.setSpacing(3)

        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText(placeholder)
        filter_edit.setClearButtonEnabled(True)
        filter_edit.setToolTip("Show only entries whose name contains this text")
        box_layout.addWidget(filter_edit)

        list_widget = CheckList()
        list_widget.setToolTip(
            "Click a row to look at it. Tick rows to choose what a change "
            "applies to; ticking here clears the other list. With nothing "
            "ticked it applies to the frame on screen."
        )
        list_widget.currentRowChanged.connect(on_change)
        list_widget.row_ticked.connect(checked_signal)
        box_layout.addWidget(list_widget)

        box_layout.addLayout(self._build_check_all(check_all_signal, title.lower()))
        return box, filter_edit, list_widget

    def _build_check_all(self, signal, noun: str) -> QHBoxLayout:
        """Return the row of tick-everything / tick-nothing buttons."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for caption, checked, hint in (
            ("Tick all", True, f"Choose every one of the {noun} listed"),
            ("None", False, f"Clear the ticks on all {noun}"),
        ):
            button = QPushButton(caption)
            button.setToolTip(hint)
            button.clicked.connect(lambda _checked=False, v=checked: signal.emit(v))
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _browse(self) -> None:
        directory = choose_directory(
            self, "Choose a folder to scan", self.path_edit.text() or None
        )
        if directory is not None:
            self.path_edit.setText(str(directory))
            self.scan_requested.emit(str(directory))

    def _project_changed(self, row: int) -> None:
        if row >= 0:
            self.project_selected.emit(row)

    def _lamella_changed(self, row: int) -> None:
        if row >= 0:
            self.lamella_selected.emit(row)

    def set_path(self, path: str) -> None:
        """Show *path* as the scan root."""
        self.path_edit.setText(path)

    def show_projects(
        self,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
    ) -> None:
        """List the discovered projects, selecting the one at *current*.

        *statuses* adds a curation-progress chip to each row; omitting it lists
        the projects with no chips, which is what happens before a scan.
        *checks* says which rows are ticked.
        """
        self._project_labels = list(labels)
        self._set_items(self.project_list, labels, current, statuses, checks)
        self.project_box.setTitle(
            self._list_title("Projects", labels, checks)
        )
        self._apply_filter(self.project_list, self._project_labels, self.project_filter.text())

    def show_lamellae(
        self,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
    ) -> None:
        """List the selected project's lamellae, with their curation progress."""
        self._lamella_labels = list(labels)
        self._set_items(self.lamella_list, labels, current, statuses, checks)
        self.lamella_box.setTitle(
            self._list_title("Lamellae", labels, checks)
        )
        self._apply_filter(self.lamella_list, self._lamella_labels, self.lamella_filter.text())

    def show_checks(
        self, project_checks: list[bool], lamella_checks: list[bool]
    ) -> None:
        """Put both lists' ticks and titles back, leaving the rows alone.

        Ticking a row must not rebuild the list: that would reset which row is
        current, and ticking is not a change of what is being looked at. Both
        lists are refreshed on every tick because ticking in one clears the
        other, and the user has to see that happen.
        """
        self.project_list.show_ticks(project_checks)
        self.lamella_list.show_ticks(lamella_checks)
        self.project_box.setTitle(
            self._list_title("Projects", self._project_labels, project_checks)
        )
        self.lamella_box.setTitle(
            self._list_title("Lamellae", self._lamella_labels, lamella_checks)
        )

    @staticmethod
    def _list_title(noun: str, labels: list[str], checks: list[bool] | None) -> str:
        """Return the box title, saying how many rows are ticked when any are."""
        return ticked_caption(
            noun, len(labels), sum(1 for checked in checks or [] if checked)
        )

    def show_summary(self, text: str) -> None:
        """Show a short description of what is loaded."""
        self.summary.setText(text)

    def announce(self, text: str) -> None:
        """Show *text* and paint it now, before the caller blocks.

        Scanning holds the interface thread, so an ordinary ``setText`` would
        not reach the screen until the scan it is announcing had finished.
        ``repaint`` draws this one widget synchronously; it does not process
        events, so nothing is re-entered on the way -- the distinction
        :class:`~.busy_dialog.BusyDialog` exists to keep.
        """
        self.summary.setText(text)
        self.summary.repaint()

    def _set_items(
        self,
        widget: CheckList,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
    ) -> None:
        """Replace a list's contents, ticks and chips together.

        The chip cannot carry its whole meaning on its own, so the long form
        goes in the tooltip rather than into the row text, which is already
        carrying a path.
        """
        widget.set_values(
            labels,
            ticked=checks or [],
            icons=[_status_icon(status) for status in statuses or []],
            tooltips=[
                f"{label}\n{status.summary}"
                for label, status in zip(labels, statuses or [])
            ],
            current=current,
        )

    def _apply_filter(self, widget: CheckList, labels: list[str], text: str) -> None:
        """Hide entries that do not match *text*."""
        widget.hide_rows_without(text, labels)


#: A curation status mapped to the colour that represents it.
#:
#: Reuses the timeline's progress ramp rather than inventing a third colour
#: vocabulary for the same idea: a status here is exactly the states in that
#: row, summarised. "not started" is the selected blue because that is what an
#: untouched image is; "done" is the corrected teal; "in progress" is drawn as
#: a split of the two rather than given a colour of its own.
STATUS_COLOURS = {
    "not started": STATE_COLOURS["selected"],
    "done": STATE_COLOURS["corrected"],
}

#: Nothing is drawn for a modality with no images. An absent modality should
#: read as absent rather than as a state, and many older datasets have no FIB.
STATUS_EMPTY = "empty"

def _status_icon(status: CurationStatus) -> QIcon:
    """Return a two-band chip: SEM above, FIB below.

    One icon rather than two so a list row stays one line, and split rather
    than combined because a lamella whose SEM work is finished and whose FIB has
    not started is a normal situation that one mark cannot express. Stacked
    rather than side by side; :data:`SWATCH_WIDTH` says why.
    """
    pixmap = QPixmap(SWATCH_WIDTH, SWATCH_HEIGHT)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        for index, modality in enumerate(("SEM", "FIB")):
            top = index * (SWATCH_BAND + SWATCH_GAP)
            _paint_status(painter, top, status.of(modality))
    finally:
        # status.of() raises on an unexpected modality, and a painter left
        # alive past its pixmap crashes the process instead of reporting it.
        painter.end()
    return chip_icon(pixmap)


def _paint_status(painter: QPainter, top: int, status: str) -> None:
    """Paint one modality's band at *top*, outlined so it reads anywhere."""
    if status == STATUS_EMPTY:
        return
    if status == "in progress":
        # Part started, part finished, filling left to right like any progress
        # bar -- which reads as "between the two" without needing a fourth
        # colour explained anywhere.
        half = SWATCH_WIDTH // 2
        painter.fillRect(0, top, half, SWATCH_BAND, STATUS_COLOURS["done"])
        painter.fillRect(
            half, top, SWATCH_WIDTH - half, SWATCH_BAND, STATUS_COLOURS["not started"]
        )
    else:
        painter.fillRect(
            0, top, SWATCH_WIDTH, SWATCH_BAND,
            STATUS_COLOURS.get(status, QColor(Qt.transparent)),
        )
    painter.setPen(SWATCH_OUTLINE)
    painter.drawRect(0, top, SWATCH_WIDTH - 1, SWATCH_BAND - 1)
