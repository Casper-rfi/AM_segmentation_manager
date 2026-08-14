"""Left panel: choosing what to review.

Holds the scan root, the projects found under it, and the lamellae of the
selected project. Scanning happens on the caller's terms -- this panel only
reports that the user asked for it.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..controller import CurationStatus
from .frame_strip import MISMATCH, STATE_COLOURS


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

    def __init__(self) -> None:
        super().__init__()
        self._emitting = False
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
        )
        self.project_filter.textChanged.connect(
            lambda text: self._apply_filter(self.project_list, self._project_labels, text)
        )
        self._add_path_menu(
            self.project_list, "project",
            self.project_path_requested, self.project_reveal_requested,
        )
        self.lamella_box, self.lamella_filter, self.lamella_list = self._build_list(
            "Lamellae", "Filter lamellae", self._lamella_changed,
        )
        self.lamella_filter.textChanged.connect(
            lambda text: self._apply_filter(self.lamella_list, self._lamella_labels, text)
        )
        self._add_path_menu(
            self.lamella_list, "lamella",
            self.lamella_path_requested, self.lamella_reveal_requested,
        )

        lists = QSplitter(Qt.Vertical)
        lists.addWidget(self.project_box)
        lists.addWidget(self.lamella_box)
        lists.setSizes([200, 300])
        layout.addWidget(lists, stretch=1)

        self.summary = QLabel("No project loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def _build_list(self, title: str, placeholder: str, on_change):
        """Build a titled, filterable list box."""
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 4, 6, 6)
        box_layout.setSpacing(3)

        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText(placeholder)
        filter_edit.setClearButtonEnabled(True)
        filter_edit.setToolTip("Show only entries whose name contains this text")
        box_layout.addWidget(filter_edit)

        list_widget = QListWidget()
        list_widget.currentRowChanged.connect(on_change)
        box_layout.addWidget(list_widget)
        return box, filter_edit, list_widget

    def _add_path_menu(
        self, widget: QListWidget, noun: str, copy_signal, reveal_signal
    ) -> None:
        """Give a list a right-click menu for the row's filesystem path.

        The row under the cursor is used, not the selected row: right-clicking
        an entry and acting on a different one would be a surprising way to lose
        track of which path you copied.
        """
        widget.setContextMenuPolicy(Qt.CustomContextMenu)

        def show_menu(point) -> None:
            item = widget.itemAt(point)
            if item is None:
                return
            row = widget.row(item)
            menu = QMenu(widget)
            copy = menu.addAction(f"Copy {noun} path")
            reveal = menu.addAction(f"Reveal {noun} in file manager")
            chosen = menu.exec_(widget.viewport().mapToGlobal(point))
            if chosen is copy:
                copy_signal.emit(row)
            elif chosen is reveal:
                reveal_signal.emit(row)

        widget.customContextMenuRequested.connect(show_menu)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder to scan", self.path_edit.text() or str(Path.home())
        )
        if directory:
            self.path_edit.setText(directory)
            self.scan_requested.emit(directory)

    def _project_changed(self, row: int) -> None:
        if not self._emitting and row >= 0:
            self.project_selected.emit(row)

    def _lamella_changed(self, row: int) -> None:
        if not self._emitting and row >= 0:
            self.lamella_selected.emit(row)

    def set_path(self, path: str) -> None:
        """Show *path* as the scan root."""
        self.path_edit.setText(path)

    def show_projects(
        self, labels: list[str], current: int, statuses: list[CurationStatus] | None = None
    ) -> None:
        """List the discovered projects, selecting the one at *current*.

        *statuses* adds a curation-progress chip to each row; omitting it lists
        the projects with no chips, which is what happens before a scan.
        """
        self._project_labels = list(labels)
        self._set_items(self.project_list, labels, current, statuses)
        self.project_box.setTitle(f"Projects ({len(labels)})" if labels else "Projects")
        self._apply_filter(self.project_list, self._project_labels, self.project_filter.text())

    def show_lamellae(
        self, labels: list[str], current: int, statuses: list[CurationStatus] | None = None
    ) -> None:
        """List the selected project's lamellae, with their curation progress."""
        self._lamella_labels = list(labels)
        self._set_items(self.lamella_list, labels, current, statuses)
        self.lamella_box.setTitle(f"Lamellae ({len(labels)})" if labels else "Lamellae")
        self._apply_filter(self.lamella_list, self._lamella_labels, self.lamella_filter.text())

    def show_summary(self, text: str) -> None:
        """Show a short description of what is loaded."""
        self.summary.setText(text)

    def _set_items(
        self,
        widget: QListWidget,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
    ) -> None:
        """Replace a list's contents without echoing selection signals back."""
        self._emitting = True
        try:
            widget.clear()
            widget.addItems(labels)
            for row, status in enumerate(statuses or []):
                item = widget.item(row)
                if item is None:
                    break
                item.setIcon(_status_icon(status))
                # The chip cannot carry the whole meaning on its own, so the
                # long form goes in the tooltip rather than into the row text,
                # which is already carrying a path.
                item.setToolTip(f"{item.text()}\n{status.summary}")
            if labels:
                widget.setCurrentRow(max(0, min(current, len(labels) - 1)))
        finally:
            self._emitting = False

    def _apply_filter(self, widget: QListWidget, labels: list[str], text: str) -> None:
        """Hide entries that do not match *text*.

        Entries are hidden rather than removed so that a row number still means
        the same project or lamella whether or not a filter is active; rebuilding
        the list would require mapping filtered rows back to real indices.
        """
        needle = text.strip().lower()
        for row in range(widget.count()):
            item = widget.item(row)
            label = labels[row] if row < len(labels) else item.text()
            item.setHidden(bool(needle) and needle not in label.lower())


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
    "needs repair": MISMATCH,
}

#: Nothing is drawn for a modality with no images. An absent modality should
#: read as absent rather than as a state, and many older datasets have no FIB.
STATUS_EMPTY = "empty"

#: Every chip gets a dark outline. Without one the "not started" blue is nearly
#: the same colour as a list row's selection highlight, so the chip on the row
#: you are looking at -- the one you most want to read -- disappeared. An
#: outline makes a chip legible on any background rather than tuning the fill
#: against one particular theme's highlight colour.
SWATCH_OUTLINE = QColor("#1a1a1a")

SWATCH_WIDTH = 14
SWATCH_HEIGHT = 10


def _status_icon(status: CurationStatus) -> QIcon:
    """Return a two-part chip: SEM on the left, FIB on the right.

    One icon rather than two so a list row stays one line, and split rather
    than combined because a lamella whose SEM work is finished and whose FIB has
    not started is a normal situation that one mark cannot express.
    """
    pixmap = QPixmap(SWATCH_WIDTH * 2 + 2, SWATCH_HEIGHT)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        for index, modality in enumerate(("SEM", "FIB")):
            left = index * (SWATCH_WIDTH + 2)
            _paint_status(painter, left, status.of(modality))
    finally:
        # status.of() raises on an unexpected modality, and a painter left
        # alive past its pixmap crashes the process instead of reporting it.
        painter.end()
    return QIcon(pixmap)


def _paint_status(painter: QPainter, left: int, status: str) -> None:
    """Paint one modality's chip at *left*, outlined so it reads anywhere."""
    if status == STATUS_EMPTY:
        return
    if status == "in progress":
        # Half started, half finished -- reads as "between the two" without
        # needing a fourth colour explained anywhere.
        half = SWATCH_WIDTH // 2
        painter.fillRect(left, 0, half, SWATCH_HEIGHT, STATUS_COLOURS["not started"])
        painter.fillRect(
            left + half, 0, SWATCH_WIDTH - half, SWATCH_HEIGHT, STATUS_COLOURS["done"]
        )
    else:
        painter.fillRect(
            left, 0, SWATCH_WIDTH, SWATCH_HEIGHT,
            STATUS_COLOURS.get(status, QColor(Qt.transparent)),
        )
    painter.setPen(SWATCH_OUTLINE)
    painter.drawRect(left, 0, SWATCH_WIDTH - 1, SWATCH_HEIGHT - 1)
