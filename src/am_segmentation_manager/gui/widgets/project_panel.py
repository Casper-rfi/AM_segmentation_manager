"""Left panel: choosing what to review.

Holds the scan root, the projects found under it, and the lamellae of the
selected project. Scanning happens on the caller's terms -- this panel only
reports that the user asked for it.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
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
        # One row, no group box. A "Scan root" title said the same thing as the
        # field's placeholder and the Scan button beside it -- three statements
        # of one fact across two rows. The placeholder carries the explanation
        # where it is useful (visible while the field is empty, gone once a path
        # is in it), and the tooltips carry what neither a label nor a button
        # caption can say.
        # The path gets a row to itself. Putting it beside both buttons was
        # tried and left roughly 140px for the field in a 270px panel, which
        # showed the tail of a path and nothing else -- and reading the path is
        # the entire reason the field is there. The buttons share a second row,
        # right-aligned so they read as actions on the field above.
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

    def show_projects(self, labels: list[str], current: int) -> None:
        """List the discovered projects, selecting the one at *current*."""
        self._project_labels = list(labels)
        self._set_items(self.project_list, labels, current)
        self.project_box.setTitle(f"Projects ({len(labels)})" if labels else "Projects")
        self._apply_filter(self.project_list, self._project_labels, self.project_filter.text())

    def show_lamellae(self, labels: list[str], current: int) -> None:
        """List the selected project's lamellae."""
        self._lamella_labels = list(labels)
        self._set_items(self.lamella_list, labels, current)
        self.lamella_box.setTitle(f"Lamellae ({len(labels)})" if labels else "Lamellae")
        self._apply_filter(self.lamella_list, self._lamella_labels, self.lamella_filter.text())

    def show_summary(self, text: str) -> None:
        """Show a short description of what is loaded."""
        self.summary.setText(text)

    def _set_items(self, widget: QListWidget, labels: list[str], current: int) -> None:
        """Replace a list's contents without echoing selection signals back."""
        self._emitting = True
        try:
            widget.clear()
            widget.addItems(labels)
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
