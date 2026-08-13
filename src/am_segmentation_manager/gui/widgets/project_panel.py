"""Left panel: choosing what to review.

Holds the scan root, the projects found under it, and the lamellae of the
selected project. Scanning happens on the caller's terms -- this panel only
reports that the user asked for it.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
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

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # -- scan root -----------------------------------------------------
        root_box = QGroupBox("Scan root")
        root_layout = QVBoxLayout(root_box)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder to scan for Autolamella projects")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        root_layout.addLayout(row)

        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(
            lambda: self.scan_requested.emit(self.path_edit.text())
        )
        root_layout.addWidget(self.scan_button)
        layout.addWidget(root_box)

        # -- projects ------------------------------------------------------
        project_box = QGroupBox("Projects")
        project_layout = QVBoxLayout(project_box)
        self.project_list = QListWidget()
        self.project_list.currentRowChanged.connect(self._project_changed)
        project_layout.addWidget(self.project_list)
        layout.addWidget(project_box)

        # -- lamellae ------------------------------------------------------
        lamella_box = QGroupBox("Lamellae")
        lamella_layout = QVBoxLayout(lamella_box)
        self.lamella_list = QListWidget()
        self.lamella_list.currentRowChanged.connect(self._lamella_changed)
        lamella_layout.addWidget(self.lamella_list)
        layout.addWidget(lamella_box)

        self.summary = QLabel("No project loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self._emitting = False

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
        self._set_items(self.project_list, labels, current)

    def show_lamellae(self, labels: list[str], current: int) -> None:
        """List the selected project's lamellae."""
        self._set_items(self.lamella_list, labels, current)

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
