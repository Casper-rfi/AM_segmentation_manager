"""The Metadata tab: see a project's descriptive metadata, and edit it.

Shows each field's resolved value beside the file that supplied it, which is
most of the point -- a value that looks wrong is explained by which of the two
YAML files it came from, and nothing else in the application surfaces that.

Editing is locked until it is switched on, matching the curation view: this is
mostly used for reading, and a text field you can change by accident is a poor
guard on a file people have typed into.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..metadata_controller import (
    DESCRIPTIVE_FIELDS,
    LAMELLA_TYPES,
    TODO_MARKER,
    MetadataController,
    MetadataValidationError,
)

logger = logging.getLogger(__name__)


class MetadataPanel(QWidget):
    """Project list on the left, its fields on the right."""

    #: Emitted after a successful write, so the window can refresh elsewhere.
    metadata_changed = Signal()

    def __init__(self, controller: MetadataController) -> None:
        super().__init__()
        self.controller = controller
        self._loading = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        layout.addWidget(self._build_projects(), stretch=0)
        layout.addWidget(self._build_fields(), stretch=1)

    def _build_projects(self) -> QWidget:
        box = QGroupBox("Projects")
        box.setMinimumWidth(280)
        column = QVBoxLayout(box)
        self.projects = QListWidget()
        self.projects.currentRowChanged.connect(self._select)
        column.addWidget(self.projects)
        return box

    def _build_fields(self) -> QWidget:
        box = QGroupBox("Metadata")
        column = QVBoxLayout(box)

        self.summary = QLabel("No project selected")
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)

        form = QFormLayout()
        self.editors: dict[str, QWidget] = {}
        self.sources: dict[str, QLabel] = {}
        for field in DESCRIPTIVE_FIELDS:
            if field == "lamella_type":
                editor: QWidget = QComboBox()
                editor.setEditable(False)
                editor.addItems(["", *LAMELLA_TYPES, TODO_MARKER])
            else:
                editor = QLineEdit()
                editor.setPlaceholderText("unset")
            source = QLabel("")
            source.setStyleSheet("color: palette(mid);")
            source.setToolTip("Which file this value came from")

            row = QHBoxLayout()
            row.addWidget(editor, stretch=1)
            row.addWidget(source)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(field.replace("_", " ").title(), holder)
            self.editors[field] = editor
            self.sources[field] = source
        column.addLayout(form)

        self.enable = QCheckBox("Enable editing")
        self.enable.setToolTip(
            "Unlock the fields. Only AM_metadata.yaml is written; "
            "experiment.yaml is never touched."
        )
        self.enable.toggled.connect(self._set_editing)
        column.addWidget(self.enable)

        buttons = QHBoxLayout()
        self.preview_button = QPushButton("Preview changes")
        self.preview_button.clicked.connect(self._preview)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        column.addLayout(buttons)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.status)
        column.addStretch(1)

        self._set_editing(False)
        return box

    # -- behaviour ---------------------------------------------------------

    def set_projects(self, labels: list[str]) -> None:
        self._loading = True
        try:
            self.projects.clear()
            self.projects.addItems(labels)
            if labels:
                self.projects.setCurrentRow(0)
        finally:
            self._loading = False
        self.refresh()

    def _select(self, row: int) -> None:
        if self._loading or row < 0:
            return
        self.controller.select(row)
        self.refresh()

    def _set_editing(self, enabled: bool) -> None:
        self.controller.editing_enabled = enabled
        for editor in self.editors.values():
            editor.setEnabled(enabled)
        self.save_button.setEnabled(enabled)

    def _edits(self) -> dict[str, str | None]:
        """Return the fields whose editor differs from the resolved value.

        Only differences are sent, so saving does not rewrite fields nobody
        touched -- and an empty box means "remove this", which is a real
        operation distinct from leaving it alone.
        """
        current = self.controller.values()
        edits: dict[str, str | None] = {}
        for field, editor in self.editors.items():
            text = (
                editor.currentText() if isinstance(editor, QComboBox) else editor.text()
            ).strip()
            if text == (current.get(field) or ""):
                continue
            edits[field] = text or None
        return edits

    def _preview(self) -> None:
        edits = self._edits()
        if not edits:
            self.status.setText("No changes")
            return
        try:
            change = self.controller.plan(edits)
        except (MetadataValidationError, ValueError) as error:
            self.status.setText(f"Cannot apply: {error}")
            return
        self.status.setText("\n".join(change.describe()))

    def _save(self) -> None:
        edits = self._edits()
        if not edits:
            self.status.setText("No changes")
            return
        try:
            change = self.controller.apply(edits)
        except (MetadataValidationError, ValueError, PermissionError) as error:
            QMessageBox.warning(self, "Cannot save", str(error))
            self.status.setText(f"Not saved: {error}")
            return
        self.status.setText("Saved.\n" + "\n".join(change.describe()))
        self.refresh()
        self.metadata_changed.emit()

    def refresh(self) -> None:
        """Redraw the fields from the controller."""
        self._loading = True
        try:
            values = self.controller.values()
            sources = self.controller.sources()
            outstanding = set(self.controller.outstanding())
            for field, editor in self.editors.items():
                value = values.get(field, "")
                if isinstance(editor, QComboBox):
                    editor.setCurrentText(value)
                else:
                    editor.setText(value)
                note = sources.get(field, "")
                if field in outstanding:
                    note = TODO_MARKER
                self.sources[field].setText(f"[{note}]" if note else "")
            self.summary.setText(self.controller.summary())
        finally:
            self._loading = False
