"""The Add data tab: see a project's descriptive metadata, and edit it.

Each field's value is shown beside the file that supplied it -- a value that
looks wrong is explained by which of the two YAML files it came from, and
nothing else surfaces that. The project list marks how much of each project is
described, so what is left to do is visible without opening every one.

Editing is locked until switched on, matching the curation view, and a
suggestion reaches a file only when the user saves.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
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

from ...metadata import split_values
from ..metadata_controller import (
    FIELDS,
    SUGGESTED_FIELDS,
    TODO_MARKER,
    VALUE_SEPARATOR,
    MetadataController,
    MetadataValidationError,
    ProjectSummary,
)
from .common import (
    ChipEditor,
    GripSplitter,
    ToggleSwitch,
    add_path_menu,
    chip_icon,
    match_anywhere,
)
from .frame_strip import MISMATCH, STATE_COLOURS
from .theme import BADGE_STYLES, SWATCH_HEIGHT, SWATCH_OUTLINE, SWATCH_WIDTH

logger = logging.getLogger(__name__)

#: How wide the completeness bar is. The same as a curation chip in the other
#: list, so the two line up rather than each sitting at its own size.
COMPLETENESS_WIDTH = SWATCH_WIDTH


#: Narrowest the project list and the actions under it are allowed to get.
#: The rows carry paths, and the buttons under them clip rather than elide, so
#: this is the width both are laid out to fit.
PROJECT_PANE_MIN_WIDTH = 360


def _completeness_icon(summary: ProjectSummary) -> QIcon:
    """Return a bar filled in proportion to how much of a project is described.

    A fill fraction rather than three colour buckets: how full the bar is *is*
    the message, so there is no second vocabulary to learn, and it separates a
    project with one field left from one with five.

    The two colours are the ones the curate list already uses for the same two
    ideas -- the corrected teal for finished, the mismatch amber for needing
    attention -- so a colour means the same thing in both lists.
    """
    pixmap = QPixmap(COMPLETENESS_WIDTH, SWATCH_HEIGHT)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        share = summary.filled / summary.total if summary.total else 0.0
        colour = STATE_COLOURS["corrected"] if summary.is_complete else MISMATCH
        painter.fillRect(0, 0, round(COMPLETENESS_WIDTH * share), SWATCH_HEIGHT, colour)
        painter.setPen(SWATCH_OUTLINE)
        painter.drawRect(0, 0, COMPLETENESS_WIDTH - 1, SWATCH_HEIGHT - 1)
    finally:
        # A painter outliving its pixmap crashes the process instead of
        # reporting it.
        painter.end()
    return chip_icon(pixmap)


def editor_text(editor: QWidget) -> str:
    """Return what an editor is showing, whichever kind of editor it is.

    As one string even where the editor holds several values, because that is
    what the resolved value formats to: the panel decides whether a field was
    edited by comparing the two, and a comparison of a list against a string is
    a field that always reports a change.
    """
    if isinstance(editor, ChipEditor):
        return VALUE_SEPARATOR.join(editor.values())
    text = editor.currentText() if isinstance(editor, QComboBox) else editor.text()
    return text.strip()


def set_editor_text(editor: QWidget, value: str) -> None:
    """Show *value* in an editor, whichever kind of editor it is."""
    if isinstance(editor, ChipEditor):
        editor.set_values(split_values(value))
    elif isinstance(editor, QComboBox):
        editor.setCurrentText(value)
    else:
        editor.setText(value)


def set_editor_suggestions(editor: QWidget, values: list[str]) -> None:
    """Offer the values already in use, whichever kind of editor this is."""
    if isinstance(editor, ChipEditor):
        editor.set_suggestions(values)
    elif isinstance(editor, QComboBox):
        editor.clear()
        editor.addItems(["", *values])


def _suggestion_box() -> QComboBox:
    """Return a box offering the values already in use, and accepting new ones.

    Typed text is never added to the list: the suggestions describe the
    collection, and a box that grew its own entries as you typed would offer
    back a typo you had just made.
    """
    box = QComboBox()
    box.setEditable(True)
    box.setInsertPolicy(QComboBox.NoInsert)
    box.lineEdit().setPlaceholderText("unset")
    match_anywhere(box.completer())
    return box


class MetadataPanel(QWidget):
    """Project list on the left, its fields on the right."""

    #: Emitted after a successful write, so the window can refresh elsewhere.
    metadata_changed = Signal()
    #: Emitted with the row the user right-clicked, to have its path copied or
    #: the folder opened. Resolving a path stays the controller's job.
    project_path_requested = Signal(int)
    project_reveal_requested = Signal(int)
    #: Emitted when the user asks what the scanned folder looks like, or asks
    #: to choose which of its directories become projects. Both start from the
    #: whole scanned collection, so the window supplies the root.
    survey_requested = Signal()
    init_requested = Signal()
    #: Emitted when the user asks to recompress every label in the collection.
    compress_requested = Signal()

    def __init__(self, controller: MetadataController) -> None:
        super().__init__()
        self.controller = controller
        self._loading = False
        self.root: Path | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        # In a splitter because the rows carry relative paths, which are long
        # and are the point: most projects in one folder share their metadata,
        # so the folder a row names is what tells you what to type.
        self.panes = GripSplitter(Qt.Horizontal)
        self.panes.addWidget(self._build_projects())
        self.panes.addWidget(self._build_fields())
        self.panes.setSizes([460, 900])
        layout.addWidget(self.panes)

    def _build_projects(self) -> QWidget:
        box = QGroupBox("Projects")
        box.setMinimumWidth(PROJECT_PANE_MIN_WIDTH)
        column = QVBoxLayout(box)
        self.projects = QListWidget()
        # Elided in the middle rather than the end: these paths differ in the
        # timestamp they end with, so cutting the tail off makes a run of
        # projects from one session indistinguishable. The tooltip carries the
        # whole label regardless.
        #
        # The horizontal scrollbar has to go for that to happen at all -- with
        # one, the view scrolls instead of eliding, and a row is simply cut off
        # at the edge.
        self.projects.setTextElideMode(Qt.ElideMiddle)
        self.projects.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.projects.currentRowChanged.connect(self._select)
        add_path_menu(
            self.projects, "project",
            self.project_path_requested, self.project_reveal_requested,
        )
        column.addWidget(self.projects)
        column.addWidget(self._build_collection_actions())
        return box

    def _build_collection_actions(self) -> QWidget:
        """Return the steps that act on the folder rather than on one project.

        The first two come before this list has rows in it, which is why they
        sit under the list rather than beside the fields: a collection with no
        metadata files has nothing to select, and naming its projects is what
        puts the first row there. Compressing joins them because it has no
        reason to be narrower -- see its tooltip.

        Two to a row rather than three across. A button clips its caption
        rather than eliding it, and clipping takes the ends off both sides at
        once -- so three of these in :data:`PROJECT_PANE_MIN_WIDTH` read as
        "nspect layou". Two fit, and the grid keeps them the same width as each
        other whatever they are called.
        """
        self.collection_box = box = QGroupBox("Whole collection")
        grid = QGridLayout(box)

        self.survey_button = QPushButton("Inspect layout")
        self.survey_button.setToolTip(
            "Report how the scanned folder is interpreted and what does not "
            "fit. Reads nothing but the directory structure, and changes nothing."
        )
        self.survey_button.clicked.connect(self.survey_requested.emit)
        grid.addWidget(self.survey_button, 0, 0)

        self.init_button = QPushButton("Add projects...")
        self.init_button.setToolTip(
            "Pick which directories become projects, and give those that have "
            "none an AM_metadata.yaml. Unindexed folders are proposed for you "
            "to confirm; an existing file is never overwritten."
        )
        self.init_button.clicked.connect(self.init_requested.emit)
        grid.addWidget(self.init_button, 0, 1)

        self.compress_button = QPushButton("Compress labels...")
        self.compress_button.setToolTip(
            "Re-encode every label under this folder so it takes less disk "
            "space. Pixel values and dimensions are unchanged, and a file that "
            "would not shrink is left alone."
        )
        self.compress_button.clicked.connect(self.compress_requested.emit)
        grid.addWidget(self.compress_button, 1, 0)

        # The list above shows what is indexed, which says nothing about what
        # is sitting in the folder unindexed -- the state a collection is in
        # before any of this has been done, and the one the list cannot show.
        self.unindexed = QLabel("")
        self.unindexed.setWordWrap(True)
        self.unindexed.setStyleSheet(f"color: {MISMATCH.name()};")
        grid.addWidget(self.unindexed, 2, 0, 1, 2)

        self._update_collection_buttons()
        return box

    def set_unindexed(self, count: int, images: int) -> None:
        """Say how many directories under the folder are not projects yet."""
        self.unindexed.setText(
            ""
            if not count
            else (
                f"{count} directory(ies) here hold {images} image(s) and are not "
                f"projects yet. Add projects... names them."
            )
        )

    def set_root(self, root) -> None:
        """Name the folder the two collection actions would act on.

        Named rather than left implicit because these act on everything under
        it, and the tab otherwise never says which folder is loaded.
        """
        self.root = root
        self.collection_box.setTitle(
            "Whole collection" if root is None else f"Whole collection: {root.name}"
        )
        self._update_collection_buttons()

    def _update_collection_buttons(self) -> None:
        """Offer the collection actions once they have a folder to act on.

        Inspecting the layout and naming projects are offered as soon as a
        folder is loaded; recompressing waits on the editing guard as well.
        Naming projects still writes nothing until the guard is answered --
        the window says so when the button is pressed. It is offered here
        because a *disabled* button cannot say what it is waiting for, and
        this is the one action a collection with no projects in it needs: its
        guard sits beside fields that an empty project list never fills.
        """
        has_root = self.root is not None
        self.survey_button.setEnabled(has_root)
        self.init_button.setEnabled(has_root)
        self.compress_button.setEnabled(has_root and self.controller.editing_enabled)

    def _build_fields(self) -> QWidget:
        box = QGroupBox("Metadata")
        column = QVBoxLayout(box)

        self.summary = QLabel("No project selected")
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)

        form = QFormLayout()
        self.editors: dict[str, QWidget] = {}
        self.notes: dict[str, QLabel] = {}
        self.captions: dict[str, QLabel] = {}
        # Which editor a field gets is the field table's to say: a vocabulary
        # is picked from, a value that repeats across projects is offered back,
        # and anything else is typed.
        for entry in FIELDS:
            if entry.multi:
                editor: QWidget = ChipEditor(separator=VALUE_SEPARATOR)
            elif entry.choices:
                editor = QComboBox()
                editor.setEditable(False)
                editor.addItems(["", *entry.choices, TODO_MARKER])
            elif entry.suggested:
                editor = _suggestion_box()
            else:
                editor = QLineEdit()
                editor.setPlaceholderText("unset")

            note = QLabel("")
            caption = QLabel(entry.label)
            for widget in (caption, editor):
                widget.setToolTip(entry.guidance)

            row = QHBoxLayout()
            row.addWidget(editor, stretch=1)
            row.addWidget(note)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(caption, holder)
            self.editors[entry.name] = editor
            self.notes[entry.name] = note
            self.captions[entry.name] = caption
        column.addLayout(form)

        self.enable = ToggleSwitch("Enable editing")
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

    def set_projects(self, summaries: list[ProjectSummary]) -> None:
        """List the projects, selecting the first."""
        self._loading = True
        try:
            self.projects.clear()
            self.projects.addItems([summary.label for summary in summaries])
            self._show_completeness(summaries)
            if summaries:
                self.projects.setCurrentRow(0)
        finally:
            self._loading = False
        self.refresh()

    def _show_completeness(self, summaries: list[ProjectSummary]) -> None:
        """Mark each row with how much of its metadata is filled in.

        The row text is already carrying a path, so the count and the fields
        still wanted go in the tooltip -- which is also where the full label
        goes, since the list elides long ones.
        """
        for row, summary in enumerate(summaries):
            item = self.projects.item(row)
            if item is None:
                break
            item.setIcon(_completeness_icon(summary))
            item.setToolTip(f"{summary.label}\n{summary.describe()}")

    def step_project(self, delta: int) -> None:
        """Move *delta* rows through the project list, stopping at its ends.

        Walking the collection a project at a time is the way this tab is used,
        so it is worth a key rather than a click per project.
        """
        count = self.projects.count()
        if not count:
            return
        row = max(0, min(self.projects.currentRow() + delta, count - 1))
        self.projects.setCurrentRow(row)

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
        self._update_collection_buttons()

    def _edits(self) -> dict[str, str | None]:
        """Return the fields whose editor differs from the resolved value.

        Only differences are sent, so saving does not rewrite fields nobody
        touched -- and an empty box means "remove this", which is a real
        operation distinct from leaving it alone.

        Anything half-typed becomes a value first. A chip editor holds its box
        until the value is confirmed, and someone who types one and reaches
        straight for Save means it.
        """
        for editor in self.editors.values():
            if isinstance(editor, ChipEditor):
                editor.commit_pending()
        current = self.controller.values()
        edits: dict[str, str | None] = {}
        for field, editor in self.editors.items():
            text = editor_text(editor)
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

    def _show_note(self, field: str, source: str, outstanding: bool) -> None:
        """Annotate one field with where its value came from, or that it is due.

        The two never coincide -- a field still holding a placeholder resolves
        to no value and so has no source -- but they are not the same kind of
        remark, and drawing an outstanding field in the dimmest colour on the
        panel put the state that matters most where it was hardest to see.
        """
        note = self.notes[field]
        if outstanding:
            note.setText(f"[{TODO_MARKER}]")
            note.setStyleSheet(BADGE_STYLES["warning"])
            note.setToolTip("Still a placeholder; nobody has filled this in")
        elif source:
            note.setText(f"[{source}]")
            note.setStyleSheet(BADGE_STYLES["info"])
            note.setToolTip("Which file this value came from")
        else:
            note.setText("")
            note.setToolTip("")

    def refresh(self) -> None:
        """Redraw the fields from the controller."""
        self._loading = True
        try:
            values = self.controller.values()
            sources = self.controller.sources()
            outstanding = set(self.controller.outstanding())
            for field, editor in self.editors.items():
                value = values.get(field, "")
                if field in SUGGESTED_FIELDS:
                    # Rebuilt every time, so a value saved on one project is
                    # offered on the next without rescanning.
                    set_editor_suggestions(editor, self.controller.suggestions(field))
                set_editor_text(editor, value)
                self._show_note(field, sources.get(field, ""), field in outstanding)
            # Saving is exactly when a project stops being incomplete, so the
            # marks in the list must not be left describing the state before it.
            self._show_completeness(self.controller.project_summaries())
            self.summary.setText(self.controller.summary())
        finally:
            self._loading = False
