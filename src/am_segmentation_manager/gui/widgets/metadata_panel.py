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
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...metadata import split_values
from ..metadata_controller import (
    FIELDS,
    MIXED_MARKER,
    SUGGESTED_FIELDS,
    TODO_MARKER,
    VALUE_SEPARATOR,
    MetadataController,
    MetadataValidationError,
    ProjectSummary,
)
from .common import chip_icon, match_anywhere
from .controls import ChipEditor, GripSplitter, ToggleSwitch
from .frame_strip import MISMATCH, STATE_COLOURS
from .lists import ProjectTree, add_fold_buttons
from .paths import add_path_menu
from .theme import (
    BADGE_STYLES,
    MUTED,
    SWATCH_HEIGHT,
    SWATCH_OUTLINE,
    SWATCH_WIDTH,
)

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


def _describe(changes: list) -> str:
    """Return what a set of metadata changes did, at the length it deserves.

    One project reads out in full, as it always did. A folder reads as a count
    and the fields touched: a dozen identical blocks say the same thing twelve
    times and bury whether any of them failed.
    """
    if len(changes) == 1:
        return "\n".join(changes[0].describe())
    fields = sorted({
        edit.field for change in changes for edit in change.changes
    })
    touched = sum(1 for change in changes if not change.is_empty)
    if not fields:
        return f"{len(changes)} projects: nothing to change"
    return (
        f"{touched} of {len(changes)} projects: {', '.join(fields)}"
    )


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
    match_anywhere(box)
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
    #: Emitted when the user asks to name the state of the whole collection.
    savepoint_requested = Signal()
    #: Emitted when the user asks to remove the kept versions nothing needs.
    reclaim_requested = Signal()

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
        # The same tree the Curate tab lists projects in: an Autolamella
        # project is named for the minute it started, and describing a
        # collection means working down the folder somebody filed them under.
        # Eliding, the scrollbar and the folds are all the widget's own.
        # No tick boxes: this tab describes the project being looked at, one
        # at a time, so a box would offer a set it cannot act on.
        self.projects = ProjectTree(checkable=False)
        self.projects.project_selected.connect(self._select)
        self.projects.folder_selected.connect(self._select_folder)
        add_path_menu(
            self.projects, "project",
            self.project_path_requested, self.project_reveal_requested,
            row_of=self.projects.project_of,
        )
        column.addWidget(self.projects)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        add_fold_buttons(row, self.projects)
        row.addStretch(1)
        column.addLayout(row)
        column.addWidget(self._build_collection_actions())
        return box

    def _build_collection_actions(self) -> QWidget:
        """Return the steps that act on the folder rather than on one project.

        The first two come before this list has rows in it, which is why they
        sit under the list rather than beside the fields: a collection with no
        metadata files has nothing to select, and naming its projects is what
        puts the first row there. Compressing and taking a savepoint join them
        because neither has anything to narrow to: both act on the whole
        collection, which is the rule that keeps them off Curate.

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

        # Beside the other two collection-wide acts rather than in Curate,
        # whose operations all narrow to what is ticked: a savepoint covers the
        # whole dataset a model was trained on and has nothing to narrow to.
        self.savepoint_button = QPushButton("Savepoint...")
        self.savepoint_button.setToolTip(
            "Record what every label under this folder holds now, under a name "
            "you give it.\n"
            "Revert on the Curate tab can put labels back to it later. Image "
            "states and project metadata are not covered."
        )
        self.savepoint_button.clicked.connect(self.savepoint_requested.emit)
        grid.addWidget(self.savepoint_button, 1, 1)

        self.reclaim_button = QPushButton("Reclaim space...")
        self.reclaim_button.setToolTip(
            "Remove the kept label versions no savepoint needs.\n"
            "Everything since the most recent savepoint stays, and the history "
            "is unchanged: what was recorded still says what happened."
        )
        self.reclaim_button.clicked.connect(self.reclaim_requested.emit)
        grid.addWidget(self.reclaim_button, 2, 0)

        # Beside the button that acts on it, in the same slot as the unindexed
        # line below: both are collection-level facts the project list has no
        # column for. Measured when this tab is opened rather than during a
        # scan -- see OperationsController.measure_archive.
        self.archive = QLabel("")
        self.archive.setWordWrap(True)
        self.archive.setStyleSheet(f"color: {MUTED};")
        grid.addWidget(self.archive, 3, 0, 1, 2)

        # The list above shows what is indexed, which says nothing about what
        # is sitting in the folder unindexed -- the state a collection is in
        # before any of this has been done, and the one the list cannot show.
        self.unindexed = QLabel("")
        self.unindexed.setWordWrap(True)
        self.unindexed.setStyleSheet(f"color: {MISMATCH.name()};")
        grid.addWidget(self.unindexed, 4, 0, 1, 2)

        self._update_collection_buttons()
        return box

    def set_archive(self, summary: str) -> None:
        """Say what the kept label versions cost, once something has measured."""
        self.archive.setText(summary)

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
        folder is loaded; recompressing and taking a savepoint wait on the
        editing guard as well, both of them writing.
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
        self.savepoint_button.setEnabled(has_root and self.controller.editing_enabled)
        self.reclaim_button.setEnabled(has_root and self.controller.editing_enabled)

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
            self.projects.set_projects(
                [summary.folder for summary in summaries],
                [summary.own_name for summary in summaries],
                icons=[_completeness_icon(summary) for summary in summaries],
                tooltips=[
                    f"{summary.label}\n{summary.describe()}" for summary in summaries
                ],
                current=0 if summaries else None,
            )
        finally:
            self._loading = False
        self.refresh()

    def step_project(self, delta: int) -> None:
        """Move *delta* rows through the project list, stopping at its ends.

        Walking the collection a project at a time is the way this tab is used,
        so it is worth a key rather than a click per project.
        """
        rows = self.projects.project_items()
        if not rows:
            return
        current = self.projects.project_of(self.projects.currentItem() or rows[0])
        here = current if current is not None else 0
        self.projects.setCurrentItem(
            rows[max(0, min(here + delta, len(rows) - 1))]
        )

    def _select(self, row: int) -> None:
        if self._loading or row < 0:
            return
        self.controller.select(row)
        self.refresh()

    def _select_folder(self, folder: str) -> None:
        """Describe every project under a folder at once.

        A session's projects share their sample, their grid and their operator,
        so typing that once is the point of grouping them. Fields they disagree
        on show as mixed and are left alone unless touched.
        """
        if self._loading:
            return
        self.controller.select_folder(folder)
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
            changes = self.controller.plan(edits)
        except (MetadataValidationError, ValueError) as error:
            self.status.setText(f"Cannot apply: {error}")
            return
        self.status.setText(_describe(changes))

    def _save(self) -> None:
        edits = self._edits()
        if not edits:
            self.status.setText("No changes")
            return
        if not self._confirm_removals(edits):
            self.status.setText("Not saved")
            return
        try:
            changes = self.controller.apply(edits)
        except (MetadataValidationError, ValueError, PermissionError) as error:
            QMessageBox.warning(self, "Cannot save", str(error))
            self.status.setText(f"Not saved: {error}")
            return
        self.status.setText("Saved.\n" + _describe(changes))
        self.refresh()
        self.metadata_changed.emit()

    def _confirm_removals(self, edits: dict[str, str | None]) -> bool:
        """Ask before clearing a field across a folder of projects.

        Writing a value to a folder is what describing one together is *for*,
        and needs no asking. Clearing one is the asymmetric case, and it is
        reachable by a single click: a field the projects disagree on shows as
        a chip reading ``(mixed)`` with a cross on it, which looks like tidying
        away a placeholder rather than like emptying that field in every
        project under the folder.

        One project keeps the behaviour it had. The edit is visible in the box
        there, and it is the project on screen.
        """
        cleared = sorted(field for field, value in edits.items() if value is None)
        if not cleared or len(self.controller.scope) < 2:
            return True
        answer = QMessageBox.question(
            self,
            "Clear fields",
            f"Remove {', '.join(cleared)} from all "
            f"{len(self.controller.scope)} projects under "
            f"{self.controller.scope_folder}?\n\n"
            "Their values in AM_metadata.yaml are deleted. Anything "
            "experiment.yaml supplies applies again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _show_note(
        self, field: str, source: str, outstanding: bool, *, mixed: bool = False
    ) -> None:
        """Annotate one field with where its value came from, or that it is due.

        Provenance and a placeholder never coincide -- a field still holding one
        resolves to no value and so has no source -- but they are not the same
        kind of remark, and drawing an outstanding field in the dimmest colour
        on the panel put the state that matters most where it was hardest to
        see.

        *mixed* comes first of the three. It is the one that says the box is not
        showing a value at all, and it is the authoritative statement: the
        marker in the box is editable text that could in principle have been
        typed, where this is derived from the projects in scope.
        """
        note = self.notes[field]
        if mixed:
            note.setText(f"[{MIXED_MARKER.strip('()')}]")
            note.setStyleSheet(BADGE_STYLES["info"])
            note.setToolTip(
                "These projects do not agree on this field. Leave it and none "
                "of them is written; type a value and every one of them is."
            )
        elif outstanding:
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
            mixed = set(self.controller.mixed_fields())
            for field, editor in self.editors.items():
                value = values.get(field, "")
                if field in SUGGESTED_FIELDS:
                    # Rebuilt every time, so a value saved on one project is
                    # offered on the next without rescanning.
                    set_editor_suggestions(editor, self.controller.suggestions(field))
                set_editor_text(editor, value)
                self._show_note(
                    field,
                    sources.get(field, ""),
                    field in outstanding,
                    mixed=field in mixed,
                )
            # Saving is exactly when a project stops being incomplete, so the
            # marks in the list must not be left describing the state before
            # it. Put back rather than rebuilt: a save is not a change of what
            # is listed, and rebuilding would refold the tree.
            summaries = self.controller.project_summaries()
            self.projects.show_icons(
                [_completeness_icon(summary) for summary in summaries],
                [f"{summary.label}\n{summary.describe()}" for summary in summaries],
            )
            self.summary.setText(self.controller.summary())
        finally:
            self._loading = False


def completeness_legend() -> list[tuple[QPixmap, str]]:
    """Return what the bar beside each project means, as ``(sample, meaning)``.

    Drawn through :func:`_completeness_icon`, so the legend cannot describe a
    bar the list does not draw.
    """
    def sample(filled: int, total: int) -> QPixmap:
        summary = ProjectSummary(
            label="", filled=filled, total=total,
            missing=() if filled == total else ("...",),
        )
        return _completeness_icon(summary).pixmap(COMPLETENESS_WIDTH, SWATCH_HEIGHT)

    return [
        (sample(4, 4), "Every field this project can carry has a value"),
        (
            sample(2, 4),
            "How full the bar is, is how much of the project is described; "
            "hover a row for what is missing",
        ),
    ]
