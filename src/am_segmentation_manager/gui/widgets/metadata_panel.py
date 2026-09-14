"""The Add data tab: see a project's descriptive metadata, and edit it.

A dot beside each field says whether it is filled in, and whether a change to
it is still only in the window; hovering it names the YAML file the value came
from, which is what explains a value that looks wrong. The project list marks
how much of each project is described, so what is left to do is visible
without opening every one.

Editing is locked until switched on, matching the curation view, and a
suggestion reaches a file only when the user saves.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
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
from .common import MAX_LISTED_FILES, chip_icon, horizontal_rule, match_anywhere
from .controls import ChipEditor, GripSplitter, ToggleSwitch, keep_together
from .frame_strip import MISMATCH, STATE_COLOURS
from .lists import project_box
from .paths import add_path_menu
from .theme import (
    ACCENT,
    MUTED,
    SECTION_HEADING,
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


#: Side of the dot beside each field, in logical pixels.
DOT = 12

#: How wide every single-line box on the tab is, in logical pixels. One width,
#: so the column of boxes reads as a column; enough for a grid type written out,
#: and a third of the pane, leaving the rest of the row for a field's values.
FIELD_WIDTH = 260

#: How many lines a prose field shows before it scrolls.
PROSE_LINES = 4

#: How far a caption sits below the top of its row, so it is level with the
#: text in a one-line box beside it rather than with the box's top edge.
CAPTION_DROP = 5

#: What each field dot says on hover, before anything particular to the field.
DOT_HINTS = {
    "filled": "Filled in",
    "missing": "Not filled in yet",
    "mixed": (
        "These projects do not agree on this field. Leave it and none of them "
        "is written; type a value and every one of them is."
    ),
    "edited": "Changed here and not saved yet",
}


def _dot_pixmap(state: str) -> QPixmap:
    """Return the dot beside a field in *state*, one of :data:`DOT_HINTS`.

    Shape says what the field holds -- solid, a ring, half -- and colour whether
    that is saved: the list's teal and amber for a filled field and a missing
    one, the accent for a change still only in the window. Two channels, so
    neither rests on telling colours apart.
    """
    pixmap = QPixmap(DOT, DOT)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        disc = QRectF(1.5, 1.5, DOT - 3, DOT - 3)
        if state == "missing":
            pen = QPen(MISMATCH)
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(disc.adjusted(0.5, 0.5, -0.5, -0.5))
        elif state == "mixed":
            grey = QColor(MUTED)
            pen = QPen(grey)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(disc)
            painter.setPen(Qt.NoPen)
            painter.setBrush(grey)
            painter.drawPie(disc, 90 * 16, 180 * 16)
        else:
            colour = QColor(ACCENT) if state == "edited" else STATE_COLOURS["corrected"]
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(disc)
    finally:
        painter.end()
    return pixmap


def field_state(shown: str, saved: str, *, mixed: bool, outstanding: bool) -> str:
    """Return which dot a field gets, from what its box shows and what is saved.

    A change comes first: whatever the saved value was, the box no longer
    shows it. A field the projects disagree on is mixed while left alone, and
    a placeholder resolves to no value, so it is missing like an empty field.
    """
    if shown != saved:
        return "edited"
    if mixed:
        return "mixed"
    if outstanding or not saved:
        return "missing"
    return "filled"


def _dot_hint(state: str, paths: tuple[Path, ...], outstanding: bool) -> str:
    """Return what a field's dot says on hover."""
    if state == "missing" and outstanding:
        return "Still a placeholder; nobody has filled this in"
    if state == "filled":
        return f"{DOT_HINTS[state]}\n{source_tooltip(paths)}"
    return DOT_HINTS[state]


def _notify_on_change(editor: QWidget, callback) -> None:
    """Call *callback* whenever *editor*'s content changes, whichever kind it is."""
    if isinstance(editor, ChipEditor):
        editor.changed.connect(callback)
    elif isinstance(editor, QComboBox):
        editor.currentTextChanged.connect(lambda _text: callback())
    else:
        editor.textChanged.connect(lambda *_args: callback())


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
    if isinstance(editor, QPlainTextEdit):
        # One line: a value is written to YAML and to a CSV cell, and a line
        # break typed into a wrapping box is layout rather than content.
        return " ".join(editor.toPlainText().split("\n")).strip()
    text = editor.currentText() if isinstance(editor, QComboBox) else editor.text()
    return text.strip()


def set_editor_text(editor: QWidget, value: str) -> None:
    """Show *value* in an editor, whichever kind of editor it is."""
    if isinstance(editor, ChipEditor):
        editor.set_values(split_values(value))
    elif isinstance(editor, QComboBox):
        editor.setCurrentText(value)
    elif isinstance(editor, QPlainTextEdit):
        editor.setPlainText(value)
    else:
        editor.setText(value)


def set_editor_suggestions(editor: QWidget, values: list[str]) -> None:
    """Offer the values already in use, whichever kind of editor this is."""
    if isinstance(editor, ChipEditor):
        editor.set_suggestions(values)
    elif isinstance(editor, QComboBox):
        editor.clear()
        editor.addItems(["", *values])


def _prose_box() -> QPlainTextEdit:
    """Return a box for a value written as a sentence or two.

    It wraps, and scrolls once full rather than growing: held to
    :data:`PROSE_LINES` lines, so a long description does not push every field
    under it down the tab.
    """
    box = QPlainTextEdit()
    box.setPlaceholderText("unset")
    box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    box.setTabChangesFocus(True)
    margin = 2 * round(box.document().documentMargin()) + 10
    box.setFixedHeight(box.fontMetrics().lineSpacing() * PROSE_LINES + margin)
    return box


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


def source_tooltip(paths: tuple[Path, ...]) -> str:
    """Return the hover text naming the file a value came from in each project."""
    if not paths:
        return "Which file this value came from"
    if len(paths) == 1:
        return f"From {paths[0]}"
    listed = [str(path) for path in paths[:MAX_LISTED_FILES]]
    if len(paths) > MAX_LISTED_FILES:
        listed.append(f"... and {len(paths) - MAX_LISTED_FILES} more")
    return f"From this file in each of the {len(paths)} projects:\n" + "\n".join(listed)


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
    #: Emitted when the user asks to put the whole collection back to a savepoint.
    restore_requested = Signal()

    def __init__(self, controller: MetadataController) -> None:
        super().__init__()
        self.controller = controller
        self._loading = False
        self.root: Path | None = None
        #: Each field as last read: its saved value, the file behind it in each
        #: project, and whether it is a placeholder or mixed. What the dots are
        #: compared against as the boxes change.
        self._resolved: dict[str, tuple[str, tuple[Path, ...], bool, bool]] = {}

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
        # The same box the Curate and Export tabs list projects in: an
        # Autolamella project is named for the minute it started, and
        # describing a collection means working down the folder somebody filed
        # them under. The filter field comes with it -- a hundred projects is
        # a hundred timestamps to read past otherwise.
        # No tick boxes: this tab describes the folder or the project being
        # looked at, so a box would offer a set it cannot act on, and a change
        # would reach projects nobody had in view.
        box, self.project_filter, self.projects = project_box(
            "Projects",
            checkable=False,
            tooltip=(
                "Projects, under the folder each is in. Click a project to "
                "describe it, or a folder to describe everything under it at "
                "once."
            ),
        )
        box.setMinimumWidth(PROJECT_PANE_MIN_WIDTH)
        self.projects.project_selected.connect(self._select)
        self.projects.folder_selected.connect(self._select_folder)
        # The tree matches on the row's own text, which is the project's name
        # under its folder -- the labels a flat list would need are the one
        # thing it does not want.
        self.project_filter.textChanged.connect(self.projects.hide_rows_without)
        add_path_menu(
            self.projects, "project",
            self.project_path_requested, self.project_reveal_requested,
            row_of=self.projects.project_of,
        )
        box.layout().addWidget(self._build_collection_actions())
        return box

    def _build_collection_actions(self) -> QWidget:
        """Return the steps that act on the folder rather than on one project.

        Two cards. *Data* reads the folder and names its projects, and comes
        before this list has rows in it: a collection with no metadata files has
        nothing to select, and naming its projects is what puts the first row
        there. *Maintenance* acts on every label under the folder -- the rule
        that keeps these off Curate -- and restoring and reclaiming change what
        is kept, so they are held apart from the two that only read or add.

        Two to a row rather than three across. A button clips its caption
        rather than eliding it, and clipping takes the ends off both sides at
        once -- so three of these in :data:`PROJECT_PANE_MIN_WIDTH` read as
        "nspect layou". Two fit, and the grid keeps them the same width as each
        other whatever they are called.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)

        self.data_box = QGroupBox("Data")
        data = QGridLayout(self.data_box)

        self.survey_button = QPushButton("Inspect layout")
        self.survey_button.setToolTip(
            "Report how the scanned folder is interpreted and what does not "
            "fit. Reads nothing but the directory structure, and changes nothing."
        )
        self.survey_button.clicked.connect(self.survey_requested.emit)
        data.addWidget(self.survey_button, 0, 0)

        self.init_button = QPushButton("Add projects...")
        self.init_button.setToolTip(
            "Pick which directories become projects, and give those that have "
            "none an AM_metadata.yaml. Unindexed folders are proposed for you "
            "to confirm; an existing file is never overwritten."
        )
        self.init_button.clicked.connect(self.init_requested.emit)
        data.addWidget(self.init_button, 0, 1)

        # The list above shows what is indexed, which says nothing about what
        # is sitting in the folder unindexed -- the state a collection is in
        # before any of this has been done, and the one the list cannot show.
        self.unindexed = QLabel("")
        self.unindexed.setWordWrap(True)
        self.unindexed.setStyleSheet(f"color: {MISMATCH.name()};")
        data.addWidget(self.unindexed, 1, 0, 1, 2)
        column.addWidget(self.data_box)

        self.maintenance_box = QGroupBox("Maintenance")
        maintenance = QGridLayout(self.maintenance_box)
        caption = QLabel("Acts on every label under the folder, once editing is on.")
        caption.setWordWrap(True)
        caption.setStyleSheet(f"color: {MUTED};")
        maintenance.addWidget(caption, 0, 0, 1, 2)

        # Taking a savepoint beside putting one back: a savepoint is taken over
        # the whole collection here, so this is where restoring the whole
        # collection belongs. Curate's Revert keeps the same choice for a
        # selection.
        self.savepoint_button = QPushButton("Savepoint...")
        self.savepoint_button.setToolTip(
            "Record what every label under this folder holds now, under a name "
            "you give it.\n"
            "Revert on the Curate tab can put labels back to it later. Image "
            "states and project metadata are not covered."
        )
        self.savepoint_button.clicked.connect(self.savepoint_requested.emit)
        maintenance.addWidget(self.savepoint_button, 1, 0)

        self.restore_button = QPushButton("Restore savepoint...")
        self.restore_button.setToolTip(
            "Put every label under this folder back to what it held at a "
            "savepoint.\n"
            "A label with no kept version is reported and left alone, and "
            "a corrected label is never replaced without being named."
        )
        self.restore_button.clicked.connect(self.restore_requested.emit)
        maintenance.addWidget(self.restore_button, 1, 1)

        self.compress_button = QPushButton("Compress labels...")
        self.compress_button.setToolTip(
            "Re-encode every label under this folder so it takes less disk "
            "space. Pixel values and dimensions are unchanged, and a file that "
            "would not shrink is left alone."
        )
        self.compress_button.clicked.connect(self.compress_requested.emit)
        maintenance.addWidget(self.compress_button, 2, 0)

        self.reclaim_button = QPushButton("Reclaim space...")
        self.reclaim_button.setToolTip(
            "Remove the kept label versions no savepoint needs.\n"
            "Everything since the most recent savepoint stays, and the history "
            "is unchanged: what was recorded still says what happened."
        )
        self.reclaim_button.clicked.connect(self.reclaim_requested.emit)
        maintenance.addWidget(self.reclaim_button, 2, 1)

        # Beside the button that acts on it: a collection-level fact the project
        # list has no column for. Measured when this tab is opened rather than
        # during a scan -- see OperationsController.measure_archive.
        self.archive = QLabel("")
        self.archive.setWordWrap(True)
        self.archive.setStyleSheet(f"color: {MUTED};")
        maintenance.addWidget(self.archive, 3, 0, 1, 2)
        column.addWidget(self.maintenance_box)

        self._update_collection_buttons()
        return holder

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
        """Name the folder the collection actions would act on.

        Named rather than left implicit because these act on everything under
        it, and the tab otherwise never says which folder is loaded.
        """
        self.root = root
        self.data_box.setTitle("Data" if root is None else f"Data: {root.name}")
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
        self.restore_button.setEnabled(has_root and self.controller.editing_enabled)

    def _build_fields(self) -> QWidget:
        # Titled with what is being described, so a folder of a dozen projects
        # is named before anything is typed into it.
        self.fields_box = box = QGroupBox("Metadata")
        column = QVBoxLayout(box)

        # Above the fields rather than under them: it decides whether any of
        # them can be typed in, and a switch below a form is found last.
        self.enable = ToggleSwitch("Enable editing")
        self.enable.setToolTip(
            "Unlock the fields. Only AM_metadata.yaml is written; "
            "experiment.yaml is never touched."
        )
        self.enable.toggled.connect(self._set_editing)
        column.addWidget(self.enable)

        # One grid for every section, so the captions and the boxes line up
        # down the whole tab rather than within each heading.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self.editors: dict[str, QWidget] = {}
        self.dots: dict[str, QLabel] = {}
        self.captions: dict[str, QLabel] = {}
        #: The heading over each section, in the order shown.
        self.section_headings: dict[str, QLabel] = {}
        row = 0
        # Which editor a field gets is the field table's to say: a vocabulary
        # is picked from, a value that repeats across projects is offered back,
        # prose wraps, and anything else is typed.
        for entry in FIELDS:
            if entry.section not in self.section_headings:
                heading = QLabel(entry.section)
                heading.setObjectName(SECTION_HEADING)
                self.section_headings[entry.section] = heading
                grid.addWidget(heading, row, 0, 1, 2)
                grid.addWidget(horizontal_rule(), row + 1, 0, 1, 2)
                row += 2
            fills = entry.multi or entry.prose
            if entry.multi:
                editor: QWidget = ChipEditor(separator=VALUE_SEPARATOR)
                editor.entry.setFixedWidth(FIELD_WIDTH)
            elif entry.prose:
                editor = _prose_box()
            elif entry.choices:
                editor = QComboBox()
                editor.setEditable(False)
                editor.addItems(["", *entry.choices, TODO_MARKER])
            elif entry.suggested:
                editor = _suggestion_box()
            else:
                editor = QLineEdit()
                editor.setPlaceholderText("unset")
            if not fills:
                editor.setFixedWidth(FIELD_WIDTH)

            caption = QLabel(entry.label)
            for widget in (caption, editor):
                widget.setToolTip(entry.guidance)
            dot = QLabel()
            dot.setFixedSize(DOT, DOT)
            label = keep_together(dot, caption, spacing=6)
            label.layout().setContentsMargins(0, CAPTION_DROP, 0, 0)
            grid.addWidget(label, row, 0, Qt.AlignLeft | Qt.AlignTop)
            # A box of one line keeps its width; values and prose take the
            # rest of the row.
            grid.addWidget(
                editor, row, 1, Qt.AlignTop if fills else Qt.AlignLeft | Qt.AlignTop
            )
            row += 1
            self.editors[entry.name] = editor
            self.dots[entry.name] = dot
            self.captions[entry.name] = caption
            _notify_on_change(editor, self._show_states)

        # Scrolled, so the fields cannot force the window taller than the
        # screen; the buttons stay under them rather than scrolling away.
        fields = QWidget()
        fields_column = QVBoxLayout(fields)
        fields_column.setContentsMargins(0, 0, 0, 0)
        fields_column.addLayout(grid)
        fields_column.addStretch(1)
        self.fields_scroller = QScrollArea()
        self.fields_scroller.setWidgetResizable(True)
        self.fields_scroller.setFrameShape(QScrollArea.NoFrame)
        self.fields_scroller.setWidget(fields)
        column.addWidget(self.fields_scroller, stretch=1)

        buttons = QHBoxLayout()
        self.preview_button = QPushButton("Preview changes")
        self.preview_button.clicked.connect(self._preview)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save)
        self.discard_button = QPushButton("Discard changes")
        self.discard_button.setToolTip(
            "Put every field back to what is saved. Nothing is written."
        )
        self.discard_button.clicked.connect(self._discard)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.discard_button)
        buttons.addStretch(1)
        column.addLayout(buttons)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.status)

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
                # The count alone: the row already names the project, and the
                # path is a right-click away.
                tooltips=[summary.describe() for summary in summaries],
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
        self.discard_button.setEnabled(enabled)
        self._update_collection_buttons()

    def _discard(self) -> None:
        """Put every field back to what is saved, writing nothing."""
        self.refresh()
        self.status.setText("Changes discarded")

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

    def _show_states(self) -> None:
        """Put each field's dot in step with what its box shows now."""
        if self._loading:
            return
        for field, editor in self.editors.items():
            saved, paths, outstanding, mixed = self._resolved.get(
                field, ("", (), False, False)
            )
            state = field_state(
                editor_text(editor), saved, mixed=mixed, outstanding=outstanding
            )
            dot = self.dots[field]
            #: The state drawn, for anything asking which dot this is.
            dot.state = state
            dot.setPixmap(_dot_pixmap(state))
            dot.setToolTip(_dot_hint(state, paths, outstanding))

    def refresh(self) -> None:
        """Redraw the fields from the controller."""
        self._loading = True
        try:
            values = self.controller.values()
            paths = self.controller.source_paths()
            outstanding = set(self.controller.outstanding())
            mixed = set(self.controller.mixed_fields())
            for field, editor in self.editors.items():
                value = values.get(field, "")
                if field in SUGGESTED_FIELDS:
                    # Rebuilt every time, so a value saved on one project is
                    # offered on the next without rescanning.
                    set_editor_suggestions(editor, self.controller.suggestions(field))
                set_editor_text(editor, value)
                self._resolved[field] = (
                    value, paths.get(field, ()), field in outstanding, field in mixed
                )
            # Saving is exactly when a project stops being incomplete, so the
            # marks in the list must not be left describing the state before
            # it. Put back rather than rebuilt: a save is not a change of what
            # is listed, and rebuilding would refold the tree.
            summaries = self.controller.project_summaries()
            self.projects.show_icons(
                [_completeness_icon(summary) for summary in summaries],
                [summary.describe() for summary in summaries],
            )
            scope = self.controller.scope_title()
            self.fields_box.setTitle(f"Metadata: {scope}" if scope else "Metadata")
        finally:
            self._loading = False
        self._show_states()


def field_dot_legend() -> list[tuple[QPixmap, str]]:
    """Return what the dot beside each field means, drawn by :func:`_dot_pixmap`."""
    return [
        (_dot_pixmap("filled"), "Filled in; hover it for the file the value came from"),
        (_dot_pixmap("missing"), "Not filled in yet, or still a placeholder"),
        (_dot_pixmap("mixed"), "The projects in this folder do not agree on it"),
        (_dot_pixmap("edited"), "Changed here and not saved yet"),
    ]


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
