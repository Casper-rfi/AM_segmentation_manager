"""The Add data tab: see a project's descriptive metadata, and edit it.

A dot beside each field says whether it is filled in, and whether a change to
it is still only in the window; hovering it names the YAML file the value came
from, which is what explains a value that looks wrong. The project list marks
how much of each project is described, so what is left to do is visible
without opening every one.

Editing is locked until switched on, matching the curation view, and a
suggestion reaches a file only when the user saves. Under the fields, the
project's families of imaging conditions, changed through
:mod:`..families_controller` under the same switch.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...families import StaleFamilies
from ...metadata import split_values
from ...models import Project
from ...operations.families_edit import FamiliesValidationError
from ...state import StaleState
from ..families_controller import FamiliesController, FamilyNode, leaving_corrected
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
    BORDER_LIGHT,
    MUTED,
    SECTION_HEADING,
    SWATCH_HEIGHT,
    SWATCH_OUTLINE,
    SWATCH_WIDTH,
    TEXT,
    WARNING,
)

logger = logging.getLogger(__name__)

#: The Family dropdown's last entry, which asks for a name rather than being one.
NEW_FAMILY = "\x00new family"

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

    The two colours are the ones the curate list's chip uses for the same two
    ideas -- teal for finished, blue for not yet started -- so a colour means
    the same thing in both lists.
    """
    pixmap = QPixmap(COMPLETENESS_WIDTH, SWATCH_HEIGHT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        share = summary.filled / summary.total if summary.total else 0.0
        colour = STATE_COLOURS["corrected"] if summary.is_complete else MISMATCH
        painter.fillRect(0, 0, COMPLETENESS_WIDTH, SWATCH_HEIGHT, QColor(BORDER_LIGHT))
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
    that is saved: the list's teal and blue for a filled field and a missing
    one, ink for a change still only in the window. Two channels, so
    neither rests on telling colours apart.
    """
    pixmap = QPixmap(DOT, DOT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        disc = QRectF(1.5, 1.5, DOT - 3, DOT - 3)
        if state == "missing":
            pen = QPen(MISMATCH)
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(disc.adjusted(0.5, 0.5, -0.5, -0.5))
        elif state == "mixed":
            grey = QColor(MUTED)
            pen = QPen(grey)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(disc)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grey)
            painter.drawPie(disc, 90 * 16, 180 * 16)
        else:
            colour = QColor(TEXT) if state == "edited" else STATE_COLOURS["corrected"]
            painter.setPen(Qt.PenStyle.NoPen)
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


class FieldDot(QLabel):
    """The dot beside a field, remembering the state it was last drawn in."""

    #: The state drawn, for anything asking which dot this is. Unset until first drawn.
    state: str


def _dot_hint(state: str, paths: tuple[Path, ...], outstanding: bool) -> str:
    """Return what a field's dot says on hover."""
    if state == "missing" and outstanding:
        return "Still a placeholder; nobody has filled this in"
    if state == "filled":
        return f"{DOT_HINTS[state]}\n{source_tooltip(paths)}"
    return DOT_HINTS[state]


#: Every kind of box a metadata field is edited in.
Editor: TypeAlias = "ChipEditor | QComboBox | QPlainTextEdit | QLineEdit"


def _notify_on_change(editor: Editor, callback) -> None:
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


def editor_text(editor: Editor) -> str:
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


def set_editor_text(editor: Editor, value: str) -> None:
    """Show *value* in an editor, whichever kind of editor it is."""
    if isinstance(editor, ChipEditor):
        editor.set_values(split_values(value))
    elif isinstance(editor, QComboBox):
        editor.setCurrentText(value)
    elif isinstance(editor, QPlainTextEdit):
        editor.setPlainText(value)
    else:
        editor.setText(value)


def set_editor_suggestions(editor: Editor, values: list[str]) -> None:
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
    box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
    box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
    box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    line = box.lineEdit()
    assert line is not None, "an editable combo box has a line edit"
    line.setPlaceholderText("unset")
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
    #: Emitted after a families change was written, or refused as stale: the
    #: index the window holds describes families that are no longer so.
    families_changed = Signal()

    def __init__(
        self, controller: MetadataController, families: FamiliesController | None = None
    ) -> None:
        super().__init__()
        self.controller = controller
        self.families = families if families is not None else FamiliesController()
        #: Asked before a families change is written, with its project; the
        #: window answers whether label edits waiting there may be put aside.
        self.may_change_families: Callable[[Project], bool] = lambda project: True
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
        self.panes = GripSplitter(Qt.Orientation.Horizontal)
        self.panes.addWidget(self._build_projects())
        # Stacked, so neither squeezes the fields' width; a splitter, so the
        # families can be given the height a project with many conditions needs.
        self.describing = GripSplitter(Qt.Orientation.Vertical)
        self.describing.addWidget(self._build_fields())
        self.describing.addWidget(self._build_families())
        self.describing.setSizes([620, 340])
        # One switch above both boxes: either writes the project's own files.
        right = QWidget()
        right_column = QVBoxLayout(right)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.addWidget(self._build_enable())
        right_column.addWidget(self.describing, stretch=1)
        self.panes.addWidget(right)
        self._set_editing(False)
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
        box_layout = box.layout()
        assert box_layout is not None, "the box is laid out above"
        box_layout.addWidget(self._build_collection_actions())
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
        self.unindexed.setStyleSheet(f"color: {WARNING};")
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

        # One grid for every section, so the captions and the boxes line up
        # down the whole tab rather than within each heading.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self.editors: dict[str, Editor] = {}
        self.dots: dict[str, FieldDot] = {}
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
                chips = ChipEditor(separator=VALUE_SEPARATOR)
                chips.entry.setFixedWidth(FIELD_WIDTH)
                editor: Editor = chips
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
            dot = FieldDot()
            dot.setFixedSize(DOT, DOT)
            label = keep_together(dot, caption, spacing=6)
            caption_row = label.layout()
            assert caption_row is not None, "keep_together lays out what it holds"
            caption_row.setContentsMargins(0, CAPTION_DROP, 0, 0)
            grid.addWidget(label, row, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            # A box of one line keeps its width; values and prose take the
            # rest of the row.
            grid.addWidget(
                editor, row, 1, Qt.AlignmentFlag.AlignTop if fills else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
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
        self.fields_scroller.setFrameShape(QScrollArea.Shape.NoFrame)
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
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.status)
        return box

    def _build_enable(self) -> QWidget:
        """Return the switch unlocking the fields and the families.

        Above both rather than under them: it decides whether anything can be
        changed, and a switch below a form is found last.
        """
        self.enable = ToggleSwitch("Enable editing")
        self.enable.setToolTip(
            "Unlock the fields and the families. Only AM_metadata.yaml and "
            "AM_families.yaml are written; experiment.yaml is never touched."
        )
        self.enable.toggled.connect(self._set_editing)
        return self.enable

    def _build_families(self) -> QWidget:
        """Return the box listing the project's families of imaging conditions."""
        self.families_box = box = QGroupBox("Families")
        box.setToolTip(
            "Which imaging conditions share a label. A condition belongs to the "
            "family its folder name gives, unless AM_families.yaml moves it."
        )
        column = QVBoxLayout(box)
        self.families_status = QLabel("")
        self.families_status.setWordWrap(True)
        column.addWidget(self.families_status)
        self.families_tree = QTreeWidget()
        self.families_tree.setHeaderLabels(["Condition", "Stages", "Note"])
        self.families_tree.setRootIsDecorated(True)
        self.families_tree.setUniformRowHeights(True)
        self.families_tree.currentItemChanged.connect(self._update_family_controls)
        column.addWidget(self.families_tree, stretch=1)

        # What the selected condition belongs to, under the tree that shows it.
        moving = QHBoxLayout()
        self.family_label = QLabel("Family")
        self.family_combo = QComboBox()
        self.family_combo.setMinimumWidth(160)
        self.family_combo.setToolTip(
            "Move the selected condition into another family, or a new one. "
            "Its folder is not renamed: the move is kept in AM_families.yaml."
        )
        self.family_label.setToolTip(self.family_combo.toolTip())
        self.family_combo.activated.connect(self._family_chosen)
        self.primary_button = QPushButton("Make primary")
        self.primary_button.setToolTip(
            "Predict, measure and export the family from the selected condition, "
            "and open frames on it"
        )
        self.primary_button.clicked.connect(self._make_primary)
        self.default_primary_button = QPushButton("Default primary")
        self.default_primary_button.setToolTip(
            "Stand the family on its base condition again, or its first variant"
        )
        self.default_primary_button.clicked.connect(self._default_primary)
        moving.addWidget(self.family_label)
        moving.addWidget(self.family_combo)
        moving.addWidget(self.primary_button)
        moving.addWidget(self.default_primary_button)
        moving.addStretch(1)
        column.addLayout(moving)

        buttons = QHBoxLayout()
        self.families_preview_button = QPushButton("Preview changes")
        self.families_preview_button.clicked.connect(self._preview_families)
        self.families_save_button = QPushButton("Save")
        self.families_save_button.clicked.connect(self._save_families)
        self.families_discard_button = QPushButton("Discard changes")
        self.families_discard_button.setToolTip(
            "Put the families back to what is saved. Nothing is written."
        )
        self.families_discard_button.clicked.connect(self._discard_families)
        for button in (
            self.families_preview_button, self.families_save_button,
            self.families_discard_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        column.addLayout(buttons)

        self.families_message = QLabel("")
        self.families_message.setWordWrap(True)
        self.families_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        column.addWidget(self.families_message)
        return box

    def _refresh_families(self) -> None:
        """Show the families of the project in scope, beam then family then condition.

        Rebuilt whole: it changes only when the project does or a scan has
        read the project again.
        """
        self.families.set_scope(self.controller.scope)
        project = self.families.project
        self.families_box.setTitle(
            f"Families: {project.display_name}" if project is not None else "Families"
        )
        status = self.families.status()
        self.families_status.setText(status)
        self.families_status.setVisible(bool(status))
        tree = self.families_tree
        selected = self._selected_condition()
        tree.blockSignals(True)
        tree.clear()
        beams: dict[str, QTreeWidgetItem] = {}
        for node in self.families.tree():
            beam = beams.get(node.modality)
            if beam is None:
                beam = beams[node.modality] = QTreeWidgetItem(tree, [node.modality])
            family = QTreeWidgetItem(beam, [node.family])
            for row in node.conditions:
                item = QTreeWidgetItem(family, [row.name, str(row.stages), row.note])
                item.setData(0, Qt.ItemDataRole.UserRole, row.name)
                if row.name == selected:
                    tree.setCurrentItem(item)
        tree.expandAll()
        tree.resizeColumnToContents(0)
        tree.blockSignals(False)
        self._update_family_controls()

    def _selected_condition(self) -> str | None:
        """Return the condition selected in the tree, or None for a heading or nothing."""
        item = self.families_tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None

    def _selected_family(self) -> FamilyNode | None:
        """Return the family the selected condition is in, as the tree shows it."""
        condition = self._selected_condition()
        return next(
            (
                node for node in self.families.tree()
                if any(row.name == condition for row in node.conditions)
            ),
            None,
        )

    def _update_family_controls(self) -> None:
        """Offer what can be done to the selected condition, and to the changes held."""
        condition = self._selected_condition()
        node = self._selected_family()
        editing = self.families.editing_enabled and self.families.editable
        chosen = editing and condition is not None and node is not None
        self.family_combo.blockSignals(True)
        self.family_combo.clear()
        if condition is not None and node is not None:
            for caption, family in self.families.family_options(condition):
                self.family_combo.addItem(caption, family)
            self.family_combo.addItem("New family...", NEW_FAMILY)
            current = self.families.family_of(condition).lower()
            values = [
                str(self.family_combo.itemData(row)).lower()
                for row in range(self.family_combo.count())
            ]
            self.family_combo.setCurrentIndex(values.index(current) if current in values else 0)
        self.family_combo.blockSignals(False)
        self.family_combo.setEnabled(chosen)
        row = next((row for row in node.conditions if row.name == condition), None) if node else None
        self.primary_button.setEnabled(chosen and row is not None and not row.named_primary)
        self.default_primary_button.setEnabled(
            chosen and node is not None and any(r.named_primary for r in node.conditions)
        )
        changed = self.families.has_changes
        self.families_preview_button.setEnabled(changed)
        self.families_save_button.setEnabled(editing and changed)
        self.families_discard_button.setEnabled(changed)

    def _family_chosen(self, index: int) -> None:
        """Move the selected condition into the family chosen, or a new one named now."""
        condition = self._selected_condition()
        if condition is None or index < 0:
            return
        family = self.family_combo.itemData(index)
        if family == NEW_FAMILY:
            name, accepted = QInputDialog.getText(
                self, "New family",
                f"Name the family {condition} should form or join. It starts with "
                "sem or fib and holds letters, digits, '-' and '.'.",
            )
            if not accepted or not name.strip():
                self._update_family_controls()
                return
            family = name.strip()
        try:
            self.families.set_family(condition, family)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot move it there", str(error))
        self.families_message.setText("")
        self._refresh_families()

    def _make_primary(self) -> None:
        """Name the selected condition as its family's primary, held until saved."""
        node = self._selected_family()
        condition = self._selected_condition()
        if node is not None and condition is not None:
            self.families.set_primary(node.family, condition)
            self._refresh_families()

    def _default_primary(self) -> None:
        """Return the selected condition's family to its default primary, held until saved."""
        node = self._selected_family()
        if node is not None:
            self.families.set_primary(node.family, None)
            self._refresh_families()

    def _discard_families(self) -> None:
        """Forget the families changes held, writing nothing."""
        self.families.discard()
        self.families_message.setText("")
        self._refresh_families()

    def _preview_families(self) -> None:
        """Say what saving the families changes would do, or why it cannot."""
        try:
            change = self.families.plan()
        except (FamiliesValidationError, ValueError) as error:
            self.families_message.setText(_refusal(error))
            return
        self.families_message.setText("\n".join(change.describe()))

    def _save_families(self) -> None:
        """Write the families changes, asking first where proofreading is at stake.

        Planned again rather than taken from a preview, which may be stale.
        Images leaving a corrected group lose the claim, so that is asked;
        label edits waiting in the project are the window's to settle, a move
        changing which label they belong to.
        """
        project = self.families.project
        if project is None or not self.families.has_changes:
            self.families_message.setText("No changes")
            return
        try:
            change = self.families.plan()
        except (FamiliesValidationError, ValueError) as error:
            self.families_message.setText(_refusal(error))
            QMessageBox.warning(self, "Cannot save", _refusal(error))
            return
        if not self.may_change_families(project):
            self.families_message.setText("Not saved")
            return
        leaving = leaving_corrected(change)
        if leaving and QMessageBox.question(
            self,
            "Leave a corrected group",
            f"{leaving} image(s) would leave a label group declared corrected, "
            "and lose that claim. The proofreading stays in the label itself.\n\n"
            "Save the change?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            self.families_message.setText("Not saved")
            return
        try:
            applied = self.families.apply(change)
        except (StaleFamilies, StaleState) as error:
            QMessageBox.warning(
                self, "Somebody else changed this project",
                f"{error}\n\nNothing was changed. Reading the folder again now.",
            )
            self.families_message.setText("Not saved")
            self.families_changed.emit()
            return
        except PermissionError as error:
            self.families_message.setText(f"Not saved: {error}")
            return
        self.families_message.setText("Saved.\n" + "\n".join(applied.describe()))
        self.families_changed.emit()

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
        self.families.editing_enabled = enabled
        for editor in self.editors.values():
            editor.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.discard_button.setEnabled(enabled)
        self._update_collection_buttons()
        self._update_family_controls()

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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

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
        self._refresh_families()


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


def _refusal(error: Exception) -> str:
    """Return why a families change cannot be made, one reason a line."""
    problems = getattr(error, "problems", None) or [str(error)]
    return "Cannot save:\n" + "\n".join(problems)
