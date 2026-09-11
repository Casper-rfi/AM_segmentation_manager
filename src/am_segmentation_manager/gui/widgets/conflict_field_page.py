"""The conflict resolver's metadata page: the tree beside the two values.

Shaped like Add data -- projects under their folders on the left, their fields
on the right -- with each conflicting field a row of *Field | Here | Copy*, so
two values are read side by side rather than out of a sentence. A value is
chosen by clicking it and unchosen by clicking it again, as a mark in the tree
decides a whole folder or project. Decisions are held by the resolution.
"""
from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...operations.merge import OURS
from ..conflict_resolution import ALL, NONE, ConflictNode, ConflictResolution
from .conflict_tree import MARK_SIZE, SIDE_COLUMNS, ConflictTree, decide_all_buttons, mark_icon
from .controls import GripSplitter
from .theme import ALTERNATE, MUTED, SELECTED

__all__ = ["FieldConflictPage", "UNSET"]

#: What a side shows for a field it holds no value for.
UNSET = "unset"

#: The table's columns. The two sides sit where the tree puts their marks,
#: so a side is in the same place on both halves of the page.
HEADERS: tuple[str, ...] = ("Field", "Here", "Copy")

#: Where the field's name sits.
FIELD_COLUMN = 0


class FieldConflictPage(QWidget):
    """Decides metadata conflicts: the tree on the left, the values on the right."""

    #: Emitted whenever a decision changes, so the dialog can offer the next step.
    changed = Signal()

    def __init__(self, resolution: ConflictResolution, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.resolution = resolution
        self._keys_by_row: dict[int, str] = {}
        self._rows_by_key: dict[str, int] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        total = sum(len(node.keys) for node in resolution.fields)
        intro = QLabel(
            f"{total} project metadata field(s) were changed differently here "
            f"and in the copy since it was taken. Click the value to keep, or a "
            f"mark in the tree to decide a whole folder or project; the history "
            f"keeps both values either way."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.panes = GripSplitter(Qt.Horizontal)
        self.panes.addWidget(self._build_tree())
        self.panes.addWidget(self._build_table())
        self.panes.setSizes([380, 820])
        layout.addWidget(self.panes, stretch=1)

        # Opened to the projects only: the table lists their fields, and a
        # tree listing them too is the same rows twice.
        self.tree.set_nodes(resolution.fields, resolution, depth=0)
        if resolution.fields:
            self.show_node(resolution.fields[0])
        self._decided(announce=False)

    def _build_tree(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.tree = ConflictTree()
        self.tree.setToolTip(
            "Every field both sides changed differently, under its folder and "
            "project.\nClick Here or Copy on a folder or project to decide "
            "every field under it; click a full mark again to undecide."
        )
        self.tree.row_selected.connect(self.show_node)
        self.tree.side_requested.connect(self._side_requested)
        column.addWidget(self.tree, stretch=1)
        column.addLayout(decide_all_buttons("field", self.decide_everything))
        self.outstanding = QLabel("")
        self.outstanding.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.outstanding)
        return box

    def _build_table(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel("")
        self.heading.setWordWrap(True)
        column.addWidget(self.heading)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(list(HEADERS))
        self.table.horizontalHeaderItem(SIDE_COLUMNS[OURS]).setToolTip(
            "What this collection holds now. Click one to keep it."
        )
        self.table.horizontalHeaderItem(len(HEADERS) - 1).setToolTip(
            "What the copy holds. Click one to take it."
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.setIconSize(QSize(MARK_SIZE, MARK_SIZE))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(FIELD_COLUMN, QHeaderView.ResizeToContents)
        for side_column in SIDE_COLUMNS.values():
            header.setSectionResizeMode(side_column, QHeaderView.Stretch)
        # A long description wraps, and how many lines it wraps to depends on
        # how wide its column is now.
        header.sectionResized.connect(lambda *_: self.table.resizeRowsToContents())
        self.table.cellClicked.connect(self._cell_clicked)
        column.addWidget(self.table, stretch=1)
        return box

    # -- what is on screen -------------------------------------------------

    def show_node(self, node: ConflictNode) -> None:
        """List every field conflict under *node*; a field lists its whole project."""
        field = node if node.is_conflict else None
        if field is not None and node.parent is not None:
            node = node.parent
        self._fill(node)
        count = len(node.keys)
        self.heading.setText(f"{node.caption}: {count} field(s) in conflict")
        if field is not None:
            self.table.scrollToItem(
                self.table.item(self._rows_by_key[field.keys[0]], FIELD_COLUMN)
            )

    def row_of(self, key: str) -> int:
        """Return the table row showing the conflict *key*.

        Raises:
            KeyError: If it is not among those listed.
        """
        return self._rows_by_key[key]

    def _fill(self, node: ConflictNode) -> None:
        """List *node*'s field conflicts, each project headed by its own row."""
        by_project: dict[str, list[str]] = {}
        for key in node.keys:
            project = self.resolution.conflict(key).row.project
            by_project.setdefault(project, []).append(key)

        self.table.clearSpans()
        self.table.setRowCount(0)
        self._keys_by_row = {}
        self._rows_by_key = {}
        for project, keys in by_project.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            heading = QTableWidgetItem(project)
            heading.setFlags(Qt.ItemIsEnabled)
            heading.setBackground(QBrush(QColor(ALTERNATE)))
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            self.table.setItem(row, FIELD_COLUMN, heading)
            self.table.setSpan(row, FIELD_COLUMN, 1, len(HEADERS))
            for key in keys:
                self._add_field(key)
        self._show_choices()
        self.table.resizeRowsToContents()

    def _add_field(self, key: str) -> None:
        """Add one field's row: its name, then what each side holds."""
        conflict = self.resolution.conflict(key)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(conflict.caption)
        name.setFlags(Qt.ItemIsEnabled)
        name.setToolTip(conflict.row.metadata_field)
        self.table.setItem(row, FIELD_COLUMN, name)
        for side, column in SIDE_COLUMNS.items():
            value = (
                conflict.row.our_value if side == OURS else conflict.row.their_value
            )
            item = QTableWidgetItem(UNSET if value is None else value)
            item.setFlags(Qt.ItemIsEnabled)
            if value is None:
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setForeground(QBrush(QColor(MUTED)))
            self.table.setItem(row, column, item)
        self._keys_by_row[row] = key
        self._rows_by_key[key] = row

    def _show_choices(self) -> None:
        """Mark, shade and embolden the value each listed field has been given."""
        for row, key in self._keys_by_row.items():
            chosen = self.resolution.side(key)
            for side, column in SIDE_COLUMNS.items():
                item = self.table.item(row, column)
                picked = side == chosen
                item.setIcon(mark_icon(ALL if picked else NONE))
                item.setBackground(QBrush(QColor(SELECTED)) if picked else QBrush())
                font = item.font()
                font.setBold(picked)
                item.setFont(font)

    # -- deciding ------------------------------------------------------------

    def decide_everything(self, side: str | None) -> None:
        """Decide every field one way, or undecide them all with None."""
        self.resolution.decide_all(self.resolution.fields, side)
        self._decided()

    def _cell_clicked(self, row: int, column: int) -> None:
        """Choose the value clicked, or unchoose it where it was already chosen."""
        key = self._keys_by_row.get(row)
        for side, side_column in SIDE_COLUMNS.items():
            if key is not None and column == side_column:
                self.resolution.toggle(self.resolution.conflict(key), side)
                self._decided()

    def _side_requested(self, node: ConflictNode, side: str) -> None:
        self.resolution.toggle(node, side)
        self._decided()

    def _decided(self, *, announce: bool = True) -> None:
        """Bring every mark, value and count into step with the decisions."""
        self.tree.show_decisions(self.resolution)
        self._show_choices()
        left = self.resolution.outstanding(self.resolution.fields)
        total = sum(len(node.keys) for node in self.resolution.fields)
        self.outstanding.setText(
            f"{left} of {total} still to decide" if left else f"All {total} decided"
        )
        if announce:
            self.changed.emit()
