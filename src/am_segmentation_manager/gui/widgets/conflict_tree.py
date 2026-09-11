"""A merge's conflicts, arranged as the collection is.

Two marks beside each row, *Here* and *Copy*, say how much of what is under it
keeps this collection's version and how much takes the copy's -- all, some or
none, drawn filled, dotted or empty. Both empty is undecided, and a heading
decided both ways shows some of each, which one control per row could not.
Clicking a mark asks for that side; the tree decides nothing itself.
"""
from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence

from qtpy.QtCore import QPointF, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
)

from ...operations.merge import OURS, THEIRS
from ..conflict_resolution import ALL, SOME, ConflictNode, ConflictResolution
from .lists import TREE_INDENT
from .theme import ACCENT, WARNING

__all__ = ["ConflictTree", "SIDE_COLUMNS", "decide_all_buttons", "mark_icon"]

#: Which column holds which side's mark.
SIDE_COLUMNS: dict[str, int] = {OURS: 1, THEIRS: 2}

#: What the columns are headed, and what each heading explains.
HEADERS: tuple[str, ...] = ("", "Here", "Copy")
HEADER_HINTS: dict[int, str] = {
    SIDE_COLUMNS[OURS]: "Keep what this collection holds now",
    SIDE_COLUMNS[THEIRS]: "Take what the copy holds",
}

#: Where a row keeps the number of the node it shows. A row is not a position
#: -- headings take positions too -- so the number travels on the item.
NODE_INDEX = Qt.UserRole + 1

#: How large a mark is drawn, in logical pixels.
MARK_SIZE = 14


class ConflictTree(QTreeWidget):
    """The conflicts of one tree of a :class:`ConflictResolution`, a mark per side."""

    #: Emitted with the node of the row the user moved to.
    row_selected = Signal(object)
    #: Emitted with a node and the side whose mark was clicked on its row.
    side_requested = Signal(object, str)

    def __init__(self, passed_keys: Iterable[int] = ()) -> None:
        """Build an empty tree.

        Args:
            passed_keys: Keys handed on to the parent rather than used to move
                between rows or to search -- the ones a page steps and decides
                with, which the tree would otherwise take first.
        """
        super().__init__()
        self._passed = frozenset(passed_keys)
        self._nodes: list[ConflictNode] = []
        self._items: dict[ConflictNode, QTreeWidgetItem] = {}
        self.setColumnCount(len(HEADERS))
        self.setHeaderLabels(list(HEADERS))
        for column, hint in HEADER_HINTS.items():
            self.headerItem().setToolTip(column, hint)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in SIDE_COLUMNS.values():
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.setIndentation(TREE_INDENT)
        self.setTextElideMode(Qt.ElideMiddle)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setIconSize(QSize(MARK_SIZE, MARK_SIZE))
        self.itemClicked.connect(self._clicked)
        self.currentItemChanged.connect(self._current_changed)

    # -- filling -----------------------------------------------------------

    def set_nodes(
        self,
        nodes: Sequence[ConflictNode],
        resolution: ConflictResolution,
        *,
        depth: int = 1,
    ) -> None:
        """Replace the rows with *nodes* and everything under them.

        Opened *depth* levels down -- by default to the lamellae, whose
        conflicts open as they are stepped into: a merge of a thousand
        conflicts is otherwise a thousand rows to read past to find the second
        lamella.
        """
        self.blockSignals(True)
        try:
            self.clear()
            self._nodes = []
            self._items = {}
            for node in nodes:
                self.addTopLevelItem(self._build(node, resolution))
        finally:
            self.blockSignals(False)
        self.expandToDepth(depth)
        self.show_decisions(resolution)

    def _build(self, node: ConflictNode, resolution: ConflictResolution) -> QTreeWidgetItem:
        """Return the row for *node*, with the rows under it."""
        item = QTreeWidgetItem([node.caption, "", ""])
        item.setData(0, NODE_INDEX, len(self._nodes))
        self._nodes.append(node)
        self._items[node] = item
        tooltip = node.row.title if node.row is not None else node.caption
        proofread = resolution.proofread(node)
        if proofread:
            # On the row, not counted at the top: which of these is proofread
            # work decides how the choice is made.
            item.setForeground(0, QBrush(QColor(WARNING)))
            tooltip += (
                "\nProofread work here: taking the copy's replaces it."
                if node.is_conflict else
                f"\n{proofread} of the {len(node.keys)} here are proofread work."
            )
        item.setToolTip(0, tooltip)
        for child in node.children:
            item.addChild(self._build(child, resolution))
        return item

    def show_decisions(self, resolution: ConflictResolution) -> None:
        """Redraw every row's two marks from *resolution*."""
        for node, item in self._items.items():
            for side, column in SIDE_COLUMNS.items():
                item.setIcon(column, mark_icon(resolution.coverage(node, side)))

    # -- where the reader is -------------------------------------------------

    def node_at(self, item: QTreeWidgetItem) -> ConflictNode:
        """Return the node *item* shows."""
        return self._nodes[item.data(0, NODE_INDEX)]

    def item_of(self, node: ConflictNode) -> QTreeWidgetItem:
        """Return the row showing *node*."""
        return self._items[node]

    def show_current(self, node: ConflictNode) -> None:
        """Move to *node*'s row, opening what it is under, without reporting it."""
        item = self._items.get(node)
        if item is None:
            return
        self.blockSignals(True)
        try:
            self.setCurrentItem(item)
        finally:
            self.blockSignals(False)
        self.scrollToItem(item)

    def _current_changed(self, current, _previous) -> None:
        if current is not None:
            self.row_selected.emit(self.node_at(current))

    def _clicked(self, item: QTreeWidgetItem, column: int) -> None:
        for side, side_column in SIDE_COLUMNS.items():
            if column == side_column:
                self.side_requested.emit(self.node_at(item), side)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Hand the page's keys on rather than moving between rows or searching."""
        if event.key() in self._passed:
            event.ignore()
            return
        super().keyPressEvent(event)


def decide_all_buttons(noun: str, on_decide) -> QHBoxLayout:
    """Return Keep all here, Take all the copy's and Clear, as one row.

    Args:
        noun: What a page's conflicts are, for the tooltips -- "label", "field".
        on_decide: Called with the side to give every conflict, or None to
            undecide them all.
    """
    row = QHBoxLayout()
    for caption, side, hint in (
        ("Keep all here", OURS, f"Keep this collection's version of every {noun}"),
        ("Take all the copy's", THEIRS, f"Take the copy's version of every {noun}"),
        ("Clear", None, f"Undecide every {noun}"),
    ):
        button = QPushButton(caption)
        button.setAutoDefault(False)
        button.setToolTip(hint)
        button.clicked.connect(lambda _checked=False, s=side: on_decide(s))
        row.addWidget(button)
    row.addStretch(1)
    return row


@functools.cache
def mark_icon(coverage: str) -> QIcon:
    """Return the mark for how much of a row one side holds.

    Told apart by shape as well as by fill, so none of the three depends on
    seeing a colour.
    """
    pixmap = QPixmap(MARK_SIZE, MARK_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    colour = QColor(ACCENT)
    painter.setPen(QPen(colour, 1.5))
    painter.setBrush(QBrush(colour) if coverage == ALL else Qt.NoBrush)
    ring = QRectF(1.5, 1.5, MARK_SIZE - 3.0, MARK_SIZE - 3.0)
    painter.drawEllipse(ring)
    if coverage == SOME:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(colour))
        painter.drawEllipse(QPointF(ring.center()), 2.5, 2.5)
    painter.end()
    return QIcon(pixmap)
