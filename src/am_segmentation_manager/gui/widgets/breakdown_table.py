"""A breakdown as a table: counts that sort, and a copy that reads as the CLI's.

What the Export tab counts, shown as rows rather than a block of padded text. The
rows are built by plain functions, so what a table holds is testable without a
widget; the copy is :mod:`am_segmentation_manager.reporting`'s text, so pasting it
gives what ``am-seg stats`` prints. **A total row stays last** whichever column is
sorted, and values sort by :func:`~am_segmentation_manager.aggregate.value_order`.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...aggregate import totals, value_order
from .theme import MUTED


@dataclass(frozen=True)
class Cell:
    """One cell: what it shows, and what it sorts by."""

    text: str
    key: tuple


@dataclass(frozen=True)
class Rows:
    """A table's worth of cells: the header, the rows that sort, and a total row."""

    headers: tuple[str, ...]
    body: tuple[tuple[Cell, ...], ...]
    footer: tuple[Cell, ...] = ()


def _cell(value: Hashable) -> Cell:
    return Cell(str(value), value_order(value))


def _share(count: int, total: int) -> Cell:
    share = 100 * count / total if total else 0.0
    return Cell(f"{share:.1f}%" if total else "-", value_order(share))


def count_rows(counts: Mapping[Hashable, int], label: str, total: int) -> Rows:
    """Return a one-property breakdown as rows of value, image count and share.

    Args:
        total: What each share is of -- more than the rows add up to where an
            image can be counted under several values.
    """
    return Rows(
        (label, "Images", "Share"),
        tuple(
            (_cell(value), _cell(count), _share(count, total))
            for value, count in counts.items()
        ),
    )


def cross_rows(
    table: Mapping[tuple[Hashable, Hashable], int],
    row_title: str,
    *,
    row_order: Sequence[Hashable] = (),
    column_order: Sequence[Hashable] = (),
) -> Rows:
    """Return a two-property breakdown as a grid, with a total column and row."""
    rows, columns = totals(table, row_order, column_order)
    body = []
    for row in rows:
        counts = [table.get((row, column), 0) for column in columns]
        body.append((_cell(row), *map(_cell, counts), _cell(sum(counts))))
    column_totals = [
        sum(table.get((row, column), 0) for row in rows) for column in columns
    ]
    footer = (
        Cell("total", value_order("total")),
        *map(_cell, column_totals),
        _cell(sum(table.values())),
    )
    return Rows(
        (row_title, *(str(column) for column in columns), "Total"), tuple(body), footer
    )


def sorted_body(
    body: Sequence[tuple[Cell, ...]], column: int, descending: bool
) -> tuple[tuple[Cell, ...], ...]:
    """Return *body* ordered by *column*, ties keeping the order they came in."""
    return tuple(sorted(body, key=lambda cells: cells[column].key, reverse=descending))


class BreakdownTable(QWidget):
    """Counts as a table that sorts on a header click, and a copy of them as text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.caption)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {MUTED};")
        self.note.setHidden(True)
        column.addWidget(self.note)

        self.grid = QTableWidget(0, 0)
        self.grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.grid.verticalHeader().setVisible(False)
        header = self.grid.horizontalHeader()
        # The cells' own font, bold. macOS gives a header a smaller system font
        # of its own, which set a heading below the values it names.
        heading = QFont(self.grid.font())
        heading.setBold(True)
        header.setFont(heading)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionsClickable(True)
        header.setToolTip("Click a heading to sort by it, and again to reverse")
        header.sectionClicked.connect(self._sort_by)
        column.addWidget(self.grid, stretch=1)

        row = QHBoxLayout()
        row.addStretch(1)
        self.copy_button = QPushButton("Copy as text")
        self.copy_button.setToolTip(
            "Copy these counts as aligned text, as am-seg stats prints them."
        )
        self.copy_button.clicked.connect(self._copy)
        row.addWidget(self.copy_button)
        column.addLayout(row)

        self._rows: Rows | None = None
        self._text = ""
        #: The column sorted by and whether descending, or None for the order
        #: the counts came in -- a scale's declared order, or largest first.
        self._sorted: tuple[int, bool] | None = None

    def text(self) -> str:
        """Return what Copy as text would copy."""
        return self._text

    def show_message(self, text: str) -> None:
        """Say *text* in place of a table, and copy that if asked."""
        self._rows = None
        self._text = text
        self.caption.setText(text)
        self.note.setHidden(True)
        self.grid.setRowCount(0)
        self.grid.setColumnCount(0)
        self.grid.setHidden(True)
        self.copy_button.setEnabled(bool(text))

    def show_rows(
        self, title: str, rows: Rows, lines: Sequence[str], *, note: str = ""
    ) -> None:
        """Show *rows* under *title*, in the order they came, copying as *lines*."""
        self._rows = rows
        self._text = "\n".join(lines)
        self._sorted = None
        self.caption.setText(title)
        self.note.setText(note)
        self.note.setHidden(not note)
        self.grid.horizontalHeader().setSortIndicatorShown(False)
        self.grid.setHidden(False)
        self.copy_button.setEnabled(True)
        self._fill(rows.body)

    def _fill(self, body: Sequence[tuple[Cell, ...]]) -> None:
        rows = self._rows
        assert rows is not None, "rows are shown before they are filled"
        lines = [*body, rows.footer] if rows.footer else list(body)
        self.grid.clear()
        self.grid.setColumnCount(len(rows.headers))
        self.grid.setHorizontalHeaderLabels(list(rows.headers))
        self.grid.setRowCount(len(lines))
        for row, cells in enumerate(lines):
            is_footer = bool(rows.footer) and row == len(lines) - 1
            for column, cell in enumerate(cells):
                item = QTableWidgetItem(cell.text)
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if is_footer:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.grid.setItem(row, column, item)

    def _sort_by(self, column: int) -> None:
        """Sort the rows by *column*, reversing on a second click; the total stays last."""
        if self._rows is None:
            return
        descending = self._sorted == (column, False)
        self._sorted = (column, descending)
        header = self.grid.horizontalHeader()
        header.setSortIndicatorShown(True)
        header.setSortIndicator(
            column, Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder
        )
        self._fill(sorted_body(self._rows.body, column, descending))

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._text)
