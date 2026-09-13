"""The Export tab's breakdown as a table: what its rows hold, and how they sort.

The rows are plain values, so most of this needs no widget. The widget's own
promises are the two a text block could not keep: a column sorts on a click,
and a total row stays under the rows it totals.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.aggregate import MISSING
    from am_segmentation_manager.gui.widgets.breakdown_table import (
        BreakdownTable,
        count_rows,
        cross_rows,
        sorted_body,
    )


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


def _texts(cells) -> list[str]:
    return [cell.text for cell in cells]


def _first_column(widget) -> list[str]:
    """Return what the table shows down its first column, top to bottom."""
    return [widget.grid.item(row, 0).text() for row in range(widget.grid.rowCount())]


def test_a_breakdown_row_holds_its_value_count_and_share():
    rows = count_rows({"SEM": 3, "FIB": 1}, "modality", total=4)

    assert rows.headers == ("modality", "Images", "Share")
    assert [_texts(row) for row in rows.body] == [
        ["SEM", "3", "75.0%"],
        ["FIB", "1", "25.0%"],
    ]
    assert rows.footer == ()


def test_a_share_is_of_the_images_counted_not_of_the_rows():
    """An image under several values counts in each row, so the rows add up
    to more than the images and the shares to more than 100."""
    rows = count_rows({"infected": 3, "differentiated": 3}, "sample", total=4)

    assert [row[2].text for row in rows.body] == ["75.0%", "75.0%"]


def test_numbers_sort_as_numbers_and_the_unrecorded_last():
    rows = count_rows({10.0: 1, MISSING: 5, 2.0: 1, 1.8: 1}, "voltage (kV)", total=8)

    ascending = sorted_body(rows.body, 0, descending=False)

    assert [row[0].text for row in ascending] == ["1.8", "2.0", "10.0", MISSING]


def test_a_grid_carries_a_total_column_and_a_total_row():
    table = {("SEM", "selected"): 2, ("SEM", "corrected"): 1, ("FIB", "selected"): 4}

    rows = cross_rows(table, "modality")

    assert rows.headers == ("modality", "corrected", "selected", "Total")
    assert {row[0].text: _texts(row[1:]) for row in rows.body} == {
        "FIB": ["0", "4", "4"],
        "SEM": ["1", "2", "3"],
    }
    assert _texts(rows.footer) == ["total", "1", "6", "7"]


def test_a_heading_is_bold_and_as_large_as_the_values_under_it(qt_app):
    """macOS gives a header its own smaller font, which put the headings below
    the numbers they name."""
    widget = BreakdownTable()
    header = widget.grid.horizontalHeader().font()

    assert header.bold()
    assert header.pointSizeF() == widget.grid.font().pointSizeF()


def test_clicking_a_heading_sorts_by_it_and_again_reverses(qt_app):
    widget = BreakdownTable()
    rows = count_rows({"a": 1, "b": 5, "c": 3}, "letter", total=9)
    widget.show_rows("Images by letter", rows, ["copied"])

    header = widget.grid.horizontalHeader()

    assert _first_column(widget) == ["a", "b", "c"], "opens in the order the counts came in"
    header.sectionClicked.emit(1)
    assert _first_column(widget) == ["a", "c", "b"]
    header.sectionClicked.emit(1)
    assert _first_column(widget) == ["b", "c", "a"]


def test_the_total_row_stays_last_whichever_way_a_column_sorts(qt_app):
    widget = BreakdownTable()
    table = {("x", "k"): 5, ("y", "k"): 1, ("z", "k"): 3}
    widget.show_rows("Images by letter and kind", cross_rows(table, "letter"), ["copied"])
    header = widget.grid.horizontalHeader()
    last = widget.grid.rowCount() - 1

    for _ in range(2):
        header.sectionClicked.emit(1)
        assert widget.grid.item(last, 0).text() == "total"
        assert widget.grid.item(last, 0).font().bold()


def test_copy_as_text_copies_the_command_line_s_text(qt_app):
    widget = BreakdownTable()
    widget.show_rows(
        "Images by modality", count_rows({"SEM": 1}, "modality", total=1),
        ["Images by modality  (1 image(s))", "  SEM  1  100.0%"],
    )

    widget.copy_button.click()

    assert QApplication.clipboard().text() == (
        "Images by modality  (1 image(s))\n  SEM  1  100.0%"
    )


def test_a_message_takes_the_place_of_the_table(qt_app):
    widget = BreakdownTable()
    widget.show_rows("t", count_rows({"a": 1}, "letter", total=1), ["x"])

    widget.show_message("Counting by crack count needs the labels measured.")

    assert widget.grid.isHidden()
    assert widget.text() == "Counting by crack count needs the labels measured."
    widget.grid.horizontalHeader().sectionClicked.emit(0)
