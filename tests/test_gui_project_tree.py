"""Grouping the projects under the folder each one sits in.

An Autolamella project is named for the minute it started, so a flat list of a
hundred of them is a hundred timestamps. The folder somebody filed them under is
what a person recognises, and it is what these group by.

Driven through the widget rather than the window: what a folder row is, and what
happens to the ticks and the folds, is the tree's own to answer.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.gui.widgets.project_panel import ProjectTree

#: One folder of three, one of two, and a project sitting at the scan root.
FOLDERS = ["session_a", "session_a", "session_a", "session_b", "session_b", ""]
LABELS = [
    "001: AutoLamella-13-00",
    "002: AutoLamella-13-01",
    "003: AutoLamella-13-02",
    "004: AutoLamella-14-00",
    "005: AutoLamella-14-01",
    "006: Grid21_adaptive_milling",
]


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tree(qt_app) -> ProjectTree:
    widget = ProjectTree()
    widget.set_projects(FOLDERS, LABELS)
    return widget


def _folder(tree, name: str):
    """Return the folder row called *name*."""
    found = [
        tree.topLevelItem(row)
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).text(0) == name
    ]
    assert found, f"no folder row called {name!r}"
    return found[0]


# ---------------------------------------------------------------------------
# What the tree is made of
# ---------------------------------------------------------------------------

def test_every_project_is_there_once(tree):
    assert [tree.project_of(item) for item in tree.project_items()] == list(range(6))


def test_a_project_hangs_under_its_folder(tree):
    assert _folder(tree, "session_a").childCount() == 3
    assert _folder(tree, "session_b").childCount() == 2


def test_a_project_at_the_scan_root_has_no_folder_row(tree):
    """Rather than one called "(root)", which would name a directory that is
    not there."""
    root_level = [
        tree.topLevelItem(row).text(0) for row in range(tree.topLevelItemCount())
    ]

    assert root_level == ["session_a", "session_b", LABELS[5]]


def test_a_folder_is_not_a_project(tree):
    """The number travels on the item, so a folder taking a position in the
    tree cannot be mistaken for the project at that position."""
    assert tree.project_of(_folder(tree, "session_a")) is None
    assert tree.project_of(tree.project_item(0)) == 0


# ---------------------------------------------------------------------------
# Ticking
# ---------------------------------------------------------------------------

def test_ticking_a_folder_ticks_its_projects_and_no_others(tree):
    reported: list[tuple[int, bool]] = []
    tree.row_ticked.connect(lambda index, on: reported.append((index, on)))

    _folder(tree, "session_a").setCheckState(0, Qt.Checked)

    assert tree.ticked_rows() == [0, 1, 2]
    assert sorted(reported) == [(0, True), (1, True), (2, True)]


def test_a_folder_follows_its_projects_back(tree):
    """Qt's tri-state, both ways: the folder is a convenience over the
    projects and never a fourth thing that can disagree with them."""
    folder = _folder(tree, "session_a")
    folder.setCheckState(0, Qt.Checked)

    tree.project_item(1).setCheckState(0, Qt.Unchecked)

    assert folder.checkState(0) == Qt.PartiallyChecked
    assert tree.ticked_rows() == [0, 2]


def test_putting_ticks_back_reports_nothing(tree):
    """A redraw is not a gesture; see CheckList.set_values."""
    reported: list = []
    tree.row_ticked.connect(lambda index, on: reported.append((index, on)))

    tree.show_ticks([True, False, True, False, False, False])

    assert tree.ticked_rows() == [0, 2]
    assert reported == []


# ---------------------------------------------------------------------------
# Folding, and what survives a rebuild
# ---------------------------------------------------------------------------

def test_a_folder_starts_open(tree):
    """Nothing a scan found is hidden by a preference nobody expressed."""
    assert _folder(tree, "session_a").isExpanded()


def test_a_folded_folder_stays_folded_across_a_rescan(tree):
    _folder(tree, "session_a").setExpanded(False)

    tree.set_projects(FOLDERS, LABELS)

    assert not _folder(tree, "session_a").isExpanded()
    assert _folder(tree, "session_b").isExpanded()


def test_a_folder_that_is_new_opens(tree):
    """Folded is remembered by name, so a folder never seen before is open."""
    _folder(tree, "session_a").setExpanded(False)

    tree.set_projects([*FOLDERS, "session_c"], [*LABELS, "007: AutoLamella-15-00"])

    assert not _folder(tree, "session_a").isExpanded()
    assert _folder(tree, "session_c").isExpanded()


def test_the_folder_holding_the_current_project_is_opened(tree):
    """Selecting something inside a folded folder has to show it."""
    _folder(tree, "session_b").setExpanded(False)

    tree.set_projects(FOLDERS, LABELS, current=4)

    assert _folder(tree, "session_b").isExpanded()
    assert tree.currentItem() is tree.project_item(4)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_filtering_hides_the_projects_that_do_not_match(tree):
    tree.hide_rows_without("14-0", LABELS)

    assert tree.project_item(0).isHidden()
    assert not tree.project_item(3).isHidden()


def test_a_folder_with_nothing_left_in_it_goes_too(tree):
    """Somewhere to look that has nothing in it is worse than one row fewer."""
    tree.hide_rows_without("14-0", LABELS)

    assert _folder(tree, "session_a").isHidden()
    assert not _folder(tree, "session_b").isHidden()


def test_clearing_the_filter_brings_everything_back(tree):
    tree.hide_rows_without("14-0", LABELS)

    tree.hide_rows_without("", LABELS)

    assert not any(item.isHidden() for item in tree.project_items())
    assert not _folder(tree, "session_a").isHidden()


def test_a_project_at_the_root_is_filtered_like_any_other(tree):
    """It is a top-level row, so the folder branch must not skip it."""
    tree.hide_rows_without("Grid21", LABELS)

    assert not tree.project_item(5).isHidden()
    assert tree.project_item(0).isHidden()
