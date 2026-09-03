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

    from am_segmentation_manager.gui.widgets.lists import ProjectTree

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

def test_a_folder_starts_closed(tree):
    """Thirty sessions open as thirty rows to look down, not as a hundred
    projects -- which is the flat list this replaced."""
    assert not _folder(tree, "session_a").isExpanded()


def test_an_opened_folder_stays_open_across_a_rescan(tree):
    """An operation is followed by a rescan, which must not undo the reader's
    arranging."""
    _folder(tree, "session_a").setExpanded(True)

    tree.set_projects(FOLDERS, LABELS)

    assert _folder(tree, "session_a").isExpanded()
    assert not _folder(tree, "session_b").isExpanded()


def test_a_folder_that_is_new_stays_closed(tree):
    """Open is remembered by name, so a folder never seen before is closed."""
    _folder(tree, "session_a").setExpanded(True)

    tree.set_projects([*FOLDERS, "session_c"], [*LABELS, "007: AutoLamella-15-00"])

    assert _folder(tree, "session_a").isExpanded()
    assert not _folder(tree, "session_c").isExpanded()


def test_every_folder_can_be_opened_and_closed_at_once(tree):
    """What the two buttons under the tree do, with a hundred projects being
    too many to fold one at a time."""
    tree.expandAll()
    assert all(
        _folder(tree, name).isExpanded() for name in ("session_a", "session_b")
    )

    tree.collapseAll()
    assert not any(
        _folder(tree, name).isExpanded() for name in ("session_a", "session_b")
    )


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


def test_a_folder_holding_a_match_is_opened_to_show_it(tree):
    """With the folders closed, hiding the folder's siblings and leaving its
    own matches folded away reads as having found nothing."""
    tree.hide_rows_without("14-0", LABELS)

    assert _folder(tree, "session_b").isExpanded()


def test_clearing_the_filter_brings_everything_back(tree):
    tree.hide_rows_without("14-0", LABELS)

    tree.hide_rows_without("", LABELS)

    assert not any(item.isHidden() for item in tree.project_items())
    assert not _folder(tree, "session_a").isHidden()


def test_clearing_the_filter_puts_the_folds_back_as_the_reader_left_them(tree):
    """Opening a folder to search inside it is the search's doing, not a
    change of mind about which folders the reader wanted open."""
    _folder(tree, "session_a").setExpanded(True)

    tree.hide_rows_without("14-0", LABELS)
    tree.hide_rows_without("", LABELS)

    assert _folder(tree, "session_a").isExpanded()
    assert not _folder(tree, "session_b").isExpanded()


def test_a_project_at_the_root_is_filtered_like_any_other(tree):
    """It is a top-level row, so the folder branch must not skip it."""
    tree.hide_rows_without("Grid21", LABELS)

    assert not tree.project_item(5).isHidden()
    assert tree.project_item(0).isHidden()


# ---------------------------------------------------------------------------
# Which series a project holds
# ---------------------------------------------------------------------------

def _status(**cells):
    """Build a status from the cells named, the rest holding no images."""
    from am_segmentation_manager.gui.controller import CurationStatus
    from am_segmentation_manager.layout import MODALITIES, SERIES

    grid = {(s, m): "empty" for s in SERIES for m in MODALITIES}
    grid.update({tuple(key.rsplit("_", 1)): value for key, value in cells.items()})
    return CurationStatus(grid)


def _chip(status) -> bytes:
    """Return the pixels of a row's chip, so two can be told apart."""
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import SWATCH_HEIGHT, SWATCH_WIDTH

    icon = _status_icon(status)
    return icon.pixmap(SWATCH_WIDTH, SWATCH_HEIGHT).toImage().constBits().tobytes()


def test_a_project_missing_a_series_is_drawn_differently(qt_app):
    """The whole point: the row with nothing to proofread has to be findable
    among the ones that have."""
    all_three = _chip(_status(
        adaptive_SEM="not started", reference_SEM="not started",
        alignment_FIB="not started",
    ))
    nothing_to_do = _chip(_status(
        reference_SEM="not started", alignment_FIB="not started",
    ))

    assert nothing_to_do != all_three


def test_the_cells_are_read_by_position(qt_app):
    """Three columns in series order, so which series a mark is about is its
    place rather than something to be guessed."""
    assert _chip(_status(adaptive_SEM="not started")) != _chip(
        _status(reference_SEM="not started")
    )
    assert _chip(_status(reference_SEM="not started")) != _chip(
        _status(alignment_SEM="not started")
    )


def test_a_series_nobody_labels_claims_no_progress(qt_app):
    """Reference images sit at the default state for ever. Drawing that as
    "not started" would claim work is outstanding that nobody intends to do,
    so presence is all the cell says."""
    started = _chip(_status(reference_SEM="not started"))
    finished = _chip(_status(reference_SEM="done"))

    assert started == finished


def test_a_series_anyone_labels_shows_its_progress(qt_app):
    """The column that does vary, which is the one worth looking at."""
    assert _chip(_status(adaptive_SEM="not started")) != _chip(
        _status(adaptive_SEM="done")
    )


def test_the_tooltip_reads_out_only_the_progress_that_means_anything(qt_app):
    status = _status(
        adaptive_SEM="done", adaptive_FIB="not started",
        reference_SEM="not started",
    )

    summary = status.summary
    assert "Adaptive Polishing — SEM: done · FIB: not started" in summary
    assert "Also holds reference images" in summary
    assert "reference — SEM" not in summary


def test_the_tooltip_says_when_there_is_nothing_to_proofread(qt_app):
    summary = _status(reference_SEM="not started").summary

    assert summary.startswith("No Adaptive Polishing images: nothing to proofread")


# ---------------------------------------------------------------------------
# Choosing a folder
# ---------------------------------------------------------------------------

def test_moving_to_a_folder_reports_the_folder(tree):
    """A folder is not a project, so it asks a different question rather than
    silently changing nothing."""
    reported: list[str] = []
    tree.folder_selected.connect(reported.append)

    tree.setCurrentItem(_folder(tree, "session_b"))

    assert reported == ["session_b"]


def test_moving_to_a_folder_reports_no_project(tree):
    """Nothing about the frame on screen changes: a folder holds no view."""
    reported: list[int] = []
    tree.project_selected.connect(reported.append)

    tree.setCurrentItem(_folder(tree, "session_b"))

    assert reported == []


def test_moving_back_to_a_project_reports_the_project(tree):
    tree.setCurrentItem(_folder(tree, "session_b"))
    projects: list[int] = []
    folders: list[str] = []
    tree.project_selected.connect(projects.append)
    tree.folder_selected.connect(folders.append)

    tree.setCurrentItem(tree.project_item(3))

    assert projects == [3]
    assert folders == []


# ---------------------------------------------------------------------------
# The cursor on a folder
# ---------------------------------------------------------------------------

def test_a_folder_can_be_the_row_the_tree_is_on(tree):
    """A folder's lamellae are listed below it, so the folder has to come back
    as the row being looked at when the tree is rebuilt -- otherwise the list
    says one thing and the tree shows nothing chosen."""
    tree.set_projects(FOLDERS, LABELS, current_folder="session_b")

    assert tree.currentItem() is _folder(tree, "session_b")


def test_a_folder_made_current_is_opened(tree):
    """A chosen row folded away is a selection nobody can see."""
    tree.collapseAll()

    tree.set_projects(FOLDERS, LABELS, current_folder="session_a")

    assert _folder(tree, "session_a").isExpanded()


def test_a_folder_that_has_gone_leaves_the_cursor_alone(tree):
    """A rescan can take a folder with it. Naming one that is not there asks a
    question about the tree on screen, and the tree is what answers."""
    tree.set_projects(FOLDERS, LABELS, current_folder="renamed_away")

    assert tree.currentItem() is None or tree.project_of(tree.currentItem()) is not None


def test_a_folder_beats_a_project_for_the_cursor(tree):
    """Both are passed on every rebuild: the project position is what the
    controller last moved to, and the folder is what is actually listed."""
    tree.set_projects(FOLDERS, LABELS, current=2, current_folder="session_b")

    assert tree.currentItem() is _folder(tree, "session_b")


def test_a_tree_can_be_built_without_tick_boxes(qt_app):
    """Add data describes the project being looked at, one at a time. A box
    there offers a set the tab cannot act on, which is worse than no box."""
    from qtpy.QtCore import Qt

    from am_segmentation_manager.gui.widgets.lists import ProjectTree

    tree = ProjectTree(checkable=False)
    tree.set_projects(FOLDERS, LABELS)

    for item in tree.project_items():
        assert not item.flags() & Qt.ItemIsUserCheckable
    assert not _folder(tree, "session_a").flags() & Qt.ItemIsUserCheckable


def test_a_tree_carries_tick_boxes_by_default(tree):
    """Curate and Export both act on a ticked set."""
    from qtpy.QtCore import Qt

    for item in tree.project_items():
        assert item.flags() & Qt.ItemIsUserCheckable
