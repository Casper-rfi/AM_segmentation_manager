"""What survives picking a project out of a long list.

Rebuilding a list from inside its own ``currentRowChanged`` is delivered while
Qt is still part-way through the mouse press that caused it, and it throws the
reviewer back to the top of the list. Both follow from the same thing, so the
property pinned here is the mechanism: selecting a project leaves the project
list's rows exactly as they were, and rebuilds only the lamellae.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from conftest import destroy

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGE_STACK),
    reason="Qt binding or optional image libs not installed",
)

if HAS_QT and HAS_IMAGE_STACK:
    import numpy as np
    import tifffile
    from qtpy.QtCore import QSettings, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

#: Enough projects that losing your place in the list is the reported problem.
PROJECT_COUNT = 20


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A scan root of many projects, each with two named lamellae."""
    root = tmp_path / "scan"
    for index in range(PROJECT_COUNT):
        project_root = root / f"AutoLamella-2026-07-28-13-{index:02d}"
        project_root.mkdir(parents=True)
        (project_root / "experiment.yaml").write_text(f"name: project-{index}")
        for lamella in (f"01-p{index}a", f"02-p{index}b"):
            stage = project_root / lamella / "adaptive_polish_2026-07-28-11-15-36AM"
            path = stage / "sem" / f"{lamella}_AP_img_000_SEM.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(path), np.full((32, 32), 100, dtype=np.uint8))
    return root


@pytest.fixture
def window(qt_app, archive: Path, tmp_path: Path):
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    # Shown, and at a real size. Offscreen, but a widget that has never been
    # laid out has no viewport to scroll and no row at any position, which is
    # exactly what the scrolling test needs to be true.
    window.resize(1200, 800)
    window.show()
    window.scan(str(archive))
    qt_app.processEvents()
    yield window
    destroy(window)


def _project_rows(window) -> list:
    """The project rows, folders aside -- what a rebuild would replace."""
    return list(window.project_panel.project_list.project_items())


def _select(window, index: int) -> None:
    """Move the cursor to a project, as clicking its row does."""
    tree = window.project_panel.project_list
    tree.setCurrentItem(tree.project_item(index))


def test_the_archive_is_listed(window):
    assert len(_project_rows(window)) == PROJECT_COUNT


def test_picking_a_project_leaves_the_project_rows_untouched(window):
    """Not merely equal -- the same objects.

    A rebuilt list is a list whose scroll position has gone back to the top,
    and a list rebuilt from inside its own signal is one Qt is still holding a
    position in. Identity is what says neither happened.
    """
    before = _project_rows(window)

    _select(window, PROJECT_COUNT - 2)

    after = _project_rows(window)
    assert len(after) == len(before)
    assert all(new is old for new, old in zip(after, before))


def test_picking_a_project_keeps_the_list_where_it_was_scrolled(window, qt_app):
    """The reported symptom, driven the way it was met: with the pointer.

    ``setCurrentRow`` will not do here. It scrolls to the row itself, so it
    hides the fault -- what loses the reviewer's place is the rebuild's own
    ``setCurrentRow`` afterwards, which re-frames the list around the row
    instead of leaving it where they had put it.
    """
    listing = window.project_panel.project_list
    listing.resize(200, 120)
    listing.scrollToItem(listing.project_item(PROJECT_COUNT - 1))
    qt_app.processEvents()

    scrolled = listing.verticalScrollBar().value()
    assert scrolled > 0, "the list did not scroll; the test cannot say anything"
    target = _a_visible_row(listing)

    _click_row(qt_app, listing, target)

    assert listing.currentItem() is target
    assert listing.verticalScrollBar().value() == scrolled


def _a_visible_row(listing):
    """Return a project row the reviewer could actually click on."""
    on_screen = [
        item
        for item in listing.project_items()
        if listing.viewport().rect().contains(listing.visualItemRect(item).center())
    ]
    assert on_screen, "no row is on screen; the test cannot click one"
    return on_screen[len(on_screen) // 2]


def _click_row(app, listing, item) -> None:
    """Press and release the pointer on *item*, as Qt would deliver it."""
    point = listing.visualItemRect(item).center()
    for kind in (QMouseEvent.MouseButtonPress, QMouseEvent.MouseButtonRelease):
        app.sendEvent(
            listing.viewport(),
            QMouseEvent(
                kind,
                point,
                listing.viewport().mapToGlobal(point),
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier,
            ),
        )
    app.processEvents()


def test_picking_a_project_does_rebuild_its_lamellae(window):
    """The half that must still happen: the lamellae are the project's own."""
    _select(window, 3)
    listed = [
        window.project_panel.lamella_list.item(row).text()
        for row in range(window.project_panel.lamella_list.count())
    ]

    assert listed == ["01-p3a", "02-p3b"]


def test_stepping_a_lamella_leaves_the_project_rows_untouched(window):
    """Moving within a project is not a change to the list of projects."""
    _select(window, 5)
    before = _project_rows(window)

    window._step_lamella(1)

    assert all(new is old for new, old in zip(_project_rows(window), before))
    assert window.controller.lamella.name == "02-p5b"


def test_a_scan_does_rebuild_both_lists(window, archive: Path):
    """The rows are stale after a scan, so that one must replace them."""
    before = _project_rows(window)

    window.scan(str(archive))

    assert all(new is not old for new, old in zip(_project_rows(window), before))


# ---------------------------------------------------------------------------
# Choosing a folder
# ---------------------------------------------------------------------------

@pytest.fixture
def foldered(qt_app, tmp_path: Path):
    """A window over two session folders, which is what the tree is for."""
    root = tmp_path / "sessions"
    for folder, count in (("session_a", 2), ("session_b", 1)):
        for index in range(count):
            project = root / folder / f"AutoLamella-2026-07-28-13-{index:02d}"
            project.mkdir(parents=True)
            (project / "experiment.yaml").write_text("name: p")
            stage = project / "01-alpha" / "adaptive_polish_2026-07-28-11-15-36AM"
            path = stage / "sem" / "01-alpha_AP_img_000_SEM.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(path), np.full((32, 32), 100, dtype=np.uint8))

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    window.scan(str(root))
    qt_app.processEvents()
    yield window
    destroy(window)


def _folder_row(window, name: str):
    tree = window.project_panel.project_list
    return next(
        tree.topLevelItem(row)
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).text(0) == name
    )


def test_choosing_a_folder_says_what_is_under_it(foldered):
    """The question a closed folder raises: is there anything in here worth
    opening."""
    tree = foldered.project_panel.project_list

    tree.setCurrentItem(_folder_row(foldered, "session_a"))

    summary = foldered.project_panel.summary.text()
    assert "session_a" in summary
    assert "2 projects" in summary
    assert "2 lamellae" in summary


def test_choosing_a_folder_leaves_the_view_where_it_was(foldered):
    """A folder holds no frame, so moving to one must not move the view."""
    tree = foldered.project_panel.project_list
    before = foldered.controller.project.name

    tree.setCurrentItem(_folder_row(foldered, "session_b"))

    assert foldered.controller.project.name == before


def test_choosing_a_project_again_describes_the_frame(foldered):
    """The folder's count is what the tree is on, not a mode it stays in."""
    tree = foldered.project_panel.project_list
    tree.setCurrentItem(_folder_row(foldered, "session_a"))

    tree.setCurrentItem(tree.project_item(0))

    assert "frames" in foldered.project_panel.summary.text()


def test_a_refresh_does_not_replace_a_folder_count_with_the_frame(foldered):
    """Held as state rather than written straight to the label: everything the
    window does ends in a refresh, and one of those must not undo the answer."""
    tree = foldered.project_panel.project_list
    tree.setCurrentItem(_folder_row(foldered, "session_a"))

    foldered.refresh()

    assert "2 projects" in foldered.project_panel.summary.text()
