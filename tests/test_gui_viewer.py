"""The viewer: the curation window opened only to look.

Drives a real window built over a read-only controller, against a project
written by :func:`conftest.build_dataset` with real image files, offscreen.
What these pin is that a viewer shows the Curate view and nothing that writes,
offers no key that would, and leaves the collection exactly as it found it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from conftest import build_dataset, destroy, settle

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGE_STACK),
    reason="Qt binding or optional image libs not installed",
)

if HAS_QT and HAS_IMAGE_STACK:
    import numpy as np
    import tifffile
    from qtpy.QtCore import QSettings

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.frame_strip import CURATE_BAND
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow
    from am_segmentation_manager.identity import VIEWER_NAME
    from am_segmentation_manager.launcher import menu_label

#: The smallest display the window is held to; see test_gui_window.py.
SMALLEST_SUPPORTED_SCREEN = (1440, 900)


def _collection(root: Path) -> Path:
    """Write a two-lamella project under *root* whose images can be read.

    ``build_dataset`` leaves images empty, which is enough for a scan but not
    for a frame to be drawn; each is given a small real image here.
    """
    build_dataset(root, lamellae=2, acquisitions=2, labels="annotated", plots=False)
    for image in root.rglob("*.tif"):
        if image.stat().st_size == 0:
            tifffile.imwrite(str(image), np.full((16, 16), 100, dtype=np.uint8))
    return root


def _snapshot(root: Path) -> dict[str, tuple[int, int] | None]:
    """Return every path under *root* with its size and modification time."""
    return {
        str(path.relative_to(root)): (
            None if path.is_dir() else (path.stat().st_size, path.stat().st_mtime_ns)
        )
        for path in sorted(root.rglob("*"))
    }


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    """A throwaway preferences file, so no test touches the real window's."""
    return QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    return _collection(tmp_path / "scan")


def _window(qt_app, settings: QSettings, *, read_only: bool) -> CurationWindow:
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 2: "lamella"},
    )
    controller = CurationController(cache=FrameCache(), config=config, read_only=read_only)
    return CurationWindow(controller, settings=settings)


@pytest.fixture
def viewer(qt_app, settings: QSettings, collection: Path):
    """A viewer with the collection scanned."""
    window = _window(qt_app, settings, read_only=True)
    window.scan(str(collection))
    settle(qt_app, lambda: window.controller.project is not None, what="the scan")
    yield window
    destroy(window)


def test_a_viewer_shows_the_curate_view_alone(viewer):
    """No tab row, and no column holding the state, families or operations."""
    assert viewer.tabs.count() == 1
    assert viewer.tabs.currentWidget() is viewer.curate
    assert not viewer.tabs.tabBar().isVisibleTo(viewer)
    assert viewer.curate.count() == 2
    for hidden in (
        viewer.state_panel.enable_check,
        viewer.state_panel,
        viewer.families_panel,
        viewer.operations_panel,
        viewer.side_tabs,
        viewer.metadata_panel,
        viewer.label_panel,
        viewer.export_panel,
    ):
        assert not hidden.isVisibleTo(viewer), type(hidden).__name__


def test_a_viewer_can_still_open_a_folder_and_ask_for_help(viewer):
    """The corner's row survives the tab bar going, less who would record."""
    corner = viewer._corner
    for shown in (viewer.scan_bar, corner.settings_button, corner.help_button, corner.mark):
        assert shown.isVisibleTo(viewer)
    assert not corner.author.isVisibleTo(viewer)


def test_a_viewer_says_it_is_one_in_its_title(viewer, collection):
    assert viewer.windowTitle().startswith(VIEWER_NAME)
    assert collection.name in viewer.windowTitle()


def test_a_viewer_offers_no_launcher_for_the_window_that_edits(viewer):
    label = menu_label()
    if label is None:
        pytest.skip("this platform has no launcher to offer")
    assert label not in viewer.menu_actions()


def test_a_viewer_binds_no_key_that_changes_anything(viewer):
    keys = {binding.key for binding in viewer._keys.bindings if binding.tab == "curate"}

    assert not any(binding.edits for binding in viewer._keys.bindings)
    assert not keys & {"D", "S", "C", "Esc"}
    assert {"Left", "Right", "Up", "Down", "V"} <= keys


def test_a_viewer_lists_no_key_or_gesture_that_changes_anything(viewer):
    """F1 must not teach what the window will not do, nor name a switch it
    does not have."""
    listed = viewer._keys.help_html()

    for absent in ("Set state", "Enable editing", "state change", "Operations"):
        assert absent not in listed
    assert "Next frame" in listed


def test_a_viewer_timeline_does_not_explain_a_band_it_never_draws(viewer):
    viewer.timeline._show_card(0)
    card = viewer.timeline._card
    assert card is not None
    assert CURATE_BAND not in card.meanings
    assert "Where you are" in card.meanings


def test_a_ctrl_click_on_the_timeline_picks_nothing_in_a_viewer(viewer):
    viewer.timeline.frame_toggled.emit(1)
    viewer.timeline.extend_started.emit(0)
    viewer.timeline.extend_reached.emit(1)

    assert viewer.controller.selection.is_empty


def test_a_viewer_fits_the_smallest_screen_supported(viewer):
    wide, tall = SMALLEST_SUPPORTED_SCREEN
    assert viewer.minimumSizeHint().width() <= wide
    assert viewer.minimumSizeHint().height() <= tall


def test_a_viewer_keeps_its_layout_apart_from_the_full_window(qt_app, settings, viewer):
    """Its Curate splitter has two panes to the full window's three, so each
    restoring the other's would misplace a divider; and it has no model box
    whose contents to remember."""
    viewer._save_layout()
    settings.sync()

    keys = set(settings.allKeys())
    assert {"viewer/geometry", "viewer/splitter/curate", "viewer/splitter/lists"} <= keys
    assert not {key for key in keys if key.startswith(("geometry", "splitter", "model"))}

    full = _window(qt_app, settings, read_only=False)
    try:
        assert full.curate.count() == 3
        assert all(size > 0 for size in full.curate.sizes())
    finally:
        destroy(full)


def test_browsing_in_a_viewer_leaves_the_collection_as_it_found_it(
    qt_app, viewer, collection
):
    before = _snapshot(collection)

    viewer._step_frame(1)
    viewer._step_lamella(1)
    viewer._set_mode("fib", from_user=True)
    viewer._set_mode("grid", from_user=True)
    viewer.rescan()
    settle(qt_app, lambda: viewer.controller.project is not None, what="the rescan")
    viewer.close()

    assert _snapshot(collection) == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_viewer_reads_a_collection_it_cannot_write_to(qt_app, settings, tmp_path):
    """As on a microscope given the share read-only."""
    root = _collection(tmp_path / "locked")
    paths = sorted(root.rglob("*"), reverse=True) + [root]
    for path in paths:
        path.chmod(0o555 if path.is_dir() else 0o444)
    window = _window(qt_app, settings, read_only=True)
    try:
        window.scan(str(root))
        settle(qt_app, lambda: window.controller.project is not None, what="the scan")
        assert len(window.controller.lamellae) == 2
    finally:
        destroy(window)
        for path in reversed(paths):
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_build_window_opens_a_viewer_when_asked(qt_app, settings, monkeypatch):
    """`am-seg gui --viewer` reaches the window through app.build_window."""
    from am_segmentation_manager.gui import app as gui_app

    monkeypatch.setattr(gui_app, "install_error_reporting", lambda: None)
    monkeypatch.setattr(gui_app, "_apply_icon", lambda application: None)
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.theme.apply_theme",
        lambda application, accent=None: None,
    )

    _application, window = gui_app.build_window(settings=settings, viewer=True)
    try:
        qt_app.processEvents()
        assert window.read_only
        assert window.controller.read_only
    finally:
        destroy(window)


@pytest.mark.parametrize("flag", [True, False], ids=["--viewer", "plain"])
def test_the_gui_command_opens_a_viewer_only_when_asked(monkeypatch, flag):
    """Parsed and handled without `cli.main`, which would open the window's
    log file in the user's own configuration area."""
    from am_segmentation_manager.cli.curating import _cmd_gui
    from am_segmentation_manager.cli.main import build_parser
    from am_segmentation_manager.gui import app as gui_app

    opened: list[bool] = []
    monkeypatch.setattr(
        gui_app, "run",
        lambda start_path=None, config_path=None, *, viewer=False: opened.append(viewer) or 0,
    )
    args = build_parser().parse_args(["gui", *(["--viewer"] if flag else [])])

    assert _cmd_gui(args) == 0
    assert opened == [flag]
