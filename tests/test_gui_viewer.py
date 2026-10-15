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

from conftest import build_dataset, destroy, settle, widest_layout

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
    assert viewer.tabs.currentWidget() is viewer.curate_panel
    assert not viewer.tabs.tabBar().isVisibleTo(viewer)
    assert viewer.curate_panel.columns.count() == 2
    for hidden in (
        viewer.curate_panel.state_panel.enable_check,
        viewer.curate_panel.state_panel,
        viewer.curate_panel.families_panel,
        viewer.curate_panel.operations_panel,
        viewer.curate_panel.side_tabs,
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


def _has_tick_box(item, *column) -> bool:
    """Whether a list or tree row can be ticked, or draws a box as if it could."""
    from qtpy.QtCore import Qt

    return bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable) or (
        item.data(*column, Qt.ItemDataRole.CheckStateRole) is not None
    )


def test_a_viewer_lists_carry_no_tick_boxes(viewer):
    """A tick says what a change applies to, and a viewer makes none -- not
    even after the window puts the ticks back, as it does after every pick."""
    from qtpy.QtWidgets import QPushButton

    panel = viewer.curate_panel.project_panel
    viewer._refresh_checks()

    lamellae = [panel.lamella_list.item(row) for row in range(panel.lamella_list.count())]
    projects = panel.project_list.project_items()
    assert lamellae and projects
    assert not any(_has_tick_box(item) for item in lamellae)
    assert not any(_has_tick_box(item, 0) for item in projects)
    assert not [
        button.text() for button in panel.findChildren(QPushButton)
        if button.text() in ("Tick all", "None")
    ]


def test_a_viewer_settings_ask_no_name_and_offer_only_what_it_opens_on(qt_app, settings):
    from qtpy.QtWidgets import QGroupBox

    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings), viewer=True)
    try:
        assert dialog.author_edit is None
        assert dialog.author is None
        assert set(dialog.values) == {
            entry.key for _title, entries in preferences.groups(viewer=True) for entry in entries
        }
        assert [box.title() for box in dialog.findChildren(QGroupBox)] == [
            preferences.APPEARANCE, preferences.SHOWN,
        ]
    finally:
        destroy(dialog)


def test_keeping_a_viewers_settings_leaves_the_recorded_name_alone(
    viewer, settings, monkeypatch
):
    """With no name asked, keeping the dialog must not read as clearing one."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets import main_window
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    touched: list[str] = []
    monkeypatch.setattr(main_window, "set_author", lambda name: touched.append(name))
    monkeypatch.setattr(main_window, "clear_author", lambda: touched.append("cleared"))

    def keep(dialog):
        dialog._editors["label_opacity"].setValue(55)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SettingsDialog, "exec_", keep)

    viewer.show_settings()

    assert touched == []
    assert preferences.read(settings)["label_opacity"] == 55


def test_a_viewer_says_it_is_one_in_its_title(viewer, collection):
    assert viewer.windowTitle().startswith(VIEWER_NAME)
    assert collection.name in viewer.windowTitle()


def test_a_viewer_offers_its_own_launcher_first(viewer, monkeypatch, tmp_path):
    """How someone at the microscope adds the viewer without a terminal."""
    from am_segmentation_manager.gui.widgets import main_window
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog
    from am_segmentation_manager.launcher import VIEWER, Places, ThisComputer

    label = menu_label()
    if label is None:
        pytest.skip("this platform has no launcher to offer")
    places = Places(menu=tmp_path / "menu", desktop=tmp_path / "Desktop", icons=tmp_path / "icons")
    monkeypatch.setattr(
        main_window, "this_computer", lambda: ThisComputer("linux", places, Path("/env/python"))
    )
    shown: list = []
    monkeypatch.setattr(LauncherDialog, "exec_", lambda self: shown.append(self) or 0)

    viewer.menu_actions()[label].trigger()

    assert [dialog.app for dialog in shown] == [VIEWER]
    destroy(shown[0])


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
    viewer.curate_panel.timeline._show_card(0)
    card = viewer.curate_panel.timeline._card
    assert card is not None
    assert CURATE_BAND not in card.meanings
    assert "Where you are" in card.meanings


def test_a_ctrl_click_on_the_timeline_picks_nothing_in_a_viewer(viewer):
    viewer.curate_panel.timeline.frame_toggled.emit(1)
    viewer.curate_panel.timeline.extend_started.emit(0)
    viewer.curate_panel.timeline.extend_reached.emit(1)

    assert viewer.controller.selection.is_empty


def test_a_viewer_fits_the_smallest_screen_supported(viewer):
    wide, tall = SMALLEST_SUPPORTED_SCREEN
    assert viewer.minimumSizeHint().width() <= widest_layout(wide)
    assert viewer.minimumSizeHint().height() <= tall


def test_a_viewer_keeps_its_layout_apart_from_the_full_window(qt_app, settings, viewer):
    """Its Curate splitter has two panes to the full window's three, so each
    restoring the other's would misplace a divider; and it has no model box
    whose contents to remember."""
    viewer._save_layout()
    settings.sync()

    assert sorted(settings.allKeys()) == [
        "scan_root", "viewer/geometry", "viewer/splitter/curate", "viewer/splitter/lists",
    ]

    full = _window(qt_app, settings, read_only=False)
    try:
        assert full.curate_panel.columns.count() == 3
        assert all(size > 0 for size in full.curate_panel.columns.sizes())
    finally:
        destroy(full)


def test_browsing_in_a_viewer_leaves_the_collection_as_it_found_it(
    qt_app, viewer, collection
):
    before = _snapshot(collection)

    viewer.curate_panel.step_frame(1)
    viewer.curate_panel.step_lamella(1)
    viewer.curate_panel.set_mode("fib", from_user=True)
    viewer.curate_panel.set_mode("grid", from_user=True)
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


def test_build_window_opens_a_viewer_named_as_the_viewer(qt_app, settings, monkeypatch):
    """`am-seg gui --viewer` reaches the window through app.build_window, and
    names itself as the viewer's launcher does: the menu bar's name on macOS,
    and the desktop entry a Wayland desktop finds the window's icon by."""
    from qtpy.QtGui import QGuiApplication

    from am_segmentation_manager.gui import app as gui_app
    from am_segmentation_manager.identity import VIEWER_ID

    named: list[str] = []
    monkeypatch.setattr(gui_app, "install_error_reporting", lambda: None)
    monkeypatch.setattr(gui_app, "_apply_icon", lambda application: None)
    monkeypatch.setattr(gui_app, "set_bundle_name", lambda name: named.append(name) or True)
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.theme.apply_theme",
        lambda application, accent=None: None,
    )
    previous = QGuiApplication.desktopFileName()

    _application, window = gui_app.build_window(settings=settings, viewer=True)
    try:
        qt_app.processEvents()
        assert window.read_only
        assert window.controller.read_only
        assert named == [VIEWER_NAME]
        assert QGuiApplication.desktopFileName() == VIEWER_ID
    finally:
        destroy(window)
        QGuiApplication.setDesktopFileName(previous)


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
