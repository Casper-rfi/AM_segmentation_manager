"""How the window asks for a path.

Every dialog is configured in one place and shown by nobody here: the
configuration is what has gone wrong in practice -- a platform dialog that
greyed out the CSVs it was asked to offer -- and it can be asserted without a
dialog ever appearing.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")

if HAS_QT:
    from qtpy.QtWidgets import QApplication, QFileDialog

    from am_segmentation_manager.gui.export_controller import ExportController
    from am_segmentation_manager.gui.widgets import export_panel as export_panel_module
    from am_segmentation_manager.gui.widgets.export_panel import ExportPanel
    from am_segmentation_manager.gui.widgets.paths import (
        ALL_FILES_FILTER,
        CSV_FILTER,
        build_path_dialog,
    )


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def panel(qt_app) -> ExportPanel:
    return ExportPanel(ExportController())


def open_dialog(**kwargs) -> QFileDialog:
    return build_path_dialog(
        None, "Open", mode=QFileDialog.ExistingFile,
        filters=(CSV_FILTER, ALL_FILES_FILTER), **kwargs,
    )


# Built conditionally: parametrize arguments are evaluated when the module is
# imported, which happens before pytestmark can skip anything. Naming a Qt enum
# at module level is what stops the whole file collecting in an environment
# without a Qt binding -- the inference environment is one.
DIALOG_MODES = (
    [QFileDialog.ExistingFile, QFileDialog.AnyFile, QFileDialog.Directory]
    if HAS_QT else []
)


# -- configuration ----------------------------------------------------------

@pytest.mark.parametrize("mode", DIALOG_MODES)
def test_no_dialog_is_left_to_the_platform(qt_app, mode):
    """The reason this helper exists: the macOS panel under Qt 5 offered no
    files at all, whatever name filter it was given."""
    dialog = build_path_dialog(None, "Pick", mode=mode)
    assert dialog.testOption(QFileDialog.DontUseNativeDialog)


def test_an_open_dialog_offers_csvs_first_and_everything_else_beside_them(qt_app):
    assert open_dialog().nameFilters() == [CSV_FILTER, ALL_FILES_FILTER]


def test_a_save_dialog_supplies_the_extension_that_was_not_typed(qt_app):
    dialog = build_path_dialog(
        None, "Save", mode=QFileDialog.AnyFile, save=True, default_suffix="csv"
    )
    assert dialog.defaultSuffix() == "csv"
    assert dialog.acceptMode() == QFileDialog.AcceptSave


def test_a_directory_dialog_offers_only_directories(qt_app):
    dialog = build_path_dialog(None, "Folder", mode=QFileDialog.Directory)
    assert dialog.fileMode() == QFileDialog.Directory
    assert dialog.testOption(QFileDialog.ShowDirsOnly)


def test_a_dialog_opens_where_it_is_asked_to(qt_app, tmp_path):
    dialog = open_dialog(directory=tmp_path)
    assert Path(dialog.directory().absolutePath()) == tmp_path


def test_a_dialog_asked_for_nowhere_opens_at_home(qt_app):
    assert Path(open_dialog().directory().absolutePath()) == Path.home()


# -- what the export panel does with the answer -----------------------------

def test_the_csv_to_check_is_reported_as_a_path(panel, tmp_path, monkeypatch):
    chosen = tmp_path / "dataset_train.csv"
    monkeypatch.setattr(
        export_panel_module, "choose_open_file", lambda *args, **kwargs: chosen
    )
    seen: list[str] = []
    panel.csv_check_requested.connect(seen.append)

    panel._check()

    assert seen == [str(chosen)]


def test_a_cancelled_dialog_asks_for_nothing(panel, monkeypatch):
    monkeypatch.setattr(
        export_panel_module, "choose_open_file", lambda *args, **kwargs: None
    )
    seen: list[str] = []
    panel.csv_check_requested.connect(seen.append)

    panel._check()

    assert seen == []


def test_the_next_dialog_opens_where_the_last_one_left_off(panel, tmp_path, monkeypatch):
    """Writing a CSV and then checking it should not mean navigating twice."""
    written = tmp_path / "dataset_train.csv"
    monkeypatch.setattr(
        export_panel_module, "choose_save_file", lambda *args, **kwargs: written
    )
    panel._choose(panel.image_metadata_export_requested, "Per-image", False)

    assert panel._last_directory == tmp_path

    opened: list[Path | None] = []

    def record(_parent, _caption, directory=None, *args, **kwargs):
        opened.append(directory)
        return None

    monkeypatch.setattr(export_panel_module, "choose_open_file", record)
    panel._check()

    assert opened == [tmp_path]
