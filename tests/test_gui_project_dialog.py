"""Choosing which directories become projects.

The dialog carries the one decision in the window that the code must not make
for itself: which level of a tree is "the project". These tests drive it
without showing it, because what matters is what it proposes, what it ticks and
what it hands back -- none of which needs a screen.
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
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.gui.operations_controller import OperationsController
    from am_segmentation_manager.gui.widgets.project_dialog import (
        ROW_KIND,
        ROW_PATH,
        AddProjectsDialog,
    )


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _image(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "img.tif").write_bytes(b"i")


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """A marked project with no metadata file, an unindexed dataset, and a
    folder two datasets of different eras were gathered in."""
    _image(tmp_path / "indexed" / "lam_001" / "sem")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")

    _image(tmp_path / "unindexed" / "lam_001" / "sem")

    _image(tmp_path / "to sort" / "2024_dataset" / "lam_001" / "sem")
    _image(tmp_path / "to sort" / "2026_dataset" / "lam_001" / "stage" / "sem")
    return tmp_path


@pytest.fixture
def dialog(qt_app, collection: Path) -> AddProjectsDialog:
    controller = OperationsController()
    controller.set_start_path(collection)
    return AddProjectsDialog(None, controller, collection)


def _rows(dialog) -> list[tuple[str, str, bool]]:
    """Return every non-heading row as (name, kind, ticked)."""
    return [
        (
            Path(item.data(ROW_PATH)).name,
            item.data(ROW_KIND),
            item.checkState() == Qt.Checked,
        )
        for row in range(dialog.rows.count())
        if (item := dialog.rows.item(row)).data(ROW_KIND) != "heading"
    ]


def test_the_two_groups_are_listed_separately(dialog):
    """A project needing only its descriptive fields is not the same decision
    as a directory that may not be a project at all."""
    assert _rows(dialog) == [
        ("indexed", "project", True),
        ("to sort", "candidate", False),
        ("unindexed", "candidate", True),
    ]


def test_a_gathering_folder_is_listed_but_not_ticked(dialog):
    """Unticked rather than hidden: it is the row you act on, and hiding it
    would leave the datasets inside it unreachable."""
    ticked = dict((name, state) for name, _kind, state in _rows(dialog))
    assert ticked["to sort"] is False
    assert "to sort" not in [path.name for path in dialog.chosen()]


def test_a_gathering_folder_says_why_it_was_not_ticked(dialog):
    row = next(
        dialog.rows.item(index)
        for index in range(dialog.rows.count())
        if dialog.rows.item(index).data(ROW_PATH)
        and Path(dialog.rows.item(index).data(ROW_PATH)).name == "to sort"
    )
    assert "gathers several datasets" in row.text()
    assert "2024_dataset" in row.text()


def test_looking_inside_replaces_the_row_with_what_is_in_it(dialog, collection: Path):
    """The action that makes a wrong level correctable rather than fatal."""
    index = next(
        row
        for row in range(dialog.rows.count())
        if dialog.rows.item(row).data(ROW_PATH) == str(collection / "to sort")
    )
    dialog.rows.setCurrentRow(index)
    assert dialog.descend_button.isEnabled()

    dialog._descend()

    names = [name for name, _kind, _ticked in _rows(dialog)]
    assert "to sort" not in names
    assert names == ["indexed", "2024_dataset", "2026_dataset", "unindexed"]
    assert [path.name for path in dialog.chosen()] == [
        "indexed", "2024_dataset", "2026_dataset", "unindexed",
    ]


def test_looking_inside_is_offered_only_for_a_proposal(dialog, collection: Path):
    """An indexed project has no level to argue about."""
    index = next(
        row
        for row in range(dialog.rows.count())
        if dialog.rows.item(row).data(ROW_PATH) == str(collection / "indexed")
    )
    dialog.rows.setCurrentRow(index)
    assert not dialog.descend_button.isEnabled()


def test_adding_the_folder_itself_is_refused_for_a_collection(
    qt_app, collection: Path, monkeypatch
):
    """The escape hatch is for data the scan cannot see, not for overriding the
    check that stops several samples landing under one metadata record."""
    controller = OperationsController()
    controller.set_start_path(collection)
    dialog = AddProjectsDialog(None, controller, collection / "to sort")

    warned: list = []
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.project_dialog.QMessageBox.warning",
        lambda *args, **kwargs: warned.append(args),
    )
    dialog._add_itself()

    assert warned
    assert str(collection / "to sort") not in [str(path) for path in dialog.chosen()]


def test_adding_the_folder_itself_offers_a_folder_the_scan_cannot_see(
    qt_app, tmp_path: Path
):
    """Images that are not in a directory named sem or fib are invisible to the
    search, which is the whole reason this button exists."""
    unrecognised = tmp_path / "odd_dataset" / "pictures"
    unrecognised.mkdir(parents=True)
    (unrecognised / "img.tif").write_bytes(b"i")

    controller = OperationsController()
    controller.set_start_path(tmp_path)
    dialog = AddProjectsDialog(None, controller, tmp_path / "odd_dataset")

    assert dialog.chosen() == []

    dialog._add_itself()

    assert [path.name for path in dialog.chosen()] == ["odd_dataset"]


def test_nothing_to_do_leaves_the_write_button_off(qt_app, tmp_path: Path):
    controller = OperationsController()
    controller.set_start_path(tmp_path)
    dialog = AddProjectsDialog(None, controller, tmp_path)

    assert dialog.chosen() == []
    from qtpy.QtWidgets import QDialogButtonBox

    assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_ticking_a_row_enables_writing(qt_app, tmp_path: Path):
    """A list whose only proposal gathers datasets opens with nothing ticked,
    so ticking is what has to wake the button."""
    from qtpy.QtWidgets import QDialogButtonBox

    _image(tmp_path / "indexed" / "lam_001" / "sem")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")
    (tmp_path / "indexed" / "AM_metadata.yaml").write_text("metadata:\n")
    _image(tmp_path / "to sort" / "2024_dataset" / "lam_001" / "sem")
    _image(tmp_path / "to sort" / "2026_dataset" / "lam_001" / "stage" / "sem")

    controller = OperationsController()
    controller.set_start_path(tmp_path)
    dialog = AddProjectsDialog(None, controller, tmp_path)
    write = dialog.buttons.button(QDialogButtonBox.Ok)

    assert dialog.chosen() == []
    assert not write.isEnabled()

    row = next(
        dialog.rows.item(index)
        for index in range(dialog.rows.count())
        if dialog.rows.item(index).data(ROW_KIND) == "candidate"
    )
    row.setCheckState(Qt.Checked)

    assert write.isEnabled()
    assert [path.name for path in dialog.chosen()] == ["to sort"]
