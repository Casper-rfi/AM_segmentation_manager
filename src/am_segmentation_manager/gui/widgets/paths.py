"""Asking the user for a path, and offering one that is already on screen.

Every dialog here is Qt's own rather than the platform's, for the reason
`docs/display.md` gives under The binding: this package decides what a file
dialog does rather than letting a platform decide it.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import (
    Qt,
)
from qtpy.QtWidgets import (
    QFileDialog,
    QMenu,
    QWidget,
)

logger = logging.getLogger(__name__)


#: The name filters the CSV dialogs offer. ``All files`` is offered beside the
#: CSV filter when opening, because a CSV that arrived with another extension is
#: still readable and refusing to show it would be the dialog deciding.
CSV_FILTER = "CSV files (*.csv)"


ALL_FILES_FILTER = "All files (*)"


def build_path_dialog(
    parent: QWidget | None,
    caption: str,
    *,
    mode: QFileDialog.FileMode,
    save: bool = False,
    directory: Path | str | None = None,
    filters: tuple[str, ...] = (),
    default_suffix: str | None = None,
) -> QFileDialog:
    """Return a configured file dialog, without showing it.

    Every dialog in the window is built here, non-native, so that one binding
    and one platform cannot behave differently from another; see
    ``docs/display.md`` § The binding.

    Args:
        mode: What the user is picking -- a file, or a directory.
        save: Whether the dialog names a file to write rather than one to read.
        directory: Where the dialog opens. Falls back to the home directory.
        filters: Name filters, the first of which is selected.
        default_suffix: Appended when a save name is typed without one, so a
            file cannot be written with no extension.
    """
    dialog = QFileDialog(parent, caption)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(mode)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave if save else QFileDialog.AcceptMode.AcceptOpen)
    if mode == QFileDialog.FileMode.Directory:
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    if filters:
        dialog.setNameFilters(list(filters))
    if default_suffix:
        dialog.setDefaultSuffix(default_suffix)
    dialog.setDirectory(str(directory) if directory else str(Path.home()))
    return dialog


def _chosen(dialog: QFileDialog) -> Path | None:
    """Show *dialog* and return what was picked, or None if it was cancelled."""
    if not dialog.exec_():
        return None
    selected = dialog.selectedFiles()
    return Path(selected[0]) if selected else None


def choose_open_file(
    parent: QWidget | None,
    caption: str,
    directory: Path | str | None = None,
    filters: tuple[str, ...] = (CSV_FILTER, ALL_FILES_FILTER),
) -> Path | None:
    """Ask for an existing file to read."""
    return _chosen(
        build_path_dialog(
            parent,
            caption,
            mode=QFileDialog.FileMode.ExistingFile,
            directory=directory,
            filters=filters,
        )
    )


def choose_save_file(
    parent: QWidget | None,
    caption: str,
    directory: Path | str | None = None,
    filters: tuple[str, ...] = (CSV_FILTER,),
    default_suffix: str = "csv",
) -> Path | None:
    """Ask for a file to write, confirming a replacement if one is picked."""
    return _chosen(
        build_path_dialog(
            parent,
            caption,
            mode=QFileDialog.FileMode.AnyFile,
            save=True,
            directory=directory,
            filters=filters,
            default_suffix=default_suffix,
        )
    )


def choose_directory(
    parent: QWidget | None, caption: str, directory: Path | str | None = None
) -> Path | None:
    """Ask for a directory."""
    return _chosen(
        build_path_dialog(
            parent, caption, mode=QFileDialog.FileMode.Directory, directory=directory
        )
    )


def add_path_menu(
    widget, noun: str, copy_signal, reveal_signal, row_of=None
) -> None:
    """Give a list or tree a right-click menu for the row's filesystem path.

    The row under the cursor is used, not the selected row: right-clicking an
    entry and acting on a different one would be a surprising way to lose track
    of which path you copied. The row number is emitted rather than a path,
    because resolving one is the controller's job, not a widget's.

    Args:
        row_of: How to turn the item under the cursor into the number the
            controller knows it by, for a widget where that is not its position.
            Returning None offers no menu, which is what a tree's folder rows
            need: a folder is not a project and has no path to copy.
    """
    widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    resolve = row_of if row_of is not None else widget.row

    def show_menu(point) -> None:
        item = widget.itemAt(point)
        if item is None:
            return
        row = resolve(item)
        if row is None:
            return
        menu = QMenu(widget)
        copy = menu.addAction(f"Copy {noun} path")
        reveal = menu.addAction(f"Reveal {noun} in file manager")
        chosen = menu.exec_(widget.viewport().mapToGlobal(point))
        if chosen is copy:
            copy_signal.emit(row)
        elif chosen is reveal:
            reveal_signal.emit(row)

    widget.customContextMenuRequested.connect(show_menu)
