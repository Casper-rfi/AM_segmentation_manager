"""Small widget behaviours shared between panels.

What lands here is something more than one panel *does* -- a context menu, an
event filter, asking the user for a path. What the window looks like belongs in
:mod:`theme`, anything that knows about projects, labels or the filesystem
belongs in a controller, and anything only one panel uses belongs in that panel.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QIcon, QPixmap, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QMenu,
    QWidget,
)

logger = logging.getLogger(__name__)

#: The name filters the CSV dialogs offer. ``All files`` is offered beside the
#: CSV filter when opening, because a CSV that arrived with another extension is
#: still readable and refusing to show it would be the dialog deciding.
CSV_FILTER = "CSV files (*.csv)"
ALL_FILES_FILTER = "All files (*)"


def chip_icon(pixmap: QPixmap) -> QIcon:
    """Return an icon Qt will draw the same way on every row.

    A view draws the icon of the row being looked at in ``QIcon.Selected``
    mode, and the variant Qt generates for that mode blends the pixmap with the
    highlight colour. For a status chip that is wrong twice over: the colour
    *is* the information, and it gets washed out on exactly the row the user is
    on -- a "done" teal came out as a blue-green against a blue-tinted row.

    Supplying the pixmap for every mode and state leaves nothing for Qt to
    generate.
    """
    icon = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active, QIcon.Disabled):
        for state in (QIcon.On, QIcon.Off):
            icon.addPixmap(pixmap, mode, state)
    return icon


#: Monospaced families to try, in the order they are looked for. The style hint
#: below is only consulted for a family that does not exist, so the list has to
#: carry a real name for each platform rather than relying on the hint alone.
_FIXED_WIDTH_FAMILIES = ("Menlo", "Consolas", "DejaVu Sans Mono", "Courier New")


def fixed_width_font() -> QFont:
    """Return a font whose characters are all the same width.

    Anything padded into columns with spaces -- the breakdown table, the
    operation reports -- is only aligned in a font like this.

    Not ``QFontDatabase.systemFont(FixedFont)``, which is the obvious call and
    returns the proportional UI font on macOS: it reports the family it was
    asked for while resolving to something else, so the padding stays padding
    and nothing lines up. Naming families and letting the style hint cover the
    ones that are missing is what actually resolves.
    """
    font = QFont()
    font.setFamilies(list(_FIXED_WIDTH_FAMILIES))
    font.setStyleHint(QFont.Monospace)
    font.setFixedPitch(True)
    return font


def caption_for(text: str, widget: QWidget, tooltip: str) -> QLabel:
    """Return a caption for *widget*, putting *tooltip* on both.

    A user reaches for the word naming a control, not the control itself, so a
    tooltip on the control alone is one nobody finds. Returning the label keeps
    the two impossible to write apart.
    """
    label = QLabel(text)
    label.setToolTip(tooltip)
    widget.setToolTip(tooltip)
    return label


class StepCombo(QComboBox):
    """A combo box whose Up and Down keys always move through the list.

    Qt's own handling of those keys turns on ``SH_ComboBox_Popup``: where a
    style sets it, the keys open the popup instead of stepping. Under PyQt5
    here they step, but that hint is a style's to answer and this window runs
    under a different binding and a different macOS than it is developed on --
    the gap ``docs/decisions.md`` § Environment is about. One code path rather
    than a style's is the same choice already made for the file dialogs.

    Stepping skips disabled rows, so a grouped list arrows through its entries
    and not its headings. Alt-Down still opens the popup, and every other key
    is Qt's.
    """

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and not event.modifiers():
            self.step(1 if event.key() == Qt.Key_Down else -1)
            event.accept()
            return
        super().keyPressEvent(event)

    def step(self, delta: int) -> None:
        """Move to the next selectable row in *delta*'s direction.

        Stops at the ends rather than wrapping: a list that jumps from the
        last row back to the first loses your place in a long one.
        """
        wanted = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        index = self.currentIndex() + delta
        while 0 <= index < self.count():
            # Asked of the model rather than the item, so this works whatever
            # model the combo was given.
            if self.model().index(index, self.modelColumn()).flags() & wanted == wanted:
                self.setCurrentIndex(index)
                return
            index += delta


def fill_grouped_combo(
    combo: QComboBox,
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    first: tuple[str, str] | None = None,
) -> None:
    """Fill *combo* with rows gathered under non-selectable headings.

    Each row carries its identifier as item data, so a caller reads
    ``currentData()`` and never has to map a display label back to a name.
    A heading carries none, which is what distinguishes the two.

    Args:
        groups: ``(heading, rows)`` pairs, where each row is
            ``(identifier, label)``.
        first: A row to place above every group, for an entry that belongs to
            no group -- an "any" or "none" option.
    """
    model = QStandardItemModel(combo)
    if first is not None:
        name, label = first
        item = QStandardItem(label)
        item.setData(name, Qt.UserRole)
        model.appendRow(item)
    for heading, rows in groups:
        item = QStandardItem(heading)
        # NoItemFlags rather than clearing ItemIsSelectable alone: a heading
        # must also be skipped when stepping through the list with the arrow
        # keys, and Qt skips disabled rows for free.
        item.setFlags(Qt.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        model.appendRow(item)
        for name, label in rows:
            row = QStandardItem(label)
            row.setData(name, Qt.UserRole)
            model.appendRow(row)
    combo.setModel(model)


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
    ``docs/decisions.md`` § Environment.

    Args:
        mode: What the user is picking -- a file, or a directory.
        save: Whether the dialog names a file to write rather than one to read.
        directory: Where the dialog opens. Falls back to the home directory.
        filters: Name filters, the first of which is selected.
        default_suffix: Appended when a save name is typed without one, so a
            file cannot be written with no extension.
    """
    dialog = QFileDialog(parent, caption)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(mode)
    dialog.setAcceptMode(QFileDialog.AcceptSave if save else QFileDialog.AcceptOpen)
    if mode == QFileDialog.Directory:
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
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
            mode=QFileDialog.ExistingFile,
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
            mode=QFileDialog.AnyFile,
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
            parent, caption, mode=QFileDialog.Directory, directory=directory
        )
    )


def add_path_menu(
    widget: QListWidget, noun: str, copy_signal, reveal_signal
) -> None:
    """Give a list a right-click menu for the row's filesystem path.

    The row under the cursor is used, not the selected row: right-clicking an
    entry and acting on a different one would be a surprising way to lose track
    of which path you copied. The row number is emitted rather than a path,
    because resolving one is the controller's job, not a widget's.
    """
    widget.setContextMenuPolicy(Qt.CustomContextMenu)

    def show_menu(point) -> None:
        item = widget.itemAt(point)
        if item is None:
            return
        row = widget.row(item)
        menu = QMenu(widget)
        copy = menu.addAction(f"Copy {noun} path")
        reveal = menu.addAction(f"Reveal {noun} in file manager")
        chosen = menu.exec_(widget.viewport().mapToGlobal(point))
        if chosen is copy:
            copy_signal.emit(row)
        elif chosen is reveal:
            reveal_signal.emit(row)

    widget.customContextMenuRequested.connect(show_menu)
