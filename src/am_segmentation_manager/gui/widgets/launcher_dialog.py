"""Adding the window to this computer's applications, from the window itself.

The launcher ``am-seg create-shortcut`` writes, through
:mod:`~am_segmentation_manager.launcher`, for someone who never opens a
terminal. Every path is shown before anything is written. A launcher am-seg
wrote before may be written again or removed; anything else in the way
disables adding, and the dialog says what it is.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QPushButton,
    QVBoxLayout,
)

from ...identity import PRODUCT_NAME
from ...launcher import (
    WHERE_TO_FIND,
    Launcher,
    LauncherRefused,
    ThisComputer,
    existing,
    remove,
    write,
)
from .controls import ToggleSwitch, WrappingLabel

__all__ = ["LauncherDialog"]

#: How wide the dialog opens, in logical pixels: a path on one line where it can be.
DIALOG_WIDTH = 520


class LauncherDialog(QDialog):
    """Show the launcher this computer would get, and write or remove it when asked."""

    def __init__(self, parent, computer: ThisComputer) -> None:
        super().__init__(parent)
        self.computer = computer
        self.setWindowTitle(f"Add {PRODUCT_NAME} to this computer")
        self.setMinimumWidth(DIALOG_WIDTH)

        intro = WrappingLabel(
            f"Start {PRODUCT_NAME} like any other application. It starts with the "
            f"Python environment this window is running in, {computer.python}, so "
            "add it again if that environment moves."
        )
        intro.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.desktop = ToggleSwitch("Also on the Desktop")
        self.desktop.setChecked(computer.desktop_by_default)
        self.desktop.toggled.connect(self.refresh)
        self.paths = WrappingLabel()
        self.paths.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status = WrappingLabel()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove)
        buttons.addButton(self.add_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.remove_button, QDialogButtonBox.ButtonRole.ActionRole)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.desktop)
        layout.addWidget(self.paths)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

        self.launcher: Launcher
        self.refresh()

    def refresh(self) -> None:
        """Show what adding would write now, and whether it may."""
        self.launcher = self.computer.launcher(self.desktop.isChecked())
        ours, others = existing(self.launcher)
        # Removing reaches the Desktop whether or not it is being added to.
        removable, _ = existing(self.computer.launcher(desktop=True))
        written = "\n".join(str(entry.path) for entry in self.launcher.entries)
        self.paths.setText(f"Writes:\n{written}")
        self.add_button.setText("Write again" if ours else "Add")
        self.add_button.setEnabled(not others)
        self.remove_button.setEnabled(bool(removable))
        if others:
            self.status.setText(f"{_in_the_way(others)}\nMove it elsewhere to add this one.")
        elif ours:
            self.status.setText(
                "Already added. Writing it again makes it start this environment."
            )
        else:
            self.status.setText("")

    def add(self) -> None:
        """Write the launcher shown, again where am-seg wrote it before."""
        try:
            write(self.launcher, replace=True)
        except LauncherRefused as error:
            self.refresh()
            self.status.setText(str(error))
            return
        self.refresh()
        self.status.setText(f"Added. {WHERE_TO_FIND[self.launcher.platform]}")

    def remove(self) -> None:
        """Remove every launcher am-seg wrote here, the Desktop's included."""
        removed, left = remove(self.computer.launcher(desktop=True))
        self.refresh()
        said = "Removed." if removed else "Nothing to remove."
        self.status.setText(f"{said} {_in_the_way(left)}" if left else said)


def _in_the_way(paths: list[Path]) -> str:
    """Say which paths hold something am-seg did not write, and so will not touch."""
    listed = "\n".join(str(path) for path in paths)
    return f"Not written by am-seg, so left as it is:\n{listed}"
