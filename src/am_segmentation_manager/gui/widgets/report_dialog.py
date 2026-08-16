"""A scrollable, monospace report, shown before and after an operation runs.

Every operation the window offers produces a list of lines -- files it would
write, labels it would re-encode, groups it would repair -- and a message box
is the wrong shape for them: it wraps, it proportional-spaces paths whose
alignment is the point, and it grows off the screen rather than scrolling.

The same dialog does both jobs. With an action label it asks; without one it
reports. Keeping them one widget is what makes a preview and its outcome read
as the same thing seen twice.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
    QVBoxLayout,
)


class ReportDialog(QDialog):
    """Shows lines of report text, optionally asking whether to go ahead."""

    def __init__(
        self,
        parent,
        title: str,
        lines: Sequence[str],
        action_label: str = "",
    ) -> None:
        """Build the dialog.

        Args:
            parent: The window it belongs to.
            title: The window title, naming the operation.
            lines: The report, one line each. Shown verbatim.
            action_label: Text for the button that goes ahead, e.g. ``"Apply"``.
                Empty makes this a report with nothing to accept.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 460)

        layout = QVBoxLayout(self)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        # Paths line up only in a fixed-width font, and a report is read by
        # scanning down the column where the paths differ.
        self.view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setPlainText("\n".join(lines))
        layout.addWidget(self.view)

        if action_label:
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.button(QDialogButtonBox.Ok).setText(action_label)
            # Cancel is the default: these dialogs open in front of a keyboard
            # gesture, and Return should not be able to apply one by accident.
            buttons.button(QDialogButtonBox.Cancel).setDefault(True)
        else:
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

    def text(self) -> str:
        """Return what the dialog is showing."""
        return self.view.toPlainText()
