"""The modal panel a long operation runs behind.

**Nothing here processes events.** ``QProgressDialog`` calls ``processEvents()``
from ``setValue()`` when modal, which re-enters the event loop inside a
queued-signal handler: the worker's *finished* can then close the dialog while one
of its own methods is still on the stack. So progress is a plain widget update,
and the dialog closes exactly when the window closes it.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)


class BusyDialog(QDialog):
    """Says which operation is running, and how far it has got.

    Opens indeterminate, because most operations cannot say how much work they
    are about to do until they have started. The first :meth:`set_progress`
    turns it into a count; an operation that never reports simply keeps the
    moving bar.
    """

    #: Emitted when the user asks a stoppable operation to stop. The window
    #: owns the :class:`~...progress.Cancellation`; this only reports the press.
    stop_requested = Signal()

    def __init__(self, parent, description: str, *, stoppable: bool = False) -> None:
        """Build the dialog for *description*, with a Stop button if allowed.

        Args:
            parent: The window it blocks.
            description: Names the operation, in the title and the first line.
            stoppable: Whether the operation can be interrupted safely. An
                operation that cannot must not be offered a button that looks
                as though it could.
        """
        super().__init__(parent)
        self._stoppable = stoppable
        self.setWindowTitle(description)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # No close button: this dialog goes away when the work does, and a
        # cross that dismissed it would leave the operation running behind a
        # window that said it had stopped.
        self.setWindowFlags(
            (self.windowFlags() | Qt.WindowType.CustomizeWindowHint)
            & ~Qt.WindowType.WindowCloseButtonHint
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        layout = QVBoxLayout(self)
        self.message = QLabel(f"{description}...")
        layout.addWidget(self.message)

        self.bar = QProgressBar()
        # 0 to 0 is Qt's indeterminate bar: a moving block rather than a
        # position, which is the honest picture until the work says otherwise.
        self.bar.setRange(0, 0)
        self.bar.setMinimumWidth(320)
        layout.addWidget(self.bar)

        if stoppable:
            self.buttons = QDialogButtonBox()
            stop = self.buttons.addButton("Stop", QDialogButtonBox.ButtonRole.RejectRole)
            stop.setToolTip(
                "Stop after the file being written. What is already done is "
                "kept."
            )
            self.buttons.rejected.connect(self._stop)
            layout.addWidget(self.buttons)

    def set_progress(self, done: int, total: int) -> None:
        """Show *done* of *total*, turning the bar into a count on the first call."""
        if self.bar.maximum() != total:
            self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.message.setText(f"{done} of {total}...")

    def _stop(self) -> None:
        """Report the press, and stay open until the work actually stops."""
        self.message.setText("Stopping after the current file...")
        self.stop_requested.emit()

    def reject(self) -> None:
        """Refuse Escape, which would otherwise dismiss the dialog outright.

        On a stoppable operation the key asks it to stop, which is what a user
        pressing Escape means; on any other it does nothing, because there is
        nothing it could safely do.
        """
        if self._stoppable:
            self._stop()
