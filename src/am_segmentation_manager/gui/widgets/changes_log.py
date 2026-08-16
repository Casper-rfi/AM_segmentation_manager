"""What has been moved this session, kept where it can be read.

Across the foot of the window rather than in a column beside the images. Two
landscape images sit side by side above it, so width is the scarce resource
here and height is not -- a log in the right-hand column was taking about
250 px of the width the images are downscaled to fit.
"""
from __future__ import annotations

from qtpy.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout


class ChangesLog(QGroupBox):
    """A read-only record of the state changes made since the window opened."""

    #: Lines kept before the oldest are dropped. Long enough to cover a
    #: session's curating, short enough that the widget never grows unbounded.
    MAX_LINES = 500

    def __init__(self) -> None:
        super().__init__("Changes")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(self.MAX_LINES)
        self.view.setToolTip(
            "Every state change made since the window opened. Nothing here is "
            "written to disk -- the log of record is what the files themselves say."
        )
        layout.addWidget(self.view)

    def append(self, message: str) -> None:
        """Record a change, so what happened stays visible."""
        self.view.appendPlainText(message)

    def text(self) -> str:
        """Return everything recorded, for tests and for copying out."""
        return self.view.toPlainText()
