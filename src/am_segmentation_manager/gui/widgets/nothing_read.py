"""What the image area shows before any folder has been read.

A surface onto the two ways to read one, in the middle of the window rather than
in the corner: nothing here decides anything, and the window connects both
signals to the same handlers the tab row's Browse and Scan use.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .theme import HEADING, MUTED

#: What the page says above the buttons.
HEADLINE = "Nothing read yet"

#: What it says under them where a folder is remembered, and where none is.
REMEMBERED_HINT = "Browse and Scan, at the top right, do the same thing."
FIRST_RUN_HINT = (
    "Choose the folder your Autolamella projects are in. "
    "Browse, at the top right, does the same thing."
)


class NothingReadPage(QWidget):
    """The empty image area, offering the folder a scan would read.

    Shown while nothing has been read, which is the one state in which no part
    of the window does anything. The controls are here because this is where
    somebody looks when a window appears empty, and hunting the tab row for a
    button drawn like every other button is what they did instead.
    """

    #: Emitted with the folder to read.
    scan_requested = Signal(str)
    #: Emitted when another folder is wanted, which opens the same chooser
    #: Browse does.
    browse_requested = Signal()

    def __init__(self) -> None:
        """Build the page, empty until :meth:`show_folder` fills it."""
        super().__init__()
        self._folder = ""

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        self.headline = QLabel(HEADLINE)
        self.headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.headline.setStyleSheet(f"color: {HEADING}; font-weight: 600;")
        layout.addWidget(self.headline)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignmentFlag.AlignCenter)
        buttons.setSpacing(8)
        self.scan_button = QPushButton()
        # Tinted by the theme's rule for a default button, as the tab row's
        # Scan is: one button drawn as the thing to press, in two places that
        # do the one thing.
        self.scan_button.setDefault(True)
        self.scan_button.setAutoDefault(False)
        self.scan_button.clicked.connect(self._scan)
        buttons.addWidget(self.scan_button)

        self.browse_button = QPushButton("Choose another folder...")
        self.browse_button.setAutoDefault(False)
        self.browse_button.clicked.connect(self.browse_requested)
        buttons.addWidget(self.browse_button)
        layout.addLayout(buttons)

        #: The folder a scan would read, under the button that would read it:
        #: pressing it is then a confirmation rather than a guess at which
        #: folder the window remembers.
        self.path = QLabel()
        self.path.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path.setWordWrap(True)
        self.path.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.path)

        self.hint = QLabel()
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.hint)

        self.show_folder("")

    def show_folder(self, folder: str) -> None:
        """Offer to read *folder*, or to choose one where it is empty.

        The folder is named on the button rather than only beneath it, so what
        pressing it would read is legible without reading a path.
        """
        self._folder = folder
        remembered = bool(folder)
        self.scan_button.setText(
            f"Scan {Path(folder).name}" if remembered else "Choose a folder..."
        )
        self.browse_button.setVisible(remembered)
        self.path.setText(folder)
        self.path.setVisible(remembered)
        self.hint.setText(REMEMBERED_HINT if remembered else FIRST_RUN_HINT)

    def _scan(self) -> None:
        """Ask for the folder to be read, or for one to be chosen."""
        if self._folder:
            self.scan_requested.emit(self._folder)
        else:
            self.browse_requested.emit()
