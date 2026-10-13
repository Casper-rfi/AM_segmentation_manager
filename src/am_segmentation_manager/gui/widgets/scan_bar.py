"""The folder the window reads, chosen from the tab row so every tab shows it.

A scan serves all four tabs, so the path, Browse and Scan sit in the tab bar's
corner rather than inside one of them. This widget only asks for a scan;
reading the folder is the window's to do.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from .paths import choose_directory

#: How wide the path field is, in logical pixels. It shares the tab row with the
#: tabs and the mark, which together must fit the smallest screen supported; a
#: path typed or chosen shows its end, the folder's own name.
SCAN_FIELD_WIDTH = 320

#: How narrow it may be squeezed. A tab row too narrow for everything in it
#: takes the room from the tabs, which Qt shrinks to slivers nobody can hit,
#: while the corner keeps its width -- so the field gives way first. Still wide
#: enough to read a folder name in, which is what the field is for.
SCAN_FIELD_MINIMUM = 120

#: What the field says about itself, under what the folder held once it is read.
SCAN_HINT = (
    "Folder searched recursively for Autolamella projects "
    "(any folder containing an experiment.yaml or AM_metadata.yaml)"
)


class ScanBar(QWidget):
    """The path of the folder being read, and the two ways to read one."""

    #: Emitted with the directory to scan.
    scan_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder to scan for Autolamella projects")
        self.path_edit.setToolTip(SCAN_HINT)
        # Preferred rather than fixed: see SCAN_FIELD_MINIMUM.
        self.path_edit.setMinimumWidth(SCAN_FIELD_MINIMUM)
        self.path_edit.setMaximumWidth(SCAN_FIELD_WIDTH)
        self.path_edit.setSizePolicy(
            QSizePolicy.Policy.Preferred, self.path_edit.sizePolicy().verticalPolicy()
        )
        # Clicked into, never tabbed into: a focused line edit claims Left and
        # Right, which step through frames on the tabs beside it.
        self.path_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.path_edit.returnPressed.connect(self._scan_typed)
        row.addWidget(self.path_edit)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Choose a folder to scan")
        self.browse_button.clicked.connect(self.browse)
        row.addWidget(self.browse_button)

        self.scan_button = QPushButton("Scan")
        self.scan_button.setAutoDefault(False)
        self.scan_button.clicked.connect(self._scan_typed)
        row.addWidget(self.scan_button)
        self.show_as_next_step(True)

    def browse(self) -> None:
        """Ask for a folder to scan, and request a scan of it if one is chosen.

        Public because the File menu offers the same thing: two ways in, one
        behaviour, rather than the menu opening a dialog of its own.
        """
        directory = choose_directory(
            self, "Choose a folder to scan", self.path_edit.text() or None
        )
        if directory is not None:
            self.set_path(str(directory))
            self.scan_requested.emit(str(directory))

    def show_as_next_step(self, waiting: bool) -> None:
        """Draw Scan as the thing to press next, or as one button among several.

        Nothing in the window works until a folder has been read, and until
        then the button that reads one looked like every other button. Drawn in
        the accent through Qt's *default* state, which the theme already tints:
        a second way of marking one button would be a second thing to keep in
        step with the accent. Enter is not bound to it -- that belongs to
        dialogs, and the path field answers Enter itself.
        """
        self.scan_button.setDefault(waiting)
        self.scan_button.setToolTip(
            "Read this folder and list what it holds" if waiting else
            "Re-read the folder from disk. Use after changing files outside AM-seg."
        )

    def set_path(self, path: str) -> None:
        """Show *path* as the folder being read."""
        self.path_edit.setText(path)

    def show_summary(self, summary: str) -> None:
        """Say on hover which folder was read and what it held, as *summary*."""
        self.path_edit.setToolTip(f"{self.path_edit.text()}\n{summary}\n\n{SCAN_HINT}")

    def _scan_typed(self) -> None:
        self.scan_requested.emit(self.path_edit.text())
