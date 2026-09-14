"""The one dialog that writes proofread work.

Its own module because the button order is a decision rather than a default,
and because a save is the only thing in the window that can lose somebody's own
work. `docs/display.md` says why the row reads as it does.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
)

#: Names the save in every dialog and log line it produces.
SAVE_TITLE = "Save labels"


class SaveLabelsDialog(QDialog):
    """The one dialog that writes proofread work, laid out on purpose.

    Not a message box. Fusion reports the Windows button order wherever it runs,
    so a message box ends the row with *Cancel* -- the button that changes
    nothing sitting where the eye lands last, past both ways of saving. The
    order here is the shape of the choice: leave, save, save and finish. The
    last is the default, and therefore the accented one, because proofreading a
    label and not marking it is the exception.

    Overriding the style hint that decides the order would fix every dialog at
    once, and was tried: ``QProxyStyle.styleHint`` segfaults PySide6 while a
    message box is being built.
    """

    def __init__(
        self, parent, message: str, note: str, *, offer_corrected: bool
    ) -> None:
        """Ask how to save.

        Args:
            parent: The window the save belongs to.
            message: What the save would do.
            note: A second line, or empty. Says how to reach what is not on
                offer when *offer_corrected* is False.
            offer_corrected: Whether marking the reviewed set corrected is
                available, which follows the editing guard.
        """
        super().__init__(parent)
        self.setWindowTitle(SAVE_TITLE)
        self.mark_corrected = False

        layout = QVBoxLayout(self)
        said = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion).pixmap(40, 40)
        )
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        said.addWidget(icon)
        said.addSpacing(8)

        text = QLabel(message if not note else f"{message}\n\n{note}")
        text.setWordWrap(True)
        said.addWidget(text, stretch=1)
        layout.addLayout(said)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        buttons.addStretch()

        # "Save only" only means something against a second way of saving.
        save_only = QPushButton("Save only" if offer_corrected else "Save")
        save_only.clicked.connect(self.accept)
        buttons.addWidget(save_only)

        if offer_corrected:
            mark = QPushButton("Save and mark corrected")
            mark.clicked.connect(self._mark_and_accept)
            mark.setDefault(True)
            buttons.addWidget(mark)
        else:
            save_only.setDefault(True)
        layout.addLayout(buttons)

    def _mark_and_accept(self) -> None:
        """Accept, having chosen to mark the reviewed set corrected."""
        self.mark_corrected = True
        self.accept()


