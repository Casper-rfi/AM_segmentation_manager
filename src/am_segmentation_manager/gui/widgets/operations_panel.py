"""The operations that act on whatever is ticked, under the state controls.

These are here rather than only on the command line because their input is the
selection on screen: the same ticks that decide what a state change moves decide
what gets converted, recompressed or repaired. Running one from a terminal means
naming that selection again in filters, which is the part the window has already
done.

Each button previews first and applies only after the preview is accepted, so
nothing here is a single click away from changing files.
"""
from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import BADGE_STYLES


class OperationsPanel(QWidget):
    """Convert predictions, compress labels, repair state: preview then apply."""

    #: Emitted when the user asks to convert the selection's predictions.
    convert_requested = Signal()
    #: Emitted when the user asks to recompress the selection's labels.
    compress_requested = Signal()
    #: Emitted when the user asks to repair inconsistent groups.
    repair_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # What these act on goes in the title rather than on a line of its own:
        # the state panel directly above already says it, and repeating the
        # sentence in a narrow column reads as two different selections.
        self.box = box = QGroupBox("Operations")
        box.setToolTip(
            "These act on the same selection the state buttons do: the ticked "
            "projects or lamellae, the frames picked out on the timeline, or "
            "the frame on screen."
        )
        column = QVBoxLayout(box)

        self.convert_button = QPushButton("Convert predictions...")
        self.convert_button.setToolTip(
            "Write a label from each selected image's prediction.\n"
            "Replacing a label that is already there is offered and has to be "
            "confirmed; a proofread label in corrected/ is never replaced.\n"
            "Post-processed (clean/) predictions are never used: their "
            "intensity values no longer match the segmentation classes."
        )
        self.convert_button.clicked.connect(self.convert_requested.emit)
        column.addWidget(self.convert_button)

        # Above the button's own report rather than inside it: whether there is
        # anything here to convert is a question about the data, and a dialog
        # you have to open to find out the answer was "nothing" is the version
        # of this that sends you looking for the predictions by hand.
        self.predictions = QLabel("")
        self.predictions.setWordWrap(True)
        self.predictions.setStyleSheet(BADGE_STYLES["info"])
        self.predictions.setToolTip(
            "What converting this selection would find, counted from the last "
            "scan. Images sharing a scene share one label, so the preview can "
            "write fewer files than there are images ready."
        )
        column.addWidget(self.predictions)

        self.compress_button = QPushButton("Compress labels...")
        self.compress_button.setToolTip(
            "Re-encode the selection's labels so they take less disk space. "
            "Pixel values and dimensions are unchanged."
        )
        self.compress_button.clicked.connect(self.compress_requested.emit)
        column.addWidget(self.compress_button)

        self.repair_button = QPushButton("Repair state...")
        self.repair_button.setToolTip(
            "Move images to agree with the label they share, for the frames the "
            "timeline marks in amber. The label is never moved."
        )
        self.repair_button.clicked.connect(self.repair_requested.emit)
        column.addWidget(self.repair_button)

        layout.addWidget(box)
        self.buttons = (self.convert_button, self.compress_button, self.repair_button)

    def show_selection(self, description: str) -> None:
        """Say what the buttons would act on."""
        self.box.setTitle(f"Operations on {description}")

    def show_predictions(self, summary: str) -> None:
        """Say what converting the selection would find.

        An empty summary hides the line: with nothing selected there is no
        count to report, and "0 of 0" reads as a fault rather than as silence.
        """
        self.predictions.setText(summary)
        self.predictions.setVisible(bool(summary))

    def set_busy(self, busy: bool) -> None:
        """Lock the buttons while an operation runs.

        One at a time: two batch operations over the same files would each have
        validated a set the other is changing underneath it.
        """
        for button in self.buttons:
            button.setEnabled(not busy)
