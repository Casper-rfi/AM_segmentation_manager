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
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..operations_controller import PREDICTION_KINDS

#: How the empty prediction kind -- "whatever this record has" -- reads in a
#: menu, where an empty row would look like a mistake.
ANY_KIND = "any"


class OperationsPanel(QWidget):
    """Convert predictions, compress labels, repair state: preview then apply."""

    #: Emitted with the prediction kind to convert from ("" means any).
    convert_requested = Signal(str)
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

        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems([kind or ANY_KIND for kind in PREDICTION_KINDS])
        self.kind.setToolTip(
            "Which predictions/ subfolder to convert from. 'any' takes whatever "
            "prediction a record has."
        )
        self.convert_button = QPushButton("Convert predictions...")
        self.convert_button.setToolTip(
            "Write a label from each selected image's prediction. Images that "
            "already have a label are never touched."
        )
        self.convert_button.clicked.connect(
            lambda: self.convert_requested.emit(self.prediction_kind())
        )
        row.addWidget(self.convert_button, stretch=1)
        row.addWidget(self.kind)
        column.addLayout(row)

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

    def prediction_kind(self) -> str:
        """Return the chosen prediction kind, with "any" back as an empty string."""
        text = self.kind.currentText()
        return "" if text == ANY_KIND else text

    def show_selection(self, description: str) -> None:
        """Say what the buttons would act on."""
        self.box.setTitle(f"Operations on {description}")

    def set_busy(self, busy: bool) -> None:
        """Lock the buttons while an operation runs.

        One at a time: two batch operations over the same files would each have
        validated a set the other is changing underneath it.
        """
        for button in self.buttons:
            button.setEnabled(not busy)
