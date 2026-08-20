"""The operations that act on whatever is ticked, under the state controls.

These are here rather than only on the command line because their input is the
selection on screen: the same ticks that decide what a state change moves decide
what gets predicted or converted. Running one from a terminal means naming that
selection again in filters, which is the part the window has already done. An
operation with no reason to be narrowed does not belong here at all; compressing
labels is on the Add data tab, over the whole collection.

Each button previews first and applies only after the preview is accepted, so
nothing here is a single click away from changing files.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .common import ALL_FILES_FILTER, choose_open_file
from .theme import BADGE_STYLES

#: Weights files, as PyTorch writes them.
MODEL_FILTER = "Model weights (*.pth *.pt)"


class OperationsPanel(QWidget):
    """Predict and convert the ticked selection: preview, then apply."""

    #: Emitted with the chosen weights file when the user asks to predict.
    predict_requested = Signal(str)
    #: Emitted when the user asks to convert the selection's predictions.
    convert_requested = Signal()
    #: Emitted when the user asks to repair inconsistent groups.
    repair_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._last_model_directory: Path | None = None
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

        # Before Convert, which is the order the two are used in: a prediction
        # has to exist before it can become a label.
        #
        # The model is a box rather than a dialog on the button, because it is
        # chosen once and used for many runs -- and because a dialog opening
        # before the window has checked that editing is even on asks for a file
        # it may be about to refuse to use.
        model_row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Model weights (.pth)")
        self.model_edit.setToolTip(
            "The trained model to predict with. Type a path or press Browse.\n"
            "Settings the weights do not record are read from a .yaml of the "
            "same name beside them."
        )
        model_row.addWidget(self.model_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Choose the model weights file.")
        self.browse_button.clicked.connect(self._choose_model)
        model_row.addWidget(self.browse_button)
        column.addLayout(model_row)

        self.predict_button = QPushButton("Predict labels...")
        self.predict_button.setToolTip(
            "Run the model above over the selected images.\n"
            "Only images of the modality it was trained on, one per scene, and "
            "never one this model has already predicted."
        )
        self.predict_button.clicked.connect(
            lambda: self.predict_requested.emit(self.model_edit.text().strip())
        )
        column.addWidget(self.predict_button)

        self.convert_button = QPushButton("Convert predictions...")
        self.convert_button.setToolTip(
            "Write a label from each selected image's prediction.\n"
            "Replacing an existing label has to be confirmed, and a proofread "
            "one is never replaced. Post-processed (clean/) predictions are "
            "not used."
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

        layout.addWidget(box)
        self.buttons = (
            self.predict_button, self.browse_button, self.convert_button,
        )

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

    def _choose_model(self) -> None:
        """Fill the model box from a file dialog. Browsing predicts nothing."""
        current = Path(self.model_edit.text()).parent if self.model_edit.text() else None
        path = choose_open_file(
            self, "Choose model weights", current or self._last_model_directory,
            filters=(MODEL_FILTER, ALL_FILES_FILTER),
        )
        if path is not None:
            self._last_model_directory = path.parent
            self.model_edit.setText(str(path))

    def model(self) -> str:
        """Return the model path as typed or chosen."""
        return self.model_edit.text().strip()

    def set_model(self, path: str) -> None:
        """Show *path* as the model to predict with, from a previous session."""
        self.model_edit.setText(path)
        if path:
            self._last_model_directory = Path(path).parent

    def set_prediction_available(self, available: bool) -> None:
        """Offer or withhold the predict button, saying why when withholding.

        The inference dependencies are a separate install, so a window running
        without them should say so on the button rather than at the end of a
        dialog the user opened expecting it to work. The full explanation is a
        terminal's job; this is a hover.
        """
        self.predict_button.setEnabled(available)
        self.model_edit.setEnabled(available)
        self.browse_button.setEnabled(available)
        if not available:
            self.predict_button.setToolTip(
                "The inference dependencies are not installed.\n"
                "Run scripts/setup_ml.sh to build an environment with them."
            )

    def set_busy(self, busy: bool) -> None:
        """Lock the buttons while an operation runs.

        One at a time: two batch operations over the same files would each have
        validated a set the other is changing underneath it.
        """
        for button in self.buttons:
            button.setEnabled(not busy)
