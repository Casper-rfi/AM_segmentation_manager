"""The operations that act on whatever is ticked, under the state controls.

These are here rather than only on the command line because their input is the
selection on screen: the same ticks that decide what a state change moves decide
what gets predicted or converted. Running one from a terminal means naming that
selection again in filters, which is the part the window has already done. An
operation with no reason to be narrowed does not belong here at all; compressing
labels is on the Add data tab, over the whole collection.

Each button that writes previews first and applies only after the preview is
accepted, so nothing here is a single click away from changing files. The one
that only reads has no preview: what it would find is what it costs to find.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...layout import DEFAULT_PREDICTION_ORIGIN, PredictionOrigin
from ...operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
    PREDICTION_PIXEL_BLOCK,
)
from .common import caption_for, horizontal_rule
from .controls import SegmentedControl
from .paths import ALL_FILES_FILTER, choose_open_file
from .theme import BADGE_STYLES

#: Weights files, as PyTorch writes them.
MODEL_FILTER = "Model weights (*.pth *.pt)"

#: What the device box offers before torch has been asked. "auto" takes the
#: most capable device the machine has, and "cpu" is the one every machine has;
#: what lies between them differs per machine -- several CUDA devices on a
#: workstation, an MPS one on a Mac -- so the rest is filled in from torch once
#: asking is free. The box is editable because
#: :func:`~am_segmentation_manager.ml.environment.resolve_device` passes a named
#: device through unexamined, so a device this package has never seen can be
#: typed rather than waiting for support.
BASE_DEVICES = ("auto", "cpu")


class OperationsPanel(QWidget):
    """Predict and convert the ticked selection: preview, then apply."""

    #: Emitted with the chosen weights file when the user asks to predict.
    predict_requested = Signal(str)
    #: Emitted when the device box is opened, so the window can fill it in if
    #: torch is loaded. The widget does not ask torch itself: importing it to
    #: populate a dropdown is what this whole arrangement exists to avoid.
    devices_requested = Signal()
    #: Emitted when the user asks to convert the selection's predictions.
    convert_requested = Signal()
    #: Emitted with the prediction source chosen for converting.
    origin_changed = Signal(str)
    #: Emitted with the largest component to report when the user asks which
    #: of the selection's labels hold specks.
    specks_requested = Signal(int)
    #: Emitted when the user asks to put the selection's labels back.
    revert_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._last_model_directory: Path | None = None
        #: The three independent reasons a button here may be locked. Kept
        #: apart and applied together by :meth:`_apply_lock`, so that lifting
        #: one cannot offer a button another still forbids.
        self._prediction_available = True
        self._series_is_editable = True
        self._busy = False
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
            "Kept per series and beam, so the box follows the view.\n"
            "Settings the weights do not record come from a .yaml beside them."
        )
        model_row.addWidget(self.model_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Choose the model weights file.")
        self.browse_button.clicked.connect(self._choose_model)
        model_row.addWidget(self.browse_button)
        column.addLayout(model_row)

        device_row = QHBoxLayout()
        self.device_box = _DeviceCombo(self.devices_requested)
        self.device_box.setEditable(True)
        self.device_box.addItems(BASE_DEVICES)
        device_row.addWidget(
            caption_for(
                "Device",
                self.device_box,
                "Where the model runs. 'auto' takes the most capable device "
                "this machine has -- a GPU where there is one, otherwise the "
                "CPU. Any torch device can be typed, such as cuda:1 on a "
                "machine with several.",
            )
        )
        device_row.addWidget(self.device_box, stretch=1)
        column.addLayout(device_row)

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

        # Above Convert, as the model is above Predict: what a button acts with
        # sits over it. Segments rather than a switch, this asking which of two
        # rather than whether one is on; the preview says what each label used.
        self.origin_control = SegmentedControl([
            (
                "am-seg", "am-seg",
                "Convert the predictions Predict labels made.\n"
                "An image without one uses the microscope's.",
            ),
            (
                "microscope", "Microscope",
                "Convert the predictions the microscope's pipeline wrote.\n"
                "An image without one uses am-seg's.",
            ),
        ])
        self.origin_control.set_current(DEFAULT_PREDICTION_ORIGIN)
        self.origin_control.chosen.connect(self.origin_changed.emit)
        origin_row = QHBoxLayout()
        origin_row.addWidget(
            caption_for(
                "From",
                self.origin_control,
                "Which prediction Convert takes where an image has both. The "
                "preview says which each label comes from.",
            )
        )
        origin_row.addWidget(self.origin_control)
        origin_row.addStretch(1)
        column.addLayout(origin_row)

        self.convert_button = QPushButton("Convert predictions...")
        self.convert_button.setToolTip(
            "Write a label from each selected image's prediction.\n"
            "Replacing an existing label has to be confirmed, and a corrected "
            "one is never replaced. Post-processed (clean/) predictions are "
            "not used."
        )
        self.convert_button.clicked.connect(self.convert_requested.emit)
        column.addWidget(self.convert_button)

        # Directly under Convert, which is what it usually undoes: trying a
        # newer model over proofread labels is a decision rather than a
        # commitment, and the two belong within a glance of each other.
        self.revert_button = QPushButton("Revert...")
        self.revert_button.setToolTip(
            "Put the selection's labels back to what they held before the "
            "last conversion or merge, or to a savepoint.\n"
            "A label with no kept version, or edited since, is reported and "
            "left alone."
        )
        self.revert_button.clicked.connect(self.revert_requested.emit)
        column.addWidget(self.revert_button)

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

        # Below the two that write, and set apart from them: this one only
        # reads. It is here rather than in Add data because what it answers is
        # a curation question -- which datasets are worth going back to -- and
        # because scoping it to a few ticked projects is what makes it take
        # seconds instead of minutes.
        column.addWidget(horizontal_rule())
        speck_row = QHBoxLayout()
        self.speck_size = QSpinBox()
        self.speck_size.setRange(MIN_SPECK_PIXELS, MAX_SPECK_PIXELS)
        self.speck_size.setValue(DEFAULT_SPECK_PIXELS)
        # The unit in the box itself, not only in the tooltip: a bare number in
        # a spin box is one you have to hover to understand, and this one is
        # read every time before it is pressed.
        self.speck_size.setSuffix(" px")
        # One tooltip, set through `caption_for`, which puts it on the caption
        # and the box alike -- so a second one set here would be overwritten
        # and the unit would be nowhere.
        speck_row.addWidget(
            caption_for(
                "At most",
                self.speck_size,
                "Largest component reported, in pixels. Set here rather than "
                "on the Label tab, so it is visible where it is pressed. "
                f"{PREDICTION_PIXEL_BLOCK} is one pixel of a prediction: "
                "anything smaller was painted by hand.",
            )
        )
        speck_row.addWidget(self.speck_size)
        speck_row.addStretch(1)
        column.addLayout(speck_row)

        self.specks_button = QPushButton("Find specks...")
        self.specks_button.setToolTip(
            "Report which projects hold labels with components likely to be "
            "too small to be real -- some are genuine, a small hole often "
            "is.\nReads every label in the selection: slow, and changes nothing."
        )
        self.specks_button.clicked.connect(
            lambda: self.specks_requested.emit(self.speck_size.value())
        )
        column.addWidget(self.specks_button)

        layout.addWidget(box)
        self.buttons = (
            self.predict_button, self.browse_button, self.convert_button,
            self.specks_button, self.revert_button,
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

    def prediction_origin(self) -> PredictionOrigin:
        """Return the source Convert takes where an image has both."""
        return self.origin_control.current() or DEFAULT_PREDICTION_ORIGIN

    def model(self) -> str:
        """Return the model path as typed or chosen."""
        return self.model_edit.text().strip()

    def device(self) -> str:
        """Return the device to predict on, as chosen or typed."""
        return self.device_box.currentText().strip() or BASE_DEVICES[0]

    def set_device(self, device: str) -> None:
        """Show *device* as the one to predict on, from a previous session."""
        if device:
            self.device_box.setCurrentText(device)

    def show_devices(self, devices) -> None:
        """Offer the devices torch reports, keeping what is currently chosen.

        Called when the box is opened and again after a run, by which time
        torch is loaded and the real list is free to ask for. The base entries
        stay: "auto" is not a torch device and so is never in the list, and a
        machine whose GPU has gone missing should still be able to choose the
        CPU.
        """
        chosen = self.device_box.currentText()
        offered = list(BASE_DEVICES[:1]) + [
            device for device in devices if device not in BASE_DEVICES[:1]
        ]
        for fallback in BASE_DEVICES[1:]:
            if fallback not in offered:
                offered.append(fallback)
        if offered == [self.device_box.itemText(i) for i in range(self.device_box.count())]:
            return
        self.device_box.clear()
        self.device_box.addItems(offered)
        self.device_box.setCurrentText(chosen)

    def set_model(self, path: str) -> None:
        """Show *path* as the model to predict with, from a previous session."""
        self.model_edit.setText(path)
        if path:
            self._last_model_directory = Path(path).parent

    def apply_defaults(self, values) -> None:
        """Open the speck size at what the settings say. See `ImagePanel`."""
        self.speck_size.setValue(int(values["speck_pixels"]))

    def set_prediction_available(self, available: bool) -> None:
        """Offer or withhold the predict button, saying why when withholding.

        The inference dependencies are a separate install, so a window running
        without them should say so on the button rather than at the end of a
        dialog the user opened expecting it to work. The full explanation is a
        terminal's job; this is a hover.
        """
        self._prediction_available = available
        if not available:
            self.predict_button.setToolTip(
                "The inference dependencies are not installed.\n"
                "Run scripts/setup_ml.sh to build an environment with them."
            )
        self._apply_lock()

    def set_series_editable(self, editable: bool) -> None:
        """Offer or withhold every button for the series on screen.

        All of them, not only the ones that write: each of these reads or
        writes a label, and the series this withholds for has none. A button
        that would answer "nothing selected" over a full timeline is worse than
        one that is plainly unavailable and says why.
        """
        self._series_is_editable = bool(editable)
        self.box.setToolTip(
            ""
            if editable
            else "These images have no labels, so none of these operations applies."
        )
        self._apply_lock()

    def set_busy(self, busy: bool) -> None:
        """Lock the buttons while an operation runs.

        One at a time: two batch operations over the same files would each have
        validated a set the other is changing underneath it.
        """
        self._busy = bool(busy)
        self._apply_lock()

    def _apply_lock(self) -> None:
        """Enable each control only where every reason allows it.

        One place rather than each setter undoing the others: re-enabling
        everything after an operation used to offer the Predict button on a
        machine with no inference stack, whether it is installed being asked
        once and never again.
        """
        usable = self._series_is_editable and not self._busy
        for button in self.buttons:
            button.setEnabled(usable)
        self.device_box.setEnabled(usable)
        self.origin_control.setEnabled(usable)
        self.model_edit.setEnabled(self._prediction_available and usable)
        if not self._prediction_available:
            self.predict_button.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.device_box.setEnabled(False)


class _DeviceCombo(QComboBox):
    """A combo that asks to be filled in the moment it is opened.

    Which devices torch can see costs an import to answer, and the window
    refuses to spend that on building a dropdown. Asking when the box is opened
    puts the cost where the user has just expressed interest -- and by then
    pressing Predict once has usually loaded torch already, so there is no cost
    left to pay.
    """

    def __init__(self, asked: Signal) -> None:
        super().__init__()
        self._asked = asked

    def showPopup(self) -> None:  # noqa: N802 - Qt's spelling
        self._asked.emit()
        super().showPopup()
