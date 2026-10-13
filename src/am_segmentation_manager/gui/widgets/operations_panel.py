"""The operations that act on whatever is ticked, under the state controls.

Their input is the selection on screen: the ticks that decide what a state change
moves decide what is predicted or converted, so an operation with no reason to be
narrowed does not belong here -- compressing labels is on the Add data tab. Each
button that writes previews first and applies only once that is accepted; the one
that only reads has no preview, what it would find being what it costs to find.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
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

from ...layout import DEFAULT_PREDICTION_ORIGIN, PREDICTION_ORIGINS, PredictionOrigin
from ...operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
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

#: Between a device and the description shown after it in the box, so what is
#: typed or chosen can be read back as the device alone.
DESCRIPTION_SEPARATOR = " · "

#: What the device box says on hover, before anything is known of the machine.
DEVICE_HINT = (
    "'auto' takes the GPU with the most free memory, or the alternative best device. GPU numbers "
    "as set by nvidia-smi command."
)

#: The hint where torch cannot use the machine's GPUs, which is then what a
#: person choosing a device needs to know. The reason is announced in full and
#: logged, being longer than a hover is read.
UNUSABLE_GPU_HINT = (
    "Torch cannot use this machine's NVIDIA GPUs, so 'auto' runs on the CPU."
)

#: What the box says on hover while its operations apply to the series on screen.
OPERATIONS_HINT = (
    "Operations act on the current selection: ticked projects or lamellae, frames picked on "
    "the timeline, or the frame on screen."
)


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
        box.setToolTip(OPERATIONS_HINT)
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
            "Path to model weights file used to predict with."
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
        device_row.addWidget(caption_for("Device", self.device_box, DEVICE_HINT))
        device_row.addWidget(self.device_box, stretch=1)
        column.addLayout(device_row)

        self.predict_button = QPushButton("Predict labels...")
        self.predict_button.setToolTip(
            "Run the model above over the selected images.\n"
            "Runs on one modality at a time: set Apply to to SEM or FIB."
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
                "Use predictions made by AM-seg.",
            ),
            (
                "microscope", "Microscope",
                "Use predictions made by Adaptive Polishing on the microscope.",
            ),
        ])
        self.origin_control.set_current(DEFAULT_PREDICTION_ORIGIN)
        self.origin_control.chosen.connect(self.origin_changed.emit)
        origin_row = QHBoxLayout()
        origin_row.addWidget(
            caption_for(
                "From",
                self.origin_control,
                "Select which prediction to convert when an image has both.",
            )
        )
        origin_row.addWidget(self.origin_control)
        origin_row.addStretch(1)
        column.addLayout(origin_row)

        self.convert_button = QPushButton("Convert predictions...")
        self.convert_button.setToolTip(
            "Convert (resize) predictions and write them as labels. Labels marked as corrected "
            "are never overwritten."
        )
        self.convert_button.clicked.connect(self.convert_requested.emit)
        column.addWidget(self.convert_button)

        # Directly under Convert, which is what it usually undoes: trying a
        # newer model over proofread labels is a decision rather than a
        # commitment, and the two belong within a glance of each other.
        self.revert_button = QPushButton("Revert...")
        self.revert_button.setToolTip(
            "Undo the last label change from a conversion or merge."
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
            "Labels that would be created or replaced on the current selection."
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
                "Set the size in pixels below which specks will be reported.",
            )
        )
        speck_row.addWidget(self.speck_size)
        speck_row.addStretch(1)
        column.addLayout(speck_row)

        self.specks_button = QPushButton("Find specks...")
        self.specks_button.setToolTip(
            "Reports which images of the selection have \"specks\", which are likely incorrect "
            "annotations."
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
        chosen = self.origin_control.current()
        for origin in PREDICTION_ORIGINS:
            if chosen == origin:
                return origin
        return DEFAULT_PREDICTION_ORIGIN

    def model(self) -> str:
        """Return the model path as typed or chosen."""
        return self.model_edit.text().strip()

    def device(self) -> str:
        """Return the device to predict on, as chosen or typed, without its description."""
        typed = self.device_box.currentText().split(DESCRIPTION_SEPARATOR, 1)[0]
        return typed.strip() or BASE_DEVICES[0]

    def set_device(self, device: str) -> None:
        """Show *device* as the one to predict on, from a previous session."""
        if not device:
            return
        for index in range(self.device_box.count()):
            if self.device_box.itemData(index) == device:
                self.device_box.setCurrentIndex(index)
                return
        self.device_box.setCurrentText(device)

    def show_devices(
        self,
        devices: Iterable[str],
        descriptions: Mapping[str, str] | None = None,
        problem: str = "",
    ) -> None:
        """Offer the devices torch reports, described, keeping what is currently chosen.

        The base entries stay: "auto" is not a torch device and so is never in
        the list, and a machine whose GPU has gone missing should still be able
        to choose the CPU. Each item shows its description where there is one
        and carries the bare device, which is what :meth:`device` reads back.
        A *problem* -- why torch cannot use the machine's GPUs -- replaces the hint.
        """
        descriptions = descriptions or {}
        chosen = self.device()
        offered: list[str] = list(BASE_DEVICES[:1]) + [
            device for device in devices if device not in BASE_DEVICES[:1]
        ]
        for fallback in BASE_DEVICES[1:]:
            if fallback not in offered:
                offered.append(fallback)
        self.device_box.setToolTip(UNUSABLE_GPU_HINT if problem else DEVICE_HINT)
        shown = [descriptions.get(device, device) for device in offered]
        if shown == [self.device_box.itemText(i) for i in range(self.device_box.count())]:
            return
        self.device_box.clear()
        for device, text in zip(offered, shown, strict=True):
            self.device_box.addItem(text, device)
        self.set_device(chosen)

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

        The inference dependencies are an optional extra, so a window running
        without them should say so on the button rather than at the end of a
        dialog the user opened expecting it to work. The full explanation is a
        terminal's job; this is a hover.
        """
        self._prediction_available = available
        if not available:
            self.predict_button.setToolTip(
                "The inference dependencies are not installed.\n"
                "Install them with: uv sync --extra gui --extra ml"
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
            OPERATIONS_HINT
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

        One place rather than each setter undoing the others, so re-enabling
        after an operation cannot offer the Predict button on a machine with no
        inference stack, whether it is installed being asked only once.
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
