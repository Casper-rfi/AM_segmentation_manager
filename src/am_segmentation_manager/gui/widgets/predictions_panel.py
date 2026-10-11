"""The Train tab's Predictions view: the ticked runs' models on a few validation images, in a grid.

A view of :class:`~am_segmentation_manager.gui.preview_controller.PreviewController`
and nothing more. A row is an image and a column a run; every pane zooms and pans
with every other. **Nothing here writes**: the predictions are the controller's, in
memory.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...progress import Cancellation
from ...training_preview import MOST_PREVIEW_IMAGES, PREVIEW_FILTERS
from ..preview_controller import PreviewController
from .controls import SLIDER_WIDTH, ToggleSwitch, ValueSlider, WrappingLabel
from .image_view import ImageView, ViewLink
from .theme import MUTED, WARNING
from .versions_bar import COMPARE_KEY, COMPARE_KEY_CODE

logger = logging.getLogger(__name__)

#: What the busy dialog says while the models predict.
PREDICTING = "Predicting to compare"

#: What stands for no choice in each of the lists an image is chosen by.
ANY = "any"

Work = Callable[[Callable[[int, int], None] | None, Cancellation | None], object]


class PredictionsPanel(QWidget):
    """Choose a few validation images, predict them with each run looked at, and compare.

    Attributes:
        run_stoppable: Called as ``run(work, done, description)`` to do *work*
            off the interface thread, given a progress callback and a
            cancellation, and hand its result to *done*. The window's, behind
            its busy dialog; inline where nothing replaces it.
        device: Returns the device to predict on; the Train tab's.
    """

    def __init__(self, controller: PreviewController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.run_stoppable: Callable[[Work, Callable[[object], None], str], None]
        self.run_stoppable = lambda work, done, _description: done(work(None, None))
        self.device: Callable[[], str] = lambda: "auto"
        self.views: dict[tuple[int, int], ImageView] = {}
        self.notes: dict[tuple[int, int], QLabel] = {}
        self._link = ViewLink()
        self._proofread = False

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        choosing = QHBoxLayout()
        self.filter_boxes: dict[str, QComboBox] = {}
        for caption, name in PREVIEW_FILTERS:
            label = _muted(caption)
            box = QComboBox()
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            hint = (
                f"Draw only images whose {caption.lower()} is the one chosen, as the "
                f"training export recorded it."
            )
            label.setToolTip(hint)
            box.setToolTip(hint)
            box.activated.connect(lambda _index, name=name: self._filter_chosen(name))
            self.filter_boxes[name] = box
            choosing.addWidget(label)
            choosing.addWidget(box)
        choosing.addWidget(_muted("Images"))
        self.count_box = QSpinBox()
        self.count_box.setRange(1, MOST_PREVIEW_IMAGES)
        self.count_box.setValue(controller.count)
        self.count_box.setToolTip("How many images to draw, a row each.")
        self.count_box.valueChanged.connect(self._count_changed)
        choosing.addWidget(self.count_box)
        self.pick_button = QPushButton("Pick images")
        self.pick_button.setToolTip(
            "Draw other validation images at random and predict them with each ticked "
            "run's marked or best epoch.\nNothing is written to the collection."
        )
        self.pick_button.clicked.connect(self.pick)
        choosing.addWidget(self.pick_button)
        self.predict_button = QPushButton("Predict the rest")
        self.predict_button.setToolTip(
            "Predict what is not yet drawn: for a run ticked since, or after stopping."
        )
        self.predict_button.clicked.connect(self.predict)
        choosing.addWidget(self.predict_button)
        choosing.addStretch(1)
        column.addLayout(choosing)

        drawing = QHBoxLayout()
        self.label_check = ToggleSwitch("Label")
        self.label_check.setChecked(controller.label_shown)
        self.label_check.setToolTip("Draw the predicted classes over each image, or the image alone.")
        self.label_check.toggled.connect(self._label_toggled)
        drawing.addWidget(self.label_check)
        opacity = _muted("Opacity")
        self.alpha_slider = ValueSlider()
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(round(controller.alpha * 100))
        self.alpha_slider.setFixedWidth(SLIDER_WIDTH)
        self.alpha_slider.setToolTip("How strongly the classes are drawn.")
        opacity.setToolTip(self.alpha_slider.toolTip())
        self.alpha_slider.valueChanged.connect(self._alpha_moved)
        drawing.addWidget(opacity)
        drawing.addWidget(self.alpha_slider)
        self.proofread_button = QPushButton(f"Hold to show the label ({COMPARE_KEY})")
        self.proofread_button.setCheckable(True)
        self.proofread_button.setToolTip(
            "Show each image's own label in place of every prediction, so the two can "
            f"be flicked between. Hold {COMPARE_KEY} to do the same from a picture."
        )
        self.proofread_button.toggled.connect(self.set_proofread)
        drawing.addWidget(self.proofread_button)
        self.unseen_check = ToggleSwitch("Unseen only")
        self.unseen_check.setChecked(controller.unseen_only)
        self.unseen_check.setToolTip(
            "Draw only images none of the ticked runs trained on: a model looks good "
            "on what it has seen. Off, a picture says where its run trained on the image."
        )
        self.unseen_check.toggled.connect(self._unseen_toggled)
        drawing.addWidget(self.unseen_check)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Show every image whole again.")
        self.fit_button.clicked.connect(self._link.fit)
        drawing.addWidget(self.fit_button)
        drawing.addStretch(1)
        column.addLayout(drawing)

        self.status = WrappingLabel()
        self.status.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.status)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        column.addWidget(self.grid_host, stretch=1)
        self._show_choices()
        self._show_status()

    # -- what is looked at ----------------------------------------------------

    def set_models(self, models) -> None:
        """Look at *models*, a column each, keeping the images drawn where they can still be."""
        if not self.controller.set_models(models):
            return
        offered = {image.image for image in self.controller.offered()}
        self.controller.images = [
            image for image in self.controller.images if image.image in offered
        ]
        self._show_choices()
        self._build_grid()

    def pick(self) -> None:
        """Draw other images, then predict them."""
        self.controller.pick()
        self._build_grid()
        self.predict()

    def predict(self) -> None:
        """Predict what is missing off the interface thread, then draw it."""
        self.controller.forget_failures()
        if not any(self.controller.missing()):
            self._draw()
            return
        device = self.device()
        self.run_stoppable(
            lambda on_progress, cancellation: self.controller.predict(
                device, on_progress=on_progress, cancellation=cancellation
            ),
            self._predicted,
            PREDICTING,
        )

    def _predicted(self, error: object) -> None:
        self._draw()
        if error:
            self.status.setText(f"Nothing was predicted: {error}")
            self.status.setStyleSheet(f"color: {WARNING};")

    def set_proofread(self, shown: bool) -> None:
        """Show each image's own label in place of the predictions, or the predictions again."""
        if self.proofread_button.isChecked() != shown:
            self.proofread_button.setChecked(shown)
            return
        if shown != self._proofread:
            self._proofread = shown
            self._draw()

    def hold_label(self) -> None:
        """Show each image's own label, from the keyboard; letting the key go shows the predictions."""
        self.set_proofread(True)

    # -- the controls ---------------------------------------------------------

    def _show_choices(self) -> None:
        """Offer, for each thing an image is chosen by, the values the images offered hold."""
        controller = self.controller
        for name, box in self.filter_boxes.items():
            values = controller.values(name)
            if controller.chosen[name] not in values:
                controller.chosen[name] = ""
            box.blockSignals(True)
            box.clear()
            box.addItem(ANY, "")
            for value in values:
                box.addItem(value, value)
            box.setCurrentIndex(max(box.findData(controller.chosen[name]), 0))
            box.setEnabled(bool(values))
            box.blockSignals(False)

    def _show_status(self) -> None:
        controller = self.controller
        said = controller.describe_offered()
        if controller.models and controller.offered() and not controller.images:
            said += " Pick images to draw some."
        self.status.setText(said)
        self.status.setStyleSheet(f"color: {MUTED};")
        usable = any(model.checkpoint is not None for model in controller.models)
        self.pick_button.setEnabled(usable and bool(controller.offered()))
        self.predict_button.setEnabled(usable and any(controller.missing()))
        for widget in (self.label_check, self.alpha_slider, self.proofread_button, self.fit_button):
            widget.setEnabled(bool(self.views))

    def _filter_chosen(self, name: str) -> None:
        self.controller.chosen[name] = self.filter_boxes[name].currentData() or ""

    def _count_changed(self, count: int) -> None:
        self.controller.count = count

    def _label_toggled(self, shown: bool) -> None:
        self.controller.label_shown = shown
        self._draw()

    def _alpha_moved(self, value: int) -> None:
        self.controller.alpha = value / 100.0
        self._draw()

    def _unseen_toggled(self, unseen: bool) -> None:
        self.controller.unseen_only = unseen
        offered = {image.image for image in self.controller.offered()}
        self.controller.images = [
            image for image in self.controller.images if image.image in offered
        ]
        self._show_choices()
        self._build_grid()

    # -- the grid -------------------------------------------------------------

    def _build_grid(self) -> None:
        """Lay out a pane for every image under every run, and draw them."""
        self._link.clear()
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.removeEventFilter(self)
                widget.deleteLater()
        self.views, self.notes = {}, {}
        controller = self.controller
        if controller.images:
            for across, model in enumerate(controller.models):
                heading = QLabel(model.caption())
                heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
                heading.setToolTip(str(model.checkpoint.path) if model.checkpoint else model.which)
                self.grid.addWidget(heading, 0, across + 1)
                self.grid.setColumnStretch(across + 1, 1)
        for down, image in enumerate(controller.images):
            said = "\n".join(part for part in (*image.describe().split(" · "), image.image.name) if part)
            heading = QLabel(said)
            heading.setToolTip(str(image.image))
            heading.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.grid.addWidget(heading, 2 * down + 1, 0, 2, 1)
            self.grid.setRowStretch(2 * down + 1, 1)
            for across in range(len(controller.models)):
                view = ImageView()
                view.installEventFilter(self)
                self._link.add(view)
                note = QLabel()
                note.setAlignment(Qt.AlignmentFlag.AlignCenter)
                note.setStyleSheet(f"color: {WARNING};")
                self.grid.addWidget(view, 2 * down + 1, across + 1)
                self.grid.addWidget(note, 2 * down + 2, across + 1)
                self.views[down, across] = view
                self.notes[down, across] = note
        self._draw()
        # Once the panes have their sizes: a picture fitted before then is fitted to no
        # pane. Given this widget, so the call is dropped if it has gone by then.
        QTimer.singleShot(0, self, self._fit_when_laid_out)

    def _fit_when_laid_out(self) -> None:
        self.grid.activate()
        self._link.fit()

    def _draw(self) -> None:
        """Draw every pane as the controller has it now, keeping where each is looking."""
        controller = self.controller
        for (down, across), view in self.views.items():
            model, image = controller.models[across], controller.images[down]
            picture = controller.picture(model, image, proofread=self._proofread)
            if picture is None:
                view.clear()
            else:
                view.set_array(picture)
            shown = controller.shown(image)
            said = shown if isinstance(shown, str) else controller.note(model, image)
            self.notes[down, across].setText(said)
            self.notes[down, across].setVisible(bool(said))
        self._show_status()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        """Hold the compare key over a picture to see the image's own label."""
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == COMPARE_KEY_CODE and not event.isAutoRepeat():
                self.set_proofread(event.type() == QEvent.Type.KeyPress)
                return True
        return super().eventFilter(watched, event)


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {MUTED};")
    return label
