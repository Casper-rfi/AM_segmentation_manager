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
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...progress import Cancellation
from ...training_preview import MOST_PREVIEW_IMAGES, PREVIEW_FILTERS
from ..preview_controller import PreviewController
from .common import breakable
from .controls import (
    SLIDER_WIDTH,
    ElidingLabel,
    FlowArea,
    FlowLayout,
    SegmentedControl,
    ToggleSwitch,
    ValueSlider,
    WrappingLabel,
    keep_paired,
    keep_together,
)
from .image_view import ImageView, ViewLink
from .theme import MUTED, WARNING
from .versions_bar import COMPARE_KEY, COMPARE_KEY_CODE

logger = logging.getLogger(__name__)

#: What the busy dialog says while the models predict.
PREDICTING = "Predicting to compare"

#: How long the opacity slider rests before the pictures are drawn again, each being
#: the size of its image.
OPACITY_REDRAW_MILLISECONDS = 120

#: The key that shows and hides the predicted classes, as it does a label on Curate and Label.
OVERLAY_KEY = "V"

#: The segment that shows every image, beside one for each row.
SHOW_ALL = "all"

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
        self.notes: dict[tuple[int, int], ElidingLabel] = {}
        self._rows: list[list[QWidget]] = []
        self._link = ViewLink()
        self._proofread = False
        self._key_held = False
        self._line_allowed = True

        self.filter_boxes: dict[str, QComboBox] = {}
        captions: dict[str, QLabel] = {}
        for caption, name in PREVIEW_FILTERS:
            captions[name] = _muted(caption)
            box = QComboBox()
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            hint = "Only draw images with this value."
            captions[name].setToolTip(hint)
            box.setToolTip(hint)
            box.activated.connect(lambda _index, name=name: self._filter_chosen(name))
            self.filter_boxes[name] = box
        self.count_box = QSpinBox()
        self.count_box.setRange(1, MOST_PREVIEW_IMAGES)
        self.count_box.setValue(controller.count)
        self.count_box.setToolTip("How many images to draw.")
        self.count_box.valueChanged.connect(self._count_changed)
        self.unseen_check = ToggleSwitch("Unseen only")
        self.unseen_check.setChecked(controller.unseen_only)
        self.unseen_check.setToolTip("Only images no ticked run trained on.")
        self.unseen_check.toggled.connect(self._unseen_toggled)
        self.pick_button = QPushButton("Pick images")
        self.pick_button.setToolTip("Draw new images at random and predict them.")
        self.pick_button.clicked.connect(self.pick)
        self.predict_button = QPushButton("Predict the rest")
        self.predict_button.setToolTip("Predict what is not drawn yet.")
        self.predict_button.clicked.connect(self.predict)

        self.label_check = ToggleSwitch("Prediction")
        self.label_check.setChecked(controller.label_shown)
        self.label_check.setToolTip(f"Show the predicted classes ({OVERLAY_KEY}).")
        self.label_check.toggled.connect(self._label_toggled)
        opacity = _muted("Opacity")
        self.alpha_slider = ValueSlider()
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(round(controller.alpha * 100))
        self.alpha_slider.setFixedWidth(SLIDER_WIDTH)
        self.alpha_slider.setToolTip("Opacity of the classes.")
        opacity.setToolTip(self.alpha_slider.toolTip())
        self.alpha_slider.valueChanged.connect(self._alpha_moved)
        self._alpha_timer = QTimer(self)
        self._alpha_timer.setSingleShot(True)
        self._alpha_timer.setInterval(OPACITY_REDRAW_MILLISECONDS)
        self._alpha_timer.timeout.connect(self._draw)
        self.proofread_button = QPushButton("Show label")
        self.proofread_button.setCheckable(True)
        self.proofread_button.setToolTip(f"Hold {COMPARE_KEY} to show the proofread label.")
        self.proofread_button.toggled.connect(self.set_proofread)
        self.show_caption = _muted("Show")
        self.show_control = SegmentedControl([
            (SHOW_ALL, "All", "Show every image."),
            *(
                (str(down), str(down + 1), f"Show image {down + 1} alone ({down + 1}).")
                for down in range(MOST_PREVIEW_IMAGES)
            ),
        ])
        self.show_control.set_current(SHOW_ALL)
        self.show_control.chosen.connect(
            lambda key: self.show_only(None if key == SHOW_ALL else int(key))
        )
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Show every image whole.")
        self.fit_button.clicked.connect(self._link.fit)

        # Pairs in a flow: which images are drawn, then how they are shown, a caption
        # over a caption and a button over a button, wrapping a pair at a time.
        self.controls = FlowArea()
        flow = FlowLayout(spacing=14)
        self.controls.setLayout(flow)
        microscope, lamella, voltage = (name for _, name in PREVIEW_FILTERS)
        left = {(0, 0): Qt.AlignmentFlag.AlignLeft, (1, 0): Qt.AlignmentFlag.AlignLeft}
        for top, bottom, align in (
            (
                (captions[microscope], self.filter_boxes[microscope]),
                (captions[lamella], self.filter_boxes[lamella]), None,
            ),
            (
                (captions[voltage], self.filter_boxes[voltage]),
                (_muted("Images"), self.count_box), {(1, 1): Qt.AlignmentFlag.AlignLeft},
            ),
            ((self.unseen_check,), (keep_together(self.pick_button, self.predict_button),), left),
            ((self.label_check,), (keep_together(opacity, self.alpha_slider),), left),
            (
                (keep_together(self.show_caption, self.show_control),),
                (keep_together(self.proofread_button, self.fit_button),), left,
            ),
        ):
            flow.addWidget(keep_paired(top, bottom, align=align))
        # What there is to draw from, in a line: the pictures want the height.
        self.status_line = ElidingLabel()

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.controls)
        column.addWidget(self.status_line)
        # Said in full while there is no picture for it to take the room of.
        self.status = WrappingLabel()
        column.addWidget(self.status)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        column.addWidget(self.grid_host, stretch=1)
        self._show_choices()
        self._show_rows()
        self._show_status()

    # -- what is looked at ----------------------------------------------------

    def set_models(self, models) -> None:
        """Look at *models*, a column each, keeping the images drawn where they can still be."""
        if not self.controller.set_models(models):
            return
        self._show_choices()
        self._build_grid()

    def pick(self) -> None:
        """Draw other images, then predict those that are on this machine."""
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
            self._say(f"Nothing was predicted: {error}", WARNING)

    def set_proofread(self, shown: bool) -> None:
        """Show each image's own label in place of the predictions, or the predictions again."""
        if self.proofread_button.isChecked() != shown:
            self.proofread_button.setChecked(shown)
            return
        if shown != self._proofread:
            self._proofread = shown
            self._draw()

    def show_only(self, down: int | None) -> None:
        """Show the image of row *down* alone, or every image for None or the row already alone.

        Where the pictures are looking is kept, each pane being another size after.
        """
        controller = self.controller
        if down is not None and not 0 <= down < len(controller.images):
            return
        image = None if down is None else controller.images[down]
        if image is not None and controller.alone == image.image:
            image = None
        looking = self._link.looking_at()
        controller.show_alone(image)
        self._show_rows()
        QTimer.singleShot(0, self, lambda: self._fit_when_laid_out(looking))

    def hold_label(self) -> None:
        """Show each image's own label, from the keyboard; letting the key go shows the predictions.

        The key is watched for across the application until it is let go, whatever
        has the keyboard then.
        """
        application = QApplication.instance()
        if not self._key_held and application is not None:
            self._key_held = True
            application.installEventFilter(self)
        self.set_proofread(True)

    def _let_go(self) -> None:
        application = QApplication.instance()
        if self._key_held and application is not None:
            self._key_held = False
            application.removeEventFilter(self)
        self.set_proofread(False)

    def toggle_label(self) -> None:
        """Show or hide the predicted classes, from the keyboard."""
        self.label_check.setChecked(not self.label_check.isChecked())

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
        not_found = controller.describe_not_found()
        if not_found:
            said = not_found
        elif controller.models and controller.offered() and not controller.images:
            said += " Pick images to draw some."
        self._say(said, WARNING if not_found else MUTED)
        usable = any(model.checkpoint is not None for model in controller.models)
        self.pick_button.setEnabled(usable and bool(controller.offered()))
        self.predict_button.setEnabled(usable and any(controller.missing()))
        for widget in (self.label_check, self.alpha_slider, self.proofread_button, self.fit_button):
            widget.setEnabled(bool(self.views))

    def _say(self, said: str, colour: str) -> None:
        """Say *said* beside the controls while pictures are drawn, and in full while none is."""
        self.status.setText(breakable(said))
        self.status_line.setText(said)
        for label in (self.status, self.status_line):
            label.setStyleSheet(f"color: {colour};")
        self.status.setVisible(not self.views)
        self.status_line.setVisible(self._line_allowed and bool(self.views))

    def show_status_line(self, allowed: bool) -> None:
        """Let the line beside the controls be shown, or keep it hidden: for whoever places it."""
        self._line_allowed = allowed
        self.status_line.setVisible(allowed and bool(self.views))

    def _filter_chosen(self, name: str) -> None:
        self.controller.chosen[name] = self.filter_boxes[name].currentData() or ""

    def _count_changed(self, count: int) -> None:
        self.controller.count = count

    def _label_toggled(self, shown: bool) -> None:
        self.controller.label_shown = shown
        self._draw()

    def _alpha_moved(self, value: int) -> None:
        """Take the opacity from the slider now, and draw the pictures once it rests."""
        self.controller.alpha = value / 100.0
        self._alpha_timer.start()

    def _unseen_toggled(self, unseen: bool) -> None:
        self.controller.unseen_only = unseen
        self.controller.keep_offered()
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
        # A layout keeps the stretch of a row it no longer holds anything in.
        for row in range(self.grid.rowCount()):
            self.grid.setRowStretch(row, 0)
        for across in range(self.grid.columnCount()):
            self.grid.setColumnStretch(across, 0)
        self.views, self.notes, self._rows = {}, {}, []
        controller = self.controller
        if controller.images:
            for across, model in enumerate(controller.models):
                heading = QLabel(model.caption())
                heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
                heading.setToolTip(str(model.checkpoint.path) if model.checkpoint else model.which)
                self.grid.addWidget(heading, 0, across + 1)
                self.grid.setColumnStretch(across + 1, 1)
        for down, image in enumerate(controller.images):
            # The file's name is long and says little: it is with the rest, on hovering.
            said = "\n".join(part for part in (str(down + 1), *image.describe().split(" · ")) if part)
            heading = QLabel(said)
            heading.setToolTip("\n".join((*image.details(), str(image.image))))
            heading.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.grid.addWidget(heading, 2 * down + 1, 0, 2, 1)
            self._rows.append([heading])
            for across in range(len(controller.models)):
                view = ImageView()
                view.installEventFilter(self)
                self._link.add(view)
                note = ElidingLabel()
                note.setAlignment(Qt.AlignmentFlag.AlignCenter)
                note.setStyleSheet(f"color: {WARNING};")
                self.grid.addWidget(view, 2 * down + 1, across + 1)
                self.grid.addWidget(note, 2 * down + 2, across + 1)
                self.views[down, across] = view
                self.notes[down, across] = note
                self._rows[down] += [view, note]
        self._show_rows()
        self._draw()
        # Once the panes have their sizes: a picture fitted before then is fitted to no
        # pane. Given this widget, so the call is dropped if it has gone by then.
        QTimer.singleShot(0, self, self._fit_when_laid_out)

    def _show_rows(self) -> None:
        """Show each row whose image is shown, and say on the control which that is."""
        controller = self.controller
        current = SHOW_ALL
        for down, widgets in enumerate(self._rows):
            shown = controller.shows(controller.images[down])
            if shown and controller.alone is not None:
                current = str(down)
            self.grid.setRowStretch(2 * down + 1, 1 if shown else 0)
            for widget in widgets:
                # A note with nothing to say stays hidden; drawing shows it again.
                widget.setVisible(shown and not (isinstance(widget, ElidingLabel) and not widget.text()))
        several = len(self._rows) > 1
        self.show_caption.setVisible(several)
        self.show_control.setVisible(several)
        for key, button in self.show_control.buttons.items():
            button.setVisible(key == SHOW_ALL or int(key) < len(self._rows))
        self.show_control.set_current(current)

    def _fit_when_laid_out(self, looking: tuple[float, float, float] | None = None) -> None:
        self.grid.activate()
        if looking is None:
            self._link.fit()
        else:
            self._link.look_at(*looking)

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
            self.notes[down, across].setVisible(
                bool(said) and controller.shows(image)
            )
        self._show_status()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        """Show each image's own label while the compare key is down, and the predictions after."""
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == COMPARE_KEY_CODE and not event.isAutoRepeat():
                if event.type() == QEvent.Type.KeyPress:
                    self.hold_label()
                else:
                    self._let_go()
                return True
        return super().eventFilter(watched, event)


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {MUTED};")
    return label
