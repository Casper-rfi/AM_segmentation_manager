"""The Train tab: set a run up on the left, follow the runs there are on the right.

A view of :class:`~am_segmentation_manager.gui.training_controller.TrainingController`
and nothing more: it decides nothing a run does. Everything on the right is read
from the training directory again while a run is live and the tab is showing, and
never otherwise. **Nothing is written or deleted without a dialog saying what first.**
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...operations.training_process import LOG_FILENAME, TrainingNotStarted
from ...operations.training_run import TrainingPlan, TrainingRunRefused
from ...plots import has_plot_support, training_figure
from ...training_runs import PER_CLASS_METRICS
from ..training_controller import RunSummary, TrainingController
from .chart_view import ChartView
from .controls import ElidingLabel, GripSplitter, SegmentedControl, WrappingLabel
from .paths import ALL_FILES_FILTER, choose_directory, choose_open_file, choose_save_file
from .report_dialog import ReportDialog
from .theme import ERROR, MUTED, SECTION_HEADING

logger = logging.getLogger(__name__)

#: How often the runs are read again while one is live and the tab is showing.
#: A run writes once an epoch, which is minutes apart.
POLL_MILLISECONDS = 10_000

MANIFEST_FILTER = "Training export manifest (*_manifest.yaml)"
MODEL_FILTER = "Model weights (*.pth *.pt)"

#: About what one checkpoint takes on disk, for the save-every-epoch box.
CHECKPOINT_MEGABYTES = 170

MODE_SEGMENTS = (
    ("new", "New model", "Train a model from ImageNet weights, which are downloaded once."),
    ("refine", "Refine a model", "Start from an existing model's weights and train it further."),
)

METRIC_CAPTIONS = {
    "iou": "IoU", "dice": "Dice", "f1": "F1",
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
}


class _DeviceCombo(QComboBox):
    """A combo that lists the machine's devices the moment it is opened, memory free included."""

    def __init__(self, fill: Callable[[], None]) -> None:
        super().__init__()
        self._fill = fill

    def showPopup(self) -> None:  # noqa: N802 - Qt's spelling
        self._fill()
        super().showPopup()


class TrainingPanel(QWidget):
    """Set a training run up, start it, and follow and tidy the runs there are.

    Attributes:
        run_in_background: Called as ``run(work, done)`` to do *work* off the
            interface thread and hand its result to *done*. The window's, behind
            its busy dialog; inline where nothing replaces it.
    """

    #: Emitted when something the window remembers between sessions has changed.
    remembered = Signal()

    def __init__(self, controller: TrainingController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.run_in_background: Callable[[Callable[[], object], Callable[[object], None]], None]
        self.run_in_background = lambda work, done: done(work())
        self._drawn: tuple | None = None

        split = GripSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_form())
        split.addWidget(self._build_runs())
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1000])
        layout = QHBoxLayout(self)
        layout.addWidget(split)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MILLISECONDS)
        self._timer.timeout.connect(self.poll)
        self.refresh()

    # -- building -------------------------------------------------------------

    def _build_form(self) -> QWidget:
        form = QWidget()
        column = QVBoxLayout(form)
        column.addWidget(_heading("New run"))

        column.addWidget(_muted("Training set"))
        self.manifest_label = ElidingLabel()
        column.addWidget(self.manifest_label)
        self.set_summary = WrappingLabel()
        self.set_summary.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.set_summary)
        set_row = QHBoxLayout()
        self.last_export_button = QPushButton("Use the last export")
        self.last_export_button.setToolTip(
            "The training set most recently written with Training CSV... on Export."
        )
        self.last_export_button.clicked.connect(self._use_last_export)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip(
            "Choose the <name>_manifest.yaml a training export wrote beside its CSV files."
        )
        self.browse_button.clicked.connect(self._browse_manifest)
        set_row.addWidget(self.last_export_button)
        set_row.addWidget(self.browse_button)
        set_row.addStretch(1)
        column.addLayout(set_row)

        self.mode = SegmentedControl(MODE_SEGMENTS)
        self.mode.chosen.connect(self._mode_chosen)
        # Kept to its own width, or the column stretches its two halves apart.
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode)
        mode_row.addStretch(1)
        column.addLayout(mode_row)

        self.weights_row = QWidget()
        weights = QHBoxLayout(self.weights_row)
        weights.setContentsMargins(0, 0, 0, 0)
        self.weights_label = ElidingLabel()
        self.weights_button = QPushButton("Weights...")
        self.weights_button.setToolTip("Choose the model weights (.pth) to refine.")
        self.weights_button.clicked.connect(self._browse_weights)
        weights.addWidget(self.weights_label, stretch=1)
        weights.addWidget(self.weights_button)
        column.addWidget(self.weights_row)

        fields = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "What the run is called: its folder, and the start of its checkpoints' names."
        )
        self.name_edit.textEdited.connect(self._name_edited)
        fields.addRow("Name", self.name_edit)
        self.device_box = _DeviceCombo(self._fill_devices)
        self.device_box.setEditable(True)
        self.device_box.setToolTip(
            "Where to train. auto takes the NVIDIA GPU with the most memory free; cuda:N is "
            "nvidia-smi's GPU N. AM-model-training trains on an NVIDIA GPU or the CPU."
        )
        self.device_box.addItem("auto", "auto")
        self.device_box.addItem("cpu", "cpu")
        self.device_box.currentTextChanged.connect(self._device_changed)
        fields.addRow("Device", self.device_box)
        column.addLayout(fields)

        self.advanced_check = QCheckBox("Advanced settings")
        self.advanced_check.setToolTip("Epochs, learning rates, batch sizes and what is kept.")
        column.addWidget(self.advanced_check)
        self.advanced: QWidget = self._build_advanced()
        self.advanced.setVisible(False)
        self.advanced_check.toggled.connect(self.advanced.setVisible)
        column.addWidget(self.advanced)

        self.problems_label = WrappingLabel()
        self.problems_label.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.problems_label)
        self.start_button = QPushButton("Start...")
        self.start_button.setDefault(True)
        self.start_button.setToolTip(
            "Check the training set, each label opened, then say what will be done and ask."
        )
        self.start_button.clicked.connect(self.start)
        column.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignRight)
        column.addStretch(1)
        return form

    def _build_advanced(self) -> QWidget:
        advanced = QWidget()
        fields = QFormLayout(advanced)
        fields.setContentsMargins(0, 0, 0, 0)
        self.max_epochs = _spin(1, 10_000, "The most epochs to train; a run stops early once it stops improving.")
        self.frozen_epochs = _spin(0, 10_000, "Epochs the encoder is held frozen for before it is trained.")
        self.initial_rate = _rate("Where the learning rate starts.")
        self.max_rate = _rate("The learning rate's peak.")
        self.batch = _spin(1, 256, "Images per training step; fewer needs less GPU memory.")
        self.validation_batch = _spin(1, 256, "Images per validation step.")
        self.seed = _spin(0, 2**31 - 1, "Fixes augmentation and shuffling.")
        self.save_all = QCheckBox(f"Keep every epoch (about {CHECKPOINT_MEGABYTES} MB each)")
        self.save_all.setToolTip(
            "Keep a checkpoint after every epoch, rather than only after those that improved."
        )
        for caption, widget, field in (
            ("Maximum epochs", self.max_epochs, "max_epochs"),
            ("Frozen epochs", self.frozen_epochs, "frozen_epochs"),
            ("Initial learning rate", self.initial_rate, "initial_learning_rate"),
            ("Maximum learning rate", self.max_rate, "max_learning_rate"),
            ("Training batch", self.batch, "training_batch_size"),
            ("Validation batch", self.validation_batch, "validation_batch_size"),
            ("Seed", self.seed, "seed"),
        ):
            fields.addRow(caption, widget)
            if isinstance(widget, QSpinBox):
                widget.valueChanged.connect(lambda value, field=field: self._changed(field, value))
            else:
                widget.editingFinished.connect(lambda widget=widget, field=field: self._rate_edited(field, widget))
        fields.addRow(self.save_all)
        self.save_all.toggled.connect(lambda kept: self._changed("save_all", kept))
        self.reset_button = QPushButton("Published settings")
        self.reset_button.setToolTip("Put every setting back to the one the models were published with.")
        self.reset_button.clicked.connect(self._reset_settings)
        fields.addRow(self.reset_button)
        return advanced

    def _build_runs(self) -> QWidget:
        runs = QWidget()
        column = QVBoxLayout(runs)
        top = QHBoxLayout()
        top.addWidget(_heading("Runs"))
        self.directory_label = ElidingLabel()
        self.directory_button = QPushButton("Training directory...")
        self.directory_button.setToolTip(
            "Where runs are kept, each in a folder of its own. Keep it outside the collection."
        )
        self.directory_button.clicked.connect(self._choose_directory)
        top.addWidget(self.directory_label, stretch=1)
        top.addWidget(self.directory_button)
        column.addLayout(top)

        self.runs_list = QListWidget()
        self.runs_list.setMaximumHeight(150)
        self.runs_list.currentItemChanged.connect(self._run_chosen)
        column.addWidget(self.runs_list)

        self.status_label = WrappingLabel()
        column.addWidget(self.status_label)
        self.asked_label = WrappingLabel()
        self.asked_label.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.asked_label)

        chart_row = QHBoxLayout()
        chart_row.addWidget(_muted("Metric per class"))
        self.metric_box = QComboBox()
        for metric in PER_CLASS_METRICS:
            self.metric_box.addItem(METRIC_CAPTIONS[metric], metric)
        self.metric_box.currentIndexChanged.connect(lambda _index: self._show_selected())
        chart_row.addWidget(self.metric_box)
        chart_row.addStretch(1)
        column.addLayout(chart_row)
        self.chart = ChartView()
        # The counts behind a breakdown; a run's are its metrics files.
        self.chart.export_button.setVisible(False)
        column.addWidget(self.chart, stretch=1)

        column.addWidget(_muted("Checkpoints"))
        self.checkpoints_list = QListWidget()
        self.checkpoints_list.setFlow(QListWidget.Flow.LeftToRight)
        self.checkpoints_list.setWrapping(True)
        self.checkpoints_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.checkpoints_list.setMaximumHeight(72)
        self.checkpoints_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.checkpoints_list.itemSelectionChanged.connect(self._show_buttons)
        column.addWidget(self.checkpoints_list)

        buttons = QHBoxLayout()
        self.copy_button = QPushButton("Save a copy...")
        self.copy_button.setToolTip(
            "Copy the selected epoch's weights to wherever you keep your models. "
            "The run keeps its own."
        )
        self.copy_button.clicked.connect(self.copy_selected)
        self.delete_button = QPushButton("Delete the others...")
        self.delete_button.setToolTip(
            "Delete this run's checkpoints except the selected ones, to free the space. "
            "Asks first; no other run is touched."
        )
        self.delete_button.clicked.connect(self.delete_others)
        self.log_button = QPushButton("Open log")
        self.log_button.setToolTip("Open the run's train.log, where a failure says why.")
        self.log_button.clicked.connect(self._open_log)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip(
            "End the run. The epoch in progress is lost; checkpoints already written are kept."
        )
        self.stop_button.clicked.connect(self.stop_selected)
        for button in (self.copy_button, self.delete_button, self.log_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        column.addLayout(buttons)
        return runs

    # -- showing --------------------------------------------------------------

    def refresh(self) -> None:
        """Show the controller as it is now: the form, then the runs."""
        self._show_form()
        self._show_runs()

    def poll(self) -> None:
        """Read the runs again if one may have changed and the tab is showing."""
        if self.isVisible() and self.controller.any_live():
            self._show_runs()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        super().hideEvent(event)
        self._timer.stop()

    def _show_form(self) -> None:
        controller = self.controller
        manifest = controller.manifest
        self.manifest_label.setText(str(manifest) if manifest else "None chosen")
        self.set_summary.setText(controller.training_set_summary())
        self.last_export_button.setEnabled(
            controller.last_export is not None and controller.last_export != manifest
        )
        self.mode.set_current(controller.mode)
        refining = controller.mode == "refine"
        self.weights_row.setVisible(refining)
        self.weights_label.setText(
            str(controller.base_weights) if controller.base_weights else "No weights chosen"
        )
        if not controller.name and controller.training_directory is not None:
            controller.name = controller.suggest_name()
        if self.name_edit.text() != controller.name:
            self.name_edit.setText(controller.name)
        if self._device() != controller.device:
            self.device_box.setCurrentText(controller.device)

        settings = controller.settings
        for widget, value in (
            (self.max_epochs, settings.max_epochs), (self.frozen_epochs, settings.frozen_epochs),
            (self.batch, settings.training_batch_size),
            (self.validation_batch, settings.validation_batch_size), (self.seed, settings.seed),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.initial_rate.setText(f"{settings.initial_learning_rate:g}")
        self.max_rate.setText(f"{settings.max_learning_rate:g}")
        self.save_all.blockSignals(True)
        self.save_all.setChecked(settings.save_all)
        self.save_all.blockSignals(False)
        self._show_problems()

    def _show_problems(self) -> None:
        problems = self.controller.problems()
        if problems and not self.controller.can_train():
            problems = [f"{problems[0]}. {self.controller.install_hint()}", *problems[1:]]
        self.problems_label.setText("\n".join(problems))
        self.start_button.setEnabled(not problems)

    def _show_runs(self) -> None:
        controller = self.controller
        directory = controller.training_directory
        self.directory_label.setText(str(directory) if directory else "No training directory chosen")
        runs = controller.runs()
        self.runs_list.blockSignals(True)
        self.runs_list.clear()
        for run in runs:
            item = QListWidgetItem(run.line())
            item.setData(Qt.ItemDataRole.UserRole, str(run.directory))
            self.runs_list.addItem(item)
            if run.directory == controller.selected:
                self.runs_list.setCurrentItem(item)
        if self.runs_list.currentItem() is None and runs:
            controller.selected = runs[0].directory
            self.runs_list.setCurrentRow(0)
        self.runs_list.blockSignals(False)
        self._show_selected()

    def _show_selected(self) -> None:
        run = self.controller.selected_run()
        if run is None:
            self.status_label.setText(
                "No run yet." if self.controller.training_directory else
                "Choose a training directory to see the runs in it."
            )
            self.asked_label.setText("")
            self.checkpoints_list.clear()
            self._draw(None)
            self._show_buttons()
            return
        self.status_label.setText(
            run.problem or f"{run.name}: {run.status.describe() if run.status else ''}"
        )
        self.status_label.setStyleSheet(
            f"color: {ERROR};" if run.problem or (run.status and run.status.state == "failed")
            else ""
        )
        self.asked_label.setText(_asked(run))
        self._show_checkpoints(run)
        self._draw(run)
        self._show_buttons()

    def _show_checkpoints(self, run: RunSummary) -> None:
        kept = {item.data(Qt.ItemDataRole.UserRole) for item in self.checkpoints_list.selectedItems()}
        best = run.best.epoch if run.best else None
        self.checkpoints_list.blockSignals(True)
        self.checkpoints_list.clear()
        for checkpoint in run.written.checkpoints if run.written else ():
            notes = [note for note, holds in (
                ("best", checkpoint.epoch == best), ("incomplete", not checkpoint.whole()),
            ) if holds]
            caption = f"epoch {checkpoint.epoch}" + (f" ({', '.join(notes)})" if notes else "")
            item = QListWidgetItem(caption)
            item.setData(Qt.ItemDataRole.UserRole, checkpoint.epoch)
            self.checkpoints_list.addItem(item)
            item.setSelected(checkpoint.epoch in kept)
        self.checkpoints_list.blockSignals(False)

    def _draw(self, run: RunSummary | None) -> None:
        """Draw the run's curves, only where they would differ from what is drawn."""
        metric = self.metric_box.currentData()
        written = run.written if run else None
        key = (
            run.directory if run else None, metric,
            len(written.train) if written else 0, len(written.val) if written else 0,
            written.finished if written else False,
        )
        if key == self._drawn:
            return
        self._drawn = key
        if written is None or not (written.train or written.val):
            self.chart.show_message(
                "No epoch has finished yet." if run is not None and run.live
                else "Nothing to draw: the run wrote no epoch." if run is not None else ""
            )
        elif not has_plot_support():
            self.chart.show_message("Charts need matplotlib, which is not installed.")
        else:
            self.chart.show_chart(lambda _shading: training_figure(written, metric))

    def _show_buttons(self) -> None:
        run = self.controller.selected_run()
        chosen = len(self.checkpoints_list.selectedItems())
        ended = run is not None and not run.problem and not run.live
        self.copy_button.setEnabled(chosen == 1)
        self.delete_button.setEnabled(
            ended and 0 < chosen < self.checkpoints_list.count()
        )
        self.log_button.setEnabled(run is not None and (run.directory / LOG_FILENAME).is_file())
        self.stop_button.setEnabled(
            run is not None and run.status is not None and run.status.state == "running"
        )

    # -- the form's edits -----------------------------------------------------

    def _use_last_export(self) -> None:
        self.controller.manifest = self.controller.last_export
        self._show_form()

    def _browse_manifest(self) -> None:
        current = self.controller.manifest
        chosen = choose_open_file(
            self, "Choose a training set", current.parent if current else None,
            filters=(MANIFEST_FILTER, ALL_FILES_FILTER),
        )
        if chosen is not None:
            self.controller.manifest = chosen
            self._show_form()

    def _browse_weights(self) -> None:
        current = self.controller.base_weights
        chosen = choose_open_file(
            self, "Choose model weights to refine", current.parent if current else None,
            filters=(MODEL_FILTER, ALL_FILES_FILTER),
        )
        if chosen is not None:
            self.controller.base_weights = chosen
            self._show_form()
            self.remembered.emit()

    def _choose_directory(self) -> None:
        chosen = choose_directory(
            self, "Choose the training directory", self.controller.training_directory
        )
        if chosen is not None:
            self.controller.training_directory = chosen
            self.controller.selected = None
            self.controller.name = ""
            self.refresh()
            self.remembered.emit()

    def _mode_chosen(self, mode: str) -> None:
        suggested = self.controller.name == self.controller.suggest_name()
        self.controller.set_mode("refine" if mode == "refine" else "new")
        if suggested:
            self.controller.name = ""
        self._show_form()

    def _name_edited(self, name: str) -> None:
        self.controller.name = name.strip()
        self._show_problems()

    def _device(self) -> str:
        """Return the device chosen or typed, without its description."""
        data = self.device_box.currentData()
        text = self.device_box.currentText().strip()
        if data and self.device_box.itemText(self.device_box.currentIndex()) == text:
            return str(data)
        return text.split(" ", 1)[0] or "auto"

    def _device_changed(self, _text: str) -> None:
        self.controller.device = self._device()
        self.remembered.emit()

    def _fill_devices(self) -> None:
        chosen = self._device()
        self.device_box.blockSignals(True)
        self.device_box.clear()
        for device, caption in self.controller.devices():
            self.device_box.addItem(caption, device)
        index = self.device_box.findData(chosen)
        if index >= 0:
            self.device_box.setCurrentIndex(index)
        else:
            self.device_box.setCurrentText(chosen)
        self.device_box.blockSignals(False)

    def _changed(self, field: str, value: object) -> None:
        self.controller.change(**{field: value})
        self._show_problems()

    def _rate_edited(self, field: str, edit: QLineEdit) -> None:
        try:
            rate = float(edit.text())
        except ValueError:
            # Not a number: what the controller holds is put back.
            self._show_form()
            return
        self._changed(field, rate)

    def _reset_settings(self) -> None:
        self.controller.reset_settings()
        self._show_form()

    # -- starting, stopping ---------------------------------------------------

    def start(self) -> None:
        """Check the run in full off the interface thread, then ask before starting it."""
        if self.controller.problems():
            self._show_problems()
            return

        def check() -> object:
            try:
                return self.controller.plan()
            except TrainingRunRefused as refused:
                return refused

        self.run_in_background(check, self._checked)

    def _checked(self, outcome: object) -> None:
        if isinstance(outcome, TrainingRunRefused):
            self._report("This run cannot be started", list(outcome.reasons))
            return
        assert isinstance(outcome, TrainingPlan)
        lines = [*outcome.describe(), "", "Training goes on after this window is closed."]
        if not self._confirm("Start training", lines, "Start"):
            return
        try:
            self.controller.start(outcome)
        except (OSError, TrainingNotStarted) as error:
            self._warn("Training did not start", str(error))
        self.controller.name = ""
        self.refresh()

    def stop_selected(self) -> None:
        """Ask, then stop the selected run."""
        run = self.controller.selected_run()
        if run is None:
            return
        if not self._ask(
            "Stop training?",
            f"Stop {run.name}? The epoch in progress is lost, and a stopped run is not "
            f"continued. Checkpoints already written are kept.",
            "Stop",
        ):
            return
        self.controller.stop(run.directory)
        self._show_runs()

    # -- checkpoints ----------------------------------------------------------

    def _selected_epochs(self) -> list[int]:
        return sorted(
            int(item.data(Qt.ItemDataRole.UserRole)) for item in self.checkpoints_list.selectedItems()
        )

    def copy_selected(self) -> None:
        """Copy the one selected checkpoint to a file the person names."""
        run = self.controller.selected_run()
        epochs = self._selected_epochs()
        if run is None or run.written is None or len(epochs) != 1:
            return
        checkpoint = run.written.checkpoint_for(epochs[0])
        if checkpoint is None:
            return
        destination = self._ask_destination(f"{run.name}_epoch{checkpoint.epoch:03}.pth")
        if destination is None:
            return
        try:
            self.controller.copy_checkpoint(checkpoint, destination)
        except (OSError, ValueError) as error:
            self._warn("The checkpoint was not copied", str(error))
            return
        self._report("Saved a copy", [
            f"Epoch {checkpoint.epoch} of {run.name} is at:", f"  {destination}", "",
            "To predict with it, choose that file as the model on Curate.",
        ])

    def delete_others(self) -> None:
        """Say which of the run's checkpoints would go, ask, then delete them."""
        run = self.controller.selected_run()
        keep = self._selected_epochs()
        if run is None or not keep:
            return
        try:
            doomed = self.controller.other_checkpoints(run.directory, keep)
        except ValueError as error:
            self._warn("Nothing was deleted", str(error))
            return
        if not doomed:
            return
        size = sum(checkpoint.path.stat().st_size for checkpoint in doomed) / 1e9
        if not self._ask(
            "Delete the other checkpoints?",
            f"Delete {len(doomed)} checkpoint(s) of {run.name}, {size:.1f} GB, keeping "
            f"epoch(s) {', '.join(map(str, keep))}?\n\nThis cannot be undone. The run's "
            f"curves and records stay, and no other run is touched.",
            "Delete",
        ):
            return
        try:
            self.controller.remove_other_checkpoints(run.directory, keep)
        except (OSError, ValueError) as error:
            self._warn("Not every checkpoint was deleted", str(error))
        self._show_runs()

    def _open_log(self) -> None:
        run = self.controller.selected_run()
        if run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.directory / LOG_FILENAME)))

    def _run_chosen(self, item: QListWidgetItem | None, _previous=None) -> None:
        self.controller.selected = Path(item.data(Qt.ItemDataRole.UserRole)) if item else None
        self.checkpoints_list.clear()
        self._show_selected()

    # -- dialogs, each one place so a test can answer it -----------------------

    def _confirm(self, title: str, lines: list[str], action: str) -> bool:
        return ReportDialog(self, title, lines, action).exec() == QDialog.DialogCode.Accepted

    def _report(self, title: str, lines: list[str]) -> None:
        ReportDialog(self, title, lines).exec()

    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _ask(self, title: str, text: str, action: str) -> bool:
        """Ask a question whose default is not to go ahead."""
        box = QMessageBox(QMessageBox.Icon.Warning, title, text, parent=self)
        go = box.addButton(action, QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(box.addButton(QMessageBox.StandardButton.Cancel))
        box.exec()
        return box.clickedButton() is go

    def _ask_destination(self, suggested: str) -> Path | None:
        return choose_save_file(
            self, "Save a copy of the checkpoint",
            filters=(MODEL_FILTER,), default_suffix="pth", name=suggested,
        )


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(SECTION_HEADING)
    return label


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {MUTED};")
    return label


def _spin(minimum: int, maximum: int, tooltip: str) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setToolTip(tooltip)
    return box


def _rate(tooltip: str) -> QLineEdit:
    edit = QLineEdit()
    edit.setToolTip(f"{tooltip} A number such as 1e-5.")
    return edit


def _asked(run: RunSummary) -> str:
    """Return what the run was asked to do, in a line."""
    if run.record is None:
        return ""
    settings, base = run.record.settings, run.record.base
    start = f"refining {Path(str(base['file'])).name}" if base else "a new model"
    frozen = f", encoder frozen for {settings.frozen_epochs}" if settings.frozen_epochs else ""
    by = f" · prepared by {run.record.created_by}" if run.record.created_by else ""
    return (
        f"{start} · at most {settings.max_epochs} epochs{frozen} · learning rate "
        f"{settings.initial_learning_rate:g} to {settings.max_learning_rate:g} · "
        f"batches of {settings.training_batch_size}{by}"
    )
