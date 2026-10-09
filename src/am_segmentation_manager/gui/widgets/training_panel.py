"""The Train tab: set a run up on the left, follow the runs there are on the right.

The runs are a tree under the folders they are kept in, as Curate's projects are.

A view of :class:`~am_segmentation_manager.gui.training_controller.TrainingController`
and nothing more: it decides nothing a run does. Everything on the right is read
from the training directory again while a run is live and the tab is showing, and
never otherwise. **Nothing is written or deleted without a dialog saying what first.**
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import QPointF, QSize, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap, QPolygonF
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
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

from ...io.tables import write_csv
from ...operations.training_process import LOG_FILENAME, TrainingNotStarted
from ...operations.training_run import TrainingPlan, TrainingRunRefused
from ...operations.validation_run import Rows, ValidationPlan
from ...plots import class_colour, comparison_figure, has_plot_support, training_figure
from ...training_compare import EVERY_SET, RANGED_VALUES, VALUE_CAPTIONS, Comparison
from ...training_runs import DEFAULT_LABEL_NAMES, PER_CLASS_METRICS
from ..training_controller import RunSummary, TrainingController
from .chart_view import ChartView
from .common import chip_icon
from .controls import (
    ElidingLabel,
    FlowArea,
    FlowLayout,
    GripSplitter,
    SegmentedControl,
    WrappingLabel,
)
from .lists import FOLD_GAP, ProjectTree, add_check_all_buttons, add_fold_buttons
from .paths import (
    ALL_FILES_FILTER,
    add_path_menu,
    choose_directory,
    choose_open_file,
    choose_save_file,
)
from .project_panel import CELL_COLOURS
from .report_dialog import ReportDialog
from .theme import (
    ERROR,
    MUTED,
    SECTION_HEADING,
    SWATCH_CELL,
    SWATCH_GAP,
    SWATCH_HEIGHT,
    SWATCH_WIDTH,
    TEXT,
)

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

VIEW_SEGMENTS = (
    ("curves", "Curves", "The run looked at, epoch by epoch."),
    ("table", "Table", "The ticked runs side by side: one value per run and class."),
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
    #: Emitted with a run's row when its path is asked for, to copy or to reveal.
    run_path_requested = Signal(int)
    run_reveal_requested = Signal(int)

    def __init__(self, controller: TrainingController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.run_in_background: Callable[[Callable[[], object], Callable[[object], None]], None]
        self.run_in_background = lambda work, done: done(work())
        self._drawn: object = None
        self._listed: list[RunSummary] = []
        self.class_boxes: dict[str, QCheckBox] = {}
        self.set_boxes: dict[str, QCheckBox] = {}

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
        self.batch = _spin(
            1, 256,
            "Images per training step. Lower it, to 1 if need be, where the GPU runs out of "
            "memory; a smaller batch trains differently from the default.",
        )
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
        self.reset_button = QPushButton("Default settings")
        self.reset_button.setToolTip("Put every setting back to its default, which is what the models were published with.")
        self.reset_button.clicked.connect(self._reset_settings)
        fields.addRow(self.reset_button)
        return advanced

    def _build_runs(self) -> QWidget:
        listed = QWidget()
        column = QVBoxLayout(listed)
        column.setContentsMargins(0, 0, 0, 0)
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

        # The row looked at decides whose curves and checkpoints show; a tick
        # only puts a run in the set the controller holds.
        self.runs_tree = ProjectTree()
        self.runs_tree.project_selected.connect(self._run_chosen)
        self.runs_tree.row_ticked.connect(self._run_ticked)
        add_path_menu(
            self.runs_tree, "run", self.run_path_requested, self.run_reveal_requested,
            row_of=self.runs_tree.project_of,
        )
        column.addWidget(self.runs_tree, stretch=1)
        acting = QHBoxLayout()
        acting.setSpacing(4)
        add_check_all_buttons(acting, self._tick_all, "runs")
        acting.addSpacing(FOLD_GAP)
        add_fold_buttons(acting, self.runs_tree)
        acting.addStretch(1)
        column.addLayout(acting)

        shown = QWidget()
        column = QVBoxLayout(shown)
        column.setContentsMargins(0, 0, 0, 0)
        self.status_label = WrappingLabel()
        column.addWidget(self.status_label)
        self.asked_label = WrappingLabel()
        self.asked_label.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.asked_label)

        chart_row = QHBoxLayout()
        self.view_control = SegmentedControl(VIEW_SEGMENTS)
        self.view_control.set_current(self.controller.view)
        self.view_control.chosen.connect(self._view_chosen)
        chart_row.addWidget(self.view_control)
        chart_row.addWidget(_muted("Metric per class"))
        self.metric_box = QComboBox()
        for metric in PER_CLASS_METRICS:
            self.metric_box.addItem(METRIC_CAPTIONS[metric], metric)
        self.metric_box.currentIndexChanged.connect(lambda _index: self._show_selected())
        chart_row.addWidget(self.metric_box)
        # Wrapped: a model's classes are as many as it has, in a pane of any width.
        self.class_area = FlowArea()
        self.class_area.setLayout(FlowLayout(spacing=10))
        chart_row.addWidget(self.class_area, stretch=1)
        column.addLayout(chart_row)
        sets_row = QHBoxLayout()
        sets_row.addWidget(_muted("Validated on"))
        self.validated_box = QComboBox()
        self.validated_box.setToolTip(
            "Which images the validation side is of: the list each run validated on as it "
            "trained, a validation made of it since with Validate..., or every one of those."
        )
        self.validated_box.activated.connect(lambda _index: self._sets_chosen())
        sets_row.addWidget(self.validated_box)
        sets_row.addWidget(_muted("split by"))
        self.split_box = QComboBox()
        self.split_box.setToolTip(
            "Divide the validation chosen by what its images are of, as the export "
            "recorded it: the instrument, the voltage, the kind of lamella."
        )
        self.split_box.activated.connect(lambda _index: self._sets_chosen())
        sets_row.addWidget(self.split_box)
        self.subset_box = QComboBox()
        self.subset_box.setToolTip("The part of the validation the curves are of.")
        self.subset_box.activated.connect(lambda _index: self._sets_chosen())
        sets_row.addWidget(self.subset_box)
        # Wrapped: a column of the images file holds as many kinds as it does.
        self.set_area = FlowArea()
        self.set_area.setLayout(FlowLayout(spacing=10))
        sets_row.addWidget(self.set_area, stretch=8)
        # Takes the room the boxes leave, or all of it while they are hidden,
        # so the three lists keep to their own width.
        sets_row.addStretch(1)
        for box in (self.validated_box, self.split_box, self.subset_box):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        column.addLayout(sets_row)

        self.table_row = QWidget()
        table_row = QHBoxLayout(self.table_row)
        table_row.setContentsMargins(0, 0, 0, 0)
        table_row.addWidget(_muted("Each cell"))
        self.value_box = QComboBox()
        for value, caption in VALUE_CAPTIONS.items():
            self.value_box.addItem(caption, value)
        self.value_box.setToolTip(
            "What a cell holds: the metric at the run's best epoch, at the epoch marked as "
            "best (the run's own where none is marked), or its mean or median over epochs."
        )
        self.value_box.currentIndexChanged.connect(lambda _index: self._table_changed())
        table_row.addWidget(self.value_box)
        self.range_label = _muted("over epochs")
        table_row.addWidget(self.range_label)
        self.first_epoch = _spin(1, 10_000, "The first epoch the mean or median is taken over.")
        self.last_epoch = _spin(1, 10_000, "The last epoch the mean or median is taken over.")
        for spin, epoch in zip((self.first_epoch, self.last_epoch), self.controller.epoch_range):
            spin.setValue(epoch)
            spin.valueChanged.connect(lambda _value: self._table_changed())
            table_row.addWidget(spin)
        table_row.addStretch(1)
        column.addWidget(self.table_row)

        self.chart = ChartView()
        self.chart.export_table_requested.connect(self.export_table)
        self.chart.export_button.setToolTip(
            "Write the table as a CSV, a row per run, as am-seg compare-training --output does"
        )
        column.addWidget(self.chart, stretch=1)

        column.addWidget(_muted("Checkpoints"))
        self.checkpoints_list = QListWidget()
        self.checkpoints_list.setFlow(QListWidget.Flow.LeftToRight)
        self.checkpoints_list.setWrapping(True)
        self.checkpoints_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.checkpoints_list.setMaximumHeight(72)
        self.checkpoints_list.setIconSize(QSize(2 * BEST_MARK, BEST_MARK))
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
        self.keep_best_check = QCheckBox("Keep the run's best")
        self.keep_best_check.setChecked(True)
        self.keep_best_check.setToolTip(
            "Delete the others... also keeps the epoch the run counts as best. The "
            "epoch you marked as best is kept whatever is ticked here."
        )
        self.mark_button = QPushButton("Mark as best...")
        self.mark_button.setToolTip(
            "Count the selected epoch as this run's best in place of the one the run "
            "chose; with none selected, asks which epoch. Written beside the run."
        )
        self.mark_button.clicked.connect(self.mark_best)
        self.unmark_button = QPushButton("Clear best")
        self.unmark_button.setToolTip("Take back the epoch marked as best; the run's own stays.")
        self.unmark_button.clicked.connect(self.clear_best)
        self.validate_button = QPushButton("Validate...")
        self.validate_button.setToolTip(
            "Score this run's checkpoints on the rows of another training export, an "
            "image at a time, so any part of them can be compared afterwards."
        )
        self.validate_button.clicked.connect(self.validate)
        self.stop_validating_button = QPushButton("Stop validating")
        self.stop_validating_button.setToolTip(
            "End this run's validations that are scoring. The epochs already scored are "
            "kept; a stopped validation is not continued."
        )
        self.stop_validating_button.clicked.connect(self.stop_validating)
        self.log_button = QPushButton("Open log")
        self.log_button.setToolTip("Open the run's train.log, where a failure says why.")
        self.log_button.clicked.connect(self._open_log)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip(
            "End the run. The epoch in progress is lost; checkpoints already written are kept."
        )
        self.stop_button.clicked.connect(self.stop_selected)
        for button in (
            self.copy_button, self.delete_button, self.keep_best_check,
            self.mark_button, self.unmark_button, self.validate_button,
            self.stop_validating_button, self.log_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        column.addLayout(buttons)

        runs = GripSplitter(Qt.Orientation.Vertical)
        runs.addWidget(listed)
        runs.addWidget(shown)
        runs.setStretchFactor(1, 1)
        runs.setSizes([220, 600])
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
        runs = self._listed = controller.runs()
        directories = [run.directory for run in runs]
        if controller.selected not in directories:
            controller.selected = directories[0] if directories else None
        self.runs_tree.set_projects(
            [run.folder for run in runs],
            [run.line() for run in runs],
            ticked=[run.directory in controller.ticked for run in runs],
            icons=[run_chip(run) for run in runs],
            tooltips=[
                str(run.directory) if run.problem else f"{run.directory}\n{run.progress()}"
                for run in runs
            ],
            current=(
                directories.index(controller.selected) if controller.selected is not None else None
            ),
        )
        self._show_selected()

    def _show_selected(self) -> None:
        self._show_sets()
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
        self.status_label.setText(run.describe() + _chosen_line(run) + _validations_line(run))
        self.status_label.setStyleSheet(
            f"color: {ERROR};" if run.problem or run.state == "failed" else ""
        )
        self.asked_label.setText(_asked(run))
        self._show_classes(run.written.label_names if run.written else DEFAULT_LABEL_NAMES)
        self._show_checkpoints(run)
        self._draw(run)
        self._show_buttons()

    def _show_checkpoints(self, run: RunSummary) -> None:
        kept = {item.data(Qt.ItemDataRole.UserRole) for item in self.checkpoints_list.selectedItems()}
        best = run.best.epoch if run.best else None
        chosen = run.chosen.epoch if run.chosen else None
        self.checkpoints_list.blockSignals(True)
        self.checkpoints_list.clear()
        for checkpoint in run.written.checkpoints if run.written else ():
            notes = [note for note, holds in (
                ("run's best", checkpoint.epoch == best),
                ("marked best", checkpoint.epoch == chosen),
                ("incomplete", not checkpoint.whole()),
            ) if holds]
            caption = f"epoch {checkpoint.epoch}" + (f" ({', '.join(notes)})" if notes else "")
            item = QListWidgetItem(
                best_marks(checkpoint.epoch == best, checkpoint.epoch == chosen), caption
            )
            item.setData(Qt.ItemDataRole.UserRole, checkpoint.epoch)
            self.checkpoints_list.addItem(item)
            item.setSelected(checkpoint.epoch in kept)
        self.checkpoints_list.blockSignals(False)

    def _show_classes(self, names: tuple[str, ...]) -> None:
        """Offer a box per class, each beside the colour its curve is drawn in."""
        if tuple(self.class_boxes) != names:
            layout = self.class_area.layout()
            assert layout is not None, "the area was given its layout when built"
            for box in self.class_boxes.values():
                layout.removeWidget(box)
                box.deleteLater()
            self.class_boxes = {}
            for index, name in enumerate(names):
                box = QCheckBox(name)
                box.setToolTip(f"Draw the curve for {name}. The weighted averages stay over every class.")
                if has_plot_support():
                    swatch = QPixmap(SWATCH_CELL, SWATCH_CELL)
                    swatch.fill(QColor(class_colour(index)))
                    box.setIcon(QIcon(swatch))
                    box.setIconSize(QSize(SWATCH_CELL, SWATCH_CELL))
                box.toggled.connect(lambda shown, name=name: self._class_toggled(name, shown))
                layout.addWidget(box)
                self.class_boxes[name] = box
            self.class_area.sync_height()
        shown = self.controller.shown_classes(names)
        for name, box in self.class_boxes.items():
            box.blockSignals(True)
            box.setChecked(name in shown)
            box.blockSignals(False)

    def _class_toggled(self, name: str, shown: bool) -> None:
        self.controller.show_class(name, shown)
        self._draw(self.controller.selected_run())

    def _show_sets(self) -> None:
        """Offer what the runs listed are validated on, and what that can be divided by."""
        controller = self.controller
        names = controller.validation_names(self._listed)
        if controller.validated_on not in ("", EVERY_SET, *names):
            controller.validated_on = ""
        _fill(
            self.validated_box, controller.validated_on,
            [("", "each run's own list"), *((name, name) for name in names),
             *([(EVERY_SET, "every set")] if names else [])],
        )
        columns = controller.dividing_columns(self._listed)
        if controller.split_by not in ("", *columns):
            controller.split_by = ""
        _fill(self.split_box, controller.split_by, [("", "nothing"), *((c, c) for c in columns)])
        self.split_box.setEnabled(bool(columns))

        run = controller.selected_run()
        parts = list(controller.sets_of(run)) if run is not None and controller.split_by else []
        if controller.subset not in parts:
            controller.subset = parts[0] if parts else ""
        _fill(self.subset_box, controller.subset, [(part, part) for part in parts])

        ticked = [run for run in self._listed if run.directory in controller.ticked]
        offered = controller.set_names([controller.sets_of(run) for run in ticked])
        if tuple(self.set_boxes) != offered:
            layout = self.set_area.layout()
            assert layout is not None, "the area was given its layout when built"
            for box in self.set_boxes.values():
                layout.removeWidget(box)
                box.deleteLater()
            self.set_boxes = {}
            for name in offered:
                box = QCheckBox(name)
                box.setToolTip(f"Give {name} a column under each class.")
                box.setChecked(name not in controller.hidden_sets)
                box.toggled.connect(lambda shown, name=name: self._set_toggled(name, shown))
                layout.addWidget(box)
                self.set_boxes[name] = box
            self.set_area.sync_height()

    def _sets_chosen(self) -> None:
        """Take what the validation side is of from the three boxes, and draw again."""
        controller = self.controller
        controller.validated_on = self.validated_box.currentData() or ""
        controller.split_by = self.split_box.currentData() or ""
        controller.subset = self.subset_box.currentData() or ""
        self._show_sets()
        self._draw(controller.selected_run())

    def _set_toggled(self, name: str, shown: bool) -> None:
        if shown:
            self.controller.hidden_sets.discard(name)
        else:
            self.controller.hidden_sets.add(name)
        self._draw(self.controller.selected_run())

    def _view_chosen(self, view: str) -> None:
        self.controller.view = view
        self._draw(self.controller.selected_run())

    def _table_changed(self) -> None:
        """Take the cell's value and the range of epochs from the controls, and draw again."""
        first, last = self.first_epoch.value(), self.last_epoch.value()
        self.controller.table_value = self.value_box.currentData()
        self.controller.epoch_range = (min(first, last), max(first, last))
        self._draw(self.controller.selected_run())

    def _draw_table(self) -> None:
        """Draw the ticked runs side by side, only where the table would differ from the one drawn."""
        metric = self.metric_box.currentData()
        ticked = [run for run in self._listed if run.directory in self.controller.ticked]
        table = self.controller.comparison(metric, ticked)
        if table == self._drawn:
            return
        self._drawn = table
        if not ticked:
            self.chart.show_message("Tick the runs to compare.")
        elif not table.rows:
            self.chart.show_message("None of the ticked runs has validated an epoch yet.")
        elif not table.columns:
            self.chart.show_message("Tick a class to compare the runs on.")
        elif not has_plot_support():
            self.chart.show_message("Charts need matplotlib, which is not installed.")
        else:
            self.chart.show_chart(lambda _shading: comparison_figure(table))

    def export_table(self) -> None:
        """Write the table drawn to a CSV the person names."""
        table = self._drawn
        if self.controller.view != "table" or not isinstance(table, Comparison):
            return
        destination = self._ask_table_destination()
        if destination is None:
            return
        try:
            write_csv(destination, table.header(), table.cells())
        except (OSError, ValueError) as error:
            self._warn("The table was not written", str(error))

    def _draw(self, run: RunSummary | None) -> None:
        """Draw the view chosen, only where it would differ from what is drawn."""
        tabled = self.controller.view == "table"
        ranged = self.controller.table_value in RANGED_VALUES
        self.table_row.setVisible(tabled)
        for widget in (self.range_label, self.first_epoch, self.last_epoch):
            widget.setVisible(ranged)
        # The numbers behind the table; a run's curves are its metrics files.
        self.chart.export_button.setVisible(tabled)
        # A part of a set is one curve; the parts side by side are the table's columns.
        self.subset_box.setVisible(not tabled and self.subset_box.count() > 0)
        self.set_area.setVisible(tabled and len(self.set_boxes) > 1)
        if tabled:
            self._draw_table()
            return
        metric = self.metric_box.currentData()
        written, validated_on = self.controller.curve_run(run) if run else (None, "")
        classes = self.controller.shown_classes(written.label_names) if written else ()
        chosen = run.chosen.epoch if run and run.chosen else None
        key = (
            run.directory if run else None, metric, classes, chosen, validated_on,
            len(written.train) if written else 0, len(written.val) if written else 0,
            written.finished if written else False,
        )
        if key == self._drawn:
            return
        self._drawn = key
        if written is None or not (written.train or written.val):
            self.chart.show_message(
                "No epoch has finished yet." if run is not None and run.training
                else "Nothing to draw: the run wrote no epoch." if run is not None else ""
            )
        elif not has_plot_support():
            self.chart.show_message("Charts need matplotlib, which is not installed.")
        else:
            self.chart.show_chart(
                lambda _shading: training_figure(
                    written, metric, classes, chosen, validated_on,
                    best=run.best.epoch if run is not None and run.best else None,
                ),
                hover=True,
            )

    def _show_buttons(self) -> None:
        run = self.controller.selected_run()
        chosen = len(self.checkpoints_list.selectedItems())
        ended = run is not None and not run.problem and not run.live
        self.copy_button.setEnabled(chosen == 1)
        self.delete_button.setEnabled(
            ended and 0 < chosen < self.checkpoints_list.count()
        )
        self.keep_best_check.setEnabled(self.delete_button.isEnabled())
        self.mark_button.setEnabled(
            run is not None and run.written is not None and bool(run.written.val) and chosen <= 1
        )
        self.unmark_button.setEnabled(run is not None and run.chosen is not None)
        self.validate_button.setEnabled(
            run is not None and not run.problem and bool(run.kept) and self.controller.can_train()
        )
        self.stop_validating_button.setEnabled(
            run is not None and any(validation.live for validation in run.validations)
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
        keep_run_best = self.keep_best_check.isChecked()
        try:
            doomed = self.controller.other_checkpoints(run.directory, keep, keep_run_best)
        except ValueError as error:
            self._warn("Nothing was deleted", str(error))
            return
        if not doomed:
            return
        size = sum(checkpoint.path.stat().st_size for checkpoint in doomed) / 1e9
        going = {checkpoint.epoch for checkpoint in doomed}
        also = [
            f"epoch {epoch} ({why})"
            for epoch, why in (
                (run.best.epoch if run.best and keep_run_best else None, "the run's best"),
                (run.chosen.epoch if run.chosen else None, "marked as best"),
            )
            if epoch is not None and epoch not in keep and epoch not in going
            and run.written is not None and run.written.checkpoint_for(epoch) is not None
        ]
        kept_too = f" {' and '.join(also).capitalize()} is kept as well." if also else ""
        if not self._ask(
            "Delete the other checkpoints?",
            f"Delete {len(doomed)} checkpoint(s) of {run.name}, {size:.1f} GB, keeping "
            f"epoch(s) {', '.join(map(str, keep))}?{kept_too}\n\nThis cannot be undone. "
            f"The run's curves and records stay, and no other run is touched.",
            "Delete",
        ):
            return
        try:
            self.controller.remove_other_checkpoints(run.directory, keep, keep_run_best)
        except (OSError, ValueError) as error:
            self._warn("Not every checkpoint was deleted", str(error))
        self._show_runs()

    def validate(self) -> None:
        """Ask what to score the selected run on, check it off the interface thread, then ask before starting."""
        run = self.controller.selected_run()
        if run is None:
            return
        asked = self._ask_validation(run)
        if asked is None:
            return
        manifest, name, rows = asked

        def check() -> object:
            try:
                return self.controller.plan_validation(run.directory, manifest, name, rows)
            except TrainingRunRefused as refused:
                return refused

        self.run_in_background(check, self._validation_checked)

    def _validation_checked(self, outcome: object) -> None:
        if isinstance(outcome, TrainingRunRefused):
            self._report("This validation cannot be started", list(outcome.reasons))
            return
        assert isinstance(outcome, ValidationPlan)
        lines = [*outcome.describe(), "", "Scoring goes on after this window is closed."]
        if not self._confirm("Start validating", lines, "Start"):
            return
        try:
            self.controller.start_validation(outcome)
        except (OSError, TrainingNotStarted) as error:
            self._warn("The validation did not start", str(error))
        self._show_runs()

    def stop_validating(self) -> None:
        """Ask, then stop the selected run's validations that are scoring."""
        run = self.controller.selected_run()
        if run is None:
            return
        if not self._ask(
            "Stop validating?",
            f"Stop the validation(s) of {run.name} that are scoring? The epochs already "
            f"scored are kept, and a stopped validation is not continued.",
            "Stop",
        ):
            return
        self.controller.stop_validations(run.directory)
        self._show_runs()

    def mark_best(self) -> None:
        """Mark the selected epoch as the run's best, asking which where none is selected."""
        run = self.controller.selected_run()
        if run is None or run.written is None or not run.written.val:
            return
        epochs = self._selected_epochs()
        epoch = epochs[0] if len(epochs) == 1 else self._ask_epoch(run)
        if epoch is None:
            return
        try:
            self.controller.choose_best(run.directory, epoch)
        except (OSError, ValueError) as error:
            self._warn("No epoch was marked as best", str(error))
        self._show_runs()

    def clear_best(self) -> None:
        """Take back the epoch marked as the selected run's best."""
        run = self.controller.selected_run()
        if run is None:
            return
        try:
            self.controller.clear_best(run.directory)
        except OSError as error:
            self._warn("The mark was not cleared", str(error))
        self._show_runs()

    def _open_log(self) -> None:
        run = self.controller.selected_run()
        if run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.directory / LOG_FILENAME)))

    def _run_chosen(self, row: int) -> None:
        self.controller.selected = self._listed[row].directory
        self.checkpoints_list.clear()
        self._show_selected()

    def _run_ticked(self, row: int, ticked: bool) -> None:
        self.controller.tick(self._listed[row].directory, ticked)
        self._show_sets()
        self._draw(self.controller.selected_run())

    def _tick_all(self, ticked: bool) -> None:
        self.controller.tick_all(ticked, self._listed)
        self.runs_tree.show_ticks([ticked] * len(self._listed))
        self._show_sets()
        self._draw(self.controller.selected_run())

    def run_path(self, row: int) -> Path | None:
        """Return the directory of the run listed at *row*, or None if it is no longer listed."""
        return self._listed[row].directory if 0 <= row < len(self._listed) else None

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

    def _ask_validation(self, run: RunSummary) -> tuple[Path, str, Rows] | None:
        """Ask which export to score *run* on, what to call it and which of its rows."""
        dialog = ValidationDialog(self, run, self.controller.manifest or self.controller.last_export)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.asked()

    def _ask_epoch(self, run: RunSummary) -> int | None:
        """Ask which validated epoch to mark, offering the one marked or the run's own best."""
        assert run.written is not None
        validated = [epoch.epoch for epoch in run.written.val]
        start = run.chosen.epoch if run.chosen else run.best.epoch if run.best else validated[-1]
        epoch, accepted = QInputDialog.getInt(
            self, "Mark an epoch as best",
            f"Epoch of {run.name} to count as its best ({validated[0]} to {validated[-1]}):",
            start, validated[0], validated[-1],
        )
        return epoch if accepted else None

    def _ask_table_destination(self) -> Path | None:
        return choose_save_file(self, "Export table as CSV", name="training_comparison.csv")

    def _ask_destination(self, suggested: str) -> Path | None:
        return choose_save_file(
            self, "Save a copy of the checkpoint",
            filters=(MODEL_FILTER,), default_suffix="pth", name=suggested,
        )


#: What each bar of a run's chip is filled with: the epochs by how the run
#: stands, the weights in a hue of their own, and what is left of either.
CHIP_FINISHED = CELL_COLOURS["done"]
CHIP_LIVE = CELL_COLOURS["in progress"]
CHIP_ENDED_EARLY = CELL_COLOURS["unlabelled"]
CHIP_WEIGHTS = CELL_COLOURS["not started"]
CHIP_TRACK = CELL_COLOURS["empty"]


def chip_fills(run: RunSummary) -> tuple[tuple[float, QColor], tuple[float, QColor]]:
    """Return how much of each bar of *run*'s chip is filled, and with what.

    The upper bar is the epochs trained of the most it may train, left empty
    where nothing says how many that is; the lower, the saved weights still on disk.
    """
    most = run.most_epochs
    colour = CHIP_LIVE if run.training else CHIP_FINISHED if run.state == "finished" else CHIP_ENDED_EARLY
    saved = len(run.saved_epochs)
    return (
        (min(run.epochs / most, 1.0) if most else 0.0, colour),
        (len(run.kept) / saved if saved else 0.0, CHIP_WEIGHTS),
    )


def run_chip(run: RunSummary) -> QIcon:
    """Return the chip beside a run: epochs trained above, weights on disk below."""
    return chip_icon(_bars(chip_fills(run)))


def _bars(fills: tuple[tuple[float, QColor], ...]) -> QPixmap:
    """Paint one bar per fill, each a track filled from the left."""
    pixmap = QPixmap(SWATCH_WIDTH, SWATCH_HEIGHT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        for row, (share, colour) in enumerate(fills):
            top = row * (SWATCH_CELL + SWATCH_GAP)
            painter.fillRect(0, top, SWATCH_WIDTH, SWATCH_CELL, CHIP_TRACK)
            # A share above nothing shows a pixel: one epoch of a hundred is not none.
            width = max(round(share * SWATCH_WIDTH), 1) if share > 0 else 0
            painter.fillRect(0, top, width, SWATCH_CELL, colour)
    finally:
        # A painter left alive past its pixmap crashes the process.
        painter.end()
    return pixmap


def run_legend() -> list[tuple[QPixmap, str]]:
    """Return what the chip beside each run means, as ``(sample, meaning)``."""
    return [
        (
            _bars(((0.6, CHIP_LIVE), (0.3, CHIP_WEIGHTS))),
            "Above, the epochs trained of the most the run may train. Below, how many of "
            "the weights it saved are still on disk.",
        ),
        (_bars(((0.6, CHIP_LIVE), (0.0, CHIP_WEIGHTS))), "Training, or may still be"),
        (_bars(((0.6, CHIP_FINISHED), (0.0, CHIP_WEIGHTS))), "Finished, often before its last epoch"),
        (
            _bars(((0.6, CHIP_ENDED_EARLY), (0.0, CHIP_WEIGHTS))),
            "Ended without finishing: stopped, failed or interrupted",
        ),
        (
            _bars(((0.0, CHIP_LIVE), (0.3, CHIP_WEIGHTS))),
            "The upper bar stays empty while a run am-seg did not start has not said how "
            "many epochs it may train, which it writes when it ends",
        ),
    ]


class ValidationDialog(QDialog):
    """Asks what to score a run on: an export's manifest, a name, and which of its rows."""

    def __init__(self, parent: QWidget, run: RunSummary, manifest: Path | None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Validate {run.name}")
        self._manifest = manifest
        self._taken = {validation.name for validation in run.validations}
        layout = QVBoxLayout(self)
        layout.addWidget(_muted("Training set to score on"))
        row = QHBoxLayout()
        self.manifest_label = ElidingLabel()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip(
            "Choose the <name>_manifest.yaml a training export wrote beside its CSV files."
        )
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.manifest_label, stretch=1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        fields = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("What the validation is called, here and in the run's folder.")
        self.name_edit.textChanged.connect(lambda _text: self._show())
        fields.addRow("Name", self.name_edit)
        self.rows_box = QComboBox()
        self.rows_box.addItem("Its validation list", "validation")
        self.rows_box.addItem("Every row it names", "all")
        self.rows_box.setToolTip(
            "Every row is for an export made of data the run never trained on."
        )
        fields.addRow("Rows", self.rows_box)
        layout.addLayout(fields)
        self.problem_label = WrappingLabel()
        self.problem_label.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.problem_label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.check_button = QPushButton("Check...")
        self.check_button.setDefault(True)
        self.check_button.setToolTip("Open each label, then say what would be scored and ask.")
        self.check_button.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self.check_button)
        layout.addLayout(buttons)
        self._suggest()
        self._show()

    def asked(self) -> tuple[Path, str, Rows]:
        """Return the manifest, the name and the rows chosen."""
        assert self._manifest is not None
        rows: Rows = "all" if self.rows_box.currentData() == "all" else "validation"
        return self._manifest, self.name_edit.text().strip(), rows

    def _browse(self) -> None:
        chosen = choose_open_file(
            self, "Choose a training set to score on",
            self._manifest.parent if self._manifest else None,
            filters=(MANIFEST_FILTER, ALL_FILES_FILTER),
        )
        if chosen is not None:
            self._manifest = chosen
            self._suggest()
            self._show()

    def _suggest(self) -> None:
        """Offer the export's own name where the person has typed none."""
        if self._manifest is not None and not self.name_edit.text():
            self.name_edit.setText(self._manifest.name.removesuffix("_manifest.yaml"))

    def _show(self) -> None:
        name = self.name_edit.text().strip()
        problem = (
            "Choose a training set" if self._manifest is None
            else "Name the validation" if not name
            else f"This run already has a validation named {name}" if name in self._taken
            else ""
        )
        self.manifest_label.setText(str(self._manifest) if self._manifest else "None chosen")
        self.problem_label.setText(problem)
        self.check_button.setEnabled(not problem)


def _fill(box: QComboBox, current: str, entries: list[tuple[str, str]]) -> None:
    """Put *entries* in *box* as ``(value, caption)`` and choose *current*, reporting nothing."""
    box.blockSignals(True)
    box.clear()
    for value, caption in entries:
        box.addItem(caption, value)
    box.setCurrentIndex(max(box.findData(current), 0))
    box.blockSignals(False)


def _validations_line(run: RunSummary) -> str:
    """Return a line saying how each validation of *run* stands, or "" where it has none."""
    if not run.validations:
        return ""
    return "\nValidations: " + "; ".join(validation.line() for validation in run.validations) + "."


#: Side of one of a checkpoint's two marks, in logical pixels.
BEST_MARK = 14


def best_marks(run_best: bool, chosen: bool) -> QIcon:
    """Return a checkpoint's marks: a hollow star for the run's best, a filled one for the epoch marked.

    Each has a place of its own, so an epoch that is both shows both and the
    captions of every checkpoint line up.
    """
    pixmap = QPixmap(2 * BEST_MARK, BEST_MARK)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(TEXT)
        for place, (shown, filled) in enumerate(((run_best, False), (chosen, True))):
            if not shown:
                continue
            painter.setPen(QPen(colour, 1.2))
            painter.setBrush(colour if filled else Qt.BrushStyle.NoBrush)
            centre = place * BEST_MARK + BEST_MARK / 2
            outer, inner = BEST_MARK * 0.40, BEST_MARK * 0.17
            painter.drawPolygon(QPolygonF([
                QPointF(
                    centre + radius * math.sin(math.pi * point / 5),
                    BEST_MARK / 2 + 0.5 - radius * math.cos(math.pi * point / 5),
                )
                for point in range(10)
                for radius in ((outer if point % 2 == 0 else inner),)
            ]))
    finally:
        # A painter left alive past its pixmap crashes the process.
        painter.end()
    return chip_icon(pixmap)


def best_legend() -> list[tuple[QPixmap, str]]:
    """Return what the stars beside a checkpoint mean, as ``(sample, meaning)``."""
    size = QSize(2 * BEST_MARK, BEST_MARK)
    return [
        (best_marks(True, False).pixmap(size), "The epoch the run counts as its best"),
        (best_marks(False, True).pixmap(size), "The epoch you marked as best with Mark as best..."),
    ]


def _chosen_line(run: RunSummary) -> str:
    """Return a line saying which epoch is marked as best, or "" where none is."""
    if run.chosen is None:
        return ""
    by = f" by {run.chosen.chosen_by}" if run.chosen.chosen_by else ""
    held = run.written is not None and run.written.checkpoint_for(run.chosen.epoch) is not None
    gone = "" if held else "; its weights are not in this directory"
    return f"\nMarked as best: epoch {run.chosen.epoch}{by}{gone}."


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
