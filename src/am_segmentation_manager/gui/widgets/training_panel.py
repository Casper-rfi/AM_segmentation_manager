"""The Train tab: set a run up on the left, follow the runs there are on the right.

The runs are a tree under the folders they are kept in, as Curate's projects are.

A view of :class:`~am_segmentation_manager.gui.training_controller.TrainingController`
and nothing more: it decides nothing a run does. Everything on the right is read
from the training directory again while a run is live and the tab is showing, the
end of the log of the run looked at more often than the runs, and never otherwise. **Nothing is written or deleted without a dialog saying what first.**
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

from qtpy.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig, load_segmentation_config
from ...io.tables import write_csv
from ...operations.training_process import LOG_FILENAME, TrainingNotStarted
from ...operations.training_run import TrainingPlan, TrainingRunRefused
from ...operations.validation_run import Rows, ValidationPlan, validation_name
from ...plots import (
    class_colour,
    comparison_figure,
    comparison_ramp,
    has_plot_support,
    training_figure,
)
from ...training_compare import (
    COLOUR_STEP,
    EVERY_SET,
    RANGED_VALUES,
    VALUE_CAPTIONS,
    Comparison,
)
from ...training_log import TAIL_LINES
from ...training_preview import PreviewModel
from ...training_runs import DEFAULT_LABEL_NAMES, PER_CLASS_METRICS
from ..preview_controller import PreviewController
from ..training_controller import RUN_COLUMNS, RunSummary, TrainingController
from .chart_view import ChartView
from .common import chip_icon, fill_grouped_combo, fixed_width_font
from .controls import (
    ElidingLabel,
    FlowArea,
    FlowLayout,
    GripSplitter,
    RangeSlider,
    SegmentedControl,
    ToggleSwitch,
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
from .predictions_panel import PredictionsPanel
from .project_panel import CELL_COLOURS
from .report_dialog import ReportDialog
from .theme import (
    ERROR,
    MUTED,
    TEXT,
    WARNING,
)

logger = logging.getLogger(__name__)

#: How many colours of the table's ramp the slider between its two values is
#: drawn from, how many steps that slider has from 0 to 1, and how long a
#: handle rests before the table is drawn again.
RAMP_STEPS = 9
COLOUR_STEPS = round(1 / COLOUR_STEP)
COLOUR_REDRAW_MILLISECONDS = 120
#: The least width of that slider: scores worth comparing lie a few hundredths
#: apart, and two handles that close on a short track sit on each other.
COLOUR_SLIDER_WIDTH = 240

#: How wide the column of runs and the form opens: what the run columns want
#: for a run by its dated name, in a folder, with no validation of it scoring.
LEFT_COLUMN_WIDTH = 560

#: What the window says while a run or a validation is checked in full.
CHECKING = "Checking the training set"

#: How often the runs are read again while one is live and the tab is showing.
#: A run writes once an epoch, which is minutes apart.
POLL_MILLISECONDS = 10_000

#: How often the log of the run looked at is read again while it is live: a bar
#: moves twice a second, and the read is of the file's end only.
LOG_POLL_MILLISECONDS = 2_000

#: How tall the log opens, in lines of its own font, and what its box takes
#: around them: its title, the line saying how the run stands, its margins.
LOG_LINES_SHOWN = 6
LOG_SURROUND = 100

#: What Start says it does, where nothing stands in its way.
START_HINT = (
    "Check the training set, including every label, and show a summary before training starts."
)

MANIFEST_FILTER = "Training export manifest (*_manifest.yaml)"
MODEL_FILTER = "Model weights (*.pth *.pt)"

#: About what one checkpoint takes on disk, for the save-every-epoch box.
CHECKPOINT_MEGABYTES = 170

MODE_SEGMENTS = (
    ("new", "New model", "Train a new model, starting from ImageNet weights (downloaded once)."),
    ("refine", "Refine a model", "Continue training from the weights of an existing model."),
)

VIEW_SEGMENTS = (
    ("curves", "Curves", "Show the curves of the selected run per epoch."),
    ("table", "Table", "Compare the ticked runs in a table: one value per run and class."),
    (
        "predictions", "Predictions",
        "Compare predictions of the ticked runs on a few validation images.",
    ),
)

#: What the button beside the epochs says, by whether the epoch shown is the one marked.
MARK, UNMARK = "Mark as best", "Unmark"

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
        run_in_background: Called as ``run(work, done, description)`` to do
            *work* off the interface thread and hand its result to *done*. The
            window's, behind its busy dialog; inline where nothing replaces it.
    """

    #: Emitted when something the window remembers between sessions has changed.
    remembered = Signal()
    #: Emitted with a run's row when its path is asked for, to copy or to reveal.
    run_path_requested = Signal(int)
    run_reveal_requested = Signal(int)

    def __init__(
        self,
        controller: TrainingController,
        parent: QWidget | None = None,
        config: SegmentationConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.predictions = PredictionsPanel(
            PreviewController(config if config is not None else load_segmentation_config())
        )
        self.predictions.device = lambda: self.controller.device
        self._chart_only: list[QWidget] = []
        self.run_in_background: Callable[
            [Callable[[], object], Callable[[object], None], str], None
        ]
        self.run_in_background = lambda work, done, _description: done(work())
        self._drawn: object = None
        self._drawn_colours: tuple[float, float] | None = None
        self._listed: list[RunSummary] = []
        self.class_boxes: dict[str, QCheckBox] = {}
        self.set_boxes: dict[str, QCheckBox] = {}

        # The runs above the form, as Export has its projects above its
        # filters: the list picked from is on the left, what it shows beside it.
        self.lists = GripSplitter(Qt.Orientation.Vertical)
        self.lists.addWidget(self._build_runs())
        self.lists.addWidget(self._build_form())
        self.lists.setStretchFactor(0, 1)
        self.lists.setSizes([480, 340])
        left = QWidget()
        left_column = QVBoxLayout(left)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(4)
        left_column.addWidget(self._build_directory())
        left_column.addWidget(self.lists, stretch=1)

        self.columns = GripSplitter(Qt.Orientation.Horizontal)
        # The log under what it explains, the divider between them the person's to move.
        self.shown = GripSplitter(Qt.Orientation.Vertical)
        self.shown.addWidget(self._build_shown())
        self.shown.addWidget(self._build_log())
        self.shown.setStretchFactor(0, 1)
        self.shown.setCollapsible(0, False)
        self.shown.setSizes(
            [700, self.log_view.fontMetrics().lineSpacing() * LOG_LINES_SHOWN + LOG_SURROUND]
        )
        self.columns.addWidget(left)
        self.columns.addWidget(self.shown)
        self.columns.setStretchFactor(1, 1)
        self.columns.setSizes([LEFT_COLUMN_WIDTH, 880])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.columns)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MILLISECONDS)
        self._timer.timeout.connect(self.poll)
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(LOG_POLL_MILLISECONDS)
        self._log_timer.timeout.connect(self.poll_log)
        self.refresh()

    # -- building -------------------------------------------------------------

    def _build_directory(self) -> QWidget:
        """Build the training directory's row, which heads the column as Series does elsewhere.

        Above both boxes: it is where the runs listed are read from and where
        a new one is written.
        """
        hint = "Folder where training runs are stored. Keep it outside the collection."
        block = QWidget()
        row = QHBoxLayout(block)
        row.setContentsMargins(4, 0, 4, 0)
        caption = QLabel("Training directory")
        caption.setToolTip(hint)
        self.directory_label = ElidingLabel()
        self.directory_label.setStyleSheet(f"color: {MUTED};")
        self.directory_button = QPushButton("Browse...")
        self.directory_button.setToolTip(hint)
        self.directory_button.clicked.connect(self._choose_directory)
        row.addWidget(caption)
        row.addWidget(self.directory_label, stretch=1)
        row.addWidget(self.directory_button)
        return block

    def _build_form(self) -> QWidget:
        form = QWidget()
        outer = QVBoxLayout(form)
        outer.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("New run")
        outer.addWidget(box)
        # Under the box, so it is as tall as what it holds and no taller.
        outer.addStretch(1)
        column = QVBoxLayout(box)

        column.addWidget(_muted("Training set"))
        self.manifest_label = ElidingLabel()
        column.addWidget(self.manifest_label)
        self.set_summary = WrappingLabel()
        self.set_summary.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.set_summary)
        set_row = QHBoxLayout()
        self.last_export_button = QPushButton("Use the last export")
        self.last_export_button.setToolTip(
            "Use the training set last written with Training CSV... on the Export tab."
        )
        self.last_export_button.clicked.connect(self._use_last_export)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip(
            "Select the <name>_manifest.yaml of a training export."
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
        self.weights_button.setToolTip("Select the model weights (.pth) to refine.")
        self.weights_button.clicked.connect(self._browse_weights)
        weights.addWidget(self.weights_label, stretch=1)
        weights.addWidget(self.weights_button)
        column.addWidget(self.weights_row)

        fields = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "Name of the run, used for its folder and checkpoint file names."
        )
        self.name_edit.textEdited.connect(self._name_edited)
        fields.addRow("Name", self.name_edit)
        self.device_box = _DeviceCombo(self._fill_devices)
        self.device_box.setEditable(True)
        self.device_box.setToolTip(
            "Device to train on. auto takes the NVIDIA GPU with the most free memory. GPU numbers "
            "as set by nvidia-smi command."
        )
        self.device_box.addItem("auto", "auto")
        self.device_box.addItem("cpu", "cpu")
        self.device_box.currentTextChanged.connect(self._device_changed)
        fields.addRow("Device", self.device_box)
        column.addLayout(fields)

        self.advanced_check = QCheckBox("Advanced settings")
        self.advanced_check.setToolTip("Show epochs, learning rates, batch sizes and checkpoint "
                                       "settings.")
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
        self.start_button.clicked.connect(self.start)
        column.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignRight)
        # Scrolled, so the list of runs above can be given room whatever the
        # form holds: with its advanced settings open it is most of the column.
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setWidget(form)
        return scroller

    def _build_advanced(self) -> QWidget:
        advanced = QWidget()
        fields = QFormLayout(advanced)
        fields.setContentsMargins(0, 0, 0, 0)
        self.max_epochs = _spin(1, 10_000, "Maximum number of epochs. Training stops early once "
                                           "the model stops improving.")
        self.frozen_epochs = _spin(0, 10_000, "Number of epochs the encoder stays frozen before "
                                              "it is trained.")
        self.initial_rate = _rate("Learning rate at the start of training.")
        self.max_rate = _rate("Maximum learning rate reached during training.")
        self.batch = _spin(
            1, 256,
            "Images per training step. Lower it if the GPU runs out of memory; a smaller batch "
            "can change the training result.",
        )
        self.validation_batch = _spin(1, 256, "Images per validation step.")
        self.seed = _spin(0, 2**31 - 1, "Seed for augmentation and shuffling, to make training "
                                        "reproducible.")
        self.save_all = QCheckBox(f"Keep every epoch (~{CHECKPOINT_MEGABYTES} MB each)")
        self.save_all.setToolTip(
            "Save a checkpoint after every epoch, instead of only after improved epochs."
        )
        for caption, widget, field in (
            ("Max epochs", self.max_epochs, "max_epochs"),
            ("Frozen epochs", self.frozen_epochs, "frozen_epochs"),
            ("Initial rate", self.initial_rate, "initial_learning_rate"),
            ("Max rate", self.max_rate, "max_learning_rate"),
            ("Training batch", self.batch, "training_batch_size"),
            ("Validation batch", self.validation_batch, "validation_batch_size"),
            ("Seed", self.seed, "seed"),
        ):
            fields.addRow(caption, widget)
            if isinstance(widget, QSpinBox):
                widget.valueChanged.connect(lambda value, field=field: self._changed(field, value))
            else:
                widget.editingFinished.connect(lambda widget=widget, field=field: self._rate_edited(field, widget))
        self.memory_label = _muted("")
        self.memory_label.setToolTip(
            "Estimated GPU memory needed for the training batch. Start... checks whether the "
            "selected GPU has enough."
        )
        batch_row = cast("tuple[int, object]", fields.getWidgetPosition(self.batch))[0]
        fields.insertRow(batch_row + 1, "", self.memory_label)
        fields.addRow(self.save_all)
        self.save_all.toggled.connect(lambda kept: self._changed("save_all", kept))
        self.my_defaults_button = QPushButton("My defaults")
        self.my_defaults_button.setToolTip(
            "Reset all settings to your defaults from Settings (F2)."
        )
        self.my_defaults_button.clicked.connect(self._reset_settings)
        self.recommended_button = QPushButton("Recommended")
        self.recommended_button.setToolTip(
            "Reset all settings to the values the published models were trained with."
        )
        self.recommended_button.clicked.connect(self._reset_recommended)
        resets = QHBoxLayout()
        resets.addWidget(self.my_defaults_button)
        resets.addWidget(self.recommended_button)
        resets.addStretch(1)
        fields.addRow(resets)
        return advanced

    def _build_runs(self) -> QWidget:
        listed = self.runs_box = QGroupBox("Runs")
        column = QVBoxLayout(listed)
        # The row looked at decides whose curves and checkpoints show; a tick
        # only puts a run in the set the controller holds.
        self.runs_tree = ProjectTree()
        self.runs_tree.set_headers(RUN_COLUMNS)
        self.runs_tree.setIconSize(QSize(RUN_MARK, RUN_MARK))
        self.runs_tree.project_selected.connect(self._run_chosen)
        self.runs_tree.row_ticked.connect(self._run_ticked)
        add_path_menu(
            self.runs_tree, "run", self.run_path_requested, self.run_reveal_requested,
            row_of=self.runs_tree.project_of,
        )
        column.addWidget(self.runs_tree, stretch=1)
        # What is listed over what is done to the run looked at: on one row
        # they are wider than the column.
        listing = QHBoxLayout()
        listing.setSpacing(4)
        add_check_all_buttons(listing, self._tick_all, "runs")
        listing.addSpacing(FOLD_GAP)
        add_fold_buttons(listing, self.runs_tree)
        listing.addStretch(1)
        column.addLayout(listing)
        acting = QHBoxLayout()
        acting.setSpacing(4)
        *watching, stop = self._build_run_actions()
        # A flow, so that the buttons wrap where the column is narrower than
        # they are together and do not set how narrow the window can be.
        self.run_actions = FlowArea()
        flow = FlowLayout(spacing=4)
        self.run_actions.setLayout(flow)
        for button in watching:
            flow.addWidget(button)
        acting.addWidget(self.run_actions, stretch=1)
        # Apart from the rest, as what ends a run.
        acting.addWidget(stop, alignment=Qt.AlignmentFlag.AlignTop)
        column.addLayout(acting)
        return listed

    def _build_shown(self) -> QWidget:
        self.plots_box = QGroupBox("Curves, table and predictions")
        column = QVBoxLayout(self.plots_box)

        # One grid, so that what is stacked is aligned: a caption, the list it
        # names at the width of the widest of them, then whatever goes with it.
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        self.view_control = SegmentedControl(VIEW_SEGMENTS)
        self.view_control.set_current(self.controller.view)
        self.view_control.chosen.connect(self._view_chosen)
        grid.addWidget(self.view_control, 0, 3, alignment=Qt.AlignmentFlag.AlignTop)
        # The pictures' own controls, in the rows the chart's leave while they are shown.
        grid.addWidget(self.predictions.controls, 0, 0, 2, 3)
        grid.addWidget(self.predictions.status_line, 1, 3)
        metric_caption = _muted("Metric")
        metric_caption.setToolTip("Select the metric to show for each class.")
        grid.addWidget(metric_caption, 0, 0)
        validated_caption, split_caption = _muted("Validated on"), _muted("split by")
        self.metric_box = QComboBox()
        for metric in PER_CLASS_METRICS:
            self.metric_box.addItem(METRIC_CAPTIONS[metric], metric)
        self.metric_box.currentIndexChanged.connect(lambda _index: self._show_selected())
        grid.addWidget(self.metric_box, 0, 1)
        # Wrapped: a model's classes are as many as it has, in a pane of any width.
        self.class_area = FlowArea()
        self.class_area.setLayout(FlowLayout(spacing=10))
        grid.addWidget(self.class_area, 0, 2)
        grid.addWidget(validated_caption, 1, 0)
        sets_row = QHBoxLayout()
        self.validated_box = QComboBox()
        self.validated_box.setToolTip("Select which validation set the scores are from.")
        self.validated_box.activated.connect(lambda _index: self._sets_chosen())
        grid.addWidget(self.validated_box, 1, 1)
        sets_row.addWidget(split_caption)
        self.split_box = QComboBox()
        self.split_box.activated.connect(lambda _index: self._sets_chosen())
        sets_row.addWidget(self.split_box)
        self.subset_box = QComboBox()
        self.subset_box.setToolTip("Select which subset the curves show, with its number of "
                                   "images.")
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
        grid.addLayout(sets_row, 1, 2, 1, 2)

        self.value_caption = _muted("Each cell")
        grid.addWidget(self.value_caption, 2, 0)
        self.table_row = QWidget()
        table_row = QHBoxLayout(self.table_row)
        table_row.setContentsMargins(0, 0, 0, 0)
        self.value_box = QComboBox()
        for value, caption in VALUE_CAPTIONS.items():
            self.value_box.addItem(caption, value)
        self.value_box.setToolTip(
            "Select the value shown per cell: the metric at the run's best epoch, at the epoch "
            "marked as best, or the mean or median over epochs."
        )
        self.value_box.currentIndexChanged.connect(lambda _index: self._table_changed())
        grid.addWidget(self.value_box, 2, 1)
        self.range_label = _muted("epochs")
        table_row.addWidget(self.range_label)
        self.first_epoch = _spin(1, 10_000, "First epoch included in the mean or median.")
        self.last_epoch = _spin(1, 10_000, "Last epoch included in the mean or median.")
        for spin, epoch in zip((self.first_epoch, self.last_epoch), self.controller.epoch_range):
            spin.setValue(epoch)
            spin.valueChanged.connect(lambda _value: self._table_changed())
            table_row.addWidget(spin)
        table_row.addSpacing(12)
        table_row.addWidget(_muted("Colours"))
        hint = (
            "Set the values for the lightest and darkest colour.\n"
            "Double-click a handle to reset; the arrow keys step by 0.01."
        )
        # The control is the scale: the span between its handles is the ramp
        # the cells run through, and the number at each end is what it stands for.
        self.colour_slider = RangeSlider(0, COLOUR_STEPS, 0, COLOUR_STEPS)
        self.colour_slider.set_ramp(comparison_ramp(RAMP_STEPS))
        self.colour_slider.setMinimumWidth(COLOUR_SLIDER_WIDTH)
        self.colour_slider.setToolTip(hint)
        self.colour_slider.range_changed.connect(self._colours_moved)
        self.colour_low_label, self.colour_high_label = _muted(""), _muted("")
        for widget in (self.colour_low_label, self.colour_slider, self.colour_high_label):
            table_row.addWidget(widget)
        # A drag reports every step of the way, and each table is a figure built anew.
        self._colour_timer = QTimer(self)
        self._colour_timer.setSingleShot(True)
        self._colour_timer.setInterval(COLOUR_REDRAW_MILLISECONDS)
        self._colour_timer.timeout.connect(lambda: self._draw(self.controller.selected_run()))
        self.auto_colours = ToggleSwitch("Auto")
        self.auto_colours.setChecked(self.controller.auto_colours)
        self.auto_colours.setToolTip(
            "Scale the colours automatically from the lowest to the highest value in the table. "
            "Moving a handle switches this off."
        )
        self.auto_colours.toggled.connect(self._auto_colours_toggled)
        table_row.addWidget(self.auto_colours)
        table_row.addStretch(1)
        grid.addWidget(self.table_row, 2, 2, 1, 2)
        column.addLayout(grid)

        self._chart_only = [
            metric_caption, self.metric_box, self.class_area, validated_caption,
            self.validated_box, split_caption, self.split_box,
        ]

        self.chart = ChartView()
        self.chart.export_table_requested.connect(self.export_table)
        self.chart.export_button.setToolTip(
            "Save the table as a CSV file, with one row per run"
        )
        # The chart and the pictures in one place, the switch above choosing between them.
        self.pages = QStackedWidget()
        self.pages.addWidget(self.chart)
        self.pages.addWidget(self.predictions)
        column.addWidget(self.pages, stretch=1)

        column.addWidget(self._build_checkpoints())
        return self.plots_box

    def _build_checkpoints(self) -> QWidget:
        """Build the row under the chart: the weights the run looked at has on disk, and what is done with one."""
        self.checkpoints_row = QWidget()
        row = QHBoxLayout(self.checkpoints_row)
        row.setContentsMargins(0, 0, 0, 0)
        caption = _muted("Epoch")
        hint = (
            "Epochs with weights on disk, with their score. A hollow star marks the run's best "
            "epoch, a filled star the epoch you marked."
        )
        caption.setToolTip(hint)
        self.epoch_box = QComboBox()
        self.epoch_box.setToolTip(hint)
        self.epoch_box.setIconSize(QSize(2 * BEST_MARK, BEST_MARK))
        self.epoch_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.epoch_box.currentIndexChanged.connect(lambda _index: self._show_buttons())
        self.mark_button = QPushButton(MARK)
        self.mark_button.clicked.connect(self.mark_best)
        self.copy_button = QPushButton("Save a copy...")
        self.copy_button.setToolTip(
            "Save a copy of this epoch's weights to another location."
        )
        self.copy_button.clicked.connect(self.copy_selected)
        self.delete_button = QPushButton("Delete the others...")
        self.delete_button.setToolTip(
            "Delete all checkpoints of this run except this epoch's, to free up disk space. Asks "
            "for confirmation."
        )
        self.delete_button.clicked.connect(self.delete_others)
        self.keep_best_check = ToggleSwitch("Keep the run's best")
        self.keep_best_check.setChecked(True)
        self.keep_best_check.setToolTip(
            "Also keep the run's best epoch when using Delete the others... The epoch you marked "
            "as best is always kept."
        )
        for widget in (caption, self.epoch_box, self.mark_button, self.copy_button):
            row.addWidget(widget)
        # What cannot be undone, apart from what can.
        row.addStretch(1)
        row.addWidget(self.delete_button)
        row.addWidget(self.keep_best_check)
        return self.checkpoints_row

    def _build_log(self) -> QWidget:
        """Build what says how the run looked at stands: a line in words over the end of its log."""
        self.log_box = QGroupBox("Log")
        column = QVBoxLayout(self.log_box)
        heading = QHBoxLayout()
        self.standing_label = ElidingLabel()
        self.log_button = QPushButton("Open log")
        self.log_button.setToolTip("Open the full train.log of the run.")
        self.log_button.clicked.connect(self._open_log)
        heading.addWidget(self.standing_label, stretch=1)
        heading.addWidget(self.log_button)
        column.addLayout(heading)
        self.advice_label = WrappingLabel()
        self.advice_label.setStyleSheet(f"color: {ERROR};")
        column.addWidget(self.advice_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(fixed_width_font())
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setMaximumBlockCount(TAIL_LINES)
        self.log_view.setToolTip(
            "The end of the log of the run. Use Open log for the full log."
        )
        self.log_view.setMinimumHeight(self.log_view.fontMetrics().lineSpacing() * 2)
        column.addWidget(self.log_view, stretch=1)
        return self.log_box

    def _build_run_actions(self) -> tuple[QPushButton, ...]:
        """Build what is done to the run looked at, in the order it sits under the list."""
        self.score_own_button = QPushButton("Score own list")
        self.score_own_button.clicked.connect(self.score_own)
        self.validate_button = QPushButton("Validate on export...")
        self.validate_button.setToolTip("Score the checkpoints of this run on another training "
                                        "export.")
        self.validate_button.clicked.connect(self.validate)
        self.stop_validating_button = QPushButton("Stop validating")
        self.stop_validating_button.setToolTip(
            "Stop scoring this run. Scored epochs are kept, but it cannot be resumed."
        )
        self.stop_validating_button.clicked.connect(self.stop_validating)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip(
            "Stop the run. The current epoch is lost; saved checkpoints are kept."
        )
        self.stop_button.clicked.connect(self.stop_selected)
        return (
            self.score_own_button, self.validate_button, self.stop_validating_button,
            self.stop_button,
        )

    # -- showing --------------------------------------------------------------

    def refresh(self) -> None:
        """Show the controller as it is now: the form, then the runs."""
        self._show_form()
        self._show_runs()

    def poll(self) -> None:
        """Read the runs again, and show them if one may have changed and the tab is showing."""
        if not self.isVisible():
            return
        runs = self.controller.runs()
        if any(run.live for run in runs):
            self._show_runs(runs)

    def poll_log(self) -> None:
        """Show the log of the run looked at again, if it may have grown and the tab is showing.

        From the run as last listed: the log is read here, not the run.
        """
        if not self.isVisible():
            return
        run = next(
            (run for run in self._listed if run.directory == self.controller.selected), None
        )
        if run is not None and run.live:
            self._show_log(run)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        super().showEvent(event)
        self._timer.start()
        self._log_timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        super().hideEvent(event)
        self._timer.stop()
        self._log_timer.stop()

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
        self.memory_label.setText(self.controller.memory_needed())
        problems = self.controller.problems()
        # How to install is said once, on the label: a tooltip holding it is too long to read.
        self.start_button.setToolTip("\n".join(problems) or START_HINT)
        if problems and not self.controller.can_train():
            problems = [f"{problems[0]}. {self.controller.install_hint()}", *problems[1:]]
        self.problems_label.setText("\n".join(problems))
        self.start_button.setEnabled(not problems)

    def _show_runs(self, runs: list[RunSummary] | None = None) -> None:
        """List the runs, read now unless the caller has just read them."""
        controller = self.controller
        directory = controller.training_directory
        self.directory_label.setText(str(directory) if directory else "No training directory chosen")
        runs = self._listed = controller.runs() if runs is None else runs
        directories = [run.directory for run in runs]
        if controller.selected not in directories:
            controller.selected = directories[0] if directories else None
        self.runs_tree.set_projects(
            [run.folder for run in runs],
            [f"{run.label}: {run.problem}" if run.problem else run.label for run in runs],
            cells=[run.cells() for run in runs],
            ticked=[run.directory in controller.ticked for run in runs],
            icons=[run_mark(run.state) for run in runs],
            tooltips=[
                str(run.directory) if run.problem
                else f"{run.directory}\n{run.progress()}\n{run.details()}"
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
        self._show_log(run)
        if run is None:
            self.epoch_box.clear()
            self._draw(None)
            self._show_buttons()
            return
        self._show_classes(run.written.label_names if run.written else DEFAULT_LABEL_NAMES)
        self._show_checkpoints(run)
        self._draw(run)
        self._show_buttons()

    def _show_log(self, run: RunSummary | None) -> None:
        """Say how *run* stands, and show the end of its log, kept at its end unless scrolled away from."""
        failed = run is not None and run.status is not None and run.status.state == "failed"
        self.standing_label.setText(run.describe() if run is not None else "")
        self.standing_label.setStyleSheet(f"color: {ERROR if failed else MUTED};")
        advice = run.advice if run is not None else ""
        self.advice_label.setText(advice)
        self.advice_label.setVisible(bool(advice))
        text = "\n".join(self.controller.log_tail(run)) if run is not None else ""
        if text == self.log_view.toPlainText():
            return
        bar = self.log_view.verticalScrollBar()
        at_end, where = bar.value() >= bar.maximum(), bar.value()
        self.log_view.setPlainText(text)
        bar.setValue(bar.maximum() if at_end else where)

    def _show_checkpoints(self, run: RunSummary) -> None:
        """List the epochs of *run* that can be acted on, keeping the one shown where it is still listed."""
        best = run.best.epoch if run.best else None
        chosen = run.chosen.epoch if run.chosen else None
        written = run.written
        on_disk = {checkpoint.epoch: checkpoint for checkpoint in run.kept}
        # A run whose weights are elsewhere can still have an epoch marked.
        listed = set(on_disk) or {epoch.epoch for epoch in (written.val if written else ())}
        if chosen is not None:
            listed.add(chosen)
        scores = {}
        if written is not None:
            for scored in written.val:
                try:
                    scores[scored.epoch] = scored.score(written.key_metrics)
                except (KeyError, ZeroDivisionError):
                    continue
        shown = self._shown_epoch()
        if shown not in listed:
            shown = next((epoch for epoch in (chosen, best) if epoch in listed), max(listed, default=None))
        self.epoch_box.blockSignals(True)
        self.epoch_box.clear()
        for epoch in sorted(listed):
            checkpoint = on_disk.get(epoch)
            weights = (
                "no weights here" if checkpoint is None
                else "" if checkpoint.whole() else "incomplete"
            )
            notes = [note for note, holds in (
                ("run's best", epoch == best), ("marked best", epoch == chosen), (weights, weights),
            ) if holds]
            # The stars say which is best; only what has no mark is written out.
            caption = str(epoch)
            if epoch in scores:
                caption += f" · {scores[epoch]:.3f}"
            if weights:
                caption += f" ({weights})"
            self.epoch_box.addItem(best_marks(epoch == best, epoch == chosen), caption, epoch)
            self.epoch_box.setItemData(
                self.epoch_box.count() - 1,
                f"Epoch {epoch}" + (f": {', '.join(notes)}" if notes else ""),
                Qt.ItemDataRole.ToolTipRole,
            )
        if shown is not None:
            self.epoch_box.setCurrentIndex(self.epoch_box.findData(shown))
        self.epoch_box.blockSignals(False)

    def _shown_epoch(self) -> int | None:
        """Return the epoch the row is showing, or None where it lists none."""
        epoch = self.epoch_box.currentData()
        return None if epoch is None else int(epoch)

    def _show_classes(self, names: tuple[str, ...]) -> None:
        """Offer a switch per class, each in the colour its curve is drawn in."""
        if tuple(self.class_boxes) != names:
            layout = self.class_area.layout()
            assert layout is not None, "the area was given its layout when built"
            for box in self.class_boxes.values():
                layout.removeWidget(box)
                box.deleteLater()
            self.class_boxes = {}
            for index, name in enumerate(names):
                box = ToggleSwitch(name)
                box.setToolTip(f"Show the curve for {name}. Weighted averages always include every class.")
                if has_plot_support():
                    # The colour its curve is drawn in, which a swatch beside it would say twice.
                    box.set_colour(class_colour(index))
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
        self.split_box.blockSignals(True)
        fill_grouped_combo(
            self.split_box, controller.grouped_dividing_columns(self._listed), first=("", "nothing")
        )
        self.split_box.setCurrentIndex(max(self.split_box.findData(controller.split_by), 0))
        self.split_box.blockSignals(False)
        self.split_box.setEnabled(bool(columns))
        self.split_box.setToolTip(controller.split_unavailable(self._listed) or SPLIT_HINT)

        run = controller.selected_run()
        parts = list(controller.sets_of(run)) if run is not None and controller.split_by else []
        if controller.subset not in parts:
            controller.subset = parts[0] if parts else ""
        sizes = controller.sizes_of(run) if run is not None and parts else {}
        _fill(
            self.subset_box, controller.subset,
            [(part, f"{part} ({sizes[part]})" if part in sizes else part) for part in parts],
        )

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
                box = ToggleSwitch(name)
                box.setToolTip(f"Show a column for {name}.")
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
        colours = self.controller.colours_for(table)
        self._show_colours()
        if table == self._drawn and colours == self._drawn_colours:
            return
        self._drawn, self._drawn_colours = table, colours
        if not ticked:
            self.chart.show_message("Tick the runs to compare.")
        elif not table.rows:
            self.chart.show_message("None of the ticked runs has validated an epoch yet.")
        elif not table.columns:
            self.chart.show_message("Tick a class to compare the runs on.")
        elif not has_plot_support():
            self.chart.show_message("Charts need matplotlib, which is not installed.")
        else:
            self.chart.show_chart(lambda _shading: comparison_figure(table, colours))

    def _show_colours(self) -> None:
        """Put the range and whether it follows the table into their controls, without their saying so."""
        low, high = self.controller.colour_range
        self.colour_slider.set_range(
            round(low * COLOUR_STEPS), round(high * COLOUR_STEPS), notify=False
        )
        self.colour_low_label.setText(f"{low:.2f}")
        self.colour_high_label.setText(f"{high:.2f}")
        self.auto_colours.blockSignals(True)
        self.auto_colours.setChecked(self.controller.auto_colours)
        self.auto_colours.blockSignals(False)

    def _colours_moved(self, low: int, high: int) -> None:
        """Take the range from the slider now, and draw the table once the handle rests."""
        self.controller.set_colour_range(low / COLOUR_STEPS, high / COLOUR_STEPS)
        self._show_colours()
        self._colour_timer.start()

    def _auto_colours_toggled(self, following: bool) -> None:
        self.controller.auto_colours = following
        self._draw_table()

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
        pictured = self.controller.view == "predictions"
        self.pages.setCurrentWidget(self.predictions if pictured else self.chart)
        for widget in self._chart_only:
            widget.setVisible(not pictured)
        self.predictions.controls.setVisible(pictured)
        self.predictions.show_status_line(pictured)
        if pictured:
            for widget in (
                self.value_caption, self.value_box, self.table_row, self.subset_box, self.set_area,
            ):
                widget.setVisible(False)
            self.predictions.set_models(self._preview_models())
            return
        ranged = self.controller.table_value in RANGED_VALUES
        for widget in (self.value_caption, self.value_box, self.table_row):
            widget.setVisible(tabled)
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
        # What the run was asked: its own files say only once it has ended.
        frozen = run.record.settings.frozen_epochs if run and run.record else None
        key = (
            run.directory if run else None, metric, classes, chosen, validated_on, frozen,
            len(written.train) if written else 0, len(written.val) if written else 0,
            written.finished if written else False,
        )
        if key == self._drawn:
            return
        self._drawn = key
        if written is None or not (written.train or written.val):
            self.chart.show_message(
                "No epoch has finished yet. The log below says how far the first is."
                if run is not None and run.training
                else "Nothing to draw: the run wrote no epoch." if run is not None
                else "No run yet." if self.controller.training_directory
                else "Choose a training directory to see the runs in it."
            )
        elif not has_plot_support():
            self.chart.show_message("Charts need matplotlib, which is not installed.")
        else:
            self.chart.show_chart(
                lambda _shading: training_figure(
                    written, metric, classes, chosen, validated_on,
                    best=run.best.epoch if run is not None and run.best else None,
                    frozen=frozen,
                ),
                hover=True,
            )

    def _preview_models(self) -> list[PreviewModel]:
        """Return the ticked runs as the Predictions view looks at them, in the order listed."""
        return [
            self.controller.preview_model(run)
            for run in self._listed if run.directory in self.controller.ticked and not run.problem
        ]

    def _show_buttons(self) -> None:
        run = self.controller.selected_run()
        ended = run is not None and not run.problem and not run.live
        epoch = self._shown_epoch()
        written = run.written if run is not None else None
        held = written is not None and epoch is not None and written.checkpoint_for(epoch) is not None
        marked = run is not None and run.chosen is not None and run.chosen.epoch == epoch
        self.copy_button.setEnabled(held)
        self.delete_button.setEnabled(ended and held and run is not None and len(run.kept) > 1)
        self.keep_best_check.setEnabled(self.delete_button.isEnabled())
        self.mark_button.setText(UNMARK if marked else MARK)
        self.mark_button.setToolTip(
            "Remove your mark. The run's own best epoch is used again." if marked
            else "Mark this epoch as the best of the run, instead of the one chosen by the run."
        )
        self.mark_button.setEnabled(
            marked or (written is not None and epoch in {validated.epoch for validated in written.val})
        )
        self.validate_button.setEnabled(
            run is not None and not run.problem and bool(run.kept) and self.controller.can_train()
        )
        against = "Select a run." if run is None else self.controller.can_score_own(run)
        self.score_own_button.setEnabled(not against)
        self.score_own_button.setToolTip(against or SCORE_OWN_HINT)
        self.stop_validating_button.setEnabled(
            run is not None and any(validation.live for validation in run.validations)
        )
        logged = run.logged_in if run is not None else None
        self.log_button.setEnabled(logged is not None and (logged / LOG_FILENAME).is_file())
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

    def _reset_recommended(self) -> None:
        self.controller.reset_recommended()
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

        self.run_in_background(check, self._checked, CHECKING)

    def _checked(self, outcome: object) -> None:
        if isinstance(outcome, TrainingRunRefused):
            self._report("This run cannot be started", list(outcome.reasons))
            return
        assert isinstance(outcome, TrainingPlan)
        lines = [*outcome.describe(), "", "Training goes on after this window is closed."]
        cautions = [
            *self.controller.memory_cautions(outcome),
            *self.controller.start_cautions(outcome, self._listed),
        ]
        if cautions:
            lines += ["", *cautions]
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
        directory = run.directory
        self.run_in_background(
            lambda: self.controller.stop(directory),
            lambda _stopped: self._show_runs(),
            f"Stopping {run.name}",
        )

    # -- checkpoints ----------------------------------------------------------

    def copy_selected(self) -> None:
        """Copy the checkpoint of the epoch shown to a file the person names."""
        run = self.controller.selected_run()
        epoch = self._shown_epoch()
        if run is None or run.written is None or epoch is None:
            return
        checkpoint = run.written.checkpoint_for(epoch)
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
        epoch = self._shown_epoch()
        if run is None or epoch is None:
            return
        keep = [epoch]
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

        self.run_in_background(check, self._validation_checked, CHECKING)

    def score_own(self) -> None:
        """Check a scoring of the selected run's own validation list off the interface thread, then ask before starting."""
        run = self.controller.selected_run()
        if run is None:
            return
        self._check_own(run, None)

    def _check_own(self, run: RunSummary, export: Path | None) -> None:
        def check() -> object:
            try:
                return self.controller.plan_own_validation(run, export)
            except TrainingRunRefused as refused:
                return refused

        asked = export is not None
        self.run_in_background(
            check, lambda outcome: self._own_checked(run, outcome, asked), CHECKING
        )

    def _own_checked(self, run: RunSummary, outcome: object, asked: bool) -> None:
        """Ask once for the run's export where nothing says what its images are of, then go on as a validation does."""
        if isinstance(outcome, ValidationPlan) and len(outcome.columns) == 1 and not asked:
            export = self._ask_own_export(run)
            if export is not None:
                self._check_own(run, export)
                return
        self._validation_checked(outcome)

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
        directory = run.directory
        self.run_in_background(
            lambda: self.controller.stop_validations(directory),
            lambda _stopped: self._show_runs(),
            f"Stopping the validations of {run.name}",
        )

    def mark_best(self) -> None:
        """Mark the epoch shown as the run's best, or take the mark back where it is the one marked."""
        run = self.controller.selected_run()
        epoch = self._shown_epoch()
        if run is None or epoch is None:
            return
        if run.chosen is not None and run.chosen.epoch == epoch:
            self.clear_best()
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
        if run is not None and run.logged_in is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.logged_in / LOG_FILENAME)))

    def _run_chosen(self, row: int) -> None:
        self.controller.selected = self._listed[row].directory
        self.epoch_box.clear()
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
        dialog = ValidationDialog(
            self, run, self.controller.manifest or self.controller.last_export,
            lambda manifest: self.controller.suggested_rows(run.directory, manifest),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.asked()

    def _ask_own_export(self, run: RunSummary) -> Path | None:
        """Ask for the export *run* was prepared from, which says what its validation images are of."""
        return choose_open_file(
            self, f"Choose the training export {run.name} was prepared from (Cancel to score without)",
            run.directory.parent, filters=(MANIFEST_FILTER, ALL_FILES_FILTER),
        )

    def _ask_table_destination(self) -> Path | None:
        return choose_save_file(self, "Export table as CSV", name="training_comparison.csv")

    def _ask_destination(self, suggested: str) -> Path | None:
        return choose_save_file(
            self, "Save a copy of the checkpoint",
            filters=(MODEL_FILTER,), default_suffix="pth", name=suggested,
        )


#: What scoring a run's own list is for, said while it can be done.
SCORE_OWN_HINT = (
    "Score this run on its own validation set per image, so that the results can be split. Only "
    "needed for runs trained with older versions of AM-seg."
)

#: What splitting is for, said while it can be done.
SPLIT_HINT = "Break the validation down by a property of its images, such as the microscope."

#: Side of the mark before a run, in logical pixels; laid out on a grid of sixteen.
RUN_MARK = 14
_MARK_GRID = 16

#: The mark before a run for each way it can stand: ``(shape, colour, meaning)``.
#: No two share a shape, so none is told from another by its colour alone.
RUN_MARKS: dict[str, tuple[str, str, str]] = {
    "prepared": ("ring", MUTED, "Not started"),
    "running": ("triangle", CELL_COLOURS["in progress"].name(), "Training"),
    "finished": ("tick", CELL_COLOURS["done"].name(), "Finished"),
    "stopped": ("square", MUTED, "Stopped by you"),
    "interrupted": ("exclamation", WARNING, "Interrupted"),
    "failed": ("cross", ERROR, "Failed"),
    "unknown": ("question", MUTED, "Started on another computer"),
}

#: What a run am-seg did not start is drawn as, its files saying less than a process does.
_MARKED_AS = {"training": "running", "not finished": "interrupted"}


def run_mark(state: str) -> QIcon:
    """Return the mark before a run that stands as *state*, or no icon where it says nothing."""
    pixmap = _run_mark_pixmap(state)
    return chip_icon(pixmap) if pixmap is not None else QIcon()


def _run_mark_pixmap(state: str, size: int = RUN_MARK) -> QPixmap | None:
    mark = RUN_MARKS.get(_MARKED_AS.get(state, state))
    if mark is None:
        return None
    shape, colour, _meaning = mark
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(size / _MARK_GRID, size / _MARK_GRID)
        _draw_run_mark(painter, shape, QColor(colour))
    finally:
        # A painter left alive past its pixmap crashes the process.
        painter.end()
    return pixmap


def _draw_run_mark(painter: QPainter, shape: str, colour: QColor) -> None:
    """Draw *shape* on the sixteen-pixel grid: stroked where it is a line, filled where an area."""
    pen = QPen(colour)
    pen.setWidthF(2.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if shape in ("triangle", "square"):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
    else:
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
    if shape == "ring":
        painter.drawEllipse(QRectF(3.5, 3.5, 9.0, 9.0))
    elif shape == "triangle":
        painter.drawPolygon(QPolygonF([QPointF(4.5, 2.5), QPointF(13.5, 8.0), QPointF(4.5, 13.5)]))
    elif shape == "tick":
        painter.drawPolyline(QPolygonF([QPointF(3.0, 8.5), QPointF(6.5, 12.0), QPointF(13.0, 4.0)]))
    elif shape == "square":
        painter.drawRect(QRectF(4.0, 4.0, 8.0, 8.0))
    elif shape == "cross":
        painter.drawLine(QPointF(4.0, 4.0), QPointF(12.0, 12.0))
        painter.drawLine(QPointF(12.0, 4.0), QPointF(4.0, 12.0))
    elif shape in ("exclamation", "question"):
        if shape == "exclamation":
            painter.drawLine(QPointF(8.0, 3.0), QPointF(8.0, 9.0))
        else:
            hook = QPainterPath()
            hook.arcMoveTo(QRectF(5.0, 2.5, 6.0, 6.0), 180)
            hook.arcTo(QRectF(5.0, 2.5, 6.0, 6.0), 180, -270)
            hook.lineTo(8.0, 10.0)
            painter.drawPath(hook)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(QPointF(8.0, 13.2), 1.4, 1.4)


def run_legend() -> list[tuple[QPixmap, str]]:
    """Return what the mark before each run means, as ``(sample, meaning)``."""
    return [
        (pixmap, meaning)
        for state, (_shape, _colour, meaning) in RUN_MARKS.items()
        if (pixmap := _run_mark_pixmap(state)) is not None
    ]


class ValidationDialog(QDialog):
    """Asks what to score a run on: an export's manifest, and which of its rows.

    The validation is called what the export is, which nobody types: runs
    validated on one export are then compared with nothing to match up.
    """

    def __init__(
        self, parent: QWidget, run: RunSummary, manifest: Path | None,
        rows_for: Callable[[Path], Rows] | None = None,
    ) -> None:
        """
        Args:
            manifest: The export offered first, or None.
            rows_for: Says which rows of an export the run is best scored on.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Validate {run.name} on another export")
        self._manifest = manifest
        self._taken = {validation.name for validation in run.validations}
        self._rows_for = rows_for
        layout = QVBoxLayout(self)
        layout.addWidget(_muted("Training export to score on"))
        row = QHBoxLayout()
        self.manifest_label = ElidingLabel()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Select the manifest of a training export.")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.manifest_label, stretch=1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        fields = QFormLayout()
        self.rows_box = QComboBox()
        self.rows_box.addItem("Its validation list", "validation")
        self.rows_box.addItem("Every row it names", "all")
        self.rows_box.setToolTip("Use every row only for data the run was not trained on.")
        fields.addRow("Rows", self.rows_box)
        layout.addLayout(fields)
        self.name_label = WrappingLabel()
        self.name_label.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.name_label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.check_button = QPushButton("Check...")
        self.check_button.setDefault(True)
        self.check_button.setToolTip("Check every label and show a summary before scoring starts.")
        self.check_button.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self.check_button)
        layout.addLayout(buttons)
        self._show()

    def name(self) -> str:
        """Return what the validation will be called, or "" before an export is chosen."""
        return validation_name(self._manifest, self._taken) if self._manifest is not None else ""

    def asked(self) -> tuple[Path, str, Rows]:
        """Return the manifest, the name and the rows chosen."""
        assert self._manifest is not None
        rows: Rows = "all" if self.rows_box.currentData() == "all" else "validation"
        return self._manifest, self.name(), rows

    def _browse(self) -> None:
        chosen = choose_open_file(
            self, "Choose a training export to score on",
            self._manifest.parent if self._manifest else None,
            filters=(MANIFEST_FILTER, ALL_FILES_FILTER),
        )
        if chosen is not None:
            self._manifest = chosen
            self._show()

    def _show(self) -> None:
        """Show the export chosen, the rows it is best scored on, and what the validation will be called."""
        manifest = self._manifest
        self.manifest_label.setText(str(manifest) if manifest else "None chosen")
        self.check_button.setEnabled(manifest is not None)
        if manifest is None:
            self.name_label.setText("Choose a training export.")
            return
        if self._rows_for is not None:
            self.rows_box.setCurrentIndex(self.rows_box.findData(self._rows_for(manifest)))
        name, plain = self.name(), validation_name(manifest)
        self.name_label.setText(
            f"It will be called {name}, after the export." if name == plain
            else f"It will be called {name}: this run already has a validation called {plain}."
        )


def _fill(box: QComboBox, current: str, entries: list[tuple[str, str]]) -> None:
    """Put *entries* in *box* as ``(value, caption)`` and choose *current*, reporting nothing."""
    box.blockSignals(True)
    box.clear()
    for value, caption in entries:
        box.addItem(caption, value)
    box.setCurrentIndex(max(box.findData(current), 0))
    box.blockSignals(False)


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
        (best_marks(False, True).pixmap(size), "The epoch you marked as best with Mark as best"),
    ]


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
