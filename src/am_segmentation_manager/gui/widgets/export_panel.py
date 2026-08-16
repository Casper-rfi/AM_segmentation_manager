"""The Export tab: choose a subset, see what it holds, write it out.

The reason this is in the window rather than only on the command line: choosing
a training set is iterative. You filter, look at what survives, and adjust. The
live count in the middle is the whole point -- ``keep_one_condition_per_scene``
and the empty-label rule both drop images silently on the command line, and
seeing what they drop is what makes the composition of an export visible before
it is written.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...filtering import METADATA_FILTERS
from ..export_controller import MODALITIES, STATES, ExportController

logger = logging.getLogger(__name__)


class ExportPanel(QWidget):
    """Filters, a live count, a breakdown, and the export buttons."""

    #: Emitted when the user asks for one of the exports, with the chosen path.
    training_export_requested = Signal(str)
    image_metadata_export_requested = Signal(str)
    project_metadata_export_requested = Signal(str)

    #: Emitted with a CSV to check the recorded paths of.
    csv_check_requested = Signal(str)

    def __init__(self, controller: ExportController) -> None:
        super().__init__()
        self.controller = controller
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        layout.addWidget(self._build_filters(), stretch=0)
        layout.addWidget(self._build_middle(), stretch=1)
        layout.addWidget(self._build_actions(), stretch=0)

    # -- construction ------------------------------------------------------

    def _build_filters(self) -> QWidget:
        box = QGroupBox("Filters")
        box.setMinimumWidth(260)
        column = QVBoxLayout(box)

        self.modality = QComboBox()
        self.modality.addItems(MODALITIES)
        self.modality.currentTextChanged.connect(self._filters_changed)
        column.addWidget(QLabel("Modality"))
        column.addWidget(self.modality)

        self.state = QComboBox()
        self.state.addItems(STATES)
        self.state.currentTextChanged.connect(self._filters_changed)
        column.addWidget(QLabel("Workflow state"))
        column.addWidget(self.state)

        column.addSpacing(6)
        self.load_button = QPushButton("Load image metadata")
        self.load_button.setToolTip(
            "Read every image's header, which is what the acquisition filters "
            "below need. About half a millisecond per image."
        )
        self.load_button.clicked.connect(self._load_metadata)
        column.addWidget(self.load_button)

        # One list per metadata property, populated from what the data actually
        # holds -- a free-text box would need you to know the answer already.
        #
        # Hidden until the headers are read. Eight empty boxes read as broken,
        # and they are not usable before the values exist to choose from.
        self.metadata_box = QWidget()
        metadata_column = QVBoxLayout(self.metadata_box)
        metadata_column.setContentsMargins(0, 0, 0, 0)
        self.metadata_lists: dict[str, QListWidget] = {}
        for name in sorted(METADATA_FILTERS):
            metadata_column.addWidget(QLabel(name.replace("-", " ").title()))
            widget = QListWidget()
            widget.setSelectionMode(QListWidget.MultiSelection)
            widget.setMaximumHeight(64)
            widget.setToolTip(
                f"Keep only images whose {METADATA_FILTERS[name].replace('_', ' ')} "
                f"is one of the selected values"
            )
            widget.itemSelectionChanged.connect(self._filters_changed)
            self.metadata_lists[name] = widget
            metadata_column.addWidget(widget)
        self.metadata_box.setVisible(False)

        # Scrolled, so eight filter lists cannot force the window taller than
        # the screen.
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.NoFrame)
        scroller.setWidget(self.metadata_box)
        column.addWidget(scroller, stretch=1)
        return box

    def _build_middle(self) -> QWidget:
        box = QGroupBox("Selection")
        column = QVBoxLayout(box)

        self.summary = QLabel("No data loaded")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.summary)

        row = QHBoxLayout()
        row.addWidget(QLabel("Break down by"))
        self.breakdown = QComboBox()
        self.breakdown.addItems(self.controller.available_breakdowns())
        self.breakdown.setCurrentText("modality")
        self.breakdown.currentTextChanged.connect(self.refresh)
        row.addWidget(self.breakdown, stretch=1)
        column.addLayout(row)

        self.table = QPlainTextEdit()
        self.table.setReadOnly(True)
        self.table.setLineWrapMode(QPlainTextEdit.NoWrap)
        column.addWidget(self.table, stretch=1)
        return box

    def _build_actions(self) -> QWidget:
        box = QGroupBox("Export")
        box.setMinimumWidth(240)
        column = QVBoxLayout(box)

        self.conditions = QCheckBox("One condition per scene")
        self.conditions.setChecked(True)
        self.conditions.setToolTip(
            "Imaging conditions of one acquisition share a label and are "
            "near-identical. Exporting all of them multiplies the dataset by "
            "copies of the same scene."
        )
        self.conditions.toggled.connect(self._filters_changed)
        column.addWidget(self.conditions)

        self.include_empty = QCheckBox("Include empty labels")
        self.include_empty.setToolTip(
            "An empty label teaches a model that the structure is background. "
            "Turn this on only to build negative examples deliberately."
        )
        self.include_empty.toggled.connect(self._filters_changed)
        column.addWidget(self.include_empty)

        column.addSpacing(8)
        for label, signal, suffix in (
            ("Training CSV...", self.training_export_requested, ""),
            ("Per-image metadata CSV...", self.image_metadata_export_requested, ".csv"),
            ("Per-project metadata CSV...", self.project_metadata_export_requested, ".csv"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, s=signal, x=suffix, t=label: self._choose(s, x, t)
            )
            column.addWidget(button)

        # Beside the buttons that write the CSVs, because it answers the
        # question they raise later: a training CSV records paths, and curating
        # the collection afterwards moves the files they point at.
        column.addSpacing(8)
        self.check_button = QPushButton("Check exported CSV...")
        self.check_button.setToolTip(
            "Read a CSV written earlier and report which recorded paths have "
            "moved or disappeared. Changes nothing."
        )
        self.check_button.clicked.connect(self._check)
        column.addWidget(self.check_button)

        column.addStretch(1)
        return box

    def _check(self) -> None:
        """Ask which CSV to check. The window does the checking."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Check exported CSV", "", "CSV files (*.csv)"
        )
        if path:
            self.csv_check_requested.emit(path)

    # -- behaviour ---------------------------------------------------------

    def _choose(self, signal, suffix: str, title: str) -> None:
        """Ask where to write, then emit. The window does the writing."""
        if suffix:
            path, _ = QFileDialog.getSaveFileName(self, title, "", "CSV files (*.csv)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose an output directory")
        if path:
            signal.emit(path)

    def _load_metadata(self) -> None:
        count = self.controller.load_metadata()
        self._populate_metadata_lists()
        self.metadata_box.setVisible(True)
        self.load_button.setText(f"{count} header(s) read")
        self.load_button.setEnabled(False)
        self.refresh()

    def _populate_metadata_lists(self) -> None:
        """Fill each filter list with the values present in the data."""
        self._updating = True
        try:
            for name, widget in self.metadata_lists.items():
                chosen = {item.text() for item in widget.selectedItems()}
                widget.clear()
                for value in self.controller.metadata_values(name):
                    item = QListWidgetItem(value)
                    widget.addItem(item)
                    item.setSelected(value in chosen)
        finally:
            self._updating = False

    def _filters_changed(self) -> None:
        if self._updating:
            return
        filters = self.controller.filters
        filters.modality = self.modality.currentText()
        filters.state = self.state.currentText()
        filters.conditions = "primary" if self.conditions.isChecked() else "all"
        filters.include_empty_labels = self.include_empty.isChecked()
        for name, widget in self.metadata_lists.items():
            self.controller.set_metadata_filter(
                name, [item.text() for item in widget.selectedItems()]
            )
        self.refresh()

    def refresh(self) -> None:
        """Recount and redraw. Cheap: the records are already in memory."""
        from ...reporting import format_counts

        if not self.controller.records:
            self.summary.setText("No data loaded. Scan a folder in the Curate tab.")
            self.table.setPlainText("")
            return

        self.summary.setText(self.controller.preview().summary())
        name = self.breakdown.currentText()
        try:
            counts = self.controller.breakdown(name)
        except ValueError as error:  # unknown breakdown, should not happen
            self.table.setPlainText(str(error))
            return
        self.table.setPlainText("\n".join(format_counts(counts, f"By {name}")))
        if self.controller.metadata_loaded():
            self._populate_metadata_lists()

    def output_directory(self) -> Path:
        return Path.home()
