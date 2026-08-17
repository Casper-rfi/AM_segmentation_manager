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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...filtering import METADATA_FILTERS, filter_property
from ..export_controller import (
    DEFAULT_PREFIX,
    MODALITIES,
    STATES,
    ExportController,
)
from .common import choose_directory, choose_open_file, choose_save_file

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
        # Where the last dialog left off, so writing a CSV and then checking it
        # does not mean navigating to the same folder twice. Held for the
        # session only: the window's QSettings are injected for test isolation,
        # and a panel reaching for the real ones would break that.
        self._last_directory: Path | None = None

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
        self.modality.currentTextChanged.connect(self._controls_changed)
        column.addWidget(QLabel("Modality"))
        column.addWidget(self.modality)

        self.state = QComboBox()
        self.state.addItems(STATES)
        self.state.currentTextChanged.connect(self._controls_changed)
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
        # Hidden until the headers are read. A column of empty boxes reads as
        # broken, and they are not usable before the values exist to choose from.
        self.metadata_box = QWidget()
        metadata_column = QVBoxLayout(self.metadata_box)
        metadata_column.setContentsMargins(0, 0, 0, 0)
        self.metadata_lists: dict[str, QListWidget] = {}
        for name in sorted(METADATA_FILTERS):
            described = filter_property(name)
            # Only the first letter: capitalize() would lowercase the rest and
            # turn the units it carries into "(pa)" and "(kv)".
            metadata_column.addWidget(QLabel(described[:1].upper() + described[1:]))
            widget = QListWidget()
            widget.setSelectionMode(QListWidget.MultiSelection)
            widget.setMaximumHeight(64)
            widget.setToolTip(
                f"Keep only images whose {filter_property(name)} is one of the "
                f"selected values"
            )
            widget.itemSelectionChanged.connect(self._controls_changed)
            self.metadata_lists[name] = widget
            metadata_column.addWidget(widget)

        # Scrolled, so the filter lists cannot force the window taller than the
        # screen. The scroller is what gets hidden, rather than the box inside
        # it: QScrollArea.setWidget shows whatever it is handed, and an empty
        # scroller still draws its scrollbar over the space it keeps.
        self.metadata_scroller = QScrollArea()
        self.metadata_scroller.setWidgetResizable(True)
        self.metadata_scroller.setFrameShape(QScrollArea.NoFrame)
        self.metadata_scroller.setWidget(self.metadata_box)
        self.metadata_scroller.setVisible(False)
        column.addWidget(self.metadata_scroller, stretch=1)

        # Holds the column's spare height while the lists are hidden. A layout
        # shares that height between whatever is left, so without something to
        # take it the two combo boxes above drift apart down the column. A
        # widget rather than a stretch, because a stretch cannot be hidden and
        # would go on competing with the lists once they appear.
        self.filter_space = QWidget()
        self.filter_space.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        column.addWidget(self.filter_space, stretch=1)
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
        self.conditions.toggled.connect(self._controls_changed)
        column.addWidget(self.conditions)

        self.include_empty = QCheckBox("Include empty labels")
        self.include_empty.setToolTip(
            "An empty label teaches a model that the structure is background. "
            "Turn this on only to build negative examples deliberately."
        )
        self.include_empty.toggled.connect(self._controls_changed)
        column.addWidget(self.include_empty)

        column.addSpacing(8)
        column.addWidget(self._build_split())

        column.addSpacing(8)
        # The training export writes two files it names itself, so it asks for a
        # directory; the other two write one file the user names.
        for label, signal, directory in (
            ("Training CSV...", self.training_export_requested, True),
            ("Per-image metadata CSV...", self.image_metadata_export_requested, False),
            ("Per-project metadata CSV...", self.project_metadata_export_requested, False),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, s=signal, d=directory, t=label: self._choose(s, t, d)
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

    def _build_split(self) -> QWidget:
        """Build the controls for how a training export is divided and named.

        These belong beside the training button rather than among the filters:
        they change nothing about which images are exported, only how the two
        files are written. The preview above shows what they come out as.
        """
        box = QGroupBox("Training split")
        grid = QVBoxLayout(box)

        row = QHBoxLayout()
        self.train_share = QSpinBox()
        self.train_share.setRange(0, 100)
        self.train_share.setValue(80)
        self.train_share.setSuffix(" %")
        self.train_share.setToolTip(
            "Share of the exportable pairs written to the training file, the "
            "rest to validation. 100% writes no validation file."
        )
        self.train_share.valueChanged.connect(self._controls_changed)
        row.addWidget(QLabel("Training"))
        row.addWidget(self.train_share, stretch=1)
        grid.addLayout(row)

        row = QHBoxLayout()
        self.seeded = QCheckBox("Seed")
        self.seeded.setChecked(True)
        self.seeded.setToolTip(
            "Shuffle reproducibly, so the same selection and seed always give "
            "the same two files. Clear it to shuffle differently every time."
        )
        self.seeded.toggled.connect(self._seeding_toggled)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(42)
        self.seed.valueChanged.connect(self._controls_changed)
        row.addWidget(self.seeded)
        row.addWidget(self.seed, stretch=1)
        grid.addLayout(row)

        row = QHBoxLayout()
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText(DEFAULT_PREFIX)
        self.prefix.setToolTip(
            f"Names the two files: <prefix>_train.csv and <prefix>_val.csv. "
            f"Left empty, they are called {DEFAULT_PREFIX}_train.csv and "
            f"{DEFAULT_PREFIX}_val.csv."
        )
        self.prefix.textChanged.connect(self._controls_changed)
        row.addWidget(QLabel("Name"))
        row.addWidget(self.prefix, stretch=1)
        grid.addLayout(row)
        return box

    def _seeding_toggled(self, seeded: bool) -> None:
        """Grey out the seed when the shuffle is not to be reproducible."""
        self.seed.setEnabled(seeded)
        self._controls_changed()

    def _check(self) -> None:
        """Ask which CSV to check. The window does the checking."""
        path = choose_open_file(self, "Check exported CSV", self._last_directory)
        if path is not None:
            self._remember(path)
            self.csv_check_requested.emit(str(path))

    # -- behaviour ---------------------------------------------------------

    def _choose(self, signal, title: str, directory: bool) -> None:
        """Ask where to write, then emit. The window does the writing."""
        if directory:
            path = choose_directory(self, "Choose an output directory", self._last_directory)
        else:
            path = choose_save_file(self, title, self._last_directory)
        if path is not None:
            self._remember(path)
            signal.emit(str(path))

    def _remember(self, path: Path) -> None:
        """Open the next dialog where this one left off."""
        self._last_directory = path if path.is_dir() else path.parent

    def _load_metadata(self) -> None:
        count = self.controller.load_metadata()
        self._populate_metadata_lists()
        self.metadata_scroller.setVisible(True)
        self.filter_space.setVisible(False)
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

    def _controls_changed(self) -> None:
        """Copy every control into the controller, then recount."""
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
        training = self.controller.training
        training.train_ratio = self.train_share.value() / 100
        training.seed = self.seed.value() if self.seeded.isChecked() else None
        training.prefix = self.prefix.text().strip() or DEFAULT_PREFIX
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
