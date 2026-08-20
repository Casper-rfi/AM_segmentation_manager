"""The Export tab: choose a subset, see what it holds, write it out.

The reason this is in the window rather than only on the command line: choosing
a training set is iterative. You filter, look at what survives, and adjust. The
live count in the middle is the whole point -- the empty-label rule drops images
silently on the command line, reported only in the log, and seeing what it drops
is what makes the composition of an export visible before it is written.

**Three boxes, three jobs.** Filters decides which images are in play, Breakdown
counts them, Export writes files. Nothing on the right narrows anything: a
control that shrank the export without shrinking the table would leave two
numbers on screen describing different collections.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...aggregate import AREA_BINS_PER_DECADE
from ..export_controller import (
    ANY_LABEL,
    DEFAULT_PREFIX,
    LABEL_FILTERS,
    MODALITIES,
    STATES,
    ExportController,
)
from .chart_view import ChartView
from .common import (
    StepCombo,
    caption_for,
    choose_directory,
    choose_open_file,
    choose_save_file,
    fill_grouped_combo,
    fixed_width_font,
)

logger = logging.getLogger(__name__)

#: Said wherever the pane has nothing to show, so the table and the chart give
#: the same answer rather than two phrasings of it.
NOTHING_SCANNED = "No data loaded. Scan a folder in the Curate tab."
NOTHING_MATCHES = "Nothing matches these filters"

#: Item data of the second dropdown's "(none)" row, meaning no
#: cross-tabulation. Not ``None``, which is what a group heading carries.
NO_CROSS = ""


class ExportPanel(QWidget):
    """Filters, a live count, a breakdown, and the export buttons."""

    #: Emitted when the user asks for one of the exports, with the chosen path.
    training_export_requested = Signal(str)
    image_metadata_export_requested = Signal(str)
    project_metadata_export_requested = Signal(str)

    #: Emitted with a CSV to check the recorded paths of.
    csv_check_requested = Signal(str)

    #: Emitted when the user asks for the labels to be measured. The window
    #: runs it, because it owns the worker thread and the busy dialog.
    measure_requested = Signal()

    def __init__(self, controller: ExportController) -> None:
        super().__init__()
        self.controller = controller
        self._updating = False
        # Where the last dialog left off, so writing a CSV and then checking it
        # does not mean navigating to the same folder twice. Held for the
        # session only: the window's QSettings are injected for test isolation,
        # and a panel reaching for the real ones would break that.
        self._last_directory: Path | None = None
        # Held rather than only added, so which box a control ended up in can be
        # asserted on: that is the rule the layout exists to keep.
        self._filters_box = self._build_filters()
        self._breakdown_box = self._build_middle()
        self._export_box = self._build_actions()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        layout.addWidget(self._filters_box, stretch=0)
        layout.addWidget(self._breakdown_box, stretch=1)
        layout.addWidget(self._export_box, stretch=0)

    # -- construction ------------------------------------------------------

    def _build_filters(self) -> QWidget:
        box = QGroupBox("Filters")
        box.setMinimumWidth(260)
        column = QVBoxLayout(box)

        self.modality = QComboBox()
        self.modality.addItems(MODALITIES)
        self.modality.currentTextChanged.connect(self._controls_changed)
        column.addWidget(
            caption_for(
                "Modality",
                self.modality,
                "Keep only images taken with one beam. This is the directory "
                "the image sits in, not what its header reports. Narrows the "
                "breakdown and every export.",
            )
        )
        column.addWidget(self.modality)

        self.state = QComboBox()
        self.state.addItems(STATES)
        self.state.currentTextChanged.connect(self._controls_changed)
        column.addWidget(
            caption_for(
                "Workflow state",
                self.state,
                "Keep only images in one state: selected for proofreading, "
                "deselected, or corrected once it is done. Narrows the "
                "breakdown and every export.",
            )
        )
        column.addWidget(self.state)

        self.label_state = QComboBox()
        self.label_state.addItems(LABEL_FILTERS)
        self.label_state.setCurrentText(ANY_LABEL)
        self.label_state.currentTextChanged.connect(self._controls_changed)
        column.addWidget(
            caption_for(
                "Label",
                self.label_state,
                "Keep only images whose label is in one state. A blank label "
                "is written for every image, so 'has a label' and 'annotated' "
                "are different sets. Narrows the breakdown and every export.",
            )
        )
        column.addWidget(self.label_state)

        self.conditions = QCheckBox("One image per label")
        self.conditions.setToolTip(
            "Several imaging conditions of one scene share one label. "
            "Ticked, keep one image per label rather than every condition. "
            "Narrows the breakdown and every export."
        )
        self.conditions.toggled.connect(self._controls_changed)
        column.addWidget(self.conditions)

        column.addSpacing(6)
        self.load_button = QPushButton("Load image metadata")
        self.load_button.setToolTip(
            "Read every image's header, which the acquisition filters and "
            "breakdowns need. About half a millisecond per image."
        )
        self.load_button.clicked.connect(self._load_metadata)
        column.addWidget(self.load_button)

        # One list per filterable property, populated from what the data
        # actually holds -- a free-text box would need you to know the answer
        # already. Which properties these are comes from the breakdowns, so a
        # property can be counted by and filtered on without being declared
        # twice.
        #
        # A list whose values need the headers stays hidden until they are
        # read: a column of empty boxes reads as broken, and none of them is
        # usable before the values exist to choose from.
        self.filter_box = QWidget()
        filter_column = QVBoxLayout(self.filter_box)
        filter_column.setContentsMargins(0, 0, 0, 0)
        self.filter_lists: dict[str, QListWidget] = {}
        self.filter_captions: dict[str, QLabel] = {}
        for name, described in self.controller.filterable_breakdowns():
            widget = QListWidget()
            widget.setSelectionMode(QListWidget.MultiSelection)
            widget.setMaximumHeight(64)
            # A lamella is named "project / lamella" and a stage one level
            # deeper, so these rows are long and differ at the *end*. A list
            # with a horizontal scrollbar scrolls rather than eliding, which
            # leaves a column of identical-looking project prefixes; the same
            # treatment the metadata panel's project list needed.
            widget.setTextElideMode(Qt.ElideMiddle)
            widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            widget.itemSelectionChanged.connect(self._controls_changed)
            # Only the first letter: capitalize() would lowercase the rest and
            # turn the units the label carries into "(pa)" and "(kv)".
            caption = caption_for(
                described[:1].upper() + described[1:],
                widget,
                f"Keep only images with the selected {described}. Select none "
                f"to keep them all. Narrows the breakdown and every export.",
            )
            filter_column.addWidget(caption)
            filter_column.addWidget(widget)
            self.filter_lists[name] = widget
            self.filter_captions[name] = caption

        # Scrolled, so the filter lists cannot force the window taller than the
        # screen. The scroller is what gets hidden, rather than the box inside
        # it: QScrollArea.setWidget shows whatever it is handed, and an empty
        # scroller still draws its scrollbar over the space it keeps.
        self.filter_scroller = QScrollArea()
        self.filter_scroller.setWidgetResizable(True)
        self.filter_scroller.setFrameShape(QScrollArea.NoFrame)
        self.filter_scroller.setWidget(self.filter_box)
        self.filter_scroller.setVisible(False)
        column.addWidget(self.filter_scroller, stretch=1)

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
        box = QGroupBox("Breakdown")
        column = QVBoxLayout(box)

        self.summary = QLabel("No data loaded")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.summary)

        groups = self.controller.available_breakdowns()

        row = QHBoxLayout()
        self.breakdown = StepCombo()
        fill_grouped_combo(self.breakdown, groups)
        self.breakdown.setCurrentIndex(self.breakdown.findData("modality"))
        self.breakdown.currentIndexChanged.connect(self.refresh)
        row.addWidget(
            caption_for(
                "Break down by",
                self.breakdown,
                "Group the images the filters match by one property. Counts "
                "everything they match, including images no export could "
                "write.",
            )
        )
        row.addWidget(self.breakdown, stretch=1)
        column.addLayout(row)

        row = QHBoxLayout()
        self.cross = StepCombo()
        fill_grouped_combo(self.cross, groups, first=(NO_CROSS, "(none)"))
        self.cross.currentIndexChanged.connect(self.refresh)
        row.addWidget(
            caption_for(
                "Then by",
                self.cross,
                "Cross-tabulate against a second property. The empty cells are "
                "the combinations nothing was acquired for.",
            )
        )
        row.addWidget(self.cross, stretch=1)
        column.addLayout(row)

        # Its own button, and never triggered by choosing a breakdown: opening
        # every label is the one thing in this tab slow enough that it has to
        # be asked for. What it costs is on the button before it is pressed.
        self.measure_button = QPushButton("Measure labels")
        self.measure_button.setToolTip(
            "Open every label and measure what it holds, which the Label "
            "measurements breakdowns need. Done once, and there is no way to "
            "stop it."
        )
        self.measure_button.clicked.connect(self.measure_requested)

        column.addWidget(self.measure_button)

        column.addLayout(self._build_measurement_settings())

        column.addLayout(self._build_view_choice())

        self.table = QPlainTextEdit()
        self.table.setReadOnly(True)
        # reporting.py pads its columns with spaces, which only line up in a
        # fixed-width font. The window's own font is proportional, so without
        # this the table is padded but not aligned.
        self.table.setFont(fixed_width_font())
        self.table.setLineWrapMode(QPlainTextEdit.NoWrap)

        self.chart_view = ChartView()

        # One pane showing one of the two, rather than a split: they are the
        # same numbers, and half a chart beside half a table reads worse than
        # either whole.
        self.views = QStackedWidget()
        self.views.addWidget(self.table)
        self.views.addWidget(self.chart_view)
        column.addWidget(self.views, stretch=1)
        return box

    def _build_view_choice(self) -> QHBoxLayout:
        """Build the Table/Chart switch, directly over the pane it acts on."""
        row = QHBoxLayout()
        self.as_table = QRadioButton("Table")
        self.as_chart = QRadioButton("Chart")
        self.as_table.setChecked(True)
        for button, tip in (
            (self.as_table, "Show the counts as numbers."),
            (self.as_chart, "Show the same counts as a chart, here in the pane."),
        ):
            button.setToolTip(tip)
            row.addWidget(button)
        self.as_chart.toggled.connect(self._view_changed)
        row.addStretch(1)
        return row

    def _view_changed(self, _checked: bool = False) -> None:
        """Switch the pane, drawing the chart only once it is being looked at."""
        self.views.setCurrentWidget(
            self.chart_view if self.as_chart.isChecked() else self.table
        )
        self.refresh()

    def _build_measurement_settings(self) -> QHBoxLayout:
        """Build the two controls deciding what the measured numbers mean.

        Beside *Measure labels* rather than among the filters: neither narrows
        anything. They say how what was measured is read -- how small a thing
        still counts as a crack, and how finely the area scale is cut.
        """
        row = QHBoxLayout()

        self.crack_threshold = QDoubleSpinBox()
        self.crack_threshold.setDecimals(3)
        self.crack_threshold.setRange(0.0, 10_000.0)
        self.crack_threshold.setSingleStep(0.05)
        self.crack_threshold.setValue(self.controller.min_crack_area_um2)
        self.crack_threshold.setSuffix(" um^2")
        self.crack_threshold.valueChanged.connect(self._controls_changed)
        row.addWidget(
            caption_for(
                "Crack from",
                self.crack_threshold,
                "Smallest component counted as a crack; anything smaller is "
                "left out of the crack breakdowns.",
            )
        )
        row.addWidget(self.crack_threshold)

        # A combo rather than a spin box: the supported settings are 1, 2, 3
        # and 6, and a spin box spanning them would offer 4 and 5, which are
        # not subdivisions of a decade into round edges.
        self.area_bins = StepCombo()
        for supported in AREA_BINS_PER_DECADE:
            self.area_bins.addItem(str(supported), supported)
        self.area_bins.setCurrentIndex(
            self.area_bins.findData(self.controller.area_bins_per_decade)
        )
        self.area_bins.currentIndexChanged.connect(self._controls_changed)
        row.addWidget(
            caption_for(
                "Area bins per decade",
                self.area_bins,
                "How finely the area breakdowns cut each factor of ten. "
                "Raise it when a class fills a single bin.",
            )
        )
        row.addWidget(self.area_bins)
        row.addStretch(1)
        return row

    def _build_actions(self) -> QWidget:
        box = QGroupBox("Export")
        box.setMinimumWidth(240)
        column = QVBoxLayout(box)

        column.addWidget(self._build_split())

        column.addSpacing(8)
        # The training export writes two files it names itself, so it asks for a
        # directory; the other two write one file the user names.
        for label, signal, directory, tooltip in (
            (
                "Training CSV...",
                self.training_export_requested,
                True,
                "Write the two training files into a directory you choose: "
                "one row per image and the label it is paired with.",
            ),
            (
                "Per-image metadata CSV...",
                self.image_metadata_export_requested,
                False,
                "One row per matching image: where it is, its workflow and "
                "label state, and every setting read from its header.",
            ),
            (
                "Per-project metadata CSV...",
                self.project_metadata_export_requested,
                False,
                "One row per project in the scan, from its recorded "
                "metadata. Not narrowed by the filters: the unit here is the "
                "project, not the image.",
            ),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, s=signal, d=directory, t=label: self._choose(s, t, d)
            )
            column.addWidget(button)

        # Beside the buttons that write the CSVs, because it answers the
        # question they raise later: a training CSV records paths, and a file
        # can be renamed, archived or lost between writing one and using it.
        column.addSpacing(8)
        self.check_button = QPushButton("Check exported CSV...")
        self.check_button.setToolTip(
            "Read a CSV written earlier and report which recorded paths no "
            "longer resolve. Changes nothing."
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
        self.train_share.valueChanged.connect(self._controls_changed)
        row.addWidget(
            caption_for(
                "Training",
                self.train_share,
                "Share of the exportable pairs written to the training file, "
                "the rest to validation. 100% writes no validation file.",
            )
        )
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
        self.seed.setToolTip(self.seeded.toolTip())
        self.seed.valueChanged.connect(self._controls_changed)
        row.addWidget(self.seeded)
        row.addWidget(self.seed, stretch=1)
        grid.addLayout(row)

        row = QHBoxLayout()
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText(DEFAULT_PREFIX)
        self.prefix.textChanged.connect(self._controls_changed)
        row.addWidget(
            caption_for(
                "Name",
                self.prefix,
                f"Names the two files: <prefix>_train.csv and <prefix>_val.csv. "
                f"Left empty, they are called {DEFAULT_PREFIX}_train.csv and "
                f"{DEFAULT_PREFIX}_val.csv.",
            )
        )
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
        self.controller.load_metadata()
        self._populate_filter_lists()
        self.refresh()

    def _show_metadata_state(self) -> None:
        """Spend the Load button once the headers have been read.

        Read from the controller rather than set where the button is pressed:
        measuring the labels needs the headers for the scale and keeps them, so
        the button can be spent by something other than itself.
        """
        if not self.controller.metadata_loaded():
            return
        self.load_button.setText(
            f"{len(self.controller.metadata_map)} header(s) read"
        )
        self.load_button.setEnabled(False)

    def _populate_filter_lists(self) -> None:
        """Fill each filter list with the values present in the data.

        A list is shown only when it can narrow anything: one offering a single
        value filters nothing, and this collection holds far more properties
        than any one dataset varies along. A list the user has already chosen
        from stays visible whatever it now offers, so a filter cannot become
        impossible to clear.
        """
        self._updating = True
        try:
            shown = False
            for name, widget in self.filter_lists.items():
                if self.controller.needs_metadata(name) and not self.controller.metadata_loaded():
                    self._show_filter(name, False)
                    continue
                chosen = {item.text() for item in widget.selectedItems()}
                widget.clear()
                values = self.controller.choice_values(name)
                for value in values:
                    item = QListWidgetItem(value)
                    # Elided rows lose their middle; the tooltip keeps it.
                    item.setToolTip(value)
                    widget.addItem(item)
                    item.setSelected(value in chosen)
                visible = len(values) > 1 or bool(chosen)
                self._show_filter(name, visible)
                shown = shown or visible
        finally:
            self._updating = False
        self.filter_scroller.setVisible(shown)
        # Something has to hold the column's spare height while the lists are
        # hidden, or the controls above drift apart down the column.
        self.filter_space.setVisible(not shown)

    def _show_filter(self, name: str, visible: bool) -> None:
        """Show or hide one filter list and the caption naming it."""
        self.filter_lists[name].setVisible(visible)
        self.filter_captions[name].setVisible(visible)

    def _controls_changed(self) -> None:
        """Copy every control into the controller, then recount."""
        if self._updating:
            return
        filters = self.controller.filters
        filters.modality = self.modality.currentText()
        filters.state = self.state.currentText()
        filters.label = self.label_state.currentText()
        filters.conditions = "primary" if self.conditions.isChecked() else "all"
        for name, widget in self.filter_lists.items():
            self.controller.set_choice_filter(
                name, [item.text() for item in widget.selectedItems()]
            )
        # Before the refresh below: both change what the table would say, and
        # setting the threshold re-derives the measurements when there are any.
        self.controller.set_crack_threshold(self.crack_threshold.value())
        self.controller.set_area_bins(self.area_bins.currentData())
        training = self.controller.training
        training.train_ratio = self.train_share.value() / 100
        training.seed = self.seed.value() if self.seeded.isChecked() else None
        training.prefix = self.prefix.text().strip() or DEFAULT_PREFIX
        self.refresh()

    def refresh(self) -> None:
        """Recount and redraw. Cheap: the records are already in memory."""
        if not self.controller.records:
            self.summary.setText(NOTHING_SCANNED)
            self.table.setPlainText("")
            self.chart_view.show_message(NOTHING_SCANNED)
            return

        self.summary.setText(self.controller.preview().summary())
        self._show_measure_state()
        self._show_metadata_state()
        try:
            lines = self._counted()
        except ValueError as error:  # unknown breakdown, should not happen
            self.table.setPlainText(str(error))
            return
        self.table.setPlainText("\n".join(lines))
        if self.as_chart.isChecked():
            self._draw_inline_chart()
        self._populate_filter_lists()

    def _draw_inline_chart(self) -> None:
        """Draw the current breakdown into the pane, or say why it cannot be.

        The chart type follows the data rather than being chosen: one property
        is bars, two are a grid. Asking would be asking the user to know which
        shape suits a shape they have not seen yet.
        """
        from ...plots import bar_figure, has_plot_support, heatmap_figure

        if not has_plot_support():
            from ...plots import INSTALL_HINT

            self.chart_view.show_message(
                f"Drawing charts requires matplotlib.\n\n{INSTALL_HINT}"
            )
            return
        if not self.controller.records:
            self.chart_view.show_message(NOTHING_SCANNED)
            return

        breakdowns = self.controller.breakdowns
        name = self.breakdown.currentData()
        other = self.cross.currentData()
        unmeasured = [
            chosen for chosen in (name, other)
            if chosen and self.controller.needs_measurements(chosen)
        ]
        if unmeasured and not self.controller.measurements_loaded():
            self.chart_view.show_message(
                "This breakdown needs the labels measured first."
            )
            return

        row = breakdowns[name].label
        if other:
            table = self.controller.cross_tabulation(name, other)
            if not table:
                self.chart_view.show_message(NOTHING_MATCHES)
                return
            column = breakdowns[other].label
            title = f"Images by {row} and {column}"
            self.chart_view.show_chart(
                lambda normalise: heatmap_figure(
                    table, row, column, title=title, normalise=normalise,
                    row_order=breakdowns[name].order,
                    column_order=breakdowns[other].order,
                ),
                normalisable=True,
            )
            return

        counts = self.controller.breakdown(name)
        if not counts:
            self.chart_view.show_message(NOTHING_MATCHES)
            return
        title = f"Images by {row}"
        self.chart_view.show_chart(
            lambda _normalise: bar_figure(
                counts, title, xlabel=row, ordered=bool(breakdowns[name].order),
            )
        )

    def _show_measure_state(self) -> None:
        """Put the size of the job on the button, or say it is done.

        The cost scales with the collection and there is nothing to cancel, so
        how much is being asked for belongs on the button rather than in a
        dialog that appears once it is too late.
        """
        if self.controller.measurements_loaded():
            self.measure_button.setText("Labels measured")
            self.measure_button.setEnabled(False)
            return
        labels = self.controller.measurable_labels()
        self.measure_button.setText(f"Measure {labels} label(s)")
        self.measure_button.setEnabled(bool(labels))

    def _counted(self) -> list[str]:
        """Return the table for whichever of the two dropdowns are in use."""
        from ...reporting import format_counts, format_cross_tabulation

        breakdowns = self.controller.breakdowns
        name = self.breakdown.currentData()
        row = breakdowns[name].label
        other = self.cross.currentData()

        unmeasured = [
            breakdowns[chosen].label
            for chosen in (name, other)
            if chosen and self.controller.needs_measurements(chosen)
        ]
        if unmeasured and not self.controller.measurements_loaded():
            # Counting anyway would put every image under "no-label", which
            # reads as a finding rather than as a question not yet asked.
            return [
                f"Counting by {' and '.join(unmeasured)} needs the labels "
                f"measured.",
                "",
                "Press Measure labels above.",
            ]

        if not other:
            counts = self.controller.breakdown(name)
            return format_counts(counts, f"Images by {row}")

        column = breakdowns[other].label
        table = self.controller.cross_tabulation(name, other)
        return format_cross_tabulation(
            table,
            row,
            column,
            title=f"Images by {row} and {column}",
            row_order=breakdowns[name].order,
            column_order=breakdowns[other].order,
        )
