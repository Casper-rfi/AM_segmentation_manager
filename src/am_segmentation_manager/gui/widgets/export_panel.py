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
from pathlib import Path, PurePath

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import plots
from ...aggregate import AREA_BINS_PER_DECADE, bin_mantissas
from ...layout import SERIES, SERIES_LABELS
from ...plots import INSTALL_HINT
from ...reporting import format_counts, format_cross_tabulation
from ..export_controller import (
    ANY_LABEL,
    DEFAULT_PREFIX,
    DEFAULT_SERIES,
    LABEL_FILTERS,
    MODALITIES,
    PROJECT_FILTER,
    STATES,
    ExportController,
)
from .chart_view import ChartView
from .common import (
    caption_for,
    fill_grouped_combo,
    fixed_width_font,
    horizontal_rule,
    ticked_caption,
)
from .controls import GripSplitter, SegmentedControl, StepCombo, ToggleSwitch
from .lists import CheckList, add_check_all_buttons, project_box
from .paths import choose_directory, choose_open_file, choose_save_file
from .theme import COMPACT_BUTTON

logger = logging.getLogger(__name__)

#: Said wherever the pane has nothing to show, so the table and the chart give
#: the same answer rather than two phrasings of it.
NOTHING_SCANNED = "No data loaded. Scan a folder in the Curate tab."
NOTHING_MATCHES = "Nothing matches these filters"

#: Item data of the second dropdown's "(none)" row, meaning no
#: cross-tabulation. Not ``None``, which is what a group heading carries.
NO_CROSS = ""

#: How tall the project box may shrink to, in pixels. A minimum rather than a
#: fixed height: it shares the Filters column with the rows below it, and which
#: of the two wants the room is the reader's to decide.
PROJECT_BOX_MIN_HEIGHT = 150

#: How tall a filter list is at least, in pixels. Four ticked rows: enough to
#: see that a list has been narrowed without any one property taking the
#: column, and the caption says how many are ticked whether or not they are in
#: view. A floor rather than a cap, the rows sharing a splitter with the
#: projects above them: a reader who wants to see thirty values at once should
#: be able to drag the room to do it.
FILTER_LIST_HEIGHT = 84

#: Item data of a filter row's "(choose a property)" entry, meaning the row is
#: there but filtering nothing yet. Not ``None``, which is what a group heading
#: carries.
NO_PROPERTY = ""

#: What that entry reads as. A row starts on it, so the box shows what it is
#: for before it does anything.
CHOOSE_PROPERTY = "(choose a property)"

#: What the button removing a filter is labelled. U+2715, which every font here
#: carries; the button is named :data:`COMPACT_BUTTON` so the stylesheet's
#: usual padding does not clip it away.
REMOVE_GLYPH = "\u2715"


def _folder_of(value: str) -> str:
    """Return the directory part of a relative path, empty at the root."""
    parent = PurePath(value).parent
    return "" if str(parent) == "." else str(parent)


def _name_of(value: str) -> str:
    """Return the last part of a relative path, the folder having said the rest."""
    return PurePath(value).name


class FilterRow(QWidget):
    """One filter: which property it is on, and which of its values are kept.

    A row the user adds rather than one list per property shown at once. The
    breakdowns cover far more properties than any one export is chosen by, so a
    column holding every list left too little of each to read, and the three
    worth the most space were buried among a dozen that are never touched.

    The value list appears only once a property is chosen: an empty box under
    an empty dropdown reads as broken rather than as unused.
    """

    #: Emitted when what this row keeps has changed -- its property or its ticks.
    changed = Signal()

    #: Emitted with this row when the user asks for it to go.
    removed = Signal(object)

    def __init__(self, controller: ExportController) -> None:
        super().__init__()
        self.controller = controller
        self._filling = False
        #: The property this row is on, so the one it leaves can be cleared.
        #: The combo cannot answer that -- by the time it reports a change it
        #: already holds the new one.
        self._property = NO_PROPERTY

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 4)
        column.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.property_box = StepCombo()
        self.property_box.setToolTip(
            "Which property this filter is on. Every property here can also be "
            "counted by, under Break down by."
        )
        self.property_box.currentIndexChanged.connect(self._property_chosen)
        row.addWidget(self.property_box, stretch=1)

        self.remove_button = QPushButton(REMOVE_GLYPH)
        self.remove_button.setObjectName(COMPACT_BUTTON)
        self.remove_button.setToolTip(
            "Take this filter away, and with it the values it was keeping."
        )
        self.remove_button.setFixedWidth(26)
        self.remove_button.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(self.remove_button)
        column.addLayout(row)

        self.count = QLabel("")
        self.count.setVisible(False)
        column.addWidget(self.count)

        self.values = CheckList()
        self.values.setMinimumHeight(FILTER_LIST_HEIGHT)
        # A lamella is named "project / lamella" and a stage one level deeper,
        # so these rows are long and differ at the *end*. A list with a
        # horizontal scrollbar scrolls rather than eliding, which leaves a
        # column of identical-looking prefixes.
        self.values.setTextElideMode(Qt.ElideMiddle)
        self.values.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.values.setToolTip(
            "Keep only images with the ticked values. Tick none to keep them "
            "all. Narrows the breakdown and every export."
        )
        self.values.row_ticked.connect(lambda _row, _on: self.changed.emit())
        self.values.setVisible(False)
        column.addWidget(self.values)

        # Under the list rather than beside the property box: they act on what
        # is above them, and a row on no property yet has nothing to tick.
        self.buttons = QWidget()
        button_row = QHBoxLayout(self.buttons)
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(4)
        add_check_all_buttons(button_row, self._tick_all, "values")
        button_row.addStretch(1)
        self.buttons.setVisible(False)
        column.addWidget(self.buttons)

    # -- what this row is on -----------------------------------------------

    @property
    def property_name(self) -> str:
        """Return the breakdown this row filters on, or "" for none yet."""
        return self._property

    def ticked_values(self) -> list[str]:
        """Return the values this row keeps. Empty means it narrows nothing."""
        return self.values.ticked_values()

    def _tick_all(self, wanted: bool) -> None:
        """Tick or clear every value this row offers.

        Reported once rather than per row: the list is what the caller asked
        about, and a signal per value would sweep the collection for each.
        """
        self.values.show_ticks([wanted] * self.values.count())
        self.changed.emit()

    def offer(self, groups, taken: set[str]) -> None:
        """Fill the dropdown with the properties this row may be put on.

        Args:
            groups: ``(heading, rows)`` from
                :meth:`ExportController.filterable_groups`.
            taken: Properties another row already holds. Left out, so one
                property is filtered in one place and two rows cannot disagree
                about it.
        """
        self._filling = True
        try:
            offered = [
                (
                    heading,
                    [row for row in rows if row[0] not in taken],
                )
                for heading, rows in groups
            ]
            fill_grouped_combo(
                self.property_box,
                [(heading, rows) for heading, rows in offered if rows],
                first=(NO_PROPERTY, CHOOSE_PROPERTY),
            )
            index = self.property_box.findData(self._property)
            self.property_box.setCurrentIndex(max(index, 0))
        finally:
            self._filling = False

    def _property_chosen(self) -> None:
        """Move this row onto another property, clearing the one it leaves."""
        if self._filling:
            return
        chosen = self.property_box.currentData() or NO_PROPERTY
        if chosen == self._property:
            return
        self.forget()
        self._property = chosen
        self.changed.emit()

    def forget(self) -> None:
        """Clear whatever this row was keeping, at the property it was on.

        Called when the row changes property and when it is taken away: a
        filter left behind by a control that has gone narrows every count in
        the tab with nothing on screen to say so.
        """
        if self._property != NO_PROPERTY:
            self.controller.set_choice_filter(self._property, [])
        self._property = NO_PROPERTY

    # -- what it offers ----------------------------------------------------

    def show_values(self, values: list[str], chosen: set[str], label: str) -> None:
        """Put the property's values in the list, keeping what was ticked."""
        self._filling = True
        try:
            self.values.set_values(
                values,
                ticked=[value in chosen for value in values],
                # Elided rows lose their middle; the tooltip keeps it.
                tooltips=values,
            )
        finally:
            self._filling = False
        self.count.setText(ticked_caption(label, len(values), len(chosen)))

    def show_empty(self) -> None:
        """Hide the list, for a row not on a property yet."""
        self.values.setVisible(False)
        self.buttons.setVisible(False)
        self.count.setVisible(False)

    def show_filled(self) -> None:
        """Show the list and the buttons acting on it."""
        self.values.setVisible(True)
        self.buttons.setVisible(True)
        self.count.setVisible(True)

    def untick(self) -> None:
        """Untick every value, without changing which property this is on."""
        self._filling = True
        try:
            self.values.show_ticks([])
        finally:
            self._filling = False


class ExportPanel(QWidget):
    """Filters, a live count, a breakdown, and the export buttons."""

    #: Emitted when the user asks for one of the exports, with the chosen path.
    training_export_requested = Signal(str)
    image_metadata_export_requested = Signal(str)
    project_metadata_export_requested = Signal(str)
    #: Emitted with the directory to copy the matching files into.
    files_export_requested = Signal(str)
    #: Emitted with the copy to bring back when the user asks to merge.
    merge_requested = Signal(str)

    #: Emitted with a CSV to check the recorded paths of.
    csv_check_requested = Signal(str)

    #: Emitted when the user asks for the labels to be measured. The window
    #: runs it, because it owns the worker thread and the busy dialog.
    measure_requested = Signal()

    def __init__(self, controller: ExportController) -> None:
        super().__init__()
        self.controller = controller
        self._updating = False
        #: How the tab catches up after a filter row changes.
        #:
        #: **A tick may not be acted on while it is still being delivered.**
        #: Catching up refills every filter row, and refilling one clears the
        #: list or tree it is drawn in -- so acting on a tick straight away
        #: destroys the very item whose signal is being emitted, while Qt is
        #: still propagating the change to that folder's other children.
        #: Ticking a folder of nine projects left one of them ticked and
        #: corrupted the heap on the way, which shows as a crash a few gestures
        #: later rather than there and then. It is the rule the curation lists
        #: already keep, reached through a timer because the tick arrives from
        #: Qt rather than from this class.
        #:
        #: Zero-delay and single-shot, so a gesture that moves many controls at
        #: once recounts once: ticking that folder is nine signals, each of
        #: which would otherwise sweep the whole collection's records.
        self._recount = QTimer(self)
        self._recount.setSingleShot(True)
        self._recount.setInterval(0)
        self._recount.timeout.connect(self._controls_changed)
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
        # In a splitter, as the other two tabs divide their columns: how much
        # of the width a long project path or a wide table wants is the
        # reader's to decide, and this tab was the one that decided for them.
        # The window saves where the dividers were left, by the same name.
        self.columns = GripSplitter(Qt.Horizontal)
        self.columns.addWidget(self._filters_box)
        self.columns.addWidget(self._breakdown_box)
        self.columns.addWidget(self._export_box)
        self.columns.setStretchFactor(1, 1)
        # The middle is the point of this tab, so the filter column opens
        # narrower than Curate's and the project box's buttons take two lines
        # there. Dragged wider they take one; clipped they never are.
        self.columns.setSizes([300, 900, 280])
        layout.addWidget(self.columns)

    def apply_defaults(self, values) -> None:
        """Open this tab's controls at what the settings say. See `ImagePanel`.

        Set while `_updating` holds, so filling eight controls asks for one
        recount at the end rather than eight sweeps of the collection.
        """
        from .. import preferences

        self._updating = True
        try:
            wanted = preferences.series_from(values)
            for name, box in self.series_boxes.items():
                box.setChecked(name in wanted)
            self.conditions.setChecked(bool(values["one_per_label"]))
            self.train_share.setValue(int(values["train_share"]))
            self.seeded.setChecked(bool(values["seeded"]))
            self.seed.setValue(int(values["seed"]))
            self.seed.setEnabled(bool(values["seeded"]))
        finally:
            self._updating = False
        self._recount.start()

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

        # Checkboxes rather than a dropdown: the three are independent, and an
        # export covering two of them is a real request. The only filter that
        # does not start on everything -- see export_controller.DEFAULT_SERIES.
        self.series_boxes: dict[str, ToggleSwitch] = {}
        series_holder = QWidget()
        series_column = QVBoxLayout(series_holder)
        series_column.setContentsMargins(0, 0, 0, 0)
        series_column.setSpacing(0)
        for name in SERIES:
            checkbox = ToggleSwitch(SERIES_LABELS[name])
            checkbox.setChecked(name in DEFAULT_SERIES)
            checkbox.toggled.connect(self._controls_changed)
            self.series_boxes[name] = checkbox
            series_column.addWidget(checkbox)
        column.addWidget(
            caption_for(
                "Series",
                series_holder,
                "What the images were taken for. Only Adaptive Polishing is "
                "ticked to start with, so an export covers what it always "
                "did. Narrows the breakdown and every export.",
            )
        )
        column.addWidget(series_holder)

        self.conditions = ToggleSwitch("One image per label")
        self.conditions.setToolTip(
            "Several imaging conditions of one scene share one label. "
            "Ticked, keep one image per label rather than every condition. "
            "Narrows the breakdown and every export."
        )
        self.conditions.toggled.connect(self._controls_changed)
        column.addWidget(self.conditions)

        column.addSpacing(6)

        # Filters are added as they are wanted rather than all shown at once.
        # Which properties are on offer comes from the breakdowns, so a property
        # can be counted by and filtered on without being declared twice.
        self.filter_box = QWidget()
        self.filter_column = QVBoxLayout(self.filter_box)
        self.filter_column.setContentsMargins(0, 0, 0, 0)
        self.filter_column.setSpacing(0)
        self.filter_rows: list[FilterRow] = []

        # The scroller stretches this column to its own height, and a layout
        # shares the surplus between whatever is in it -- which pushed every
        # row a long way from the list it names. The spare height goes to the
        # bottom instead, into a stretch, which unlike a widget cannot itself
        # be hidden again.
        self.filter_column.addStretch(1)

        # Scrolled, so the filter rows cannot force the window taller than the
        # screen. setWidget shows whatever it is handed, so a widget hidden
        # before it goes in comes back on screen: hide the scroller rather than
        # what it holds.
        self.filter_scroller = QScrollArea()
        self.filter_scroller.setWidgetResizable(True)
        self.filter_scroller.setFrameShape(QScrollArea.NoFrame)
        self.filter_scroller.setWidget(self.filter_box)

        # The projects and the rows share this column, and which of the two
        # wants the room depends on what is being looked for: thirty sessions
        # to scroll, or four filters each with a long list. So they are given a
        # divider rather than a ratio decided here.
        self.filters = GripSplitter(Qt.Vertical)
        self.filters.addWidget(self._build_projects())
        self.filters.addWidget(self.filter_scroller)
        self.filters.setSizes([260, 340])
        column.addWidget(self.filters, stretch=1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.add_button = QPushButton("Add filter")
        self.add_button.setToolTip(
            "Add another filter, on a property not already filtered on."
        )
        self.add_button.clicked.connect(self._add_filter)
        row.addWidget(self.add_button)

        # Under the rows rather than over them: it undoes what they hold, and a
        # control that clears something reads best after the thing it clears.
        # It unticks rather than removing, so the filters stay where they were
        # put -- clearing what a filter keeps and deciding not to filter on it
        # at all are different intentions, and the remove button is the other.
        self.clear_button = QPushButton("Clear all filters")
        self.clear_button.setToolTip(
            "Untick every value in the filters above, leaving the filters "
            "themselves in place. The controls at the top of the box are not "
            "touched -- they always hold one of their choices."
        )
        self.clear_button.clicked.connect(self._clear_filters)
        row.addWidget(self.clear_button)
        column.addLayout(row)

        # One to begin with, on nothing: the box says what it offers without
        # spending the column on lists nobody asked for.
        self._add_filter()
        return box

    def _build_projects(self) -> QWidget:
        """Build the project filter, which every other tab draws as a list too.

        Shown rather than added, and with no button to take it away: which
        projects an export covers is the first question anyone asks of one, and
        it was a row somebody had to know to go and add -- indistinguishable,
        once added, from a filter on the microscope or the sample.

        **Every row ticked is written as no filter at all.** The values on
        offer are recomputed from the records the other filters pass, so a
        literal list of all of them would quietly exclude a project that
        appeared afterwards -- from a rescan, or from widening another filter.
        Opening with everything ticked says plainly that nothing is excluded;
        `_projects_changed` is what keeps that true underneath.
        """
        box, self.project_filter, self.project_tree = project_box(
            "Autolamella projects",
            on_check_all=self._tick_all_projects,
            tooltip=(
                "Which projects the counts and every export cover, under the "
                "folder each sits in. All of them to start with; untick to "
                "narrow."
            ),
        )
        box.setMinimumHeight(PROJECT_BOX_MIN_HEIGHT)
        self.project_filter.textChanged.connect(self.project_tree.hide_rows_without)
        # Queued through the same timer the filter rows use: a tick arrives
        # from Qt while it is still being delivered to the folder's other
        # children, and catching up rebuilds the very tree it came from.
        self.project_tree.row_ticked.connect(lambda _row, _on: self._recount.start())
        #: The relative paths behind the tree's rows, which are addressed by
        #: number. Held because a row says only its own name, the folder above
        #: it saying the rest.
        self._project_values: list[str] = []
        return box

    def _tick_all_projects(self, wanted: bool) -> None:
        """Tick or clear every project, then catch up once."""
        self.project_tree.show_ticks([wanted] * len(self._project_values))
        self._recount.start()

    def _show_projects(self) -> None:
        """List the projects the rest of the selection still holds.

        What is ticked comes from the tree rather than from the controller,
        the way a filter row's does: the two agree except while somebody is
        part-way through a gesture, and re-deriving would undo them mid-way.
        With everything ticked the answer is everything, so a project that
        appears later -- from a rescan, or from widening another filter --
        joins the export rather than being silently left out of it.
        """
        values = self.controller.choice_values(PROJECT_FILTER)
        if self._all_projects_ticked():
            ticked = [True] * len(values)
        else:
            held = {
                self._project_values[row]
                for row in self.project_tree.ticked_rows()
                if row is not None and row < len(self._project_values)
            }
            ticked = [value in held for value in values]
        self._project_values = list(values)
        self.project_tree.set_projects(
            [_folder_of(value) for value in values],
            [_name_of(value) for value in values],
            ticked=ticked,
            tooltips=values,
        )

    def _all_projects_ticked(self) -> bool:
        """Return whether every project on offer is ticked, an empty tree included.

        An empty one counts as everything: a tab that has not scanned yet
        excludes nothing, and reading it as "none of them" would open the
        Export tab counting zero images.
        """
        return len(self.project_tree.ticked_rows()) == len(self._project_values)

    def _projects_changed(self) -> None:
        """Copy the ticked projects into the controller.

        Every row ticked is stored as no filter, which is what it means: see
        :meth:`_build_projects`. None ticked is stored as an empty choice,
        which matches nothing -- unlike a filter row, where ticking none of a
        property's values is how you stop filtering on it.
        """
        ticked = [
            self._project_values[row]
            for row in self.project_tree.ticked_rows()
            if row is not None and row < len(self._project_values)
        ]
        if self._all_projects_ticked():
            self.controller.set_choice_filter(PROJECT_FILTER, [])
            return
        self.controller.set_choice_filter(
            PROJECT_FILTER, ticked, empty_keeps_everything=False
        )

    # -- the filter rows ---------------------------------------------------

    def _add_filter(self) -> None:
        """Put another filter row above the Add button."""
        row = FilterRow(self.controller)
        # Queued through the timer, never called straight from the row's
        # signal: see _recount.
        row.changed.connect(self._recount.start)
        row.removed.connect(self._remove_filter)
        # Before the trailing stretch, which is what holds the spare height.
        self.filter_column.insertWidget(len(self.filter_rows), row)
        self.filter_rows.append(row)
        self._refresh_filter_rows()
        self._show_add_state()

    def _remove_filter(self, row: FilterRow) -> None:
        """Take a filter away, and with it whatever it was keeping.

        The last one is emptied rather than removed. A box with no filter in it
        still says what could be filtered on; no box at all leaves the column
        blank with nothing to explain what is missing.
        """
        row.forget()
        if len(self.filter_rows) == 1:
            row.property_box.setCurrentIndex(0)
        else:
            self.filter_rows.remove(row)
            self.filter_column.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._controls_changed()
        self._show_add_state()

    def _show_add_state(self) -> None:
        """Grey out Add once every property already has a filter of its own."""
        used = sum(1 for row in self.filter_rows if row.property_name)
        self.add_button.setEnabled(
            used < len(self.controller.filterable_breakdowns())
        )



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
        """Build the Table/Chart switch, directly over the pane it acts on.

        A segmented run like every other one-of-a-set in the window: two ways
        of showing the same numbers, both worth having on screen.
        """
        row = QHBoxLayout()
        self.view_control = SegmentedControl([
            ("table", "Table", "Show the counts as numbers."),
            (
                "chart", "Chart",
                "Show the same counts as a chart, here in the pane.",
            ),
        ])
        #: The segments, so a caller can reach one at a time.
        self.as_table = self.view_control.buttons["table"]
        self.as_chart = self.view_control.buttons["chart"]
        self.view_control.set_current("table")
        self.view_control.chosen.connect(self._view_changed)
        row.addWidget(self.view_control)
        row.addStretch(1)
        return row

    def _view_changed(self, _chosen: str = "") -> None:
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
        #
        # Each row carries the values it cuts at rather than only the count.
        # "3" is a number to look up; "3 per decade (1, 2, 5)" is a scale a
        # reader recognises, and the choice is between scales.
        self.area_bins = StepCombo()
        for supported in AREA_BINS_PER_DECADE:
            cuts = ", ".join(f"{value:g}" for value in bin_mantissas(supported))
            self.area_bins.addItem(f"{supported} per decade ({cuts})", supported)
        self.area_bins.setCurrentIndex(
            self.area_bins.findData(self.controller.area_bins_per_decade)
        )
        self.area_bins.currentIndexChanged.connect(self._controls_changed)
        row.addWidget(
            caption_for(
                "Area bins",
                self.area_bins,
                "How finely each factor of ten is cut. Raising it splits a bin "
                "rather than moving its edges, so counts stay comparable "
                "across settings.",
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

        # Under a rule and apart from the four above, which all write or read a
        # list of paths. These two move the images and labels themselves, and
        # reading either as a fifth kind of CSV is how a copy of gigabytes gets
        # started by accident.
        column.addSpacing(10)
        column.addWidget(horizontal_rule())
        column.addSpacing(10)
        self.files_button = QPushButton("Export images and labels...")
        self.files_button.setToolTip(
            "Copy the matching images, their labels and each project's "
            "files into a folder, keeping the hierarchy.\n"
            "Nothing is overwritten; an earlier run of it resumes."
        )
        self.files_button.clicked.connect(
            lambda: self._choose(
                self.files_export_requested, self.files_button.text(), True
            )
        )
        column.addWidget(self.files_button)

        # Directly under the export it undoes: a copy goes out here and comes
        # back here, and putting the two together is what makes the round trip
        # read as one thing rather than two unrelated buttons.
        self.merge_button = QPushButton("Merge a copy back...")
        self.merge_button.setToolTip(
            "Bring the work done in a copy back into this collection.\n"
            "Nothing is merged while a conflict stands, and proofread work "
            "here is never written over unasked."
        )
        self.merge_button.clicked.connect(self._choose_copy)
        column.addWidget(self.merge_button)

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
        self.seeded = ToggleSwitch("Seed")
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

    def _choose_copy(self) -> None:
        """Ask which copy to bring back. It is read, not written to."""
        path = choose_directory(
            self, "Choose the copy to merge back", self._last_directory
        )
        if path is not None:
            self._remember(path)
            self.merge_requested.emit(str(path))

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

    def _refresh_filter_rows(self) -> None:
        """Offer each row its choices, and fill the one it is on.

        The values are those of the records passing every *other* filter, which
        the controller works out -- so narrowing one filter does not empty the
        list of the one being used. Only the properties actually in a row are
        asked for, so a column holding two filters costs two passes rather than
        one per property in the collection.
        """
        self._updating = True
        try:
            self._show_projects()
            groups = self.controller.filterable_groups()
            held = {row.property_name for row in self.filter_rows if row.property_name}
            for row in self.filter_rows:
                row.offer(groups, held - {row.property_name})
                name = row.property_name
                if not name:
                    row.show_empty()
                    continue
                chosen = set(row.ticked_values())
                row.show_values(
                    self.controller.choice_values(name),
                    chosen,
                    self._noun(name),
                )
                row.show_filled()
        finally:
            self._updating = False

    def _noun(self, name: str) -> str:
        """Return a property's label as a caption starts it.

        Only the first letter is raised: ``capitalize()`` would lowercase the
        rest and turn the units a label carries into "(pa)" and "(kv)".
        """
        label = self.controller.breakdowns[name].label
        return label[:1].upper() + label[1:]

    def _clear_filters(self) -> None:
        """Untick every filter's values, then recount once.

        The projects are put back to *all* ticked rather than to none: there,
        every row ticked is what "not filtered" looks like, so clearing them
        would narrow the export to nothing at the press of a button called
        Clear.
        """
        self._updating = True
        try:
            for row in self.filter_rows:
                row.untick()
            self.project_tree.show_ticks([True] * len(self._project_values))
        finally:
            self._updating = False
        self._controls_changed()

    def _controls_changed(self) -> None:
        """Copy every control into the controller, then recount."""
        if self._updating:
            return
        filters = self.controller.filters
        filters.modality = self.modality.currentText()
        filters.state = self.state.currentText()
        filters.label = self.label_state.currentText()
        filters.series = tuple(
            name for name, box in self.series_boxes.items() if box.isChecked()
        )
        filters.conditions = "primary" if self.conditions.isChecked() else "all"
        self._projects_changed()
        for row in self.filter_rows:
            if row.property_name:
                self.controller.set_choice_filter(row.property_name, row.ticked_values())
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
        try:
            lines = self._counted()
        except ValueError as error:  # unknown breakdown, should not happen
            self.table.setPlainText(str(error))
            return
        self.table.setPlainText("\n".join(lines))
        if self.as_chart.isChecked():
            self._draw_inline_chart()
        self._refresh_filter_rows()

    def _draw_inline_chart(self) -> None:
        """Draw the current breakdown into the pane, or say why it cannot be.

        The chart type follows the data rather than being chosen: one property
        is bars, two are a grid. Asking would be asking the user to know which
        shape suits a shape they have not seen yet.
        """

        if not plots.has_plot_support():

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
                lambda normalise: plots.heatmap_figure(
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
            lambda _normalise: plots.bar_figure(
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
            return format_counts(
                counts,
                f"Images by {row}",
                total=len(self.controller.selected_records()),
                multi=breakdowns[name].multi,
            )

        column = breakdowns[other].label
        table = self.controller.cross_tabulation(name, other)
        return format_cross_tabulation(
            table,
            row,
            column,
            title=f"Images by {row} and {column}",
            row_order=breakdowns[name].order,
            column_order=breakdowns[other].order,
            multi=breakdowns[name].multi or breakdowns[other].multi,
        )
