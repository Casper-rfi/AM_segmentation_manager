"""Left panel: choosing what to review, and what to change.

The switches for which series the lists cover, the projects a scan found, and the
selected project's lamellae; the folder is chosen by :mod:`.scan_bar`, and the
modality and state beside the buttons that act. In the lists the *current* row is
what is looked at, the *checked* rows what a state change applies to, and the chip
reports curation progress -- see ``docs/decisions.md`` § Workflow state. Only one
list carries ticks at a time, so everything a change will touch is visible.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...layout import LABELS_EXPECTED, MODALITIES, SERIES, SERIES_LABELS
from ..controller import CurationStatus
from .common import caption_for, chip_icon, ticked_caption
from .controls import FlowArea, FlowLayout, GripSplitter, ToggleSwitch
from .lists import CheckList, add_check_all_buttons, project_box
from .paths import add_path_menu
from .theme import (
    SWATCH_CELL,
    SWATCH_GAP,
    SWATCH_HEIGHT,
    SWATCH_WIDTH,
    swatch_width,
)

#: What the series switches are for, said on the caption and on each switch.
SHOWN_HINT = (
    "Which series the lamella list, the chips and the series menu cover. A "
    "lamella holding only series switched off leaves the list and is counted "
    "in its title. Opens as Settings (F2) say."
)


class ProjectPanel(QWidget):
    """Series switches, project list and lamella list."""

    #: Emitted with the index of the project the user picked.
    project_selected = Signal(int)
    #: Emitted with the folder the user picked, which is a question about
    #: what is under it rather than a change of what is on screen.
    folder_selected = Signal(str)
    #: Emitted with the index of the lamella the user picked.
    lamella_selected = Signal(int)
    #: Emitted with the row the user right-clicked, to have its path copied.
    #: The panel holds display strings, not model objects, so it asks for the
    #: path by row rather than resolving one itself -- that stays the window's
    #: job, via the controller.
    project_path_requested = Signal(int)
    lamella_path_requested = Signal(int)
    #: As above, but to open the folder in the desktop file manager.
    project_reveal_requested = Signal(int)
    lamella_reveal_requested = Signal(int)
    #: Emitted with a row and whether it is now ticked.
    project_checked = Signal(int, bool)
    lamella_checked = Signal(int, bool)
    #: Emitted with whether to tick or clear every row of that list.
    projects_check_all = Signal(bool)
    lamellae_check_all = Signal(bool)
    #: Emitted with the series switched on, in `SERIES` order, whenever one is
    #: switched. Never empty: the last one cannot be switched off.
    shown_series_changed = Signal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self._project_labels: list[str] = []
        self._lamella_labels: list[str] = []
        #: The series each chip has a column for.
        self._chip_series: tuple[str, ...] = SERIES
        #: Lamellae the list leaves out, for its title.
        self._hidden_lamellae = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 2, 4)
        layout.setSpacing(4)

        # -- series shown --------------------------------------------------
        # Switches, as the Export tab's series are: independent, and several at
        # once is the ordinary case. In a flow under their caption rather than
        # beside it, so the column can still be narrowed: a row that cannot
        # wrap is a floor under the whole panel.
        self.shown_boxes: dict[str, ToggleSwitch] = {}
        for name in SERIES:
            switch = ToggleSwitch(SERIES_LABELS[name])
            switch.setChecked(True)
            switch.toggled.connect(
                lambda checked, name=name: self._shown_toggled(name, checked)
            )
            self.shown_boxes[name] = switch
        self.shown_flow = FlowArea()
        flow = FlowLayout(spacing=6)
        self.shown_flow.setLayout(flow)
        for name in SERIES:
            flow.addWidget(self.shown_boxes[name])
        layout.addWidget(caption_for("Series shown", self.shown_flow, SHOWN_HINT))
        layout.addWidget(self.shown_flow)

        # -- projects and lamellae -----------------------------------------
        # In a splitter so that an archive with many projects but few lamellae
        # (or the reverse) can be given the space where it is needed.
        self.project_box, self.project_filter, self.project_list = self._build_tree(
            "Projects"
        )
        self.project_filter.textChanged.connect(
            lambda text: self._apply_filter(self.project_list, self._project_labels, text)
        )
        add_path_menu(
            self.project_list, "project",
            self.project_path_requested, self.project_reveal_requested,
            row_of=self.project_list.project_of,
        )
        self.lamella_box, self.lamella_filter, self.lamella_list = self._build_list(
            "Lamellae", "Filter lamellae", self._lamella_changed,
            self.lamella_checked, self.lamellae_check_all,
        )
        self.lamella_filter.textChanged.connect(
            lambda text: self._apply_filter(self.lamella_list, self._lamella_labels, text)
        )
        add_path_menu(
            self.lamella_list, "lamella",
            self.lamella_path_requested, self.lamella_reveal_requested,
        )

        self.lists = GripSplitter(Qt.Orientation.Vertical)
        self.lists.addWidget(self.project_box)
        self.lists.addWidget(self.lamella_box)
        self.lists.setSizes([200, 300])
        layout.addWidget(self.lists, stretch=1)
        self.set_chip_series(SERIES)

        self.summary = QLabel("No project loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def _build_tree(self, title: str):
        """Build the project tree, boxed and filterable like the lamella list.

        The furniture comes from `lists.project_box`, which the Add data and
        Export tabs use too: three tabs listing the same projects had three
        arrangements of the same widget, and which of them a reader is looking
        at is not a thing the window should have to teach.
        """
        box, filter_edit, tree = project_box(
            title,
            on_check_all=self.projects_check_all.emit,
            tooltip=(
                "Projects, under the folder each is in. Click a row to look at "
                "it; tick rows to choose what a change applies to. Ticking a "
                "folder ticks its projects, and ticking here clears the "
                "lamellae."
            ),
        )
        tree.project_selected.connect(self._project_changed)
        tree.folder_selected.connect(self.folder_selected)
        tree.row_ticked.connect(self.project_checked)
        return box, filter_edit, tree

    def _build_list(
        self, title: str, placeholder: str, on_change, checked_signal, check_all_signal
    ):
        """Build a titled, filterable list box with tick-all controls."""
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 4, 6, 6)
        box_layout.setSpacing(3)

        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText(placeholder)
        filter_edit.setClearButtonEnabled(True)
        filter_edit.setToolTip("Show only entries whose name contains this text")
        box_layout.addWidget(filter_edit)

        list_widget = CheckList()
        list_widget.setToolTip(
            "Click a row to look at it. Tick rows to choose what a change "
            "applies to; ticking here clears the other list. With nothing "
            "ticked it applies to the frame on screen."
        )
        list_widget.currentRowChanged.connect(on_change)
        list_widget.row_ticked.connect(checked_signal)
        box_layout.addWidget(list_widget)

        box_layout.addLayout(self._build_check_all(check_all_signal, title.lower()))
        return box, filter_edit, list_widget

    def _build_check_all(self, signal, noun: str) -> QHBoxLayout:
        """Return the row of tick-everything / tick-nothing buttons."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        add_check_all_buttons(row, signal.emit, noun)
        row.addStretch(1)
        return row

    def _project_changed(self, row: int) -> None:
        if row >= 0:
            self.project_selected.emit(row)

    def _lamella_changed(self, row: int) -> None:
        if row >= 0:
            self.lamella_selected.emit(row)

    def shown_series(self) -> tuple[str, ...]:
        """Return the series switched on, in `SERIES` order."""
        return tuple(name for name in SERIES if self.shown_boxes[name].isChecked())

    def _shown_toggled(self, name: str, checked: bool) -> None:
        """Say which series are on, refusing to switch the last one off.

        None at all would list nothing and draw an empty chip, which reads as a
        fault rather than a choice.
        """
        if not checked and not self.shown_series():
            switch = self.shown_boxes[name]
            switch.blockSignals(True)
            switch.setChecked(True)
            switch.blockSignals(False)
            return
        self.shown_series_changed.emit(self.shown_series())

    def apply_defaults(self, values: Mapping[str, int | bool]) -> None:
        """Open the series switches at what the settings say. See `ImagePanel`.

        Set quietly and announced once, so opening does not redraw the lists
        once per switch.
        """
        from .. import preferences

        wanted = preferences.shown_series_from(values)
        for name, switch in self.shown_boxes.items():
            switch.blockSignals(True)
            switch.setChecked(name in wanted)
            switch.blockSignals(False)
        self.shown_series_changed.emit(self.shown_series())

    def set_chip_series(self, series: Sequence[str]) -> None:
        """Draw the chips with a column for each of *series* only.

        Both lists are told the chip's size, or Qt scales it to a text line's
        height and the cells come out a smudge. The rows are drawn again by
        :meth:`show_projects` and :meth:`show_lamellae`, not here.
        """
        self._chip_series = tuple(series)
        size = QSize(swatch_width(len(self._chip_series)), SWATCH_HEIGHT)
        self.project_list.setIconSize(size)
        self.lamella_list.setIconSize(size)

    def show_projects(
        self,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
        folders: list[str] | None = None,
        current_folder: str | None = None,
    ) -> None:
        """List the discovered projects, selecting the one at *current*.

        *statuses* adds a curation-progress chip to each row; omitting it lists
        the projects with no chips, which is what happens before a scan.
        *checks* says which rows are ticked. *folders* says which folder each
        project hangs under, one entry per label; omitting it puts them all at
        the top level, which is a flat list by another name. *current_folder*
        puts the cursor on a folder row instead of a project, for when the
        lamella list is that folder's rather than one project's.
        """
        self._project_labels = list(labels)
        self.project_list.set_projects(
            folders if folders is not None else [""] * len(labels),
            labels,
            ticked=checks or [],
            icons=[_status_icon(status, self._chip_series) for status in statuses or []],
            tooltips=[
                f"{label}\n{status.summary}"
                for label, status in zip(labels, statuses or [])
            ],
            current=current,
            current_folder=current_folder,
        )
        self.project_box.setTitle(
            self._list_title("Projects", labels, checks)
        )
        self._apply_filter(self.project_list, self._project_labels, self.project_filter.text())

    def show_lamellae(
        self,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
        hidden: int = 0,
    ) -> None:
        """List the selected project's lamellae, with their curation progress.

        *hidden* counts the lamellae left out for holding only series switched
        off, which the title says so that leaving them out is never silent.
        """
        self._lamella_labels = list(labels)
        self._hidden_lamellae = hidden
        self._set_items(self.lamella_list, labels, current, statuses, checks)
        self.lamella_box.setTitle(
            self._list_title("Lamellae", labels, checks, hidden)
        )
        self._apply_filter(self.lamella_list, self._lamella_labels, self.lamella_filter.text())

    def show_checks(
        self, project_checks: list[bool], lamella_checks: list[bool]
    ) -> None:
        """Put both lists' ticks and titles back, leaving the rows alone.

        Ticking a row must not rebuild the list: that would reset which row is
        current, and ticking is not a change of what is being looked at. Both
        lists are refreshed on every tick because ticking in one clears the
        other, and the user has to see that happen.
        """
        self.project_list.show_ticks(project_checks)
        self.lamella_list.show_ticks(lamella_checks)
        self.project_box.setTitle(
            self._list_title("Projects", self._project_labels, project_checks)
        )
        self.lamella_box.setTitle(
            self._list_title(
                "Lamellae", self._lamella_labels, lamella_checks, self._hidden_lamellae
            )
        )

    @staticmethod
    def _list_title(
        noun: str, labels: list[str], checks: list[bool] | None, hidden: int = 0
    ) -> str:
        """Return the box title, saying how many rows are ticked and hidden."""
        return ticked_caption(
            noun, len(labels), sum(1 for checked in checks or [] if checked), hidden
        )

    def show_summary(self, text: str) -> None:
        """Show a short description of what is loaded."""
        self.summary.setText(text)

    def announce(self, text: str) -> None:
        """Show *text* and paint it now, before the caller blocks.

        Scanning holds the interface thread, so an ordinary ``setText`` would
        not reach the screen until the scan it is announcing had finished.
        ``repaint`` draws this one widget synchronously; it does not process
        events, so nothing is re-entered on the way -- the distinction
        :class:`~.busy_dialog.BusyDialog` exists to keep.
        """
        self.summary.setText(text)
        self.summary.repaint()

    def _set_items(
        self,
        widget: CheckList,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
    ) -> None:
        """Replace a list's contents, ticks and chips together.

        The chip cannot carry its whole meaning on its own, so the long form
        goes in the tooltip rather than into the row text, which is already
        carrying a path.
        """
        widget.set_values(
            labels,
            ticked=checks or [],
            icons=[_status_icon(status, self._chip_series) for status in statuses or []],
            tooltips=[
                f"{label}\n{status.summary}"
                for label, status in zip(labels, statuses or [])
            ],
            current=current,
        )

    def _apply_filter(self, widget: CheckList, labels: list[str], text: str) -> None:
        """Hide entries that do not match *text*."""
        widget.hide_rows_without(text, labels)


#: What each thing a cell can say is filled with, lightest to darkest.
#:
#: A ramp in lightness rather than in hue, and the order here is the order of
#: the ramp: brightness is what tells the steps apart at this size, and the
#: timeline's two hues only confirm it. `docs/display.md` § Controls says why,
#: and the F1 list says it to the reader.
CELL_COLOURS = {
    "empty": QColor("#e9ecf1"),
    "unlabelled": QColor("#aeb6c2"),
    "not started": QColor("#9dbcea"),
    "in progress": QColor("#4c9c8b"),
    "done": QColor("#12604f"),
}

#: A modality with no images. It gets the palest fill rather than nothing drawn:
#: absent has to read as absent, and many older datasets have no FIB at all.
STATUS_EMPTY = "empty"

#: A series that is present and that nobody labels. Off the ramp entirely, so
#: it cannot be read as a step of one.
STATUS_UNLABELLED = "unlabelled"

def _status_icon(status: CurationStatus, series: Sequence[str] = SERIES) -> QIcon:
    """Return the chip: a cell per series and beam, SEM above FIB.

    A column for each of *series* only -- the ones switched on -- in the order
    given, so a series switched off takes no room rather than an empty column.

    One icon rather than two, so a row stays one line and the two questions a
    reader has -- what is here, and how far along is it -- are answered in one
    glance at one mark. The grid replaced a pair of bands plus a row of
    presence marks beside them: the bands could not say *which* series they
    were about, and saying it separately cost half again the width for
    something the cells already carry.

    Every cell is a flat tile from :data:`CELL_COLOURS`. Only Adaptive Polishing
    is in :data:`~...layout.LABELS_EXPECTED`, so only its column ever varies --
    which is the point rather than a shortcoming, a reference column that
    changed would be claiming work nobody intends to do.
    """
    pixmap = QPixmap(swatch_width(len(series)), SWATCH_HEIGHT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        for column, name in enumerate(series):
            for row, modality in enumerate(MODALITIES):
                _paint_cell(
                    painter,
                    column * (SWATCH_CELL + SWATCH_GAP),
                    row * (SWATCH_CELL + SWATCH_GAP),
                    status.of(name, modality),
                    expected=name in LABELS_EXPECTED,
                )
    finally:
        # A painter left alive past its pixmap crashes the process instead of
        # reporting it.
        painter.end()
    return chip_icon(pixmap)


def _paint_cell(
    painter: QPainter, left: int, top: int, status: str, *, expected: bool
) -> None:
    """Paint one series-and-beam cell as a flat tile of one colour.

    One colour rather than a part-filled one: anything divided inside seven
    pixels is a smudge whatever it is made of, so the middle step is a colour
    instead of a boundary.
    """
    painter.fillRect(
        left, top, SWATCH_CELL, SWATCH_CELL, _cell_colour(status, expected=expected)
    )


def _cell_colour(status: str, *, expected: bool) -> QColor:
    """Return the fill for a cell in *status*, in a series anyone labels or not.

    A series nobody labels reports presence and claims nothing about progress:
    reference and alignment images sit at the default state for ever, and a
    progress colour there would say work is outstanding that nobody intends to
    do.
    """
    if status == STATUS_EMPTY:
        return CELL_COLOURS[STATUS_EMPTY]
    if not expected:
        return CELL_COLOURS[STATUS_UNLABELLED]
    return CELL_COLOURS[status]


#: How big a single colour is drawn in the F1 legend. Twice a cell, so it reads
#: as a colour beside a line of text where a lone cell reads as a speck; the
#: fill is the one the chip actually uses, which is the part that has to match.
LEGEND_SWATCH = 2 * SWATCH_CELL


def _legend_swatch(status: str, *, expected: bool) -> QPixmap:
    """Return one of the chip's fills, alone and big enough to see."""
    pixmap = QPixmap(LEGEND_SWATCH, LEGEND_SWATCH)
    pixmap.fill(_cell_colour(status, expected=expected))
    return pixmap


def status_legend() -> list[tuple[QPixmap, str]]:
    """Return what the chip beside each row means, as ``(sample, meaning)``.

    The samples are drawn by the same painter as the chips themselves, so a
    colour cannot be explained here as one thing and drawn there as another.
    """
    example = CurationStatus(
        {
            **{(series, modality): "not started"
               for series in SERIES for modality in MODALITIES},
            ("adaptive", "SEM"): "done",
            ("adaptive", "FIB"): "in progress",
        }
    )
    columns = ", ".join(SERIES_LABELS[series] for series in SERIES)
    return [
        (
            _status_icon(example).pixmap(SWATCH_WIDTH, SWATCH_HEIGHT),
            f"Left to right, those switched on under Series shown: {columns}. "
            "SEM above, FIB below.",
        ),
        (
            _legend_swatch(STATUS_EMPTY, expected=True),
            "No images of that series at all",
        ),
        (
            _legend_swatch("not started", expected=False),
            "Images are there, and nobody labels that series",
        ),
        (
            _legend_swatch("not started", expected=True),
            "Nothing corrected yet",
        ),
        (
            _legend_swatch("in progress", expected=True),
            "Part corrected",
        ),
        (
            _legend_swatch("done", expected=True),
            "Corrected, or deselected as not wanted",
        ),
    ]
