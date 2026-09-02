"""Left panel: choosing what to review, and what to change.

Holds the scan root, the projects found under it, and the lamellae of the
selected project. Scanning happens on the caller's terms -- this panel only
reports that the user asked for it. Which *rows* a change covers is chosen
here; which modality, and which state, are chosen beside the buttons that act.

The lists do two jobs at once: the *current* row is what is being looked at,
the *checked* rows are what a state change applies to, and the chip beside them
reports curation progress. Three separate questions, three separate marks --
see `docs/decisions.md` § Workflow state.

Only one of the two lists carries ticks at a time, and the ticked lamellae are
the selected project's own -- so everything a change will touch is a tick
visible on screen.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...layout import LABELS_EXPECTED, MODALITIES, SERIES, SERIES_LABELS
from ..controller import CurationStatus
from .common import (
    CheckList,
    GripSplitter,
    ProjectTree,
    add_fold_buttons,
    add_path_menu,
    chip_icon,
    choose_directory,
    ticked_caption,
)
from .theme import (
    SWATCH_CELL,
    SWATCH_GAP,
    SWATCH_HEIGHT,
    SWATCH_WIDTH,
)


class ProjectPanel(QWidget):
    """Scan root, project list and lamella list."""

    #: Emitted with the directory to scan.
    scan_requested = Signal(str)
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

    def __init__(self) -> None:
        super().__init__()
        self._project_labels: list[str] = []
        self._lamella_labels: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 2, 4)
        layout.setSpacing(4)

        # -- scan root -----------------------------------------------------
        # No group box. A "Scan root" title said the same thing as the field's
        # placeholder and the Scan button beside it -- one fact stated three
        # times. The placeholder carries the explanation where it is useful
        # (visible while the field is empty, gone once a path is in it), and the
        # tooltips carry what neither a label nor a button caption can say.
        #
        # The path keeps a row to itself. Putting it beside both buttons was
        # tried and left roughly 140px for the field in a 270px panel, showing
        # the tail of a path and nothing else -- and reading the path is the
        # entire reason the field is there.
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder to scan for Autolamella projects")
        self.path_edit.setToolTip(
            "Folder searched recursively for Autolamella projects "
            "(any folder containing an experiment.yaml or AM_metadata.yaml)"
        )
        self.path_edit.returnPressed.connect(
            lambda: self.scan_requested.emit(self.path_edit.text())
        )
        layout.addWidget(self.path_edit)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addStretch(1)

        browse = QPushButton("Browse...")
        browse.setToolTip("Choose a folder to scan")
        browse.clicked.connect(self.browse)
        row.addWidget(browse)

        # Not full width any more: Scan is not the window's primary action, and
        # a full-width button implied that it was.
        self.scan_button = QPushButton("Scan")
        self.scan_button.setToolTip(
            "Re-read the folder from disk. Use this after changing files "
            "outside the window."
        )
        self.scan_button.clicked.connect(
            lambda: self.scan_requested.emit(self.path_edit.text())
        )
        row.addWidget(self.scan_button)

        layout.addLayout(row)

        # -- projects and lamellae -----------------------------------------
        # In a splitter so that an archive with many projects but few lamellae
        # (or the reverse) can be given the space where it is needed.
        self.project_box, self.project_filter, self.project_list = self._build_tree(
            "Projects", "Filter projects",
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

        self.lists = GripSplitter(Qt.Vertical)
        self.lists.addWidget(self.project_box)
        self.lists.addWidget(self.lamella_box)
        self.lists.setSizes([200, 300])
        layout.addWidget(self.lists, stretch=1)

        self.summary = QLabel("No project loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def _build_tree(self, title: str, placeholder: str):
        """Build the project tree, boxed and filterable like the lamella list.

        Same furniture as :meth:`_build_list` deliberately: the two sit one
        above the other, and a tree that framed itself differently would read
        as a different kind of thing rather than the same list, grouped.
        """
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 4, 6, 6)
        box_layout.setSpacing(3)

        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText(placeholder)
        filter_edit.setClearButtonEnabled(True)
        filter_edit.setToolTip("Show only entries whose name contains this text")
        box_layout.addWidget(filter_edit)

        tree = ProjectTree()
        tree.setToolTip(
            "Projects, under the folder each is in. Click a row to look at it; "
            "tick rows to choose what a change applies to. Ticking a folder "
            "ticks its projects, and ticking here clears the lamellae."
        )
        tree.project_selected.connect(self._project_changed)
        tree.folder_selected.connect(self.folder_selected)
        tree.row_ticked.connect(self.project_checked)
        box_layout.addWidget(tree)

        # Opening and closing every folder goes on the same row as ticking
        # every project, both being "do this to the whole tree" -- but past the
        # stretch, so what the ticks decide stays apart from what is merely on
        # screen. Neither reaches the controller: which folders are open is the
        # widget's own business and nothing else asks.
        box_layout.addLayout(
            add_fold_buttons(
                self._build_check_all(self.projects_check_all, "projects"), tree
            )
        )
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
        for caption, checked, hint in (
            ("Tick all", True, f"Choose every one of the {noun} listed"),
            ("None", False, f"Clear the ticks on all {noun}"),
        ):
            button = QPushButton(caption)
            button.setToolTip(hint)
            button.clicked.connect(lambda _checked=False, v=checked: signal.emit(v))
            row.addWidget(button)
        row.addStretch(1)
        return row

    def browse(self) -> None:
        """Ask for a folder to scan, and request a scan of it if one is chosen.

        Public because the File menu offers the same thing: two ways in, one
        behaviour, rather than the menu opening a dialog of its own.
        """
        directory = choose_directory(
            self, "Choose a folder to scan", self.path_edit.text() or None
        )
        if directory is not None:
            self.path_edit.setText(str(directory))
            self.scan_requested.emit(str(directory))

    def _project_changed(self, row: int) -> None:
        if row >= 0:
            self.project_selected.emit(row)

    def _lamella_changed(self, row: int) -> None:
        if row >= 0:
            self.lamella_selected.emit(row)

    def set_path(self, path: str) -> None:
        """Show *path* as the scan root."""
        self.path_edit.setText(path)

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
            icons=[_status_icon(status) for status in statuses or []],
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
    ) -> None:
        """List the selected project's lamellae, with their curation progress."""
        self._lamella_labels = list(labels)
        self._set_items(self.lamella_list, labels, current, statuses, checks)
        self.lamella_box.setTitle(
            self._list_title("Lamellae", labels, checks)
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
            self._list_title("Lamellae", self._lamella_labels, lamella_checks)
        )

    @staticmethod
    def _list_title(noun: str, labels: list[str], checks: list[bool] | None) -> str:
        """Return the box title, saying how many rows are ticked when any are."""
        return ticked_caption(
            noun, len(labels), sum(1 for checked in checks or [] if checked)
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
            icons=[_status_icon(status) for status in statuses or []],
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

def _status_icon(status: CurationStatus) -> QIcon:
    """Return the chip: a cell per series and beam, SEM above FIB.

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
    pixmap = QPixmap(SWATCH_WIDTH, SWATCH_HEIGHT)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        for column, series in enumerate(SERIES):
            for row, modality in enumerate(MODALITIES):
                _paint_cell(
                    painter,
                    column * (SWATCH_CELL + SWATCH_GAP),
                    row * (SWATCH_CELL + SWATCH_GAP),
                    status.of(series, modality),
                    expected=series in LABELS_EXPECTED,
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

    One colour rather than a part-filled one. A part-finished cell used to be
    split between the two progress colours, which at this size showed two pixels
    of each; anything divided inside seven pixels is a smudge whatever it is
    made of, so the middle step is a colour instead of a boundary.
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
            f"Left to right: {columns}. SEM above, FIB below.",
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
            "Nothing proofread yet",
        ),
        (
            _legend_swatch("in progress", expected=True),
            "Part proofread",
        ),
        (
            _legend_swatch("done", expected=True),
            "Proofread, or deselected as not wanted",
        ),
    ]
