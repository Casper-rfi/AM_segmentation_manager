"""Left panel: choosing what to review, and what to change.

Holds the scan root, the projects found under it, and the lamellae of the
selected project. Scanning happens on the caller's terms -- this panel only
reports that the user asked for it.

The lists do two jobs at once: the *current* row is what is being looked at,
the *checked* rows are what a state change applies to, and the chip beside them
reports curation progress. Three separate questions, three separate marks --
see `docs/decisions.md` § Workflow state.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..controller import CurationStatus
from .common import add_path_menu, chip_icon, choose_directory
from .frame_strip import MISMATCH, STATE_COLOURS
from .theme import (
    SWATCH_BAND,
    SWATCH_GAP,
    SWATCH_HEIGHT,
    SWATCH_OUTLINE,
    SWATCH_WIDTH,
)


class ProjectPanel(QWidget):
    """Scan root, project list and lamella list."""

    #: Emitted with the directory to scan.
    scan_requested = Signal(str)
    #: Emitted with the index of the project the user picked.
    project_selected = Signal(int)
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
    #: Emitted with the modality a state change applies to.
    target_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._emitting = False
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
        browse.clicked.connect(self._browse)
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
        self.project_box, self.project_filter, self.project_list = self._build_list(
            "Projects", "Filter projects", self._project_changed,
            self.project_checked, self.projects_check_all,
        )
        self.project_filter.textChanged.connect(
            lambda text: self._apply_filter(self.project_list, self._project_labels, text)
        )
        add_path_menu(
            self.project_list, "project",
            self.project_path_requested, self.project_reveal_requested,
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

        self.lists = QSplitter(Qt.Vertical)
        self.lists.addWidget(self.project_box)
        self.lists.addWidget(self.lamella_box)
        self.lists.setSizes([200, 300])
        layout.addWidget(self.lists, stretch=1)

        layout.addWidget(self._build_target())

        self.summary = QLabel("No project loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

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

        list_widget = QListWidget()
        list_widget.setToolTip(
            "Click a row to look at it. Tick rows to choose what a state change "
            "applies to; with none ticked it applies to the frame on screen."
        )
        list_widget.currentRowChanged.connect(on_change)
        list_widget.itemChanged.connect(
            lambda item, w=list_widget, s=checked_signal: self._item_checked(w, item, s)
        )
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

    def _build_target(self) -> QWidget:
        """Return the modality control.

        Beside the ticks rather than beside the state buttons: both answer
        "what does this apply to", and the state buttons answer "to what
        state". It stays usable with editing off, because choosing what *would*
        move changes nothing on disk and the reach line above the state buttons
        recounts as you go.
        """
        box = QGroupBox("Apply to")
        box.setToolTip(
            "Which images a state change moves. The two modalities can be in "
            "different states."
        )
        row = QHBoxLayout(box)
        self.target_buttons: dict[str, QRadioButton] = {}
        group = QButtonGroup(self)
        for target, caption, hint in (
            ("SEM", "SEM", "Change the SEM image only"),
            ("FIB", "FIB", "Change the FIB image only"),
            ("both", "Both", "Change both images together"),
        ):
            button = QRadioButton(caption)
            button.setToolTip(hint)
            button.toggled.connect(
                lambda checked, t=target: checked and self.target_changed.emit(t)
            )
            group.addButton(button)
            row.addWidget(button)
            self.target_buttons[target] = button
        self.target_buttons["SEM"].setChecked(True)
        return box

    def _item_checked(self, widget: QListWidget, item, signal) -> None:
        """Report a tick, ignoring the ones the panel makes itself."""
        if self._emitting:
            return
        signal.emit(widget.row(item), item.checkState() == Qt.Checked)

    def set_target(self, target: str) -> None:
        """Reflect the current modality without re-emitting a change."""
        button = self.target_buttons[target]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)

    def _browse(self) -> None:
        directory = choose_directory(
            self, "Choose a folder to scan", self.path_edit.text() or None
        )
        if directory is not None:
            self.path_edit.setText(str(directory))
            self.scan_requested.emit(str(directory))

    def _project_changed(self, row: int) -> None:
        if not self._emitting and row >= 0:
            self.project_selected.emit(row)

    def _lamella_changed(self, row: int) -> None:
        if not self._emitting and row >= 0:
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
    ) -> None:
        """List the discovered projects, selecting the one at *current*.

        *statuses* adds a curation-progress chip to each row; omitting it lists
        the projects with no chips, which is what happens before a scan.
        *checks* says which rows are ticked.
        """
        self._project_labels = list(labels)
        self._set_items(self.project_list, labels, current, statuses, checks)
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

    def show_check_counts(
        self, project_checks: list[bool], lamella_checks: list[bool]
    ) -> None:
        """Update both titles' ticked counts, leaving the rows alone.

        Ticking a row must not rebuild the list: that would reset which row is
        current, and ticking is not a change of what is being looked at.
        """
        self.project_box.setTitle(
            self._list_title("Projects", self._project_labels, project_checks)
        )
        self.lamella_box.setTitle(
            self._list_title("Lamellae", self._lamella_labels, lamella_checks)
        )

    @staticmethod
    def _list_title(noun: str, labels: list[str], checks: list[bool] | None) -> str:
        """Return the box title, saying how many rows are ticked when any are.

        The ticks decide what a state change moves, and the lists are
        scrollable -- so how many are ticked has to be legible without
        scrolling through looking for them.
        """
        if not labels:
            return noun
        ticked = sum(1 for checked in checks or [] if checked)
        if ticked:
            return f"{noun} ({len(labels)}, {ticked} ticked)"
        return f"{noun} ({len(labels)})"

    def show_summary(self, text: str) -> None:
        """Show a short description of what is loaded."""
        self.summary.setText(text)

    def _set_items(
        self,
        widget: QListWidget,
        labels: list[str],
        current: int,
        statuses: list[CurationStatus] | None = None,
        checks: list[bool] | None = None,
    ) -> None:
        """Replace a list's contents without echoing selection signals back.

        Every refresh rebuilds the rows from scratch, so the ticks have to be
        put back from *checks* -- they live in the controller, keyed by path,
        precisely so that rebuilding the list cannot lose them.
        """
        self._emitting = True
        try:
            widget.clear()
            widget.addItems(labels)
            for row in range(widget.count()):
                item = widget.item(row)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                ticked = checks[row] if checks and row < len(checks) else False
                item.setCheckState(Qt.Checked if ticked else Qt.Unchecked)
            for row, status in enumerate(statuses or []):
                item = widget.item(row)
                if item is None:
                    break
                item.setIcon(_status_icon(status))
                # The chip cannot carry the whole meaning on its own, so the
                # long form goes in the tooltip rather than into the row text,
                # which is already carrying a path.
                item.setToolTip(f"{item.text()}\n{status.summary}")
            if labels:
                widget.setCurrentRow(max(0, min(current, len(labels) - 1)))
        finally:
            self._emitting = False

    def _apply_filter(self, widget: QListWidget, labels: list[str], text: str) -> None:
        """Hide entries that do not match *text*.

        Entries are hidden rather than removed so that a row number still means
        the same project or lamella whether or not a filter is active; rebuilding
        the list would require mapping filtered rows back to real indices.
        """
        needle = text.strip().lower()
        for row in range(widget.count()):
            item = widget.item(row)
            label = labels[row] if row < len(labels) else item.text()
            item.setHidden(bool(needle) and needle not in label.lower())


#: A curation status mapped to the colour that represents it.
#:
#: Reuses the timeline's progress ramp rather than inventing a third colour
#: vocabulary for the same idea: a status here is exactly the states in that
#: row, summarised. "not started" is the selected blue because that is what an
#: untouched image is; "done" is the corrected teal; "in progress" is drawn as
#: a split of the two rather than given a colour of its own.
STATUS_COLOURS = {
    "not started": STATE_COLOURS["selected"],
    "done": STATE_COLOURS["corrected"],
    "needs repair": MISMATCH,
}

#: Nothing is drawn for a modality with no images. An absent modality should
#: read as absent rather than as a state, and many older datasets have no FIB.
STATUS_EMPTY = "empty"

def _status_icon(status: CurationStatus) -> QIcon:
    """Return a two-band chip: SEM above, FIB below.

    One icon rather than two so a list row stays one line, and split rather
    than combined because a lamella whose SEM work is finished and whose FIB has
    not started is a normal situation that one mark cannot express. Stacked
    rather than side by side; :data:`SWATCH_WIDTH` says why.
    """
    pixmap = QPixmap(SWATCH_WIDTH, SWATCH_HEIGHT)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        for index, modality in enumerate(("SEM", "FIB")):
            top = index * (SWATCH_BAND + SWATCH_GAP)
            _paint_status(painter, top, status.of(modality))
    finally:
        # status.of() raises on an unexpected modality, and a painter left
        # alive past its pixmap crashes the process instead of reporting it.
        painter.end()
    return chip_icon(pixmap)


def _paint_status(painter: QPainter, top: int, status: str) -> None:
    """Paint one modality's band at *top*, outlined so it reads anywhere."""
    if status == STATUS_EMPTY:
        return
    if status == "in progress":
        # Part started, part finished, filling left to right like any progress
        # bar -- which reads as "between the two" without needing a fourth
        # colour explained anywhere.
        half = SWATCH_WIDTH // 2
        painter.fillRect(0, top, half, SWATCH_BAND, STATUS_COLOURS["done"])
        painter.fillRect(
            half, top, SWATCH_WIDTH - half, SWATCH_BAND, STATUS_COLOURS["not started"]
        )
    else:
        painter.fillRect(
            0, top, SWATCH_WIDTH, SWATCH_BAND,
            STATUS_COLOURS.get(status, QColor(Qt.transparent)),
        )
    painter.setPen(SWATCH_OUTLINE)
    painter.drawRect(0, top, SWATCH_WIDTH - 1, SWATCH_BAND - 1)
