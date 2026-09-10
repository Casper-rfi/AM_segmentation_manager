"""The two list widgets every tab picks rows in.

A chosen row is ticked, never highlighted, and a list reports nothing while its
contents are replaced -- `docs/display.md` says why both, and why the projects
are grouped under the folder each sits in. Nothing here knows what a project
is beyond its name and its chip; deciding which rows to show belongs to a
controller.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from qtpy.QtCore import (
    QSize,
    Qt,
    Signal,
)
from qtpy.QtGui import (
    QIcon,
)
from qtpy.QtWidgets import (
    QGroupBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .controls import FlowArea, FlowLayout
from .theme import SWATCH_HEIGHT, SWATCH_WIDTH

logger = logging.getLogger(__name__)


#: Where a project row keeps the number the controller knows it by. A tree row
#: is not a row number -- the folders take positions too -- so the number
#: travels on the item rather than being counted off the widget.
PROJECT_INDEX = Qt.UserRole + 1


#: How far a project is indented under its folder. Enough to read as beneath it
#: and no more: the panel is narrow and the rows carry names that need the room.
TREE_INDENT = 12


def add_fold_buttons(row, tree: ProjectTree):
    """Append Expand and Collapse for *tree* to an existing row, and return it.

    A folder opens when it is clicked and stays as the reader left it, which is
    right for working down one session and wrong at the two moments when the
    whole shape is the question: finding a project whose folder you have
    forgotten, and putting thirty open folders away again.

    Appended rather than given a row of their own, so a panel can put them
    beside whatever else acts on the whole tree. Neither reaches a controller:
    which folders are open is the widget's own business and nothing else asks.
    """
    for caption, hint, act in (
        ("Expand", "Open every folder", tree.expandAll),
        ("Collapse", "Close every folder", tree.collapseAll),
    ):
        button = QPushButton(caption)
        button.setToolTip(hint)
        button.clicked.connect(act)
        row.addWidget(button)
    return row


def add_check_all_buttons(row, on_all, noun: str):
    """Append Tick all and None to an existing row, and return it.

    Appended like :func:`add_fold_buttons`, and usually beside it: what a
    change applies to and what is merely on screen are different questions,
    which is why the two runs are separate and why the stretch between them
    belongs to the caller.

    Args:
        on_all: Called with True to tick every row and False to clear them.
        noun: What the rows are, for the tooltips -- "projects", "lamellae".
    """
    for caption, wanted, hint in (
        ("Tick all", True, f"Choose every one of the {noun} listed"),
        ("None", False, f"Clear the ticks on all {noun}"),
    ):
        button = QPushButton(caption)
        button.setToolTip(hint)
        button.clicked.connect(lambda _checked=False, value=wanted: on_all(value))
        row.addWidget(button)
    return row


#: Between the buttons deciding what a change covers and those deciding only
#: what is on screen. A gap rather than a stretch: the run wraps, and a
#: stretch in a flow layout is a spacer of no width. Narrow, because the four
#: buttons want 284px before it and the columns they sit in are ~300.
FOLD_GAP = 8

#: What a project box's filter field says while it is empty, and what its rows
#: are called in the tick-all tooltips. One spelling, so three tabs cannot
#: describe the same list differently.
PROJECT_PLACEHOLDER = "Filter projects"
PROJECT_NOUN = "projects"


def project_box(
    title: str,
    *,
    checkable: bool = True,
    on_check_all=None,
    tooltip: str = "",
) -> tuple[QGroupBox, QLineEdit, ProjectTree]:
    """Return a titled box holding a filter field, a project tree and its buttons.

    The three tabs that list projects all want the same furniture around the
    same widget, and had three arrangements of it: one with a filter field and
    ticks, one with neither, one with ticks and no field. Which of them a
    reader is looking at is not a thing the window should have to teach.

    Args:
        checkable: Whether rows carry tick boxes. False where the tab acts on
            what is being looked at rather than on a set.
        on_check_all: Called with True or False to tick or clear every row.
            None leaves the tick-all buttons off, which is what a tree with no
            boxes wants.
        tooltip: What the tree says about itself, which differs by what the
            ticks mean on that tab.

    Returns:
        The box, its filter field and the tree -- the field and the tree
        because the caller has to connect them to whatever it filters and
        selects.
    """
    box = QGroupBox(title)
    column = QVBoxLayout(box)
    column.setContentsMargins(6, 4, 6, 6)
    column.setSpacing(3)

    filter_edit = QLineEdit()
    filter_edit.setPlaceholderText(PROJECT_PLACEHOLDER)
    filter_edit.setClearButtonEnabled(True)
    filter_edit.setToolTip("Show only entries whose name contains this text")
    column.addWidget(filter_edit)

    tree = ProjectTree(checkable=checkable)
    if tooltip:
        tree.setToolTip(tooltip)
    column.addWidget(tree)

    # Opening and closing every folder goes on the same row as ticking every
    # project, both being "do this to the whole tree" -- but past a gap, so
    # what the ticks decide stays apart from what is merely on screen.
    #
    # Wrapped, because four buttons want 284px and the Export tab's filter
    # column is narrower than that: a row that cannot wrap is clipped, and a
    # button reading "ollaps" is worse than one on a second line.
    buttons = FlowArea()
    row = FlowLayout(spacing=4)
    buttons.setLayout(row)
    if on_check_all is not None:
        add_check_all_buttons(row, on_check_all, PROJECT_NOUN)
        row.addSpacing(FOLD_GAP)
    add_fold_buttons(row, tree)
    column.addWidget(buttons)
    return box, filter_edit, tree


class ProjectTree(QTreeWidget):
    """The projects, gathered under the folder each one sits in.

    An Autolamella project is named for the minute it started, so a flat list of
    them is a hundred timestamps; the folder somebody filed them under is what
    says what they were. The folder is a row of its own and the projects sit
    under it, which is also what lets the ones being ignored be folded away.

    Ticking works as it does in :class:`CheckList` and means the same
    thing -- what a change applies to. A folder's tick is Qt's own tri-state,
    so ticking one ticks its projects and each reports itself: the folder is a
    convenience over the projects, never a thing that can be ticked instead.
    """

    #: Emitted with a project's number and whether it is now ticked. Not
    #: emitted for the ticks the tree puts back itself when it is refilled.
    row_ticked = Signal(int, bool)
    #: Emitted with the number of the project the user moved to.
    project_selected = Signal(int)
    #: Emitted with the name of the folder the user moved to. A folder is not a
    #: project and moving to one does not change the frame on screen: it asks
    #: what is under the folder, which is the question a closed one raises.
    folder_selected = Signal(str)

    def __init__(self, *, checkable: bool = True) -> None:
        """Build the tree.

        Args:
            checkable: Whether the rows carry tick boxes. False where a tab
                acts on the row being *looked at* rather than on a set -- a box
                there offers a choice the tab cannot honour, which is worse
                than no box at all.
        """
        super().__init__()
        self._filling = False
        self._checkable = checkable
        #: Which folders were open when a filter was last typed into, so
        #: clearing it can put them back. None while no filter is in force.
        self._before_filter: set[str] | None = None
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setExpandsOnDoubleClick(False)
        # A row carries a path, and both ends of one are what identify it: the
        # session folder at the front and the minute at the back. Eliding needs
        # the horizontal scrollbar gone, or the view scrolls rather than elides
        # -- the same pairing the lists use.
        self.setTextElideMode(Qt.ElideMiddle)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Narrower than Qt's default, which spends most of a 270px panel on
        # saying that a project is one level down.
        self.setIndentation(TREE_INDENT)
        # Or Qt scales the chip to a text line's height and the cells
        # come out a smudge.
        self.setIconSize(QSize(SWATCH_WIDTH, SWATCH_HEIGHT))
        self.itemChanged.connect(self._item_changed)
        self.currentItemChanged.connect(self._current_changed)

    # -- filling -----------------------------------------------------------

    def set_projects(
        self,
        folders: Sequence[str],
        labels: Sequence[str],
        *,
        ticked: Sequence[bool] = (),
        icons: Sequence[QIcon] | None = None,
        tooltips: Sequence[str] | None = None,
        current: int | None = None,
        current_folder: str | None = None,
    ) -> None:
        """Replace the rows, keeping what the reader had arranged.

        Which folders were opened and where the tree was scrolled to are the
        reader's, not the data's: a rescan after an operation must not undo the
        arranging and send the tree back to the top.

        **A folder seen for the first time is closed.** An archive of a hundred
        projects across thirty sessions opens as thirty rows a reader can look
        down, where opening them all is the flat list this replaced. The folder
        holding the current project is opened even so, a selection nobody can
        see being worse than a row nobody asked for.

        Reports nothing while it runs, for the reason
        :meth:`CheckList.set_values` gives.
        """
        opened = self._opened()
        scrolled = self.verticalScrollBar().value()

        self._filling = True
        self.blockSignals(True)
        try:
            self.clear()
            parents: dict[str, QTreeWidgetItem] = {}
            for row, (folder, label) in enumerate(zip(folders, labels)):
                item = QTreeWidgetItem([label])
                item.setData(0, PROJECT_INDEX, row)
                self._set_checkable(item)
                if self._checkable:
                    item.setCheckState(
                        0,
                        Qt.Checked
                        if row < len(ticked) and ticked[row]
                        else Qt.Unchecked,
                    )
                if icons is not None and row < len(icons):
                    item.setIcon(0, icons[row])
                if tooltips is not None and row < len(tooltips):
                    item.setToolTip(0, tooltips[row])
                self._parent_for(folder, parents).addChild(item)

            for folder, parent in parents.items():
                parent.setExpanded(folder in opened)
            if current_folder is not None:
                self._make_folder_current(current_folder, parents)
            elif current is not None:
                self._make_current(current)
        finally:
            self.blockSignals(False)
            self._filling = False

        self.verticalScrollBar().setValue(scrolled)

    def show_icons(
        self,
        icons: Sequence[QIcon],
        tooltips: Sequence[str] | None = None,
    ) -> None:
        """Replace the chips without rebuilding the rows.

        A chip changes for a reason that is not a change of what is listed --
        a project described, a state set -- and rebuilding would reset which
        row is current and refold what the reader had opened.
        """
        self._filling = True
        self.blockSignals(True)
        try:
            for item in self.project_items():
                row = self.project_of(item)
                if row is None or row >= len(icons):
                    continue
                item.setIcon(0, icons[row])
                if tooltips is not None and row < len(tooltips):
                    item.setToolTip(0, tooltips[row])
        finally:
            self.blockSignals(False)
            self._filling = False

    def _set_checkable(self, item: QTreeWidgetItem) -> None:
        """Add or take away a row's tick box.

        **Taken away explicitly.** ``ItemIsUserCheckable`` is on by default, and
        Qt merely draws no box while no check state has been set -- so a tree
        built without ticks would report every row as checkable, and the first
        thing to call ``setCheckState`` would put boxes back on a tab that
        cannot act on them.
        """
        if self._checkable:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        else:
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)

    def _parent_for(
        self, folder: str, parents: dict[str, QTreeWidgetItem]
    ) -> QTreeWidgetItem:
        """Return the row a project of *folder* hangs from.

        A project at the scan root has no folder and hangs from the tree
        itself: inventing a "(root)" row for it would name something that is
        not there.
        """
        if not folder:
            return self.invisibleRootItem()
        if folder not in parents:
            item = QTreeWidgetItem([folder])
            self._set_checkable(item)
            if self._checkable:
                # Tri-state and Qt keeps the folder in step with its projects,
                # in both directions, without this class knowing how to count
                # them.
                item.setFlags(item.flags() | Qt.ItemIsAutoTristate)
                item.setCheckState(0, Qt.Unchecked)
            self.addTopLevelItem(item)
            parents[folder] = item
        return parents[folder]

    def _opened(self) -> set[str]:
        """Return the folders the reader had opened."""
        return {
            self.topLevelItem(row).text(0)
            for row in range(self.topLevelItemCount())
            if self.topLevelItem(row).childCount()
            and self.topLevelItem(row).isExpanded()
        }

    def _make_folder_current(
        self, folder: str, parents: dict[str, QTreeWidgetItem]
    ) -> None:
        """Put the cursor back on a folder row, and open it.

        A rescan rebuilds the tree, and a folder whose lamellae are listed must
        come back as the row being looked at -- otherwise the list below says
        one thing and the tree shows nothing chosen.
        """
        item = parents.get(folder)
        if item is None:
            return
        item.setExpanded(True)
        self.setCurrentItem(item)

    def _make_current(self, index: int) -> None:
        """Put the cursor on a project, opening the folder holding it."""
        item = self.project_item(index)
        if item is None:
            return
        if item.parent() is not None:
            item.parent().setExpanded(True)
        self.setCurrentItem(item)

    # -- reading -----------------------------------------------------------

    def project_of(self, item: QTreeWidgetItem) -> int | None:
        """Return the project number *item* stands for, or None for a folder."""
        return item.data(0, PROJECT_INDEX)

    def project_item(self, index: int) -> QTreeWidgetItem | None:
        """Return the row standing for a project number, or None if it is gone."""
        return next(
            (item for item in self.project_items() if self.project_of(item) == index),
            None,
        )

    def project_items(self) -> list[QTreeWidgetItem]:
        """Return every project row in project order, the folders aside.

        The mapping from a project number to a row is this class's to know:
        a tree row is not a row number, and every caller that needs one --
        the context menu, stepping, a test -- asks here rather than counting.
        """
        items = []
        for row in range(self.topLevelItemCount()):
            top = self.topLevelItem(row)
            if self.project_of(top) is not None:
                items.append(top)
            items.extend(top.child(child) for child in range(top.childCount()))
        return items

    def ticked_rows(self) -> list[int]:
        """Return the numbers of the ticked projects."""
        return [
            self.project_of(item)
            for item in self.project_items()
            if item.checkState(0) == Qt.Checked
        ]

    # -- changing ----------------------------------------------------------

    def show_ticks(self, ticked: Sequence[bool]) -> None:
        """Put the ticks back without rebuilding the rows."""
        self._filling = True
        self.blockSignals(True)
        try:
            for item in self.project_items():
                index = self.project_of(item)
                item.setCheckState(
                    0,
                    Qt.Checked
                    if index < len(ticked) and ticked[index]
                    else Qt.Unchecked,
                )
        finally:
            self.blockSignals(False)
            self._filling = False

    def hide_rows_without(self, text: str, labels: Sequence[str] | None = None) -> None:
        """Show only the projects matching *text*, and the folders holding them.

        A folder whose every project is hidden goes with them: leaving it
        behind would offer somewhere to look that has nothing in it.

        **A folder holding a match is opened**, or with the folders closed the
        filter would answer by showing the folder and hiding what it found in
        it -- which reads as finding nothing. The folds in force before the
        filter are put back when it is cleared, opening a folder to search
        being the search's doing rather than the reader's.
        """
        wanted = text.strip().lower()
        if wanted and self._before_filter is None:
            self._before_filter = self._opened()

        for row in range(self.topLevelItemCount()):
            top = self.topLevelItem(row)
            if self.project_of(top) is not None:
                top.setHidden(bool(wanted) and wanted not in top.text(0).lower())
                continue
            shown = 0
            for child in range(top.childCount()):
                item = top.child(child)
                hidden = bool(wanted) and wanted not in item.text(0).lower()
                item.setHidden(hidden)
                shown += not hidden
            top.setHidden(shown == 0)
            if wanted and shown:
                top.setExpanded(True)

        if not wanted and self._before_filter is not None:
            for row in range(self.topLevelItemCount()):
                top = self.topLevelItem(row)
                if top.childCount():
                    top.setExpanded(top.text(0) in self._before_filter)
            self._before_filter = None

    # -- reporting ---------------------------------------------------------

    def _item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        """Report a tick the user made, and none that this class put back."""
        if self._filling:
            return
        index = self.project_of(item)
        if index is None:
            return
        self.row_ticked.emit(index, item.checkState(0) == Qt.Checked)

    def _current_changed(self, current, _previous) -> None:
        """Report a move, saying whether it landed on a project or a folder."""
        if self._filling or current is None:
            return
        index = self.project_of(current)
        if index is not None:
            self.project_selected.emit(index)
        else:
            self.folder_selected.emit(current.text(0))


class CheckList(QListWidget):
    """A list whose rows are ticked rather than highlighted.

    Highlighting cannot say what it means: it is also how Qt shows the row you
    are looking at, it is lost the moment the list is rebuilt, and in a list
    that scrolls there is no way to see what is chosen without scrolling
    through looking for it. A tick is none of those things.

    Rows are addressed by position, because the caller holds the mapping from a
    row to whatever it stands for -- a path, a metadata value -- and a list
    widget should not be a second copy of it.
    """

    #: Emitted with the row and whether it is now ticked. Not emitted for the
    #: ticks the list puts back itself when its contents are replaced.
    row_ticked = Signal(int, bool)

    def __init__(self) -> None:
        super().__init__()
        self._filling = False
        self.itemChanged.connect(self._item_changed)

    def set_values(
        self,
        values: Sequence[str],
        *,
        ticked: Sequence[bool] = (),
        icons: Sequence[QIcon] | None = None,
        tooltips: Sequence[str] | None = None,
        current: int | None = None,
    ) -> None:
        """Replace the rows, putting the ticks back without reporting them.

        Every refresh rebuilds the rows, so the ticks have to be supplied: they
        live in whatever holds the state, keyed by something more stable than a
        row number, precisely so that rebuilding cannot lose them.

        Args:
            values: The row labels, in order.
            ticked: Whether each row is ticked, by position. Missing entries
                are unticked.
            icons: A chip for each row, where the caller draws one.
            tooltips: Hover text for each row. Rows carrying paths elide, so
                this is where the whole label goes.
            current: The row to make current, for a list where that means
                something. None leaves it alone.

        Reports nothing while it runs. Replacing the contents is a redraw, not
        a gesture: an owner that rebuilds its list in response to the current
        row changing would otherwise drive itself round in a circle.
        """
        self._filling = True
        self.blockSignals(True)
        try:
            self.clear()
            for row, value in enumerate(values):
                item = QListWidgetItem(value)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.Checked
                    if row < len(ticked) and ticked[row]
                    else Qt.Unchecked
                )
                if icons is not None and row < len(icons):
                    item.setIcon(icons[row])
                if tooltips is not None and row < len(tooltips):
                    item.setToolTip(tooltips[row])
                self.addItem(item)
            if current is not None and values:
                self.setCurrentRow(max(0, min(current, len(values) - 1)))
        finally:
            self.blockSignals(False)
            self._filling = False

    def show_ticks(self, ticked: Sequence[bool]) -> None:
        """Put the ticks back without rebuilding the rows.

        Rebuilding would reset which row is current, and a tick is not a change
        of what is being looked at.
        """
        self._filling = True
        try:
            for row in range(self.count()):
                self.item(row).setCheckState(
                    Qt.Checked
                    if row < len(ticked) and ticked[row]
                    else Qt.Unchecked
                )
        finally:
            self._filling = False

    def ticked_rows(self) -> list[int]:
        """Return the positions of the ticked rows."""
        return [
            row for row in range(self.count())
            if self.item(row).checkState() == Qt.Checked
        ]

    def ticked_values(self) -> list[str]:
        """Return the labels of the ticked rows."""
        return [self.item(row).text() for row in self.ticked_rows()]

    def hide_rows_without(self, text: str, labels: Sequence[str] | None = None) -> None:
        """Hide rows not containing *text*.

        Hidden rather than removed, so a row number still means the same thing
        whether or not a filter is active. *labels* supplies what to match
        against where the row text is elided or decorated.
        """
        needle = text.strip().lower()
        for row in range(self.count()):
            item = self.item(row)
            label = labels[row] if labels and row < len(labels) else item.text()
            item.setHidden(bool(needle) and needle not in label.lower())

    def _item_changed(self, item) -> None:
        """Report a tick, ignoring the ones this widget makes itself."""
        if self._filling:
            return
        self.row_ticked.emit(self.row(item), item.checkState() == Qt.Checked)
