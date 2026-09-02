"""Small widget behaviours shared between panels.

What lands here is something more than one panel *does* -- a context menu, an
event filter, asking the user for a path, the two list widgets every tab picks
rows in. What the window looks like belongs in :mod:`theme`, anything that
knows about projects, labels or the filesystem belongs in a controller, and
anything only one panel uses belongs in that panel.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import Sequence
from pathlib import Path

from qtpy.QtCore import (
    QBuffer,
    QByteArray,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QSplitterHandle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig
from . import theme
from .theme import SWATCH_HEIGHT, SWATCH_WIDTH

logger = logging.getLogger(__name__)

#: The name filters the CSV dialogs offer. ``All files`` is offered beside the
#: CSV filter when opening, because a CSV that arrived with another extension is
#: still readable and refusing to show it would be the dialog deciding.
CSV_FILTER = "CSV files (*.csv)"
ALL_FILES_FILTER = "All files (*)"


#: What each label status is called, and how strongly it is said. Shared by the
#: curation and proofreading panels: the two report the same fact about the same
#: file, and a second wording for it would read as a second meaning.
LABEL_BADGES = {
    "missing": ("no label", "warning"),
    "empty": ("empty label - not yet annotated", "warning"),
    "present": ("", "info"),
}


def class_legend_html(config: SegmentationConfig) -> str:
    """Return a swatch and name for each foreground class, as rich text.

    Background is left out: it covers most of a frame and is never drawn, so
    naming it in the legend would describe a colour nothing uses.
    """
    parts = []
    for class_index in config.foreground_classes:
        red, green, blue = config.rgb(class_index)
        parts.append(
            f'<span style="color: rgb({red},{green},{blue});">&#9632;</span> '
            f"{config.name(class_index)}"
        )
    return "&nbsp;&nbsp;".join(parts)


def chip_icon(pixmap: QPixmap) -> QIcon:
    """Return an icon Qt will draw the same way on every row.

    A view draws the icon of the row being looked at in ``QIcon.Selected``
    mode, and the variant Qt generates for that mode blends the pixmap with the
    highlight colour. For a status chip that is wrong twice over: the colour
    *is* the information, and it gets washed out on exactly the row the user is
    on -- a "done" teal came out as a blue-green against a blue-tinted row.

    Supplying the pixmap for every mode and state leaves nothing for Qt to
    generate.
    """
    icon = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active, QIcon.Disabled):
        for state in (QIcon.On, QIcon.Off):
            icon.addPixmap(pixmap, mode, state)
    return icon


def mark_image(pixmap: QPixmap) -> str:
    """Return *pixmap* as an ``<img>`` tag Qt's rich text will draw inline.

    So that a legend shows the mark itself rather than a description of it: the
    pixmap comes from the same painter that draws the real thing, and the two
    cannot drift apart. The bytes travel in the tag as a ``data:`` URI, there
    being no file to point at and ``QLabel`` offering no document to attach a
    resource to.

    The byte array is held in a name rather than passed as a temporary:
    ``QBuffer(QByteArray())`` lets the array die under the buffer and segfaults
    PySide6 on the write.
    """
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.WriteOnly)
    try:
        pixmap.save(buffer, "PNG")
    finally:
        buffer.close()
    encoded = base64.b64encode(bytes(data)).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}">'



#: Where a project row keeps the number the controller knows it by. A tree row
#: is not a row number -- the folders take positions too -- so the number
#: travels on the item rather than being counted off the widget.
PROJECT_INDEX = Qt.UserRole + 1

#: How far a project is indented under its folder. Enough to read as beneath it
#: and no more: the panel is narrow and the rows carry names that need the room.
TREE_INDENT = 12


def add_fold_buttons(row: QHBoxLayout, tree: ProjectTree) -> QHBoxLayout:
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
        # Or Qt scales the chip to a text line's height and the six cells
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

#: Monospaced families to try, in the order they are looked for. The style hint
#: below is only consulted for a family that does not exist, so the list has to
#: carry a real name for each platform rather than relying on the hint alone.
_FIXED_WIDTH_FAMILIES = ("Menlo", "Consolas", "DejaVu Sans Mono", "Courier New")


def fixed_width_font() -> QFont:
    """Return a font whose characters are all the same width.

    Anything padded into columns with spaces -- the breakdown table, the
    operation reports -- is only aligned in a font like this.

    Not ``QFontDatabase.systemFont(FixedFont)``, which is the obvious call and
    returns the proportional UI font on macOS: it reports the family it was
    asked for while resolving to something else, so the padding stays padding
    and nothing lines up. Naming families and letting the style hint cover the
    ones that are missing is what actually resolves.
    """
    font = QFont()
    font.setFamilies(list(_FIXED_WIDTH_FAMILIES))
    font.setStyleHint(QFont.Monospace)
    font.setFixedPitch(True)
    return font


def horizontal_rule() -> QFrame:
    """Return a hairline for dividing one group of controls from the next.

    Painted rather than left to the style, which draws a sunken frame as two
    lines of its own choosing -- a groove cut across a panel whose every other
    edge is flat. :data:`theme.BORDER` rather than the lighter separator: this
    rule is the only thing saying the groups differ, and against a group box
    the light one cannot be seen at all.
    """
    rule = QFrame()
    rule.setFrameShape(QFrame.HLine)
    rule.setFrameShadow(QFrame.Plain)
    rule.setFixedHeight(1)
    rule.setStyleSheet(f"background: {theme.BORDER}; border: none;")
    return rule


def caption_for(text: str, widget: QWidget, tooltip: str) -> QLabel:
    """Return a caption for *widget*, putting *tooltip* on both.

    A user reaches for the word naming a control, not the control itself, so a
    tooltip on the control alone is one nobody finds. Returning the label keeps
    the two impossible to write apart.
    """
    label = QLabel(text)
    label.setToolTip(tooltip)
    widget.setToolTip(tooltip)
    return label


def ticked_caption(noun: str, total: int, ticked: int) -> str:
    """Return a list's caption, saying how many of its rows are ticked.

    Spelled "3 of 12 ticked" rather than as two bare numbers: the ticks decide
    what an operation covers, the lists scroll, and a reader should not have to
    work out which number is which.
    """
    if not total:
        return noun
    if ticked:
        return f"{noun} ({ticked} of {total} ticked)"
    return f"{noun} ({total})"


def match_anywhere(box: QComboBox) -> None:
    """Have *box* suggest on any part of a value, and let Tab take the suggestion.

    **The popup is not cosmetic.** An editable ``QComboBox`` builds its
    completer in ``InlineCompletion`` mode, which assumes the match is a prefix
    and keeps what was typed while appending the rest of the completion. Set to
    match anywhere and left inline, typing ``a`` against "UltrAuFoil Au R2/2 200
    mesh" writes "altrAuFoil Au R2/2 200 mesh" into the field -- a plausible
    value nobody typed, in metadata that is exported. A popup replaces the whole
    text instead, which is what makes matching anywhere safe.

    **Tab has to be taught to accept**, which is the other half of the same
    fault. A completer's popup answers to Return and to a click; Tab is not its
    key, so it reaches the field, moves the focus, and leaves whatever was
    half-typed behind as the value -- one letter, committed as a chip nobody
    meant. So Tab takes the suggestion on show and stays in the field.

    **The filter goes on the popup, and on the field.** An open popup grabs the
    keyboard, so the key arrives at the popup; ``QCompleter`` then passes it on
    by calling ``widget->event()``, which reaches the field's own handlers but
    **not the event filters installed on it**. Watching only the field is
    therefore watching the one route the key does not take -- except under the
    offscreen platform, which cannot grab a keyboard and so delivers straight to
    the field. Both are watched, because a test that only proves the second
    proves nothing about the window.

    Matching anywhere is worth both: "Quantifoil" and "UltrAuFoil" are told
    apart by their middles, not by how they start.
    """
    completer = box.completer()
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    completer.setFilterMode(Qt.MatchContains)
    completer.setCompletionMode(QCompleter.PopupCompletion)

    taking = _CompleteOnTab(box)
    line = box.lineEdit()
    if line is not None:
        line.installEventFilter(taking)
    # Creates the popup if it does not exist yet, which is what makes it
    # filterable now rather than on first use. It outlives the suggestions
    # being replaced, so it is installed on once.
    completer.popup().installEventFilter(taking)


class _CompleteOnTab(QObject):
    """Makes Tab take what the completer's popup is offering.

    Installed on both the popup and the field, one of which gets the key
    depending on whether the platform grants the popup a keyboard grab. Parented
    to the box, an event filter being kept alive by whoever installed it and by
    nothing else.
    """

    def __init__(self, box: QComboBox) -> None:
        super().__init__(box)
        self._box = box

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        """Take the suggestion on Tab, and let every other key through."""
        if event.type() != QEvent.KeyPress or event.key() != Qt.Key_Tab:
            return False
        completer = self._box.completer()
        if completer is None or not completer.popup().isVisible():
            return False
        chosen = self._offered(completer)
        if not chosen:
            return False
        completer.popup().hide()
        self._box.setCurrentText(chosen)
        # Hidden again after setting the text: writing to the field is an edit
        # like any other, so the completer answers it by offering the value it
        # has just been given.
        completer.popup().hide()
        line = self._box.lineEdit()
        if line is not None:
            line.setCursorPosition(len(chosen))
        return True

    @staticmethod
    def _offered(completer: QCompleter) -> str:
        """Return the row the popup is on, or the first it would offer.

        A popup that has just appeared has no current row until someone arrows
        into it, and that is exactly when Tab is pressed.
        """
        index = completer.popup().currentIndex()
        if index.isValid():
            return str(index.data())
        return completer.currentCompletion()


class ToggleSwitch(QCheckBox):
    """A standalone on/off control, drawn as a switch rather than a tick box.

    **A subclass of** :class:`QCheckBox` **on purpose.** Only the painting
    differs, so ``isChecked``, ``setChecked``, ``toggled``, the caption, Space,
    focus and the disabled state all behave exactly as they did -- which is
    what let every control that should be a switch become one by changing a
    word, with nothing else about any panel touched.

    For a control that is on or off by itself. A row of a list is ticked, not
    switched: see :class:`CheckList`, where the tick means something a switch
    cannot say.
    """

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the track, the knob on it, and the caption beside it.

        Everything, rather than deferring to ``QStyle``: half a switch drawn
        here and half by the style would be two controls on top of each other.
        """
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            track = self._track()
            radius = track.height() / 2

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(self._track_colour()))
            painter.drawRoundedRect(track, radius, radius)

            if self.hasFocus():
                # A control reachable by keyboard has to show where the
                # keyboard is; nothing else here would say so.
                ring = track.adjusted(-2.5, -2.5, 2.5, 2.5)
                pen = QPen(QColor(theme.ACCENT))
                pen.setWidthF(1.5)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SWITCH_KNOB))
            painter.drawEllipse(self._knob(track))

            painter.setPen(QColor(self._text_colour()))
            painter.drawText(
                self._caption(), Qt.AlignLeft | Qt.AlignVCenter, self.text()
            )
        finally:
            painter.end()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return the room the switch, its focus ring and its caption need."""
        metrics = QFontMetrics(self.font())
        width = theme.SWITCH_WIDTH + 2 * theme.SWITCH_FOCUS
        if self.text():
            width += theme.SWITCH_GAP + metrics.horizontalAdvance(self.text())
        height = max(theme.SWITCH_HEIGHT + 2 * theme.SWITCH_FOCUS, metrics.height())
        return QSize(int(width), int(height))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return self.sizeHint()

    def _track(self) -> QRectF:
        """Return the rounded bar, centred on the widget's height.

        Indented by :data:`theme.SWITCH_FOCUS`, which is the room the focus
        ring is drawn into.
        """
        top = (self.height() - theme.SWITCH_HEIGHT) / 2
        return QRectF(
            theme.SWITCH_FOCUS, top, theme.SWITCH_WIDTH, theme.SWITCH_HEIGHT
        )

    def _knob(self, track: QRectF) -> QRectF:
        """Return the circle on the track, at whichever end it belongs."""
        size = track.height() - 2 * theme.SWITCH_INSET
        left = (
            track.right() - theme.SWITCH_INSET - size
            if self.isChecked()
            else track.left() + theme.SWITCH_INSET
        )
        return QRectF(left, track.top() + theme.SWITCH_INSET, size, size)

    def _caption(self) -> QRectF:
        """Return where the text goes, to the right of the track."""
        left = theme.SWITCH_FOCUS + theme.SWITCH_WIDTH + theme.SWITCH_GAP
        return QRectF(left, 0.0, max(0.0, self.width() - left), float(self.height()))

    def _track_colour(self) -> str:
        if not self.isEnabled():
            return theme.BORDER_LIGHT
        return theme.ACCENT if self.isChecked() else theme.SWITCH_OFF

    def _text_colour(self) -> str:
        return theme.TEXT if self.isEnabled() else theme.DISABLED_TEXT


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


class GripSplitter(QSplitter):
    """A splitter whose handles show that they can be dragged.

    Qt draws a divider that looks the same whether or not it moves, so the only
    way to find out was to sweep the pointer along it until the cursor changed.
    The handle carries a grip instead, and takes the accent under the pointer
    like every other control that reacts to being pointed at.
    """

    def __init__(self, orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self.setHandleWidth(theme.HANDLE_WIDTH)

    def createHandle(self) -> QSplitterHandle:  # noqa: N802 - Qt naming
        """Return a handle that paints a grip."""
        return _GripHandle(self.orientation(), self)


class _GripHandle(QSplitterHandle):
    """One divider of a :class:`GripSplitter`, drawn as a grip.

    Painted rather than styled: a stylesheet can put a flat colour or an image
    file on a handle and nothing else, and this package ships no images.
    """

    def __init__(self, orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self._pointed_at = False

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._pointed_at = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._pointed_at = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the handle as the window's surface with a grip on it.

        The grip takes the accent while the handle is pointed at; the surface
        under it is left alone. Filling the whole band instead draws a coloured
        column the height of the window to answer "this line moves", which the
        cursor has said already by the time anyone is close enough to see it.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.WINDOW))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.ACCENT if self._pointed_at else theme.GRIP))
        painter.drawRoundedRect(
            self._grip(), theme.GRIP_THICKNESS / 2, theme.GRIP_THICKNESS / 2
        )

    def _grip(self) -> QRectF:
        """Return the bar to draw, centred in the handle and along it.

        A horizontal splitter divides its children left from right, so its
        handle is the upright bar and the grip runs down it.
        """
        thickness, length = theme.GRIP_THICKNESS, theme.GRIP_LENGTH
        if self.orientation() == Qt.Horizontal:
            width, height = thickness, length
        else:
            width, height = length, thickness
        centre = QPointF(self.rect().center()) + QPointF(0.5, 0.5)
        return QRectF(
            centre.x() - width / 2, centre.y() - height / 2, width, height
        )


class ElidingLabel(QLabel):
    """A label that shortens its text to fit rather than being cut off by it.

    Qt clips a label that is too narrow, and clipping takes the ends off
    without saying it did. This shows a middle ellipsis instead and keeps the
    whole line in the tooltip -- the same treatment the path lists get, for the
    same reason: the useful part of these strings is at both ends.

    Set the text through :meth:`setText` as usual; :meth:`full_text` returns
    what was set rather than what is shown.
    """

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        """Keep *text* whole, and show as much of it as there is room for."""
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        """Return the text as set, whatever is currently on screen."""
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        super().setText(
            self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, self.width())
        )


class StepCombo(QComboBox):
    """A combo box whose Up and Down keys always move through the list.

    Qt's own handling of those keys turns on ``SH_ComboBox_Popup``: where a
    style sets it, the keys open the popup instead of stepping. Under PyQt5
    here they step, but that hint is a style's to answer and this window runs
    under a different binding and a different macOS than it is developed on --
    the gap ``docs/display.md`` § The binding is about. One code path rather
    than a style's is the same choice already made for the file dialogs.

    Stepping skips disabled rows, so a grouped list arrows through its entries
    and not its headings. Alt-Down still opens the popup, and every other key
    is Qt's.
    """

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and not event.modifiers():
            self.step(1 if event.key() == Qt.Key_Down else -1)
            event.accept()
            return
        super().keyPressEvent(event)

    def step(self, delta: int) -> None:
        """Move to the next selectable row in *delta*'s direction.

        Stops at the ends rather than wrapping: a list that jumps from the
        last row back to the first loses your place in a long one.
        """
        wanted = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        index = self.currentIndex() + delta
        while 0 <= index < self.count():
            # Asked of the model rather than the item, so this works whatever
            # model the combo was given.
            if self.model().index(index, self.modelColumn()).flags() & wanted == wanted:
                self.setCurrentIndex(index)
                return
            index += delta


def fill_choice_combo(
    combo: QComboBox,
    entries: Sequence[tuple[str, object]],
    current: object,
) -> None:
    """Offer *entries* in *combo*, select *current*, and grey it if there is one.

    Rebuilt only when the entries have actually changed. The view controls are
    refilled on every step through a lamella, and a combo cleared that often
    closes its own popup under the pointer.

    Args:
        entries: ``(caption, value)`` pairs, the value being what a reader
            takes from ``itemData`` and never has to map back from a caption.
        current: The value to show. A value not among *entries* falls to the
            first, so a caller need not check.
    """
    entries = list(entries)
    held = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
    combo.blockSignals(True)
    if held != entries:
        combo.clear()
        for caption, value in entries:
            combo.addItem(caption, value)
    values = [value for _caption, value in entries]
    combo.setCurrentIndex(values.index(current) if current in values else 0)
    combo.blockSignals(False)
    # Greyed rather than hidden where there is nothing to choose: the control
    # still says what is on screen, and the row beside it does not move as the
    # reviewer steps onto a lamella that has only one.
    combo.setEnabled(len(entries) > 1)


def fill_grouped_combo(
    combo: QComboBox,
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    first: tuple[str, str] | None = None,
) -> None:
    """Fill *combo* with rows gathered under non-selectable headings.

    Each row carries its identifier as item data, so a caller reads
    ``currentData()`` and never has to map a display label back to a name.
    A heading carries none, which is what distinguishes the two.

    Args:
        groups: ``(heading, rows)`` pairs, where each row is
            ``(identifier, label)``.
        first: A row to place above every group, for an entry that belongs to
            no group -- an "any" or "none" option.
    """
    model = QStandardItemModel(combo)
    if first is not None:
        name, label = first
        item = QStandardItem(label)
        item.setData(name, Qt.UserRole)
        model.appendRow(item)
    for heading, rows in groups:
        item = QStandardItem(heading)
        # NoItemFlags rather than clearing ItemIsSelectable alone: a heading
        # must also be skipped when stepping through the list with the arrow
        # keys, and Qt skips disabled rows for free.
        item.setFlags(Qt.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        model.appendRow(item)
        for name, label in rows:
            row = QStandardItem(label)
            row.setData(name, Qt.UserRole)
            model.appendRow(row)
    combo.setModel(model)


def build_path_dialog(
    parent: QWidget | None,
    caption: str,
    *,
    mode: QFileDialog.FileMode,
    save: bool = False,
    directory: Path | str | None = None,
    filters: tuple[str, ...] = (),
    default_suffix: str | None = None,
) -> QFileDialog:
    """Return a configured file dialog, without showing it.

    Every dialog in the window is built here, non-native, so that one binding
    and one platform cannot behave differently from another; see
    ``docs/display.md`` § The binding.

    Args:
        mode: What the user is picking -- a file, or a directory.
        save: Whether the dialog names a file to write rather than one to read.
        directory: Where the dialog opens. Falls back to the home directory.
        filters: Name filters, the first of which is selected.
        default_suffix: Appended when a save name is typed without one, so a
            file cannot be written with no extension.
    """
    dialog = QFileDialog(parent, caption)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(mode)
    dialog.setAcceptMode(QFileDialog.AcceptSave if save else QFileDialog.AcceptOpen)
    if mode == QFileDialog.Directory:
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
    if filters:
        dialog.setNameFilters(list(filters))
    if default_suffix:
        dialog.setDefaultSuffix(default_suffix)
    dialog.setDirectory(str(directory) if directory else str(Path.home()))
    return dialog


def _chosen(dialog: QFileDialog) -> Path | None:
    """Show *dialog* and return what was picked, or None if it was cancelled."""
    if not dialog.exec_():
        return None
    selected = dialog.selectedFiles()
    return Path(selected[0]) if selected else None


def choose_open_file(
    parent: QWidget | None,
    caption: str,
    directory: Path | str | None = None,
    filters: tuple[str, ...] = (CSV_FILTER, ALL_FILES_FILTER),
) -> Path | None:
    """Ask for an existing file to read."""
    return _chosen(
        build_path_dialog(
            parent,
            caption,
            mode=QFileDialog.ExistingFile,
            directory=directory,
            filters=filters,
        )
    )


def choose_save_file(
    parent: QWidget | None,
    caption: str,
    directory: Path | str | None = None,
    filters: tuple[str, ...] = (CSV_FILTER,),
    default_suffix: str = "csv",
) -> Path | None:
    """Ask for a file to write, confirming a replacement if one is picked."""
    return _chosen(
        build_path_dialog(
            parent,
            caption,
            mode=QFileDialog.AnyFile,
            save=True,
            directory=directory,
            filters=filters,
            default_suffix=default_suffix,
        )
    )


def choose_directory(
    parent: QWidget | None, caption: str, directory: Path | str | None = None
) -> Path | None:
    """Ask for a directory."""
    return _chosen(
        build_path_dialog(
            parent, caption, mode=QFileDialog.Directory, directory=directory
        )
    )


def add_path_menu(
    widget, noun: str, copy_signal, reveal_signal, row_of=None
) -> None:
    """Give a list or tree a right-click menu for the row's filesystem path.

    The row under the cursor is used, not the selected row: right-clicking an
    entry and acting on a different one would be a surprising way to lose track
    of which path you copied. The row number is emitted rather than a path,
    because resolving one is the controller's job, not a widget's.

    Args:
        row_of: How to turn the item under the cursor into the number the
            controller knows it by, for a widget where that is not its position.
            Returning None offers no menu, which is what a tree's folder rows
            need: a folder is not a project and has no path to copy.
    """
    widget.setContextMenuPolicy(Qt.CustomContextMenu)
    resolve = row_of if row_of is not None else widget.row

    def show_menu(point) -> None:
        item = widget.itemAt(point)
        if item is None:
            return
        row = resolve(item)
        if row is None:
            return
        menu = QMenu(widget)
        copy = menu.addAction(f"Copy {noun} path")
        reveal = menu.addAction(f"Reveal {noun} in file manager")
        chosen = menu.exec_(widget.viewport().mapToGlobal(point))
        if chosen is copy:
            copy_signal.emit(row)
        elif chosen is reveal:
            reveal_signal.emit(row)

    widget.customContextMenuRequested.connect(show_menu)


class FlowLayout(QLayout):
    """A layout that puts its items in a row and wraps onto the next.

    Qt ships no such layout. This is needed because a field holding several
    values holds however many the user gives it, of whatever length: laid out
    in one row they run off the pane, and in a column each takes a line of the
    form to itself.
    """

    def __init__(self, spacing: int = 4) -> None:
        super().__init__()
        self._items: list[QLayoutItem] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # -- what QLayout requires ---------------------------------------------

    def addItem(self, item) -> None:  # noqa: N802 - Qt naming
        self._items.append(item)

    def addSpacing(self, size: int) -> None:  # noqa: N802 - Qt naming
        """Add a fixed gap, separating one group of items from the next.

        ``QBoxLayout`` has this and ``QLayout`` does not, so a row moved here
        from a box layout would otherwise lose the gaps that group it. A gap
        landing at the start of a wrapped line is left as it falls: nudging it
        would mean knowing where the wrap will be, which is the width's answer
        rather than this one's.
        """
        self.addItem(QSpacerItem(size, 0, QSizePolicy.Fixed, QSizePolicy.Minimum))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt naming
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt naming
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt naming
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self._lay_out(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:  # noqa: N802 - Qt naming
        super().setGeometry(rect)
        self._lay_out(rect, apply=True)

    def sizeHint(self):  # noqa: N802 - Qt naming
        """Return the room the items need in one row, before any wrapping.

        The width this would *like*. Given it, nothing wraps; given less, it
        wraps down to :meth:`minimumSize`. The two have to differ: made equal,
        one is a floor and the layout can only ever force its container wide.
        """
        margins = self.contentsMargins()
        width = sum(item.sizeHint().width() for item in self._items)
        width += self.spacing() * max(len(self._items) - 1, 0)
        return QSize(width + margins.left() + margins.right(), 0).expandedTo(
            self.minimumSize()
        )

    def minimumSize(self):  # noqa: N802 - Qt naming
        """Return one row's height, and no width at all.

        No width because a minimum is a floor: asked for the widest item, the
        pane could never be narrower than the longest value in it, and a
        layout that wraps is exactly the one that should not force that. The
        height is what :meth:`heightForWidth` replaces once there is a width to
        wrap against.
        """
        height = max((item.sizeHint().height() for item in self._items), default=0)
        margins = self.contentsMargins()
        return QSize(0, height + margins.top() + margins.bottom())

    # -- the arithmetic ----------------------------------------------------

    def _lay_out(self, rect, *, apply: bool) -> int:
        """Place the items across *rect*, and return the height they need."""
        margins = self.contentsMargins()
        left = rect.x() + margins.left()
        right = rect.right() - margins.right()
        x, y = left, rect.y() + margins.top()
        row_height = 0

        for item in self._items:
            # Clamped to the row: an item wider than the space it has spills
            # past the edge otherwise, where nothing clips it and it is drawn
            # over whatever sits beside the layout.
            size = item.sizeHint().boundedTo(QSize(right - left + 1, 1 << 20))
            if row_height and x + size.width() > right:
                x = left
                y += row_height + self.spacing()
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), size))
            x += size.width() + self.spacing()
            row_height = max(row_height, size.height())

        return y + row_height - rect.y() + margins.bottom()


class FlowArea(QWidget):
    """A widget whose height is whatever the flow layout inside it needs.

    Height-for-width alone is not enough inside a form: the form asks how tall
    the widget would be at one width and then hands it another, and the rows
    that do not fit are laid out below the bottom edge -- present,
    addressable, and invisible. Answering from the width it actually has,
    every time it gets one, is the only account that cannot disagree with
    itself.

    Used for a field holding several values and for the image panel's row of
    controls, which are the same problem: however many things there are, at
    whatever width the pane happens to have.
    """

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self.sync_height()

    def sync_height(self) -> None:
        """Take the height this widget's contents need at its current width."""
        layout = self.layout()
        if layout is not None:
            self.setFixedHeight(layout.heightForWidth(self.width()))


class ChipEditor(QWidget):
    """Several values at once, each a chip that can be taken off.

    For a field that holds more than one thing -- two organisations, a sample
    that is a host and the pathogen in it. A chip apiece rather than one line
    of separated text, because where one value ends and the next begins is the
    whole point, and a separator typed by hand is a separator got wrong.

    Values are read and written as a list; nothing outside this class needs to
    know what separates them on screen.
    """

    #: Emitted whenever the values change through the widget. Not emitted by
    #: :meth:`set_values`, which is a redraw rather than an edit.
    changed = Signal()

    def __init__(self, separator: str = "; ", placeholder: str = "add a value") -> None:
        """Build an editor whose typed text is split on *separator*."""
        super().__init__()
        self._separator = separator.strip()
        self._values: list[str] = []

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(3)

        self._chips = FlowArea()
        self._chip_layout = FlowLayout()
        self._chips.setLayout(self._chip_layout)
        column.addWidget(self._chips)

        self.entry = QComboBox()
        self.entry.setEditable(True)
        self.entry.setInsertPolicy(QComboBox.NoInsert)
        self.entry.lineEdit().setPlaceholderText(placeholder)
        match_anywhere(self.entry)
        # Three ways in, and the third is the one that matters: Return, picking
        # a value already in use, and simply leaving the box. Without that last
        # one a value typed and left there is discarded by the next thing that
        # reads the editor -- which is Save, the very next thing the user does.
        # `editingFinished` covers Return and focus alike; `activated` fires
        # only on a choice the user made, where `currentIndexChanged` also
        # fires as the list is refilled.
        self.entry.lineEdit().editingFinished.connect(self._commit_entry)
        self.entry.activated.connect(lambda _index: self._commit_entry())
        column.addWidget(self.entry)

    # -- what it holds -----------------------------------------------------

    def values(self) -> list[str]:
        """Return the values, in the order they were added."""
        return list(self._values)

    def set_values(self, values: Sequence[str]) -> None:
        """Replace the values, reporting nothing."""
        self._values = [value for value in dict.fromkeys(values) if value]
        self._redraw()

    def set_suggestions(self, values: Sequence[str]) -> None:
        """Offer *values* in the box, keeping whatever is half-typed in it."""
        typed = self.entry.currentText()
        self.entry.clear()
        self.entry.addItems(["", *values])
        self.entry.setCurrentText(typed)

    # -- editing -----------------------------------------------------------

    def commit_pending(self) -> None:
        """Turn anything still in the box into a value.

        Called by whoever is about to read this editor, so that a value typed
        and not confirmed is never quietly dropped. Doing nothing is the normal
        case, the box being empty most of the time.
        """
        self._commit_entry()

    def _commit_entry(self) -> None:
        """Turn whatever is in the box into chips, and empty it."""
        text = self.entry.currentText()
        added = [part.strip() for part in text.split(self._separator)]
        wanted = [part for part in added if part and part not in self._values]
        self.entry.setCurrentText("")
        if not wanted:
            return
        self._values.extend(wanted)
        self._redraw()
        self.changed.emit()

    def _remove(self, value: str) -> None:
        if value not in self._values:
            return
        self._values.remove(value)
        self._redraw()
        self.changed.emit()

    def _redraw(self) -> None:
        """Rebuild the chips from the values."""
        while (item := self._chip_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        # Shown before the chips go in, because a widget cannot be shown into a
        # hidden parent and a hidden widget measures as nothing.
        self._chips.setVisible(bool(self._values))
        for value in self._values:
            chip = QPushButton(f"{value}  ✕")
            chip.setStyleSheet(theme.VALUE_CHIP_STYLE)
            chip.setToolTip(f"Remove {value}")
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _checked=False, v=value: self._remove(v))
            self._chip_layout.addWidget(chip)
            # Adding a widget to a layout reparents it, and reparenting hides
            # it. Qt shows it again on the next pass through the event loop --
            # but the height is wanted now, and a layout of hidden widgets
            # measures zero, so the row of chips gets no height at all and the
            # values are held, addressable, and invisible.
            chip.setVisible(True)
        # The row count changes with the values, and the form has to be told to
        # make room rather than clipping the new row.
        self._chips.sync_height()
        self.updateGeometry()


#: The keyboard the F1 diagram draws, row by row. A compact layout rather than
#: a full one: no key beyond these is bound anywhere in the window, and rows of
#: function keys nothing uses are rows the eye has to cross to reach the ones
#: that matter. ``None`` is a gap.
KEY_ROWS: tuple[tuple[str, ...], ...] = (
    ("Esc", "F1"),
    ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"),
    ("Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]"),
    ("A", "S", "D", "F", "G", "H", "J", "K", "L"),
    ("Shift", "Z", "X", "C", "V", "B", "N", "M"),
    ("Ctrl", "Space"),
)

#: How wide a key is, in units of :data:`KEY_SIZE`. Anything unnamed is one.
KEY_WIDTHS: dict[str, float] = {"Esc": 1.4, "Shift": 1.9, "Ctrl": 1.6, "Space": 5.0}

KEY_SIZE = 21
KEY_GAP = 3
KEY_RADIUS = 3.0


def _key_width(name: str) -> int:
    """Return how wide the key called *name* is drawn."""
    return int(round(KEY_SIZE * KEY_WIDTHS.get(name, 1.0)))


def _row_width(row: tuple[str, ...]) -> int:
    """Return how wide a row of keys is, gaps included."""
    return sum(_key_width(name) for name in row) + KEY_GAP * (len(row) - 1)


def keyboard_diagram(pressed: set[str]) -> QPixmap:
    """Return a keyboard with the keys in *pressed* picked out.

    Drawn rather than loaded. The tool marks in `label_panel` are painted the
    same way, and for the same reasons: it takes the window's own colours, it
    is right at any display scale, and there is no asset to keep in step with
    the bindings. What it shows comes from the bindings themselves, so a key
    added to the window appears here without anything else being edited.

    One tint for every bound key rather than a colour per group. The window has
    exactly one accent because a second would have to mean something, and what
    each key *does* is the table under this -- the picture answers only "which
    of these does anything here", which is the question a diagram is better at
    than a list.

    Args:
        pressed: Key names as :data:`KEY_ROWS` spells them, already broken out
            of any chord.
    """
    arrows = _arrow_block_width()
    width = max(_row_width(row) for row in KEY_ROWS) + KEY_GAP * 2 + arrows
    height = len(KEY_ROWS) * (KEY_SIZE + KEY_GAP) - KEY_GAP

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 2))
        painter.setFont(font)
        for row_index, row in enumerate(KEY_ROWS):
            x = 0
            y = row_index * (KEY_SIZE + KEY_GAP)
            for name in row:
                _paint_key(painter, x, y, name, name in pressed)
                x += _key_width(name) + KEY_GAP
        _paint_arrows(painter, width - arrows, height, pressed)
    finally:
        # A painter outliving its pixmap crashes the process rather than
        # reporting it.
        painter.end()
    return pixmap


def _arrow_block_width() -> int:
    """Return how wide the arrow cluster is."""
    return 3 * KEY_SIZE + 2 * KEY_GAP


def _paint_arrows(painter: QPainter, left: int, height: int, pressed: set[str]) -> None:
    """Draw the arrow cluster in its own shape, bottom-aligned.

    Kept apart from the letters and drawn as the inverted T it is on a real
    keyboard: these four are the keys most used here, and finding them by
    shape is faster than reading a label.
    """
    step = KEY_SIZE + KEY_GAP
    bottom = height - KEY_SIZE
    _paint_key(painter, left + step, bottom - step, "Up", "Up" in pressed)
    for offset, name in enumerate(("Left", "Down", "Right")):
        _paint_key(painter, left + offset * step, bottom, name, name in pressed)


def _paint_key(painter: QPainter, x: int, y: int, name: str, used: bool) -> None:
    """Draw one key, tinted when the tab in front binds it."""
    painter.setPen(QColor(theme.ACCENT if used else theme.BORDER))
    painter.setBrush(QColor(theme.ACCENT if used else theme.BASE))
    painter.drawRoundedRect(
        QRectF(x, y, _key_width(name), KEY_SIZE), KEY_RADIUS, KEY_RADIUS
    )
    painter.setPen(QColor(theme.ACCENT_TEXT if used else theme.MUTED))
    painter.drawText(
        QRectF(x, y, _key_width(name), KEY_SIZE),
        Qt.AlignCenter,
        KEY_CAPTIONS.get(name, name),
    )


#: What a key is called on the drawing, where that differs from its binding.
KEY_CAPTIONS: dict[str, str] = {
    "Up": "↑", "Down": "↓", "Left": "←", "Right": "→",
}


#: The range slider's handle radius and groove, in logical pixels, and the room
#: kept clear for a focus ring the way :data:`theme.SWITCH_FOCUS` is -- a ring
#: painted past the widget's edge comes out clipped flat.
RANGE_HANDLE = 7
RANGE_GROOVE = 4
RANGE_FOCUS = 3.0

#: How wide the control asks to be. Narrower than the two separate sliders and
#: their caption were together, which the flow row it sits in needed.
RANGE_WIDTH = 150


class RangeSlider(QWidget):
    """One control for a low and a high limit, with a handle for each.

    Two sliders for one interval leave the reader working out which is which
    and let them be dragged past each other; one control shows the interval
    itself -- the filled span between the handles is the answer, and the empty
    ends are what is being thrown away.

    **Painted entirely here**, like :class:`ToggleSwitch`, so **no stylesheet
    rule may name it**: a rule would send it back through ``QStyle`` and draw a
    second control underneath this one.

    The limits are held apart rather than allowed to cross: a picture turning
    into a threshold reads as a fault instead of as a setting.
    """

    #: Emitted with the low and high limits whenever either moves.
    range_changed = Signal(int, int)

    def __init__(
        self,
        minimum: int,
        maximum: int,
        low: int,
        high: int,
        parent: QWidget | None = None,
    ) -> None:
        """Build the control over ``minimum..maximum``, resting at *low*, *high*.

        The resting pair is where a double-click on a handle puts it back:
        Qt offers no gesture for returning a slider to neutral, and reaching
        neutral by dragging is a thing nobody quite manages -- which for a
        control that changes what an image looks like means being left slightly
        off it without noticing.
        """
        super().__init__(parent)
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        self._resting = (int(low), int(high))
        self._low = int(low)
        self._high = int(high)
        #: Which handle the pointer took hold of, and which the keys move.
        self._active = "low"
        self._dragging = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)

    # -- the values --------------------------------------------------------

    def values(self) -> tuple[int, int]:
        """Return the low and high limits."""
        return self._low, self._high

    def set_range(self, low: int, high: int, *, notify: bool = True) -> None:
        """Set both limits, clamped into range and kept at least one apart.

        Where the two would cross, the *high* one gives way, the low being the
        one a caller states first.

        Args:
            notify: Clear to reflect a change made elsewhere without reporting
                it back and round again.
        """
        self._apply(int(low), int(high), moved="high", notify=notify)

    def reset(self) -> None:
        """Put both limits back where they rest, reporting the change."""
        self.set_range(*self._resting)

    def _apply(self, low: int, high: int, *, moved: str, notify: bool = True) -> None:
        """Take a new pair, held apart and clamped, and report it if it moved."""
        low, high = self._held_apart(low, high, moved=moved)
        if (low, high) == (self._low, self._high):
            return
        self._low, self._high = low, high
        self.update()
        if notify:
            self.range_changed.emit(low, high)

    def _held_apart(self, low: int, high: int, *, moved: str) -> tuple[int, int]:
        """Return the pair clamped into range and kept at least one apart.

        Whichever handle was *moved* gives way, so a handle dragged into the
        other stops rather than pushing it along -- being pushed reads as the
        control having taken hold of the wrong one.
        """
        low = max(self._minimum, min(low, self._maximum))
        high = max(self._minimum, min(high, self._maximum))
        if low >= high:
            if moved == "low":
                low = max(self._minimum, high - 1)
            else:
                high = min(self._maximum, low + 1)
        return low, high

    # -- the pointer -------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Take hold of whichever handle is nearer the press."""
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        x = _event_x(event)
        self._active = (
            "low"
            if abs(x - self._x_of(self._low)) <= abs(x - self._x_of(self._high))
            else "high"
        )
        self._dragging = True
        self._drag_to(x)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Move the handle that was taken hold of."""
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        self._drag_to(_event_x(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Let go."""
        self._dragging = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Put the nearer handle back where it rests, leaving the other alone."""
        x = _event_x(event)
        near_low = abs(x - self._x_of(self._low)) <= abs(x - self._x_of(self._high))
        # Not dragging: the press that opened the double-click already took
        # hold, and leaving it held would drag the handle on the way out.
        self._dragging = False
        if near_low:
            self._apply(self._resting[0], self._high, moved="low")
        else:
            self._apply(self._low, self._resting[1], moved="high")
        event.accept()

    def _drag_to(self, x: float) -> None:
        """Move the active handle to the value at *x*."""
        value = self._value_at(x)
        if self._active == "low":
            self._apply(value, self._high, moved="low")
        else:
            self._apply(self._low, value, moved="high")

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Step the active handle, where the window has not claimed the key."""
        step = {Qt.Key_Left: -1, Qt.Key_Right: 1}.get(event.key())
        if step is None:
            super().keyPressEvent(event)
            return
        if self._active == "low":
            self._apply(self._low + step, self._high, moved="low")
        else:
            self._apply(self._low, self._high + step, moved="high")
        event.accept()

    # -- how it is drawn ---------------------------------------------------

    def _span(self) -> tuple[float, float]:
        """Return the x range the handles' centres move between."""
        return RANGE_HANDLE + RANGE_FOCUS, self.width() - RANGE_HANDLE - RANGE_FOCUS

    def _x_of(self, value: int) -> float:
        """Return where *value* sits along the track."""
        left, right = self._span()
        if self._maximum == self._minimum:  # pragma: no cover - a degenerate range
            return left
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return left + fraction * (right - left)

    def _value_at(self, x: float) -> int:
        """Return the value at *x* along the track."""
        left, right = self._span()
        if right <= left:  # pragma: no cover - not yet laid out
            return self._minimum
        fraction = (x - left) / (right - left)
        span = self._maximum - self._minimum
        return round(self._minimum + max(0.0, min(1.0, fraction)) * span)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        """Draw the groove, the span between the handles, and the handles."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            left, right = self._span()
            middle = self.height() / 2
            groove = QRectF(
                left, middle - RANGE_GROOVE / 2, right - left, RANGE_GROOVE
            )
            radius = RANGE_GROOVE / 2

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.BORDER))
            painter.drawRoundedRect(groove, radius, radius)

            low_x, high_x = self._x_of(self._low), self._x_of(self._high)
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawRoundedRect(
                QRectF(low_x, groove.top(), high_x - low_x, RANGE_GROOVE),
                radius, radius,
            )
            for name, x in (("low", low_x), ("high", high_x)):
                self._paint_handle(painter, x, middle, focused=name == self._active)
        finally:
            painter.end()

    def _paint_handle(
        self, painter: QPainter, x: float, middle: float, *, focused: bool
    ) -> None:
        """Draw one handle, ringed when it is the one the keys would move."""
        centre = QPointF(x, middle)
        if focused and self.hasFocus():
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawEllipse(centre, RANGE_HANDLE + RANGE_FOCUS, RANGE_HANDLE + RANGE_FOCUS)
        pen = QPen(QColor(theme.BORDER))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(QColor(theme.BASE))
        painter.drawEllipse(centre, RANGE_HANDLE, RANGE_HANDLE)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return the room the track, the handles and a focus ring need."""
        side = int(2 * (RANGE_HANDLE + RANGE_FOCUS))
        return QSize(RANGE_WIDTH, side)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return enough for both handles side by side and nothing more."""
        side = int(2 * (RANGE_HANDLE + RANGE_FOCUS))
        return QSize(4 * side, side)


def _event_x(event) -> float:
    """Return a mouse event's x, under either binding."""
    position = getattr(event, "position", None)
    return float(position().x()) if position is not None else float(event.x())
