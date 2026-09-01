"""Small widget behaviours shared between panels.

What lands here is something more than one panel *does* -- a context menu, an
event filter, asking the user for a path. What the window looks like belongs in
:mod:`theme`, anything that knows about projects, labels or the filesystem
belongs in a controller, and anything only one panel uses belongs in that panel.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from qtpy.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
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
    QLabel,
    QLayout,
    QLayoutItem,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig
from . import theme

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


def match_anywhere(completer: QCompleter) -> None:
    """Have *completer* suggest on any part of a value, and offer them in a list.

    **The popup is not cosmetic.** An editable ``QComboBox`` builds its
    completer in ``InlineCompletion`` mode, which assumes the match is a prefix
    and keeps what was typed while appending the rest of the completion. Set to
    match anywhere and left inline, typing ``a`` against "UltrAuFoil Au R2/2 200
    mesh" writes "altrAuFoil Au R2/2 200 mesh" into the field -- a plausible
    value nobody typed, in metadata that is exported. A popup replaces the whole
    text instead, which is what makes matching anywhere safe.

    Matching anywhere is worth this: "Quantifoil" and "UltrAuFoil" are told
    apart by their middles, not by how they start.
    """
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    completer.setFilterMode(Qt.MatchContains)
    completer.setCompletionMode(QCompleter.PopupCompletion)


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
    the gap ``docs/decisions.md`` § Environment is about. One code path rather
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
    ``docs/decisions.md`` § Environment.

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
    widget: QListWidget, noun: str, copy_signal, reveal_signal
) -> None:
    """Give a list a right-click menu for the row's filesystem path.

    The row under the cursor is used, not the selected row: right-clicking an
    entry and acting on a different one would be a surprising way to lose track
    of which path you copied. The row number is emitted rather than a path,
    because resolving one is the controller's job, not a widget's.
    """
    widget.setContextMenuPolicy(Qt.CustomContextMenu)

    def show_menu(point) -> None:
        item = widget.itemAt(point)
        if item is None:
            return
        row = widget.row(item)
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


class _FlowArea(QWidget):
    """A widget whose height is whatever the flow layout inside it needs.

    Height-for-width alone is not enough inside a form: the form asks how tall
    the widget would be at one width and then hands it another, and the rows
    that do not fit are laid out below the bottom edge -- present,
    addressable, and invisible. Answering from the width it actually has,
    every time it gets one, is the only account that cannot disagree with
    itself.
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

        self._chips = _FlowArea()
        self._chip_layout = FlowLayout()
        self._chips.setLayout(self._chip_layout)
        column.addWidget(self._chips)

        self.entry = QComboBox()
        self.entry.setEditable(True)
        self.entry.setInsertPolicy(QComboBox.NoInsert)
        self.entry.lineEdit().setPlaceholderText(placeholder)
        match_anywhere(self.entry.completer())
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
