"""The controls this package draws itself, rather than letting a style draw.

Each is here because Qt's own version answers a different question or is drawn
differently on each platform; `docs/display.md` under Controls says which and
why for every one. **Anything painted here may not be named in a stylesheet
rule**: a rule sends the widget back through `QStyle` and it is then drawn
twice.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from qtpy.QtCore import (
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
    QFontMetrics,
    QPainter,
    QPen,
)
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSplitter,
    QSplitterHandle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .common import match_anywhere

logger = logging.getLogger(__name__)


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


def keep_together(*widgets: QWidget, spacing: int = 4) -> QWidget:
    """Return one widget holding *widgets* in a row, so a flow cannot split them.

    A flow layout wraps between its items, and a caption is a separate item
    from the control it names: at the wrong width the two land on different
    lines, leaving the caption reading as a heading for whatever followed it.
    One widget is the same answer :class:`SegmentedControl` gives for the same
    reason, and `docs/display.md` states it there.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        row.addWidget(widget)
    return holder


def keep_paired(top, bottom, *, spacing: int = 4, align=None) -> QWidget:
    """Return one widget holding two rows of controls, in a grid that aligns them.

    Each argument is that row's widgets, left to right; the grid lines their
    columns up, so a caption above a caption and a slider above a slider read
    as a pair rather than as four loose things. One widget for the same reason
    :func:`keep_together` is one, and a pair costs half the width of the two
    controls side by side -- which is what keeps a row of them inside a pane
    the window can actually offer.

    Args:
        align: Alignment for particular cells, keyed by ``(row, column)``. A
            cell fills its column by default, which is right for the sliders
            and wrong for a lone button under a wider control.
    """
    align = align or {}
    holder = QWidget()
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(spacing)
    grid.setVerticalSpacing(2)
    for row, widgets in enumerate((top, bottom)):
        for column, widget in enumerate(widgets):
            grid.addWidget(widget, row, column, align.get((row, column), Qt.Alignment()))
    return holder


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
        #: Where each item was put, and how tall each row turned out, so the
        #: items can be centred once the row's height is known -- which is only
        #: after the last item on it has been placed.
        placed: list = []
        rows: list = [(0, 0)]

        for item in self._items:
            # Clamped to the row: an item wider than the space it has spills
            # past the edge otherwise, where nothing clips it and it is drawn
            # over whatever sits beside the layout.
            size = item.sizeHint().boundedTo(QSize(right - left + 1, 1 << 20))
            if row_height and x + size.width() > right:
                x = left
                y += row_height + self.spacing()
                row_height = 0
                rows.append((len(placed), 0))
            placed.append((item, QPoint(x, y), size))
            x += size.width() + self.spacing()
            row_height = max(row_height, size.height())
            rows[-1] = (len(placed), row_height)

        if apply:
            self._centre_rows(placed, rows)
        return y + row_height - rect.y() + margins.bottom()

    @staticmethod
    def _centre_rows(placed, rows) -> None:
        """Put each item in the middle of its row's height.

        A row mixing a two-line block of paired controls with a single button
        would otherwise hang the button from the top of it, reading as though
        the two belonged to different rows.
        """
        start = 0
        for end, height in rows:
            for item, point, size in placed[start:end]:
                offset = (height - size.height()) // 2
                item.setGeometry(QRect(QPoint(point.x(), point.y() + offset), size))
            start = end


class WrappingLabel(QLabel):
    """A word-wrapped label whose height follows the width it actually has.

    Qt asks a label for its size hint before it knows how wide it will be, and
    a word-wrapped one answers for a single line. A dialog sized from that hint
    comes up too short, and every widget below the label is squeezed towards
    its minimum -- which is a group box drawing its rows on top of each other,
    some way from the label that caused it.

    The same fault :class:`FlowArea` fixes for a row that wraps, and the same
    answer: take the height from the width, every time there is one.
    """

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setWordWrap(True)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self.setFixedHeight(self.heightForWidth(self.width()))


class FlowArea(QWidget):
    """A widget whose height is whatever the flow layout inside it needs.

    Height-for-width alone is not enough inside a form: the form asks how tall
    the widget would be at one width and then hands it another, and the rows
    that do not fit are laid out below the bottom edge -- present,
    addressable, and invisible. Answering from the width it actually has,
    every time it gets one, is the only account that cannot disagree with
    itself.

    Used for a field holding several values and for the rows of controls above
    the curation panes and the proofreading canvas, which are the same problem:
    however many things there are, at whatever width the pane happens to have.
    On those two rows it is also what stops the window being forced wider than
    the screen; see `docs/display.md`.
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


#: The range slider's handle radius and groove, in logical pixels, and the room
#: kept clear for a focus ring the way :data:`theme.SWITCH_FOCUS` is -- a ring
#: painted past the widget's edge comes out clipped flat.
RANGE_HANDLE = 7


RANGE_GROOVE = 4


RANGE_FOCUS = 3.0


#: How wide a view slider is. The range slider asks for it and a plain slider
#: paired above one is fixed to it, so the two grid cells end together.
SLIDER_WIDTH = 150


def _track_span(width: float) -> tuple[float, float]:
    """Return the x range a handle's centre moves between, in a widget *width* wide."""
    return RANGE_HANDLE + RANGE_FOCUS, width - RANGE_HANDLE - RANGE_FOCUS


def _x_along(value: int, minimum: int, maximum: int, width: float) -> float:
    """Return where *value* sits along the track of a widget *width* wide."""
    left, right = _track_span(width)
    if maximum == minimum:  # pragma: no cover - a degenerate range
        return left
    return left + (value - minimum) / (maximum - minimum) * (right - left)


def _value_along(x: float, minimum: int, maximum: int, width: float) -> int:
    """Return the value at *x* along the track of a widget *width* wide."""
    left, right = _track_span(width)
    if right <= left:  # pragma: no cover - not yet laid out
        return minimum
    fraction = max(0.0, min(1.0, (x - left) / (right - left)))
    return round(minimum + fraction * (maximum - minimum))


def _slider_side() -> int:
    """Return how tall a slider is: a handle and the focus ring round it."""
    return int(2 * (RANGE_HANDLE + RANGE_FOCUS))


def _paint_track(
    painter: QPainter,
    left: float,
    right: float,
    middle: float,
    filled_from: float,
    filled_to: float,
) -> None:
    """Draw a slider's groove, filled in the accent between two points along it."""
    groove = QRectF(left, middle - RANGE_GROOVE / 2, right - left, RANGE_GROOVE)
    radius = RANGE_GROOVE / 2
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(theme.BORDER))
    painter.drawRoundedRect(groove, radius, radius)
    painter.setBrush(QColor(theme.ACCENT))
    painter.drawRoundedRect(
        QRectF(filled_from, groove.top(), filled_to - filled_from, RANGE_GROOVE),
        radius, radius,
    )


def _paint_knob(painter: QPainter, x: float, middle: float, *, ringed: bool) -> None:
    """Draw one round handle, ringed in the accent when it is the one the keys move."""
    centre = QPointF(x, middle)
    if ringed:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(centre, RANGE_HANDLE + RANGE_FOCUS, RANGE_HANDLE + RANGE_FOCUS)
    pen = QPen(QColor(theme.BORDER))
    pen.setWidthF(1.0)
    painter.setPen(pen)
    painter.setBrush(QColor(theme.BASE))
    painter.drawEllipse(centre, RANGE_HANDLE, RANGE_HANDLE)


class ValueSlider(QSlider):
    """A horizontal slider drawn as :class:`RangeSlider` is, with one handle.

    **A subclass of** :class:`QSlider` **on purpose**, as :class:`ToggleSwitch`
    is of ``QCheckBox``: the range, the keys, the wheel and ``valueChanged``
    are Qt's own, and only the drawing and where a press lands differ. Painted
    entirely here, so no stylesheet rule may name it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self.setCursor(Qt.PointingHandCursor)
        self._dragging = False

    def _x_of(self, value: int) -> float:
        """Return where *value* sits along the track."""
        return _x_along(value, self.minimum(), self.maximum(), self.width())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Put the handle under the pointer and take hold of it."""
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._dragging = True
        self.setSliderDown(True)
        self._drag_to(_event_x(event))
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
        if not self._dragging:
            super().mouseReleaseEvent(event)
            return
        self._dragging = False
        self.setSliderDown(False)
        event.accept()

    def _drag_to(self, x: float) -> None:
        """Move the handle to the value at *x*, as a drag would."""
        self.setSliderPosition(
            _value_along(x, self.minimum(), self.maximum(), self.width())
        )

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        """Draw the groove, filled up to the handle, and the handle."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            left, right = _track_span(self.width())
            middle = self.height() / 2
            x = self._x_of(self.value())
            _paint_track(painter, left, right, middle, left, x)
            _paint_knob(painter, x, middle, ringed=self.hasFocus())
        finally:
            painter.end()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return the size a range slider asks for, so a pair ends together."""
        return QSize(SLIDER_WIDTH, _slider_side())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return enough for the handle to travel and nothing more."""
        side = _slider_side()
        return QSize(2 * side, side)


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
        return _track_span(self.width())

    def _x_of(self, value: int) -> float:
        """Return where *value* sits along the track."""
        return _x_along(value, self._minimum, self._maximum, self.width())

    def _value_at(self, x: float) -> int:
        """Return the value at *x* along the track."""
        return _value_along(x, self._minimum, self._maximum, self.width())

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        """Draw the groove, the span between the handles, and the handles."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            left, right = self._span()
            middle = self.height() / 2
            low_x, high_x = self._x_of(self._low), self._x_of(self._high)
            _paint_track(painter, left, right, middle, low_x, high_x)
            for name, x in (("low", low_x), ("high", high_x)):
                _paint_knob(
                    painter, x, middle,
                    ringed=name == self._active and self.hasFocus(),
                )
        finally:
            painter.end()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return the room the track, the handles and a focus ring need."""
        return QSize(SLIDER_WIDTH, _slider_side())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Return enough for both handles side by side and nothing more."""
        side = _slider_side()
        return QSize(4 * side, side)


def _event_x(event) -> float:
    """Return a mouse event's x, under either binding."""
    position = getattr(event, "position", None)
    return float(position().x()) if position is not None else float(event.x())


#: The object names the stylesheet joins a run of segments with. Set on each
#: button by :class:`SegmentedControl`; the rules are in :mod:`theme`, chrome
#: being decided there and nowhere else.
SEGMENT_NAMES = ("segmentOnly", "segmentFirst", "segmentMiddle", "segmentLast")


def _segment_name(index: int, count: int) -> str:
    """Return which end of the run a segment is, or that it is the whole run."""
    if count == 1:
        return "segmentOnly"
    if index == 0:
        return "segmentFirst"
    return "segmentLast" if index == count - 1 else "segmentMiddle"


class SegmentedControl(QWidget):
    """One of a small closed set, drawn as a joined run rather than as dots.

    For a question with three or four fixed answers that are all worth showing
    at once -- which beam, which view, what a change applies to. A dropdown
    hides the alternatives behind a click and a radio group spends a row of
    loose dots saying what the captions already say.

    One widget rather than loose buttons, which also means the run cannot be
    split across two lines of a flow layout: the three view buttons could wrap
    apart, leaving what is plainly one choice reading as two.

    Exclusive, so exactly one segment is chosen; :meth:`set_current` reflects a
    choice made elsewhere without reporting it back. Each segment is an
    ordinary ``QToolButton`` and they are reachable through :attr:`buttons`, so
    a caller can enable and disable them one at a time.
    """

    #: Emitted with the key of the segment the user chose.
    chosen = Signal(str)

    def __init__(
        self,
        segments: Sequence[tuple[str, str, str]],
        parent: QWidget | None = None,
    ) -> None:
        """Build a run from ``(key, caption, tooltip)`` triples, in order."""
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        # No spacing: the segments share their borders, which is what makes the
        # run read as one control rather than as a row of buttons.
        row.setSpacing(0)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict[str, QToolButton] = {}

        for index, (key, caption, hint) in enumerate(segments):
            button = QToolButton()
            button.setObjectName(_segment_name(index, len(segments)))
            button.setText(caption)
            button.setCheckable(True)
            button.setToolTip(hint)
            button.setCursor(Qt.PointingHandCursor)
            button.toggled.connect(
                lambda checked, k=key: checked and self.chosen.emit(k)
            )
            self._group.addButton(button)
            row.addWidget(button)
            self.buttons[key] = button

    def current(self) -> str | None:
        """Return the key of the chosen segment, or None while none is."""
        for key, button in self.buttons.items():
            if button.isChecked():
                return key
        return None

    def set_current(self, key: str) -> None:
        """Choose *key* without reporting it, for reflecting a change elsewhere.

        Raises:
            KeyError: If *key* is not one of the segments, a caller asking for
                one that is not there having got the vocabulary wrong.
        """
        button = self.buttons[key]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
