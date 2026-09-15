"""The frame timeline: position and curation progress in one widget.

The position marker and the state rows are drawn on the same pixels, so their
alignment is structural rather than something to maintain. Each modality gets
its own row, because a curation step often targets one, and a combined row
could only report that the two disagree. Hovering a frame shows a card naming
what it holds above what every mark means, so the legend is where the marks
are read and not only behind F1.
"""
from __future__ import annotations

from qtpy.QtCore import QPoint, Qt, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import TIMELINE_CARD, TIMELINE_LEGEND

#: Workflow state colours.
#:
#: A progress ramp -- inert, active, done -- rather than a semantic red/amber/
#: green. Two reasons: the class palette already uses saturated orange, green,
#: red and yellow, and its legend sits directly above this widget, so a
#: red/green workflow scale would collide with the label colours it is drawn
#: next to; and red against green is the pairing red-green colour deficiency
#: cannot separate.
#:
#: To switch to a semantic scale, change these three values and the matching
#: swatches in the state panel follow automatically.
STATE_COLOURS = {
    "deselected": QColor("#6b6b6b"),
    "selected": QColor("#4c78c8"),
    "corrected": QColor("#1a7f6e"),
    None: QColor("#2e2a25"),
}

BACKGROUND = QColor("#26221e")
TICK = QColor("#1c1915")
CURSOR = QColor("#f5f5f5")

#: Warning amber: something wants attention rather than reporting a state.
#: Deliberately not a workflow colour, so it cannot be read as a fourth one.
#:
#: **Nothing on this timeline is painted with it.** It is here because it sits
#: with the colours that carry meaning rather than with the chrome, and the Add
#: data tab's completeness bar uses it for a project still short of fields.
MISMATCH = QColor("#d08c00")

#: Whether each modality's label exists, and whether it holds an annotation.
#:
#: Violet for an annotation, chosen because the workflow states use grey/blue/
#: teal and the class legend sitting directly above uses orange, green, red and
#: yellow: a separate hue says "this line is about labels, not workflow state".
#:
#: An empty placeholder is **grey rather than a dim violet**. Against a near
#: black background a dark colour is nearly indistinguishable from nothing
#: drawn at all, which is exactly the distinction that has to be legible here --
#: "annotated", "placeholder" and "no label" are three different amounts of
#: work remaining. Grey is lighter than the background and unmistakably not the
#: annotation colour. It is lighter than the deselected state grey too, and
#: sits on the thin line rather than in a state row, so the two do not read as
#: the same mark.
LABEL_COLOURS = {
    "present": QColor("#a78bfa"),
    "empty": QColor("#8a8a8a"),
    "missing": None,
    None: None,
}

#: How a frame holding specks is marked while the speck filter is on.
#:
#: **Drawn over the state row, reserving no space of its own**, and only where
#: there is something: one stray pixel and thirty are different amounts of
#: work, so the count is what is drawn, and a frame with none is left blank.
#: That keeps the marks sparse enough to read as "here are the problems"
#: rather than as another row to be scanned.
#:
#: **White, which is this widget's third mark made for the reviewer rather than
#: by the data.** The cursor says where you are, the band says what you picked,
#: and this says what the filter you switched on found -- none of the three is
#: a standing property of a frame, and all three go away when the reviewer
#: stops asking. So the rule holds rather than bends: state and label keep
#: their own vocabularies and white stays what the reviewer put there.
#:
#: It is also the only ink that reads. Amber is the colour reserved for wanting
#: attention, but against the selected blue it is 1.5:1 -- enough for a solid
#: shape, not for a numeral. Near-black manages 3.5:1 and is legible, but its
#: block reads as one more tick where the frames are narrow.
SPECK_INK = CURSOR

#: How tall that count is drawn, **in pixels rather than in points**. The row
#: it sits in is a pixel measurement, and a point size is not: macOS reports 72
#: logical DPI where the offscreen platform reports 96, so one point size is a
#: third smaller on screen than in a rendered test -- which is how this was
#: first set too small to read.
#:
#: As large as the row will take. The digits are drawn by their baseline rather
#: than centred as a line box, so what has to fit is the ink and not the
#: descender space a numeral never uses: at this size the ink is exactly the
#: row's 10px, and anything larger is clipped.
SPECK_PIXEL_SIZE = 12

#: Clear space kept between a count and the frame next to it, so two of them
#: cannot run together and read as a third number that is not there. Whether a
#: count fits is asked of that count -- "3" fits where "27" does not, and a
#: lamella dense enough to refuse both is past the point of picking out single
#: frames anyway.
SPECK_TEXT_MARGIN = 2

#: The block that stands in at those widths, drawn in the same ink. It says
#: "something here" and does not pretend to say how much.
SPECK_BLOCK_HEIGHT = 3

#: Label lines are thin, so they read as a rule under or over a row rather than
#: as a fourth and fifth state bar.
LABEL_LINE_HEIGHT = 3
LABEL_LINE_GAP = 1

#: Row order, top to bottom. Matches the order of the view mode buttons.
ROW_MODALITIES = ("SEM", "FIB")

ROW_HEIGHT = 11
ROW_GAP = 2
CURSOR_WIDTH = 2

#: Frames picked out for a state change, drawn as a band above everything else.
#:
#: The cursor's white rather than a new hue. The widget already carries three
#: colour vocabularies -- workflow states, label lines, and the class legend
#: directly above it -- and a fourth would have to be explained.
#: White is already the user's own mark here: the cursor says where you are and
#: this says what you picked, and neither is a fact about the data. A horizontal
#: band cannot be confused with the vertical cursor line.
SELECTION = CURSOR
SELECTION_HEIGHT = 3
SELECTION_GAP = 2

#: How much the state rows are pushed down to make room for the band.
SELECTION_BAND = SELECTION_HEIGHT + SELECTION_GAP

#: Ticks are only drawn when frames are at least this wide, below which they
#: would occupy more of the bar than the states they separate.
MIN_TICK_SPACING = 5

#: What the white band means on each tab's timeline -- one mark, two meanings,
#: named once for the hover card and the F1 list alike.
CURATE_BAND = "Picked out for a state change"
LABEL_BAND = "This label has unsaved edits"

#: How a frame's label line is put into words on the hover card.
LABEL_WORDS = {
    "present": "label holds an annotation",
    "empty": "label is empty",
    "missing": "no label file",
}

#: Space kept between the hover card and the strip it describes.
CARD_GAP = 6


def _at(values: list, index: int):
    """Return *values* at *index*, or None past its end."""
    return values[index] if index < len(values) else None


class FrameTimeline(QWidget):
    """Position within a lamella, each modality's state, and what is selected.

    Click or drag to move to a frame. Ctrl (Cmd) and Shift pick frames out for
    a state change instead of moving to them -- deliberately separate, so that
    building a selection never sets a lamella's worth of images loading. Hover
    to see what a frame holds.
    """

    #: Emitted with a frame index when the user clicks or drags.
    frame_selected = Signal(int)
    #: Emitted with a frame to add to, or take out of, the selection.
    frame_toggled = Signal(int)
    #: Emitted when a Shift-extension begins, with the frame it began on.
    extend_started = Signal(int)
    #: Emitted with the frame an extension now reaches, repeatedly while
    #: dragging. Each one describes the whole extension rather than adding to
    #: it, so dragging back shrinks the range again.
    extend_reached = Signal(int)

    def __init__(self, band: str = CURATE_BAND) -> None:
        """Build an empty timeline.

        Args:
            band: What the white band means on the tab this timeline is on.
        """
        super().__init__()
        self._band = band
        self._card: TimelineCard | None = None
        #: The frame the card is describing, while it shows.
        self._hovered: int | None = None
        self._rows: dict[str, list] = {modality: [] for modality in ROW_MODALITIES}
        self._labels: dict[str, list] = {modality: [] for modality in ROW_MODALITIES}
        self._specks: dict[str, list] = {modality: [] for modality in ROW_MODALITIES}
        self._selected: set[int] = set()
        self._count = 0
        self._current = 0
        # Each modality's label line sits on the outside of its state row, so
        # the two label lines frame the widget and neither can be mistaken for
        # the other's row. The selection band sits above all of it.
        self.setFixedHeight(
            SELECTION_BAND
            + len(ROW_MODALITIES) * (ROW_HEIGHT + LABEL_LINE_HEIGHT + LABEL_LINE_GAP)
            + ROW_GAP
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # The hover card instead of a tooltip, which could say what the strip
        # is but not what the frame under the pointer holds.
        self.setMouseTracking(True)

    def set_states(
        self,
        rows: dict[str, list],
        current: int,
        labels: dict[str, list] | None = None,
        selected: set[int] | None = None,
        specks: dict[str, list] | None = None,
    ) -> None:
        """Show per-modality states and the current position.

        Args:
            rows: Workflow state per frame, keyed by modality. A None entry
                means that modality has no image in that frame.
            current: Index of the frame being viewed.
            labels: Label state per frame, keyed by modality: ``"present"``,
                ``"empty"``, ``"missing"``, or None where that modality has no
                image. Drawn as a thin line outside each state row, because
                annotation progress and workflow state are different questions
                and reading one as the other would be misleading.
            selected: Frames picked out for a state change.
            specks: How many specks each frame's label holds, keyed by
                modality, or None to draw none -- which is what the filter
                being off means. An entry of None is a frame nothing has
                counted yet, drawn blank: a background pass fills them in, and
                "not counted" must not read as "none".
        """
        self._rows = {modality: list(rows.get(modality, [])) for modality in ROW_MODALITIES}
        self._labels = {
            modality: list((labels or {}).get(modality, [])) for modality in ROW_MODALITIES
        }
        self._specks = {
            modality: list((specks or {}).get(modality, [])) for modality in ROW_MODALITIES
        }
        self._selected = set(selected or ())
        self._count = max((len(states) for states in self._rows.values()), default=0)
        self._current = current
        if self._hovered is not None:
            if self._count:
                self._show_card(min(self._hovered, self._count - 1), refresh=True)
            else:
                self._hide_card()
        self.update()

    def _frame_width(self) -> float:
        return self.width() / self._count if self._count else 0.0

    @staticmethod
    def _band_height() -> int:
        """Return the height of one modality's label line plus its state row."""
        return LABEL_LINE_HEIGHT + LABEL_LINE_GAP + ROW_HEIGHT

    @classmethod
    def _row_top(cls, row: int) -> int:
        """Return the y of a modality's state row.

        The SEM band is label-line-then-row; the FIB band is row-then-label-
        line, so the two label lines sit on the outside edges and the state
        rows stay adjacent in the middle where they are easiest to compare.
        """
        band = SELECTION_BAND + row * (cls._band_height() + ROW_GAP)
        if row == 0:
            return band + LABEL_LINE_HEIGHT + LABEL_LINE_GAP
        return band

    @classmethod
    def _label_line_top(cls, row: int) -> int:
        """Return the y of a modality's label line."""
        band = SELECTION_BAND + row * (cls._band_height() + ROW_GAP)
        if row == 0:
            return band
        return band + ROW_HEIGHT + LABEL_LINE_GAP

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            # Always, even if painting raised. A QPainter that outlives its
            # paint device crashes the process rather than reporting an error,
            # and under PySide6 the painter is not destroyed promptly when the
            # frame goes out of scope -- an exception here captures it in the
            # traceback, and the segfault then lands somewhere unrelated, in
            # whatever next triggers garbage collection.
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        """Draw the timeline. Called with an active painter, which the caller
        is responsible for ending."""
        painter.fillRect(self.rect(), BACKGROUND)
        if not self._count:
            return

        width = self._frame_width()
        draw_ticks = width >= MIN_TICK_SPACING

        for row, modality in enumerate(ROW_MODALITIES):
            states = self._rows[modality]
            labels = self._labels[modality]
            top = self._row_top(row)
            label_top = self._label_line_top(row)
            for index in range(self._count):
                state = states[index] if index < len(states) else None
                colour = STATE_COLOURS.get(state, STATE_COLOURS[None])
                left = index * width
                cell_width = max(1, int(width) + 1)
                painter.fillRect(int(left), top, cell_width, ROW_HEIGHT - 1, colour)
                if draw_ticks and index:
                    painter.fillRect(int(left), top, 1, ROW_HEIGHT - 1, TICK)

                label_state = labels[index] if index < len(labels) else None
                label_colour = LABEL_COLOURS.get(label_state)
                if label_colour is not None:
                    painter.fillRect(
                        int(left), label_top, cell_width, LABEL_LINE_HEIGHT, label_colour
                    )

            self._paint_specks(painter, self._specks[modality], top, width)

        for index in sorted(self._selected):
            if index >= self._count:
                continue
            painter.fillRect(
                int(index * width), 0, max(1, int(width) + 1),
                SELECTION_HEIGHT, SELECTION,
            )

        # The position marker spans both rows, so it reads as one cursor rather
        # than two independent marks.
        cursor_left = int(self._current * width + width / 2) - CURSOR_WIDTH // 2
        painter.fillRect(cursor_left, 0, CURSOR_WIDTH, self.height(), CURSOR)

    def _paint_specks(
        self, painter: QPainter, counts: list, top: int, width: float
    ) -> None:
        """Draw one modality's speck counts over its state row.

        Nothing is drawn for a frame with no specks, or for one nothing has
        counted yet -- a blank means "no mark to make here", and telling those
        two apart belongs to whatever is filling the counts in.
        """
        if not counts:
            return
        font = painter.font()
        font.setPixelSize(SPECK_PIXEL_SIZE)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(SPECK_INK)
        metrics = QFontMetrics(font)
        room = width - SPECK_TEXT_MARGIN
        # Centred on the ink rather than on the line box: the box carries room
        # for a descender that no numeral uses, so centring it wastes the
        # bottom of the row and clips the top.
        baseline = top + (ROW_HEIGHT - 1 + metrics.tightBoundingRect("0").height()) // 2
        for index in range(min(self._count, len(counts))):
            count = counts[index]
            if not count:
                continue
            left = index * width
            text = str(count)
            advance = metrics.horizontalAdvance(text)
            if advance <= room:
                painter.drawText(
                    int(left + (width - advance) / 2 + self._cursor_shift(index, width, advance)),
                    baseline, text,
                )
            else:
                # Started at the cell's own left edge, never inset: an inset
                # is wider than the cell once frames are a pixel or two apart,
                # and the mark then lands on the frame next door.
                painter.fillRect(
                    int(left),
                    top + (ROW_HEIGHT - 1 - SPECK_BLOCK_HEIGHT) // 2,
                    max(1, int(width) - 1), SPECK_BLOCK_HEIGHT, SPECK_INK,
                )

    def _cursor_shift(self, index: int, width: float, advance: float) -> float:
        """Return how far to move a count clear of the position cursor.

        Zero for every frame but the one being looked at, whose cursor is drawn
        down the middle of the cell and in the same white -- so a count centred
        there has a line through it. Shifted to the right of the cursor where
        the cell has room, and by as much of that as there is where it has not:
        a partly cleared number reads better than one struck through, and a
        cell too narrow to clear it is one where the count is a single digit.
        """
        if index != self._current:
            return 0.0
        clear = advance / 2 + CURSOR_WIDTH / 2 + SPECK_TEXT_MARGIN
        room = (width - advance) / 2
        return min(clear, max(0.0, room))

    @property
    def card(self) -> TimelineCard | None:
        """Return the hover card, or None before anything has been hovered."""
        return self._card

    def describe_frame(self, index: int) -> list[str]:
        """Return what the timeline shows of frame *index*, a line per beam.

        The hover card's text, read from what this widget draws, so the card
        cannot describe a frame differently from how the strip shows it.
        """
        heading = f"Frame {index + 1} of {self._count}"
        lines = [f"{heading} (where you are)" if index == self._current else heading]
        for modality in ROW_MODALITIES:
            state = _at(self._rows[modality], index)
            if state is None:
                lines.append(f"{modality}: no image")
                continue
            parts = [state]
            label = _at(self._labels[modality], index)
            if label in LABEL_WORDS:
                parts.append(LABEL_WORDS[label])
            specks = _at(self._specks[modality], index)
            if specks:
                parts.append(f"{specks} speck{'s' if specks != 1 else ''}")
            lines.append(f"{modality}: {', '.join(parts)}")
        if index in self._selected:
            lines.append(self._band)
        return lines

    def _show_card(self, index: int, *, refresh: bool = False) -> None:
        """Describe frame *index* on the card, over the column it is drawn in."""
        if self._card is None:
            self._card = TimelineCard(self, self._band)
        if index == self._hovered and self._card.isVisible() and not refresh:
            return
        self._hovered = index
        self._card.describe(self.describe_frame(index))
        self._card.adjustSize()
        self._card.move(self._card_position(index))
        self._card.show()

    def _hide_card(self) -> None:
        self._hovered = None
        if self._card is not None:
            self._card.hide()

    def _card_position(self, index: int) -> QPoint:
        """Return where the card goes: centred on the frame, above the strip.

        Below the strip where the screen has no room above it, and never past
        either side of the screen.
        """
        card = self._card
        assert card is not None, "the card is made before it is placed"
        width = self._frame_width()
        above = self.mapToGlobal(QPoint(int(index * width + width / 2), 0))
        x = above.x() - card.width() // 2
        y = above.y() - card.height() - CARD_GAP
        screen = self.screen()
        if screen is not None:
            room = screen.availableGeometry()
            x = max(room.left(), min(x, room.right() - card.width()))
            if y < room.top():
                y = self.mapToGlobal(QPoint(0, self.height())).y() + CARD_GAP
        return QPoint(x, y)

    def _frame_at(self, x: int) -> int:
        width = self._frame_width()
        if not width:
            return 0
        return max(0, min(int(x / width), self._count - 1))

    @staticmethod
    def _picking(modifiers) -> str:
        """Return which selection gesture *modifiers* asks for.

        ``ControlModifier`` is Cmd on macOS, which is what a Mac user expects
        for "add this one to what I already picked".
        """
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "extend"
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return "toggle"
        return "seek"

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hide_card()
        if not self._count or event.button() != Qt.MouseButton.LeftButton:
            return
        frame = self._frame_at(event.pos().x())
        gesture = self._picking(event.modifiers())
        if gesture == "toggle":
            self.frame_toggled.emit(frame)
        elif gesture == "extend":
            self.extend_started.emit(frame)
            self.extend_reached.emit(frame)
        else:
            self.frame_selected.emit(frame)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Scrub, or widen the selection, while the button is held; else describe.

        A drag with either modifier extends: dragging is how a run is picked
        out, and toggling frame by frame under the pointer would flicker items
        in and out as the pointer crossed them.
        """
        if not self._count:
            return
        frame = self._frame_at(event.pos().x())
        if not event.buttons() & Qt.MouseButton.LeftButton:
            self._show_card(frame)
            return
        self._hide_card()
        if self._picking(event.modifiers()) == "seek":
            self.frame_selected.emit(frame)
        else:
            self.extend_reached.emit(frame)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hide_card()
        super().leaveEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hide_card()
        super().hideEvent(event)


#: How wide a legend sample is drawn. Wide enough to read as a band rather than
#: a speck, and every one of them is drawn on :data:`BACKGROUND`: these marks
#: are colours against a near-black ground, and the same colour on the dialog's
#: white is a different mark.
LEGEND_WIDTH = 30


def _legend_tile(height: int | None = None) -> QPixmap:
    """Return a blank sample at the timeline's own ground colour."""
    pixmap = QPixmap(LEGEND_WIDTH, height if height is not None else ROW_HEIGHT)
    pixmap.fill(BACKGROUND)
    return pixmap


def _state_sample(state: str) -> QPixmap:
    """Return one frame's worth of a workflow state, as the timeline draws it."""
    pixmap = _legend_tile()
    painter = QPainter(pixmap)
    try:
        painter.fillRect(
            0, 0, LEGEND_WIDTH, ROW_HEIGHT - 1, STATE_COLOURS[state]
        )
    finally:
        # A painter outliving its pixmap crashes the process rather than
        # reporting it.
        painter.end()
    return pixmap


def _label_line_sample() -> QPixmap:
    """Return the three label states side by side, in the order they are named.

    One sample rather than three rows: the line is the same mark throughout and
    what varies is only how much of a label there is, which reads as a sequence.
    """
    pixmap = _legend_tile(_band_sample_height())
    painter = QPainter(pixmap)
    try:
        third = LEGEND_WIDTH // 3
        for index, state in enumerate(("present", "empty", "missing")):
            colour = LABEL_COLOURS[state]
            width = LEGEND_WIDTH - 2 * third if index == 2 else third
            if colour is not None:
                painter.fillRect(index * third, 0, width, LABEL_LINE_HEIGHT, colour)
            painter.fillRect(
                index * third,
                LABEL_LINE_HEIGHT + LABEL_LINE_GAP,
                width,
                ROW_HEIGHT - 1,
                STATE_COLOURS["selected"],
            )
    finally:
        painter.end()
    return pixmap


def _band_sample_height() -> int:
    """Return the height of a label line and the state row under it."""
    return LABEL_LINE_HEIGHT + LABEL_LINE_GAP + ROW_HEIGHT


def _whole_sample() -> QPixmap:
    """Return a miniature timeline, so the layout can be seen before the parts.

    Drawn by the widget itself rather than reproduced here: a legend showing
    something the timeline does not draw is worse than no legend, and this is
    the only way to be sure it cannot happen.
    """
    strip = FrameTimeline()
    strip.resize(3 * LEGEND_WIDTH, strip.height())
    states = ["deselected", "selected", "corrected", "corrected"]
    labels = ["missing", "empty", "present", "present"]
    strip.set_states(
        {"SEM": states, "FIB": ["deselected", *states[:3]]},
        1,
        {"SEM": labels, "FIB": ["missing", *labels[:3]]},
    )
    return strip.grab()


def timeline_legend(band: str) -> list[tuple[QPixmap, str]]:
    """Return what each mark on the timeline means, as ``(sample, meaning)``.

    Args:
        band: What the white band across the top says on the tab asking. It
            marks frames picked out for a state change while curating and
            labels with unsaved edits while proofreading -- one mark, two
            meanings, which is why the caller supplies it rather than this.
    """
    return [
        (
            _whole_sample(),
            "One column per frame, in order. SEM above, FIB below, each with a "
            "thin line for its label.",
        ),
        *_mark_legend(band),
    ]


def _mark_legend(band: str) -> list[tuple[QPixmap, str]]:
    """Return what each mark means, without the miniature of the whole strip.

    The hover card's legend, where the strip itself is right under it.
    """
    return [
        (_state_sample("deselected"), "Deselected: no label is wanted here"),
        (_state_sample("selected"), "Selected: a label is wanted and not yet finished"),
        (_state_sample("corrected"), "Corrected: the label has been proofread"),
        (
            _label_line_sample(),
            "The label line, left to right: it holds an annotation, the file is "
            "there but empty, there is no label file",
        ),
        (_selection_sample(), band),
        (_cursor_sample(), "Where you are"),
    ]


def _selection_sample() -> QPixmap:
    """Return the white band, drawn where it sits above the rows."""
    pixmap = _legend_tile(SELECTION_BAND + ROW_HEIGHT)
    painter = QPainter(pixmap)
    try:
        painter.fillRect(0, 0, LEGEND_WIDTH, SELECTION_HEIGHT, SELECTION)
        painter.fillRect(
            0, SELECTION_BAND, LEGEND_WIDTH, ROW_HEIGHT - 1,
            STATE_COLOURS["selected"],
        )
    finally:
        painter.end()
    return pixmap


def _cursor_sample() -> QPixmap:
    """Return the position marker, crossing a row and the ground around it.

    Taller than the row it crosses on purpose: drawn against a tile the row
    fills, the marker reads as a gap between two bands rather than as a line
    over them, which is the opposite of what it is.
    """
    height = SELECTION_BAND + ROW_HEIGHT
    pixmap = _legend_tile(height)
    painter = QPainter(pixmap)
    try:
        painter.fillRect(
            0, SELECTION_BAND, LEGEND_WIDTH, ROW_HEIGHT - 1, STATE_COLOURS["selected"]
        )
        painter.fillRect(
            LEGEND_WIDTH // 2 - CURSOR_WIDTH // 2, 0, CURSOR_WIDTH, height, CURSOR
        )
    finally:
        painter.end()
    return pixmap


class TimelineCard(QFrame):
    """What the frame under the pointer holds, above what every mark means.

    A window flagged as a tooltip, so it takes no focus and no room in the
    layout: a QToolTip holds text, and could not show the marks themselves.
    """

    def __init__(self, parent: QWidget, band: str) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setObjectName(TIMELINE_CARD)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 6, 8, 8)
        column.setSpacing(8)

        #: The frame's own lines, replaced as the pointer moves.
        self.frame = QLabel()
        column.addWidget(self.frame)

        legend = QGridLayout()
        legend.setHorizontalSpacing(8)
        legend.setVerticalSpacing(3)
        rows = _mark_legend(band)
        for row, (sample, meaning) in enumerate(rows):
            picture = QLabel()
            picture.setPixmap(sample)
            legend.addWidget(picture, row, 0)
            caption = QLabel(meaning)
            caption.setObjectName(TIMELINE_LEGEND)
            legend.addWidget(caption, row, 1)
        column.addLayout(legend)
        #: What each legend row says, in order.
        self.meanings = [meaning for _, meaning in rows]

    def describe(self, lines: list[str]) -> None:
        """Show *lines* as what the frame under the pointer holds."""
        self.frame.setText("\n".join(lines))
