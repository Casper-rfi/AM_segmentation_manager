"""The frame timeline: position and curation progress in one widget.

The position marker and the state rows are drawn on the same pixels, so their
alignment is structural rather than something to maintain.

Each modality gets its own row, because a curation step often targets one --
deselecting a SEM while leaving its FIB selected is routine -- and a combined row
could only report that the two disagree.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPixmap
from qtpy.QtWidgets import QSizePolicy, QWidget

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
    None: QColor("#2b2b2b"),
}

BACKGROUND = QColor("#232323")
TICK = QColor("#1a1a1a")
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


class FrameTimeline(QWidget):
    """Position within a lamella, each modality's state, and what is selected.

    Click or drag to move to a frame. Ctrl (Cmd) and Shift pick frames out for
    a state change instead of moving to them -- deliberately separate, so that
    building a selection never sets a lamella's worth of images loading.
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

    def __init__(self) -> None:
        super().__init__()
        self._rows: dict[str, list] = {modality: [] for modality in ROW_MODALITIES}
        self._labels: dict[str, list] = {modality: [] for modality in ROW_MODALITIES}
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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Curation progress: SEM above, FIB below, each with a thin line "
            "for its labels, bright where one holds an annotation.\n"
            "Click to jump; F1 says what every mark means."
        )

    def set_states(
        self,
        rows: dict[str, list],
        current: int,
        labels: dict[str, list] | None = None,
        selected: set[int] | None = None,
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
        """
        self._rows = {modality: list(rows.get(modality, [])) for modality in ROW_MODALITIES}
        self._labels = {
            modality: list((labels or {}).get(modality, [])) for modality in ROW_MODALITIES
        }
        self._selected = set(selected or ())
        self._count = max((len(states) for states in self._rows.values()), default=0)
        self._current = current
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
        if modifiers & Qt.ShiftModifier:
            return "extend"
        if modifiers & Qt.ControlModifier:
            return "toggle"
        return "seek"

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._count or event.button() != Qt.LeftButton:
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
        """Scrub, or widen the selection, while the button is held.

        A drag with either modifier extends: dragging is how a run is picked
        out, and toggling frame by frame under the pointer would flicker items
        in and out as the pointer crossed them.
        """
        if not self._count or not event.buttons() & Qt.LeftButton:
            return
        frame = self._frame_at(event.pos().x())
        if self._picking(event.modifiers()) == "seek":
            self.frame_selected.emit(frame)
        else:
            self.extend_reached.emit(frame)


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
