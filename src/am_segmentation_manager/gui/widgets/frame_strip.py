"""The frame timeline: position and curation progress in one widget.

The position marker and the state rows are drawn on the same pixels, so their
alignment is structural rather than something to maintain.

Each modality gets its own row, because a curation step often targets one --
deselecting a SEM while leaving its FIB selected is routine -- and a combined row
could only report that the two disagree.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter
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

#: Drawn over a frame whose images disagree with the label they share. Warning
#: amber rather than another workflow colour: this is not a fourth state, it is
#: a sign that the state shown cannot be trusted.
#:
#: Painted as a translucent wash over the frame's whole column rather than as
#: another stripe. The stripes are all "here is a fact about this row"; a
#: mismatch is "do not trust this frame", which needs to read differently and
#: has nowhere of its own to sit now the label lines occupy both edges.
MISMATCH = QColor("#d08c00")
MISMATCH_ALPHA = 90

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
#: The cursor's white rather than a new hue. The widget already carries four
#: colour vocabularies -- workflow states, label lines, the mismatch wash and
#: the class legend directly above it -- and a fifth would have to be explained.
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
        self._mismatched: set[int] = set()
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
            "Curation progress across the lamella: SEM on the top row, FIB "
            "below.\nThe thin line outside each row shows its labels: bright "
            "means annotated, dim means an empty placeholder, and nothing "
            "means no label file.\nAn amber wash marks a frame whose images "
            "disagree with their shared label.\n"
            "Click or drag to jump to a frame; Left/Right arrow keys step one.\n"
            "Ctrl-click picks a frame out for a state change, Shift-click "
            "extends the run, and Esc clears it. The white band on top marks "
            "what is picked."
        )

    def set_states(
        self,
        rows: dict[str, list],
        current: int,
        mismatched: set[int] | None = None,
        labels: dict[str, list] | None = None,
        selected: set[int] | None = None,
    ) -> None:
        """Show per-modality states and the current position.

        Args:
            rows: Workflow state per frame, keyed by modality. A None entry
                means that modality has no image in that frame.
            current: Index of the frame being viewed.
            mismatched: Frames whose images disagree with the label they share.
                Marked because the state drawn for them is the base image's,
                which is not the whole truth until the disagreement is
                resolved with ``am-seg repair-state``.
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
        self._mismatched = set(mismatched or ())
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

        # Drawn last, over everything, so a frame that cannot be trusted is
        # visible whichever row the disagreement is in.
        wash = QColor(MISMATCH)
        wash.setAlpha(MISMATCH_ALPHA)
        for index in sorted(self._mismatched):
            if index >= self._count:
                continue
            painter.fillRect(
                int(index * width), 0, max(1, int(width) + 1), self.height(), wash
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
