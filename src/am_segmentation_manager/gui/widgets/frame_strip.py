"""The frame timeline: position and curation progress in one widget.

This replaces a slider sitting above a separate state strip. Those two could
never line up, because the slider row carried labels the strip did not, and
keeping them aligned by padding would have been a permanent maintenance
irritation. Merging them makes alignment structural rather than something to get
right: there is one widget, and the position marker is drawn on the same pixels
as the states.

Each modality gets its own row. A curation step often targets one modality --
deselecting a SEM while leaving its FIB selected is routine -- and a single
combined row can only report "these disagree", which is exactly the state that
proved unreadable in testing.
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
MISMATCH = QColor("#d08c00")

#: Height of the mismatch stripe along the bottom of a frame's cell.
MISMATCH_HEIGHT = 3

#: Row order, top to bottom. Matches the order of the view mode buttons.
ROW_MODALITIES = ("SEM", "FIB")

ROW_HEIGHT = 11
ROW_GAP = 2
CURSOR_WIDTH = 2

#: Ticks are only drawn when frames are at least this wide, below which they
#: would occupy more of the bar than the states they separate.
MIN_TICK_SPACING = 5


class FrameTimeline(QWidget):
    """Position within a lamella, and each modality's state across it.

    Click or drag anywhere on the widget to move to that frame.
    """

    #: Emitted with a frame index when the user clicks or drags.
    frame_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._rows: dict[str, list] = {modality: [] for modality in ROW_MODALITIES}
        self._mismatched: set[int] = set()
        self._count = 0
        self._current = 0
        self.setFixedHeight(len(ROW_MODALITIES) * ROW_HEIGHT + ROW_GAP)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Curation progress across the lamella: SEM on the top row, FIB "
            "below.\nClick or drag to jump to a frame; Left/Right arrow keys "
            "step one frame."
        )

    def set_states(
        self,
        rows: dict[str, list],
        current: int,
        mismatched: set[int] | None = None,
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
        """
        self._rows = {modality: list(rows.get(modality, [])) for modality in ROW_MODALITIES}
        self._mismatched = set(mismatched or ())
        self._count = max((len(states) for states in self._rows.values()), default=0)
        self._current = current
        self.update()

    def _frame_width(self) -> float:
        return self.width() / self._count if self._count else 0.0

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if not self._count:
            return

        width = self._frame_width()
        draw_ticks = width >= MIN_TICK_SPACING

        for row, modality in enumerate(ROW_MODALITIES):
            states = self._rows[modality]
            top = row * (ROW_HEIGHT + ROW_GAP)
            for index in range(self._count):
                state = states[index] if index < len(states) else None
                colour = STATE_COLOURS.get(state, STATE_COLOURS[None])
                left = index * width
                painter.fillRect(
                    int(left), top, max(1, int(width) + 1), ROW_HEIGHT - 1, colour
                )
                if draw_ticks and index:
                    painter.fillRect(int(left), top, 1, ROW_HEIGHT - 1, TICK)

        # Mismatches are drawn after the states, as a stripe spanning the full
        # height, so a frame whose state cannot be trusted is visible without
        # having to compare the two rows against each other.
        for index in sorted(self._mismatched):
            if index >= self._count:
                continue
            left = index * width
            painter.fillRect(
                int(left),
                self.height() - MISMATCH_HEIGHT,
                max(1, int(width) + 1),
                MISMATCH_HEIGHT,
                MISMATCH,
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

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._count and event.button() == Qt.LeftButton:
            self.frame_selected.emit(self._frame_at(event.pos().x()))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Scrub while the button is held."""
        if self._count and event.buttons() & Qt.LeftButton:
            self.frame_selected.emit(self._frame_at(event.pos().x()))
