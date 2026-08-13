"""A strip showing the workflow state of every frame in a lamella.

Curation is a sweep through a lamella, and the question that matters between
sessions is "how far did I get?". Scrubbing the slider to find out is tedious,
so the state of every frame is drawn at once, under the slider that moves
through them.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import QSizePolicy, QWidget

from ..controller import MIXED

#: Colours for each workflow state. Deliberately distinct in hue rather than
#: only in brightness, so the strip is readable at a glance.
STATE_COLOURS = {
    "selected": QColor("#4c8f4c"),
    "deselected": QColor("#6b6b6b"),
    "corrected": QColor("#3b6ea5"),
    MIXED: QColor("#c08a34"),
    None: QColor("#2b2b2b"),
}

CURRENT_MARKER = QColor("#f0f0f0")


class FrameStateStrip(QWidget):
    """Draws one tick per frame, coloured by workflow state.

    Clicking a tick jumps to that frame.
    """

    #: Emitted with the frame index when the user clicks the strip.
    frame_clicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._states: list = []
        self._current = 0
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Workflow state of each frame; click to jump")

    def set_states(self, states: list, current: int) -> None:
        """Show *states*, highlighting the frame at *current*."""
        self._states = list(states)
        self._current = current
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), STATE_COLOURS[None])
        if not self._states:
            return

        width = self.width() / len(self._states)
        for index, state in enumerate(self._states):
            colour = STATE_COLOURS.get(state, STATE_COLOURS[None])
            left = index * width
            painter.fillRect(
                int(left), 0, max(1, int(width) + 1), self.height(), colour
            )

        # Mark where the user is, so the strip doubles as a position indicator.
        marker_left = int(self._current * width)
        painter.fillRect(marker_left, 0, max(2, int(width)), 3, CURRENT_MARKER)
        painter.fillRect(
            marker_left, self.height() - 3, max(2, int(width)), 3, CURRENT_MARKER
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._states or event.button() != Qt.LeftButton:
            return
        index = int(event.pos().x() / (self.width() / len(self._states)))
        self.frame_clicked.emit(max(0, min(index, len(self._states) - 1)))
