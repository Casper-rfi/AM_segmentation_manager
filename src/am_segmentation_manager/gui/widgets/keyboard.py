"""The keyboard drawn for the shortcut list.

Only the F1 list uses this. It is here rather than beside that list because it
is a picture with its own geometry, and because what it may draw is decided by
the bindings rather than by this: `_keys_of` reads them, so the diagram cannot
claim a key the window does not have.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import (
    QRectF,
    Qt,
)
from qtpy.QtGui import (
    QColor,
    QPainter,
    QPixmap,
)

from . import theme

logger = logging.getLogger(__name__)


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
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
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


#: What a key is called on the drawing, where that differs from its binding.
KEY_CAPTIONS: dict[str, str] = {
    "Up": "↑", "Down": "↓", "Left": "←", "Right": "→",
}


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
    painter.setPen(QColor(theme.active_accent().colour if used else theme.BORDER))
    painter.setBrush(QColor(theme.active_accent().colour if used else theme.BASE))
    painter.drawRoundedRect(
        QRectF(x, y, _key_width(name), KEY_SIZE), KEY_RADIUS, KEY_RADIUS
    )
    painter.setPen(QColor(theme.active_accent().text if used else theme.MUTED))
    painter.drawText(
        QRectF(x, y, _key_width(name), KEY_SIZE),
        Qt.AlignmentFlag.AlignCenter,
        KEY_CAPTIONS.get(name, name),
    )
