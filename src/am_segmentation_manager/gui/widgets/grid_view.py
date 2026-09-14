"""The pane showing a project's grid overview with its lamellae marked.

An :class:`.ImageView` with marks placed in the picture's own pixel coordinates
and told to ignore the view's transform, so zooming magnifies the picture and
leaves each mark its size. **One accent and no second colour**: every mark is the
window's accent and the lamella being looked at is drawn heavier with its name;
orange and green are class colours, which ``docs/display.md`` says may not mark a
position.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QBrush, QColor, QFont, QPen
from qtpy.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from ...overview import LamellaMarker
from .image_view import ImageView
from .theme import ACCENT, ACCENT_TEXT

#: Radius of a mark, in screen pixels. Large enough to find on a 2048-pixel
#: overview holding eighteen of them, small enough not to hide the square it
#: points at -- which is the whole reason the mark is a ring and not a dot.
MARK_RADIUS: float = 9.0

#: Radius of the mark on the lamella being looked at.
CURRENT_MARK_RADIUS: float = 15.0

#: Pen widths for an ordinary mark and for the current one.
MARK_WIDTH: float = 2.0
CURRENT_MARK_WIDTH: float = 3.0

#: How far above a mark its name sits, in screen pixels.
LABEL_OFFSET: float = 6.0

#: The margin between the name and the edge of the chip behind it.
LABEL_PADDING: float = 3.0

#: Drawn over the picture. Above :data:`~.image_view.BACKDROP_Z` and the pixmap,
#: which sit at the default zero.
MARK_Z: float = 10.0


class GridView(ImageView):
    """A grid overview with a mark on each lamella of the project."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._marks: list[QGraphicsItem] = []

    def set_markers(
        self,
        markers: tuple[LamellaMarker, ...],
        current: str | None,
        size: tuple[int, int],
    ) -> None:
        """Draw a mark per lamella, naming the one being looked at.

        Args:
            markers: Where each lamella falls, as fractions of the picture.
            current: The lamella on screen, drawn heavier and named. None
                where the project is selected but no lamella within it is.
            size: ``(width, height)`` the picture was drawn at, which the
                fractions are multiplied by. Passed rather than read off the
                item so that a caller preparing a reduced picture needs to
                agree with nothing.

        Markers falling outside the picture are not drawn: a lamella milled
        after the grid moved has a real position and no place on this image,
        and a mark pinned to the nearest edge would claim otherwise.
        """
        self.clear_markers()
        width, height = size
        for marker in markers:
            if not marker.inside:
                continue
            self._add_mark(
                marker.x * width, marker.y * height, marker.name == current, marker.name
            )

    def clear_markers(self) -> None:
        """Take every mark off, leaving the picture."""
        for item in self._marks:
            self.scene().removeItem(item)
        self._marks.clear()

    def _add_mark(self, x: float, y: float, current: bool, name: str) -> None:
        """Put one ring on the picture, and its name where it is the current one."""
        colour = QColor(ACCENT)
        radius = CURRENT_MARK_RADIUS if current else MARK_RADIUS
        ring = QGraphicsEllipseItem(
            QRectF(-radius, -radius, radius * 2, radius * 2)
        )
        ring.setPen(QPen(colour, CURRENT_MARK_WIDTH if current else MARK_WIDTH))
        ring.setBrush(QBrush(Qt.NoBrush))
        ring.setPos(x, y)
        # The ring keeps its size as the picture is zoomed: it points at a
        # square rather than covering one, and a ring that grew would.
        ring.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        ring.setZValue(MARK_Z)
        self.scene().addItem(ring)
        self._marks.append(ring)

        if not current:
            return
        self._add_label(x, y, name, colour, radius)

    def _add_label(
        self, x: float, y: float, name: str, colour: QColor, radius: float
    ) -> None:
        """Put the lamella's name above its ring, on a ground of its own.

        **Filled rather than drawn straight onto the picture.** The overview is
        a greyscale micrograph of arbitrary brightness, so text of any single
        colour is unreadable somewhere on it; the accent filled with
        :data:`~.theme.ACCENT_TEXT` on top is the same pair every filled
        control in the window uses, and its contrast does not depend on what
        the name happens to land on.
        """
        font = QFont()
        font.setBold(True)
        text = QGraphicsSimpleTextItem(name)
        text.setFont(font)
        text.setBrush(QBrush(QColor(ACCENT_TEXT)))

        bounds = text.boundingRect()
        chip = QGraphicsRectItem(
            QRectF(
                -bounds.width() / 2 - LABEL_PADDING,
                -(radius + LABEL_OFFSET + bounds.height() + LABEL_PADDING * 2),
                bounds.width() + LABEL_PADDING * 2,
                bounds.height() + LABEL_PADDING * 2,
            )
        )
        chip.setPen(QPen(Qt.NoPen))
        chip.setBrush(QBrush(colour))
        chip.setPos(x, y)
        chip.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        chip.setZValue(MARK_Z)
        self.scene().addItem(chip)
        self._marks.append(chip)

        # Parented to the chip, so it inherits the position and the flag and
        # cannot drift from its own background.
        text.setParentItem(chip)
        text.setPos(
            -bounds.width() / 2,
            -(radius + LABEL_OFFSET + bounds.height() + LABEL_PADDING),
        )
