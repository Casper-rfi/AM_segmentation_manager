"""The window's drawn icons: the Label tab's tools, and the settings gear.

Each is a few QPainter calls on a transparent square, so the window carries no
image files and every icon takes the theme's colour. **Each icon is the shape of
the thing it does** -- the brush is the disc it lays down -- so the tools read
without labels. Only chrome lives here: an icon saying something about the *data*
belongs beside the panel and legend that explain it, drawn by the same code.
"""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF

from .common import chip_icon
from .theme import TEXT

#: Side of the tool icons, in logical pixels.
ICON = 16


def _tool_pixmap(size: int = ICON) -> QPixmap:
    """Return a transparent square *size* pixels across to draw a tool icon on."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    return pixmap


def _drawn_at(painter: QPainter, size: int) -> None:
    """Scale *painter* so a mark laid out on the sixteen-pixel grid fills *size*.

    Drawn at the size wanted rather than scaled up afterwards, so a larger
    mark is as sharp as the others.
    """
    painter.scale(size / ICON, size / ICON)


def brush_icon(size: int = ICON):
    """Return the brush's icon: the disc it paints.

    Drawn rather than shipped, so the window carries no image files. A disc
    because that is literally what the brush lays down -- the icon and the
    tool are the same shape.
    """
    pixmap = _tool_pixmap(size)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _drawn_at(painter, size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT))
        painter.drawEllipse(3, 3, ICON - 6, ICON - 6)
    finally:
        # Always: a QPainter outliving its paint device crashes the process
        # rather than reporting an error.
        painter.end()
    return chip_icon(pixmap)


def fill_icon(size: int = ICON):
    """Return the fill's icon: a drop, for the area one click floods.

    A drop rather than the usual bucket: a bucket needs a body, a handle and a
    pouring lip, and at sixteen pixels those merge into a blob. The pair reads
    as what each tool does -- the brush lays down a disc, the fill pours.
    """
    pixmap = _tool_pixmap(size)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _drawn_at(painter, size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT))
        # A round bottom with a pointed top, drawn as two overlapping shapes of
        # one colour rather than as a single path: an arc joined to a curve
        # leaves a seam wherever the join is a degree out, and at this size the
        # seam is the most visible thing in the icon.
        painter.drawEllipse(QPointF(8.0, 9.8), 4.2, 4.2)
        painter.drawPolygon(
            QPolygonF([QPointF(8.0, 1.4), QPointF(4.4, 9.4), QPointF(11.6, 9.4)])
        )
    finally:
        painter.end()
    return chip_icon(pixmap)


def lock_icon(locked: bool = True, size: int = ICON):
    """Return a padlock, its shackle open or closed, *size* pixels square.

    The shackle is the whole of the difference between the two states, so it is
    drawn as a stroke rather than filled -- at sixteen pixels a filled arch and
    an open one are the same blob. Drawn at *size* rather than scaled up after,
    so a larger padlock is as sharp as the others.
    """
    pixmap = _tool_pixmap(size)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _drawn_at(painter, size)
        pen = QPen(QColor(TEXT))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Closed: the shackle sits over the body. Open: it is lifted and turned
        # aside, which is the shape a padlock actually makes.
        left = 4.6 if locked else 7.0
        painter.drawArc(QRectF(left, 2.0, 6.8, 7.6), 0, 180 * 16)
        painter.drawLine(QPointF(left, 5.8), QPointF(left, 8.4))
        painter.drawLine(
            QPointF(left + 6.8, 5.8), QPointF(left + 6.8, 8.4 if locked else 6.6)
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT))
        painter.drawRoundedRect(QRectF(3.0, 8.0, 10.0, 6.6), 1.4, 1.4)
    finally:
        painter.end()
    return chip_icon(pixmap)


def save_icon(size: int = ICON):
    """Return the save button's icon: a disk, its shutter and label punched out.

    The one mark on this bar that stands for writing to the filesystem rather
    than for a tool, so it is the familiar shape rather than an invention: at
    sixteen pixels a reader recognises the outline before reading the caption.
    Both openings are inset from the body's edges -- reaching them cuts the
    body into two columns joined by a band, which draws an H.
    """
    pixmap = _tool_pixmap(size)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _drawn_at(painter, size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT))
        # The body, its lower right corner cut as a real one is.
        painter.drawPolygon(
            QPolygonF([
                QPointF(2.0, 2.0), QPointF(11.8, 2.0), QPointF(14.0, 4.2),
                QPointF(14.0, 14.0), QPointF(2.0, 14.0),
            ])
        )
        # Punched out rather than drawn over, so the two read as openings in
        # the body and the shape is a disk rather than a square with marks on.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.drawRect(QRectF(5.6, 3.0, 4.4, 3.2))
        painter.drawRect(QRectF(4.6, 8.6, 6.8, 3.4))
    finally:
        # Always: a QPainter outliving its paint device crashes the process
        # rather than reporting an error.
        painter.end()
    return chip_icon(pixmap)


def settings_icon():
    """Return the settings button's icon: a gear.

    The one shape a reader recognises without a caption, which is what this
    button needs -- it sits in the corner beside the mark, where there is no
    room for a word and nothing around it to give it context. Drawn as a ring
    of teeth around a punched hub rather than as a filled disc with notches:
    at sixteen pixels a gear is read by its outline, and a solid centre leaves
    it looking like a flower.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TEXT))
        painter.translate(ICON / 2.0, ICON / 2.0)
        # Eight teeth, drawn one rotation apart rather than as one path: a
        # polygon of sixteen points hand-placed at this size drifts, and a
        # tooth a degree out is the most visible thing in the icon.
        for _ in range(8):
            painter.drawRect(QRectF(-1.45, -7.0, 2.9, 2.7))
            painter.rotate(45)
        painter.drawEllipse(QPointF(0.0, 0.0), 4.9, 4.9)
        # Punched out, so the hub is a hole in the body rather than a mark on
        # it -- the same treatment the disk's shutter gets.
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.drawEllipse(QPointF(0.0, 0.0), 1.95, 1.95)
    finally:
        # Always: a QPainter outliving its paint device crashes the process
        # rather than reporting an error.
        painter.end()
    return chip_icon(pixmap)


def _history_icon(forwards: bool, size: int = ICON):
    """Return an arrow curving over and back on itself, pointing left or right.

    One drawing for undo and redo, mirrored about the centre line: they are one
    gesture in two directions, and two arrows drawn by hand at this size never
    quite match. The head sits at the end of the arc and points across it, an
    arc and a head that overlap being a blob at sixteen pixels.
    """
    pixmap = _tool_pixmap(size)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _drawn_at(painter, size)
        if forwards:
            painter.translate(ICON, 0)
            painter.scale(-1.0, 1.0)
        pen = QPen(QColor(TEXT))
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # The top half of an ellipse: over the top and down both sides.
        painter.drawArc(QRectF(3.4, 4.6, 9.6, 9.0), 0, 180 * 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TEXT))
        # On the left leg, pointing down: where the stroke is going.
        painter.drawPolygon(
            QPolygonF([QPointF(0.9, 8.6), QPointF(5.9, 8.6), QPointF(3.4, 13.4)])
        )
    finally:
        painter.end()
    return chip_icon(pixmap)


def undo_icon(size: int = ICON):
    """Return the undo button's icon: an arrow curving back the way it came."""
    return _history_icon(forwards=False, size=size)


def redo_icon(size: int = ICON):
    """Return the redo button's icon: the undo arrow, mirrored."""
    return _history_icon(forwards=True, size=size)
