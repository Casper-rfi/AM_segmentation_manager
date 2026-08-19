"""The proofreading canvas: an acquisition, its label, and the tools to change it.

The image and the label are two stacked graphics items rather than one composed
array. That is what makes the opacity control free -- it is the label item's own
opacity, not a re-blend of every pixel -- and it is what lets the label layer be
redrawn without touching the image. The curation view composes instead, which is
the right choice there: it shows a reduced frame and never changes one.

The label is drawn as an 8-bit indexed image whose colour table carries the
alpha, so the background class is transparent without a fourth channel: a
full-resolution label costs one byte a pixel here rather than four.

The canvas draws into an array it does not own. Whoever supplies that array owns
what happens to the edit; this widget reports each stroke and holds nothing.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QCursor, QImage, QPen, QPixmap
from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsView

from ...config import BACKGROUND_CLASS, UNKNOWN_CLASS_COLOR, SegmentationConfig
from ...display import DEFAULT_ALPHA
from ...operations.label_edit import fill, paint, patch_between
from .image_view import ImageView

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: Entries in an 8-bit indexed image's colour table.
COLOUR_TABLE_SIZE = 256

#: Drawn above the image, which sits at the default zero. The brush outline goes
#: above both, so it stays visible over an opaque label.
LABEL_Z = 1.0
BRUSH_Z = 2.0

#: The two ways of writing a class. The brush draws where it is dragged; the
#: fill floods the run of one class under the pointer, which is how an area
#: enclosed by an annotated boundary is filled without tracing it.
TOOL_BRUSH = "brush"
TOOL_FILL = "fill"
TOOLS = (TOOL_BRUSH, TOOL_FILL)

#: Brush radius, in label pixels, before the reviewer chooses one.
DEFAULT_BRUSH_RADIUS = 8

#: The widest and narrowest brush offered. The upper end is a whole structure at
#: once on a 3000-pixel acquisition; the lower is a single pixel, for the last
#: correction along an edge.
MIN_BRUSH_RADIUS = 0
MAX_BRUSH_RADIUS = 200


def label_colour_table(config: SegmentationConfig) -> list[int]:
    """Return the ARGB colour table for drawing a label as an indexed image.

    The background class is fully transparent, so it needs no special case when
    drawing. Every class the config does not define takes
    :data:`~am_segmentation_manager.config.UNKNOWN_CLASS_COLOR` opaquely, on the
    same reasoning as the composed overlay: an unexpected label value has to be
    visible rather than silently invisible. The config is asked only about the
    classes it declares, so filling the table cannot emit a warning per unused
    entry.
    """
    known = set(config.classes)
    table = []
    for index in range(COLOUR_TABLE_SIZE):
        if index == BACKGROUND_CLASS:
            table.append(0)
            continue
        red, green, blue = (
            config.rgb(index) if index in known else UNKNOWN_CLASS_COLOR
        )
        table.append((0xFF << 24) | (red << 16) | (green << 8) | blue)
    return table


def label_to_qimage(label: np.ndarray, colour_table: list[int]) -> QImage:
    """Convert a label array to an indexed 8-bit QImage with *colour_table*.

    Values above the table's range are clamped to its last entry rather than
    wrapped, so a class too large to draw shows as an unknown class instead of
    as some other class. The buffer is copied, because the array it came from
    may be edited or evicted while Qt is still drawing.

    Args:
        label: 2D array of class indices, of any integer dtype.
        colour_table: 256 ARGB values, as :func:`label_colour_table` returns.
    """
    import numpy as np

    if label.dtype != np.uint8:
        label = np.clip(label, 0, COLOUR_TABLE_SIZE - 1).astype(np.uint8)
    label = np.ascontiguousarray(label)
    height, width = label.shape[:2]
    image = QImage(label.data, width, height, width, QImage.Format_Indexed8)
    image.setColorTable(colour_table)
    return image.copy()


def event_position(event):
    """Return a mouse event's position as a QPoint, under either Qt version.

    Qt 6 replaced ``pos()`` with ``position()``, and this package supports both
    bindings.
    """
    position = getattr(event, "position", None)
    return position().toPoint() if position is not None else event.pos()


class LabelCanvas(ImageView):
    """One acquisition with its label above it, and a brush that edits it.

    Inherits the wheel zoom, fit and drag panning of the curation view's panes,
    so proofreading and reviewing behave the same way where they do the same
    thing. While editing, the left button paints instead of panning; hold space,
    or drag with the middle button, to pan.
    """

    #: Emitted with the :class:`~...operations.label_edit.EditPatch` of a
    #: completed stroke. One signal per stroke, not per mouse move, so a drag is
    #: one undo step.
    edited = Signal(object)

    #: Emitted with +1 or -1 when Shift and the wheel ask for a bigger or
    #: smaller brush. Reported rather than applied: which sizes are offered is
    #: the panel's, and the slider has to move with the wheel.
    brush_stepped = Signal(int)

    def __init__(self, config: SegmentationConfig) -> None:
        """Create a canvas drawing classes from *config*."""
        super().__init__()
        self.config = config
        self._colour_table = label_colour_table(config)

        self._label_item = QGraphicsPixmapItem()
        self._label_item.setZValue(LABEL_Z)
        self._label_item.setOpacity(DEFAULT_ALPHA)
        self.scene().addItem(self._label_item)
        self._label_shown = True

        self._brush_item = QGraphicsEllipseItem()
        self._brush_item.setZValue(BRUSH_Z)
        pen = QPen(Qt.white)
        # Cosmetic, so the outline stays one pixel wide at any zoom rather than
        # growing into a band that hides the edge being painted.
        pen.setCosmetic(True)
        self._brush_item.setPen(pen)
        self._brush_item.setBrush(Qt.NoBrush)
        self._brush_item.setVisible(False)
        self.scene().addItem(self._brush_item)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._label_array: np.ndarray | None = None
        self._editable = False
        self._tool = TOOL_BRUSH
        self._brush_radius = DEFAULT_BRUSH_RADIUS
        foreground = config.foreground_classes
        self._paint_value = foreground[0] if foreground else BACKGROUND_CLASS
        self._protected: frozenset[int] = frozenset()

        self._stroke: list[tuple[int, int]] | None = None
        self._snapshot: np.ndarray | None = None
        self._pan_from = None
        self._space_held = False

    # -- what is shown -----------------------------------------------------

    def set_frame(self, image: np.ndarray, label: np.ndarray | None) -> None:
        """Show *image*, with *label* over it when there is one.

        Args:
            image: 2D uint8 greyscale array at full resolution.
            label: The array to draw and, when editing is on, to paint into.
                Held by reference, not copied: the caller owns the edit.
        """
        self.set_array(image)
        self.set_label(label)

    def set_label(self, label: np.ndarray | None) -> None:
        """Redraw the label layer alone, leaving the image and the zoom alone."""
        self._label_array = label
        if label is None:
            self._label_item.setPixmap(QPixmap())
            self._label_item.setVisible(False)
            return
        self._label_item.setPixmap(
            QPixmap.fromImage(label_to_qimage(label, self._colour_table))
        )
        self._label_item.setVisible(self._label_shown)

    def set_label_opacity(self, alpha: float) -> None:
        """Set how strongly the label is drawn, 0.0 to 1.0."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self._label_item.setOpacity(alpha)

    def set_label_shown(self, shown: bool) -> None:
        """Show or hide the label layer, keeping what it holds."""
        self._label_shown = shown
        self._label_item.setVisible(shown and not self._label_item.pixmap().isNull())

    @property
    def label_shown(self) -> bool:
        """Return whether the label layer is being drawn."""
        return self._label_shown

    def clear(self) -> None:
        """Show nothing, in every layer."""
        super().clear()
        self._label_array = None
        self._label_item.setPixmap(QPixmap())
        self._label_item.setVisible(False)
        self._brush_item.setVisible(False)

    # -- the tools ---------------------------------------------------------

    def set_editable(self, editable: bool) -> None:
        """Allow or forbid painting.

        Switched off for a frame that must not be edited -- one whose label does
        not match its image -- and while there is no label array to write into.
        """
        self._editable = editable
        if not editable:
            self._cancel_stroke()
            self._brush_item.setVisible(False)
        else:
            # A frame is prepared on another thread, so editing usually becomes
            # possible while the pointer is already sitting still over the pane.
            # Without this the brush is invisible until it is moved.
            self._show_brush_at_pointer()
        self._apply_drag_mode()
        self._apply_tool_cursor()

    @property
    def editable(self) -> bool:
        """Return whether painting is allowed."""
        return self._editable and self._label_array is not None

    def set_tool(self, tool: str) -> None:
        """Choose whether the left button brushes or fills.

        Raises:
            ValueError: If *tool* is not one of :data:`TOOLS`.
        """
        if tool not in TOOLS:
            raise ValueError(f"Unknown tool: {tool}")
        self._tool = tool
        self._cancel_stroke()
        self._apply_tool_cursor()

    @property
    def tool(self) -> str:
        """Return which tool the left button uses."""
        return self._tool

    def _apply_tool_cursor(self) -> None:
        """Show the brush outline for the brush, and a crosshair for the fill."""
        if not self.editable or self._space_held:
            self._brush_item.setVisible(False)
            self.unsetCursor()
            return
        if self._tool == TOOL_FILL:
            self._brush_item.setVisible(False)
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def set_brush_radius(self, radius: int) -> None:
        """Set the brush radius in label pixels."""
        self._brush_radius = max(MIN_BRUSH_RADIUS, min(int(radius), MAX_BRUSH_RADIUS))
        self._resize_brush_cursor()

    @property
    def brush_radius(self) -> int:
        """Return the brush radius in label pixels."""
        return self._brush_radius

    def set_paint_value(self, value: int) -> None:
        """Set the class the brush writes. Background is how a stroke erases."""
        self._paint_value = int(value)

    @property
    def paint_value(self) -> int:
        """Return the class the brush writes."""
        return self._paint_value

    def set_protected(self, classes) -> None:
        """Lock *classes* so no tool can paint over them."""
        self._protected = frozenset(int(value) for value in classes)

    @property
    def protected(self) -> frozenset[int]:
        """Return the locked classes."""
        return self._protected

    # -- pointer -----------------------------------------------------------

    def _label_point(self, position) -> tuple[int, int]:
        """Return the label pixel under a view position."""
        scene = self.mapToScene(position)
        return int(scene.x() // 1), int(scene.y() // 1)

    def _apply_drag_mode(self) -> None:
        """Give the left button to the brush, or to panning."""
        painting = self.editable and not self._space_held
        self.setDragMode(
            QGraphicsView.NoDrag if painting else QGraphicsView.ScrollHandDrag
        )

    def _resize_brush_cursor(self) -> None:
        """Redraw the brush outline at the current radius, where it already is."""
        rect = self._brush_item.rect()
        centre = rect.center()
        size = 2 * self._brush_radius + 1
        self._brush_item.setRect(
            centre.x() - size / 2, centre.y() - size / 2, size, size
        )

    def _move_brush_cursor(self, position) -> None:
        """Put the brush outline under the pointer."""
        if not self.editable or self._space_held:
            self._brush_item.setVisible(False)
            return
        scene = self.mapToScene(position)
        size = 2 * self._brush_radius + 1
        self._brush_item.setRect(
            scene.x() - size / 2, scene.y() - size / 2, size, size
        )
        self._brush_item.setVisible(True)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Take the keyboard and show the brush when the pointer arrives.

        Focus follows the pointer here because the keys that matter while
        painting -- space to pan above all -- are read by this widget, and a
        tool button clicked a moment ago would otherwise still hold the
        keyboard and swallow them.
        """
        self.setFocus(Qt.MouseFocusReason)
        self._move_brush_cursor(event_position(event))
        super().enterEvent(event)

    def _pointer_position(self):
        """Return where the pointer is, in viewport coordinates."""
        return self.viewport().mapFromGlobal(QCursor.pos())

    def _show_brush_at_pointer(self) -> None:
        """Draw the brush outline where the pointer is, if it is over the pane."""
        position = self._pointer_position()
        if self.viewport().rect().contains(position):
            self._move_brush_cursor(position)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Hide the brush outline when the pointer leaves the pane."""
        self._brush_item.setVisible(False)
        super().leaveEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop panning when the keyboard goes elsewhere.

        The release of a held space arrives at whatever has focus, so without
        this a window switch mid-pan leaves the left button panning with no
        key held down and no way to get the brush back.
        """
        self._release_space()
        super().focusOutEvent(event)

    # -- painting ----------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Start a stroke, or start panning."""
        if event.button() == Qt.MiddleButton:
            self._pan_from = event_position(event)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.editable and not self._space_held:
            point = self._label_point(event_position(event))
            if self._tool == TOOL_FILL:
                self._fill(point)
            else:
                self._begin_stroke(point)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Extend a stroke, pan, or just move the brush outline."""
        position = event_position(event)
        if self._pan_from is not None:
            self._pan_by(position)
            event.accept()
            return
        if self._tool == TOOL_BRUSH:
            self._move_brush_cursor(position)
        if self._stroke is not None:
            self._extend_stroke(self._label_point(position))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Finish a stroke or a pan, reporting the stroke as one edit."""
        if event.button() == Qt.MiddleButton and self._pan_from is not None:
            self._pan_from = None
            self._apply_tool_cursor()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._stroke is not None:
            self._end_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _pan_by(self, position) -> None:
        """Scroll by how far the pointer moved since the last report."""
        delta = position - self._pan_from
        self._pan_from = position
        horizontal = self.horizontalScrollBar()
        vertical = self.verticalScrollBar()
        horizontal.setValue(horizontal.value() - delta.x())
        vertical.setValue(vertical.value() - delta.y())

    def _begin_stroke(self, point: tuple[int, int]) -> None:
        """Take the before-image of a stroke and lay down its first dab.

        The whole label is copied rather than the region the stroke will cover,
        because where a stroke will go is not known until it ends.
        """
        if self._label_array is None:
            return
        self._snapshot = self._label_array.copy()
        self._stroke = [point]
        self._paint([point])

    def _extend_stroke(self, point: tuple[int, int]) -> None:
        """Draw from the last reported position to this one."""
        if self._stroke is None:
            return
        last = self._stroke[-1]
        if point == last:
            return
        self._stroke.append(point)
        self._paint([last, point])

    def _paint(self, points: list[tuple[int, int]]) -> None:
        """Apply one segment of a stroke and redraw the label layer."""
        if self._label_array is None:
            return
        paint(
            self._label_array,
            points,
            self._brush_radius,
            self._paint_value,
            protected=self._protected,
        )
        self.set_label(self._label_array)

    def _fill(self, point: tuple[int, int]) -> None:
        """Flood from *point* and report it, as one edit.

        Unlike a stroke there is nothing to accumulate: a fill is a single
        operation, so it is applied and reported in one step.
        """
        if self._label_array is None:
            return
        patch = fill(
            self._label_array,
            point,
            self._paint_value,
            protected=self._protected,
        )
        if patch is None:
            return
        self.set_label(self._label_array)
        self.edited.emit(patch)

    def _end_stroke(self) -> None:
        """Report the whole stroke as a single reversible edit."""
        snapshot, self._snapshot = self._snapshot, None
        self._stroke = None
        if snapshot is None or self._label_array is None:
            return
        patch = patch_between(snapshot, self._label_array)
        if patch is not None:
            self.edited.emit(patch)

    def _cancel_stroke(self) -> None:
        """Drop a stroke in progress without reporting it."""
        self._stroke = None
        self._snapshot = None

    def refresh_label(self) -> None:
        """Redraw the label layer from the array it was given.

        Called after an undo, which changes the array without going through a
        stroke.
        """
        self.set_label(self._label_array)

    # -- panning by keyboard -----------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zoom, or resize the brush when Shift is held.

        macOS turns Shift and the wheel into a *horizontal* scroll, so the
        delta arrives on the x axis there and on the y axis everywhere else.
        Reading both is what makes this work on either.
        """
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta()
            step = delta.y() or delta.x()
            if step:
                self.brush_stepped.emit(1 if step > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Hold space to pan with the left button instead of painting."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self._cancel_stroke()
            self._brush_item.setVisible(False)
            self._apply_drag_mode()
            self._apply_tool_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Give the left button back to the brush."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._release_space()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _release_space(self) -> None:
        """Give the left button back to the brush, however space was let go."""
        if not self._space_held:
            return
        self._space_held = False
        self._apply_drag_mode()
        self._apply_tool_cursor()
