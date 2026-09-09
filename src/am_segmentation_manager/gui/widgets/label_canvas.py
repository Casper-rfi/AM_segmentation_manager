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

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QCursor, QImage, QPen, QPixmap
from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsView

from ...config import BACKGROUND_CLASS, UNKNOWN_CLASS_COLOR, SegmentationConfig
from ...display import DEFAULT_ALPHA
from ...operations.label_edit import fill, paint, patch_between
from ..preferences import DEFAULT_BRUSH_RADIUS
from .image_view import LABEL_MIN_FIT_FRACTION, ImageView, event_position
from .versions_bar import COMPARE_KEY_CODE

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: Entries in an 8-bit indexed image's colour table.
COLOUR_TABLE_SIZE = 256

#: Drawn above the image, which sits at the default zero. The brush outline goes
#: above both, so it stays visible over an opaque label. Speck rings sit between
#: them: over the annotation they point at, under the tool doing the work.
LABEL_Z = 1.0
SPECK_Z = 1.5
BRUSH_Z = 2.0

#: The ring drawn around a speck, in **screen** pixels.
#:
#: Screen rather than image pixels because the thing being pointed at is four
#: pixels or fewer: at a zoom that fits a 3000-pixel acquisition in a pane, a
#: ring scaled with the image would be smaller than the speck inside it.
SPECK_RING = 9.0
SPECK_PEN = 2.0

#: Magenta, which is the one hue not already spoken for. The classes use
#: orange, green, red and yellow, the workflow states grey, blue and teal, the
#: label lines violet, and white is the cursor and the selection.
SPECK_COLOUR = "#ff2fd0"

#: The two ways of writing a class. The brush draws where it is dragged; the
#: fill floods the run of one class under the pointer, which is how an area
#: enclosed by an annotated boundary is filled without tracing it.
TOOL_BRUSH = "brush"
TOOL_FILL = "fill"
TOOLS = (TOOL_BRUSH, TOOL_FILL)

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


#: How often the pointer is looked up while the window is not the one in
#: front, in milliseconds. macOS delivers no pointer motion at all to a window
#: that is not active, so the brush outline would otherwise sit where it was
#: last drawn until something was clicked. Fast enough to read as tracking,
#: and it runs only while there is an outline to track with.
POINTER_POLL_MS = 50


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

    #: Emitted while the compare key is held, so the panel can put an earlier
    #: version on the canvas and take it off again. The canvas holds the
    #: keyboard; what is worth showing is not its business.
    compare_held = Signal(bool)

    def __init__(self, config: SegmentationConfig) -> None:
        """Create a canvas drawing classes from *config*."""
        super().__init__()
        # One step out from fit, unlike the curation panes: an annotation is
        # easiest to leave a pixel short at the frame's edge, and that edge has
        # to be clear of the pane's border to be checked. See
        # :data:`LABEL_MIN_FIT_FRACTION`.
        self.min_fit_fraction = LABEL_MIN_FIT_FRACTION
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
        self._brush_item.setBrush(QBrush(Qt.NoBrush))
        self._brush_item.setVisible(False)
        self.scene().addItem(self._brush_item)

        #: One ring per speck, rebuilt whenever the specks change.
        self._speck_items: list[QGraphicsEllipseItem] = []

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Started and stopped by _apply_tool_cursor, which already runs on
        # every state that decides whether there is an outline at all.
        self._pointer_poll = QTimer(self)
        self._pointer_poll.setInterval(POINTER_POLL_MS)
        self._pointer_poll.timeout.connect(self._poll_pointer)

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

        # Last, because it is the first thing that reads the state above. A fit
        # changes the transform without necessarily moving a scrollbar, so
        # scrollContentsBy alone does not cover every way the scene moves.
        self.zoomed.connect(self._sync_brush)

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

    def show_specks(self, specks) -> None:
        """Ring each of *specks*, replacing whatever was ringed before.

        The rings ignore the view transform, so each stays the same size on
        screen at any zoom -- see :data:`SPECK_RING`. Passing nothing clears
        them, which is what switching the filter off does.
        """
        for item in self._speck_items:
            self.scene().removeItem(item)
        self._speck_items.clear()

        pen = QPen(QColor(SPECK_COLOUR))
        pen.setWidthF(SPECK_PEN)
        pen.setCosmetic(True)
        for speck in specks:
            ring = QGraphicsEllipseItem(
                -SPECK_RING, -SPECK_RING, 2 * SPECK_RING, 2 * SPECK_RING
            )
            ring.setPen(pen)
            ring.setBrush(QBrush(Qt.NoBrush))
            ring.setZValue(SPECK_Z)
            ring.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            # The middle of the pixel, not its corner.
            ring.setPos(speck.x + 0.5, speck.y + 0.5)
            self.scene().addItem(ring)
            self._speck_items.append(ring)

    def centre_on(self, x: float, y: float) -> None:
        """Bring a point of the label into the middle of the pane."""
        self.centerOn(x + 0.5, y + 0.5)

    def clear(self) -> None:
        """Show nothing, in every layer."""
        super().clear()
        self._label_array = None
        self._label_item.setPixmap(QPixmap())
        self._label_item.setVisible(False)
        self._sync_brush()
        self.show_specks(())

    # -- the tools ---------------------------------------------------------

    def set_editable(self, editable: bool) -> None:
        """Allow or forbid painting.

        Switched off for a frame that must not be edited -- one whose label does
        not match its image -- and while there is no label array to write into.
        """
        self._editable = editable
        if not editable:
            self._cancel_stroke()
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
        """Show the cursor for whatever the left button will do.

        Set on the **viewport**, and set for every state rather than left to
        ``setDragMode``. Qt puts the panning hand on the viewport as a side
        effect of the drag mode *changing* -- so it does nothing at all when the
        mode is already what it is being set to -- and ``QGraphicsView`` puts a
        stored cursor back whenever the pointer moves across a scene whose items
        carry no cursor of their own. Naming the cursor here, at the level Qt
        manipulates, is what keeps the hand on screen for as long as space is
        held.

        The outline goes wherever the cursor does: every state this reads is
        also a state that decides whether there is a brush to draw, and keeping
        the two apart is what left a ring on screen after a fill was chosen and
        no ring at all after a pan ended. The pointer poll starts and stops
        here for the same reason -- it exists to move that outline, so it runs
        exactly while there is one.
        """
        viewport = self.viewport()
        self._sync_brush()
        if self._brush_is_live:
            self._pointer_poll.start()
        else:
            self._pointer_poll.stop()
        if not self.editable or self._space_held:
            # The left button pans in both cases, so the hand is the truth.
            viewport.setCursor(Qt.OpenHandCursor)
        elif self._tool == TOOL_FILL:
            viewport.setCursor(Qt.CrossCursor)
        else:
            viewport.unsetCursor()

    def set_brush_radius(self, radius: int) -> None:
        """Set the brush radius in label pixels."""
        self._brush_radius = max(MIN_BRUSH_RADIUS, min(int(radius), MAX_BRUSH_RADIUS))
        self._sync_brush()

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

    @property
    def _brush_is_live(self) -> bool:
        """Return whether the left button would paint, so there is a ring to draw."""
        return self.editable and not self._space_held and self._tool == TOOL_BRUSH

    def _poll_pointer(self) -> None:
        """Follow the pointer while no event says where it is.

        A window that is not in front receives no pointer motion on macOS, so
        the outline stopped tracking until something was clicked -- one of the
        three faults that share that cause. ``QCursor.pos`` is answered by the
        window system rather than by an event, so it is right either way; this
        only asks it often enough for the answer to be worth having, and does
        nothing at all while the window is the one in front and real events are
        arriving.
        """
        if not self.isActiveWindow():
            self._sync_brush()

    def _move_brush_cursor(self, position) -> None:
        """Put the brush outline under a pointer at *position*, or take it away.

        The one place that decides whether the outline should be showing. Every
        other caller either reports where the pointer is or asks
        :meth:`_sync_brush` to find out.
        """
        if not self._brush_is_live:
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

    def _sync_brush(self) -> None:
        """Redraw the brush outline against a pointer that has not reported in.

        The outline is drawn in *scene* coordinates, and almost everything moves
        the scene under a stationary pointer: a middle-button pan, a fit, going
        to the next speck, a frame arriving on the loader thread. Each leaves a
        ring around a pixel the pointer is no longer over -- and each state
        change that hides it leaves it hidden until the pointer is nudged. So
        every one of them ends here instead of waiting for a mouse move.
        """
        position = self._pointer_position()
        if self.viewport().rect().contains(position):
            self._move_brush_cursor(position)
        else:
            self._brush_item.setVisible(False)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Hide the brush outline when the pointer leaves the pane."""
        self._brush_item.setVisible(False)
        super().leaveEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802 - Qt naming
        """Follow the scene as it moves under the pointer.

        The one hook every kind of view movement passes through: a pan by drag
        or by middle button, a fit, and centring on a speck all end in the
        scrollbars.
        """
        super().scrollContentsBy(dx, dy)
        self._sync_brush()

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
            self.viewport().setCursor(Qt.ClosedHandCursor)
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
        self._move_brush_cursor(position)
        if self._stroke is not None:
            self._extend_stroke(self._label_point(position))
            event.accept()
            return
        super().mouseMoveEvent(event)
        self._keep_pan_cursor()

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

    def _keep_pan_cursor(self) -> None:
        """Put the panning hand back if Qt has just taken it off.

        ``QGraphicsView`` restores a stored viewport cursor as the pointer moves
        across a scene whose items carry no cursor of their own, so a hand set
        when space went down does not survive the drag that follows. Nothing
        changes state during a drag, so this is the only place left to say it.
        Qt's own closed hand, shown while the button is down, is left alone.
        """
        if not self._space_held:
            return
        if self.viewport().cursor().shape() not in (
            Qt.OpenHandCursor, Qt.ClosedHandCursor
        ):
            self.viewport().setCursor(Qt.OpenHandCursor)

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
        """Hold space to pan, or the compare key to look at an earlier version."""
        if event.key() == COMPARE_KEY_CODE and not event.isAutoRepeat():
            self.compare_held.emit(True)
            event.accept()
            return
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self._cancel_stroke()
            self._apply_drag_mode()
            self._apply_tool_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Give the left button back to the brush, and stop comparing."""
        if event.key() == COMPARE_KEY_CODE and not event.isAutoRepeat():
            self.compare_held.emit(False)
            event.accept()
            return
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
