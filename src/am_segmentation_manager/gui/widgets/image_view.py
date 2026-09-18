"""Zoomable image panes that stay in step with one another.

The plain image and the same image under its label overlay are only useful side
by side if they show the same region at the same magnification, so the two views
share their zoom and scroll position: pan one and the other follows.

A pane can be scrolled and zoomed past the edges of what it holds, and shows no
scrollbars; see `docs/display.md`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from qtpy.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...display import HIDDEN_IMAGE_GREY

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

ZOOM_STEP = 1.25
MIN_SCALE = 0.05
MAX_SCALE = 40.0

#: How far below the fit scale a view may be zoomed out, as a fraction of it.
#: Nothing below fit: the whole frame is already on screen, and shrinking it
#: further only strands it in an empty pane. What a view is *for* decides
#: whether it needs the exception below.
MIN_FIT_FRACTION = 1.0

#: The proofreading canvas's floor. Checking that a label runs all the way to
#: the edge of the frame means seeing that edge clear of the pane's border, so
#: one step out from fit is allowed and lands here: :meth:`ImageView.zoom_by`
#: snaps to the floor rather than overshooting it, which makes a step of 0.8
#: stop at a band of 5% on each side rather than the 18% two steps would leave.
LABEL_MIN_FIT_FRACTION = 1 / 1.1

#: Below the picture, which sits at the default zero. The proofreading canvas
#: puts its label above both; nothing is drawn under this.
BACKDROP_Z = -1.0

#: Room around the image to scroll into, as a fraction of its size. Without it
#: the furthest an edge can be brought is the border of the pane, where half the
#: brush is off-screen -- and the edges are exactly where an annotation is
#: easiest to leave a pixel short.
SCENE_MARGIN = 0.25


def event_position(event):
    """Return a mouse or wheel event's position as a QPoint, under either binding.

    Qt 6 replaced ``pos()`` with ``position()``, and this package supports
    both. Here rather than beside the canvas that paints, because the wheel
    zoom reads it too: the event is the one thing that always knows where the
    pointer was.
    """
    position = getattr(event, "position", None)
    return position().toPoint() if position is not None else event.pos()


def to_qimage(array: np.ndarray) -> QImage:
    """Convert a display array to a QImage.

    Accepts the two shapes the display pipeline produces: 2D greyscale and
    ``(H, W, 3)`` RGB, both 8-bit. The buffer is copied, because the numpy array
    may be evicted from the cache while Qt is still drawing from it.
    """
    import numpy as np

    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        height, width = array.shape
        image = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        image = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    else:
        raise ValueError(f"Cannot display an array of shape {array.shape}")
    return image.copy()


def _with_margin(rect):
    """Return *rect* with :data:`SCENE_MARGIN` of room added on every side."""
    return rect.adjusted(
        -SCENE_MARGIN * rect.width(), -SCENE_MARGIN * rect.height(),
        SCENE_MARGIN * rect.width(), SCENE_MARGIN * rect.height(),
    )


class ImageView(QGraphicsView):
    """A single pane showing one image, with wheel zoom and drag panning."""

    #: Emitted after the view is zoomed, so a linked view can follow.
    #: Zoom must be reported explicitly: an image that fits its pane has no
    #: scrollbars, so scroll signals alone cannot carry it.
    zoomed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        # Under the picture, and the size of it. Drawn only where the picture
        # is hidden, so what is left on screen still says where the frame ends;
        # see :data:`~...display.HIDDEN_IMAGE_GREY`.
        self._backdrop = QGraphicsRectItem()
        self._backdrop.setPen(QPen(Qt.PenStyle.NoPen))
        self._backdrop.setBrush(QColor(*HIDDEN_IMAGE_GREY))
        self._backdrop.setZValue(BACKDROP_Z)
        self._backdrop.setVisible(False)
        self._scene.addItem(self._backdrop)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # No scrollbars: the pane is driven by dragging and the wheel, and the
        # scrollable area is deliberately larger than the image (SCENE_MARGIN),
        # so bars would be permanently visible and permanently not full.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # NoAnchor, and the wheel anchors itself: see zoom_by. Qt's own
        # AnchorUnderMouse anchors on the scene point recorded by the last
        # mouse event this view handled, which is not where the pointer is
        # whenever no such event arrived -- and two ordinary things stop them
        # arriving. A window that is not in front receives no pointer motion at
        # all on macOS; and the proofreading canvas accepts a mouse move
        # without passing it on for the whole of a stroke, so painting freezes
        # the recorded point until some later move happens to fall through.
        # Either way the next wheel zoom pulls the image towards a pixel the
        # pointer left some time ago.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._shape: tuple[int, int] | None = None
        self._image_shown = True
        #: Where the view is looking, as a fraction of the picture, held across
        #: a change of resolution. Kept rather than re-derived every time
        #: because deriving it goes through the scroll position, which is a
        #: whole number: re-deriving on every swap loses a fraction of a pixel
        #: the same way each time, and stepping through a series drifts.
        #: Dropped as soon as the user moves the view themselves.
        self._anchor: tuple[float, float] | None = None
        self._centring = False
        #: How far below fit this view zooms out; see :data:`MIN_FIT_FRACTION`.
        #: Per view rather than per module: the curation panes are read at fit
        #: and the proofreading canvas is worked at its edges.
        self.min_fit_fraction = MIN_FIT_FRACTION

    def set_array(
        self,
        array: np.ndarray,
        *,
        device_pixel_ratio: float = 1.0,
        native: bool = False,
        same_picture: bool = False,
    ) -> None:
        """Show *array*, fitting the view when the image dimensions change.

        The zoom is otherwise kept, so stepping through a lamella holds position
        on the same region instead of resetting on every frame.

        Args:
            array: The image to show.
            device_pixel_ratio: How many physical pixels the array has per
                logical pixel. Telling Qt this stops it magnifying an array
                that was already prepared at the screen's real resolution.
            native: Show at exactly one array pixel per physical pixel instead
                of scaling to fit. Only correct when the array was prepared to
                fit the pane; it is what removes the last resampling step for
                content whose sharpness matters, such as rendered text.
            same_picture: The array shows what is already on screen at a
                different resolution. Dimensions changing normally means a new
                subject and the view refits; here refitting would throw away
                where the user had zoomed to, at the moment they are looking
                at it, so the region on screen is kept instead.
        """
        shape = array.shape[:2]
        before = self._item.boundingRect()
        centre = self.mapToScene(self.viewport().rect().center())
        pixmap = QPixmap.fromImage(to_qimage(array))
        if device_pixel_ratio and device_pixel_ratio != 1.0:
            pixmap.setDevicePixelRatio(device_pixel_ratio)
        # Replacing the picture moves the scrollbars by itself, and so does
        # every step of the rescale below. None of that is the user looking
        # somewhere else, so the held viewpoint has to survive all of it.
        self._centring = True
        try:
            self._set_pixmap(pixmap, shape, before, centre, native, same_picture)
        finally:
            self._centring = False

    def _set_pixmap(self, pixmap, shape, before, centre, native, same_picture) -> None:
        """Put *pixmap* on screen and decide where the view should be looking."""
        self._item.setPixmap(pixmap)
        self._sync_backdrop()
        self._scene.setSceneRect(_with_margin(self._item.boundingRect()))
        if shape == self._shape:
            return
        self._shape = shape
        if same_picture and before.width() and self._item.boundingRect().width():
            self._rescale_onto(before, centre)
        elif native:
            # An identity transform means Qt blits the pixmap rather than
            # resampling it; any other scale, even 0.996, filters every pixel.
            # Centred as fitting would centre: resetting the transform says
            # nothing about where the view is looking, and the scene is larger
            # than the picture, so on its own it leaves the picture off to one
            # side of the pane.
            self.resetTransform()
            self.centerOn(self._item)
            self.zoomed.emit()
        else:
            self.fit()

    def _rescale_onto(self, before, centre) -> None:
        """Keep the region on screen across a change of resolution.

        Scene coordinates are the array's own, so the same picture at twice the
        pixels is the same view at half the scale about a doubled centre.
        """
        after = self._item.boundingRect()
        ratio = before.width() / after.width()
        anchor = self._anchor or (
            centre.x() / before.width(), centre.y() / before.height()
        )
        scale = self.transform().m11() * ratio
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(anchor[0] * after.width(), anchor[1] * after.height())
        self._anchor = anchor
        self.zoomed.emit()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802 - Qt naming
        """Note that the view moved, unless this widget moved it itself."""
        super().scrollContentsBy(dx, dy)
        if not self._centring:
            self._anchor = None

    def set_smooth_scaling(self, smooth: bool) -> None:
        """Choose how the pixmap is resampled when it is not drawn 1:1.

        ``QGraphicsPixmapItem`` overrides the view's render hint with its own
        transformation mode, whose default is nearest-neighbour -- so a pane
        showing something much larger than itself drops pixels rather than
        averaging them, which on a figure full of one-pixel rules and small
        text is the difference between reading it and not.

        Off for acquisitions: zoomed *in* on a label, nearest neighbour is what
        shows where one pixel ends and the next begins, which is the whole
        point of looking that closely.
        """
        self._item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation
        )

    def set_image_shown(self, shown: bool) -> None:
        """Show or hide the image, keeping what it holds and where it sits.

        Hiding it leaves anything drawn above it -- the label layer of the
        proofreading canvas -- alone against a flat ground, which is what makes
        an annotation of a few pixels findable. That ground is the backdrop
        rather than the pane behind it, so the frame's own edge is still
        visible with nothing drawn along it.
        """
        self._image_shown = shown
        self._item.setVisible(shown)
        self._sync_backdrop()

    def _sync_backdrop(self) -> None:
        """Keep the backdrop the size of the picture, and shown only without it."""
        self._backdrop.setRect(self._item.boundingRect())
        self._backdrop.setVisible(not self._image_shown)

    @property
    def image_shown(self) -> bool:
        """Return whether the image is being drawn."""
        return self._image_shown

    def clear(self) -> None:
        """Show nothing."""
        self._item.setPixmap(QPixmap())
        self._sync_backdrop()
        self._shape = None

    def fit(self) -> None:
        """Scale so the whole image is visible."""
        self._anchor = None
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoomed.emit()

    def zoom_fully_out(self) -> None:
        """Show the whole image centred, with this view's margin around it.

        The furthest out the wheel goes, and where :meth:`zoom_by` lands when
        it reaches the floor. Centred rather than scaled in place: the anchor
        that makes zooming *in* follow the pointer would otherwise leave a
        frame small enough to fit the pane pushed into a corner of it.

        :attr:`min_fit_fraction` sets the margin, and at 1.0 -- its largest --
        this is exactly :meth:`fit`.
        """
        rect = self._item.sceneBoundingRect()
        if rect.isEmpty():
            return
        margin = (1.0 / self.min_fit_fraction - 1.0) / 2.0
        self.fitInView(
            rect.adjusted(
                -margin * rect.width(), -margin * rect.height(),
                margin * rect.width(), margin * rect.height(),
            ),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.zoomed.emit()

    def fit_scale(self) -> float:
        """Return the scale at which the image exactly fills the pane."""
        rect = self._item.boundingRect()
        if rect.isEmpty():
            return MIN_SCALE
        viewport = self.viewport().rect()
        return min(viewport.width() / rect.width(), viewport.height() / rect.height())

    def zoom_by(self, factor: float, at=None) -> None:
        """Scale the view by *factor*, within limits, and report the change.

        Zooming out stops at :attr:`min_fit_fraction` of the fit scale, which
        is fit itself unless the view asked for room around the frame, and
        lands there through :meth:`zoom_fully_out`. Below that there is nothing
        to gain: shrinking further only strands the image in an empty view.
        Zooming *in* is limited only by :data:`MAX_SCALE`.

        The lower limit deliberately applies to one direction only. Applying it
        to both blocks zooming in whenever the current scale happens to sit
        below it -- which it does after every :meth:`fit`, because ``fitInView``
        leaves room for the frame -- and the view then refuses to zoom at all.

        Separate from :meth:`wheelEvent` so the limits can be exercised without
        synthesising wheel events.

        Args:
            at: The point in the viewport to hold still, or None for its
                centre. Passed rather than looked up, because the event that
                asks for the zoom is the only thing that reliably knows where
                the pointer was -- see the anchor note in ``__init__``.
        """
        current = self.transform().m11()
        target = current * factor
        # The user is choosing where to look, so whatever was being held across
        # a change of resolution no longer describes it.
        self._anchor = None

        if factor < 1.0:
            if target < self.fit_scale() * self.min_fit_fraction:
                # Land on the floor rather than overshooting past it. Through
                # fitInView rather than by scaling, so the frame arrives
                # centred and with the same two-pixel border fit() leaves --
                # scaling to the computed floor would crop it.
                self.zoom_fully_out()
                return
        elif target > MAX_SCALE:
            return

        anchor = self.viewport().rect().center() if at is None else at
        before = self.mapToScene(anchor)
        self.scale(factor, factor)
        # Put back under *anchor* whatever was under it, by moving the view
        # rather than the transform: the linked pane copies this one's
        # transform, so the transform has to stay a plain scale.
        self.centerOn(
            self.mapToScene(self.viewport().rect().center())
            + (before - self.mapToScene(anchor))
        )
        self.zoomed.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zoom about the pointer, which the event itself says where it is."""
        self.zoom_by(
            ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP,
            at=event_position(event),
        )


class LinkedImageViews(QWidget):
    """Two captioned panes kept at the same zoom and scroll position."""

    #: Gap between the two panes. Deliberately tight: the panes are the point of
    #: the window, and every pixel of margin is taken from the image.
    PANE_SPACING = 4

    def __init__(self, left_caption: str, right_caption: str) -> None:
        super().__init__()
        self.left = ImageView()
        self.right = ImageView()
        self._syncing = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self.PANE_SPACING)

        for view, caption in ((self.left, left_caption), (self.right, right_caption)):
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            label = QLabel(caption)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(label)
            column.addWidget(view)
            row.addLayout(column)

        self._link(self.left, self.right)
        self._link(self.right, self.left)

    def _link(self, source: ImageView, target: ImageView) -> None:
        """Mirror *source*'s zoom and scroll position onto *target*.

        Zoom is mirrored from its own signal rather than inferred from scroll
        events. An image that fits its pane has no scrollbars to move, so a
        scroll-only link silently stops synchronising at exactly the point the
        user zooms out far enough to see the whole frame.
        """
        source.zoomed.connect(lambda: self._mirror_zoom(source, target))
        source.horizontalScrollBar().valueChanged.connect(
            lambda value: self._mirror_scroll(target.horizontalScrollBar(), value)
        )
        source.verticalScrollBar().valueChanged.connect(
            lambda value: self._mirror_scroll(target.verticalScrollBar(), value)
        )

    def _mirror_zoom(self, source: ImageView, target: ImageView) -> None:
        """Copy *source*'s transform onto *target*, without recursing."""
        if self._syncing:
            return
        self._syncing = True
        try:
            if target.transform() != source.transform():
                target.setTransform(source.transform())
            target.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
            target.verticalScrollBar().setValue(source.verticalScrollBar().value())
        finally:
            self._syncing = False

    def _mirror_scroll(self, scrollbar, value: int) -> None:
        """Copy a scroll position across, without recursing."""
        if self._syncing:
            return
        self._syncing = True
        try:
            scrollbar.setValue(value)
        finally:
            self._syncing = False

    def fit(self) -> None:
        """Fit both panes to their images."""
        self.left.fit()
        self.right.fit()

    def clear(self) -> None:
        """Empty both panes."""
        self.left.clear()
        self.right.clear()
