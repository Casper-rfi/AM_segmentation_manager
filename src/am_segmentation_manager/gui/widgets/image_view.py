"""Zoomable image panes that stay in step with one another.

The plain image and the same image under its label overlay are only useful side
by side if they show the same region at the same magnification, so the two views
share their zoom and scroll position: pan one and the other follows.

A pane can be scrolled and zoomed past the edges of what it holds, and shows no
scrollbars; see `docs/decisions.md` § Display.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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

#: Room around the image to scroll into, as a fraction of its size. Without it
#: the furthest an edge can be brought is the border of the pane, where half the
#: brush is off-screen -- and the edges are exactly where an annotation is
#: easiest to leave a pixel short.
SCENE_MARGIN = 0.25


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
        image = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
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
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # No scrollbars: the pane is driven by dragging and the wheel, and the
        # scrollable area is deliberately larger than the image (SCENE_MARGIN),
        # so bars would be permanently visible and permanently not full.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.black)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._shape: tuple[int, int] | None = None
        self._image_shown = True
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
        """
        shape = array.shape[:2]
        pixmap = QPixmap.fromImage(to_qimage(array))
        if device_pixel_ratio and device_pixel_ratio != 1.0:
            pixmap.setDevicePixelRatio(device_pixel_ratio)
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(_with_margin(self._item.boundingRect()))
        if shape != self._shape:
            self._shape = shape
            if native:
                # An identity transform means Qt blits the pixmap rather than
                # resampling it; any other scale, even 0.996, filters every
                # pixel.
                self.resetTransform()
                self.zoomed.emit()
            else:
                self.fit()

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
            Qt.SmoothTransformation if smooth else Qt.FastTransformation
        )

    def set_image_shown(self, shown: bool) -> None:
        """Show or hide the image, keeping what it holds and where it sits.

        Hiding it leaves anything drawn above it -- the label layer of the
        proofreading canvas -- alone against the pane's black, which is what
        makes an annotation of a few pixels findable.
        """
        self._image_shown = shown
        self._item.setVisible(shown)

    @property
    def image_shown(self) -> bool:
        """Return whether the image is being drawn."""
        return self._image_shown

    def clear(self) -> None:
        """Show nothing."""
        self._item.setPixmap(QPixmap())
        self._shape = None

    def fit(self) -> None:
        """Scale so the whole image is visible."""
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.KeepAspectRatio)
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
            Qt.KeepAspectRatio,
        )
        self.zoomed.emit()

    def fit_scale(self) -> float:
        """Return the scale at which the image exactly fills the pane."""
        rect = self._item.boundingRect()
        if rect.isEmpty():
            return MIN_SCALE
        viewport = self.viewport().rect()
        return min(viewport.width() / rect.width(), viewport.height() / rect.height())

    def zoom_by(self, factor: float) -> None:
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
        """
        current = self.transform().m11()
        target = current * factor

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

        self.scale(factor, factor)
        self.zoomed.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zoom about the cursor."""
        self.zoom_by(ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP)


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
            label.setAlignment(Qt.AlignCenter)
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
