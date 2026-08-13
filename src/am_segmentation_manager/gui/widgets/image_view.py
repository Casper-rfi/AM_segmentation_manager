"""Zoomable image panes that stay in step with one another.

The plain image and the same image under its label overlay are only useful side
by side if they show the same region at the same magnification, so the two views
share their zoom and scroll position: pan one and the other follows.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
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


def to_qimage(array: "np.ndarray") -> QImage:
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


class ImageView(QGraphicsView):
    """A single pane showing one image, with wheel zoom and drag panning."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.black)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._shape: tuple[int, int] | None = None

    def set_array(self, array: "np.ndarray") -> None:
        """Show *array*, fitting the view when the image dimensions change.

        The zoom is otherwise kept, so stepping through a lamella holds position
        on the same region instead of resetting on every frame.
        """
        shape = array.shape[:2]
        pixmap = QPixmap.fromImage(to_qimage(array))
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())
        if shape != self._shape:
            self._shape = shape
            self.fit()

    def clear(self) -> None:
        """Show nothing."""
        self._item.setPixmap(QPixmap())
        self._shape = None

    def fit(self) -> None:
        """Scale so the whole image is visible."""
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zoom about the cursor, within sane limits."""
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        scale = self.transform().m11() * factor
        if MIN_SCALE <= scale <= MAX_SCALE:
            self.scale(factor, factor)


class LinkedImageViews(QWidget):
    """Two captioned panes kept at the same zoom and scroll position."""

    def __init__(self, left_caption: str, right_caption: str) -> None:
        super().__init__()
        self.left = ImageView()
        self.right = ImageView()
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        panes = QWidget()
        panes_layout = QVBoxLayout(panes)
        panes_layout.setContentsMargins(0, 0, 0, 0)

        from qtpy.QtWidgets import QHBoxLayout

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for view, caption in ((self.left, left_caption), (self.right, right_caption)):
            column = QVBoxLayout()
            label = QLabel(caption)
            label.setAlignment(Qt.AlignCenter)
            column.addWidget(label)
            column.addWidget(view)
            container = QWidget()
            container.setLayout(column)
            row.addWidget(container)
        panes_layout.addLayout(row)
        layout.addWidget(panes)

        self._link(self.left, self.right)
        self._link(self.right, self.left)

    def _link(self, source: ImageView, target: ImageView) -> None:
        """Mirror *source*'s scroll position onto *target*."""
        source.horizontalScrollBar().valueChanged.connect(
            lambda value: self._mirror(target.horizontalScrollBar(), value, source, target)
        )
        source.verticalScrollBar().valueChanged.connect(
            lambda value: self._mirror(target.verticalScrollBar(), value, source, target)
        )

    def _mirror(self, scrollbar, value: int, source: ImageView, target: ImageView) -> None:
        """Copy zoom and scroll from *source* to *target*, without recursing."""
        if self._syncing:
            return
        self._syncing = True
        try:
            if target.transform() != source.transform():
                target.setTransform(source.transform())
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
