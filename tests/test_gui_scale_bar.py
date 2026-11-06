"""The scale bar every acquisition and grid overview carries.

Where it is drawn is asserted on the pixels, as `docs/display.md` asks of
anything about how something renders; its length is asserted on the layout the
pane works out, the arithmetic behind it being `display.scale_bar_length_um`'s.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import destroy

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

pytestmark = pytest.mark.skipif(not (HAS_QT and HAS_NUMPY), reason="Qt binding or numpy not installed")

if HAS_QT and HAS_NUMPY:
    import numpy as np
    from qtpy.QtGui import QImage

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.display import DisplayImage
    from am_segmentation_manager.gui.cache import FrameImage
    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel
    from am_segmentation_manager.gui.widgets.image_view import ImageView

#: Grey enough that no pixel of the picture can be taken for the bar.
PICTURE_GREY = 100


def _view(qt_app, size: tuple[int, int], field_width_um: float | None) -> ImageView:
    """Return a pane of *size* showing a 300 x 200 grey frame, fitted."""
    view = ImageView()
    view.resize(*size)
    view.show()
    qt_app.processEvents()
    view.set_array(np.full((200, 300), PICTURE_GREY, dtype=np.uint8))
    view.fit()
    view.set_field_width(field_width_um)
    qt_app.processEvents()
    return view


def _white(view: ImageView) -> list[tuple[int, int]]:
    """Return every pure white pixel in the pane, as (x, y)."""
    image = view.viewport().grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    ratio = image.devicePixelRatio()
    return [
        (int(x / ratio), int(y / ratio))
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) & 0xFFFFFF == 0xFFFFFF
    ]


def _picture(view: ImageView):
    """Return the picture's corners in the pane's pixels."""
    rect = view._item.sceneBoundingRect()
    return view.mapFromScene(rect.topLeft()), view.mapFromScene(rect.bottomRight())


@pytest.mark.parametrize(
    "size, where",
    [((400, 600), "below"), ((700, 300), "beside")],
    ids=["room-below", "room-beside-only"],
)
def test_the_scale_bar_sits_off_the_picture_by_its_bottom_right_corner(qt_app, size, where):
    """Under the corner where the pane has room, else beside it: never over the picture."""
    view = _view(qt_app, size, field_width_um=30.0)
    try:
        top_left, bottom_right = _picture(view)
        white = _white(view)

        assert white, "a frame with a scale is drawn with a bar"
        assert not [
            (x, y) for x, y in white
            if top_left.x() <= x <= bottom_right.x() and top_left.y() <= y <= bottom_right.y()
        ], "nothing of the bar is drawn over the picture"
        if where == "below":
            assert all(y > bottom_right.y() for _, y in white)
            assert max(x for x, _ in white) <= bottom_right.x()
        else:
            assert all(x > bottom_right.x() for x, _ in white)
            assert max(y for _, y in white) <= bottom_right.y()
    finally:
        destroy(view)


@pytest.mark.parametrize(
    "field_width_um, zoom", [(None, 1.0), (30.0, 6.0)], ids=["no-scale", "corner-off-the-pane"]
)
def test_no_scale_bar_is_drawn_without_a_scale_or_room_beside_the_picture(
    qt_app, field_width_um, zoom
):
    view = _view(qt_app, (400, 600), field_width_um)
    try:
        view.zoom_by(zoom)
        qt_app.processEvents()

        assert view.scale_bar_layout() is None
        assert _white(view) == []
    finally:
        destroy(view)


@pytest.mark.parametrize("field_width_um", [7.0, 30.0, 1800.0])
def test_fitted_every_frame_s_bar_is_about_a_fifth_of_its_width(qt_app, field_width_um):
    """Rounded to 1, 2 or 5, so between two fifths of the target and all of it."""
    view = _view(qt_app, (400, 600), field_width_um)
    try:
        layout = view.scale_bar_layout()
        shown = view._item.sceneBoundingRect().width() * view.transform().m11()

        assert 0.08 * shown <= layout.bar.width() <= 0.2 * shown + 1
    finally:
        destroy(view)


def test_zooming_in_chooses_a_shorter_bar(qt_app):
    view = _view(qt_app, (400, 600), field_width_um=30.0)
    try:
        fitted = view.scale_bar_layout().length_um
        view.zoom_by(1.25 ** 3)
        # Zoomed in about the middle the corner leaves the pane, and the bar
        # with it, so the corner is brought back to where the bar has room.
        view.centerOn(view._item.sceneBoundingRect().bottomRight())

        assert view.scale_bar_layout().length_um < fitted
    finally:
        destroy(view)


def test_clearing_a_pane_takes_its_scale_with_it(qt_app):
    view = _view(qt_app, (400, 600), field_width_um=30.0)
    try:
        view.clear()

        assert view.field_width_um is None
    finally:
        destroy(view)


def test_both_curate_panes_are_given_the_frame_s_scale(qt_app):
    panel = ImagePanel(SegmentationConfig(class_colors={0: (0, 0, 0)}, class_labels={0: "background"}))
    try:
        panel.show_image(
            FrameImage(DisplayImage(
                image=np.zeros((64, 96), dtype=np.uint8), field_width_um=40.0,
            )),
            show_overlay=True,
        )

        assert panel.pair.left.field_width_um == panel.pair.right.field_width_um == 40.0
    finally:
        destroy(panel)
