"""Smoke tests for the curation window.

These drive the real widgets against a small synthetic project on disk, using
Qt's offscreen platform so they need no display. They are skipped when Qt or the
optional image dependencies are not installed, which keeps a minimal
installation -- and CI without Qt -- able to run the rest of the suite.

pytest-qt is deliberately not used: a session-wide QApplication and a small
event-loop helper are all these tests need, and avoiding the dependency keeps
the test requirements the same as the package's.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import pytest

from conftest import class_config, destroy, on_every_beam_and_state, settle, widest_layout

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGE_STACK),
    reason="Qt binding or optional image libs not installed",
)

if HAS_QT and HAS_IMAGE_STACK:
    import numpy as np
    from qtpy.QtCore import QSettings, Qt
    from qtpy.QtWidgets import QApplication, QMessageBox, QToolButton

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.display import DisplayImage
    from am_segmentation_manager.gui.attribution import Attribution
    from am_segmentation_manager.gui.cache import FrameCache, FrameImage
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.label_controller import LabelController
    from am_segmentation_manager.gui.metadata_controller import MetadataController
    from am_segmentation_manager.gui.widgets.busy_dialog import BusyDialog
    from am_segmentation_manager.gui.widgets.controls import ChipEditor
    from am_segmentation_manager.gui.widgets.frame_strip import FrameTimeline
    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel
    from am_segmentation_manager.gui.widgets.image_view import MIN_FIT_FRACTION
    from am_segmentation_manager.gui.widgets.label_panel import LabelPanel
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow, _corner
    from am_segmentation_manager.gui.widgets.metadata_panel import (
        MetadataPanel,
        editor_text,
        set_editor_text,
    )
    from am_segmentation_manager.gui.widgets.operations_panel import OperationsPanel
    from am_segmentation_manager.gui.widgets.project_panel import ProjectPanel
    from am_segmentation_manager.gui.widgets.scan_bar import ScanBar
    from am_segmentation_manager.gui.widgets.state_panel import StatePanel
    from am_segmentation_manager.identity import PRODUCT_NAME
    from am_segmentation_manager.metadata import (
        DESCRIPTIVE_FIELDS,
        FIELDS,
        build_experiment_metadata,
    )
    from am_segmentation_manager.models import Project
    from am_segmentation_manager.progress import Cancellation

CONFIG = None


# ---------------------------------------------------------------------------
# One panel on its own
#
# For a test whose claim is about one panel. Each is built as the window builds
# it and freed with `destroy`; a panel with no parent is a top-level widget, and
# the suite fails a test that leaves one alive. Anything the claim depends on
# beyond the panel -- a controller, a scan, another panel -- stays on `window`.
# ---------------------------------------------------------------------------

@pytest.fixture
def image_panel(qt_app):
    """The curate tab's image panel on its own, with the synthetic project's classes."""
    panel = ImagePanel(class_config())
    yield panel
    destroy(panel)


def _first_frame() -> FrameImage:
    """The synthetic project's first frame, prepared as the cache hands it over."""
    label = np.zeros((64, 96), dtype=np.uint8)
    label[:20] = 2
    return FrameImage(
        display=DisplayImage(
            image=np.full((64, 96), 100, dtype=np.uint8),
            label=label,
            label_status="present",
        )
    )


@pytest.fixture
def image_pair(image_panel):
    """The image panel's pair of linked panes, showing the project's first frame."""
    image_panel.show_image(_first_frame(), show_overlay=True)
    return image_panel.pair


@pytest.fixture
def timeline(qt_app):
    """A frame timeline on its own, as the curate tab builds it."""
    widget = FrameTimeline()
    yield widget
    destroy(widget)


@pytest.fixture
def state_panel(qt_app):
    """The curate tab's state panel on its own, editing off as a window opens it."""
    panel = StatePanel()
    yield panel
    destroy(panel)
    # The window seats the guard in a row of its own, so alone it has no parent.
    destroy(panel.enable_check)


@pytest.fixture
def project_panel(qt_app):
    """The project and lamella lists on their own, listing the synthetic project."""
    panel = ProjectPanel()
    panel.show_projects(["AutoLamella-2026-07-28-13-57"], 0, checks=[False])
    panel.show_lamellae(["01-alpha", "02-beta"], 0, checks=[False, False])
    yield panel
    destroy(panel)


@pytest.fixture
def scan_bar(qt_app):
    """The folder box, Browse and Scan on their own, as the tab row holds them."""
    bar = ScanBar()
    yield bar
    destroy(bar)


@pytest.fixture
def corner_bar(qt_app):
    """The tab row's corner as the window builds it, with its buttons doing nothing."""
    corner = _corner(ScanBar(), lambda: None, lambda: None, lambda: None)
    yield corner
    destroy(corner)


@pytest.fixture
def label_panel(qt_app):
    """The Label tab on its own, over a controller that has read nothing."""
    panel = LabelPanel(LabelController(config=class_config()))
    yield panel
    destroy(panel)


@pytest.fixture
def operations_panel(qt_app):
    """The curate tab's batch operations on their own, as a window opens them."""
    panel = OperationsPanel()
    yield panel
    destroy(panel)


@pytest.fixture
def busy_dialog(qt_app):
    """The dialog of an operation that cannot be stopped, shown as the window shows it."""
    dialog = BusyDialog(None, "Long thing")
    dialog.show()
    yield dialog
    destroy(dialog)


# ---------------------------------------------------------------------------
# Resizable divisions
# ---------------------------------------------------------------------------

def test_every_division_that_moves_is_marked_as_one(window):
    """Which divisions can be dragged is found by eye, so one that looks like
    an ordinary line is one nobody discovers. Asserted over the same mapping
    the window saves its layout through, so a division added without a grip is
    a division added without a saved size as well."""
    from am_segmentation_manager.gui.widgets.controls import GripSplitter

    divisions = window._splitters()

    assert divisions, "the window has resizable divisions"
    assert all(
        isinstance(splitter, GripSplitter) for splitter in divisions.values()
    ), sorted(
        name for name, splitter in divisions.items()
        if not isinstance(splitter, GripSplitter)
    )


def test_the_train_tab_s_divisions_are_saved_as_every_other_tab_s_are(window):
    """Left out, they opened where the code put them however they were left."""
    divisions = window._splitters()

    assert divisions["train"] is window.training_panel.columns
    assert divisions["train_lists"] is window.training_panel.lists


# ---------------------------------------------------------------------------
# Fitting the screen
# ---------------------------------------------------------------------------

#: The smallest display the window is expected to open on, in logical pixels.
#: A 13" MacBook Air; the 14" MacBook Pro reports 1512x982 and every external
#: monitor here is larger. Not a guess at the smallest screen in the world --
#: it is the floor this package holds itself to, and a control row that cannot
#: wrap breaks it long before anything exotic does.
SMALLEST_SUPPORTED_SCREEN = (1440, 900)

#: The width a layout is held to here, which is less than the screen's where
#: this platform's fonts are the narrower; see ``conftest.widest_layout``.
WIDEST_LAYOUT = widest_layout(SMALLEST_SUPPORTED_SCREEN[0])


def test_no_tab_needs_more_room_than_the_smallest_screen_supported(window):
    """A window cannot be made narrower than its widest unwrappable row.

    Qt clamps a resize to the minimum, so a tab needing more width than the
    screen has does not scroll or wrap: the far edge sits off the display with
    no way to drag it back. That is not a layout blemish but an unreachable
    control, and it is invisible to anyone developing on a larger monitor --
    the Label tab's view controls needed 1560px, against the 1512 a 14"
    MacBook Pro has, and the right of every tab was cut off there.
    """
    wide, tall = WIDEST_LAYOUT, SMALLEST_SUPPORTED_SCREEN[1]
    too_big = {
        window.tabs.tabText(index): (
            window.tabs.widget(index).minimumSizeHint().width(),
            window.tabs.widget(index).minimumSizeHint().height(),
        )
        for index in range(window.tabs.count())
        if window.tabs.widget(index).minimumSizeHint().width() > wide
        or window.tabs.widget(index).minimumSizeHint().height() > tall
    }

    assert not too_big, f"needs more than {wide}x{tall}: {too_big}"
    assert window.minimumSizeHint().width() <= wide
    assert window.minimumSizeHint().height() <= tall


#: Set this to a directory to save a picture of every tab at each size in
#: :data:`CAPTURED_SIZES`, for a person to look at how the window lays out
#: under another platform's fonts, e.g.
#: ``AM_SEG_SCREENSHOTS=screenshots uv run pytest tests/test_gui_window.py -k captured``.
SCREENSHOTS_ENV_VAR = "AM_SEG_SCREENSHOTS"

#: The smallest screen the window must fit, and a common desktop one.
CAPTURED_SIZES = (SMALLEST_SUPPORTED_SCREEN, (1920, 1080))


needs_screenshots = pytest.mark.skipif(
    not os.environ.get(SCREENSHOTS_ENV_VAR), reason=f"{SCREENSHOTS_ENV_VAR} is not set"
)


def _capture(qt_app, window, size: tuple[int, int], label: str, tabs: list[int]) -> None:
    """Save *window* showing each of *tabs* at *size*, named for what was asked and kept."""
    import sys

    directory = Path(os.environ[SCREENSHOTS_ENV_VAR])
    directory.mkdir(parents=True, exist_ok=True)
    window.show()
    window.resize(*size)
    for index in tabs:
        window.tabs.setCurrentIndex(index)
        settle(qt_app, lambda: window.tabs.currentIndex() == index, what="the tab")
        for _ in range(5):
            qt_app.processEvents()
        name = window.tabs.tabText(index).replace(" ", "-").lower()
        shown = window.size()
        path = directory / (
            f"{sys.platform}-{size[0]}x{size[1]}-{label}{index + 1}-{name}"
            f"-at-{shown.width()}x{shown.height()}.png"
        )

        assert window.grab().save(str(path)), f"could not write {path}"


@needs_screenshots
@pytest.mark.parametrize("size", CAPTURED_SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
def test_every_tab_is_captured_for_a_person_to_look_at(window, qt_app, size):
    """Not a check: the layout tests say a row does not fit, and only a picture
    says whether that looks wrong. A window Qt will not shrink that far is
    captured at the width it keeps, which is itself the finding."""
    _capture(qt_app, window, size, "", list(range(window.tabs.count())))


@needs_screenshots
def test_the_curate_tab_before_a_scan_is_captured_for_a_person_to_look_at(qt_app, settings):
    """Built and measured as the view-controls test below measures it, so the
    picture shows what that test saw; the controls' height is in the name."""
    import sys

    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    directory = Path(os.environ[SCREENSHOTS_ENV_VAR])
    directory.mkdir(parents=True, exist_ok=True)
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.tabs.setCurrentWidget(window.curate_panel)
        for width in (SMALLEST_SUPPORTED_SCREEN[0], 1920):
            window.resize(width, 900)
            window.show()
            qt_app.processEvents()
            picture = window.grab()
            controls = window.curate_panel.image_panel.controls
            path = directory / (
                f"{sys.platform}-{width}x900-unscanned-2-curate"
                f"-controls-{controls.width()}x{controls.height()}.png"
            )

            assert picture.save(str(path)), f"could not write {path}"
    finally:
        destroy(window)


def test_the_window_is_drawn_in_letters_of_different_widths(qt_app):
    """Where the offscreen platform finds no fonts it draws every character as
    one box, and every layout test then measures boxes -- which is what
    Windows gave it until conftest named Windows' own fonts."""
    from qtpy.QtGui import QFontMetrics

    metrics = QFontMetrics(qt_app.font())

    assert metrics.horizontalAdvance("i") < metrics.horizontalAdvance("W")


def test_the_view_controls_wrap_rather_than_widening_the_window(label_panel):
    """The Label tab's controls are the row that broke the rule above.

    Asserted on the row rather than only through the window, so the reason a
    tab fits stays visible: it fits because this row wraps, and a control
    added to a plain box layout here would put it back over the edge.
    """
    from am_segmentation_manager.gui.widgets.controls import FlowArea, FlowLayout

    controls = label_panel.controls

    assert isinstance(controls, FlowArea)
    assert isinstance(controls.layout(), FlowLayout)
    assert controls.layout().minimumSize().width() == 0


def test_a_row_that_wraps_grows_when_a_control_in_it_appears(qt_app):
    """Showing a control resizes nothing, so a row measured only on a resize
    kept one line's height and laid the control that wrapped below its edge."""
    from qtpy.QtWidgets import QPushButton

    from am_segmentation_manager.gui.widgets.controls import FlowArea, FlowLayout

    area = FlowArea()
    row = FlowLayout(spacing=4)
    area.setLayout(row)
    first, second = QPushButton("First"), QPushButton("Second")
    row.addWidget(first)
    row.addWidget(second)
    second.hide()
    try:
        area.resize(first.sizeHint().width() + 10, 10)
        area.show()
        qt_app.processEvents()
        one_row = area.height()

        second.show()
        qt_app.processEvents()

        assert area.height() > one_row
        assert area.rect().contains(second.geometry())
    finally:
        destroy(area)


def test_a_long_caption_does_not_widen_the_curate_tab(image_panel):
    """The line under the frame names the project path and the image file,
    which on a grid overview runs past a screen's width. A line that cannot
    shorten is a floor under the tab, and the splitter takes that width from
    the lamella list; so the caption elides and the badges wrap."""
    panel = image_panel
    before = panel.minimumSizeHint().width()
    caption = (
        "001: " + "/".join(["AutoLamella-2026-04-08-10-01_virussample"] * 8)
        + " overview-image-2026-04-08_10-04-53-autogamma.tif"
    )

    panel.set_status(caption)
    panel._set_badges([("positions are approximate", "warning")] * 12)

    assert panel.minimumSizeHint().width() <= before
    assert panel.status.full_text() == caption
    assert panel.status.toolTip() == caption


def test_a_model_name_does_not_widen_the_curate_tab(state_panel):
    """A model's file name holds no space to wrap at, and the line repeating
    the last change was as wide as the name: a floor under the right panel,
    which the splitter then collapsed rather than squeeze."""
    panel = state_panel
    before = panel.minimumSizeHint().width()
    headline = (
        "Predict labels: Predicted 2 SEM image(s) with Franklin_eBIC_Rosalind_training_"
        "absolute_Franklin_eBIC_Rosalind_validation_absolute_epoch087.pth."
    )

    panel.show_last_change(headline)

    assert panel.minimumSizeHint().width() <= before
    assert panel.last_change.text().replace("\u200b", "") == headline

    panel.show_last_change("x" * 400)

    assert panel.minimumSizeHint().width() <= before


def test_a_chart_does_not_widen_the_export_tab(qt_app, tmp_path, settings):
    """A grid of long lamella names drawn as a chart fits where its table does.

    The chart's bar gains the shading control for a grid, and a bar that
    cannot wrap took the width from the Export column until its buttons read
    "Per-image metadata CSV.." with their first letters cut off, and then
    pushed the window past the screen.
    """
    from qtpy.QtWidgets import QPushButton

    from conftest import build_dataset

    dataset = build_dataset(
        tmp_path / "scan", name="AutoLamella-2024-07-05-12-53",
        lamellae=12, lamella_name="01-big-cicada",
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=CONFIG), settings=settings
    )
    try:
        on_every_beam_and_state(window.export_panel)
        window.resize(*SMALLEST_SUPPORTED_SCREEN)
        window.show()
        window.scan(str(dataset.root))
        panel = window.export_panel
        window.tabs.setCurrentWidget(panel)
        panel.breakdown.setCurrentIndex(panel.breakdown.findData("lamella"))
        panel.cross.setCurrentIndex(panel.cross.findData("state"))
        panel.as_chart.click()
        for _ in range(5):
            qt_app.processEvents()
        clipped = {
            button.text(): (button.width(), button.sizeHint().width())
            for button in panel._export_box.findChildren(QPushButton)
            if button.isVisible() and button.width() < button.sizeHint().width()
        }
        bar = panel.chart_view.bar
        cut_off = [
            button.text() for button in panel.chart_view._actions()
            if not bar.rect().contains(button.geometry())
        ]

        assert panel.chart_view._canvas is not None
        assert not panel.chart_view.shading.isHidden()
        assert window.minimumSizeHint().width() <= WIDEST_LAYOUT
        assert window.width() == SMALLEST_SUPPORTED_SCREEN[0]
        assert not clipped, clipped
        assert not cut_off, cut_off
    finally:
        destroy(window)


def test_a_window_larger_than_the_screen_is_pulled_onto_it(window):
    """Tested against a screen this machine does not have.

    The fault only appears on a display smaller than the one it is developed
    on, so asserting it through the real screen would assert nothing here.
    """
    from qtpy.QtCore import QRect

    window.resize(1600, 950)
    window._fit_within(QRect(0, 0, 1512, 982))

    assert window.width() <= 1512
    assert window.height() <= 982


def test_a_window_remembered_off_the_edge_is_moved_back_on(window):
    """A geometry saved on a second monitor is restored without one attached.

    Qt puts back what it was given; nothing in ``restoreGeometry`` asks
    whether the screen it was recorded on is the screen it is being put back
    on, and a window opening off the edge cannot be dragged back.
    """
    from qtpy.QtCore import QRect

    room = QRect(0, 0, 1512, 982)
    window.resize(1100, 800)
    window.move(2400, 200)

    window._fit_within(room)

    assert room.contains(window.frameGeometry())


# ---------------------------------------------------------------------------
# Linked zoom (regression: the panes stopped following each other once an
# image fitted its pane, because zoom rode on scrollbar signals)
# ---------------------------------------------------------------------------

def test_zoom_stays_linked_when_the_image_fits_its_pane(image_pair):
    """The case that used to desynchronise: no scrollbars, so no scroll events."""
    pair = image_pair
    pair.left.zoom_by(4.0)
    assert pair.right.transform().m11() == pytest.approx(pair.left.transform().m11())

    # Zoom back out past the point where the image fits.
    for _ in range(8):
        pair.left.zoom_by(0.5)

    assert pair.right.transform().m11() == pytest.approx(pair.left.transform().m11())


def test_zoom_out_stops_at_fit(image_pair):
    """Reviewing is done at fit, so there is nothing below it to see: shrinking
    the frame further only strands it in an empty pane."""
    view = image_pair.left
    view.fit()
    fitted = view.transform().m11()

    for _ in range(10):
        view.zoom_by(0.5)

    assert view.transform().m11() == pytest.approx(fitted, rel=0.001), (
        "zoomed all the way out and fitted have to be the same picture"
    )
    assert view.min_fit_fraction == MIN_FIT_FRACTION


def test_the_pane_can_be_scrolled_past_the_edges_of_the_frame(image_pair):
    view = image_pair.left
    image = view._item.boundingRect()
    scene = view.sceneRect()

    assert scene.left() < image.left() and scene.right() > image.right()
    assert scene.top() < image.top() and scene.bottom() > image.bottom(), (
        "an edge that cannot leave the border of the pane is one you paint blind"
    )


def test_the_panes_have_no_scrollbars(image_pair):
    view = image_pair.left

    assert view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_timeline_handles_an_empty_lamella(timeline):
    timeline.set_states({"SEM": [], "FIB": []}, 0)

    assert timeline._count == 0


# ---------------------------------------------------------------------------
# State panel
# ---------------------------------------------------------------------------

def test_state_options_are_in_workflow_order(state_panel):
    order = list(state_panel.state_buttons)

    assert order == ["deselected", "selected", "corrected"]


def test_state_options_carry_a_colour_swatch(state_panel):
    for button in state_panel.state_buttons.values():
        assert not button.icon().isNull()


def test_state_tooltips_name_their_shortcut(state_panel):
    assert "D" in state_panel.state_buttons["deselected"].toolTip()
    assert "C" in state_panel.state_buttons["corrected"].toolTip()


# ---------------------------------------------------------------------------
# Long lists
# ---------------------------------------------------------------------------

def test_filtering_hides_non_matching_lamellae(project_panel):
    project_panel.lamella_filter.setText("beta")

    hidden = [
        project_panel.lamella_list.item(row).isHidden()
        for row in range(project_panel.lamella_list.count())
    ]
    assert hidden == [True, False]


def test_clearing_the_filter_restores_everything(project_panel):
    project_panel.lamella_filter.setText("beta")
    project_panel.lamella_filter.setText("")

    hidden = [
        project_panel.lamella_list.item(row).isHidden()
        for row in range(project_panel.lamella_list.count())
    ]
    assert hidden == [False, False]


def test_list_titles_show_counts(project_panel):
    assert "(1)" in project_panel.project_box.title()
    assert "(2)" in project_panel.lamella_box.title()


# ---------------------------------------------------------------------------
# Status chips
#
# The colour is the information, so Qt must not be left to re-tint it.
# ---------------------------------------------------------------------------

def test_a_chip_is_the_same_pixmap_in_every_icon_mode(qt_app):
    """Left to itself, Qt generates a Selected variant blended with the
    highlight colour, which washes the chip out on the row the user is on."""
    from qtpy.QtGui import QIcon, QPixmap

    from am_segmentation_manager.gui.widgets.common import chip_icon

    pixmap = QPixmap(14, 10)
    pixmap.fill(Qt.red)
    icon = chip_icon(pixmap)

    normal = icon.pixmap(14, 10, QIcon.Normal).toImage()
    for mode in (QIcon.Selected, QIcon.Active, QIcon.Disabled):
        assert icon.pixmap(14, 10, mode).toImage() == normal


def _status(**cells):
    """Build a status from the cells named, the rest holding no images.

    Keys are ``series_MODALITY``: ``adaptive_SEM="done"``.
    """
    from am_segmentation_manager.gui.controller import CurationStatus
    from am_segmentation_manager.layout import MODALITIES, SERIES

    grid = {(s, m): "empty" for s in SERIES for m in MODALITIES}
    grid.update({tuple(key.rsplit("_", 1)): value for key, value in cells.items()})
    return CurationStatus(grid)


def _chip_image(**cells):
    """Render one status chip on its own, at its natural size."""
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import SWATCH_HEIGHT, SWATCH_WIDTH

    icon = _status_icon(_status(**cells))
    return icon.pixmap(SWATCH_WIDTH, SWATCH_HEIGHT).toImage()


def test_a_chip_has_a_column_per_series_it_is_given(qt_app):
    """A series switched off takes no room, rather than an empty column."""
    from am_segmentation_manager.gui.widgets.project_panel import _status_icon
    from am_segmentation_manager.gui.widgets.theme import SWATCH_WIDTH, swatch_width
    from am_segmentation_manager.layout import SERIES

    icon = _status_icon(_status(), ("adaptive", "reference"))

    assert swatch_width(len(SERIES)) == SWATCH_WIDTH
    assert icon.availableSizes()[0].width() == swatch_width(2)


def _cell_centre(column: int, row: int):
    """Return a point inside the cell at *column* and *row* of a chip."""
    from am_segmentation_manager.gui.widgets.theme import SWATCH_CELL, SWATCH_GAP

    step, middle = SWATCH_CELL + SWATCH_GAP, SWATCH_CELL // 2
    return column * step + middle, row * step + middle


def _luminance(colour) -> float:
    """Return the relative luminance of *colour*, as the eye weights it."""
    def channel(value: int) -> float:
        value /= 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(colour.red())
        + 0.7152 * channel(colour.green())
        + 0.0722 * channel(colour.blue())
    )


def test_the_two_modalities_are_stacked_not_side_by_side(qt_app):
    """SEM above FIB, the same arrangement as the timeline's two rows, so a
    beam is found in the same place in both."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS

    image = _chip_image(adaptive_SEM="done", adaptive_FIB="not started")

    assert image.pixelColor(*_cell_centre(0, 0)) == CELL_COLOURS["done"]
    assert image.pixelColor(*_cell_centre(0, 1)) == CELL_COLOURS["not started"]


def test_a_part_finished_modality_gets_a_colour_rather_than_a_split(qt_app):
    """A split showed two pixels of each colour once the outline had taken the
    rest, and the two were the same brightness. The cell is one flat colour."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS
    from am_segmentation_manager.gui.widgets.theme import SWATCH_CELL

    image = _chip_image(adaptive_SEM="in progress")
    row = SWATCH_CELL // 2
    across = {image.pixelColor(x, row).name() for x in range(SWATCH_CELL)}

    assert across == {CELL_COLOURS["in progress"].name()}


def test_the_progress_steps_are_told_apart_by_brightness(qt_app):
    """Hue is the channel the eye cannot use in a cell this small, so the three
    steps have to separate in luminance whatever their colour."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS

    ramp = [
        _luminance(CELL_COLOURS[state])
        for state in ("not started", "in progress", "done")
    ]

    assert ramp == sorted(ramp, reverse=True)
    assert all(
        (before + 0.05) / (after + 0.05) >= 1.5
        for before, after in zip(ramp, ramp[1:])
    )


def test_a_cell_with_no_images_is_the_palest_thing_on_the_chip(qt_app):
    """Absent has to read as absent, not as a state: many older datasets have
    no FIB whatsoever, and a quarter of the archive no Adaptive Polishing. It is
    drawn rather than left hollow, so it does not depend on what is behind."""
    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS

    image = _chip_image(adaptive_SEM="done")
    inside = image.pixelColor(*_cell_centre(0, 1))

    assert inside == CELL_COLOURS["empty"]
    assert inside.alpha() == 255
    assert _luminance(inside) > max(
        _luminance(colour)
        for state, colour in CELL_COLOURS.items()
        if state != "empty"
    )


def test_the_chip_is_an_even_number_of_pixels_tall(qt_app):
    """Qt centres a row's check indicator on the row but rounds the decoration
    differently, so an odd-height icon lands half a pixel below the tick beside
    it. The chip was 13 and sat visibly low."""
    from am_segmentation_manager.gui.widgets.theme import SWATCH_HEIGHT

    assert SWATCH_HEIGHT % 2 == 0


def test_a_chip_keeps_its_colour_on_the_row_being_looked_at(qt_app):
    """The regression: a 'done' teal came out blue-green on the current row,
    against a blue-tinted background -- so a finished lamella looked unfinished
    exactly when you selected it to check."""
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QListWidget

    from am_segmentation_manager.gui.widgets.project_panel import (
        CELL_COLOURS,
        _status_icon,
    )
    from am_segmentation_manager.gui.widgets.theme import apply_theme

    stylesheet, palette = qt_app.styleSheet(), qt_app.palette()
    apply_theme(qt_app)
    try:
        done = _status(adaptive_SEM="done", adaptive_FIB="done")
        widget = QListWidget()
        widget.addItems(["current", "other"])
        for row in range(2):
            widget.item(row).setIcon(_status_icon(done))
        widget.setCurrentRow(0)
        widget.resize(300, 60)
        image = widget.grab().toImage()

        teal = CELL_COLOURS["done"].name()
        found = [
            any(
                QColor(image.pixel(x, y)).name() == teal
                for y in range(4 + row * 22, 24 + row * 22)
                for x in range(4, 40)
            )
            for row in range(2)
        ]
        assert found == [True, True]
    finally:
        qt_app.setStyleSheet(stylesheet)
        qt_app.setPalette(palette)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def test_tabs_are_in_workflow_order(window):
    titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]

    assert titles == ["Add data", "Curate", "Label", "Export", "Train"]


def test_curate_is_the_tab_that_opens(window):
    """Left-most is the workflow's first step; open on the one used most."""
    assert window.tabs.currentIndex() == _curate_tab(window)
    assert window._current_tab() == "curate"


# ``focusWidget`` rather than ``hasFocus``: these windows are never shown, and
# Qt defers real focus until a window is activated. What is asserted is the
# widget that will hold the keyboard, which is the same either way.

def test_curate_opens_with_the_keyboard_on_the_project_list(window):
    """Not on the scan box: a focused QLineEdit claims Left and Right for its
    own cursor -- so the keys that step through frames never reach the window
    until something is clicked."""
    assert window.focusWidget() is window.curate_panel.project_panel.project_list


def test_returning_to_curate_takes_the_keyboard_back(window):
    window.tabs.setCurrentWidget(window.metadata_panel)
    window.tabs.setCurrentIndex(_curate_tab(window))

    assert window.focusWidget() is window.curate_panel.project_panel.project_list


def test_the_scan_box_is_what_would_have_swallowed_the_frame_keys(window):
    """The reason the fix is a focus change rather than a rebinding. Asserted
    on the widgets themselves so a Qt version that changed its mind about which
    keys a line edit claims is reported here."""
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QKeyEvent

    def claims(widget, key) -> bool:
        event = QKeyEvent(QEvent.ShortcutOverride, key, Qt.NoModifier)
        event.ignore()
        QApplication.sendEvent(widget, event)
        return event.isAccepted()

    assert claims(window.scan_bar.path_edit, Qt.Key_Left)
    assert not claims(window.curate_panel.project_panel.project_list, Qt.Key_Left)
    assert not claims(window.curate_panel.project_panel.project_list, Qt.Key_Down)


def test_the_folder_is_chosen_in_the_tab_row_on_every_tab(window):
    """A scan serves all four tabs, and only Curate could show which folder it
    had read."""
    bar = window.scan_bar

    assert window.tabs.cornerWidget(Qt.TopRightCorner).isAncestorOf(bar)
    assert not window.curate_panel.project_panel.isAncestorOf(bar)
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        assert not bar.isHidden(), window.tabs.tabText(index)


def test_the_corner_reads_from_the_folder_to_the_mark(window):
    """The folder nearest the tabs it serves, this machine's state beside it,
    and whose window this is at the far end."""
    corner = window._corner
    order = [
        corner.layout().indexOf(widget)
        for widget in (window.scan_bar, corner.author, corner.mark)
    ]

    assert window.tabs.cornerWidget(Qt.TopLeftCorner) is None
    assert -1 not in order and order == sorted(order)


def test_the_scan_box_is_never_reached_by_tab(scan_bar):
    """Tabbed into, it would take Left and Right from the frame keys."""
    assert scan_bar.path_edit.focusPolicy() == Qt.ClickFocus


def test_return_and_scan_both_ask_for_the_typed_folder(qt_app):
    from am_segmentation_manager.gui.widgets.scan_bar import ScanBar

    bar = ScanBar()
    asked: list[str] = []
    bar.scan_requested.connect(asked.append)
    bar.path_edit.setText("/some/folder")

    bar.path_edit.returnPressed.emit()
    bar.scan_button.click()

    assert asked == ["/some/folder", "/some/folder"]
    destroy(bar)


def test_the_scan_box_says_what_the_folder_held(window, project):
    tip = window.scan_bar.path_edit.toolTip()

    assert window.scan_bar.path_edit.text() == str(project)
    assert "1 project" in tip
    assert "2 lamellae" in tip


def test_opening_export_reads_the_headers_and_not_the_labels(window, monkeypatch):
    """Half a millisecond an image, against ten times that per label. The
    headers are what the acquisition filters are made of, so they are worth
    paying for on arrival; the labels answer a different question and stay
    behind the button that asks it."""
    submitted: list[str] = []
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: (
            submitted.append(description), on_done(work())
        ),
    )

    window.tabs.setCurrentWidget(window.export_panel)

    assert submitted and "header" in submitted[0]
    assert window.export_controller.metadata_loaded()
    assert not window.export_controller.measurements_loaded(), (
        "opening the tab must not open a single label"
    )


def test_inspect_layout_runs_with_a_stop_button(window, monkeypatch):
    started: list[tuple[str, object]] = []
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: started.append(
            (description, cancellation)
        ),
    )

    window.metadata_panel.survey_requested.emit()

    assert [description for description, _ in started] == ["Inspect layout"]
    assert started[0][1] is not None


def test_choosing_a_label_filter_reads_the_labels_behind_the_dialog(window, monkeypatch):
    """Over a real collection the labels take the better part of a minute, and
    read inside the redraw they froze the window with nothing saying why."""
    submitted: list[str] = []
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: (
            submitted.append(description), on_done(work())
        ),
    )
    window.tabs.setCurrentWidget(window.export_panel)
    submitted.clear()

    window.export_panel.label_state.setCurrentText("has a label")
    window.export_panel._controls_changed()

    assert submitted and submitted[0].startswith("Reading")
    assert window.export_controller.labels_pending() == 0
    assert "Reading" not in window.export_panel.table.text()


def test_returning_to_export_does_not_read_the_headers_again(window, monkeypatch):
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    window.tabs.setCurrentWidget(window.export_panel)
    assert window.export_controller.metadata_loaded()

    monkeypatch.setattr(
        window.runner, "run",
        lambda *args, **kwargs: pytest.fail("the headers were already read"),
    )
    window.tabs.setCurrentWidget(window.curate_panel)
    window.tabs.setCurrentWidget(window.export_panel)


def test_the_scan_says_what_it_is_doing_before_it_blocks(window, project: Path):
    """A wait cursor does not say whether a window is working or stuck, and the
    scan is what stops anything else being painted."""
    announced: list[str] = []
    original = window.curate_panel.project_panel.announce
    window.curate_panel.project_panel.announce = lambda text: (
        announced.append(text), original(text)
    )
    try:
        window.scan(str(project))
    finally:
        del window.curate_panel.project_panel.announce

    assert announced and "Scanning" in announced[0]
    assert project.name in announced[0]


def test_the_add_data_tab_opens_before_anything_is_scanned(qt_app, settings):
    """Reachable from a fresh window, which has no projects to describe yet."""
    controller = CurationController(cache=FrameCache(), config=SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    ))
    fresh = CurationWindow(controller, settings=settings)
    try:
        fresh.tabs.setCurrentWidget(fresh.metadata_panel)

        assert fresh.metadata_panel.fields_box.title() == "Metadata"
    finally:
        destroy(fresh)


# ---------------------------------------------------------------------------
# Remembering the layout
# ---------------------------------------------------------------------------

def test_the_splitter_positions_survive_closing_the_window(qt_app, project, settings):
    """Dragging a divider is a preference, and losing it every session is what
    makes a window feel like it is not listening."""
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    )
    first = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    first.curate_panel.columns.setSizes([400, 800, 400])
    sizes = first.curate_panel.columns.sizes()
    first.close()
    destroy(first)

    second = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert second.curate_panel.columns.sizes() == sizes
    finally:
        destroy(second)


def test_the_window_remembers_under_the_keys_it_always_has(window, settings, tmp_path):
    """A renamed key forgets, without a word, what every user's window held:
    its size and dividers, the folder, the model and device, where runs are
    kept. So the keys are pinned exactly, whichever code comes to write them."""
    weights = tmp_path / "base.pth"
    weights.write_bytes(b"")
    window.training_controller.training_directory = tmp_path
    window.training_controller.base_weights = weights
    window.training_panel.remembered.emit()
    window._remember_device()

    window.close()
    settings.sync()

    assert sorted(settings.allKeys()) == [
        "geometry",
        "model/adaptive/SEM",
        "predict_device",
        "scan_root",
        "splitter/add_data",
        "splitter/curate",
        "splitter/export",
        "splitter/export_filters",
        "splitter/lists",
        "splitter/train",
        "splitter/train_lists",
        "training/device",
        "training/directory",
        "training/weights",
    ]


def test_a_window_with_no_saved_layout_uses_its_defaults(qt_app, settings):
    """A first run has nothing to restore, which must not leave a collapsed
    pane behind."""
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    )
    fresh = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert all(size > 0 for size in fresh.curate_panel.columns.sizes())
    finally:
        destroy(fresh)


#: Longest tooltip worth reading in a hover, as the Export tab already holds
#: itself to. Past this it is documentation, and a tooltip nobody finishes
#: explains nothing.
MAX_TOOLTIP = 200


def _tooltips_too_long(window) -> set[str]:
    from qtpy.QtWidgets import QWidget

    return {
        widget.toolTip()
        for widget in window.findChildren(QWidget)
        if len(widget.toolTip()) > MAX_TOOLTIP
    }


def test_no_tooltip_in_the_window_is_longer_than_anyone_reads(window):
    """The ceiling is stated in gui/widgets/__init__.py. Enforced on one tab
    only, the timeline grew a 488-character legend nobody was reading."""
    assert not _tooltips_too_long(window)


def test_no_tooltip_is_longer_where_neither_stack_is_installed(
    qt_app, project: Path, settings: QSettings, monkeypatch
):
    """What a greyed control says of a missing extra is read only where it is missing.

    An environment with both installed never draws those, and one of them
    passed the ceiling there to fail in every environment without ``train``.
    """
    from am_segmentation_manager.ml import environment

    monkeypatch.setattr(environment, "ml_stack_present", lambda: False)
    monkeypatch.setattr(environment, "training_stack_present", lambda: False)
    window = CurationWindow(CurationController(cache=FrameCache(), config=CONFIG), settings=settings)
    try:
        window.scan(str(project))

        assert not window.training_controller.can_train()
        assert "not installed" in window.curate_panel.operations_panel.predict_button.toolTip()
        assert not _tooltips_too_long(window)
    finally:
        destroy(window)


def _wheel_turned_over(widget):
    """Send *widget* one turn of the wheel, and return the event as it came back."""
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication

    where = QPointF(widget.rect().center())
    turn = QWheelEvent(
        where, widget.mapToGlobal(where), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, turn)
    return turn


def test_the_wheel_changes_no_control_anywhere_in_the_window(window):
    """It moves the page, and changed whichever setting the pointer was passing over."""
    from qtpy.QtWidgets import QAbstractSpinBox, QComboBox, QSlider

    def held(control) -> object:
        if isinstance(control, QComboBox):
            return control.currentIndex()
        return control.text() if isinstance(control, QAbstractSpinBox) else control.value()

    controls = [
        control
        for kind in (QAbstractSpinBox, QComboBox, QSlider)
        for control in window.findChildren(kind)
    ]
    assert len(controls) > 10, "every tab's, not one's"
    for control in controls:
        before = held(control)
        control.setFocus()

        assert not _wheel_turned_over(control).isAccepted(), control
        assert held(control) == before, control


def test_the_wheel_still_moves_a_scroll_bar(window):
    """A scroll bar is a slider to Qt, and is what the wheel is for."""
    from qtpy.QtWidgets import QScrollBar

    from am_segmentation_manager.gui.widgets.controls import _WheelPassedOn

    bar = window.findChildren(QScrollBar)[0]

    assert not _WheelPassedOn().eventFilter(bar, _wheel_turned_over(bar))


def test_the_wheel_is_left_to_scrolling_again_where_building_a_window_fails(qt_app):
    """The filter is off while a window is assembled, and a window that fails
    to build must not leave it off for the windows already open."""
    from qtpy.QtWidgets import QSpinBox

    from am_segmentation_manager.gui.widgets.controls import (
        wheel_left_to_scrolling_once_built,
    )

    box = QSpinBox()
    try:
        with pytest.raises(RuntimeError):
            with wheel_left_to_scrolling_once_built(qt_app):
                assert _wheel_turned_over(box).isAccepted(), "off while building"
                raise RuntimeError("the window could not be built")

        box.setValue(0)
        assert not _wheel_turned_over(box).isAccepted()
        assert box.value() == 0
    finally:
        destroy(box)


# ---------------------------------------------------------------------------
# Shortcut reference
# ---------------------------------------------------------------------------

def test_every_shortcut_is_documented(window):
    assert len(window._keys.bindings) == len(window._keys.shortcuts)
    assert all(binding.description for binding in window._keys.bindings)
    assert any(binding.key == "F1" for binding in window._keys.bindings)


def test_a_key_is_live_only_on_the_tab_it_belongs_to(window):
    """Curate keys firing behind another tab is what stopped the Metadata list
    ever seeing an arrow key."""
    window.tabs.setCurrentWidget(window.metadata_panel)

    live = {b.key for b, s in zip(window._keys.bindings, window._keys.shortcuts) if s.isEnabled()}

    assert live == {"Up", "Down", "F1"}


def test_every_curate_key_runs_what_it_names(window, monkeypatch):
    """A binding is a lambda naming a method, so one left naming a method that
    moved fails only when somebody presses the key. Editing is off, so the
    state keys are refused rather than writing anything."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    for name in ("information", "warning", "question", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Cancel)
    keys = [binding for binding in window._keys.bindings if binding.tab == TAB_CURATE]
    assert keys

    for binding in keys:
        binding.action()


def test_curate_keys_come_back_with_the_curate_tab(window):
    window.tabs.setCurrentWidget(window.metadata_panel)
    window.tabs.setCurrentIndex(_curate_tab(window))

    live = {b.key for b, s in zip(window._keys.bindings, window._keys.shortcuts) if s.isEnabled()}

    assert {"Left", "Right", "V", "D", "F1"} <= live


def _curate_tab(window) -> int:
    """Return the curate tab's index, whatever order the tabs are in."""
    others = {window.metadata_panel, window.export_panel}
    return next(
        index for index in range(window.tabs.count())
        if window.tabs.widget(index) not in others
    )


def test_f1_puts_the_list_away_again(window):
    window.show_shortcuts()
    assert window._keys.help.isVisible()
    # is_in_front rather than the real thing: an offscreen window is never
    # activated, so nothing here is ever the window in front.
    window._keys.help.is_in_front = lambda: True

    window.show_shortcuts()

    assert not window._keys.help.isVisible(), "the key that opened it closes it"


def test_f1_fetches_back_a_list_left_behind_the_window(window):
    """It closed on visibility alone, and a list buried behind the window is
    visible -- so F1 dismissed, unseen, the one thing that answers "what can I
    press?", and pressing it again just put it back. That is most of what made
    F1 look as though it worked only sometimes."""
    window.show_shortcuts()
    window._keys.help.is_in_front = lambda: False

    window.show_shortcuts()

    assert window._keys.help.isVisible()


def test_f1_is_ignored_while_an_operation_holds_the_window(window, monkeypatch):
    """The key reaches the window from anywhere now, and a list opened over an
    application-modal dialog can be neither read nor closed until it ends."""
    monkeypatch.setattr(
        QApplication, "activeModalWidget", staticmethod(lambda: window)
    )

    window.show_shortcuts()

    assert window._keys.help is None


def test_f1_works_from_the_list_it_opened(window):
    """A window-scoped shortcut is dead while a second top-level window holds
    the keyboard, and the list is one -- so the key that opens it could not
    close it again."""
    from qtpy.QtCore import Qt

    f1 = next(
        shortcut for binding, shortcut in zip(window._keys.bindings, window._keys.shortcuts)
        if binding.key == "F1"
    )

    assert f1.context() == Qt.ApplicationShortcut


def test_the_window_asks_to_be_the_one_in_front(window):
    """Raised and activated, not just raised: a window on top but without the
    keyboard is one macOS sends neither key presses nor pointer motion to, so
    F1, every tooltip and the brush outline are all dead until it is clicked.
    Asserted on the calls, an offscreen window never being activated."""
    asked: list[str] = []
    window.raise_ = lambda: asked.append("raise")
    window.activateWindow = lambda: asked.append("activate")

    window.bring_to_front()

    assert asked == ["raise", "activate"]


def test_losing_the_keyboard_is_recorded(window, caplog):
    """Whether the window was in front is not recoverable after the fact, and
    it is the first thing to establish about a report that the keyboard or the
    pointer stopped working."""
    from qtpy.QtCore import QEvent

    with caplog.at_level(logging.DEBUG, logger="am_segmentation_manager.gui.widgets"):
        window.changeEvent(QEvent(QEvent.ActivationChange))

    assert "deactivated" in caplog.text


def test_the_shortcut_list_does_not_block_the_window(window):
    window.show_shortcuts()

    assert not window._keys.help.isModal(), (
        "a list you must dismiss before trying what it says is read once"
    )


def test_the_shortcut_list_follows_the_tab(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_LABEL

    window.show_shortcuts()
    window.tabs.setCurrentWidget(window.label_panel)

    assert window._keys.help._text.text() == window._keys.help_html(TAB_LABEL)


def test_the_help_lists_the_menu_keys_every_tab_shares(window):
    """F1 is the one place that answers "what can I press?", and the File menu's
    keys -- rescanning among them -- were in the menu and nowhere else."""
    from am_segmentation_manager.gui.widgets.shortcuts import (
        TAB_ADD_DATA,
        TAB_CURATE,
        TAB_EXPORT,
        TAB_LABEL,
        TAB_TRAIN,
    )

    for tab in (TAB_ADD_DATA, TAB_CURATE, TAB_LABEL, TAB_EXPORT, TAB_TRAIN):
        listed = window._keys.help_html(tab)
        assert "The File menu" in listed, tab
        assert "Rescan" in listed, tab


def test_the_menu_keys_are_read_off_the_menu_rather_than_copied(window):
    """A key spelled here would be a second table to keep in step, and each
    platform spells its own -- Ctrl+O where macOS says Cmd+O."""
    from qtpy.QtGui import QKeySequence

    from am_segmentation_manager.gui.widgets.shortcuts import menu_rows

    rows = dict(menu_rows(window))
    rescan = window.menu_actions()["Rescan"]

    assert rescan.shortcut().toString(QKeySequence.NativeText) in rows
    assert rows[rescan.shortcut().toString(QKeySequence.NativeText)] == "Rescan"
    assert all(key for key in rows), "an action with no key teaches nothing"


def test_the_help_lists_only_the_active_tab(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_ADD_DATA, TAB_CURATE

    curate = window._keys.help_html(TAB_CURATE)
    metadata = window._keys.help_html(TAB_ADD_DATA)

    assert "Next frame" in curate and "Next project" not in curate
    assert "Next project" in metadata and "Next frame" not in metadata
    # The editing guard only governs the curate tab's keys.
    assert "Enable editing" in curate and "Enable editing" not in metadata


def test_the_help_describes_the_mouse_gestures_too(window):
    """Picking frames out has no key to press, so listing only keys left the
    whole selection mechanism undiscoverable."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    help_text = window._keys.help_html(TAB_CURATE)

    assert "Ctrl-click" in help_text
    assert "Shift-click" in help_text
    assert "Tick a project or lamella" in help_text


def test_every_gesture_is_described():
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_GESTURES

    for gestures in TAB_GESTURES.values():
        assert all(
            gesture.group and gesture.action and gesture.description
            for gesture in gestures
        )


def _headings() -> set[str]:
    """The headings F1 is allowed to file a row under, named one by one.

    Deliberately not derived from the module: a heading is added by writing it
    here as well, which is the moment to ask whether the list needs another
    one or the row belongs under a heading that already exists.
    """
    from am_segmentation_manager.gui.widgets import shortcuts as shortcuts_module

    return {
        shortcuts_module.GROUP_VIEW,
        shortcuts_module.GROUP_CHOOSE,
        shortcuts_module.GROUP_STATE,
        shortcuts_module.GROUP_TOOLS,
        shortcuts_module.GROUP_CLASSES,
        shortcuts_module.GROUP_SPECKS,
        shortcuts_module.GROUP_LABELLING_GUIDE,
        shortcuts_module.GROUP_EDITS,
        shortcuts_module.GROUP_ELSEWHERE,
        shortcuts_module.GROUP_HELP,
    }


def test_every_key_and_gesture_is_filed_under_a_known_heading(window):
    """A row with nowhere to go is one the reader has to place themselves."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_GESTURES

    headings = _headings()
    assert {binding.group for binding in window._keys.bindings} <= headings
    for gestures in TAB_GESTURES.values():
        assert {gesture.group for gesture in gestures} <= headings


def test_the_key_that_hides_a_label_hides_the_classes_a_model_predicted(window):
    """V on Curate and Label; on Train the classes drawn are a prediction's."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_TRAIN

    hide = {b.tab: b.action for b in window._keys.bindings if b.key == "V"}
    predictions = window.training_panel.predictions

    hide[TAB_TRAIN]()

    assert not predictions.label_check.isChecked()
    assert "(V)" in predictions.label_check.toolTip()


def test_a_number_shows_the_predicted_image_of_that_row_alone(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_TRAIN

    keys = {b.key: b.action for b in window._keys.bindings if b.tab == TAB_TRAIN}

    assert {"1", "2", "3", "4"} <= set(keys)
    keys["2"]()  # nothing drawn: nothing to show, and no error


def test_the_pictures_controls_stand_beside_the_view_switch_and_only_with_the_pictures(window, qt_app):
    panel = window.training_panel
    panel.view_control.buttons["predictions"].click()
    qt_app.processEvents()
    controls, switch = panel.predictions.controls, panel.view_control

    assert not controls.isHidden()
    assert controls.mapTo(panel, controls.rect().topLeft()).y() == pytest.approx(
        switch.mapTo(panel, switch.rect().topLeft()).y(), abs=6
    )

    panel.view_control.buttons["curves"].click()
    qt_app.processEvents()
    assert controls.isHidden()


def test_the_key_that_fits_a_view_resets_a_chart_s(window):
    """G on Curate and Label; a chart's whole view is what Reset view gives."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_EXPORT, TAB_TRAIN

    reset = {b.tab: b.action for b in window._keys.bindings if b.key == "G"}

    assert reset[TAB_EXPORT] == window.export_panel.chart_view.reset_view
    assert reset[TAB_TRAIN] == window.training_panel.chart.reset_view
    assert "Shortcut: G" in window.training_panel.chart.reset_button.toolTip()
    reset[TAB_TRAIN]()  # nothing drawn: nothing to do, and no error


def test_both_tabs_that_draw_a_chart_say_how_to_move_about_it(window):
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_EXPORT, TAB_TRAIN

    for tab in (TAB_EXPORT, TAB_TRAIN):
        help_text = window._keys.help_html(tab)
        assert "Scroll wheel over a chart" in help_text and "Hold Space and drag a chart" in help_text


def test_the_three_ways_to_resize_the_brush_are_listed_together(window):
    """[ and ] and Shift with the wheel all do one thing, so they read as one
    group -- and the mouse rows still run together, this one leading them."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_LABEL

    help_text = window._keys.help_html(TAB_LABEL)

    assert (
        help_text.index("Larger brush")
        < help_text.index("Shift + scroll wheel")
        < help_text.index("Drag with the left button")
        < help_text.index("Click with the fill tool")
    )


def test_a_tooltip_names_only_a_key_its_tab_binds(window):
    """A tooltip's key is written by hand beside the control, apart from the
    binding, and Label's Fit taught one that nothing was bound to."""
    import re

    from qtpy.QtWidgets import QWidget

    from am_segmentation_manager.gui.widgets.shortcuts import ANY_TAB, TAB_CURATE, TAB_LABEL

    unbound = []
    for panel, tab in ((window.curate_panel.image_panel, TAB_CURATE), (window.label_panel, TAB_LABEL)):
        bound = {b.key for b in window._keys.bindings if b.tab in (tab, ANY_TAB)}
        for widget in (panel, *panel.findChildren(QWidget)):
            named = re.search(r"Shortcut: ([^\s;]+)", widget.toolTip())
            if named and named.group(1) not in bound:
                unbound.append((tab, widget.toolTip()))

    assert not unbound, unbound


def test_a_heading_is_not_drawn_like_the_rows_under_it(window):
    """Every left-hand cell is bold, being the column a reader scans, so bold
    alone left a heading looking like one more key."""
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_TOOLS, TAB_LABEL
    from am_segmentation_manager.gui.widgets.theme import HEADING

    help_text = window._keys.help_html(TAB_LABEL)

    heading = help_text[: help_text.index(GROUP_TOOLS)].rsplit("<p", 1)[-1]
    assert HEADING in heading, "a heading takes the group-title colour"
    assert "font-size" in heading and "border-bottom" in heading
    assert f"<b>{GROUP_TOOLS}</b>" not in help_text, (
        "a heading has to differ from a key, not merely repeat its weight"
    )


def test_the_curate_keys_are_grouped_by_what_they_are_for(window):
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_CHOOSE,
        GROUP_STATE,
        GROUP_VIEW,
        TAB_CURATE,
    )

    help_text = window._keys.help_html(TAB_CURATE)

    for heading in (GROUP_VIEW, GROUP_CHOOSE, GROUP_STATE):
        assert heading in help_text
    # In the order the work is done: look, choose, then change.
    assert (
        help_text.index(GROUP_VIEW)
        < help_text.index(GROUP_CHOOSE)
        < help_text.index(GROUP_STATE)
    )


def test_a_gesture_sits_under_the_same_heading_as_the_keys_it_belongs_with(window):
    """Picking frames out is one idea, done with Esc or with Ctrl-click.

    Listing the keys and then the gestures separately made the reader do the
    grouping the list is for.
    """
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_CHOOSE, GROUP_STATE, TAB_CURATE

    help_text = window._keys.help_html(TAB_CURATE)

    choose = help_text.index(GROUP_CHOOSE)
    state = help_text.index(GROUP_STATE)
    assert choose < help_text.index("Esc") < state
    assert choose < help_text.index("Ctrl-click a frame") < state


def test_the_key_that_opens_the_list_is_listed_last(window):
    from am_segmentation_manager.gui.widgets.shortcuts import GROUP_HELP, TAB_CURATE, TAB_LABEL

    for tab in (TAB_CURATE, TAB_LABEL):
        help_text = window._keys.help_html(tab)
        heading = help_text.index(GROUP_HELP)
        assert all(
            heading > help_text.index(other)
            for other in _headings()
            if other != GROUP_HELP and other in help_text
        ), tab


#: A tab nothing is bound on. Every tab the window has now holds a key, and one
#: added without any still has to say so.
KEYLESS_TAB = "a tab with no keys"


def test_a_tab_with_no_keys_of_its_own_says_so(window):
    help_text = window._keys.help_html(KEYLESS_TAB)

    assert "no keys of its own" in help_text
    assert "F1" in help_text


def test_the_project_arrows_are_scoped_to_the_project_list(window):
    """Window-wide, they would take Up/Down from the editable combo boxes on
    the same tab."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_ADD_DATA

    scoped = [b for b in window._keys.bindings if b.tab == TAB_ADD_DATA]

    assert {b.key for b in scoped} == {"Up", "Down"}
    assert all(b.scope is window.metadata_panel.projects for b in scoped)


# ---------------------------------------------------------------------------
# Zoom (regression: zooming in was refused whenever the current scale sat
# below the fit scale, which it does after every fit)
# ---------------------------------------------------------------------------

def test_zoom_in_works_from_the_fitted_scale(image_pair):
    view = image_pair.left
    view.fit()
    before = view.transform().m11()

    view.zoom_by(1.25)

    assert view.transform().m11() > before


def test_zoom_in_works_even_when_well_below_the_fit_scale(image_pair):
    """The fit limit constrains zooming out; it must not block zooming in."""
    from qtpy.QtGui import QTransform

    view = image_pair.left
    fit = view.fit_scale()
    view.setTransform(QTransform.fromScale(fit * 0.5, fit * 0.5))

    view.zoom_by(1.25)

    assert view.transform().m11() > fit * 0.5


def test_repeated_zoom_in_keeps_working(image_pair):
    view = image_pair.left
    scales = []
    for _ in range(4):
        view.zoom_by(1.25)
        scales.append(view.transform().m11())

    assert scales == sorted(scales)
    assert scales[-1] > scales[0]


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------

def test_an_export_leaving_out_a_corrected_image_names_it_and_cancel_writes_nothing(
    qt_app, window, monkeypatch, tmp_path
):
    """A corrected image with nothing to pair it with is proofreading gone
    missing, and a training set short of it has to be chosen, not found."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["FIB"].setChecked(True)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning", lambda *args, **kwargs: QMessageBox.Yes
    )
    window.curate_panel.apply_state("corrected")
    assert window.controller.state_of("FIB") == "corrected"
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda *args, **kwargs: asked.append((args[1], args[2])) or QMessageBox.Cancel,
    )
    monkeypatch.setattr(
        window.export_controller, "export_training",
        lambda *args, **kwargs: pytest.fail("wrote a training set the user cancelled"),
    )

    window._export_training(str(tmp_path))
    settle(qt_app, lambda: bool(asked), what="the export to ask about the corrected image")

    [(title, text)] = asked
    assert title == "Export without corrected images?"
    assert "01-alpha" in text and "_FIB.tif" in text
    assert list(tmp_path.glob("*.csv")) == []


def test_cancelling_the_export_warning_writes_nothing(window, monkeypatch, tmp_path):
    """The gate has to stop the write, not merely appear before it."""
    from am_segmentation_manager.models import ImageRecord

    monkeypatch.setattr(
        window.export_controller, "selected_records",
        lambda *a, **k: [ImageRecord(image_path=tmp_path / "img.tif")],
    )
    monkeypatch.setattr(window, "_confirm_dropping_corrected", lambda: False)
    monkeypatch.setattr(
        window.export_controller, "export_training",
        lambda *a, **k: pytest.fail("wrote a training set the user cancelled"),
    )

    window._export_training(str(tmp_path))

    assert list(tmp_path.glob("*.csv")) == []


def test_an_earlier_training_set_is_replaced_only_when_asked(
    window, inline_operations, monkeypatch, tmp_path
):
    """Cancel leaves the earlier files; Yes is the only way to overwrite=True."""
    from qtpy.QtWidgets import QMessageBox

    asked_to_overwrite = []

    def export(output, *, overwrite=False, on_progress=None):
        asked_to_overwrite.append(overwrite)
        if not overwrite:
            raise FileExistsError("dataset_train.csv already in here; nothing was written")
        return [(output / "dataset_train.csv", None)]

    monkeypatch.setattr(window, "_confirm_dropping_corrected", lambda: True)
    monkeypatch.setattr(window.export_controller, "export_training", export)

    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning", lambda *a, **k: QMessageBox.Cancel
    )
    window._write_training(str(tmp_path))
    assert asked_to_overwrite == [False]

    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning", lambda *a, **k: QMessageBox.Yes
    )
    window._write_training(str(tmp_path))
    assert asked_to_overwrite == [False, False, True]


@pytest.mark.parametrize(
    ("action", "export", "description"),
    [
        ("_write_training", "export_training", "Writing the training files"),
        ("_export_image_metadata", "export_image_metadata", "Writing the image metadata"),
    ],
)
def test_an_export_reading_headers_is_written_behind_the_busy_dialog(
    window, monkeypatch, tmp_path, action: str, export: str, description: str
):
    """Either can open a header per image, which on the interface thread is a
    window that looks hung."""
    from am_segmentation_manager.models import ImageRecord

    ran = []
    monkeypatch.setattr(
        window.export_controller, "selected_records",
        lambda *a, **k: [ImageRecord(image_path=tmp_path / "img.tif")],
    )
    monkeypatch.setattr(window, "_confirm_dropping_corrected", lambda: True)
    monkeypatch.setattr(
        window.runner, "run", lambda description, work, done, **_: ran.append(description)
    )
    monkeypatch.setattr(
        window.export_controller, export,
        lambda *a, **k: pytest.fail("wrote on the interface thread"),
    )

    getattr(window, action)(str(tmp_path / "out.csv"))

    assert ran == [description]


def test_an_image_metadata_export_that_fails_says_why(
    window, inline_operations, monkeypatch, tmp_path
):
    from am_segmentation_manager.models import ImageRecord

    def export(output, *, overwrite=False, on_progress=None):
        raise ValueError("the metadata file of 01-alpha could not be read")

    shown = []
    monkeypatch.setattr(
        window.export_controller, "selected_records",
        lambda *a, **k: [ImageRecord(image_path=tmp_path / "img.tif")],
    )
    monkeypatch.setattr(window.export_controller, "export_image_metadata", export)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning", lambda _parent, title, text: shown.append(title)
    )

    window._export_image_metadata(str(tmp_path / "images.csv"))

    assert shown == ["Export failed"]


def test_an_earlier_exports_leftover_folds_are_not_offered_for_replacing(
    window, inline_operations, monkeypatch, tmp_path
):
    """Replacing would leave them where they are, so the only answer is to say so."""
    from am_segmentation_manager.operations.training_export import LeftoverFolds

    def export(output, *, overwrite=False, on_progress=None):
        raise LeftoverFolds("dataset_fold4_train.csv in here belong to an earlier export")

    shown = []
    monkeypatch.setattr(window, "_confirm_dropping_corrected", lambda: True)
    monkeypatch.setattr(window.export_controller, "export_training", export)
    monkeypatch.setattr(
        window, "_confirm_overwriting", lambda *a: pytest.fail("offered to replace them")
    )
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning", lambda _parent, title, text: shown.append(title)
    )

    window._write_training(str(tmp_path))

    assert shown == ["Export failed"]


def _tick_lamella(window, row: int = 0) -> None:
    """Tick a lamella through the widget, as a user would."""
    window.curate_panel.project_panel.lamella_list.item(row).setCheckState(Qt.Checked)


# ---------------------------------------------------------------------------
# Operations run against the same selection a state change does
# ---------------------------------------------------------------------------

@pytest.fixture
def inline_operations(window, monkeypatch):
    """Run operations on the calling thread and accept every dialog.

    ``OperationRunner.run`` is the one seam between the window and its worker
    thread; substituting it leaves the whole preview-confirm-apply ladder
    running for real, which is where the behaviour worth testing is.
    """
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    monkeypatch.setattr(window.runner, "show_report", lambda title, report: None)
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window.runner, "confirm",
        lambda title, report, action: asked.append((title, action)) or True,
    )
    return asked


def test_copying_the_selection_writes_it_into_the_chosen_folder(
    window, inline_operations, tmp_path: Path
):
    """The Export tab's copy: the filters decide what, the dialog decides
    where, and the same preview-then-apply ladder decides whether."""
    destination = tmp_path / "deposit"

    window._copy_files(str(destination))

    assert inline_operations == [("Export images and labels", "Apply")]
    copied = sorted(path.name for path in destination.rglob("*.tif"))
    assert copied, "the images and their labels land under the folder"
    assert (destination / "AutoLamella-2026-07-28-13-57" / "experiment.yaml").exists(), (
        "the file that makes the copy indexable comes too"
    )


def test_looking_at_a_collection_does_not_record_anything(window):
    """The rule the whole opt-in property now rests on: only a write starts a
    log, so opening and reading a collection leaves it exactly as it was."""
    from am_segmentation_manager.history import history_file_path

    root = Path(window.controller.start_path)
    window.rescan()

    assert not any(
        history_file_path(project.path).is_file()
        for project in window.controller.projects
    ), f"a scan of {root} wrote a log"


def test_a_copy_records_what_it_was_narrowed_to(
    window, inline_operations, tmp_path: Path
):
    """In the tab's own words: a copy made here was narrowed by these controls,
    and writing it as command-line flags would describe a run nobody made."""
    from am_segmentation_manager.operations.export_record import read_export_record

    destination = tmp_path / "subset"

    window._copy_files(str(destination))

    record = read_export_record(destination)
    assert record is not None
    assert record.scope == window.export_controller.describe_filters()


def test_merging_a_folder_that_is_not_a_copy_is_refused(
    window, inline_operations, tmp_path: Path
):
    """Refused rather than half-read: the wrong ancestor discards work while
    reporting success."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    reports: list = []
    window.runner.show_report = lambda title, report: reports.append(report)

    window._merge_copy(str(stranger))

    assert inline_operations == [], "nothing was offered to apply"
    assert reports and "fork point" in reports[0].headline


def test_copying_previews_before_it_writes(window, inline_operations, monkeypatch):
    """Refusing the preview must leave the folder untouched, gigabytes being
    the usual size of this one."""
    monkeypatch.setattr(window.runner, "confirm", lambda *args: False)
    destination = Path(window.controller.start_path).parent / "refused"

    window._copy_files(str(destination))

    assert not destination.exists()


# ---------------------------------------------------------------------------
# Reverting, which acts on the selection like everything else in that panel
# ---------------------------------------------------------------------------

def revert_with(window, undoing="conversion", savepoint="", metadata=False):
    """Answer the revert dialog without opening it."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import operation_actions as module

    class _Answered:
        def __init__(self, *_args):
            self.undoing, self.savepoint, self.metadata = undoing, savepoint, metadata

        def exec_(self):
            return QDialog.Accepted

    original = module.RevertDialog
    module.RevertDialog = _Answered
    try:
        window._revert()
    finally:
        module.RevertDialog = original


def test_reverting_proofread_work_is_refused_until_it_is_named(
    window, inline_operations, monkeypatch
):
    """CLAUDE.md's rule: no automated operation replaces a corrected label
    without explicit confirmation, and the window must not make that a
    checkbox somebody clicks past."""
    from am_segmentation_manager.gui.report import OperationReport

    window.curate_panel.state_panel.enable_check.setChecked(True)
    asked: list = []
    monkeypatch.setattr(
        window.history_controller, "revert_labels",
        lambda records, **kwargs: (
            asked.append(kwargs.get("allow_corrected"))
            or OperationReport(
                "2 label(s) are corrected.", ("a", "b"), 2, blocked="corrected"
            )
        ),
    )
    monkeypatch.setattr(window.runner, "confirm", lambda *args: False)

    revert_with(window)

    assert asked == [False], "the first run asked without permission, and stopped"


def test_restoring_metadata_with_a_savepoint_reaches_the_controller(
    window, inline_operations, monkeypatch
):
    from am_segmentation_manager.gui.report import OperationReport

    window.curate_panel.state_panel.enable_check.setChecked(True)
    asked: list = []
    monkeypatch.setattr(
        window.history_controller, "revert_labels",
        lambda records, **kwargs: asked.append(kwargs) or OperationReport("Nothing to restore."),
    )

    revert_with(window, savepoint="round 1", metadata=True)

    assert asked and asked[0]["savepoint"] == "round 1" and asked[0]["metadata"] is True


def test_the_metadata_box_counts_only_with_a_savepoint_chosen(qt_app):
    from am_segmentation_manager.gui.widgets.revert_dialog import RevertDialog

    class _Point:
        id, title, reachable, targets = "abc", "round 1", 1, 1

    dialog = RevertDialog(None, "2 image(s)", [_Point()])
    dialog.metadata_check.setChecked(True)

    assert not dialog.metadata_check.isEnabled() and not dialog.metadata
    dialog.savepoint_button.setChecked(True)
    assert dialog.metadata_check.isEnabled() and dialog.metadata
    destroy(dialog)


# ---------------------------------------------------------------------------
# Savepoints, which are collection-wide and so live beside the other two
# ---------------------------------------------------------------------------

def take_savepoint(window, title="round 1", note="", gap=False):
    """Answer the savepoint dialog without opening it."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import operation_actions as module

    class _Answered:
        def __init__(self, *_args):
            self.title, self.description, self.closing_gap = title, note, gap

        def exec_(self):
            return QDialog.Accepted

    original = module.SavepointDialog
    module.SavepointDialog = _Answered
    try:
        window._take_savepoint()
    finally:
        module.SavepointDialog = original


def test_restoring_proofread_work_is_refused_until_it_is_named(window, monkeypatch):
    """A restore reaches every label under the folder, the corrected ones among
    them, so it asks for them apart as a revert of the selection does."""
    from am_segmentation_manager.gui.report import OperationReport

    window.metadata_panel.enable.setChecked(True)
    monkeypatch.setattr(
        window.history_controller, "savepoints_here", lambda projects: ["round 1"]
    )
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    asked: list = []
    monkeypatch.setattr(
        window.history_controller, "revert_labels",
        lambda records, **kwargs: (
            asked.append(kwargs.get("allow_corrected"))
            or OperationReport(
                "2 label(s) are corrected.", ("a", "b"), 2, blocked="corrected"
            )
        ),
    )
    monkeypatch.setattr(window.runner, "confirm", lambda *args: False)

    restore_savepoint(window)

    assert asked == [False], "the first run asked without permission, and stopped"


def test_a_savepoint_names_every_project_in_the_scan(window, inline_operations):
    """A round is over the dataset a model was trained on, so there is nothing
    for a savepoint to narrow to and no selection is consulted."""
    from am_segmentation_manager.operations.savepoint import list_savepoints

    window.metadata_panel.enable.setChecked(True)
    take_savepoint(window, title="round 1", note="as proofread in August")

    assert inline_operations == [("Savepoint", "Apply")]
    points = list_savepoints(window.controller.projects)
    assert [point.title for point in points] == ["round 1"]
    assert points[0].found == len(window.controller.projects)


def restore_savepoint(window, savepoint="round 1"):
    """Answer the restore chooser without opening it."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import operation_actions as module

    class _Answered:
        def __init__(self, *_args, **_kwargs):
            self.undoing, self.savepoint, self.metadata = "", savepoint, False

        def exec_(self):
            return QDialog.Accepted

    original = module.RevertDialog
    module.RevertDialog = _Answered
    try:
        window._restore_savepoint()
    finally:
        module.RevertDialog = original


def test_a_savepoint_can_be_put_back_over_the_whole_collection(
    window, inline_operations
):
    """It is taken over the whole collection from this tab, so putting the
    whole collection back belongs here too -- going to Curate and ticking
    everything first was the asymmetry this closes."""
    import numpy as np
    import tifffile

    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.label_save import (
        LabelToSave,
        save_labels,
    )

    window.metadata_panel.enable.setChecked(True)
    take_savepoint(window, title="round 1")
    inline_operations.clear()

    # Through the ordinary save, so what it replaces is archived -- a label
    # changed any other way has nothing kept to go back to.
    label = next(
        label
        for project in window.controller.projects
        for record in project.image_records
        for label in record.labels
    )
    before = tifffile.imread(str(label))
    save_labels(
        [
            LabelToSave(
                destination=label,
                label=np.zeros_like(before),
                fingerprint=fingerprint_of(label),
            )
        ],
        allow_corrected=True,
        recorder=Recorder.for_run(),
    )

    restore_savepoint(window, "round 1")

    assert inline_operations, "the restore was never previewed"
    assert inline_operations[-1][0] == "Restore savepoint"
    assert np.array_equal(tifffile.imread(str(label)), before), "it did not go back"


def test_restoring_says_so_when_no_savepoint_has_been_taken(window, monkeypatch):
    """Rather than opening a chooser with nothing in it."""
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )
    window.metadata_panel.enable.setChecked(True)

    restore_savepoint(window)

    assert shown == ["Restore savepoint"]


def test_the_archive_is_not_measured_while_nothing_keeps_versions(
    window, monkeypatch
):
    """One stat per project rather than a walk of each: the ordinary case is a
    collection that has never replaced a label, and it must cost nothing."""
    walked: list = []
    monkeypatch.setattr(
        window.history_controller, "measure_archive",
        lambda projects: walked.append(projects),
    )
    # A measurement is handed to a worker, which has not run by the time this
    # asserts: what was handed over is what can be seen from here.
    started: list[str] = []
    monkeypatch.setattr(
        window.runner, "run", lambda description, *args, **kwargs: started.append(description)
    )

    window.tabs.setCurrentWidget(window.metadata_panel)

    assert started == [], "opening the tab started a measurement"
    assert walked == [], "opening the tab walked the archive"
    assert window.metadata_panel.archive.text() == ""


def test_compressing_acts_on_the_whole_collection(
    window, inline_operations, monkeypatch
):
    """Recompression preserves the pixels exactly, so there is no label it
    should be withheld from and nothing to select.

    The fixture's second lamella is never selected, and its labels are exactly
    the ones a per-selection button would have left uncompressed."""
    window.metadata_panel.enable.setChecked(True)
    seen: list = []
    real = window.operations_controller.compress_labels
    monkeypatch.setattr(
        window.operations_controller, "compress_labels",
        lambda records, **kwargs: seen.append(list(records)) or real(records, **kwargs),
    )

    window._compress_labels()

    assert inline_operations == [("Compress labels", "Apply")]
    assert "Compress labels" in window.curate_panel.state_panel.last_change.text()
    assert seen, "the operation never ran"
    assert {image.lamella for image in seen[0]} == {"01-alpha", "02-beta"}


def test_the_predict_button_is_withheld_where_inference_is_not_installed(operations_panel):
    """An optional extra, so the window must say so on the button rather than
    at the end of a dialog opened expecting it to work."""
    operations_panel.set_prediction_available(False)

    assert not operations_panel.predict_button.isEnabled()
    assert "--extra ml" in operations_panel.predict_button.toolTip()


def test_browsing_for_a_model_predicts_nothing_by_itself(operations_panel):
    """Choosing a model fills the box. Nothing runs until Predict is pressed,
    which is what lets the window check that editing is on before a file is
    ever asked for."""
    started: list[str] = []
    operations_panel.predict_requested.connect(started.append)

    operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    assert operations_panel.model() == "/models/AM_SEM_All_V01.pth"
    assert started == []


# ---------------------------------------------------------------------------
# The model box follows the view
# ---------------------------------------------------------------------------
#
# An SEM model over a FIB image returns a confident, entirely wrong prediction,
# and the series differ from each other as much again. Which model is right
# stays the user's to say; what these pin is that the box is not offering the
# answer to a different question.

def test_the_model_box_is_kept_per_series_and_beam(window, settings):
    window.curate_panel.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    window.curate_panel.set_mode("fib", from_user=True)

    assert window.curate_panel.operations_panel.model() == ""
    assert settings.value("model/adaptive/SEM", "", type=str) == (
        "/models/AM_SEM_All_V01.pth"
    )


def test_the_model_comes_back_when_the_view_does(window):
    window.curate_panel.operations_panel.set_model("/models/AM_SEM_All_V01.pth")
    window.curate_panel.set_mode("fib", from_user=True)
    window.curate_panel.operations_panel.set_model("/models/AM_FIB_All_V02.pth")

    window.curate_panel.set_mode("sem", from_user=True)
    assert window.curate_panel.operations_panel.model() == "/models/AM_SEM_All_V01.pth"

    window.curate_panel.set_mode("fib", from_user=True)
    assert window.curate_panel.operations_panel.model() == "/models/AM_FIB_All_V02.pth"


def test_the_plot_view_leaves_the_model_box_alone(window):
    """It names no beam, so there is no slot to put a typed path into."""
    window.curate_panel.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    window.curate_panel.set_mode("plot", from_user=True)

    assert window.curate_panel.operations_panel.model() == "/models/AM_SEM_All_V01.pth"


def test_a_slot_never_filled_opens_on_the_one_path_kept_before(qt_app, settings):
    """An existing window opens on the model it always did, rather than on an
    empty box because the preference gained a shape."""
    settings.setValue("model_directory", "/models/AM_SEM_All_V01.pth")
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert window.curate_panel.operations_panel.model() == "/models/AM_SEM_All_V01.pth"
    finally:
        destroy(window)


def test_closing_keeps_what_was_typed_into_the_box(window, settings):
    """Predicting is not the only way a path gets chosen, and losing one that
    was browsed for means browsing for it again next session."""
    window.curate_panel.operations_panel.set_model("/models/AM_SEM_All_V01.pth")

    window.close()

    assert settings.value("model/adaptive/SEM", "", type=str) == (
        "/models/AM_SEM_All_V01.pth"
    )


def test_a_selection_leaving_everything_out_says_why(window, monkeypatch):
    """"Nothing selected" over ticked lamellae reads as the ticks not working."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    note = "1 ticked lamella holds no Adaptive Polishing images and is left out"
    monkeypatch.setattr(window.controller, "scoped_records", lambda: [])
    monkeypatch.setattr(window.controller, "left_out_note", lambda: note)
    flashed: list[str] = []
    monkeypatch.setattr(window, "_flash", flashed.append)

    window._predict("model.pth")

    assert flashed == [note]


def test_a_preview_opens_with_what_the_selection_leaves_out(window, monkeypatch):
    from am_segmentation_manager.gui.report import OperationReport

    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    previewed: list[OperationReport] = []
    monkeypatch.setattr(
        window.runner, "confirm",
        lambda title, report, action: previewed.append(report) and False,
    )

    window.runner.operate(
        "Predict labels",
        lambda allow, dry: OperationReport("Would predict 1.", ("a.tif: write",), 1),
        notes=("1 ticked lamella is left out.",),
    )

    assert previewed[0].lines == ("1 ticked lamella is left out.", "a.tif: write")


#: Each action on the selection, called as its button calls it.
ACTS_ON_THE_SELECTION = {
    "predict": lambda window: window._predict("model.pth"),
    "convert": lambda window: window._convert_predictions(),
    "revert": lambda window: revert_with(window),
}


@pytest.mark.parametrize(
    "act", ACTS_ON_THE_SELECTION.values(), ids=ACTS_ON_THE_SELECTION
)
def test_an_action_on_the_selection_previews_with_what_it_leaves_out(
    window, monkeypatch, act
):
    """Each hands the note over itself, so each could stop saying that a
    ticked lamella is not among what it is about to change."""
    window.curate_panel.state_panel.enable_check.setChecked(True)
    note = ("1 ticked lamella is left out.",)
    monkeypatch.setattr(window, "_scope_notes", lambda: note)
    handed: dict = {}
    monkeypatch.setattr(
        window.runner, "operate", lambda title, work, **kwargs: handed.update(kwargs)
    )

    act(window)

    assert handed["notes"] == note


def test_pressing_predict_with_an_empty_box_asks_for_a_model_last(window, monkeypatch):
    window.curate_panel.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems", lambda: []
    )
    flashed: list[str] = []
    monkeypatch.setattr(window, "_flash", flashed.append)

    window._predict("")

    assert flashed == ["Choose a model to predict with"]


def test_pressing_predict_does_not_import_torch_into_this_process(
    window, monkeypatch, inline_operations, tmp_path: Path
):
    """The window predicts in a child precisely so that torch is never loaded
    here -- and asking whether torch *would* load is asking by loading it. The
    cheap probe that greys out the button locates the modules instead."""
    from am_segmentation_manager.gui import operations_controller
    from am_segmentation_manager.operations.predict_process import ProcessOutcome

    window.curate_panel.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems",
        lambda: pytest.fail("the window must not import torch to check"),
    )
    monkeypatch.setattr(
        operations_controller, "predict_out_of_process",
        lambda *args, **kwargs: ProcessOutcome(written=0, total=0),
    )
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"x")

    window._predict(str(weights))


def test_a_missing_inference_stack_comes_back_from_the_child(
    window, monkeypatch, inline_operations, tmp_path: Path
):
    """It is the process that has to import torch that finds out, so the news
    arrives as a report rather than as a check this process could make."""
    from am_segmentation_manager.gui import operations_controller
    from am_segmentation_manager.operations.predict_process import ProcessOutcome

    window.curate_panel.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(
        operations_controller, "predict_out_of_process",
        lambda *args, **kwargs: ProcessOutcome(
            error="RuntimeError: Loading a model needs torch: not installed"
        ),
    )
    reported: list[str] = []
    monkeypatch.setattr(
        window.runner, "show_report",
        lambda title, report: reported.append(report.headline),
    )
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"x")

    window._predict(str(weights))

    assert reported and "needs torch" in reported[0]


def test_only_the_applying_run_is_given_something_to_stop(window, monkeypatch):
    """A preview opens no file and answers in milliseconds, so a Stop button on
    one is a button with nothing to stop."""
    from am_segmentation_manager.gui.report import OperationReport

    given: list[object] = []
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: (
            given.append(cancellation) or on_done(work())
        ),
    )
    monkeypatch.setattr(window.runner, "show_report", lambda title, report: None)
    monkeypatch.setattr(window.runner, "confirm", lambda title, report, action: True)
    monkeypatch.setattr(window.runner, "applied", lambda *a, **k: None)
    cancellation = Cancellation()

    window.runner.operate(
        "Predict labels",
        lambda allow, dry: OperationReport("preview" if dry else "applied", count=1),
        cancellation=cancellation,
    )

    assert given == [None, cancellation]


def test_a_copy_can_be_stopped_by_the_dialog_that_shows_it(window, monkeypatch, tmp_path):
    """Gigabytes go by in a copy, and Stop sets what the dialog was given: the
    copy has to be polling that one, on the run that writes and not the preview."""
    handed: dict = {}
    monkeypatch.setattr(
        window.runner, "operate",
        lambda title, work, **kwargs: handed.update(kwargs, work=work),
    )
    polled: list[object] = []
    monkeypatch.setattr(
        window.operations_controller, "copy_files",
        lambda *args, **kwargs: polled.append(kwargs["cancellation"]),
    )

    window._copy_files(str(tmp_path / "laptop"))
    handed["work"](False, True)
    handed["work"](False, False)

    assert isinstance(handed["cancellation"], Cancellation)
    assert polled == [None, handed["cancellation"]]


def _accept_dialog(monkeypatch, chosen, *, accepted=True, opened=None):
    """Stand in for the Add projects dialog, which is driven in its own tests."""

    class FakeDialog:
        def __init__(self, parent, controller, root):
            if opened is not None:
                opened.append(root)

        def exec_(self):
            return accepted

        def chosen(self):
            return list(chosen)

    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.operation_actions.AddProjectsDialog", FakeDialog
    )


def _unlabelled_record(window):
    """Return a record the scan found no label for, as a conversion's target."""
    return next(
        record
        for project in window.controller.projects
        for record in project.image_records
        if not record.labels
    )


def test_converting_does_not_walk_the_collection_again(
    project: Path, window, inline_operations, monkeypatch
):
    """The scan that used to follow cost a full read of the archive, which on a
    real collection is most of the time the operation takes. What it went
    looking for is the label the run just wrote, and the run says where."""
    from am_segmentation_manager.gui.report import OperationReport
    from am_segmentation_manager.operations.label_io import label_destination

    record = _unlabelled_record(window)
    written = label_destination(record)
    monkeypatch.setattr(
        window.operations_controller, "convert_predictions",
        lambda records, dry_run=False, **_: OperationReport(
            "Wrote 1 label(s) from predictions.", (), 1,
            applied=not dry_run,
            written=() if dry_run else (written,),
        ),
    )
    monkeypatch.setattr(
        window, "rescan", lambda: pytest.fail("converting should not rescan")
    )
    window.curate_panel.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)

    window._convert_predictions()

    assert record.labels == [written]


def test_predicting_does_not_walk_the_collection_again(
    project: Path, window, inline_operations, monkeypatch
):
    """Preparing labels is predict then convert against the same selection, so
    the scan was paid twice for one decision. Convert reads the predictions off
    the records, so this is also what makes the second step see the first."""
    from am_segmentation_manager.gui.report import OperationReport
    from am_segmentation_manager.operations.predict import prediction_destination

    record = _unlabelled_record(window)
    written = prediction_destination(record)
    given: list[list] = []

    def predict(records, *args, **kwargs):
        given.append(records)
        return OperationReport(
            "Predicted 1 SEM image(s).", (), 1,
            applied=not kwargs.get("dry_run"),
            written=() if kwargs.get("dry_run") else (written,),
        )

    monkeypatch.setattr(window.operations_controller, "predict", predict)
    monkeypatch.setattr(
        window, "rescan", lambda: pytest.fail("predicting should not rescan")
    )
    window.curate_panel.state_panel.enable_check.setChecked(True)
    _tick_lamella(window)
    selected = window.controller.scoped_records()

    window._predict(str(project / "model.pth"))

    assert given == [selected, selected], "the preview and the run, of the selection"
    assert record.predictions == [written]


def test_creating_projects_rescans_so_the_new_projects_appear(
    project: Path, window, inline_operations, monkeypatch
):
    """Making an unindexed dataset into a project is the point of this.

    A refresh recounts what is already in memory; only a scan can see the
    project the operation just created, and without one the Add data list keeps
    showing the collection as it was before.
    """
    unindexed = project / "loose-dataset" / "01-lamella" / "stage" / "sem"
    unindexed.mkdir(parents=True)
    (unindexed / "img.tif").write_bytes(b"i")
    window.metadata_panel.enable.setChecked(True)
    assert len(window.controller.projects) == 1

    _accept_dialog(monkeypatch, [project / "loose-dataset"])
    window._add_projects()

    assert (project / "loose-dataset" / "AM_metadata.yaml").exists()
    assert len(window.controller.projects) == 2
    assert len(window.metadata_panel.projects.project_items()) == 2


def test_cancelling_the_dialog_writes_nothing(project: Path, window, monkeypatch):
    """The dialog is where the decision is made, so dismissing it is the whole
    answer -- nothing behind it should have started."""
    unindexed = project / "loose-dataset" / "01-lamella" / "stage" / "sem"
    unindexed.mkdir(parents=True)
    (unindexed / "img.tif").write_bytes(b"i")
    window.metadata_panel.enable.setChecked(True)

    _accept_dialog(monkeypatch, [project / "loose-dataset"], accepted=False)
    # The write runs on a worker, which has not started by the time this
    # asserts, so the file alone would be absent either way.
    monkeypatch.setattr(
        window.runner, "run",
        lambda *args, **kwargs: pytest.fail("started writing after the dialog was dismissed"),
    )
    window._add_projects()

    assert not (project / "loose-dataset" / "AM_metadata.yaml").exists()


#: Each action that writes, called as its button calls it. Six guard themselves
#: in their own first lines and three through ``_editable_records``, so each
#: could lose its guard alone.
EDITING_GUARDED = {
    "predict": lambda window, tmp_path: window._predict(""),
    "convert": lambda window, tmp_path: window._convert_predictions(),
    "revert": lambda window, tmp_path: revert_with(window),
    "merge": lambda window, tmp_path: window._merge_copy(str(tmp_path / "copy")),
    "savepoint": lambda window, tmp_path: take_savepoint(window),
    "restore": lambda window, tmp_path: restore_savepoint(window),
    "reclaim": lambda window, tmp_path: window._reclaim_archive(),
    "compress": lambda window, tmp_path: window._compress_labels(),
    "add_projects": lambda window, tmp_path: window._add_projects(),
}


@pytest.mark.parametrize("act", EDITING_GUARDED.values(), ids=EDITING_GUARDED)
def test_an_action_that_writes_waits_on_the_editing_guard(
    window, monkeypatch, tmp_path: Path, act
):
    """Said before anything is asked for, and nothing is started: a model
    chosen or a dialog answered for an operation that was never going to run
    is the complaint this order exists to prevent."""
    shown: list[str] = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[1]),
    )
    flashed: list[str] = []
    monkeypatch.setattr(window, "_flash", lambda message, *a, **k: flashed.append(message))
    monkeypatch.setattr(
        window.runner, "run", lambda *args, **kwargs: pytest.fail("should not have started")
    )
    opened: list = []
    _accept_dialog(monkeypatch, [], opened=opened)

    act(window, tmp_path)

    assert shown == ["Editing is off"]
    assert flashed == []
    assert opened == []


def test_corrected_labels_are_confirmed_separately_before_being_touched(
    qt_app, window, monkeypatch
):
    """Two confirmations, in order of what is at stake.

    The same ladder apply_state uses: first that this is more than the frame on
    screen, then that proofread work is at risk. Recompressing rewrites the
    file, so a corrected label must not be reached by one Apply.
    """
    window.curate_panel.state_panel.enable_check.setChecked(True)
    window.curate_panel.state_panel.target_buttons["both"].setChecked(True)
    window.metadata_panel.enable.setChecked(True)
    _tick_lamella(window)
    monkeypatch.setattr(window.curate_panel, "_confirm_bulk", lambda state: True)
    # The fixture's FIB images have no labels, which this ladder is not about.
    monkeypatch.setattr(window.curate_panel, "_confirm_unannotated", lambda: True)
    window.curate_panel.apply_state("corrected")
    settle(qt_app, lambda: window.controller.frame_states()[0] == "corrected")

    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: on_done(work()),
    )
    monkeypatch.setattr(window.runner, "show_report", lambda title, report: None)
    asked: list[str] = []

    def refuse(title, report, action):
        asked.append(action)
        return False

    monkeypatch.setattr(window.runner, "confirm", refuse)
    window._compress_labels()
    assert asked == ["Include them"]

    # Confirming the corrected question re-previews and then asks to apply.
    asked.clear()
    monkeypatch.setattr(
        window.runner, "confirm",
        lambda title, report, action: asked.append(action) or True,
    )
    window._compress_labels()

    assert asked == ["Include them", "Apply"]


def test_finding_specks_needs_no_editing_guard(window, inline_operations, monkeypatch):
    """It reads and writes nothing, and asking someone to switch editing on to
    look at their own data would be a lock on the wrong door."""
    shown = []
    monkeypatch.setattr(window.runner, "show_report", lambda title, report: shown.append(title))
    _tick_lamella(window)
    assert not window.curate_panel.state_panel.enable_check.isChecked()

    window._find_specks(64)

    assert shown == ["Find specks"]


def test_finding_specks_needs_something_ticked(window, monkeypatch):
    """Unticked it would sweep whatever happened to be on screen, which is not
    what a report about datasets is for."""
    monkeypatch.setattr(window.controller, "scoped_records", list)
    monkeypatch.setattr(
        window.runner, "run",
        lambda *args, **kwargs: pytest.fail("should not have started"),
    )

    window._find_specks(64)


def test_the_collection_actions_wait_for_a_folder(metadata_panel):
    """All three act on the whole scan root, so none is offered before one
    exists. Only recompressing then waits on the editing guard as well: the
    other two write nothing by themselves, and naming projects is how a
    collection gets its first row -- a disabled button there cannot say what it
    is waiting for, so the window answers that when it is pressed."""
    panel = metadata_panel
    assert not panel.survey_button.isEnabled()
    assert not panel.init_button.isEnabled()
    assert not panel.compress_button.isEnabled()

    panel.set_root(Path("/somewhere"))
    assert panel.survey_button.isEnabled()
    assert panel.init_button.isEnabled()
    assert not panel.compress_button.isEnabled()

    panel.enable.setChecked(True)
    assert panel.compress_button.isEnabled()


def test_a_first_run_opens_on_curate(qt_app, settings):
    """Curate once held an empty tree and no cue about what comes first, which
    is why a first run opened on Add data instead. It now offers the folder to
    read where the frames would be, so it is where a first run can act."""
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        assert fresh.tabs.currentWidget() is fresh.curate_panel
        assert fresh.curate_panel.image_panel.stack.currentWidget() is fresh.curate_panel.image_panel.nothing_read
    finally:
        destroy(fresh)


def test_a_first_scan_leaves_the_reviewer_on_curate(qt_app, project: Path, settings):
    """Nothing hands the tab over any more: the tab a first run starts on is
    the tab that shows what the scan just found."""
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        fresh.scan(str(project))

        assert fresh.tabs.currentWidget() is fresh.curate_panel
    finally:
        destroy(fresh)


def test_a_later_scan_leaves_the_reviewer_where_they_are(window, project: Path):
    """The hand-over happens once. Pressing Scan from Export is how a reviewer
    picks up files changed outside the window, and being thrown to Curate for
    it would cost them the filters they had set."""
    window.tabs.setCurrentWidget(window.export_panel)

    window.scan(str(project))

    assert window.tabs.currentWidget() is window.export_panel


def test_the_title_names_the_folder_that_is_open(window, project: Path):
    """Two windows on two collections are told apart by this and nothing else."""
    assert project.name in window.windowTitle()
    assert PRODUCT_NAME in window.windowTitle()


def test_a_remembered_folder_still_opens_on_curate(qt_app, settings, tmp_path):
    """Which is the ordinary case: a scan root, once chosen, is remembered, so
    this changes where a first run lands and nothing else."""
    from am_segmentation_manager.gui.widgets.main_window import SETTING_SCAN_ROOT, CurationWindow

    settings.setValue(SETTING_SCAN_ROOT, str(tmp_path))
    controller = CurationController(cache=FrameCache(), config=CONFIG)
    fresh = CurationWindow(controller, settings=settings)
    try:
        assert fresh.tabs.currentWidget() is fresh.curate_panel
    finally:
        destroy(fresh)


def test_a_window_that_has_read_nothing_offers_the_folder_it_remembers(
    qt_app, settings, project: Path
):
    """The window opens on empty panes that look no different from a lamella
    holding nothing, and nothing in it works until a folder is read. Pressing
    the button in the middle is what the tab row's Scan does, where a first
    user is already looking."""
    from am_segmentation_manager.gui.widgets.main_window import SETTING_SCAN_ROOT, CurationWindow

    settings.setValue(SETTING_SCAN_ROOT, str(project))
    fresh = CurationWindow(CurationController(cache=FrameCache(), config=CONFIG), settings=settings)
    try:
        page = fresh.curate_panel.image_panel.nothing_read
        assert fresh.curate_panel.image_panel.stack.currentWidget() is page, "in place of the panes"
        assert project.name in page.scan_button.text(), "the button names what it would read"
        assert page.path.text() == str(project), "and the path says which folder that is"

        page.scan_button.click()

        assert fresh.controller.start_path == project
        assert fresh.curate_panel.image_panel.stack.currentWidget() is not page, "the page is done with"
        assert not fresh.scan_bar.scan_button.isDefault(), "and Scan is a button like any other"
    finally:
        destroy(fresh)


def test_a_window_with_no_folder_remembered_offers_to_choose_one(qt_app, settings, monkeypatch):
    """A first run of all has nothing to name, so both ways in open the chooser."""
    from am_segmentation_manager.gui.widgets import scan_bar as module

    fresh = CurationWindow(CurationController(cache=FrameCache(), config=CONFIG), settings=settings)
    asked = []
    monkeypatch.setattr(module, "choose_directory", lambda *args, **kwargs: asked.append(args) or None)
    try:
        page = fresh.curate_panel.image_panel.nothing_read
        assert fresh.curate_panel.image_panel.stack.currentWidget() is page
        assert not page.path.isVisible(), "there is no folder to name"
        assert not page.browse_button.isVisible(), "and no second way to say the same thing"

        page.scan_button.click()

        assert asked, "the one button opens the chooser"
    finally:
        destroy(fresh)


def test_rescan_reads_the_remembered_folder_before_anything_has_been_scanned(
    qt_app, settings, project: Path
):
    """A session that opens on a remembered folder has scanned nothing, and the
    key that says Rescan did nothing at all -- silently, and now that F1 lists
    it, having advertised itself first."""
    from am_segmentation_manager.gui.widgets.main_window import SETTING_SCAN_ROOT, CurationWindow

    settings.setValue(SETTING_SCAN_ROOT, str(project))
    fresh = CurationWindow(CurationController(cache=FrameCache(), config=CONFIG), settings=settings)
    try:
        assert fresh.controller.start_path is None, "nothing is read at startup"

        fresh.rescan()

        assert fresh.controller.start_path == project
        assert fresh.controller.projects, "and the projects are there to work on"
    finally:
        destroy(fresh)


def test_rescan_with_no_folder_to_read_says_so_rather_than_nothing(qt_app, settings, monkeypatch):
    """The first run of all: nothing remembered, nothing typed."""
    from am_segmentation_manager.gui.widgets import main_window as module

    said = []
    monkeypatch.setattr(module.CurationWindow, "_flash", lambda self, message, **kw: said.append(message))
    fresh = CurationWindow(CurationController(cache=FrameCache(), config=CONFIG), settings=settings)
    try:
        fresh.rescan()
    finally:
        destroy(fresh)

    assert said and "Browse" in said[0], "and names the way out"


def _menu_items(window, title: str) -> dict[str, str]:
    """Return ``{item text: shortcut}`` for one menu of the bar.

    Plain strings rather than live QActions. PySide6 keeps a wrapper alive only
    while the wrapper that vended it lives, so a QMenu read out of the bar and
    returned past its QAction -- or a QAction returned past its QMenu -- is an
    already-deleted object by the time anything asks it a question. Everything
    is read while the whole chain is still in scope. To *press* an item, go
    through ``window._menu_actions``, which the window holds for this reason.
    """
    bar_action = next(
        action
        for action in window.menuBar().actions()
        if action.text().replace("&", "") == title
    )
    menu = bar_action.menu()
    return {
        item.text(): item.shortcut().toString()
        for item in menu.actions()
        if not item.isSeparator()
    }


# ---------------------------------------------------------------------------
# Who changes are recorded under
# ---------------------------------------------------------------------------

def test_the_window_says_which_name_it_is_recording_under(window, monkeypatch):
    """Beside the mark, on every tab. A name reaches every entry written from
    anywhere in the window, so no one tab is the place to say it and there are
    far too many actions to warn at."""
    from am_segmentation_manager.history import resolve_author

    assert window._corner.author.text() in {
        "Recording without a name",
        f"Recording as {resolve_author()[0]}",
    }


#: What the window is known to hold that records. Named so this test fails
#: loudly if the search below ever stops finding anything, rather than passing
#: on an empty set; the search is what catches a *new* one.
KNOWN_RECORDERS = frozenset({
    "controller",
    "families_controller",
    "history_controller",
    "label_controller",
    "metadata_controller",
    "operations_controller",
    "label_panel.versions_controller",
})


def _things_that_record(window) -> dict[str, object]:
    """Return everything the window reaches that carries a recorder.

    Instance attributes one level into the panels as well as the window's own:
    the versions controller belongs to the Label tab rather than to the window,
    and a search that stopped at the window would miss exactly the one that was
    hardest to remember. The shared
    :class:`~am_segmentation_manager.gui.attribution.Attribution` is itself
    skipped -- it is what they record *through*, not a thing that records.
    """
    found: dict[str, object] = {}

    def offer(name: str, held: object) -> None:
        if isinstance(held, Attribution) or not hasattr(held, "recorder"):
            return
        found[name] = held

    for name, held in vars(window).items():
        offer(name, held)
        for inner, deeper in vars(held).items() if hasattr(held, "__dict__") else ():
            offer(f"{name}.{inner}", deeper)
    return found


def test_setting_a_name_reaches_everything_that_records(window, monkeypatch):
    """Including the curation controller, which records state changes -- the
    one the first version of this forgot.

    Asserted as "they all carry the same name as the line says", not as a
    literal: whether this machine has a name set is not this test's business.
    """
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module
    from am_segmentation_manager.gui.widgets.settings_dialog import author_summary

    class _Answered:
        def __init__(self, *_args, **_kwargs):
            self.author = "Ada"
            # The dialog carries the preferences too; this test is about the
            # name, so it offers none to write.
            self.values: dict[str, int | bool] = {}

        def exec_(self):
            return QDialog.Accepted

    written: list[str] = []
    monkeypatch.setattr(module, "SettingsDialog", _Answered)
    monkeypatch.setattr(module, "set_author", written.append)

    window.show_settings()

    assert written == ["Ada"], "the name was not written"
    # Found rather than listed. A list here would have to be extended by
    # whoever adds a controller, which is the same omission this test exists
    # to catch -- so it asks the window what it holds.
    recording = _things_that_record(window)
    missing = KNOWN_RECORDERS - set(recording)
    assert not missing, f"the search stopped finding {sorted(missing)}"
    # Identity, not equality: two recorders built from the same file agree by
    # accident, and would go on agreeing while one of them was never updated.
    carried = {id(holder.recorder) for holder in recording.values()}
    assert len(carried) == 1, "something that records did not pick the name up"
    assert window._corner.author.text() == author_summary(
        window.controller.recorder.by
    )


def test_a_name_can_be_taken_off_again(window, monkeypatch):
    """A name stated once would otherwise be stated for ever, and somebody who
    set the wrong one needs an undo that is not typing something else."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module

    class _Answered:
        def __init__(self, *_args, **_kwargs):
            self.author = ""
            self.values: dict[str, int | bool] = {}

        def exec_(self):
            return QDialog.Accepted

    cleared: list[bool] = []
    monkeypatch.setattr(module, "SettingsDialog", _Answered)
    monkeypatch.setattr(module, "clear_author", lambda: cleared.append(True))
    monkeypatch.setattr(module, "set_author", lambda name: pytest.fail("wrote a name"))

    window.show_settings()

    assert cleared == [True]


def test_a_name_that_cannot_be_written_takes_the_previewed_accent_back(window, monkeypatch):
    """Nothing the dialog held is kept when the name fails, and an accent left on
    screen would look kept until the window next opened."""
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module

    class _Previewed:
        def __init__(self, _parent, _values, *, preview_accent, viewer=False):
            self.author = "Ada"
            self.values = {"accent": "blue"}
            self._preview = preview_accent

        def exec_(self):
            self._preview("blue")
            return QDialog.Accepted

    def refuse(_name):
        raise OSError("read-only home directory")

    shown: list[str] = []
    monkeypatch.setattr(window, "show_accent", shown.append)
    monkeypatch.setattr(module, "SettingsDialog", _Previewed)
    monkeypatch.setattr(module, "set_author", refuse)
    monkeypatch.setattr(module.QMessageBox, "critical", lambda *args: None)

    window.show_settings()

    from am_segmentation_manager.gui.preferences import DEFAULT_ACCENT

    assert shown == ["blue", DEFAULT_ACCENT]


def test_showing_an_accent_redraws_the_grid_overview_marks(window, monkeypatch):
    """The one thing on the window that keeps the colour it was drawn in."""
    from am_segmentation_manager.gui.widgets import main_window as module

    redrawn: list[bool] = []
    monkeypatch.setattr(module, "apply_accent", lambda application, accent: None)
    monkeypatch.setattr(
        window.curate_panel.image_panel.grid_view, "refresh_accent", lambda: redrawn.append(True)
    )

    window.show_accent("blue")

    assert redrawn == [True]


def test_switching_the_accent_leaves_none_of_the_old_one_on_screen(window, qt_app):
    """Held against the whole window rather than a list of what stores a
    colour, so a widget added later that stores one is caught too."""
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui.widgets import theme

    def count(colour: str) -> int:
        wanted = QColor(colour).rgb()
        total = 0
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            qt_app.processEvents()
            image = window.grab().toImage()
            total += sum(
                1
                for y in range(0, image.height(), 2)
                for x in range(0, image.width(), 2)
                if image.pixel(x, y) == wanted
            )
        return total

    style, palette, previous = qt_app.styleSheet(), qt_app.palette(), theme.active_accent()
    theme.apply_theme(qt_app, theme.ACCENTS["orange"])
    try:
        window.resize(1400, 900)
        window.show()
        orange = theme.ACCENTS["orange"].colour
        assert count(orange), "nothing drawn in the accent to begin with"

        window.show_accent("blue")

        assert count(orange) == 0
        assert count(theme.ACCENTS["blue"].colour)
    finally:
        theme.apply_theme(qt_app, previous)
        qt_app.setStyleSheet(style)
        qt_app.setPalette(palette)


def test_the_line_says_plainly_when_nothing_is_recorded():
    """The case somebody is most likely to be in without knowing it."""
    from am_segmentation_manager.gui.widgets.settings_dialog import author_summary

    assert author_summary(None) == "Recording without a name"
    assert author_summary("Ada") == "Recording as Ada"
    assert "carry nobody" in author_summary(None, asked=True)


def test_the_window_has_the_menus_a_desktop_application_has(window):
    """Nothing here is a second way to drive the window: it is where someone
    looks for the things a window is expected to have and this one had nowhere
    to put."""
    from am_segmentation_manager.launcher import menu_label

    assert set(_menu_items(window, "File")) == {
        "Open folder...", "Rescan", "Settings...", menu_label(), "Quit",
    }
    assert set(_menu_items(window, "Help")) == {
        "Keyboard shortcuts", f"About {PRODUCT_NAME}",
    }


def _launcher_computer(tmp_path, family="linux"):
    """This computer, with its menus and Desktop under *tmp_path*."""
    from am_segmentation_manager.launcher import Places, ThisComputer

    places = Places(menu=tmp_path / "menu", desktop=tmp_path / "Desktop", icons=tmp_path / "icons")
    return ThisComputer(family, places, Path("/env/python"))


def test_the_launcher_dialog_shows_what_it_would_write_before_writing_it(qt_app, tmp_path):
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog
    from am_segmentation_manager.launcher import WINDOW

    dialog = LauncherDialog(None, _launcher_computer(tmp_path))
    try:
        assert str(tmp_path / "menu" / WINDOW.desktop_entry_name) in dialog.paths.text()
        assert str(tmp_path / "Desktop") not in dialog.paths.text()
        assert dialog.add_button.text() == "Add"
        assert not dialog.remove_button.isEnabled()

        dialog.desktop.setChecked(True)

        assert str(tmp_path / "Desktop" / WINDOW.desktop_entry_name) in dialog.paths.text()
        assert not (tmp_path / "menu").exists(), "shown, not written"
    finally:
        destroy(dialog)


def test_the_launcher_dialog_offers_the_desktop_where_the_platform_expects_it(qt_app, tmp_path):
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog

    for family, expected in (("win32", True), ("darwin", False), ("linux", False)):
        dialog = LauncherDialog(None, _launcher_computer(tmp_path, family))
        try:
            assert dialog.desktop.isChecked() is expected, family
        finally:
            destroy(dialog)


def test_the_launcher_dialog_adds_writes_again_and_removes(qt_app, tmp_path):
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog
    from am_segmentation_manager.launcher import WINDOW

    entry = tmp_path / "menu" / WINDOW.desktop_entry_name
    dialog = LauncherDialog(None, _launcher_computer(tmp_path))
    try:
        dialog.add_button.click()

        assert entry.exists()
        assert dialog.status.text().startswith("Added.")
        assert dialog.add_button.text() == "Write again"
        assert dialog.remove_button.isEnabled()

        dialog.add_button.click()
        assert entry.exists()

        dialog.remove_button.click()

        assert not entry.exists()
        assert dialog.add_button.text() == "Add"
        assert not dialog.remove_button.isEnabled()
    finally:
        destroy(dialog)


def test_something_else_in_the_way_disables_adding_and_says_what(qt_app, tmp_path):
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog
    from am_segmentation_manager.launcher import WINDOW

    theirs = tmp_path / "menu" / WINDOW.desktop_entry_name
    theirs.parent.mkdir(parents=True)
    theirs.write_text("[Desktop Entry]\nName=Someone else's\n")
    dialog = LauncherDialog(None, _launcher_computer(tmp_path))
    try:
        assert not dialog.add_button.isEnabled()
        assert not dialog.remove_button.isEnabled()
        assert str(theirs) in dialog.status.text()
    finally:
        destroy(dialog)
    assert theirs.read_text() == "[Desktop Entry]\nName=Someone else's\n"


def test_the_launcher_dialog_offers_the_viewer_beside_the_window(qt_app, tmp_path):
    """Each a launcher of its own: adding or removing one leaves the other."""
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog
    from am_segmentation_manager.launcher import VIEWER, WINDOW

    window_entry = tmp_path / "menu" / WINDOW.desktop_entry_name
    viewer_entry = tmp_path / "menu" / VIEWER.desktop_entry_name
    dialog = LauncherDialog(None, _launcher_computer(tmp_path))
    try:
        assert dialog.app == WINDOW
        dialog.add_button.click()

        dialog.choice.buttons["viewer"].click()

        assert dialog.app == VIEWER
        assert str(viewer_entry) in dialog.paths.text()
        assert dialog.add_button.text() == "Add"
        assert not dialog.remove_button.isEnabled()
        dialog.add_button.click()
        assert viewer_entry.exists() and window_entry.exists()
        assert VIEWER.name in dialog.status.text()

        dialog.remove_button.click()

        assert not viewer_entry.exists()
        assert window_entry.exists()
    finally:
        destroy(dialog)


def test_the_file_menu_opens_the_launcher_dialog(window, monkeypatch, tmp_path):
    from am_segmentation_manager.gui.widgets import main_window
    from am_segmentation_manager.gui.widgets.launcher_dialog import LauncherDialog
    from am_segmentation_manager.launcher import menu_label

    shown: list = []
    monkeypatch.setattr(main_window, "this_computer", lambda: _launcher_computer(tmp_path))
    monkeypatch.setattr(LauncherDialog, "exec_", lambda self: shown.append(self) or 0)

    window._menu_actions[menu_label()].trigger()

    assert len(shown) == 1
    assert shown[0].app.name == PRODUCT_NAME
    destroy(shown[0])


def test_a_computer_with_nowhere_to_add_the_window_says_so(window, monkeypatch):
    from am_segmentation_manager.gui.widgets import main_window
    from am_segmentation_manager.launcher import LauncherRefused

    said: list[str] = []

    def refuse():
        raise LauncherRefused("Windows did not name its Start menu and Desktop")

    monkeypatch.setattr(main_window, "this_computer", refuse)
    monkeypatch.setattr(
        main_window.QMessageBox, "warning", lambda parent, title, text: said.append(text)
    )

    window.show_launcher_dialog()

    assert said == ["Windows did not name its Start menu and Desktop"]


def test_the_shortcut_menu_item_binds_no_key_of_its_own(window):
    """F1 is already an application shortcut, and Qt answers two bindings of
    one key with an ambiguous overload where neither fires reliably. The list
    teaches its own key instead -- it holds a row for F1."""
    assert _menu_items(window, "Help")["Keyboard shortcuts"] == ""


def test_the_shortcut_menu_item_opens_the_same_list_f1_does(window, qt_app):
    window._menu_actions["Keyboard shortcuts"].trigger()
    try:
        assert window._keys.help is not None
        assert not window._keys.help.isHidden()
    finally:
        if window._keys.help is not None:
            window._keys.help.close()


def test_a_stored_default_is_what_every_control_opens_at(qt_app, settings):
    """One store, four panels, and no control left on the packaged answer.

    The window is what reads the settings and hands them round, so a panel
    built on its own still opens at what the package ships -- which is why
    this asserts through a window rather than through a panel.
    """
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    preferences.write(settings, {
        "speck_pixels": 21,
        "label_opacity": 15,
        "brush_radius": 3,
        "train_share": 60,
        "seed": 7,
        "one_per_label": True,
        "keep_together": "project",
        "folds": 4,
    })
    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        assert window.curate_panel.operations_panel.speck_size.value() == 21
        assert window.label_panel.speck_slider.value() == 21
        # Held on the controller as well: a slider moved without it would
        # report sizes nothing was counted at.
        assert window.label_panel.controller.speck_size == 21
        assert window.curate_panel.image_panel.alpha_slider.value() == 15
        assert window.label_panel.alpha_slider.value() == 15
        assert window.export_panel.train_share.value() == 60
        assert window.export_panel.seed.value() == 7
        assert window.export_panel.keep_together.currentData() == "project"
        assert window.export_panel.folds.value() == 4
        assert not window.export_panel.train_share.isEnabled()
        assert window.export_panel.conditions.isChecked()
    finally:
        destroy(window)


def test_a_default_decides_what_a_control_opens_at_and_nothing_else(qt_app, settings):
    """The rule the whole idea rests on, and the reason it is safe to keep
    these on one machine: what a run produces comes from the packaged
    constants and the controls as they stand, never from a preference reached
    behind them. So changing one while a control is on screen changes nothing
    there -- the dialog says so, and this holds it."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.curate_panel.image_panel.alpha_slider.setValue(90)

        preferences.write(settings, {"label_opacity": 10})

        assert window.curate_panel.image_panel.alpha_slider.value() == 90, (
            "a control somebody is using is not moved under them"
        )
    finally:
        destroy(window)


def test_the_settings_dialog_offers_every_preference(qt_app, settings):
    """Built from the table, so a row added there reaches the dialog without
    an edit here -- and one that did not would be a setting nobody could
    change."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        offered = dialog.values

        assert sorted(offered) == sorted(entry.key for entry in preferences.PREFERENCES)
        assert offered == preferences.read(settings)
    finally:
        destroy(dialog)


def test_training_defaults_that_cannot_be_trained_with_are_not_kept(qt_app, settings):
    """Each box holds a valid number; the pair of them need not be."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        dialog._editors["train/new/frozen_epochs"].setValue(500)
        dialog.accept()

        assert dialog.result() != SettingsDialog.DialogCode.Accepted
        assert not dialog.problem_label.isHidden()
        assert preferences.TRAIN_NEW in dialog.problem_label.text()

        dialog.restore_buttons[preferences.TRAIN_NEW].click()
        dialog.accept()

        assert dialog.result() == SettingsDialog.DialogCode.Accepted
        assert dialog.values["train/new/frozen_epochs"] == 25
    finally:
        destroy(dialog)


def test_a_learning_rate_is_typed_as_it_is_written(qt_app, settings):
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        rate = dialog._editors["train/refine/initial_learning_rate"]
        assert rate.text() == "2e-06"

        rate.setText("5e-7")
        assert dialog.values["train/refine/initial_learning_rate"] == 5e-7

        rate.setText("fast")
        assert dialog.values["train/refine/initial_learning_rate"] == 2e-6, "what it held"
    finally:
        destroy(dialog)


def test_the_train_tab_opens_at_the_defaults_kept_in_settings(qt_app, project, settings):
    """And F2 changed while the window is open is what My defaults next gives."""
    from am_segmentation_manager.gui import preferences

    preferences.write(settings, {"train/new/max_epochs": 200, "train/refine/seed": 7})
    controller = CurationController(cache=FrameCache(), config=SegmentationConfig(
        class_colors={0: (0, 0, 0)}, class_labels={0: "background"},
    ))
    window = CurationWindow(controller, settings=settings)
    try:
        training = window.training_controller
        assert training.settings.max_epochs == 200

        training.set_mode("refine")
        assert training.settings.seed == 7

        preferences.write(settings, {"train/refine/seed": 8})
        training.reset_settings()
        assert training.settings.seed == 8
    finally:
        destroy(window)


def test_the_settings_dialog_returns_the_accent_chosen_in_it(qt_app, settings):
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        dialog.show()
        qt_app.processEvents()
        run = dialog._editors["accent"]
        assert dialog.values["accent"] == preferences.DEFAULT_ACCENT
        assert run.width() == run.sizeHint().width(), "a form pulls the segments apart"

        run.buttons["blue"].click()

        assert dialog.values["accent"] == "blue"
    finally:
        destroy(dialog)


def test_choosing_an_accent_shows_it_at_once_and_cancel_puts_it_back(qt_app, settings):
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    shown: list[str] = []
    dialog = SettingsDialog(None, preferences.read(settings), preview_accent=shown.append)
    try:
        dialog._editors["accent"].buttons["blue"].click()
        assert shown == ["blue"]

        dialog.reject()

        assert shown == ["blue", preferences.DEFAULT_ACCENT]
    finally:
        destroy(dialog)


@pytest.mark.parametrize("close", ["accept", "reject"])
def test_an_accent_left_as_it_opened_is_not_shown_again(qt_app, settings, close):
    """Accepted, the previewed accent is the one kept; cancelled with nothing
    chosen, there is nothing to put back."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    shown: list[str] = []
    dialog = SettingsDialog(None, preferences.read(settings), preview_accent=shown.append)
    try:
        if close == "accept":
            dialog._editors["accent"].buttons["blue"].click()
        getattr(dialog, close)()

        assert shown == (["blue"] if close == "accept" else [])
    finally:
        destroy(dialog)


def test_each_accent_offered_carries_its_own_colour(qt_app, settings):
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets import theme
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        for key, button in dialog._editors["accent"].buttons.items():
            dot = button.icon().pixmap(theme.SWATCH_DOT).toImage()
            middle = theme.SWATCH_DOT // 2
            assert dot.pixelColor(middle, middle) == QColor(theme.ACCENTS[key].colour), key
    finally:
        destroy(dialog)


def test_a_status_line_that_starts_empty_grows_to_what_it_says(qt_app):
    """Its height followed only a resize, so a line empty when the dialog
    opened stayed no lines tall whatever it was later given to say."""
    from am_segmentation_manager.gui.widgets.controls import WrappingLabel

    label = WrappingLabel()
    try:
        label.setFixedWidth(200)
        label.show()
        qt_app.processEvents()
        assert label.maximumHeight() == 0

        label.setText("A sentence long enough to need more than one line at this width.")

        assert label.maximumHeight() > label.fontMetrics().lineSpacing()
    finally:
        destroy(label)


def test_every_group_of_settings_gets_the_height_its_rows_need(qt_app, settings):
    """A word-wrapped note reports a one-line hint, and the squeeze lands elsewhere.

    Qt asks a label for its size hint before it knows how wide it will be, so
    a dialog sized from that comes up short and every widget below is pushed
    towards its minimum -- which showed as the Export box drawing four
    switches on top of the three rows under them, some way from the label that
    caused it. `controls.WrappingLabel` is the fix; this is what noticed.
    """
    from qtpy.QtWidgets import QGroupBox

    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import SettingsDialog

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        dialog.show()
        qt_app.processEvents()
        dialog.grab()

        short = {
            box.title(): (box.height(), box.layout().sizeHint().height())
            for box in dialog.findChildren(QGroupBox)
            if box.height() < box.layout().sizeHint().height()
        }

        assert not short, f"boxes shorter than their rows need: {short}"
    finally:
        destroy(dialog)


def test_the_settings_dialog_is_capped_to_the_screen_and_scrolls(qt_app, settings):
    """It ran past a laptop's screen, leaving the last settings and OK beyond
    reach: the settings scroll instead, under buttons that stay in view. The
    offscreen screen is shorter than every setting together, as a laptop's is."""
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets.settings_dialog import (
        SCREEN_MARGIN,
        SettingsDialog,
    )

    dialog = SettingsDialog(None, preferences.read(settings))
    try:
        dialog.show()
        qt_app.processEvents()
        room = qt_app.primaryScreen().availableGeometry()

        assert dialog.height() <= room.height() - SCREEN_MARGIN
        assert dialog.width() <= room.width() - SCREEN_MARGIN
        assert dialog._body.height() > dialog._scroll.viewport().height()
        assert dialog._scroll.verticalScrollBar().isVisible()
        assert dialog.buttons.isVisible()
        assert dialog.buttons.geometry().bottom() <= dialog.rect().bottom()
    finally:
        destroy(dialog)


def test_nothing_in_the_filters_column_is_clipped_at_the_width_it_opens_at(
    qt_app, settings
):
    """Every compaction here trades height for width, and the width has to be there.

    Captions beside their boxes, a run of segments for the beam and two short
    switches on one line each save a row and cost columns -- and squeezed, Qt
    does not shorten a caption but cuts it from both ends. The first cut of
    this column drew the beam's segments blank and "Adaptive Polishing" as
    "Adaptive Po", which reads as a rendering fault rather than as a column
    that wants dragging.
    """
    from qtpy.QtWidgets import QAbstractButton, QLabel

    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.tabs.setCurrentWidget(window.export_panel)
        window.resize(1512, 900)
        window.show()
        qt_app.processEvents()
        window.grab()

        # PySide6 takes one type, not a tuple of them.
        shown = [
            widget
            for block in (window.export_panel.series_block, window.export_panel.fixed_filters)
            for widget in (
                *block.findChildren(QLabel), *block.findChildren(QAbstractButton)
            )
        ]
        clipped = {
            widget.text(): (widget.width(), widget.sizeHint().width())
            for widget in shown
            if widget.text() and widget.width() < widget.sizeHint().width()
        }

        assert not clipped, f"cut off in the Filters column: {clipped}"
    finally:
        destroy(window)


def test_the_settings_button_opens_what_f2_opens(window, monkeypatch):
    """The corner is where the name is shown, so it is where it is changed.

    F2 is the only other way in and nothing on screen said so -- the same gap
    the help button beside it fills for F1.
    """
    from qtpy.QtWidgets import QDialog

    from am_segmentation_manager.gui.widgets import main_window as module

    opened: list[bool] = []

    class _Dismissed:
        def __init__(self, *_args, **_kwargs):
            self.author = "Ada"
            self.values: dict[str, int | bool] = {}

        def exec_(self):
            opened.append(True)
            return QDialog.Rejected

    monkeypatch.setattr(module, "SettingsDialog", _Dismissed)

    window._corner.settings_button.click()

    assert opened == [True]


def test_the_corner_shows_the_mark_without_naming_the_product(corner_bar):
    """The window would otherwise say its own name twice: the title bar carries
    it on Windows and Linux, the menu bar on macOS, and it was printed here as
    well. The mark stays -- it says whose window this is without repeating it."""
    from qtpy.QtWidgets import QLabel

    from am_segmentation_manager.identity import PRODUCT_NAME

    corner = corner_bar
    labels = corner.findChildren(QLabel)

    assert not corner.mark.icon().isNull(), "the mark is what is left"
    assert not corner.mark.text()
    assert all(label.text() != PRODUCT_NAME for label in labels)
    assert PRODUCT_NAME in corner.mark.toolTip(), "named on hover, for a reader who asks"


def test_the_mark_opens_what_help_about_opens(window, monkeypatch):
    """Whose window this is, and so where a reader looks for which version it is."""
    opened = []
    monkeypatch.setattr(
        QMessageBox, "exec_", lambda self: opened.append(self.windowTitle()) or 0, raising=False
    )

    window._corner.mark.click()
    window.menu_actions()[f"About {PRODUCT_NAME}"].trigger()

    assert len(opened) == 2 and opened[0] == opened[1]


def test_the_path_field_gives_way_before_the_tabs_do(scan_bar):
    """A tab row too narrow for everything in it takes the room from the tabs,
    which Qt shrinks to slivers nobody can hit. The field is what shrinks."""
    from am_segmentation_manager.gui.widgets.scan_bar import (
        SCAN_FIELD_MINIMUM,
        SCAN_FIELD_WIDTH,
    )

    field = scan_bar.path_edit

    assert field.minimumWidth() == SCAN_FIELD_MINIMUM < SCAN_FIELD_WIDTH
    assert field.maximumWidth() == SCAN_FIELD_WIDTH, "and no wider than it reads well at"


def test_the_help_button_in_the_corner_opens_the_list(window):
    """F1 is the only way in, and nothing on screen mentions F1."""
    corner = window.tabs.cornerWidget(Qt.TopRightCorner)
    button = next(
        child for child in corner.findChildren(QToolButton) if child.text() == "?"
    )

    button.click()
    try:
        assert window._keys.help is not None
    finally:
        if window._keys.help is not None:
            window._keys.help.close()


def test_about_says_the_version_and_where_the_log_went(window, monkeypatch, tmp_path):
    """The two things anybody reporting a problem is asked for first, and
    neither was anywhere on screen. The log is read back from the handlers, so
    a run started with --log-file reports where its output actually went."""
    import logging

    from am_segmentation_manager import __version__

    shown = {}

    def capture(self):
        shown["text"] = self.text()
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", capture, raising=False)
    handler = logging.FileHandler(tmp_path / "run.log")
    logging.getLogger().addHandler(handler)
    try:
        window._show_about()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    assert __version__ in shown["text"]
    assert "run.log" in shown["text"]


def test_about_says_how_recent_this_version_is(window, monkeypatch):
    """What tells a reader whether it is worth looking for a newer one."""
    from am_segmentation_manager.gui.widgets import main_window

    shown = {}
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: shown.update(text=self.text()) or 0, raising=False)
    monkeypatch.setattr(main_window, "version_date", lambda: "Latest change: 2026-10-07")

    window._show_about()

    assert "Latest change: 2026-10-07" in shown["text"]


def test_about_says_where_to_look_for_a_newer_version_and_how_to_take_it(window, monkeypatch):
    """A link the reader follows themselves, and the steps for this install."""
    from am_segmentation_manager.gui.widgets import main_window
    from am_segmentation_manager.identity import REPOSITORY_URL

    shown = {}
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: shown.update(text=self.informativeText()) or 0, raising=False)
    monkeypatch.setattr(main_window, "update_steps", lambda: (f"Listed at {REPOSITORY_URL}", "run <this>"))

    window._show_about()

    assert f'<a href="{REPOSITORY_URL}">' in shown["text"]
    assert "run &lt;this&gt;" in shown["text"]


def test_the_tab_says_how_much_of_the_folder_is_not_indexed(
    project: Path, window
):
    """The project list shows what is indexed, which is silent about the data
    sitting beside it that nothing has indexed -- the state a collection is in
    before any of this has been done."""
    unindexed = project / "loose-dataset" / "01-lamella" / "stage" / "sem"
    unindexed.mkdir(parents=True)
    (unindexed / "img.tif").write_bytes(b"i")

    window.rescan()

    said = window.metadata_panel.unindexed.text()
    assert "1 folder(s) here contain 1 image(s)" in said
    assert "Add projects" in said


# ---------------------------------------------------------------------------
# The dialog a long operation runs behind
# ---------------------------------------------------------------------------

def test_the_busy_dialog_does_not_process_events(window):
    """QProgressDialog.setValue() calls processEvents() whenever the dialog is
    modal, which re-enters the event loop from inside a queued-signal handler:
    the worker's "finished" can then be delivered in the middle of a progress
    update and close the dialog under the code still using it. It showed as a
    freeze on the final item."""
    from qtpy.QtWidgets import QProgressDialog

    from am_segmentation_manager.gui.widgets.busy_dialog import BusyDialog

    window.runner.run("Long thing", lambda: None, lambda result: None)
    try:
        assert isinstance(window.runner._busy, BusyDialog)
        assert not isinstance(window.runner._busy, QProgressDialog)
    finally:
        window.runner._take_pending()


def test_progress_arriving_after_the_work_returned_is_harmless(window):
    """Both are queued signals from the same thread, so the last report can
    land after the result. This is the shape of the freeze: the handler must
    find no dialog rather than one being taken apart."""
    window.runner.run("Long thing", lambda: None, lambda result: None)
    window.runner._take_pending()

    window.runner._progressed(9, 9)

    assert window.runner._busy is None


def test_the_busy_dialog_reports_how_far_the_work_has_got(window):
    window.runner.run("Long thing", lambda: None, lambda result: None)
    try:
        assert window.runner._busy.bar.maximum() == 0, "opens indeterminate"

        window.runner._progressed(3, 9)

        assert window.runner._busy.bar.maximum() == 9
        assert window.runner._busy.bar.value() == 3
        assert "3 of 9" in window.runner._busy.message.text()
    finally:
        window.runner._take_pending()


def test_escape_does_not_dismiss_an_operation_that_cannot_be_stopped(busy_dialog):
    """The dialog is the only thing keeping the window out of files that are
    being written."""
    busy_dialog.reject()

    # isHidden() rather than isVisible(): a dialog shown on a window that
    # was never shown is never "visible", so isVisible() would pass here
    # whether or not Escape had dismissed it.
    assert busy_dialog.isHidden() is False


def test_escape_on_a_stoppable_operation_asks_it_to_stop(window):
    from am_segmentation_manager.progress import Cancellation

    cancellation = Cancellation()
    window.runner.run(
        "Long thing", lambda: None, lambda result: None, cancellation=cancellation
    )
    try:
        window.runner._busy.reject()

        assert cancellation.requested is True
    finally:
        window.runner._take_pending()


def test_the_collection_actions_fit_the_pane_they_sit_in(metadata_panel):
    """A QPushButton clips its caption rather than eliding it, and clipping
    takes the ends off both sides at once -- so a row too wide for the pane
    reads as "nspect layou" rather than as something obviously cut off."""
    from am_segmentation_manager.gui.widgets.metadata_panel import (
        PROJECT_PANE_MIN_WIDTH,
    )

    for box in (metadata_panel.data_box, metadata_panel.maintenance_box):
        assert box.minimumSizeHint().width() <= PROJECT_PANE_MIN_WIDTH, box.title()


def test_scanning_names_the_collection_the_actions_would_write_to(window, project: Path):
    assert window.metadata_panel.root == project
    assert project.name in window.metadata_panel.data_box.title()


def test_what_acts_on_every_label_is_held_apart_from_what_reads_the_folder(metadata_panel):
    """Restoring and reclaiming change what is kept across every project;
    inspecting the layout and adding projects only read the folder or add to it."""
    panel = metadata_panel

    for button in (panel.survey_button, panel.init_button):
        assert panel.data_box.isAncestorOf(button), button.text()
    for button in (
        panel.savepoint_button, panel.restore_button,
        panel.compress_button, panel.reclaim_button,
    ):
        assert panel.maintenance_box.isAncestorOf(button), button.text()


def test_the_editing_switch_comes_before_the_fields_it_unlocks(metadata_panel):
    """It decides whether any field can be typed in, and was found last."""
    panel = metadata_panel
    column = panel.enable.parentWidget().layout()

    assert column.indexOf(panel.enable) == 0


def test_choosing_where_predictions_come_from_recounts_the_line(window, monkeypatch):
    """The line has to describe the conversion the button would now run."""
    _tick_lamella(window)
    window.curate_panel.refresh()
    asked: list[str] = []
    describe = window.operations_controller.describe_predictions
    monkeypatch.setattr(
        window.operations_controller, "describe_predictions",
        lambda records, prefer: asked.append(prefer) or describe(records, prefer),
    )

    buttons = window.curate_panel.operations_panel.origin_control.buttons
    opened_on = window.curate_panel.operations_panel.prediction_origin()
    # Every origin, ending on the one it opened on: one origin clicked once
    # is also what a line counted from that origin alone would be asked for.
    each = [origin for origin in buttons if origin != opened_on] + [opened_on]
    assert len(each) > 1

    for origin in each:
        buttons[origin].click()

    assert asked == each
    assert window.curate_panel.operations_panel.prediction_origin() == opened_on


def test_an_operation_really_runs_off_the_interface_thread(qt_app, window, monkeypatch):
    """The seam every other test substitutes, exercised once for real.

    Asserted on the thread the work ran on: a queued connection that degraded
    to a direct call would still deliver a result, and would freeze the window
    for the minutes a compression takes. And on the thread the result arrives
    on, which is the interface thread only while the runner is a ``QObject``
    living there.

    The failure path is redirected because the window reports it in a modal
    box, which in a test run waits for a click that never comes -- an
    unexpected failure here has to fail the test, not stop the suite.
    """
    import threading

    failures: list = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda parent, title, text, *args: failures.append(text),
    )
    results: list = []
    window.runner.run(
        "Test",
        lambda: threading.current_thread().name,
        lambda name: results.append((name, threading.current_thread().name)),
    )
    settle(qt_app, lambda: bool(results) or bool(failures))

    assert not failures
    worked_on, handed_back_on = results[0]
    assert worked_on != threading.current_thread().name
    assert handed_back_on == threading.current_thread().name
    # The buttons locked for the duration are usable again.
    assert window.curate_panel.operations_panel.convert_button.isEnabled()


def test_a_failing_operation_is_reported_and_leaves_the_window_usable(
    qt_app, window, monkeypatch
):
    reported: list = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda parent, title, text, *args: reported.append(text),
    )

    window.runner.run("Test", lambda: 1 / 0, lambda result: pytest.fail("no result"))
    settle(qt_app, lambda: bool(reported))

    assert "division by zero" in reported[0]
    assert window.curate_panel.operations_panel.convert_button.isEnabled()


def test_an_operation_stopped_partway_names_what_it_wrote_on_the_records(
    qt_app, project: Path, window, monkeypatch
):
    """Failing is not the same as having changed nothing: a label written
    before the failure is on disk, and the window has to say so."""
    from am_segmentation_manager.operations.label_io import label_destination
    from am_segmentation_manager.progress import StoppedPartway

    record = _unlabelled_record(window)
    written = label_destination(record)
    reported: list = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda parent, title, text, *args: reported.append(text),
    )

    def stopped():
        raise StoppedPartway("disk full after one label", [written])

    window.runner.run("Test", stopped, lambda result: pytest.fail("no result"))
    settle(qt_app, lambda: bool(reported))

    assert "disk full" in reported[0]
    assert record.labels == [written]


def test_a_report_lines_up_in_a_fixed_width_font(qt_app):
    """A report pads its columns with spaces, which line up only where every
    character advances as far as every other. Asked of the metrics, never of
    fixedPitch(), which reports the family requested rather than the one found."""
    from qtpy.QtGui import QFontMetricsF

    from am_segmentation_manager.gui.widgets.report_dialog import ReportDialog

    dialog = ReportDialog(None, "Inspect layout", ["headline"])
    metrics = QFontMetricsF(dialog.view.font())

    assert metrics.horizontalAdvance("i") == metrics.horizontalAdvance("W")
    destroy(dialog)


def test_a_report_dialog_shows_its_lines_and_asks_only_when_it_can_act(qt_app):
    """The dialog every operation reports through, on both of its two jobs."""
    from qtpy.QtWidgets import QDialogButtonBox

    from am_segmentation_manager.gui.widgets.report_dialog import ReportDialog

    asking = ReportDialog(None, "Compress labels", ["headline", "a/b.tif"], "Apply")
    assert asking.text() == "headline\na/b.tif"
    assert asking.buttons.button(QDialogButtonBox.Ok).text() == "Apply"
    # Cancel is the default, so Return cannot apply an operation by accident.
    assert asking.buttons.button(QDialogButtonBox.Cancel).isDefault()
    # exec() rather than exec_(): present and undeprecated under both bindings.
    assert callable(asking.exec)

    telling = ReportDialog(None, "Inspect layout", ["headline"])
    assert telling.buttons.button(QDialogButtonBox.Ok) is None
    destroy(asking)
    destroy(telling)


# ---------------------------------------------------------------------------
# Plots are prepared at the size they are shown at
# ---------------------------------------------------------------------------


def test_swapping_resolution_repeatedly_does_not_drift(qt_app):
    """Stepping through a series while zoomed swaps copies twice a frame. Each
    swap used to re-derive where the view was looking through the scroll
    position, which is a whole number, losing a fraction of a pixel the same
    way every time -- so the picture crept out of the pane."""
    import numpy as np

    from am_segmentation_manager.gui.widgets.image_view import ImageView

    small = np.zeros((300, 400, 3), dtype=np.uint8)
    large = np.zeros((1200, 1600, 3), dtype=np.uint8)
    view = ImageView()
    view.resize(400, 300)
    view.show()
    qt_app.processEvents()
    try:
        view.set_array(small)
        view.zoom_by(1.25)
        view.zoom_by(1.25)

        def looking_at():
            rect = view._item.boundingRect()
            middle = view.mapToScene(view.viewport().rect().center())
            return (middle.x() / rect.width(), middle.y() / rect.height())

        for _ in range(3):                       # settles on the first swap
            view.set_array(large, same_picture=True)
            view.set_array(small, same_picture=True)
        settled = looking_at()

        for _ in range(20):
            view.set_array(large, same_picture=True)
            view.set_array(small, same_picture=True)

        assert looking_at() == pytest.approx(settled, abs=0.001)
    finally:
        view.close()


def test_a_change_of_resolution_keeps_the_region_on_screen(qt_app):
    """The mechanism the swap rests on, on its own: the same picture at more
    pixels is the same view at a smaller scale about a larger centre."""
    import numpy as np

    from am_segmentation_manager.gui.widgets.image_view import ImageView

    view = ImageView()
    view.resize(400, 300)
    view.show()
    qt_app.processEvents()
    try:
        view.set_array(np.zeros((300, 400, 3), dtype=np.uint8))
        view.zoom_by(1.25)
        qt_app.processEvents()
        drawn = view._item.boundingRect().width() * view.transform().m11()
        centre = view.mapToScene(view.viewport().rect().center())

        view.set_array(np.zeros((1200, 1600, 3), dtype=np.uint8), same_picture=True)

        after = view.mapToScene(view.viewport().rect().center())
        assert view._item.boundingRect().width() * view.transform().m11() == (
            pytest.approx(drawn, rel=0.01)
        )
        assert (after.x(), after.y()) == pytest.approx(
            (centre.x() * 4, centre.y() * 4), rel=0.01
        )
    finally:
        view.close()


def test_a_figure_of_another_size_starts_again_rather_than_inheriting_the_zoom(
    image_panel
):
    """Holding the view across a change of resolution is for the same figure
    read whole. A figure of another size is another picture, and carrying a
    zoom onto it would show it at a scale chosen for something else."""
    import numpy as np

    from am_segmentation_manager.gui.cache import FramePlot

    panel = image_panel
    panel.set_mode("plot")
    pane = panel.plot_view.viewport().rect()
    panel.show_plot(FramePlot(plot=np.zeros((pane.height(), pane.width(), 3), dtype=np.uint8)))
    panel.plot_view.zoom_by(1.25)
    panel.plot_view.zoom_by(1.25)
    assert not panel.plot_view.transform().isIdentity()

    panel.show_plot(FramePlot(plot=np.zeros((240, 320, 3), dtype=np.uint8)))

    # Identity is one array pixel per physical pixel, which is what a copy
    # prepared for this pane is drawn at.
    assert panel.plot_view.transform().isIdentity()


def test_a_plot_prepared_for_another_pane_still_fills_this_one(image_panel):
    """The copy is reduced to a box measured in physical pixels, so trusting
    the screen's ratio at drawing time leaves it at half or twice the size
    whenever the two disagree -- which is what a band of empty pane above and
    below the figure was."""
    import numpy as np

    from am_segmentation_manager.gui.cache import FramePlot

    panel = image_panel
    panel.set_mode("plot")
    viewport = panel.plot_view.viewport().rect()

    for ratio in (1, 2, 3):
        wide = round(viewport.width() * ratio)
        tall = round(viewport.height() * ratio)
        panel.show_plot(FramePlot(plot=np.zeros((tall, wide, 3), dtype=np.uint8)))

        item = panel.plot_view._item.boundingRect()
        drawn = item.width() * panel.plot_view.transform().m11()
        assert drawn == pytest.approx(viewport.width(), rel=0.02), f"ratio {ratio}"


# ---------------------------------------------------------------------------
# The metadata tab is read before it is written
# ---------------------------------------------------------------------------

@pytest.fixture
def metadata_panel(qt_app, tmp_path: Path):
    """A metadata panel over two projects, built without scanning for them.

    The YAML files are written even though the projects are built by hand: a
    save goes to disk and re-reads what landed there, so a fixture that only
    existed in memory would lose half its metadata the first time one happened.
    """
    experiment = {"metadata": {"organisation": "Rosalind Franklin Institute"}}

    def project(name: str, own: dict) -> Project:
        # Nested in a session folder, as Autolamella projects are: several
        # experiments from one sitting share a directory.
        path = tmp_path / "20260728_CB_AM_bitmap" / name
        path.mkdir(parents=True)
        (path / "experiment.yaml").write_text(
            "metadata:\n  organisation: Rosalind Franklin Institute\n"
        )
        (path / "AM_metadata.yaml").write_text(
            "metadata:\n" + "".join(f"  {key}: {value}\n" for key, value in own.items())
        )
        return Project(
            name=name,
            path=path,
            relative_path=f"20260728_CB_AM_bitmap/{name}",
            experiment_metadata=build_experiment_metadata(
                experiment,
                {"metadata": own},
                experiment_path=path / "experiment.yaml",
                am_metadata_path=path / "AM_metadata.yaml",
            ),
        )

    controller = MetadataController()
    controller.set_projects([
        project("described", {"sample": "HeLa", "user": "Casper"}),
        project("outstanding", {"sample": "TODO"}),
    ])
    panel = MetadataPanel(controller)
    panel.set_projects(controller.project_summaries())
    yield panel
    destroy(panel)


def test_a_fully_indexed_folder_says_nothing_about_unindexed_data(metadata_panel):
    """A line that stayed from the scan before would send somebody to add
    projects that are already there."""
    metadata_panel.set_unindexed(1, 1)
    assert metadata_panel.unindexed.text()

    metadata_panel.set_unindexed(0, 0)

    assert metadata_panel.unindexed.text() == ""


def test_a_locked_field_still_shows_its_value(metadata_panel):
    """The tab opens locked and is mostly used for reading.

    That the value is also drawn *legibly* while locked is a property of the
    theme, and is asserted on the pixels in ``test_gui_theme.py``.
    """
    assert editor_text(metadata_panel.editors["sample"]) == "HeLa"
    assert all(not editor.isEnabled() for editor in metadata_panel.editors.values())


def test_every_field_says_what_is_expected_in_it(metadata_panel):
    """Hovering either the name or the box has to explain the field."""
    for entry in FIELDS:
        assert metadata_panel.captions[entry.name].toolTip() == entry.guidance
        assert metadata_panel.editors[entry.name].toolTip() == entry.guidance


def test_a_field_holding_several_says_so_without_being_hovered(metadata_panel):
    """A hover explains a control to someone who already suspects it exists.
    Which fields take more than one answer has to be readable at a glance, or
    the ones that do look exactly like the ones that do not."""
    plural = {"(s)", "media"}
    for entry in FIELDS:
        caption = metadata_panel.captions[entry.name].text()
        reads_plural = any(caption.endswith(ending) for ending in plural)
        assert reads_plural == entry.multi, caption
        says_so = "Holds several" in metadata_panel.editors[entry.name].toolTip()
        assert says_so == entry.multi, entry.name


def test_a_value_says_on_hover_which_file_it_came_from(metadata_panel):
    """On the dot rather than in words beside the box, which cost every row the
    width its chips need; the whole path is there."""
    sample = metadata_panel.dots["sample"].toolTip()
    organisation = metadata_panel.dots["organisation"].toolTip()

    shown = sample.split("From ", 1)[1]
    assert Path(shown).is_absolute() and sample.endswith("AM_metadata.yaml")
    assert organisation.endswith("experiment.yaml")


def test_a_folder_names_the_file_in_every_project_it_covers():
    from am_segmentation_manager.gui.widgets.common import MAX_LISTED_FILES
    from am_segmentation_manager.gui.widgets.metadata_panel import source_tooltip

    paths = tuple(
        Path(f"/collection/project{index}/AM_metadata.yaml")
        for index in range(MAX_LISTED_FILES + 2)
    )

    tip = source_tooltip(paths)

    assert tip.startswith(f"From this file in each of the {len(paths)} projects:")
    assert str(paths[0]) in tip
    assert tip.endswith("... and 2 more")


def _current_project(panel) -> int | None:
    """Return the project number the tree's cursor is on."""
    item = panel.projects.currentItem()
    return panel.projects.project_of(item) if item is not None else None


def test_stepping_moves_to_the_next_project_and_stops_at_the_ends(metadata_panel):
    """What Up/Down are bound to on this tab. It steps through *projects*: the
    folders are rows too, and stepping onto one would describe nothing."""
    metadata_panel.step_project(1)
    assert _current_project(metadata_panel) == 1
    assert metadata_panel.controller.position == 1

    # The last project: stepping past it stays there rather than wrapping.
    metadata_panel.step_project(1)
    assert _current_project(metadata_panel) == 1

    metadata_panel.step_project(-1)
    assert _current_project(metadata_panel) == 0
    assert metadata_panel.controller.position == 0


def test_an_outstanding_field_is_marked_rather_than_dimmed(metadata_panel):
    """A placeholder is the state that most needs seeing; it used to be drawn
    in the dimmest colour on the panel, in the same grey as provenance."""
    from am_segmentation_manager.gui.widgets.metadata_panel import _dot_pixmap

    metadata_panel.projects.setCurrentItem(metadata_panel.projects.project_item(1))

    dot = metadata_panel.dots["sample"]
    assert dot.state == "missing"
    assert "placeholder" in dot.toolTip()
    assert dot.pixmap().toImage() == _dot_pixmap("missing").toImage()
    assert dot.pixmap().toImage() != _dot_pixmap("filled").toImage()
    assert editor_text(metadata_panel.editors["sample"]) == ""


def test_a_change_not_yet_saved_is_drawn_apart_from_a_missing_field(qt_app):
    """An edited field's dot is ink, neither the accent nor the colour of a
    missing field's ring, so colour stays a second channel beside shape."""
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui.widgets.frame_strip import MISMATCH
    from am_segmentation_manager.gui.widgets.metadata_panel import DOT, _dot_pixmap
    from am_segmentation_manager.gui.widgets.theme import TEXT, active_accent

    centre = _dot_pixmap("edited").toImage().pixelColor(DOT // 2, DOT // 2)

    assert centre == QColor(TEXT)
    assert centre not in (QColor(active_accent().colour), MISMATCH)


def test_the_fields_sit_under_their_section_headings(metadata_panel):
    """General, then how the sample was prepared, then the lamella: the order
    somebody describing an experiment works through it."""
    from qtpy.QtWidgets import QApplication

    metadata_panel.resize(1200, 1400)
    metadata_panel.show()
    QApplication.processEvents()
    headings = metadata_panel.section_headings

    def top(widget) -> int:
        return widget.mapTo(metadata_panel, widget.rect().topLeft()).y()

    assert list(headings) == ["General", "Sample preparation", "Lamella preparation"]
    names = list(headings)
    for entry in FIELDS:
        here = names.index(entry.section)
        field = top(metadata_panel.editors[entry.name])
        assert top(headings[entry.section]) < field, entry.name
        assert all(field < top(headings[later]) for later in names[here + 1:]), entry.name


def test_every_single_line_box_is_one_width(metadata_panel):
    """A column of boxes reads as one only if they end together."""
    from qtpy.QtWidgets import QComboBox, QLineEdit

    from am_segmentation_manager.gui.widgets.metadata_panel import FIELD_WIDTH

    boxes = [
        editor.entry if isinstance(editor, ChipEditor) else editor
        for editor in metadata_panel.editors.values()
        if isinstance(editor, (ChipEditor, QComboBox, QLineEdit))
    ]

    assert len(boxes) == len(FIELDS) - 1
    assert {box.width() for box in boxes} == {FIELD_WIDTH}


def test_the_description_wraps_and_is_saved_as_one_line(metadata_panel):
    from qtpy.QtWidgets import QPlainTextEdit

    box = metadata_panel.editors["description"]
    box.setPlainText("A lamella\nseries")

    assert box.lineWrapMode() == QPlainTextEdit.WidgetWidth
    assert box.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    assert box.height() > 2 * metadata_panel.editors["grid_id"].sizeHint().height()
    assert editor_text(box) == "A lamella series"


def test_discarding_puts_every_field_back_and_writes_nothing(metadata_panel):
    project = metadata_panel.controller.scope[0]
    before = (project.path / "AM_metadata.yaml").read_text()
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")

    metadata_panel.discard_button.click()

    assert editor_text(metadata_panel.editors["description"]) == ""
    assert metadata_panel.dots["description"].state == "missing"
    assert (project.path / "AM_metadata.yaml").read_text() == before


def test_an_edited_field_is_marked_until_it_is_saved(metadata_panel):
    """Saving is a separate press, so a change still only in the window has to
    look different from one that reached the file."""
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")

    assert metadata_panel.dots["description"].state == "edited"
    assert metadata_panel.dots["sample"].state == "filled"

    metadata_panel._save()

    assert metadata_panel.dots["description"].state == "filled"


def test_repeating_fields_offer_the_values_already_in_use(metadata_panel):
    box = metadata_panel.editors["sample"].entry

    assert box.isEditable()
    # A blank first entry, so "unset" stays reachable from the list.
    assert [box.itemText(row) for row in range(box.count())] == ["", "HeLa"]
    assert metadata_panel.editors["description"].__class__.__name__ == "QPlainTextEdit"


def test_a_single_valued_field_keeps_its_one_box(metadata_panel):
    """Only a field that holds several gets chips. A grid type is one thing,
    and offering to add a second would say it might not be."""
    from qtpy.QtWidgets import QComboBox

    assert isinstance(metadata_panel.editors["grid_type"], QComboBox)
    assert isinstance(metadata_panel.editors["lamella_type"], QComboBox)


def test_a_value_not_in_the_list_can_still_be_typed(metadata_panel):
    """The list is a shortcut, not a vocabulary -- the first project to use a
    sample has nothing to pick. And it is added to what is there rather than
    replacing it, which is the whole reason the field holds several."""
    metadata_panel.enable.setChecked(True)
    editor = metadata_panel.editors["sample"]
    editor.entry.setCurrentText("Chlamydomonas")
    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["HeLa", "Chlamydomonas"]
    assert metadata_panel._edits() == {"sample": "HeLa; Chlamydomonas"}
    # Typing must not grow the list, or a typo would be offered back later.
    assert editor.entry.count() == 2


def test_a_value_typed_and_not_confirmed_is_still_saved(metadata_panel):
    """Regression: the box held what you typed until Return or focus left it,
    so typing a value and reaching straight for Save discarded it silently --
    which is every first attempt at using the field."""
    metadata_panel.enable.setChecked(True)
    metadata_panel.editors["organisation"].entry.setCurrentText("Diamond")

    assert metadata_panel._edits() == {
        "organisation": "Rosalind Franklin Institute; Diamond"
    }


def test_leaving_the_box_turns_what_is_in_it_into_a_value(metadata_panel):
    """So that what was typed is visibly a value before Save is pressed,
    rather than only becoming one when Save reads the editor."""
    metadata_panel.enable.setChecked(True)
    editor = metadata_panel.editors["project"]
    editor.entry.setCurrentText("Adaptive milling")

    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["Adaptive milling"]
    assert editor.entry.currentText() == ""


def test_the_usual_answers_are_offered_before_anyone_has_given_one(metadata_panel):
    """A default is offered, where a vocabulary is required: plunge and high
    pressure cover almost everything, and jet freezing still has to be typable."""
    from am_segmentation_manager.metadata import FREEZING_METHODS

    box = metadata_panel.editors["freezing_method"]
    offered = [box.itemText(row) for row in range(box.count())]

    assert offered == ["", *FREEZING_METHODS]
    assert box.isEditable(), "an unusual method is typed, not refused"


def test_a_chip_can_be_taken_off_again(metadata_panel):
    """And not while editing is locked: the cross is a control on a value, and
    the lock covers every other way of changing one."""
    editor = metadata_panel.editors["sample"]
    assert editor.values() == ["HeLa"]
    editor._chip_layout.itemAt(0).widget().click()
    assert editor.values() == ["HeLa"], "locked, so the cross does nothing"

    metadata_panel.enable.setChecked(True)

    editor._chip_layout.itemAt(0).widget().click()

    assert editor.values() == []
    assert metadata_panel._edits() == {"sample": None}


def test_the_wider_project_holds_several_like_the_rest(metadata_panel):
    """It names the project several datasets belong to, so it repeats across
    them -- which is what the value lists are for. It used to carry a button
    copying the directory name in, which is a different fact entirely."""
    editor = metadata_panel.editors["project"]

    assert isinstance(editor, ChipEditor)
    assert editor.entry.isEditable()
    assert not hasattr(metadata_panel, "folder_button")


def test_the_list_says_which_projects_still_need_describing(metadata_panel):
    """Without this you have to click every project to find the empty ones."""
    described = metadata_panel.projects.project_item(0)
    outstanding = metadata_panel.projects.project_item(1)

    total = len(DESCRIPTIVE_FIELDS)
    assert f"3/{total} completed" in described.toolTip(0)
    # Only the organisation, from experiment.yaml: its sample is a placeholder.
    assert f"1/{total} completed" in outstanding.toolTip(0)
    assert not described.icon(0).isNull()


def test_the_projects_are_grouped_under_their_session_folder(metadata_panel):
    """The same tree the Curate tab lists them in. Describing a collection is
    done a session at a time, and an Autolamella project is named for the
    minute it started."""
    tree = metadata_panel.projects
    folders = [
        tree.topLevelItem(row).text(0)
        for row in range(tree.topLevelItemCount())
        if tree.project_of(tree.topLevelItem(row)) is None
    ]

    assert folders == ["20260728_CB_AM_bitmap"]
    assert len(tree.project_items()) == 2


def test_add_data_offers_expand_and_collapse(metadata_panel):
    """Describing a collection means working down one session at a time, which
    needs its folder opened; thirty left open is the other half of that."""
    tree = metadata_panel.projects
    tree.collapseAll()
    assert not any(
        tree.topLevelItem(row).isExpanded()
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).childCount()
    )

    from qtpy.QtWidgets import QPushButton

    buttons = metadata_panel.findChildren(QPushButton)
    expand = next(b for b in buttons if b.text() == "Expand")
    collapse = next(b for b in buttons if b.text() == "Collapse")
    expand.click()

    assert all(
        tree.topLevelItem(row).isExpanded()
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).childCount()
    )
    collapse.click()
    assert not any(
        tree.topLevelItem(row).isExpanded()
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).childCount()
    )


def test_the_add_data_tree_has_no_tick_boxes(metadata_panel):
    """This tab describes one project at a time, so a box would offer a set it
    cannot act on."""
    from qtpy.QtCore import Qt as _Qt

    for item in metadata_panel.projects.project_items():
        assert not item.flags() & _Qt.ItemIsUserCheckable


def test_a_row_carries_its_own_name_and_not_the_folder_again(metadata_panel):
    """The folder above has said it once, and repeating it on every row is
    what made the flat list a column of identical prefixes."""
    item = metadata_panel.projects.project_item(0)

    assert "20260728_CB_AM_bitmap" not in item.text(0)
    assert item.toolTip(0) == f"3/{len(DESCRIPTIVE_FIELDS)} completed", (
        "the row names the project, and right-click copies its path"
    )


def test_a_long_row_is_elided_rather_than_scrolled(metadata_panel):
    """These paths differ in the timestamp they end with. With a horizontal
    scrollbar the view scrolls instead of eliding, and a run of projects from
    one session is cut off at the edge into identical-looking rows."""
    item = metadata_panel.projects.project_item(0)

    assert metadata_panel.projects.textElideMode() == Qt.ElideMiddle
    assert metadata_panel.projects.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert item.text(0)


def test_saving_metadata_starts_the_record_it_writes_into(metadata_panel):
    """The third door a change comes through, and the one that never reaches
    the window: the panel calls its controller directly."""
    from am_segmentation_manager.history import history_file_path, read_log

    project = metadata_panel.controller.scope[0]
    assert not history_file_path(project.path).is_file(), "nothing recorded yet"
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")

    metadata_panel._save()

    kinds = [event.kind for event in read_log(project.path).events]
    assert "log-created" in kinds, "the write started the log"
    assert "metadata-changed" in kinds, kinds


def test_saving_updates_the_mark_in_the_list(metadata_panel):
    """Saving is exactly when a project stops being incomplete, so leaving the
    list describing the state before it is how a user learns to distrust it."""
    total = len(DESCRIPTIVE_FIELDS)
    assert f"3/{total} completed" in (
        metadata_panel.projects.project_item(0).toolTip(0)
    )
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["description"], "A lamella series")
    metadata_panel._save()

    assert f"4/{total} completed" in (
        metadata_panel.projects.project_item(0).toolTip(0)
    )


def test_right_clicking_a_project_asks_for_its_path(metadata_panel):
    """Reusing the curate tab's menu: the row is emitted, and resolving the
    path stays the controller's job."""
    rows: list[int] = []
    metadata_panel.project_path_requested.connect(rows.append)
    metadata_panel.project_path_requested.emit(1)

    assert rows == [1]
    assert metadata_panel.controller.project_path(1).is_absolute()


# ---------------------------------------------------------------------------
# Choosing what to predict on
# ---------------------------------------------------------------------------
#
# Developed on a Mac, which has MPS and a CPU. A workstation has several CUDA
# devices and no MPS, so nothing here may assume the machine it runs on.

def test_the_device_box_offers_what_every_machine_has(operations_panel):
    """Before torch is loaded, the only honest answers are "the best one" and
    "the one everything has". Asking torch costs an import, which a dropdown
    may not spend."""
    offered = [
        operations_panel.device_box.itemText(index)
        for index in range(operations_panel.device_box.count())
    ]

    assert offered == ["auto", "cpu"]
    assert operations_panel.device() == "auto"


def test_the_device_box_takes_the_machines_own_devices(operations_panel):
    """A CUDA workstation, as torch would report it."""
    operations_panel.show_devices(("cuda:0", "cuda:1", "cpu"))

    offered = [
        operations_panel.device_box.itemText(index)
        for index in range(operations_panel.device_box.count())
    ]

    assert offered == ["auto", "cuda:0", "cuda:1", "cpu"]


def test_a_device_this_package_has_never_seen_can_be_typed(operations_panel):
    """resolve_device passes a named device through unexamined, so support for
    a new accelerator is not something anyone has to wait for."""
    operations_panel.device_box.setCurrentText("xpu:0")

    assert operations_panel.device() == "xpu:0"


def test_filling_the_box_keeps_what_was_chosen(operations_panel):
    """It is filled when the box is opened, which must not undo a choice made
    before the machine's devices were known."""
    operations_panel.set_device("cuda:1")

    operations_panel.show_devices(("cuda:0", "cuda:1", "cpu"))

    assert operations_panel.device() == "cuda:1"


def test_a_gpu_is_offered_with_its_model_and_free_memory_and_read_back_bare(operations_panel):
    """Four identical cards are told apart by how full they are, which is what
    someone sharing the machine chooses by; the run is given the device alone."""
    panel = operations_panel
    described = "cuda:1 · NVIDIA A100 80GB PCIe · 71.2 of 80 GB free"

    panel.show_devices(("cuda:0", "cuda:1", "cpu"), {"cuda:1": described})
    panel.set_device("cuda:1")

    assert panel.device_box.currentText() == described
    assert panel.device() == "cuda:1"


def test_a_described_device_typed_or_pasted_is_read_back_bare(operations_panel):
    operations_panel.device_box.setCurrentText("cuda:2 · NVIDIA A100-PCIE-40GB · 39 GB free")

    assert operations_panel.device() == "cuda:2"


def test_new_free_memory_keeps_the_gpu_that_was_chosen(operations_panel):
    """The descriptions are read afresh each time the box opens."""
    panel = operations_panel
    panel.show_devices(("cuda:0", "cuda:1", "cpu"), {"cuda:1": "cuda:1 · A100 · 70 GB free"})
    panel.set_device("cuda:1")

    panel.show_devices(("cuda:0", "cuda:1", "cpu"), {"cuda:1": "cuda:1 · A100 · 12 GB free"})

    assert panel.device() == "cuda:1"
    assert panel.device_box.currentText() == "cuda:1 · A100 · 12 GB free"


def test_gpus_torch_cannot_use_are_said_so_on_the_box(operations_panel):
    from am_segmentation_manager.gui.widgets.operations_panel import (
        DEVICE_HINT,
        UNUSABLE_GPU_HINT,
    )

    panel = operations_panel
    panel.show_devices(("cpu",), problem="4 NVIDIA GPU(s) found, but torch cannot use them")
    assert panel.device_box.toolTip() == UNUSABLE_GPU_HINT

    panel.show_devices(("cuda:0", "cpu"))
    assert panel.device_box.toolTip() == DEVICE_HINT


def test_opening_the_device_box_asks_torch_once_and_nvidia_smi_every_time(window, monkeypatch):
    """Which devices exist costs a torch import; how full the GPUs are changes
    while colleagues work, and costs nvidia-smi alone."""
    from am_segmentation_manager.gui.widgets import operation_actions
    from am_segmentation_manager.ml import gpus
    from am_segmentation_manager.operations.predict_process import DeviceProbe

    probes: list[int] = []
    monkeypatch.setattr(
        operation_actions, "probe_devices",
        lambda: probes.append(1) or DeviceProbe(devices=("cuda:0", "cuda:1", "cpu")),
    )
    free = iter([70.0, 12.0])

    def inventory():
        gb = next(free)
        gpu = gpus.Gpu(1, 1, "GPU-1", "NVIDIA A100 80GB PCIe", int(gb * 1024), 80 * 1024, "0:1")
        return gpus.GpuInventory(gpus=(gpu,))

    monkeypatch.setattr(gpus, "gpu_inventory", inventory)
    items = window.curate_panel.operations_panel.device_box

    window._offer_devices()
    first = [items.itemText(index) for index in range(items.count())]
    window._offer_devices()
    second = [items.itemText(index) for index in range(items.count())]

    assert probes == [1]
    assert first[2] == "cuda:1 · NVIDIA A100 80GB PCIe · 70.0 of 80 GB free"
    assert second[2] == "cuda:1 · NVIDIA A100 80GB PCIe · 12.0 of 80 GB free"
    assert [first[0], first[1], first[3]] == ["auto", "cuda:0", "cpu"]


def test_the_chosen_device_reaches_the_run(window, monkeypatch, tmp_path: Path):
    """Every prediction from the window used to be 'auto'."""
    seen: list[str] = []
    monkeypatch.setattr(
        window.operations_controller, "predict",
        lambda records, weights, device="auto", **kwargs: seen.append(device),
    )
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems", lambda: []
    )
    monkeypatch.setattr(
        window.runner, "run",
        lambda description, work, on_done, cancellation=None: work(),
    )
    window.controller.editing_enabled = True
    window.curate_panel.operations_panel.set_device("cpu")
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"x")

    window._predict(str(weights))

    assert seen == ["cpu"]
    assert window.settings.value("predict_device", "", type=str) == "cpu"


def test_predicting_stays_off_where_it_was_never_available(operations_panel):
    """Whether the inference stack is installed is asked once. Unlocking the
    buttons after an operation must not answer it again by accident."""
    operations_panel.set_prediction_available(False)

    operations_panel.set_busy(True)
    operations_panel.set_busy(False)

    assert not operations_panel.predict_button.isEnabled()
    assert not operations_panel.device_box.isEnabled()


# ---------------------------------------------------------------------------
# The chip editor, as a widget
# ---------------------------------------------------------------------------

@pytest.fixture
def editor(qt_app):
    """A chip editor on its own, destroyed when the test ends."""
    editor = ChipEditor()
    yield editor
    destroy(editor)


def test_chips_wrap_onto_another_row_rather_than_off_the_edge(editor):
    """Values that do not fit used to be laid out below the bottom edge, where
    they are addressable and invisible -- so a field would report two
    organisations while showing one."""
    editor._chips.resize(90, 0)
    editor.set_values(["alpha"])
    one_row = editor._chips.height()

    editor.set_values(["alpha", "beta", "gamma", "delta"])

    assert one_row > 0
    assert editor._chips.height() > one_row
    assert editor._chip_layout.count() == 4


def test_a_chip_is_tall_enough_to_be_seen_as_soon_as_it_exists(editor):
    """Regression, and the worst kind: adding a widget to a layout reparents
    it, reparenting hides it, and a hidden widget measures as nothing -- so the
    row of chips was given no height. The values were held, addressable and
    saved, and simply not on screen, which reads as a field that refuses to
    take a value and as one that loses the value already in it."""
    editor.show()
    editor._chips.resize(300, 0)

    editor.set_values(["RFI"])

    assert editor._chips.isVisible()
    assert editor._chips.height() > 0
    assert editor._chip_layout.heightForWidth(300) > 0


def test_chips_come_back_after_a_project_that_had_none(editor):
    """What stepping through a collection does to one editor. A project with
    nothing in the field hides the chips, and the next project has to be able
    to show them again -- which is the case that failed, because a chip cannot
    be shown into a parent that is still hidden."""
    editor.show()
    editor._chips.resize(300, 0)

    for values in (["RFI"], [], ["RFI", "Diamond"], [], ["EMBL"]):
        editor.set_values(values)

        assert (editor._chips.height() > 0) == bool(values), values
        assert editor.values() == values


def test_a_value_sits_beside_the_box_rather_than_on_a_row_of_its_own(qt_app, editor):
    """A row per field whether or not it holds anything, so fields with values
    and fields without them are the same height."""
    editor.resize(600, 80)
    editor.show()
    editor.set_values(["RFI"])
    qt_app.processEvents()
    chip = editor._chip_layout.itemAt(0).widget()
    corner = chip.mapTo(editor, chip.rect().topLeft())

    assert corner.x() >= editor.entry.width()
    assert corner.y() < editor.entry.height()


def test_a_box_with_no_values_beside_it_stays_at_the_left(qt_app, editor):
    """Regression: with the chips hidden, a box aligned only by its top was
    centred in the room they left, out of line with every other box."""
    editor.resize(600, 40)
    editor.show()
    editor.set_values([])
    qt_app.processEvents()

    assert editor.entry.x() == 0


def test_a_chip_area_takes_no_minimum_width(editor):
    """A minimum is a floor. Asked for the widest value in it, the pane could
    never be narrower than the longest organisation anybody had typed."""
    editor.set_values(["a value considerably longer than the pane it sits in"])

    assert editor._chip_layout.minimumSize().width() == 0
    assert editor._chip_layout.sizeHint().width() > 0


def test_typing_several_values_at_once_makes_several_chips(editor):
    """Pasting a separated line is how anyone would try it first."""
    editor.entry.setCurrentText("infected; differentiated")
    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["infected", "differentiated"]
    assert editor.entry.currentText() == "", "the box empties into the chips"


def test_a_value_already_there_is_not_added_twice(editor):
    editor.set_values(["HeLa"])
    reported: list[int] = []
    editor.changed.connect(lambda: reported.append(1))

    editor.entry.setCurrentText("HeLa")
    editor.entry.lineEdit().editingFinished.emit()

    assert editor.values() == ["HeLa"]
    assert reported == [], "nothing changed, so nothing is reported"


def test_setting_the_values_is_a_redraw_rather_than_an_edit(editor):
    """The panel fills every editor whenever it refreshes, and a refresh that
    reported edits would have the window saving what it had just read."""
    reported: list[int] = []
    editor.changed.connect(lambda: reported.append(1))

    editor.set_values(["RFI", "Diamond"])

    assert editor.values() == ["RFI", "Diamond"]
    assert reported == []


# ---------------------------------------------------------------------------
# The legend
#
# A mark and its explanation must be drawn by the same code, or they drift.
# ---------------------------------------------------------------------------

def _legend_images(help_text: str) -> list[bytes]:
    """Return the pictures embedded in a help list, decoded."""
    import base64
    import re

    return [
        base64.b64decode(match)
        for match in re.findall(r'<img src="data:image/png;base64,([^"]+)">', help_text)
    ]


def test_the_legend_shows_the_marks_rather_than_describing_them(window, qt_app):
    """Every legend row carries a picture drawn by whatever draws the real
    mark. A described colour and a drawn one drift apart; the same pixmap
    cannot."""
    from qtpy.QtGui import QImage

    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    images = _legend_images(window._keys.help_html(TAB_CURATE))

    assert len(images) >= 10
    for encoded in images:
        picture = QImage.fromData(encoded, "PNG")
        assert not picture.isNull()
        assert picture.width() > 0 and picture.height() > 0


def test_the_chip_legend_is_the_chip_the_rows_carry(window, qt_app):
    """The point of the legend: the fill named here is the fill drawn there."""
    from qtpy.QtGui import QImage

    from am_segmentation_manager.gui.widgets.project_panel import CELL_COLOURS
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    shown = {
        QImage.fromData(encoded, "PNG").pixelColor(1, 1).name()
        for encoded in _legend_images(window._keys.help_html(TAB_CURATE))
    }

    assert {colour.name() for colour in CELL_COLOURS.values()} <= shown


def test_the_legend_says_what_the_tab_actually_draws(window):
    """The white band means frames picked out while curating and unsaved edits
    while proofreading. One mark, two meanings: listing both on both tabs would
    make it mean neither."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE, TAB_LABEL

    curate = window._keys.help_html(TAB_CURATE)
    label = window._keys.help_html(TAB_LABEL)

    assert "Picked out for a state change" in curate
    assert "unsaved edits" not in curate
    assert "unsaved edits" in label
    assert "Picked out for a state change" not in label


def test_the_field_dots_are_explained_by_the_dots_themselves(window):
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_FIELD_MARK,
        TAB_ADD_DATA,
    )

    help_text = window._keys.help_html(TAB_ADD_DATA)

    assert GROUP_FIELD_MARK in help_text
    assert "Changed here and not saved yet" in help_text


def test_a_tab_with_no_marks_gets_no_legend(window):
    """Export draws neither a chip nor the bar, and a heading with nothing
    under it is a question raised and not answered."""
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_ROW_MARK,
        GROUP_STRIP_MARK,
        TAB_EXPORT,
    )

    help_text = window._keys.help_html(TAB_EXPORT)

    assert GROUP_ROW_MARK not in help_text
    assert GROUP_STRIP_MARK not in help_text


def test_the_legend_comes_after_the_keys_and_before_this_list(window):
    """A reader looking for a key should not have to read past the pictures,
    and the key that opened the list is the one they have already found."""
    from am_segmentation_manager.gui.widgets.shortcuts import (
        GROUP_HELP,
        GROUP_ROW_MARK,
        GROUP_STRIP_MARK,
        GROUP_VIEW,
        TAB_CURATE,
    )

    help_text = window._keys.help_html(TAB_CURATE)
    places = [
        help_text.index(group)
        for group in (GROUP_VIEW, GROUP_ROW_MARK, GROUP_STRIP_MARK, GROUP_HELP)
    ]

    assert places == sorted(places)


def test_the_shortcut_list_is_capped_to_the_screen(window, qt_app):
    """The list runs past 800 pixels on two tabs now the legend is in it, and a
    dialog merely sized past the screen has no scrollbar the user can reach
    either. Built directly rather than through F1, which does nothing while any
    modal dialog is up."""
    from am_segmentation_manager.gui.widgets.shortcuts import SCREEN_MARGIN, TAB_LABEL, ShortcutHelp

    dialog = ShortcutHelp(window)
    dialog.show_for("Label", window._keys.help_html(TAB_LABEL))
    room = qt_app.primaryScreen().availableGeometry()

    assert dialog.height() <= room.height() - SCREEN_MARGIN
    assert dialog.width() <= room.width() - SCREEN_MARGIN
    # And the rows past the cap are still reachable rather than cut off.
    assert dialog._text.sizeHint().height() > dialog.height()
    assert dialog._scroll.verticalScrollBarPolicy() != Qt.ScrollBarAlwaysOff


def test_the_timeline_legend_describes_nothing_the_timeline_omits(qt_app):
    """`MISMATCH` was commented as a wash over frames disagreeing with their
    label, and `_paint` has never drawn one. A legend row for it would have
    been a mark the reader could look for for ever."""
    from am_segmentation_manager.gui.widgets.frame_strip import timeline_legend

    meanings = " ".join(text for _, text in timeline_legend("picked out")).lower()

    assert "amber" not in meanings
    assert "disagree" not in meanings


def test_a_mark_survives_the_round_trip_into_the_list(qt_app):
    """`QBuffer(QByteArray())` lets the array die under the buffer and
    segfaults PySide6, so the bytes have to be held in a name."""
    import base64
    import re

    from qtpy.QtGui import QColor, QImage, QPixmap

    from am_segmentation_manager.gui.widgets.common import mark_image

    pixmap = QPixmap(25, 16)
    pixmap.fill(QColor("#12604f"))
    encoded = re.search(r'base64,([^"]+)"', mark_image(pixmap)).group(1)
    restored = QImage.fromData(base64.b64decode(encoded), "PNG")

    assert restored.size() == pixmap.size()
    assert restored.pixelColor(12, 8) == QColor("#12604f")


# ---------------------------------------------------------------------------
# Describing a folder of projects at once
# ---------------------------------------------------------------------------

def _choose_folder(panel, folder: str) -> None:
    """Move the tree onto a folder row, as clicking one would."""
    tree = panel.projects
    row = next(
        tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).text(0) == folder
    )
    tree.setCurrentItem(row)


def test_choosing_a_folder_describes_its_projects_together(metadata_panel):
    """The point of grouping them: a session's projects share their sample,
    their grid and their operator."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")

    assert len(metadata_panel.controller.scope) == 2
    assert metadata_panel.controller.scope_folder == "20260728_CB_AM_bitmap"


def test_a_field_the_projects_disagree_on_shows_mixed(metadata_panel):
    """A field holding several values shows what any project holds, never a
    `(mixed)` chip: a value added beside one wrote it into every project."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    sample = metadata_panel.editors["sample"]

    assert sample.values() == ["HeLa"]
    assert sample.partial_values() == ["HeLa"]
    assert metadata_panel.dots["sample"].state == "mixed"


def test_a_value_only_some_hold_typed_again_is_saved_to_all(metadata_panel):
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    sample = metadata_panel.editors["sample"]
    sample.entry.setCurrentText("HeLa")
    sample.commit_pending()

    assert sample.partial_values() == []
    assert metadata_panel.dots["sample"].state == "edited"

    metadata_panel.enable.setChecked(True)
    metadata_panel._save()

    for project in metadata_panel.controller.scope:
        assert project.experiment_metadata.sample == ("HeLa",)


def test_a_mixed_field_left_alone_is_not_written(metadata_panel):
    """The whole rule. The marker is what the box holds *and* what the value
    resolves to, so an untouched field compares equal and never reaches a
    file -- accepted knowing it means a field can be edited without being
    written."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")

    assert metadata_panel._edits() == {}


def test_typing_over_mixed_writes_it_to_every_project(metadata_panel):
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    set_editor_text(metadata_panel.editors["sample"], "HeLa")

    assert metadata_panel._edits() == {"sample": "HeLa"}

    metadata_panel.enable.setChecked(True)
    metadata_panel._save()

    for project in metadata_panel.controller.scope:
        assert "HeLa" in (project.path / "AM_metadata.yaml").read_text()


def test_the_title_counts_the_folder_rather_than_reading_it_out(metadata_panel):
    """A dozen project names do not fit and would not help."""
    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    metadata_panel.refresh()

    assert metadata_panel.fields_box.title() == (
        "Metadata: 20260728_CB_AM_bitmap (2 projects)"
    )


def test_clearing_a_field_across_a_folder_is_confirmed_first(metadata_panel, monkeypatch):
    """A field the projects disagree on shows as a chip with a cross on it,
    which looks like tidying away a placeholder rather than like emptying that
    field in every project under the folder."""
    from qtpy.QtWidgets import QMessageBox

    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    metadata_panel.enable.setChecked(True)
    set_editor_text(metadata_panel.editors["sample"], "")
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.No)[1],
    )

    metadata_panel._save()

    assert asked and "all 2 projects" in asked[0]
    assert metadata_panel.status.text() == "Not saved"


def test_describing_one_project_needs_no_such_confirmation(metadata_panel, monkeypatch):
    """The edit is visible in the box, on the project on screen."""
    from qtpy.QtWidgets import QMessageBox

    metadata_panel.enable.setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.Yes)[1],
    )
    set_editor_text(metadata_panel.editors["user"], "")

    metadata_panel._save()

    assert not asked


def test_writing_a_value_across_a_folder_is_not_confirmed(metadata_panel, monkeypatch):
    """That is what describing them together is for."""
    from qtpy.QtWidgets import QMessageBox

    _choose_folder(metadata_panel, "20260728_CB_AM_bitmap")
    metadata_panel.enable.setChecked(True)
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.Yes)[1],
    )
    set_editor_text(metadata_panel.editors["sample"], "HeLa")

    metadata_panel._save()

    assert not asked
    for project in metadata_panel.controller.scope:
        assert "HeLa" in (project.path / "AM_metadata.yaml").read_text()


# ---------------------------------------------------------------------------
# The minimum/maximum range
# ---------------------------------------------------------------------------


def test_the_whole_range_says_nothing(image_panel):
    """A line that is always there is a line nobody reads."""
    image_panel.show_image(_first_frame(), show_overlay=True)
    assert "shown" not in image_panel.badges.text()

    image_panel.intensity_slider.set_range(40, 200)
    image_panel.show_image(_first_frame(), show_overlay=True)
    assert "shown 40-200" in image_panel.badges.text(), "said while it is narrowed"

    image_panel.reset_intensity_range()
    image_panel.show_image(_first_frame(), show_overlay=True)
    assert "shown" not in image_panel.badges.text()


def test_the_range_opens_on_everything(image_panel):
    """A frame opens showing exactly what was recorded."""
    from am_segmentation_manager.display import DEFAULT_INTENSITY_RANGE

    assert image_panel._intensity_range == DEFAULT_INTENSITY_RANGE


def test_the_overlay_is_composed_from_the_ranged_image(image_panel, monkeypatch):
    """The two panes must not disagree about how the same acquisition looks, so
    the transform runs once and the label is drawn over its result -- not over
    the array as it came out of the cache."""
    from am_segmentation_manager.gui.widgets import image_panel as module

    panel = image_panel
    shown: list = []
    composed: list = []

    def ranged(image, low, high):
        # A copy, so that what the control produced can be told from what it was given.
        shown.append(image.copy())
        return shown[-1]

    monkeypatch.setattr(module, "apply_intensity_range", ranged)
    monkeypatch.setattr(
        module, "compose_overlay",
        lambda image, *args, **kwargs: composed.append(image) or image,
    )

    panel.intensity_slider.set_range(40, 255)
    panel.show_image(_first_frame(), show_overlay=True)

    assert composed, "the frame under test carries a label to draw"
    assert composed[0] is shown[0], "composed from what the control produced"


def test_a_caption_is_never_a_flow_item_of_its_own(qt_app):
    """A wrapping row may not put a caption on one line and its control on the next.

    The failure this stops is not a control going missing but a row going
    unreadable: "Minimum / maximum" ended a line with its slider starting the
    next one, where it read as a heading for whatever followed. Asserted on
    the structure rather than on a width, because which pairs get split is
    whatever the pane happens to be -- it looked right on the monitor it was
    built on and wrong on the one beside it.
    """
    from qtpy.QtWidgets import QLabel

    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    # Held in a local: a PySide6 wrapper lives only as long as the wrapper that
    # vended it, so `ImagePanel(config).controls` hands back an object already
    # deleted. `docs/display.md` § The binding.
    panel = ImagePanel(config)
    try:
        layout = panel.controls.layout()
        loose = [
            layout.itemAt(index).widget().text()
            for index in range(layout.count())
            if isinstance(layout.itemAt(index).widget(), QLabel)
        ]

        assert not loose, f"captions that can wrap away from their control: {loose}"
    finally:
        destroy(panel)


def test_the_view_controls_are_one_block_on_the_smallest_screen_supported(
    qt_app, settings, project
):
    """Paired into two lines, they fit the centre pane rather than wrapping.

    The pane is what is left of the window after the project list and the
    state panel, so these controls have far less width than the tab does.
    Loose, they needed 1441px against the 922 a laptop leaves, and wrapped
    into an emergent shape that moved with every window size. Paired, the
    block is the same on every screen it is asked to open on, which is what
    makes it readable rather than merely reachable.
    """
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.tabs.setCurrentWidget(window.curate_panel)
        heights = []
        for width in (SMALLEST_SUPPORTED_SCREEN[0], 1920):
            window.resize(width, 900)
            window.show()
            qt_app.processEvents()
            window.grab()
            heights.append(window.curate_panel.image_panel.controls.height())
            if width == SMALLEST_SUPPORTED_SCREEN[0]:
                controls = window.curate_panel.image_panel.controls
                flow = controls.layout()
                has = controls.contentsRect().width()
                needs = flow.sizeHint().width()
                blocks = [flow.itemAt(i).sizeHint().width() for i in range(flow.count())]

        # The widths say how far over the row is, and each block's which of
        # them grew, where the heights alone say only that it wrapped.
        assert heights[0] == heights[1], (
            f"the row wraps differently at {SMALLEST_SUPPORTED_SCREEN[0]} "
            f"than at 1920: {heights}; it needs {needs}px of the {has}px the "
            f"pane has there, in blocks and gaps of {blocks}"
        )
    finally:
        destroy(window)


def test_the_series_switches_open_two_above_two(qt_app, settings):
    """Loose, they wrapped wherever their captions fell, and the second row
    lined up with nothing above it."""
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.tabs.setCurrentWidget(window.curate_panel)
        window.resize(*SMALLEST_SUPPORTED_SCREEN)
        window.show()
        qt_app.processEvents()
        window.grab()
        boxes = window.curate_panel.project_panel.shown_boxes
        at = {
            name: box.mapTo(window.curate_panel.project_panel, box.rect().topLeft())
            for name, box in boxes.items()
        }

        assert at["adaptive"].y() == at["reference"].y()
        assert at["alignment"].y() == at["other"].y() > at["adaptive"].y()
        assert at["adaptive"].x() == at["alignment"].x()
        assert at["reference"].x() == at["other"].x() > at["adaptive"].x()
    finally:
        destroy(window)


def test_a_view_that_cannot_be_chosen_says_why(image_panel):
    panel = image_panel

    panel.set_available_modes(["sem"])
    assert panel.mode_buttons["grid"].toolTip() == "No grid overview for this project"

    panel.set_available_modes(["sem", "grid"])
    assert "Shortcut: 4" in panel.mode_buttons["grid"].toolTip()


def test_every_view_slider_is_drawn_one_way(window):
    """Opacity beside Min / Max read as two kinds of control while Fusion drew
    one of them."""
    from am_segmentation_manager.gui.widgets.controls import ValueSlider

    sliders = (
        window.curate_panel.image_panel.alpha_slider,
        window.label_panel.alpha_slider,
        window.label_panel.brush_slider,
        window.label_panel.speck_slider,
    )

    assert all(isinstance(slider, ValueSlider) for slider in sliders)


def test_the_opacity_and_min_max_sliders_are_the_same_length(qt_app, settings):
    """Paired one above the other, they read as a pair only if they end together."""
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config), settings=settings
    )
    try:
        window.resize(1512, 900)
        window.show()
        widths = {}
        for tab, panel in (
            (window.curate_panel, window.curate_panel.image_panel), (window.label_panel, window.label_panel)
        ):
            window.tabs.setCurrentWidget(tab)
            qt_app.processEvents()
            window.grab()
            widths[type(panel).__name__] = (
                panel.alpha_slider.width(), panel.intensity_slider.width()
            )

        assert all(alpha == limits for alpha, limits in widths.values()), widths
    finally:
        destroy(window)


def test_a_control_row_still_wraps_where_it_has_to(qt_app):
    """Pairing halves the width it needs; it does not make wrapping optional.

    Answered from the width the row actually has: asked at one width and
    handed another, the second line lands below the bottom edge, present and
    invisible.
    """
    from am_segmentation_manager.gui.widgets.controls import FlowLayout
    from am_segmentation_manager.gui.widgets.image_panel import ImagePanel

    config = SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0)},
        class_labels={0: "background", 1: "GIS"},
    )
    panel = ImagePanel(config)
    layout = panel.controls.layout()
    assert isinstance(layout, FlowLayout)
    # A minimum with no width is what lets it wrap; a floor would force the
    # pane wider than the longest row instead.
    assert layout.minimumSize().width() == 0

    panel.resize(1900, 400)
    panel.show()
    qt_app.processEvents()
    wide = panel.controls.height()

    panel.resize(500, 400)
    qt_app.processEvents()

    assert panel.controls.height() > wide, "narrower means more lines"


def test_hiding_the_image_leaves_a_ground_the_size_of_the_frame(qt_app):
    """The proofreading canvas hides an item where Curate composes an array.

    Two mechanisms, one rule: with the acquisition gone something has to say
    where the frame ends, or a label a row of pixels short along an edge is
    indistinguishable from one that runs to it. The canvas keeps the label as
    its own layer so it can be painted into, and that layer's background is
    transparent -- so the ground has to be an item under it rather than a fill
    inside it.
    """
    import numpy as np

    from am_segmentation_manager.display import HIDDEN_IMAGE_GREY
    from am_segmentation_manager.gui.widgets.image_view import ImageView

    view = ImageView()
    try:
        view.set_array(np.zeros((40, 60), dtype=np.uint8))
        backdrop = view._backdrop

        assert not backdrop.isVisible(), "nothing to stand in for while it shows"

        view.set_image_shown(False)

        assert backdrop.isVisible()
        assert backdrop.rect() == view._item.boundingRect(), (
            "the ground is the frame, so its edge is the frame's edge"
        )
        assert backdrop.brush().color().getRgb()[:3] == HIDDEN_IMAGE_GREY
        assert backdrop.zValue() < view._item.zValue(), "under the picture"

        view.set_image_shown(True)

        assert not backdrop.isVisible()
    finally:
        destroy(view)


# ---------------------------------------------------------------------------
# The keyboard diagram
# ---------------------------------------------------------------------------

def test_the_list_opens_with_a_keyboard(window):
    """Which of the keys under your hands do anything here is a question a
    picture answers faster than a list."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE

    help_text = window._keys.help_html(TAB_CURATE)

    assert help_text.startswith("<p><img src=\"data:image/png;base64,")


def test_a_tab_with_no_keys_of_its_own_gets_no_keyboard(window):
    """An almost empty keyboard says nothing the sentence beneath it does not."""
    help_text = window._keys.help_html(KEYLESS_TAB)

    assert "<img" not in help_text
    assert "no keys of its own" in help_text


def test_a_chord_lights_its_modifiers_too():
    """Ctrl+Shift+Z is three keys somebody has to find, and a diagram showing
    only the letter would be saying the shortcut is one."""
    from am_segmentation_manager.gui.widgets.shortcuts import Binding, _keys_of

    def binding(key: str) -> Binding:
        return Binding(key, "", lambda: None, "any", "group")

    assert _keys_of([binding("Ctrl+Shift+Z")]) == {"Ctrl", "Shift", "Z"}
    assert _keys_of([binding("Left"), binding("Ctrl+S")]) == {"Left", "Ctrl", "S"}


def test_a_key_held_for_a_gesture_is_lit_on_the_diagram(window):
    """Space pans and Shift resizes the brush, and neither is a binding: the
    diagram left both dark while the table beside it named them."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_LABEL, _diagram_keys

    lit = _diagram_keys(window._keys.bindings, TAB_LABEL)

    assert {"Space", "Shift"} <= lit


def test_the_keyboard_picks_out_the_keys_it_is_given(qt_app):
    """Drawn from the bindings, so a key added to the window appears here
    without anything else being edited."""
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui.widgets.keyboard import keyboard_diagram
    from am_segmentation_manager.gui.widgets.theme import active_accent, apply_theme

    stylesheet, palette = qt_app.styleSheet(), qt_app.palette()
    apply_theme(qt_app)
    try:
        plain = keyboard_diagram(set()).toImage()
        marked = keyboard_diagram({"G"}).toImage()

        assert plain.size() == marked.size(), "one key does not change the shape"
        assert plain != marked
        accent = QColor(active_accent().colour).rgb()
        assert not any(
            plain.pixel(x, y) == accent
            for x in range(plain.width())
            for y in range(plain.height())
        )
        assert any(
            marked.pixel(x, y) == accent
            for x in range(marked.width())
            for y in range(marked.height())
        )
    finally:
        qt_app.setStyleSheet(stylesheet)
        qt_app.setPalette(palette)


def test_every_bound_key_has_somewhere_to_be_drawn(window):
    """A key the diagram has no cap for is one the reader is told about in the
    table and cannot find on the picture."""
    from am_segmentation_manager.gui.widgets.keyboard import KEY_ROWS
    from am_segmentation_manager.gui.widgets.shortcuts import _keys_of

    drawable = {name for row in KEY_ROWS for name in row}
    drawable |= {"Up", "Down", "Left", "Right"}

    assert _keys_of(window._keys.bindings) <= drawable


@pytest.mark.parametrize("passed", [True, False], ids=["given", "the user's"])
def test_the_window_opens_in_the_accent_its_settings_hold(
    qt_app, settings, monkeypatch, passed
):
    """Read before the window is built, from the same store the window then
    keeps -- including the user's own, where none is given."""
    from am_segmentation_manager.gui import app as gui_app
    from am_segmentation_manager.gui import preferences
    from am_segmentation_manager.gui.widgets import main_window, theme

    monkeypatch.setattr(gui_app, "install_error_reporting", lambda: None)
    monkeypatch.setattr(gui_app, "_apply_icon", lambda application: None)
    monkeypatch.setattr(main_window, "user_settings", lambda: settings)
    preferences.write(settings, {"accent": "blue"})
    style, palette, previous = qt_app.styleSheet(), qt_app.palette(), theme.active_accent()

    _application, window = gui_app.build_window(settings=settings if passed else None)
    try:
        # build_window queues bring_to_front; left queued, it would run on the
        # destroyed window and raise in a later test's event loop.
        qt_app.processEvents()
        assert theme.active_accent() == theme.ACCENTS["blue"]
        assert theme.ACCENTS["blue"].colour in qt_app.styleSheet()
        assert window.settings is settings
    finally:
        destroy(window)
        theme.apply_theme(qt_app, previous)
        qt_app.setStyleSheet(style)
        qt_app.setPalette(palette)


def test_starting_on_a_folder_shows_it_in_the_scan_box_and_scans_it(
    qt_app, settings, project, monkeypatch
):
    """`am-seg gui <folder>` opens the window through app.build_window, which
    no other test here goes through -- so a panel it names can move without
    anything else noticing. The event loop is not started: nothing would end it."""
    from am_segmentation_manager.gui import app as gui_app

    monkeypatch.setattr(gui_app, "install_error_reporting", lambda: None)
    monkeypatch.setattr(gui_app, "_apply_icon", lambda application: None)
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.theme.apply_theme",
        lambda application, accent=None: None,
    )

    _application, window = gui_app.build_window(project, settings=settings)
    try:
        assert window.scan_bar.path_edit.text() == str(project)

        settle(
            qt_app, lambda: window.controller.start_path is not None,
            what="the scan queued on startup",
        )

        assert window.controller.project.name == "test-project"
    finally:
        destroy(window)


def test_the_window_names_itself_to_the_desktop_as_it_opens(qt_app, settings, monkeypatch):
    """The menu bar's name before the application exists, as AppKit reads it
    only then; the desktop entry a Linux launcher writes, which is how a
    Wayland desktop finds the window's icon; and the ID a Windows shortcut
    carries, which is how the taskbar ties the window to its pin."""
    from qtpy.QtGui import QGuiApplication

    from am_segmentation_manager.gui import app as gui_app
    from am_segmentation_manager.identity import APPLICATION_ID, PRODUCT_NAME

    named: list[str] = []
    monkeypatch.setattr(gui_app, "install_error_reporting", lambda: None)
    monkeypatch.setattr(gui_app, "_apply_icon", lambda application: None)
    monkeypatch.setattr(gui_app, "set_bundle_name", lambda name: named.append(name) or True)
    identified: list[str] = []
    monkeypatch.setattr(
        gui_app, "set_app_user_model_id", lambda name: identified.append(name) or True
    )
    monkeypatch.setattr(
        "am_segmentation_manager.gui.widgets.theme.apply_theme",
        lambda application, accent=None: None,
    )
    previous = QGuiApplication.desktopFileName()

    _application, window = gui_app.build_window(settings=settings)
    try:
        qt_app.processEvents()
        assert named == [PRODUCT_NAME]
        assert identified == [APPLICATION_ID]
        assert QGuiApplication.desktopFileName() == APPLICATION_ID
    finally:
        destroy(window)
        QGuiApplication.setDesktopFileName(previous)


def test_the_window_hands_what_a_save_wrote_to_the_export_tab(window, monkeypatch):
    """Its measurements are keyed by image, so forgetting the label states is not enough."""
    from am_segmentation_manager.gui.label_controller import SavedLabels

    forgotten: list[tuple[Path, ...]] = []
    monkeypatch.setattr(
        window.export_controller, "forget_labels",
        lambda paths: forgotten.append(tuple(paths)) or 0,
    )
    label = Path("sem_labels") / "image.tif"

    window._labels_saved(SavedLabels(headline="wrote 1 label", written=(label,)))

    assert forgotten == [(label,)]


def test_a_series_name_too_long_for_its_box_ends_in_an_ellipsis(qt_app):
    """Qt clipped the shown entry, which read "Adaptive Polishir" with nothing to
    say the end was missing."""
    from am_segmentation_manager.gui.widgets.controls import ElidingCombo

    combo = ElidingCombo()
    combo.addItem("Adaptive Polishing")
    combo.resize(300, 26)
    assert combo.shown_text() == "Adaptive Polishing"

    combo.resize(70, 26)
    shown = combo.shown_text()
    assert shown.endswith("…")
    assert "Adaptive Polishing".startswith(shown[:-1])


# ---------------------------------------------------------------------------
# The Families tab
# ---------------------------------------------------------------------------

def _families_panel(root: Path, *, second: bool = False, lamella: bool = False):
    """Return a Families tab over one project, or two; *lamella* marks lam_010 as in view."""
    from am_segmentation_manager.gui.families_controller import FamiliesController
    from am_segmentation_manager.gui.widgets.families_panel import FamiliesPanel
    from am_segmentation_manager.scanner import Scanner

    names = ("20260930_adaptive_milling", "20261001_adaptive_milling")
    for name in names[: 2 if second else 1]:
        project = root / "batch" / name
        for directory in ("sem", "sem_fi4", "sem-5kV", "FIB"):
            (project / "lam_010" / directory).mkdir(parents=True)
            (project / "lam_010" / directory / "img_0004.tif").write_bytes(b"")
        (project / "lam_011" / "sem").mkdir(parents=True)
        (project / "lam_011" / "sem" / "img_0004.tif").write_bytes(b"")
        (project / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    projects = Scanner(root).build_index().all_projects()
    panel = FamiliesPanel(FamiliesController())
    shown = projects[0].lamellae[1] if lamella else None
    panel.show_scope(projects, shown)
    return panel


def _tree_text(item) -> list:
    """Return an item's first column and its children's, as nested lists."""
    return [item.text(0), [_tree_text(item.child(i)) for i in range(item.childCount())]]


def _row(panel, condition: str):
    """Return *condition*'s row, not the heading of a family it names."""
    [item] = [
        item for item in panel.tree.findItems(
            condition, Qt.MatchFlag.MatchStartsWith | Qt.MatchFlag.MatchRecursive, 0
        )
        if item.data(0, Qt.ItemDataRole.UserRole) == condition
    ]
    return item


def test_the_families_heading_says_what_a_family_is_under_its_own_line(qt_app, tmp_path: Path):
    """An eliding label makes its tooltip the line it shortened, which hid the explanation."""
    from am_segmentation_manager.gui.widgets.families_panel import FAMILIES_HINT

    panel = _families_panel(tmp_path)
    try:
        assert panel.title.toolTip() == f"{panel.title.full_text()}\n{FAMILIES_HINT}"
    finally:
        destroy(panel)


def test_the_families_tab_shows_the_projects_families(qt_app, tmp_path: Path):
    panel = _families_panel(tmp_path)
    try:
        tree = panel.tree
        assert [_tree_text(tree.topLevelItem(i)) for i in range(tree.topLevelItemCount())] == [
            ["SEM", [["sem", [["sem", []], ["sem_fi4", []]]], ["sem-5kV", [["sem-5kV", []]]]]],
            ["FIB", [["FIB", [["FIB", []]]]]],
        ]
        assert _row(panel, "sem").toolTip(0) == "In 2 stage(s) of the project; primary (default)"
        assert panel.title.text().endswith("20260930_adaptive_milling")
        assert panel.status.isHidden()
    finally:
        destroy(panel)


def test_a_family_of_several_stars_its_primary(qt_app, tmp_path: Path):
    from am_segmentation_manager.gui.widgets.common import primary_star

    panel = _families_panel(tmp_path)
    star = primary_star().pixmap(16, 16).toImage()
    try:
        assert _row(panel, "sem").icon(0).pixmap(16, 16).toImage() == star
        assert _row(panel, "sem_fi4").icon(0).pixmap(16, 16).toImage() != star
        assert _row(panel, "sem-5kV").icon(0).isNull(), "a family of one has nothing to choose"
    finally:
        destroy(panel)


def test_conditions_the_lamella_in_view_lacks_are_greyed(qt_app, tmp_path: Path):
    """lam_011 holds only sem: the others are still listed, a correction reaching
    every lamella of the project."""
    from qtpy.QtGui import QColor

    from am_segmentation_manager.gui.widgets import theme

    panel = _families_panel(tmp_path, lamella=True)
    try:
        assert _row(panel, "sem").text(0) == "sem"
        assert _row(panel, "sem_fi4").text(0) == "sem_fi4 (not here)"
        assert _row(panel, "sem_fi4").foreground(0).color().name() == QColor(theme.MUTED).name()
        assert "not in this lamella" in _row(panel, "sem_fi4").toolTip(0)
    finally:
        destroy(panel)


def test_the_families_tab_asks_for_one_project_over_a_folder(qt_app, tmp_path: Path):
    panel = _families_panel(tmp_path, second=True)
    try:
        assert panel.tree.topLevelItemCount() == 0
        assert panel.status.text() == "Select one project to show its families."
    finally:
        destroy(panel)


def _choose(panel, condition: str, family: str) -> None:
    """Select *condition* in the families tree and choose *family* for it."""
    panel.tree.setCurrentItem(_row(panel, condition))
    combo = panel.family_combo
    index = next(row for row in range(combo.count()) if combo.itemData(row) == family)
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def test_changes_are_held_until_editing_is_on(qt_app, tmp_path: Path):
    panel = _families_panel(tmp_path)
    try:
        _choose(panel, "sem_fi4", "sem-5kV")
        assert not panel.save_button.isEnabled(), "a change is held, not saved"

        panel.set_editing_enabled(True)

        assert panel.family_combo.isEnabled()
        assert panel.save_button.isEnabled()
    finally:
        destroy(panel)


def test_a_family_chosen_is_shown_and_written_on_save(qt_app, tmp_path: Path):
    panel = _families_panel(tmp_path)
    saved: list[bool] = []
    panel.families_changed.connect(lambda: saved.append(True))
    try:
        panel.set_editing_enabled(True)
        _choose(panel, "sem_fi4", "sem-5kV")
        assert _row(panel, "sem_fi4").text(0) == "sem_fi4 (moved, not saved)"

        panel.preview_button.click()
        assert "sem_fi4: family as named -> sem-5kV" in panel.message.text()
        panel.save_button.click()

        project = tmp_path / "batch" / "20260930_adaptive_milling"
        assert "sem_fi4: sem-5kV" in (project / "AM_families.yaml").read_text()
        assert saved == [True], "the window reads the folder again"
        assert panel.message.text().startswith("Saved.")
    finally:
        destroy(panel)


def test_waiting_label_edits_can_stop_a_families_save(qt_app, tmp_path: Path):
    panel = _families_panel(tmp_path)
    panel.may_change_families = lambda project: False
    try:
        panel.set_editing_enabled(True)
        _choose(panel, "sem_fi4", "sem-5kV")

        panel.save_button.click()

        assert panel.message.text() == "Not saved"
        assert not (tmp_path / "batch" / "20260930_adaptive_milling" / "AM_families.yaml").exists()
    finally:
        destroy(panel)


@pytest.mark.parametrize("answer, written", [("No", False), ("Yes", True)])
def test_leaving_a_corrected_group_is_asked_first(qt_app, tmp_path: Path, monkeypatch, answer, written):
    stage = tmp_path / "batch" / "20260930_adaptive_milling" / "lam_010"
    (stage / "sem_labels").mkdir(parents=True)
    (stage / "sem_labels" / "img_0004.tif").write_bytes(b"")
    (stage / "AM_state.yaml").write_text(
        "version: 1\nstate:\n  sem_labels/img_0004.tif: corrected\n", encoding="utf-8"
    )
    panel = _families_panel(tmp_path)
    asked: list[str] = []

    def question(parent, title, text, *args):
        asked.append(text)
        return getattr(QMessageBox.StandardButton, answer)

    from qtpy.QtWidgets import QInputDialog

    from am_segmentation_manager.gui.widgets.families_panel import NEW_FAMILY

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("sem-fi4", True))
    try:
        panel.set_editing_enabled(True)
        _choose(panel, "sem_fi4", NEW_FAMILY)
        assert _row(panel, "sem_fi4").text(0).endswith("not saved)"), "the new family is named"

        panel.save_button.click()

        assert len(asked) == 1 and asked[0].startswith("1 image(s) would leave")
        assert (stage.parent / "AM_families.yaml").exists() is written
    finally:
        destroy(panel)


def test_applies_to_is_offered_only_with_a_lamella_in_view(qt_app, tmp_path: Path):
    panel = _families_panel(tmp_path)
    try:
        assert panel.applies_row.isHidden()
    finally:
        destroy(panel)


def test_a_change_applied_to_one_lamella_is_written_to_its_section(qt_app, tmp_path: Path):
    from am_segmentation_manager.families import read_families

    panel = _families_panel(tmp_path, lamella=True)
    try:
        assert not panel.applies_row.isHidden()
        assert [panel.applies_combo.itemText(i) for i in range(2)] == [
            "Whole project", "lam_011 only",
        ]
        panel.set_editing_enabled(True)
        panel.applies_combo.setCurrentIndex(1)
        panel.applies_combo.activated.emit(1)
        _choose(panel, "sem", "sem-5kV")
        assert not panel.applies_combo.isEnabled(), "changes belong to the choice held"

        panel.save_button.click()

        on_file = read_families(tmp_path / "batch" / "20260930_adaptive_milling")
        assert on_file.project.families == {}
        assert dict(on_file.lamellae["lam_011"].families) == {"sem": "sem-5kV"}
    finally:
        destroy(panel)


def test_add_data_holds_no_families(qt_app, tmp_path: Path):
    """They moved beside the images, on Curate."""
    from am_segmentation_manager.gui.metadata_controller import MetadataController

    panel = MetadataPanel(MetadataController())
    try:
        assert not hasattr(panel, "families_tree")
        assert "AM_families.yaml" not in panel.enable.toolTip()
    finally:
        destroy(panel)


# ---------------------------------------------------------------------------
# The Train tab
# ---------------------------------------------------------------------------

def test_curate_s_keys_are_off_on_the_train_tab(window):
    """A tab the window does not know by name falls back to Curate's keys, and
    an arrow key pressed in the name box would then step through frames."""
    from am_segmentation_manager.gui.widgets.shortcuts import TAB_CURATE, TAB_TRAIN

    window.tabs.setCurrentWidget(window.training_panel)

    assert window._current_tab() == TAB_TRAIN
    live = [
        binding for binding, shortcut in zip(window._keys.bindings, window._keys.shortcuts)
        if binding.tab == TAB_CURATE and shortcut.isEnabled()
    ]
    assert not live


def test_where_the_train_tab_is_pointed_is_remembered(qt_app, project: Path, settings, tmp_path):
    runs, weights = tmp_path / "runs", tmp_path / "base.pth"
    runs.mkdir()
    weights.write_bytes(b"weights")
    first = CurationWindow(CurationController(cache=FrameCache()), settings=settings)
    try:
        first.training_controller.training_directory = runs
        first.training_controller.base_weights = weights
        first.training_controller.device = "cuda:1"
        first.training_panel.remembered.emit()
    finally:
        destroy(first)

    second = CurationWindow(CurationController(cache=FrameCache()), settings=settings)
    try:
        assert second.training_controller.training_directory == runs
        assert second.training_controller.base_weights == weights
        assert second.training_controller.device == "cuda:1"
    finally:
        destroy(second)


def test_a_training_directory_that_has_gone_is_not_restored(qt_app, settings, tmp_path):
    settings.setValue("training/directory", str(tmp_path / "unmounted"))

    opened = CurationWindow(CurationController(cache=FrameCache()), settings=settings)
    try:
        assert opened.training_controller.training_directory is None
    finally:
        destroy(opened)


@pytest.mark.parametrize("splits", [1, 2], ids=["one split", "cross-validation"])
def test_a_training_export_is_offered_on_the_train_tab_when_it_is_one_split(
    window, inline_operations, tmp_path, monkeypatch, splits: int
):
    """A run trains on one split, so a cross-validation export is not offered."""
    output = tmp_path / "export"
    monkeypatch.setattr(window, "_confirm_dropping_corrected", lambda: True)
    monkeypatch.setattr(
        window.export_controller, "export_training",
        lambda directory, overwrite=False, on_progress=None: [
            (directory / "dataset_train.csv", directory / "dataset_val.csv")
        ] * splits,
    )

    window._write_training(str(output))

    prefix = window.export_controller.training.prefix
    offered = output / f"{prefix}_manifest.yaml" if splits == 1 else None
    assert window.training_controller.manifest == offered


def test_the_train_tab_checks_a_run_behind_the_busy_dialog(window, monkeypatch):
    """Opening every label is seconds over a real set, on the interface thread
    a window that looks hung."""
    ran = []
    monkeypatch.setattr(
        window.runner, "run", lambda description, work, done, **_: ran.append(description)
    )

    window.training_panel.run_in_background(lambda: None, lambda outcome: None, "Checking")

    assert ran == ["Checking"]


def test_the_scanned_folder_reaches_the_train_tab(window, project: Path):
    assert window.training_controller.collection == project


def test_starting_a_run_checks_it_off_the_interface_thread_and_then_asks(
    window, qt_app, tmp_path, monkeypatch
):
    """The whole way round: the tab, the window's worker thread, and back to the
    dialog that asks, with nothing written when the answer is no."""
    from am_segmentation_manager.training import preparation
    from test_training_process import export_of_placeholders

    monkeypatch.setattr(preparation, "label_report", lambda path, classes: (None, None))
    controller, panel = window.training_controller, window.training_panel
    monkeypatch.setattr(controller, "can_train", lambda: True)
    controller.manifest = export_of_placeholders(tmp_path)
    # The scanned folder is where the collection is, whatever the manifest says.
    controller.collection = tmp_path / "collection"
    controller.training_directory = tmp_path / "runs"
    controller.name = "round01"
    asked = []
    monkeypatch.setattr(
        panel, "_confirm", lambda title, lines, action: asked.append(lines) or False
    )

    panel.start()
    settle(qt_app, lambda: bool(asked), what="the check to come back")

    assert "1 training and 1 validation pair(s)" in "\n".join(asked[0])
    assert not window.runner.running()
    assert not (tmp_path / "runs").exists()
