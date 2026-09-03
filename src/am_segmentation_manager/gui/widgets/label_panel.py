"""The Label tab: one acquisition at full resolution, ready to be proofread.

Reads its own controller and draws from it, so the window wires this tab up
rather than running it. Nothing here touches the filesystem: an edit is held by
the controller's session until the window saves it.

The lamella is not chosen here. It is handed in from the Curate tab, so the
collection has one place where "which lamella am I on" is decided.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...config import BACKGROUND_CLASS
from ...display import (
    DEFAULT_ALPHA,
    DEFAULT_INTENSITY_RANGE,
    apply_intensity_range,
)
from ...layout import MODALITIES, SERIES_LABELS
from ...operations.label_measure import (
    DEFAULT_SPECK_PIXELS,
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
    PREDICTION_PIXEL_BLOCK,
)
from ..label_controller import LabelController
from ..version_controller import VersionController
from .common import LABEL_BADGES, chip_icon, fill_choice_combo, intensity_badge
from .controls import ElidingLabel, RangeSlider, SegmentedControl, ToggleSwitch
from .frame_strip import FrameTimeline
from .label_canvas import (
    DEFAULT_BRUSH_RADIUS,
    TOOL_BRUSH,
    TOOL_FILL,
    LabelCanvas,
)
from .theme import BADGE_STYLES, SWITCH_RUN_GAP, TEXT
from .versions_bar import DIFFERENCE_CLASS, VersionsBar

logger = logging.getLogger(__name__)

#: Said whenever anything is waiting, because an edit that exists only in this
#: window is one closing it would throw away.
UNSAVED_NOTE = "not yet written to disk"

#: The swatch drawn on each class's paint button.
SWATCH = 12

#: How long after the last stroke the specks are counted again. Long enough
#: that a pause between strokes is not one, short enough that a speck painted
#: out stops being ringed while the reviewer is still looking at it.
SPECK_RECOUNT_MS = 400

#: The keys that choose a class to paint, one per button in the order the
#: classes are shown.
#:
#: The row above the tools, so the hand that reaches D and F reaches these
#: without moving. Five because that is what the segmentation config declares;
#: a longer config leaves the rest without a key rather than borrowing one.
#: Named here, beside the buttons they press, and bound by the window.
CLASS_KEYS = ("Q", "W", "E", "R", "T")

#: The brush sizes the slider offers, in pixels across the widest part of the
#: stroke: size 1 is a single pixel, size 3 is three pixels wide, and size n
#: covers 2n-1. Sizes are *odd* because the brush is a disc centred on a pixel.
#:
#: The list is why the slider is not a plain range. It is 120 pixels wide, so a
#: linear 1-to-200 range moves in steps of two and cannot be set to a small odd
#: size at all -- which is exactly the end that matters, since the last
#: corrections along a boundary are made a pixel or two at a time. Every size up
#: to 39 is offered; above that the steps widen, because nobody needs to pick
#: 151 out from 149 on a slider this size.
BRUSH_WIDTHS = (
    *range(1, 40, 2),
    *range(43, 100, 4),
    *range(109, 200, 10),
    *range(219, 400, 20),
)


#: Side of the tool icons, in logical pixels.
ICON = 16


def _tool_pixmap() -> QPixmap:
    """Return a transparent square to draw a tool icon on."""
    pixmap = QPixmap(ICON, ICON)
    pixmap.fill(Qt.transparent)
    return pixmap


def brush_icon():
    """Return the brush's icon: the disc it paints.

    Drawn rather than shipped, so the window carries no image files. A disc
    because that is literally what the brush lays down -- the icon and the
    tool are the same shape.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TEXT))
        painter.drawEllipse(3, 3, ICON - 6, ICON - 6)
    finally:
        # Always: a QPainter outliving its paint device crashes the process
        # rather than reporting an error.
        painter.end()
    return chip_icon(pixmap)


def fill_icon():
    """Return the fill's icon: a drop, for the area one click floods.

    A drop rather than the usual bucket: a bucket needs a body, a handle and a
    pouring lip, and at sixteen pixels those merge into a blob. The pair reads
    as what each tool does -- the brush lays down a disc, the fill pours.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
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


def edit_icon():
    """Return the edit row's icon: a pencil, angled as one is held.

    Drawn like the tool icons and to the same weight, so the four marks on this
    bar read as one set. A pencil rather than a brush: the row it heads chooses
    *what* is written, where the brush and the fill choose how.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TEXT))
        # The body, then the tip: a quadrilateral running corner to corner and
        # a triangle closing it, rather than one polygon, so the tip stays a
        # clean point at sixteen pixels instead of a notch.
        painter.drawPolygon(
            QPolygonF([
                QPointF(11.4, 1.6), QPointF(14.4, 4.6),
                QPointF(6.6, 12.4), QPointF(3.6, 9.4),
            ])
        )
        painter.drawPolygon(
            QPolygonF([QPointF(2.6, 13.4), QPointF(5.8, 13.2), QPointF(2.8, 10.2)])
        )
    finally:
        painter.end()
    return chip_icon(pixmap)


def lock_icon(locked: bool = True):
    """Return the lock row's icon: a padlock, its shackle open or closed.

    The shackle is the whole of the difference between the two states, so it is
    drawn as a stroke rather than filled -- at sixteen pixels a filled arch and
    an open one are the same blob.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(TEXT))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        # Closed: the shackle sits over the body. Open: it is lifted and turned
        # aside, which is the shape a padlock actually makes.
        left = 4.6 if locked else 7.0
        painter.drawArc(QRectF(left, 2.0, 6.8, 7.6), 0, 180 * 16)
        painter.drawLine(QPointF(left, 5.8), QPointF(left, 8.4))
        painter.drawLine(
            QPointF(left + 6.8, 5.8), QPointF(left + 6.8, 8.4 if locked else 6.6)
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TEXT))
        painter.drawRoundedRect(QRectF(3.0, 8.0, 10.0, 6.6), 1.4, 1.4)
    finally:
        painter.end()
    return chip_icon(pixmap)


def save_icon():
    """Return the save button's icon: a disk, its shutter and label punched out.

    The one mark on this bar that stands for writing to the filesystem rather
    than for a tool, so it is the familiar shape rather than an invention: at
    sixteen pixels a reader recognises the outline before reading the caption.
    Both openings are inset from the body's edges -- reaching them cuts the
    body into two columns joined by a band, which draws an H.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
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
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.drawRect(QRectF(5.6, 3.0, 4.4, 3.2))
        painter.drawRect(QRectF(4.6, 8.6, 6.8, 3.4))
    finally:
        # Always: a QPainter outliving its paint device crashes the process
        # rather than reporting an error.
        painter.end()
    return chip_icon(pixmap)


def _history_icon(forwards: bool):
    """Return an arrow curving over and back on itself, pointing left or right.

    One drawing for undo and redo, mirrored about the centre line: they are one
    gesture in two directions, and two arrows drawn by hand at this size never
    quite match. The head sits at the end of the arc and points across it, an
    arc and a head that overlap being a blob at sixteen pixels.
    """
    pixmap = _tool_pixmap()
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        if forwards:
            painter.translate(ICON, 0)
            painter.scale(-1.0, 1.0)
        pen = QPen(QColor(TEXT))
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        # The top half of an ellipse: over the top and down both sides.
        painter.drawArc(QRectF(3.4, 4.6, 9.6, 9.0), 0, 180 * 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(TEXT))
        # On the left leg, pointing down: where the stroke is going.
        painter.drawPolygon(
            QPolygonF([QPointF(0.9, 8.6), QPointF(5.9, 8.6), QPointF(3.4, 13.4)])
        )
    finally:
        painter.end()
    return chip_icon(pixmap)


def undo_icon():
    """Return the undo button's icon: an arrow curving back the way it came."""
    return _history_icon(forwards=False)


def redo_icon():
    """Return the redo button's icon: the undo arrow, mirrored."""
    return _history_icon(forwards=True)


def _row_header(icon, caption: str, hint: str) -> QWidget:
    """Return a grid row's heading: a drawn mark beside its name.

    A widget rather than a QLabel, because a QLabel shows either a pixmap or
    text and this needs both -- and the mark is what ties the row to the tool
    buttons beside it, which are drawn the same way.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    mark = QLabel()
    mark.setPixmap(icon.pixmap(ICON, ICON))
    text = QLabel(caption)
    for widget in (holder, mark, text):
        widget.setToolTip(hint)
    row.addWidget(mark)
    row.addWidget(text)
    return holder


def _equalise(buttons) -> None:
    """Give every one of *buttons* the width of the widest, and no more."""
    widest = max(button.sizeHint().width() for button in buttons)
    for button in buttons:
        button.setMinimumWidth(widest)


def _lock_button(caption: str, *, locked: bool = True) -> QToolButton:
    """Return one cell of the lock row, built like the row above it.

    Same construction as an edit button -- icon beside text, expanding to its
    column -- so the two rows line up as a table rather than as two bars that
    happen to be stacked.

    Args:
        locked: Which padlock to start with. A class button starts unlocked and
            its shackle is kept in step by :meth:`LabelPanel._locks_changed`;
            the "All" button is not a state, so it keeps the closed padlock as
            the glyph for locking.
    """
    button = QToolButton()
    button.setText(caption)
    button.setIcon(lock_icon(locked))
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return button


def _width_position(width: int) -> int:
    """Return the slider position offering *width*, or the nearest below it."""
    return max(
        (index for index, offered in enumerate(BRUSH_WIDTHS) if offered <= width),
        default=0,
    )


class LabelPanel(QWidget):
    """The proofreading view: controls, the canvas, and the frame timeline."""

    #: Emitted when the frames around this one need preparing. The window owns
    #: the loader thread, so the panel asks rather than reads.
    loads_needed = Signal()

    #: Emitted when frames in this series have specks nobody has counted, so
    #: the window can put the reading on its worker thread. Not done here: a
    #: sweep of a lamella is seconds of label reading, and the panel draws.
    speck_counts_needed = Signal()

    #: Emitted when the reviewer asks to write this lamella's edits. The window
    #: owns everything that touches the filesystem.
    save_requested = Signal()
    #: Asked when the reviewer wants the chosen earlier version put back. The
    #: window handles it, restoring being a write and so its to confirm.
    restore_requested = Signal()

    def __init__(
        self,
        controller: LabelController,
        versions: VersionController | None = None,
    ) -> None:
        """Build the tab against *controller*.

        Args:
            controller: What is being proofread.
            versions: The kept versions of the label on screen. Injected so a
                test can hand in one attributed to a known author.
        """
        super().__init__()
        self.controller = controller
        self.versions_controller = (
            versions if versions is not None else VersionController()
        )
        #: The intensities shown as black and as white while proofreading.
        #: A property of the *view*, kept across frames on purpose --
        #: comparing one frame with the next is what it is for -- and never
        #: reaching the label being painted. Separate from the curation
        #: panes' own: reviewing and proofreading are looked at differently
        #: and a shared setting would have each change the other's picture.
        self._intensity_range = DEFAULT_INTENSITY_RANGE
        #: The classes to offer, in the order they are shown and keyed.
        self._classes = list(controller.config.classes)
        #: Condition names from the project metadata; see
        #: :meth:`set_variant_descriptions`.
        self._variant_descriptions: dict[str, str] = {}
        #: The view to put back when the speck filter is switched off.
        self._view_before: tuple[bool, bool, int] | None = None
        #: Which speck the stepping key is on.
        self._speck_position = 0
        #: Whether an earlier version is on the canvas in place of the label.
        self._comparing = False
        #: Whether the canvas is marking where the two disagree.
        self._showing_differences = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        layout.addLayout(self._build_controls())

        self.canvas = LabelCanvas(controller.config)
        self.canvas.edited.connect(self._stroke_finished)
        self.canvas.brush_stepped.connect(self.step_brush)
        self.canvas.compare_held.connect(self._compare_key_held)
        layout.addWidget(self.canvas, stretch=1)

        self.versions = VersionsBar()
        self.versions.selected.connect(self._version_chosen)
        self.versions.comparing.connect(self._set_comparing)
        self.versions.differences.connect(self._set_differences)
        self.versions.restore_requested.connect(self.restore_requested.emit)
        layout.addWidget(self.versions)

        layout.addLayout(self._build_tools())
        layout.addLayout(self._build_timeline())

        self._speck_timer = QTimer(self)
        self._speck_timer.setSingleShot(True)
        self._speck_timer.timeout.connect(self._refresh_specks)
        #: Held off separately from the ring recount above: that one answers
        #: for the label on screen and is cheap, this one reads the whole
        #: series and must not start while the handle is still moving.
        self._resweep_timer = QTimer(self)
        self._resweep_timer.setSingleShot(True)
        self._resweep_timer.timeout.connect(self.speck_counts_needed.emit)

        # Every control here has a shortcut, and none of them needs the
        # keyboard: what does is the canvas, whose space-to-pan is read as a
        # key press. A radio button that kept focus after being clicked would
        # take that space and toggle itself instead.
        for widget in self.findChildren(QWidget):
            if widget is not self.canvas:
                widget.setFocusPolicy(Qt.NoFocus)

    # -- construction ------------------------------------------------------

    def _build_controls(self) -> QHBoxLayout:
        """Build the row of view controls."""
        controls = QHBoxLayout()

        self.modality_control = SegmentedControl([
            (
                modality, modality,
                f"Proofread the {modality} images of this lamella\nShortcut: {key}",
            )
            for modality, key in zip(MODALITIES, ("1", "2"))
        ])
        self.modality_control.chosen.connect(self._modality_chosen)
        #: The segments, so a caller can reach one at a time.
        self.modality_buttons = self.modality_control.buttons
        controls.addWidget(self.modality_control)
        # Chosen without emitting: the panel is still being built, and the
        # handler redraws a timeline that does not exist yet.
        self.modality_control.set_current(MODALITIES[0])

        # Beside the beam buttons and before the variant, in the same order and
        # for the same reasons as the curation view: the series decides which
        # frames there are to review, the variant only which file of one is
        # drawn. Annotating a lamella means annotating one series of it.
        self.series_label = QLabel("Series")
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(140)
        self.series_combo.setToolTip(
            "Which series to proofread. Each is its own list of frames, "
            "reviewed one at a time."
        )
        self.series_label.setToolTip(self.series_combo.toolTip())
        self.series_combo.currentIndexChanged.connect(self._series_chosen)
        controls.addSpacing(12)
        controls.addWidget(self.series_label)
        controls.addWidget(self.series_combo)

        # A dropdown rather than buttons, as in the curation view: variant names
        # run long, and most projects have none to choose between.
        self.variant_label = QLabel("Variant")
        self.variant_combo = QComboBox()
        self.variant_combo.setMinimumWidth(110)
        self.variant_combo.setToolTip(
            "Which imaging condition of this acquisition to proofread on.\n"
            "They share one label, so an edit reaches all of them."
        )
        self.variant_label.setToolTip(self.variant_combo.toolTip())
        self.variant_combo.currentIndexChanged.connect(self._variant_chosen)
        controls.addSpacing(12)
        controls.addWidget(self.variant_label)
        controls.addWidget(self.variant_combo)

        controls.addSpacing(20)
        self.overlay_check = ToggleSwitch("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip("Draw the label over the image\nShortcut: V")
        self.overlay_check.toggled.connect(self._overlay_toggled)
        controls.addWidget(self.overlay_check)

        controls.addSpacing(SWITCH_RUN_GAP)
        self.image_check = ToggleSwitch("Show image")
        self.image_check.setChecked(True)
        self.image_check.setToolTip(
            "Draw the acquisition under the label, or leave the label alone "
            "against black.\nOver black is how an annotation of a few pixels "
            "is found."
        )
        self.image_check.toggled.connect(self._image_toggled)
        controls.addWidget(self.image_check)

        controls.addSpacing(SWITCH_RUN_GAP)
        opacity_label = QLabel("Opacity")
        controls.addWidget(opacity_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(DEFAULT_ALPHA * 100))
        self.alpha_slider.setFixedWidth(120)
        self.alpha_slider.setToolTip("How strongly the label colour is drawn")
        opacity_label.setToolTip(self.alpha_slider.toolTip())
        self.alpha_slider.valueChanged.connect(self._alpha_moved)
        controls.addWidget(self.alpha_slider)

        controls.addSpacing(SWITCH_RUN_GAP)
        low, high = DEFAULT_INTENSITY_RANGE
        range_hint = (
            "The intensities shown as black and as white; outside them is "
            "clipped.\nChanges nothing on disk and reaches no label. "
            "Double-click a handle to put it back."
        )
        range_label = QLabel("Minimum / maximum")
        range_label.setToolTip(range_hint)
        controls.addWidget(range_label)
        self.intensity_slider = RangeSlider(0, 255, low, high)
        self.intensity_slider.setToolTip(range_hint)
        self.intensity_slider.range_changed.connect(self._range_moved)
        controls.addWidget(self.intensity_slider)

        controls.addSpacing(20)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Scale the image to the pane\nShortcut: 0")
        self.fit_button.clicked.connect(self.canvas_fit)
        controls.addWidget(self.fit_button)

        controls.addSpacing(20)
        self.speck_check = ToggleSwitch("Find specks")
        self.speck_check.setToolTip(
            "Ring components of a few pixels -- a stray annotation, or a hole "
            "inside one -- and count them on the timeline."
            "\nSets the view that shows them, and puts it back after."
            "\nShortcut: X; N steps to the next"
        )
        self.speck_check.toggled.connect(self._finding_toggled)
        controls.addWidget(self.speck_check)

        controls.addSpacing(SWITCH_RUN_GAP)
        self.speck_slider = QSlider(Qt.Horizontal)
        self.speck_slider.setRange(MIN_SPECK_PIXELS, MAX_SPECK_PIXELS)
        self.speck_slider.setValue(self.controller.speck_size)
        self.speck_slider.setFixedWidth(90)
        self.speck_slider.setToolTip(
            "Largest component still ringed.\n"
            f"Starts at {DEFAULT_SPECK_PIXELS}; {PREDICTION_PIXEL_BLOCK} is "
            "one pixel of a prediction, so anything smaller was painted by hand."
        )
        self.speck_slider.valueChanged.connect(self._speck_size_changed)
        controls.addWidget(self.speck_slider)

        self._show_speck_controls(False)

        controls.addStretch()

        # Where the reviewer is, at the end of the row that decides it: the
        # modality buttons on the left of this line are half of what it reports.
        # On a row of its own it cost the canvas a row of height to say one
        # short sentence, and it sat under the buttons rather than near the
        # navigation.
        # Elided rather than clipped: the line runs long on a project with
        # timestamped stage names, and this row has other controls in it.
        self.status = ElidingLabel()
        self.status.setStyleSheet(BADGE_STYLES["info"])
        self.status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls.addWidget(self.status, stretch=1)
        return controls

    def _build_tools(self) -> QHBoxLayout:
        """Build the bottom bar: the tools, the classes, and what to do next.

        Columns of two rows rather than one long row, because the controls pair
        up: the brush with its size, a class with its lock, an edit with the
        save that writes it. The paint buttons carry the class colours, which is
        what makes a separate legend unnecessary -- the button you press to
        paint a class is also the thing that tells you what colour it is.
        """
        tools = QHBoxLayout()
        tools.addLayout(self._build_tool_buttons())
        tools.addSpacing(20)
        tools.addLayout(self._build_class_table())
        tools.addSpacing(20)
        tools.addLayout(self._build_actions())
        tools.addStretch()

        # Both say something about the label being edited, so they belong with
        # the tools that edit it -- and in the width this row has spare, rather
        # than each holding a row of the canvas's height for one short line.
        self.badges = QLabel("")
        self.badges.setTextFormat(Qt.RichText)
        self.badges.setWordWrap(True)
        self.badges.setVisible(False)
        tools.addWidget(self.badges)

        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet(BADGE_STYLES["warning"])
        tools.addWidget(self.pending_label)
        return tools

    def _build_tool_buttons(self) -> QGridLayout:
        """Build the two tools, stacked, with the brush size beside the brush."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)

        self.tool_buttons: dict[str, QToolButton] = {}
        group = QButtonGroup(self)
        for row, (tool, caption, key, hint, icon) in enumerate((
            (TOOL_BRUSH, "Brush", "D", "Draw where you drag", brush_icon()),
            (
                TOOL_FILL, "Fill", "F",
                "Flood the area of one class you click in", fill_icon(),
            ),
        )):
            button = QToolButton()
            button.setText(caption)
            button.setIcon(icon)
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setCheckable(True)
            button.setToolTip(f"{hint}\nShortcut: {key}")
            button.toggled.connect(
                lambda checked, t=tool: checked and self._tool_chosen(t)
            )
            group.addButton(button)
            grid.addWidget(button, row, 0)
            self.tool_buttons[tool] = button

        self.brush_slider = QSlider(Qt.Horizontal)
        # One step of the slider is one entry in BRUSH_WIDTHS, not one pixel:
        # see why there.
        self.brush_slider.setRange(0, len(BRUSH_WIDTHS) - 1)
        self.brush_slider.setValue(_width_position(2 * DEFAULT_BRUSH_RADIUS + 1))
        self.brush_slider.setFixedWidth(120)
        self.brush_slider.setToolTip(
            "How many pixels across the brush paints\n"
            "Shortcuts: [ and ], or Shift and the scroll wheel"
        )
        self.brush_slider.valueChanged.connect(self._brush_resized)
        self.brush_value = QLabel("")
        self.brush_value.setMinimumWidth(38)
        self.brush_value.setToolTip(self.brush_slider.toolTip())
        # In the brush's own row: the size is the brush's, and means nothing to
        # the fill.
        grid.addWidget(self.brush_slider, 0, 1)
        grid.addWidget(self.brush_value, 0, 2)
        self._brush_resized(self.brush_slider.value())

        # The two are a pair and read as one: the shorter caption is widened
        # to the longer rather than both being left to their own text.
        _equalise(self.tool_buttons.values())

        brush_button = self.tool_buttons[TOOL_BRUSH]
        brush_button.blockSignals(True)
        brush_button.setChecked(True)
        brush_button.blockSignals(False)
        return grid

    def _build_class_table(self) -> QGridLayout:
        """Build the classes to edit with, above the same classes' locks.

        One column per class, so a lock sits under the class it protects and
        the two rows read as one table. Every cell is the same size: the
        columns share the width equally and each button fills its own, so a
        class with a long name does not make its lock a different shape from
        its neighbour's. Background's column carries the lock-everything button
        instead of a lock of its own: painting background is how a stroke
        erases, so locking it would take the eraser away.
        """
        self.class_grid = grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        # "Edit" rather than "Paint": the row chooses what a stroke writes, and
        # one of its buttons erases -- which is not painting by any reading.
        grid.addWidget(
            _row_header(edit_icon(), "Edit", "Which class the brush and the fill write"),
            0, 0,
        )
        grid.addWidget(
            _row_header(
                lock_icon(), "Lock",
                "A locked class cannot be painted over by any tool",
            ),
            1, 0,
        )

        self.paint_buttons: dict[int, QToolButton] = {}
        self.lock_buttons: dict[int, QToolButton] = {}
        paint_group = QButtonGroup(self)
        for column, class_index in enumerate(self._classes, start=1):
            name = self.controller.config.name(class_index)
            erasing = class_index == BACKGROUND_CLASS

            button = QToolButton()
            button.setText(f"{name} (erase)" if erasing else name)
            button.setIcon(self._swatch(class_index))
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setCheckable(True)
            hint = (
                "Paint background, which is how a stroke erases" if erasing
                else f"Paint {name}"
            )
            if column <= len(CLASS_KEYS):
                hint += f"\nShortcut: {CLASS_KEYS[column - 1]}"
            button.setToolTip(hint)
            button.toggled.connect(
                lambda checked, index=class_index: checked and self._class_chosen(index)
            )
            paint_group.addButton(button)
            grid.addWidget(button, 0, column)
            self.paint_buttons[class_index] = button

            # Equal shares, so every button in both rows is the same width
            # whatever its class is called.
            grid.setColumnStretch(column, 1)

            if erasing:
                self.lock_all_button = _lock_button("All")
                self.lock_all_button.setToolTip(
                    "Lock every class, or unlock them all\nShortcut: A"
                )
                self.lock_all_button.clicked.connect(lambda: self.toggle_all_locks())
                grid.addWidget(self.lock_all_button, 1, column)
                continue

            lock = _lock_button(name, locked=False)
            lock.setCheckable(True)
            lock.setToolTip("A locked class cannot be painted over by any tool")
            lock.toggled.connect(self._locks_changed)
            grid.addWidget(lock, 1, column)
            self.lock_buttons[class_index] = lock

        # Qt leaves nothing checked; the canvas starts on the first foreground
        # class. Choosing it here makes the bar say what the brush will actually
        # write -- a bar reading "erase" while the brush paints GIS is one that
        # costs somebody an image.
        self._select_class(self.canvas.paint_value)
        return grid

    def _build_actions(self) -> QGridLayout:
        """Build undo and redo, above the save they lead to.

        Marked like the tools beside them, and drawn the same way: these three
        are the buttons a reviewer reaches for without reading, so what each
        one is has to be legible before its caption is.
        """
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        self.undo_button = QPushButton(undo_icon(), "Undo")
        self.undo_button.setToolTip("Undo the last stroke on this label\nShortcut: Ctrl+Z")
        self.undo_button.clicked.connect(self.undo)
        grid.addWidget(self.undo_button, 0, 0)

        self.redo_button = QPushButton(redo_icon(), "Redo")
        self.redo_button.setToolTip("Redo the last undone stroke\nShortcut: Ctrl+Shift+Z")
        self.redo_button.clicked.connect(self.redo)
        grid.addWidget(self.redo_button, 0, 1)

        self.save_button = QPushButton(save_icon(), "Save labels...")
        self.save_button.setToolTip(
            "Write every label edited in this lamella, and choose whether to "
            "mark them corrected\nShortcut: Ctrl+S"
        )
        self.save_button.clicked.connect(self.save_requested.emit)
        grid.addWidget(self.save_button, 1, 0, 1, 2)
        return grid

    def _swatch(self, class_index: int):
        """Return a colour chip for one class."""
        pixmap = QPixmap(SWATCH, SWATCH)
        red, green, blue = self.controller.config.rgb(class_index)
        pixmap.fill(QColor(red, green, blue))
        return chip_icon(pixmap)

    def _build_timeline(self) -> QHBoxLayout:
        """Build the timeline row, over the frames worth proofreading."""
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(6)

        self.timeline = FrameTimeline()
        self.timeline.setToolTip(
            "The images of this lamella worth proofreading, in order.\nThe thin "
            "line is each label, bright where it holds an annotation.\nClick or "
            "drag to jump to one; Left and Right step."
        )
        self.timeline.frame_selected.connect(self.select_frame)
        row.addWidget(self.timeline, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(70)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.frame_label.setToolTip("Position among the frames to proofread")
        row.addWidget(self.frame_label)
        return row

    # -- user actions ------------------------------------------------------

    def _modality_chosen(self, modality: str) -> None:
        self.controller.set_modality(modality)
        self.refresh()

    def _series_chosen(self, index: int) -> None:
        """Proofread the chosen series instead."""
        if index < 0:
            return
        self.controller.set_series(self.series_combo.itemData(index))
        self.refresh()

    def _variant_chosen(self, index: int) -> None:
        """Show the chosen imaging condition."""
        if index < 0:
            return
        self.controller.set_variant(self.variant_combo.itemData(index))
        self.refresh()

    def set_variant_descriptions(self, descriptions: dict[str, str]) -> None:
        """Take the human-readable condition names from the project metadata.

        Handed in rather than read here: the panel has a controller, and the
        controller has the scan, but neither reads project metadata -- the
        window does, for the curation view, and this is the same answer.
        """
        self._variant_descriptions = descriptions

    def _overlay_toggled(self, shown: bool) -> None:
        self.controller.show_overlay = shown
        self.canvas.set_label_shown(shown)

    def _show_speck_controls(self, visible: bool) -> None:
        """Show the size only while anything is being looked for.

        There is no count beside it: the timeline carries one per frame, this
        frame's included, and a second copy of one number would only be a place
        for the two to disagree.
        """
        self.speck_slider.setVisible(visible)

    def _finding_toggled(self, finding: bool) -> None:
        """Start or stop pointing out specks, and set the view for it.

        Switching on takes the view the reviewer had and replaces it with the
        one that makes a four-pixel annotation visible: the label alone,
        opaque, against black. Switching off puts theirs back -- a filter that
        leaves the window rearranged is one you stop using.
        """
        if finding:
            self._view_before = (
                self.overlay_check.isChecked(),
                self.image_check.isChecked(),
                self.alpha_slider.value(),
            )
            self.overlay_check.setChecked(True)
            self.image_check.setChecked(False)
            self.alpha_slider.setValue(100)
        elif self._view_before is not None:
            overlay, image, alpha = self._view_before
            self._view_before = None
            self.overlay_check.setChecked(overlay)
            self.image_check.setChecked(image)
            self.alpha_slider.setValue(alpha)

        self.controller.finding_specks = finding
        self._show_speck_controls(finding)
        self._refresh_specks()
        self._refresh_timeline()
        if finding:
            self.speck_counts_needed.emit()

    def toggle_specks(self) -> None:
        """Start or stop looking for specks, from the keyboard."""
        self.speck_check.toggle()

    def _speck_size_changed(self, pixels: int) -> None:
        """Take a new size, and put off the sweep until the slider stops.

        Every count in the lamella is invalid at a new size, so recounting on
        each value would read the series again per pixel the handle moves.
        """
        self.controller.set_speck_size(pixels)
        self._refresh_specks()
        self._refresh_timeline()
        self._resweep_timer.start(SPECK_RECOUNT_MS)

    def _refresh_specks(self) -> None:
        """Ring what is there now, and put the new count on the timeline.

        The timeline as well as the canvas: painting out the last speck in a
        frame changes what its mark should say, and a recount that reached the
        controller without redrawing left the old number standing.
        """
        if not self.controller.finding_specks:
            self.canvas.show_specks(())
            self._speck_position = 0
            self._refresh_timeline()
            return
        self.canvas.show_specks(self.controller.specks())
        self._speck_position = 0
        self._refresh_timeline()

    def step_speck(self, delta: int) -> None:
        """Bring the next speck into the middle of the pane, wrapping round."""
        specks = self.controller.specks()
        if not self.controller.finding_specks or not specks:
            return
        self._speck_position = (self._speck_position + delta) % len(specks)
        speck = specks[self._speck_position]
        self.canvas.centre_on(speck.x, speck.y)

    def _image_toggled(self, shown: bool) -> None:
        self.controller.show_image = shown
        self.canvas.set_image_shown(shown)

    def _alpha_moved(self, value: int) -> None:
        self.canvas.set_label_opacity(value / 100.0)

    def _tool_chosen(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        # The brush size means nothing to the fill, so it is switched off rather
        # than left looking as though it applies.
        for widget in (self.brush_slider, self.brush_value):
            widget.setEnabled(tool == TOOL_BRUSH)

    def set_tool(self, tool: str) -> None:
        """Choose the tool, from the keyboard."""
        button = self.tool_buttons.get(tool)
        if button is not None:
            button.setChecked(True)

    def _class_chosen(self, class_index: int) -> None:
        self.canvas.set_paint_value(class_index)

    def _select_class(self, class_index: int) -> None:
        """Show *class_index* as the class being painted, if there is a button."""
        button = self.paint_buttons.get(class_index)
        if button is not None:
            button.setChecked(True)

    @property
    def paint_class(self) -> int:
        """Return the class the brush will write."""
        return self.canvas.paint_value

    def class_names(self) -> list[str]:
        """Return the classes that can be painted, in the order they are shown."""
        return [self.paint_buttons[index].text() for index in self._classes]

    def select_class_at(self, position: int) -> None:
        """Paint the class at *position* in the row, from the keyboard.

        Out of range does nothing: the keys are laid out for the classes a
        config declares, and a config with fewer leaves the last of them idle.
        """
        if 0 <= position < len(self._classes):
            self._select_class(self._classes[position])

    def _brush_resized(self, position: int) -> None:
        """Set the brush to the width at *position* in :data:`BRUSH_WIDTHS`."""
        width = BRUSH_WIDTHS[max(0, min(position, len(BRUSH_WIDTHS) - 1))]
        # The canvas works in the radius OpenCV draws with; the reviewer is
        # shown how wide the stroke comes out, which is the thing they can see.
        self.canvas.set_brush_radius((width - 1) // 2)
        self.brush_value.setText(f"{width} px")

    @property
    def brush_width(self) -> int:
        """Return how many pixels across the brush paints."""
        return 2 * self.canvas.brush_radius + 1

    def step_brush(self, delta: int) -> None:
        """Grow or shrink the brush by one offered size, from the keyboard."""
        self.brush_slider.setValue(self.brush_slider.value() + delta)

    def toggle_all_locks(self, locked: bool | None = None) -> None:
        """Lock every class, or unlock every class, from the keyboard.

        One key rather than two, because the question being asked is "protect
        what is already there while I work" and the answer is on or off. With
        anything unlocked it locks the lot; only when everything is already
        locked does it let go. Background has no lock: painting it is how a
        stroke erases, so locking it would take the eraser away.
        """
        if locked is None:
            locked = not all(button.isChecked() for button in self.lock_buttons.values())
        for button in self.lock_buttons.values():
            button.setChecked(locked)

    def _locks_changed(self) -> None:
        """Protect the locked classes, and open the shackle on the rest.

        The padlock is redrawn here rather than where it is toggled so that a
        lock changed from the keyboard looks like one changed by clicking it.
        A checked button whose icon says "open" is a control that reports the
        opposite of what it is doing.
        """
        locked = {
            index for index, button in self.lock_buttons.items() if button.isChecked()
        }
        for index, button in self.lock_buttons.items():
            button.setIcon(lock_icon(index in locked))
        self.canvas.set_protected(locked)

    def _stroke_finished(self, patch) -> None:
        """Record a stroke and show what it changed."""
        self.controller.record_edit(patch)
        self._label_changed()

    def undo(self) -> None:
        """Reverse the last stroke on the label on screen."""
        if self.controller.session.undo():
            self.canvas.refresh_label()
            self.controller.note_label_changed()
            self._label_changed()

    def redo(self) -> None:
        """Reapply the last undone stroke."""
        if self.controller.session.redo():
            self.canvas.refresh_label()
            self.controller.note_label_changed()
            self._label_changed()

    def _label_changed(self) -> None:
        """Bring the window into step with a label that has just changed.

        One path for every way of changing it. Undo and redo change the label
        as surely as a stroke does, and having their own way of saying so is
        how they came to say nothing: the count went on reporting what the
        stroke had made it.
        """
        self._refresh_timeline()
        self._refresh_edit_state()
        if self.controller.finding_specks:
            # Counting takes tens of milliseconds, which is nothing in a pause
            # and everything mid-drag, so it waits for the drawing to stop.
            self._speck_timer.start(SPECK_RECOUNT_MS)

    def focus_canvas(self) -> None:
        """Give the canvas the keyboard, so space pans without a click first."""
        self.canvas.setFocus(Qt.OtherFocusReason)

    def canvas_fit(self) -> None:
        """Scale the image to the pane."""
        self.canvas.fit()

    def toggle_label(self) -> None:
        """Show or hide the label, from the keyboard."""
        self.overlay_check.setChecked(not self.overlay_check.isChecked())

    def set_modality(self, modality: str) -> None:
        """Switch modality, from the keyboard, if there is anything to show."""
        button = self.modality_buttons.get(modality.upper())
        if button is not None and button.isEnabled():
            button.setChecked(True)

    def select_frame(self, position: int) -> None:
        """Move to *position* among the frames being proofread."""
        self.controller.select(position)
        self.refresh()

    def step_frame(self, delta: int) -> None:
        """Move *delta* frames."""
        self.controller.step(delta)
        self.refresh()

    # -- rendering ---------------------------------------------------------

    def refresh(self) -> None:
        """Redraw everything from the controller, and ask for what is missing."""
        self.refresh_versions()
        self._refresh_series()
        self._refresh_modalities()
        self._refresh_variants()
        self._refresh_timeline()
        self.status.setText(self.controller.status_text())
        self._refresh_edit_state()
        self.draw_current()
        self.loads_needed.emit()

    # -- earlier versions --------------------------------------------------

    def _label_to_draw(self):
        """Return what the canvas should show: the label, a version, or a mask.

        Comparing replaces the label rather than sitting beside it, so the two
        occupy the same pixels and flicking between them is a real comparison
        rather than an eye moving across a gap.
        """
        working = self.controller.working_label()
        if self._showing_differences:
            mask = self.versions_controller.difference(working)
            if mask is not None:
                import numpy as np

                return np.where(mask, DIFFERENCE_CLASS, 0).astype(np.uint8)
        if self._comparing:
            other = self.versions_controller.comparison_pixels()
            if other is not None:
                return other
        return working

    def _version_chosen(self, index: int) -> None:
        """Note which version is being offered, and whether it can be shown."""
        self.versions_controller.select(None if index < 0 else index)
        chosen = self.versions_controller.selected
        self.versions.set_available(bool(chosen and chosen.available))
        self._comparing = False
        self._showing_differences = False
        self.draw_current()

    def _set_comparing(self, comparing: bool) -> None:
        """Put an earlier version on the canvas, or take it off again."""
        self._comparing = comparing
        self._refresh_edit_state()
        self.draw_current()

    def _compare_key_held(self, held: bool) -> None:
        """Flick to the chosen version while the key is down.

        Held rather than toggled: the comparison is made by going back and
        forth, and a mode that stays on is one more thing to remember to leave.
        """
        chosen = self.versions_controller.selected
        if chosen is None or not chosen.available:
            return
        self.versions.set_comparing(held)

    def _set_differences(self, showing: bool) -> None:
        """Mark where the chosen version and the label disagree."""
        self._showing_differences = showing
        self._refresh_edit_state()
        self.draw_current()

    def refresh_versions(self) -> None:
        """Read the kept versions of the label on screen and list them."""
        self.versions_controller.set_label(self.controller.current_key)
        self.versions.set_rows(self.versions_controller.rows)

    @property
    def showing_a_version(self) -> bool:
        """Return whether the canvas holds something other than the label."""
        return self._comparing or self._showing_differences

    def _refresh_edit_state(self) -> None:
        """Reflect what can be undone, and what is waiting to be saved."""
        session = self.controller.session
        self.undo_button.setEnabled(session.can_undo)
        self.redo_button.setEnabled(session.can_redo)
        pending = self.controller.pending_count()
        self.save_button.setEnabled(bool(pending))
        self.pending_label.setText(
            f"{pending} label{'s' if pending != 1 else ''} waiting - {UNSAVED_NOTE}"
            if pending else ""
        )

    def _refresh_series(self) -> None:
        """Offer the series this lamella holds, greyed where there is one."""
        series = self.controller.available_series()
        fill_choice_combo(
            self.series_combo,
            [(SERIES_LABELS.get(name, name), name) for name in series],
            self.controller.series,
        )

    def _refresh_variants(self) -> None:
        """Offer the conditions this acquisition has, greyed where there is one.

        On screen either way, as in the curation view: a lamella with one
        condition per modality is every Autolamella project so far, so hiding
        it moved this row on nearly every step.
        """
        described = self._variant_descriptions
        fill_choice_combo(
            self.variant_combo,
            [
                ("base" if variant is None else described.get(variant, variant), variant)
                for variant in self.controller.available_variants()
            ],
            self.controller.variant,
        )

    def _refresh_modalities(self) -> None:
        """Enable only the modalities this lamella has anything to proofread in."""
        available = self.controller.available_modalities()
        for modality, button in self.modality_buttons.items():
            button.setEnabled(modality in available)
        current = self.controller.modality
        button = self.modality_buttons[current]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)

    def refresh_timeline(self) -> None:
        """Redraw the timeline. For the window, as speck counts arrive."""
        self._refresh_timeline()

    def _refresh_timeline(self) -> None:
        """Redraw the timeline and the position counter."""
        self.timeline.set_states(
            self.controller.timeline_states(),
            self.controller.position,
            labels=self.controller.timeline_labels(),
            selected=self.controller.edited_positions(),
            specks=self.controller.timeline_specks(),
        )
        count = self.controller.count
        position = self.controller.position + 1 if count else 0
        self.frame_label.setText(f"{position} / {count}")

    def _range_moved(self, low: int, high: int) -> None:
        """Take the new limits and redraw the frame under them.

        Held on the panel rather than in the controller, and never reaching a
        label: this changes what the reviewer *sees* while they paint, not what
        they paint. The two are held apart by the control, not here.
        """
        self._intensity_range = (low, high)
        self.draw_current()

    def draw_current(self) -> None:
        """Draw the frame, or say why there is nothing drawn.

        Separate from :meth:`refresh` so that a frame finishing its read can be
        drawn without asking for more to be prepared.
        """
        if self.controller.frame is None:
            self.canvas.clear()
            self.canvas.set_editable(False)
            self._set_badges([])
            return

        prepared = self.controller.cached_image()
        if prepared is None:
            # Only what is already prepared is read: a full-resolution frame
            # takes long enough that blocking here would freeze the window.
            self._set_badges([("loading...", "info")])
            return
        if prepared.failed:
            self.canvas.clear()
            self._set_badges([(f"could not read image: {prepared.error}", "error")])
            return

        display = prepared.display
        # The label is painted into as it is; only the picture under it is
        # rescaled, or a stroke would be recorded against something that is not
        # the file.
        image = apply_intensity_range(display.image, *self._intensity_range)
        self.canvas.set_frame(image, self._label_to_draw())
        self.canvas.set_label_shown(self.controller.show_overlay)
        self.canvas.set_image_shown(self.controller.show_image)
        self._refresh_specks()
        # Never while an earlier version is on the canvas: a stroke would be
        # recorded against the label underneath, which is not what is being
        # looked at.
        self.canvas.set_editable(
            self.controller.editable and not self.showing_a_version
        )
        self._set_badges(
            intensity_badge(self._intensity_range) + self._label_badges(display)
        )

    def _label_badges(self, display) -> list[tuple[str, str]]:
        """Return the badges describing what is on screen."""
        badges: list[tuple[str, str]] = []
        text, level = LABEL_BADGES[display.label_status]
        if text:
            badges.append((text, level))
        if display.label_resized:
            # Editing is refused on these: saving a label resampled to fit the
            # image would rewrite the annotation at the wrong scale.
            badges.append(
                ("label does not match the image's dimensions - not editable", "error")
            )
        return badges

    def _set_badges(self, badges: list[tuple[str, str]]) -> None:
        """Show what is worth saying about the label, or nothing at all.

        Hidden rather than emptied: an empty label still holds its row, which
        is the whole reason these moved off one of their own.
        """
        self.badges.setVisible(bool(badges))
        if not badges:
            self.badges.setText("")
            return
        self.badges.setText(
            "&nbsp;&nbsp;|&nbsp;&nbsp;".join(
                f'<span style="{BADGE_STYLES[level]}">{text}</span>'
                for text, level in badges
            )
        )
