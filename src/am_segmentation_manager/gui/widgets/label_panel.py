"""The Label tab: one acquisition at full resolution, ready to be proofread.

Reads its own controller and draws from it, so the window wires this tab up
rather than running it. Nothing here writes to disk; the tab is a viewer until
the drawing tools arrive.

The lamella is not chosen here. It is handed in from the Curate tab, so the
collection has one place where "which lamella am I on" is decided.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import QPointF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPixmap, QPolygonF
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...config import BACKGROUND_CLASS
from ...display import DEFAULT_ALPHA
from ...layout import MODALITIES
from ...operations.label_measure import DEFAULT_SPECK_PIXELS
from ..label_controller import LabelController
from .common import LABEL_BADGES, chip_icon
from .frame_strip import FrameTimeline
from .label_canvas import (
    DEFAULT_BRUSH_RADIUS,
    TOOL_BRUSH,
    TOOL_FILL,
    LabelCanvas,
)
from .theme import BADGE_STYLES, TEXT

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

#: The sizes the speck slider offers, in pixels. One pixel is the smallest
#: thing there is; the top is where "small component" stops meaning anything.
MIN_SPECK_PIXELS = 1
MAX_SPECK_PIXELS = 64

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

    #: Emitted when the reviewer asks to write this lamella's edits. The window
    #: owns everything that touches the filesystem.
    save_requested = Signal()

    def __init__(self, controller: LabelController) -> None:
        """Build the tab against *controller*."""
        super().__init__()
        self.controller = controller
        #: The classes to offer, in the order they are shown and keyed.
        self._classes = list(controller.config.classes)
        #: Condition names from the project metadata; see
        #: :meth:`set_variant_descriptions`.
        self._variant_descriptions: dict[str, str] = {}
        #: The view to put back when the speck filter is switched off.
        self._view_before: tuple[bool, bool, int] | None = None
        #: Which speck the stepping key is on.
        self._speck_position = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        layout.addLayout(self._build_controls())

        self.canvas = LabelCanvas(controller.config)
        self.canvas.edited.connect(self._stroke_finished)
        self.canvas.brush_stepped.connect(self.step_brush)
        layout.addWidget(self.canvas, stretch=1)

        layout.addLayout(self._build_tools())

        self.badges = QLabel("")
        self.badges.setTextFormat(Qt.RichText)
        layout.addWidget(self.badges)

        self.status = QLabel("")
        self.status.setStyleSheet(BADGE_STYLES["info"])
        layout.addWidget(self.status)

        layout.addLayout(self._build_timeline())

        self._speck_timer = QTimer(self)
        self._speck_timer.setSingleShot(True)
        self._speck_timer.timeout.connect(self._refresh_specks)

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

        self.modality_buttons: dict[str, QRadioButton] = {}
        group = QButtonGroup(self)
        for modality, key in zip(MODALITIES, ("1", "2")):
            button = QRadioButton(modality)
            button.setToolTip(
                f"Proofread the {modality} images of this lamella\nShortcut: {key}"
            )
            button.toggled.connect(
                lambda checked, m=modality: checked and self._modality_chosen(m)
            )
            group.addButton(button)
            controls.addWidget(button)
            self.modality_buttons[modality] = button
        # Checked without emitting: the panel is still being built, and the
        # handler redraws a timeline that does not exist yet.
        first = self.modality_buttons[MODALITIES[0]]
        first.blockSignals(True)
        first.setChecked(True)
        first.blockSignals(False)

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
        self._show_variant_controls(False)

        controls.addSpacing(20)
        self.overlay_check = QCheckBox("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip("Draw the label over the image\nShortcut: V")
        self.overlay_check.toggled.connect(self._overlay_toggled)
        controls.addWidget(self.overlay_check)

        self.image_check = QCheckBox("Show image")
        self.image_check.setChecked(True)
        self.image_check.setToolTip(
            "Draw the acquisition under the label, or leave the label alone "
            "against black.\nOver black is how an annotation of a few pixels "
            "is found."
        )
        self.image_check.toggled.connect(self._image_toggled)
        controls.addWidget(self.image_check)

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

        controls.addSpacing(20)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Scale the image to the pane\nShortcut: 0")
        self.fit_button.clicked.connect(self.canvas_fit)
        controls.addWidget(self.fit_button)

        controls.addSpacing(20)
        self.speck_check = QCheckBox("Find specks")
        self.speck_check.setToolTip(
            "Ring every annotated component of a few pixels, whatever class it "
            "is.\nSwitching it on sets the view that makes them findable, and "
            "switching it off puts your own back.\nShortcut: X, and N steps to "
            "the next one"
        )
        self.speck_check.toggled.connect(self._finding_toggled)
        controls.addWidget(self.speck_check)

        self.speck_slider = QSlider(Qt.Horizontal)
        self.speck_slider.setRange(MIN_SPECK_PIXELS, MAX_SPECK_PIXELS)
        self.speck_slider.setValue(self.controller.speck_size)
        self.speck_slider.setFixedWidth(90)
        self.speck_slider.setToolTip(
            "Largest component still ringed.\n"
            f"{DEFAULT_SPECK_PIXELS} is one pixel of a prediction: labels "
            "converted from predictions are upscaled from half resolution, so "
            "anything smaller was painted by hand."
        )
        self.speck_slider.valueChanged.connect(self._speck_size_changed)
        controls.addWidget(self.speck_slider)

        self.speck_count = QLabel("")
        self.speck_count.setMinimumWidth(96)
        self.speck_count.setToolTip(self.speck_check.toolTip())
        controls.addWidget(self.speck_count)
        self._show_speck_controls(False)

        controls.addStretch()
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

        brush_button = self.tool_buttons[TOOL_BRUSH]
        brush_button.blockSignals(True)
        brush_button.setChecked(True)
        brush_button.blockSignals(False)
        return grid

    def _build_class_table(self) -> QGridLayout:
        """Build the classes to paint, above the same classes' locks.

        One column per class, so a lock sits under the class it protects and
        the two rows read as one table. Background's column carries the
        lock-everything button instead of a lock of its own: painting background
        is how a stroke erases, so locking it would take the eraser away.
        """
        self.class_grid = grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        paint_label, lock_label = QLabel("Paint"), QLabel("Lock")
        lock_label.setToolTip("A locked class cannot be painted over by any tool")
        grid.addWidget(paint_label, 0, 0)
        grid.addWidget(lock_label, 1, 0)

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

            if erasing:
                self.lock_all_button = QToolButton()
                self.lock_all_button.setText("All")
                self.lock_all_button.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Preferred
                )
                self.lock_all_button.setToolTip(
                    "Lock every class, or unlock them all\nShortcut: A"
                )
                self.lock_all_button.clicked.connect(lambda: self.toggle_all_locks())
                grid.addWidget(self.lock_all_button, 1, column)
                continue

            lock = QToolButton()
            lock.setText(name)
            lock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            lock.setCheckable(True)
            lock.setToolTip(lock_label.toolTip())
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
        """Build undo and redo, above the save they lead to."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo the last stroke on this label\nShortcut: Ctrl+Z")
        self.undo_button.clicked.connect(self.undo)
        grid.addWidget(self.undo_button, 0, 0)

        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Redo the last undone stroke\nShortcut: Ctrl+Shift+Z")
        self.redo_button.clicked.connect(self.redo)
        grid.addWidget(self.redo_button, 0, 1)

        self.save_button = QPushButton("Save labels...")
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
            "line shows each label: bright means annotated, dim means an empty "
            "placeholder, and nothing means no label file.\nClick or drag to "
            "jump to one; Left/Right arrow keys step one."
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

    def _variant_chosen(self, index: int) -> None:
        """Show the chosen imaging condition."""
        if index < 0:
            return
        self.controller.set_variant(self.variant_combo.itemData(index))
        self.refresh()

    def _show_variant_controls(self, visible: bool) -> None:
        self.variant_label.setVisible(visible)
        self.variant_combo.setVisible(visible)

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
        """Show the size and the count only while anything is being looked for."""
        self.speck_slider.setVisible(visible)
        self.speck_count.setVisible(visible)

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

    def toggle_specks(self) -> None:
        """Start or stop looking for specks, from the keyboard."""
        self.speck_check.toggle()

    def _speck_size_changed(self, pixels: int) -> None:
        self.controller.set_speck_size(pixels)
        self._refresh_specks()

    def _refresh_specks(self) -> None:
        """Ring what is there now, and say how much of it there is."""
        if not self.controller.finding_specks:
            self.canvas.show_specks(())
            self.speck_count.setText("")
            self._speck_position = 0
            return
        specks = self.controller.specks()
        self.canvas.show_specks(specks)
        size = self.controller.speck_size
        self.speck_count.setText(
            f"{len(specks)} at {size}px or less" if specks else f"none at {size}px or less"
        )
        self._speck_position = 0

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
        self.canvas.set_protected(
            {index for index, button in self.lock_buttons.items() if button.isChecked()}
        )

    def _stroke_finished(self, patch) -> None:
        """Record a stroke and show what it changed."""
        self.controller.record_edit(patch)
        self._refresh_timeline()
        self._refresh_edit_state()
        if self.controller.finding_specks:
            # Counting takes tens of milliseconds, which is nothing in a pause
            # and everything mid-drag, so it waits for the drawing to stop.
            self._speck_timer.start(SPECK_RECOUNT_MS)

    def undo(self) -> None:
        """Reverse the last stroke on the label on screen."""
        if self.controller.session.undo():
            self.canvas.refresh_label()
            self._refresh_timeline()
            self._refresh_edit_state()

    def redo(self) -> None:
        """Reapply the last undone stroke."""
        if self.controller.session.redo():
            self.canvas.refresh_label()
            self._refresh_timeline()
            self._refresh_edit_state()

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
        self._refresh_modalities()
        self._refresh_variants()
        self._refresh_timeline()
        self.status.setText(self.controller.status_text())
        self._refresh_edit_state()
        self.draw_current()
        self.loads_needed.emit()

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

    def _refresh_variants(self) -> None:
        """Offer the conditions this acquisition has, or hide the control.

        Hidden unless there is a genuine choice, so a lamella with one condition
        per modality -- every Autolamella project so far -- shows nothing extra.
        """
        variants = self.controller.available_variants()
        if len(variants) < 2:
            self._show_variant_controls(False)
            return
        current = self.controller.variant
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        for variant in variants:
            caption = (
                "base" if variant is None
                else self._variant_descriptions.get(variant, variant)
            )
            self.variant_combo.addItem(caption, variant)
        self.variant_combo.setCurrentIndex(
            variants.index(current) if current in variants else 0
        )
        self.variant_combo.blockSignals(False)
        self._show_variant_controls(True)

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

    def _refresh_timeline(self) -> None:
        """Redraw the timeline and the position counter."""
        self.timeline.set_states(
            self.controller.timeline_states(),
            self.controller.position,
            labels=self.controller.timeline_labels(),
            selected=self.controller.edited_positions(),
        )
        count = self.controller.count
        position = self.controller.position + 1 if count else 0
        self.frame_label.setText(f"{position} / {count}")

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
        self.canvas.set_frame(display.image, self.controller.working_label())
        self.canvas.set_label_shown(self.controller.show_overlay)
        self.canvas.set_image_shown(self.controller.show_image)
        self._refresh_specks()
        self.canvas.set_editable(self.controller.editable)
        self._set_badges(self._label_badges(display))

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
        if not badges:
            self.badges.setText("")
            return
        self.badges.setText(
            "&nbsp;&nbsp;|&nbsp;&nbsp;".join(
                f'<span style="{BADGE_STYLES[level]}">{text}</span>'
                for text, level in badges
            )
        )
