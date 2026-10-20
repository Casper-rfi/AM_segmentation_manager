"""The Label tab: one acquisition at full resolution, ready to be proofread.

Reads its own controller and draws from it, so the window wires this tab up
rather than running it. Nothing here touches the filesystem: an edit is held by
the controller's session until the window saves it, and an earlier version is
put back by the versions controller once the reviewer has agreed.

The lamella is not chosen here. It is handed in from the Curate tab, so the
collection has one place where "which lamella am I on" is decided.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
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
    MAX_SPECK_PIXELS,
    MIN_SPECK_PIXELS,
)
from ..label_controller import LabelController
from ..loader import BackgroundLoader
from ..preferences import DEFAULT_BRUSH_RADIUS
from ..version_controller import VersionController, comparison_badge
from .common import (
    CONDITION_FIELD_WIDTH,
    LABEL_BADGES,
    VIEW_ONLY_BADGE,
    badge_markup,
    chip_icon,
    choice_count_badge,
    fill_choice_combo,
    fill_condition_combo,
    intensity_badge,
)
from .controls import (
    SLIDER_WIDTH,
    ElidingCombo,
    ElidingLabel,
    FlowArea,
    FlowLayout,
    RangeSlider,
    SegmentedControl,
    ToggleSwitch,
    ValueSlider,
    keep_paired,
    keep_together,
)
from .frame_strip import LABEL_BAND, FrameTimeline
from .icons import (
    brush_icon,
    fill_icon,
    lock_icon,
    redo_icon,
    save_icon,
    undo_icon,
)
from .label_canvas import (
    TOOL_BRUSH,
    TOOL_FILL,
    LabelCanvas,
)
from .labelling_guide_panel import LabellingGuidePanel
from .theme import BADGE_STYLES, CLASS_BUTTON, GROUP_GAP, PADLOCK_BUTTON
from .versions_bar import DIFFERENCE_CLASS, VersionsBar

logger = logging.getLogger(__name__)

#: Said whenever anything is waiting, because an edit that exists only in this
#: window is one closing it would throw away.
UNSAVED_NOTE = "not yet written to disk"

#: Names the restore in every dialog it opens.
RESTORE_TITLE = "Restore an earlier version"

#: The side every mark on the proofreading bars is drawn at -- the tools, the
#: class swatches, the padlocks, undo, redo and save -- so none reads as more
#: important than its neighbour. Two pixels more than the sixteen they are laid
#: out on.
MARK = 18

#: The colour square inside a paint button's mark. Short of :data:`MARK`, a
#: solid square carrying more ink than a line drawing the same size.
SWATCH = 14

#: The size undo, redo and save keep, in logical pixels: what they took with a
#: sixteen-pixel mark, held so the larger one does not grow the row.
ACTION_BUTTON = (37, 27)

#: The side of a class's padlock button, in logical pixels. It is reached for
#: between strokes with the eyes still on the canvas, so it is a target sized for
#: the pointer rather than for the icon it carries.
PADLOCK = 28


#: How wide the status line beside the view controls may get, in logical
#: pixels. A cap rather than a share: it sits beside a row that wraps, so width
#: given to it is width the controls wrap for want of. Measured against the
#: wrapping rather than chosen -- at 380 the row takes three lines on a 1200px
#: pane and at 300 it takes two, while no cap at all buys back the second line
#: on a 1920px one, those controls needing about 1730px to sit in a single row.
STATUS_MAX_WIDTH = 300

#: How long after the last stroke the specks are counted again. Long enough
#: that a pause between strokes is not one, short enough that a speck painted
#: out stops being ringed while the reviewer is still looking at it.
SPECK_RECOUNT_MS = 400

#: The keys that choose a class to paint, one per button in the order the
#: classes are shown.
#:
#: The row above the tools, so the hand that reaches D and F reaches these
#: without moving. Six, one more than the segmentation config declares; a
#: longer config leaves the rest without a key rather than borrowing one.
#: Named here, beside the buttons they press, and bound by the window.
CLASS_KEYS = ("Q", "W", "E", "R", "T", "Y")

#: The keys that lock or unlock one class: its paint key with Shift, so which
#: padlock a key reaches is read off the key that paints that class.
LOCK_KEYS = tuple(f"Shift+{key}" for key in CLASS_KEYS)

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





def _action_button(icon, hint: str, action) -> QToolButton:
    """Return a button that is its mark alone, the tooltip naming what it does."""
    button = QToolButton()
    button.setIcon(icon)
    button.setIconSize(QSize(MARK, MARK))
    button.setFixedSize(*ACTION_BUTTON)
    button.setToolTip(hint)
    button.clicked.connect(lambda _checked=False: action())
    return button


def _larger_text(button: QToolButton) -> None:
    """Set *button*'s caption a point larger, level with the mark beside it."""
    font = button.font()
    font.setPointSizeF(font.pointSizeF() + 1)
    button.setFont(font)


def _equalise(buttons) -> None:
    """Give every one of *buttons* the width of the widest, and no more."""
    widest = max(button.sizeHint().width() for button in buttons)
    for button in buttons:
        button.setMinimumWidth(widest)


def _padlock_button() -> QToolButton:
    """Return a class's padlock: checkable, open, and :data:`PADLOCK` square.

    Its shackle is kept in step with its state by :meth:`LabelPanel._locks_changed`.
    """
    button = QToolButton()
    button.setObjectName(PADLOCK_BUTTON)
    button.setCheckable(True)
    button.setIcon(lock_icon(False, MARK))
    button.setIconSize(QSize(MARK, MARK))
    button.setFixedSize(PADLOCK, PADLOCK)
    return button


def _width_position(width: int) -> int:
    """Return the slider position offering *width*, or the nearest below it."""
    return max(
        (index for index, offered in enumerate(BRUSH_WIDTHS) if offered <= width),
        default=0,
    )


class LabelPanel(QWidget):
    """The proofreading view: controls, the canvas, and the frame timeline."""

    #: Emitted when the reviewer asks to write this lamella's edits. The window
    #: owns everything that touches the filesystem.
    save_requested = Signal()
    #: A short message for the window to show at the pointer, which is where
    #: the reviewer is looking when a Shift-click is answered.
    notice = Signal(str)

    def __init__(
        self,
        controller: LabelController,
        versions: VersionController | None = None,
        loader: BackgroundLoader | None = None,
    ) -> None:
        """Build the tab against *controller*.

        Args:
            controller: What is being proofread.
            versions: The kept versions of the label on screen. Injected so a
                test can hand in one attributed to a known author.
            loader: Prepares frames and counts specks on a thread of its own,
                the tab's alone: proofreading needs every pixel, where Curate's
                loader deliberately reads less. Without one, nothing is read
                ahead and the tab draws only what the controller already holds.
        """
        super().__init__()
        self.controller = controller
        self.loader = loader
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
        self.canvas.class_picked.connect(self._class_picked)
        self.canvas.compare_held.connect(self._compare_key_held)
        # Beside the canvas rather than over it, so no step covers the image.
        self.guide_panel = LabellingGuidePanel(controller.labelling_guide.steps)
        self.guide_panel.step_toggled.connect(self._step_toggled)
        self.guide_panel.setVisible(False)
        beside = QHBoxLayout()
        beside.setContentsMargins(0, 0, 0, 0)
        beside.setSpacing(3)
        beside.addWidget(self.canvas, stretch=1)
        beside.addWidget(self.guide_panel)
        layout.addLayout(beside, stretch=1)

        self.versions = VersionsBar()
        self.versions.selected.connect(self._version_chosen)
        self.versions.comparing.connect(self._set_comparing)
        self.versions.differences.connect(self._set_differences)
        self.versions.restore_requested.connect(self.restore_version)
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
        self._resweep_timer.timeout.connect(self._sweep_specks)

        # Every control here has a shortcut, and none of them needs the
        # keyboard: what does is the canvas, whose space-to-pan is read as a
        # key press. A radio button that kept focus after being clicked would
        # take that space and toggle itself instead.
        for widget in self.findChildren(QWidget):
            if widget is not self.canvas:
                widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        if loader is not None:
            loader.worker.loaded.connect(self._on_loaded)
            loader.worker.counted.connect(self._on_speck_counted)

    # -- construction ------------------------------------------------------

    def apply_defaults(self, values) -> None:
        """Open this tab's controls at what the settings say. See `ImagePanel`.

        The speck size goes through the controller as well as the slider: it
        is held there, so a slider moved without it would report sizes nothing
        was counted at.
        """
        self.alpha_slider.setValue(int(values["label_opacity"]))
        self.brush_slider.setValue(_width_position(2 * int(values["brush_radius"]) + 1))
        self.speck_slider.setValue(int(values["speck_pixels"]))
        self.controller.set_speck_size(int(values["speck_pixels"]))

    def _build_controls(self) -> QHBoxLayout:
        """Build the row of view controls, wrapped, with the status beside it.

        Wrapped for the reason `image_panel` gives of its own row: laid out in
        one line these controls need more width than a laptop screen has, and
        a window cannot be made narrower than its widest unwrappable row. The
        status keeps the end of the row rather than joining the flow, being the
        one thing here that elides instead of taking the width it would like.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.controls = FlowArea()
        controls = FlowLayout(spacing=6)
        self.controls.setLayout(controls)

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
        # Chosen without emitting: the panel is still being built, and the
        # handler redraws a timeline that does not exist yet.
        self.modality_control.set_current(MODALITIES[0])

        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Scale the image to the pane\nShortcut: G")
        self.fit_button.clicked.connect(self.canvas_fit)
        # Which view and fitting it to the pane, over undo, redo and save: the
        # top left, where image software keeps the actions reached for most.
        controls.addWidget(keep_paired(
            (self.modality_control, self.fit_button), (self._build_actions(),),
            align={(0, 1): Qt.AlignmentFlag.AlignLeft, (1, 0): Qt.AlignmentFlag.AlignLeft},
        ))

        # Beside the beam buttons and before the variant, in the same order and
        # for the same reasons as the curation view: the series decides which
        # frames there are to review, the variant only which file of one is
        # drawn. Annotating a lamella means annotating one series of it.
        self.series_label = QLabel("Series")
        self.series_combo = ElidingCombo()
        self.series_combo.setMinimumWidth(110)
        self.series_combo.setToolTip(
            "Select which image series to proofread."
        )
        self.series_label.setToolTip(self.series_combo.toolTip())
        self.series_combo.currentIndexChanged.connect(self._series_chosen)
        controls.addSpacing(GROUP_GAP)

        # A dropdown rather than buttons, as in the curation view: variant names
        # run long, and most projects have none to choose between.
        self.variant_label = QLabel("Variant")
        self.variant_combo = ElidingCombo()
        self.variant_combo.setMinimumWidth(CONDITION_FIELD_WIDTH)
        self.variant_combo.setToolTip(
            "Which imaging condition to proofread on. The label is shared with all variants "
            "within one family."
        )
        self.variant_label.setToolTip(self.variant_combo.toolTip())
        self.variant_combo.currentIndexChanged.connect(self._condition_chosen)
        # Which images the timeline is made of, above which file of one is
        # drawn: the narrower question under the wider one.
        controls.addWidget(keep_paired(
            (self.series_label, self.series_combo, choice_count_badge(self.series_combo)),
            (self.variant_label, self.variant_combo, choice_count_badge(self.variant_combo)),
        ))

        controls.addSpacing(GROUP_GAP)
        self.overlay_check = ToggleSwitch("Label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip("Display the label\nShortcut: V")
        self.overlay_check.toggled.connect(self._overlay_toggled)

        self.image_check = ToggleSwitch("Image")
        self.image_check.setChecked(True)
        self.image_check.setToolTip(
            "Display the image"
        )
        self.image_check.toggled.connect(self._image_toggled)
        # What is drawn, one switch above the other: the same question asked of
        # the two layers, and stacked they cost one caption's width.
        controls.addWidget(keep_paired((self.overlay_check,), (self.image_check,)))

        controls.addSpacing(GROUP_GAP)
        opacity_label = QLabel("Opacity")
        self.alpha_slider = ValueSlider()
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(DEFAULT_ALPHA * 100))
        self.alpha_slider.setFixedWidth(SLIDER_WIDTH)
        self.alpha_slider.setToolTip("Opacity of the label")
        opacity_label.setToolTip(self.alpha_slider.toolTip())
        self.alpha_slider.valueChanged.connect(self._alpha_moved)

        low, high = DEFAULT_INTENSITY_RANGE
        range_hint = (
            "Set min and max pixel intensities of how the image is displayed."
        )
        range_label = QLabel("Min / max")
        range_label.setToolTip(range_hint)
        self.intensity_slider = RangeSlider(0, 255, low, high)
        self.intensity_slider.setToolTip(range_hint)
        self.intensity_slider.range_changed.connect(self._range_moved)
        # How the label is drawn, above how the image under it is: two
        # sliders, aligned, and the long caption no longer sets the width alone.
        controls.addWidget(keep_paired(
            (opacity_label, self.alpha_slider),
            (range_label, self.intensity_slider),
        ))


        controls.addSpacing(GROUP_GAP)
        self.speck_check = ToggleSwitch("Find specks")
        self.speck_check.setToolTip(
            "Highlight specks: components of a few pixels that are likely incorrect "
            "annotations.\nShortcut: X; N steps to the next"
        )
        self.speck_check.toggled.connect(self._finding_toggled)
        self.speck_slider = ValueSlider()
        self.speck_slider.setRange(MIN_SPECK_PIXELS, MAX_SPECK_PIXELS)
        self.speck_slider.setValue(self.controller.speck_size)
        self.speck_slider.setFixedWidth(90)
        self.speck_slider.setToolTip(
            "Set the size in pixels below which specks are highlighted."
        )
        self.speck_slider.valueChanged.connect(self._speck_size_changed)


        self._show_speck_controls(False)

        # Off by default and per session. Comparing is wanted a few times a
        # year against a view read every day, so the strip earns its space only
        # while somebody is using it -- and the count says whether opening it
        # would show anything before it is opened.
        self.versions_check = ToggleSwitch("Compare versions")
        self.versions_check.setToolTip(
            "List earlier versions and predictions of this label, to compare with the current "
            "label."
        )
        self.versions_check.toggled.connect(self._versions_toggled)
        # The two ways of looking harder at a label, one above the other: ring
        # what is too small to see, or put an earlier version in its place.
        controls.addWidget(keep_paired(
            (self.speck_check, self.speck_slider),
            (self.versions_check,),
        ))

        # Off by default and per session, like the versions: an aid somebody
        # turns to, which C opens as well as stepping through.
        controls.addSpacing(GROUP_GAP)
        self.guide_check = ToggleSwitch("Labelling guide")
        self.guide_check.setToolTip(
            "Show the proofreading steps, to tick off for each label.\n"
            "Shortcut: C ticks the current step; Shift+C goes back"
        )
        self.guide_check.toggled.connect(self._guide_toggled)
        controls.addWidget(self.guide_check)

        row.addWidget(self.controls, stretch=1)

        # Where the reviewer is, at the end of the row that decides it: the
        # modality buttons on the left of this line are half of what it reports.
        # On a row of its own it cost the canvas a row of height to say one
        # short sentence, and it sat under the buttons rather than near the
        # navigation.
        # Elided rather than clipped: the line runs long on a project with
        # timestamped stage names, and this row has other controls in it.
        # Beside the flow rather than in it: a flow layout gives every item the
        # width its size hint asks for, which is the whole line here -- so the
        # status would take a row to itself and never elide at all. Capped, or
        # sharing the surplus with a row that wraps hands it half the width
        # while the controls below wrap for want of it.
        self.status = ElidingLabel()
        self.status.setStyleSheet(BADGE_STYLES["info"])
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status.setMaximumWidth(STATUS_MAX_WIDTH)
        row.addWidget(self.status, stretch=1)
        return row

    def _build_tools(self) -> QHBoxLayout:
        """Build the bottom bar: the tools, then the classes, in one row that wraps.

        The paint buttons carry the class colours, which is what makes a
        separate legend unnecessary -- the button you press to paint a class is
        also the thing that tells you what colour it is. A flow, so a narrow
        pane wraps the row rather than widening the window.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tools = FlowArea()
        flow = FlowLayout(spacing=6)
        self.tools.setLayout(flow)
        brush, fill, size = self._build_tool_buttons()
        flow.addWidget(keep_together(brush, fill))
        flow.addWidget(size)
        flow.addSpacing(GROUP_GAP)
        for cell in self._build_class_cells():
            flow.addWidget(cell)
        row.addWidget(self.tools, stretch=1)

        # Both say something about the label being edited, so they belong with
        # the tools that edit it -- and in the width this row has spare, rather
        # than each holding a row of the canvas's height for one short line.
        self.badges = QLabel("")
        self.badges.setTextFormat(Qt.TextFormat.RichText)
        self.badges.setWordWrap(True)
        self.badges.setMaximumWidth(STATUS_MAX_WIDTH)
        self.badges.setVisible(False)
        row.addWidget(self.badges)

        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet(BADGE_STYLES["warning"])
        row.addWidget(self.pending_label)
        return row

    def _build_tool_buttons(self) -> tuple[QToolButton, QToolButton, QWidget]:
        """Build the brush and the fill, and the size that belongs to the brush."""
        self.tool_buttons: dict[str, QToolButton] = {}
        group = QButtonGroup(self)
        for tool, caption, key, hint, icon in (
            (TOOL_BRUSH, "Brush", "D", "Draw where you drag", brush_icon(MARK)),
            (
                TOOL_FILL, "Fill", "F",
                "Fill the area of one class that you click in", fill_icon(MARK),
            ),
        ):
            button = QToolButton()
            button.setText(caption)
            button.setIcon(icon)
            button.setIconSize(QSize(MARK, MARK))
            _larger_text(button)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setFixedHeight(PADLOCK)
            button.setCheckable(True)
            button.setToolTip(f"{hint}\nShortcut: {key}")

            def choose_tool(checked: bool, t: str = tool) -> None:
                if checked:
                    self._tool_chosen(t)

            button.toggled.connect(choose_tool)
            group.addButton(button)
            self.tool_buttons[tool] = button

        self.brush_slider = ValueSlider()
        # One step of the slider is one entry in BRUSH_WIDTHS, not one pixel:
        # see why there.
        self.brush_slider.setRange(0, len(BRUSH_WIDTHS) - 1)
        self.brush_slider.setValue(_width_position(2 * DEFAULT_BRUSH_RADIUS + 1))
        self.brush_slider.setFixedWidth(120)
        self.brush_slider.setToolTip(
            "Brush size in pixels\nShortcuts: [ and ], or Shift and the scroll wheel"
        )
        self.brush_slider.valueChanged.connect(self._brush_resized)
        self.brush_value = QLabel("")
        self.brush_value.setMinimumWidth(38)
        self.brush_value.setToolTip(self.brush_slider.toolTip())
        self._brush_resized(self.brush_slider.value())

        # The two are a pair and read as one: the shorter caption is widened
        # to the longer rather than both being left to their own text.
        _equalise(self.tool_buttons.values())

        brush_button = self.tool_buttons[TOOL_BRUSH]
        brush_button.blockSignals(True)
        brush_button.setChecked(True)
        brush_button.blockSignals(False)
        return (
            brush_button,
            self.tool_buttons[TOOL_FILL],
            keep_together(self.brush_slider, self.brush_value),
        )

    def _build_class_cells(self) -> list[QWidget]:
        """Build a cell per class: its paint button beside the padlock protecting it.

        A class and its lock are one cell, so which lock is whose needs no
        reading, and every cell is one width, so a class with a long name does
        not make its cell a different shape from its neighbour's. **Background
        has a lock too**: locked, a stroke changes only pixels another class
        already holds, and painting background over an unlocked class still
        erases -- which its tooltip says, its caption being its name alone.
        """
        self.paint_buttons: dict[int, QToolButton] = {}
        self.lock_buttons: dict[int, QToolButton] = {}
        paint_group = QButtonGroup(self)
        cells: list[QWidget] = []
        for position, class_index in enumerate(self._classes):
            name = self.controller.config.name(class_index)
            erasing = class_index == BACKGROUND_CLASS

            button = QToolButton()
            button.setText(name)
            button.setObjectName(CLASS_BUTTON)
            button.setIcon(self._swatch(class_index))
            button.setIconSize(QSize(MARK, MARK))
            _larger_text(button)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setFixedHeight(PADLOCK)
            button.setCheckable(True)
            hint = (
                "Paint background, which erases the label" if erasing
                else f"Paint {name}"
            )
            if position < len(CLASS_KEYS):
                hint += f"\nShortcut: {CLASS_KEYS[position]}"
            hint += "\nOr Shift-click this class in the label"
            button.setToolTip(hint)

            def choose_class(checked: bool, index: int = class_index) -> None:
                if checked:
                    self._class_chosen(index)

            button.toggled.connect(choose_class)
            paint_group.addButton(button)
            self.paint_buttons[class_index] = button

            lock = _padlock_button()
            lock_hint = (
                "Lock background, so that a stroke only changes existing labels" if erasing
                else f"Lock {name}, so that it cannot be painted over"
            )
            if position < len(LOCK_KEYS):
                lock_hint += f"\nShortcut: {LOCK_KEYS[position]}"
            if not erasing:
                lock_hint += "\nA locks or unlocks all classes except background"
            lock.setToolTip(lock_hint)
            lock.toggled.connect(self._locks_changed)
            self.lock_buttons[class_index] = lock

            cell = QWidget()
            pair = QHBoxLayout(cell)
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(2)
            pair.addWidget(button, stretch=1)
            pair.addWidget(lock)
            cells.append(cell)

        _equalise(cells)

        # Qt leaves nothing checked; the canvas starts on the first foreground
        # class. Choosing it here makes the bar say what the brush will actually
        # write -- a bar reading "background" while the brush paints GIS is one
        # that costs somebody an image.
        self._select_class(self.canvas.paint_value)
        return cells

    def _build_actions(self) -> QWidget:
        """Build undo, redo and save as marks alone, in one row.

        Under the beam buttons, where image software keeps them. They are
        reached for without reading, so the mark is the whole of the button and
        the tooltip carries the name and the key.
        """
        self.undo_button = _action_button(
            undo_icon(MARK), "Undo the last stroke on this label\nShortcut: Ctrl+Z", self.undo
        )
        self.redo_button = _action_button(
            redo_icon(MARK), "Redo the last undone stroke\nShortcut: Ctrl+Shift+Z", self.redo
        )
        self.save_button = _action_button(
            save_icon(MARK),
            "Save all edited labels in this lamella, with the option to mark them "
            "corrected\nShortcut: Ctrl+S",
            self.save_requested.emit,
        )
        return keep_together(self.undo_button, self.redo_button, self.save_button, spacing=2)

    def _swatch(self, class_index: int):
        """Return a colour chip for one class, a square centred in a mark."""
        pixmap = QPixmap(MARK, MARK)
        pixmap.fill(Qt.GlobalColor.transparent)
        red, green, blue = self.controller.config.rgb(class_index)
        inset = (MARK - SWATCH) // 2
        painter = QPainter(pixmap)
        try:
            painter.fillRect(inset, inset, SWATCH, SWATCH, QColor(red, green, blue))
        finally:
            # A painter outliving its pixmap crashes the process.
            painter.end()
        return chip_icon(pixmap)

    def _build_timeline(self) -> QHBoxLayout:
        """Build the timeline row, over the frames worth proofreading."""
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(6)

        self.timeline = FrameTimeline(band=LABEL_BAND)
        self.timeline.frame_selected.connect(self.select_frame)
        row.addWidget(self.timeline, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(70)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
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

    def _condition_chosen(self, index: int) -> None:
        """Show the chosen imaging condition."""
        if index < 0:
            return
        self.controller.set_condition(self.variant_combo.itemData(index))
        self.refresh()

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
        opaque, against the flat ground that stands in for the acquisition.
        Switching off puts theirs back -- a filter that leaves the window
        rearranged is one you stop using.
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
            self._sweep_specks()

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

    def _class_picked(self, class_index: int) -> None:
        """Paint with the class a Shift-click found, if the configuration has it."""
        button = self.paint_buttons.get(class_index)
        if button is None:
            self.notice.emit(
                f"Class {class_index} is not in the configuration, so it cannot be painted"
            )
            return
        self._select_class(class_index)
        self.notice.emit(f"Painting {button.text()}")

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

    def toggle_lock_at(self, position: int) -> None:
        """Lock or unlock the class at *position* in the row, from the keyboard.

        Background included: a key names one padlock, and that padlock is
        background's as much as any class's.
        """
        if 0 <= position < len(self._classes):
            button = self.lock_buttons[self._classes[position]]
            button.setChecked(not button.isChecked())

    def toggle_all_locks(self, locked: bool | None = None) -> None:
        """Lock every class but background, or unlock them, from the keyboard.

        One key rather than two, because the question being asked is "protect
        what is already there while I work" and the answer is on or off. With
        any of them unlocked it locks the lot; only when all are locked does it
        let go. **Background's lock is left as it is**: locking it changes what a
        stroke can reach at all, which is a choice of its own.
        """
        foreground = [
            button for index, button in self.lock_buttons.items()
            if index != BACKGROUND_CLASS
        ]
        if locked is None:
            locked = not all(button.isChecked() for button in foreground)
        for button in foreground:
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
            button.setIcon(lock_icon(index in locked, MARK))
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
        if (self._saved_label() is None) != (self.versions_controller.saved is None):
            # The first edit lists the label as saved, and undoing the last
            # takes it off again; every stroke in between leaves the list be.
            self.refresh_versions()
        if self.controller.finding_specks:
            # Counting takes tens of milliseconds, which is nothing in a pause
            # and everything mid-drag, so it waits for the drawing to stop.
            self._speck_timer.start(SPECK_RECOUNT_MS)

    def focus_canvas(self) -> None:
        """Give the canvas the keyboard, so space pans without a click first."""
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def canvas_fit(self) -> None:
        """Scale the image to the pane."""
        self.canvas.fit()

    def hold_compare(self) -> None:
        """Show the chosen earlier version, from the keyboard.

        The press half of holding the key. The canvas turns it off again on
        release; this exists so the key has a binding to appear in the F1 list
        under, the binding machinery knowing nothing about a key you hold.
        """
        self._compare_key_held(True)

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
        self._refresh_conditions()
        self._refresh_timeline()
        self.status.setText(self.controller.status_text())
        self._refresh_edit_state()
        # Asked for before drawing, for the reason CuratePanel.refresh gives.
        self._schedule_loads()
        self.draw_current()

    def may_discard_edits(self) -> bool:
        """Ask before throwing away unsaved label edits; return whether to go on.

        Deliberately not offered as "save these first": saving runs off the
        interface thread and asks its own questions, and threading a navigation
        through the far side of that is how a half-finished save ends up moving
        files under a lamella nobody is looking at any more.
        """
        pending = self.controller.pending_count()
        if not pending:
            return True
        lamella = self.controller.lamella
        name = lamella.display_name if lamella is not None else "this lamella"
        answer = QMessageBox.warning(
            self,
            "Unsaved label edits",
            f"{pending} label{'s' if pending != 1 else ''} edited in {name} "
            f"{'have' if pending != 1 else 'has'} not been saved.\n"
            "\n"
            "Save with Ctrl+S first, or discard the edits to continue.",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Discard:
            return False
        self.controller.discard_edits()
        return True

    def stop_loading(self) -> None:
        """Stop the loader's thread, waiting for what it is reading."""
        if self.loader is not None:
            self.loader.stop()

    def _schedule_loads(self) -> None:
        """Ask the loader to prepare the frames around this one."""
        if self.loader is None:
            return
        token = self.loader.next_token()
        requests = self.controller.pending_loads()
        if requests:
            self.loader.load_batch(requests, token)
        self._sweep_specks(token)

    def _sweep_specks(self, token: int | None = None) -> None:
        """Count the specks of frames nothing has counted yet.

        On the loader's thread and behind whatever frame preparation is queued
        there, which is the order that matters: the frame on screen is what the
        reviewer is waiting for and the marks are what they will look at next.

        Sharing the loader's token means moving to another lamella abandons a
        sweep in flight. That costs the label it was reading and nothing else,
        the sweep asking each time only for what is still uncounted. A sweep
        started on its own raises the token through :meth:`_schedule_loads`, so
        the frame on screen is asked for again under it rather than left loading.
        """
        if self.loader is None:
            return
        labels = self.controller.uncounted_labels()
        if not labels:
            return
        if token is None:
            self._schedule_loads()
            return
        self.loader.count_specks(labels, token, self.controller.speck_size)

    def _on_speck_counted(self, token: int, key: str, count: int) -> None:
        """Take one label's count from the sweep and redraw the marks."""
        if self.loader is None or token != self.loader.token:
            return
        self.controller.record_speck_count(Path(key), count)
        self._refresh_timeline()

    def _on_loaded(self, token: int, frame_index: int) -> None:
        """Redraw the canvas when the frame on screen finishes preparing.

        A frame whose label was rewritten while it was being read is discarded
        by the redraw, and asked for again here rather than left loading.
        """
        if self.loader is None or token != self.loader.token:
            return
        if frame_index == self.controller.frame_index:
            self.draw_current()
            if self.controller.cached_image() is None:
                self._schedule_loads()

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
            other = self.versions_controller.comparison_pixels(self._frame_shape(working))
            if other is not None:
                return other
        return working

    def _frame_shape(self, working) -> tuple[int, int] | None:
        """Return the shape a label on this frame has, the one a conversion writes."""
        if working is not None:
            return working.shape[:2]
        prepared = self.controller.cached_image()
        if prepared is None or prepared.display is None:
            return None
        return prepared.display.image.shape[:2]

    def _version_chosen(self, index: int) -> None:
        """Note which version is being offered, and whether it can be shown.

        Redrawn only where a version was on the canvas: otherwise it holds the
        label already, and a stroke that lists the saved label would count its
        specks mid-edit, ahead of the pause the recount waits for.
        """
        self.versions_controller.select(None if index < 0 else index)
        chosen = self.versions_controller.selected
        self.versions.set_available(
            bool(chosen and chosen.available),
            refusal=chosen.not_restorable_because if chosen else "",
        )
        showing = self.showing_a_version
        self._comparing = False
        self._showing_differences = False
        if showing:
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

    def restore_version(self) -> None:
        """Put the chosen earlier version of this label back, having asked.

        What is on disk now is kept first, which makes this reversible, and the
        reviewer is told that before deciding rather than after.
        """
        versions = self.versions_controller
        chosen = versions.selected
        if chosen is None:
            self.notice.emit("No earlier version selected")
            return
        if self.controller.pending_count():
            QMessageBox.warning(
                self,
                RESTORE_TITLE,
                "Save or discard your edits first: restoring overwrites this label, and unsaved "
                "edits would be lost.",
            )
            return

        corrected = self.controller.is_reviewed()
        if QMessageBox.question(
            self,
            RESTORE_TITLE,
            f"Put back the version from {chosen.when or 'an unrecorded time'}"
            f", saved by {chosen.who}?\n\n"
            "What is on disk now is kept first, so this can itself be undone."
            + (
                "\n\nThis label is marked corrected. Restoring writes over "
                "the corrected version."
                if corrected else ""
            ),
        ) != QMessageBox.StandardButton.Yes:
            return

        try:
            versions.restore_selected(corrected=corrected, confirmed=True)
        except (ValueError, PermissionError, RuntimeError) as error:
            QMessageBox.critical(self, RESTORE_TITLE, str(error))
            return
        self.controller.discard_edits()
        self.refresh()
        self.notice.emit("Earlier version restored")

    def refresh_versions(self) -> None:
        """List the versions of the label on screen, kept and predicted."""
        record = self.controller.record
        self.versions_controller.set_label(
            self.controller.current_key,
            record.predictions if record is not None else (),
            saved=self._saved_label(),
        )
        self.versions.set_rows(
            self.versions_controller.rows, self.versions_controller.default_index
        )
        self._refresh_versions_switch()

    def _saved_label(self):
        """Return the label as it is on disk, where it has unsaved edits, or None.

        The prepared frame's own array, which painting never writes through to,
        so listing it reads no file.
        """
        key = self.controller.current_key
        if key is None or not self.controller.session.is_edited(key):
            return None
        prepared = self.controller.cached_image()
        if prepared is None or prepared.display is None:
            return None
        return prepared.display.label

    def _refresh_versions_switch(self) -> None:
        """Say how many versions this label has, without the strip being open.

        The number is what makes the switch worth reading while it is off: a
        label with nothing to compare against says so, and is not opened.
        """
        count = self.versions_controller.count
        self.versions_check.setText(
            "Compare versions" if not count else f"Compare versions ({count})"
        )

    def _versions_toggled(self, showing: bool) -> None:
        """Open or close the strip, and put the canvas back when it closes."""
        self.versions.set_open(showing)
        if not showing:
            self._comparing = False
            self._showing_differences = False
            self.draw_current()

    # -- the labelling guide -------------------------------------------------

    def _guide_toggled(self, shown: bool) -> None:
        self.guide_panel.setVisible(shown)
        self._refresh_guide()

    def next_step(self) -> None:
        """Tick the current step, from the keyboard; open the labelling guide first if closed.

        At the save step, ask for the save, which is what ticks it.
        """
        if not self.guide_check.isChecked():
            self.guide_check.setChecked(True)
            return
        if self.controller.advance_guide():
            self.save_requested.emit()
        self._refresh_guide()
        self._refresh_edit_state()

    def previous_step(self) -> None:
        """Untick the step before the current one, from the keyboard."""
        if not self.guide_check.isChecked():
            self.guide_check.setChecked(True)
            return
        self.controller.retreat_guide()
        self._refresh_guide()
        self._refresh_edit_state()

    def _step_toggled(self, step: int, ticked: bool) -> None:
        self.controller.tick_step(step, ticked)
        self._refresh_guide()
        self._refresh_edit_state()

    def _refresh_guide(self) -> None:
        """Show the progress of the label on screen."""
        if not self.guide_check.isChecked():
            return
        key = self.controller.guide_key
        guide = self.controller.labelling_guide
        if key is None:
            self.guide_panel.show_progress(None, 0)
        else:
            self.guide_panel.show_progress(guide.ticked(key), guide.current(key))

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
        # A label ticked through its labelling guide can be marked corrected unchanged.
        self.save_button.setEnabled(bool(pending or self.controller.checked_keys()))
        # Marked while anything waits, its caption being gone; re-polished, a
        # stylesheet reading a property only when a widget is polished.
        if self.save_button.property("pending") != bool(pending):
            self.save_button.setProperty("pending", bool(pending))
            self.save_button.style().unpolish(self.save_button)
            self.save_button.style().polish(self.save_button)
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

    def _refresh_conditions(self) -> None:
        """Offer the conditions this acquisition has, greyed where there is one.

        On screen either way, as in the curation view: a lamella with one
        condition per modality is every Autolamella project so far, so hiding
        it moved this row on nearly every step.
        """
        record = self.controller.record
        fill_condition_combo(
            self.variant_combo,
            self.controller.condition_choices(),
            record.condition if record is not None else None,
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
            # The canvas still holds the frame before, which no stroke is
            # recorded against any more.
            self.canvas.set_editable(False)
            self._set_badges([("loading...", "info")])
            return
        if prepared.failed:
            self.canvas.clear()
            self._set_badges([(f"could not read image: {prepared.error}", "error")])
            return

        display = prepared.display
        assert display is not None, "a frame that did not fail holds its display"
        # The label is painted into as it is; only the picture under it is
        # rescaled, or a stroke would be recorded against something that is not
        # the file.
        image = apply_intensity_range(display.image, *self._intensity_range)
        self.canvas.set_frame(image, self._label_to_draw())
        self.canvas.set_label_shown(self.controller.show_overlay)
        self.canvas.set_image_shown(self.controller.show_image)
        self._refresh_specks()
        self._refresh_guide()
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
        # Said first and loudly: an earlier version looks exactly like the
        # label, and a reviewer who cannot tell which is on screen is
        # comparing two pictures without knowing which is which.
        chosen = self.versions_controller.selected
        if self._showing_differences:
            badges.append(("showing what changed, not a label", "warning"))
        elif self._comparing and chosen is not None:
            badges.append((comparison_badge(chosen), "warning"))
        if not self.controller.series_is_labelled:
            badges.append(VIEW_ONLY_BADGE)
        else:
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
        self.badges.setText(badge_markup(badges))
