"""The Label tab: one acquisition at full resolution, ready to be proofread.

Reads its own controller and draws from it, so the window wires this tab up
rather than running it. Nothing here writes to disk; the tab is a viewer until
the drawing tools arrive.

The lamella is not chosen here. It is handed in from the Curate tab, so the
collection has one place where "which lamella am I on" is decided.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...config import BACKGROUND_CLASS
from ...display import DEFAULT_ALPHA
from ...layout import MODALITIES
from ..label_controller import LabelController
from .common import LABEL_BADGES, chip_icon, class_legend_html
from .frame_strip import FrameTimeline
from .label_canvas import (
    DEFAULT_BRUSH_RADIUS,
    MAX_BRUSH_RADIUS,
    MIN_BRUSH_RADIUS,
    TOOL_BRUSH,
    TOOL_FILL,
    LabelCanvas,
)
from .theme import BADGE_STYLES

logger = logging.getLogger(__name__)

#: Said whenever anything is waiting, because an edit that exists only in this
#: window is one closing it would throw away.
UNSAVED_NOTE = "not yet written to disk"

#: The swatch drawn beside each class in the brush's class list.
SWATCH = 12


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        layout.addLayout(self._build_controls())

        self.canvas = LabelCanvas(controller.config)
        self.canvas.edited.connect(self._stroke_finished)
        layout.addWidget(self.canvas, stretch=1)

        layout.addLayout(self._build_tools())

        self.badges = QLabel("")
        self.badges.setTextFormat(Qt.RichText)
        layout.addWidget(self.badges)

        self.legend = QLabel(class_legend_html(controller.config))
        self.legend.setTextFormat(Qt.RichText)
        layout.addWidget(self.legend)

        self.status = QLabel("")
        self.status.setStyleSheet(BADGE_STYLES["info"])
        layout.addWidget(self.status)

        layout.addLayout(self._build_timeline())

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

        controls.addSpacing(20)
        self.overlay_check = QCheckBox("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip("Draw the label over the image\nShortcut: V")
        self.overlay_check.toggled.connect(self._overlay_toggled)
        controls.addWidget(self.overlay_check)

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

        controls.addStretch()
        return controls

    def _build_tools(self) -> QHBoxLayout:
        """Build the row of drawing tools."""
        tools = QHBoxLayout()

        self.tool_buttons: dict[str, QRadioButton] = {}
        tool_group = QButtonGroup(self)
        for tool, caption, key, hint in (
            (TOOL_BRUSH, "Brush", "B", "Draw where you drag"),
            (TOOL_FILL, "Fill", "G", "Flood the area of one class you click in"),
        ):
            button = QRadioButton(caption)
            button.setToolTip(f"{hint}\nShortcut: {key}")
            button.toggled.connect(
                lambda checked, t=tool: checked and self._tool_chosen(t)
            )
            tool_group.addButton(button)
            tools.addWidget(button)
            self.tool_buttons[tool] = button
        brush_button = self.tool_buttons[TOOL_BRUSH]
        brush_button.blockSignals(True)
        brush_button.setChecked(True)
        brush_button.blockSignals(False)

        tools.addSpacing(16)
        tools.addWidget(QLabel("Paint"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(150)
        self.class_combo.setToolTip(
            "Which class the brush writes. Background is how a stroke erases."
            "\nShortcuts: D and F step through the classes."
        )
        for class_index in self.controller.config.classes:
            name = self.controller.config.name(class_index)
            if class_index == BACKGROUND_CLASS:
                name = f"{name} (erase)"
            self.class_combo.addItem(self._swatch(class_index), name, class_index)
        self.class_combo.currentIndexChanged.connect(self._class_chosen)
        tools.addWidget(self.class_combo)

        tools.addSpacing(16)
        self.brush_label = brush_label = QLabel("Brush")
        tools.addWidget(brush_label)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(MIN_BRUSH_RADIUS, MAX_BRUSH_RADIUS)
        self.brush_slider.setValue(DEFAULT_BRUSH_RADIUS)
        self.brush_slider.setFixedWidth(120)
        self.brush_slider.setToolTip(
            "Brush radius in image pixels\nShortcuts: [ and ] step the size"
        )
        brush_label.setToolTip(self.brush_slider.toolTip())
        self.brush_slider.valueChanged.connect(self._brush_resized)
        tools.addWidget(self.brush_slider)
        self.brush_value = QLabel(str(DEFAULT_BRUSH_RADIUS))
        self.brush_value.setMinimumWidth(28)
        tools.addWidget(self.brush_value)

        tools.addSpacing(16)
        lock_label = QLabel("Lock")
        lock_label.setToolTip("Locked classes cannot be painted over by any tool")
        tools.addWidget(lock_label)
        self.lock_checks: dict[int, QCheckBox] = {}
        for class_index in self.controller.config.foreground_classes:
            check = QCheckBox(self.controller.config.name(class_index))
            check.setToolTip(lock_label.toolTip())
            check.toggled.connect(self._locks_changed)
            tools.addWidget(check)
            self.lock_checks[class_index] = check

        tools.addSpacing(16)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo the last stroke on this label\nShortcut: Ctrl+Z")
        self.undo_button.clicked.connect(self.undo)
        tools.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Redo the last undone stroke\nShortcut: Ctrl+Shift+Z")
        self.redo_button.clicked.connect(self.redo)
        tools.addWidget(self.redo_button)

        tools.addSpacing(16)
        self.save_button = QPushButton("Save labels...")
        self.save_button.setToolTip(
            "Write every label edited in this lamella, and choose whether to "
            "mark them corrected\nShortcut: Ctrl+S"
        )
        self.save_button.clicked.connect(self.save_requested.emit)
        tools.addWidget(self.save_button)

        tools.addStretch()
        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet(BADGE_STYLES["warning"])
        tools.addWidget(self.pending_label)
        return tools

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

    def _overlay_toggled(self, shown: bool) -> None:
        self.controller.show_overlay = shown
        self.canvas.set_label_shown(shown)

    def _alpha_moved(self, value: int) -> None:
        self.canvas.set_label_opacity(value / 100.0)

    def _tool_chosen(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        # The brush size means nothing to the fill, so it is switched off rather
        # than left looking as though it applies.
        for widget in (self.brush_label, self.brush_slider, self.brush_value):
            widget.setEnabled(tool == TOOL_BRUSH)

    def set_tool(self, tool: str) -> None:
        """Choose the tool, from the keyboard."""
        button = self.tool_buttons.get(tool)
        if button is not None:
            button.setChecked(True)

    def _class_chosen(self, index: int) -> None:
        if index >= 0:
            self.canvas.set_paint_value(self.class_combo.itemData(index))

    def step_class(self, delta: int) -> None:
        """Move to the next or previous class, from the keyboard."""
        count = self.class_combo.count()
        if count:
            self.class_combo.setCurrentIndex(
                (self.class_combo.currentIndex() + delta) % count
            )

    def _brush_resized(self, radius: int) -> None:
        self.canvas.set_brush_radius(radius)
        self.brush_value.setText(str(self.canvas.brush_radius))

    def step_brush(self, delta: int) -> None:
        """Grow or shrink the brush, from the keyboard."""
        self.brush_slider.setValue(self.brush_slider.value() + delta)

    def _locks_changed(self) -> None:
        self.canvas.set_protected(
            {index for index, check in self.lock_checks.items() if check.isChecked()}
        )

    def _stroke_finished(self, patch) -> None:
        """Record a stroke and show what it changed."""
        self.controller.record_edit(patch)
        self._refresh_timeline()
        self._refresh_edit_state()

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
