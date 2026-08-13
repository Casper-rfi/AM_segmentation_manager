"""Middle panel: the frame itself.

Shows the acquisition beside the same acquisition under its label, or the
summary plot, according to the selected view. Also carries the class legend and
the badges that explain what is *not* being shown -- a missing label, an
auto-generated empty one, a resampled one, or a file that would not read.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig
from ...display import DEFAULT_ALPHA, compose_overlay
from ..cache import FrameImage, FramePlot
from ..controller import DisplayMode
from .image_view import ImageView, LinkedImageViews

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...frames import LamellaFrame

logger = logging.getLogger(__name__)

BADGE_STYLES = {
    "warning": "color: #c08a34; font-weight: bold;",
    "error": "color: #c0504d; font-weight: bold;",
    "info": "color: #888888;",
}

LABEL_BADGES = {
    "missing": ("no label", "warning"),
    "empty": ("empty label - not yet annotated", "warning"),
    "present": ("", "info"),
}


class ImagePanel(QWidget):
    """The frame view, its mode controls, legend and badges."""

    #: Emitted with the view the user selected.
    mode_changed = Signal(str)
    #: Emitted when the overlay is switched on or off.
    overlay_toggled = Signal(bool)
    #: Emitted with the new label opacity, 0.0-1.0.
    alpha_changed = Signal(float)

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        self.config = config
        self._alpha = DEFAULT_ALPHA

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        # -- view controls -------------------------------------------------
        controls = QHBoxLayout()
        self.mode_buttons: dict[str, QRadioButton] = {}
        group = QButtonGroup(self)
        for mode, caption, key in (
            ("sem", "SEM", "1"), ("fib", "FIB", "2"), ("plot", "Plots", "3"),
        ):
            button = QRadioButton(caption)
            button.setToolTip(f"Show the {caption} view\nShortcut: {key}")
            button.toggled.connect(
                lambda checked, m=mode: checked and self.mode_changed.emit(m)
            )
            group.addButton(button)
            controls.addWidget(button)
            self.mode_buttons[mode] = button
        self.mode_buttons["sem"].setChecked(True)

        controls.addSpacing(20)
        self.overlay_check = QCheckBox("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip(
            "Draw the label over the right-hand image\nShortcut: Space"
        )
        self.overlay_check.toggled.connect(self.overlay_toggled.emit)
        controls.addWidget(self.overlay_check)

        opacity_label = QLabel("Opacity")
        controls.addWidget(opacity_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(DEFAULT_ALPHA * 100))
        self.alpha_slider.setFixedWidth(120)
        self.alpha_slider.setToolTip("How strongly the label colour is drawn over the image")
        opacity_label.setToolTip(self.alpha_slider.toolTip())
        self.alpha_slider.valueChanged.connect(self._alpha_moved)
        controls.addWidget(self.alpha_slider)
        controls.addStretch()
        layout.addLayout(controls)

        # -- the views -----------------------------------------------------
        self.pair = LinkedImageViews("image", "image + label")
        self.plot_view = ImageView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.pair)
        self.stack.addWidget(self.plot_view)
        layout.addWidget(self.stack, stretch=1)

        # -- badges and legend ---------------------------------------------
        self.badges = QLabel("")
        self.badges.setTextFormat(Qt.RichText)
        layout.addWidget(self.badges)

        self.legend = QLabel("")
        self.legend.setTextFormat(Qt.RichText)
        self._build_legend()
        layout.addWidget(self.legend)

        self.status = QLabel("")
        self.status.setStyleSheet(BADGE_STYLES["info"])
        layout.addWidget(self.status)

    # -- controls ----------------------------------------------------------

    def _alpha_moved(self, value: int) -> None:
        self._alpha = value / 100.0
        self.alpha_changed.emit(self._alpha)

    def set_mode(self, mode: DisplayMode) -> None:
        """Reflect the current view without re-emitting a change."""
        button = self.mode_buttons[mode]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
        self.stack.setCurrentIndex(1 if mode == "plot" else 0)

    def set_available_modes(self, modes: list[str]) -> None:
        """Disable the views this frame has nothing to show for."""
        for mode, button in self.mode_buttons.items():
            button.setEnabled(mode in modes)

    def _build_legend(self) -> None:
        """Render a colour swatch and name for each foreground class."""
        parts = []
        for class_index in self.config.foreground_classes:
            red, green, blue = self.config.rgb(class_index)
            parts.append(
                f'<span style="color: rgb({red},{green},{blue});">&#9632;</span> '
                f"{self.config.name(class_index)}"
            )
        self.legend.setText("&nbsp;&nbsp;".join(parts))

    # -- rendering ---------------------------------------------------------

    def show_image(self, frame_image: FrameImage | None, show_overlay: bool) -> None:
        """Draw an acquisition, or explain why there is nothing to draw."""
        if frame_image is None:
            self.pair.clear()
            self._set_badges([("no image for this modality", "warning")])
            return
        if frame_image.failed:
            self.pair.clear()
            self._set_badges([(f"could not read image: {frame_image.error}", "error")])
            return

        display = frame_image.display
        self.pair.left.set_array(display.image)

        if display.label is not None and show_overlay:
            overlay = compose_overlay(
                display.image, display.label, self.config, alpha=self._alpha
            )
            self.pair.right.set_array(overlay)
        else:
            self.pair.right.set_array(display.image)

        self._set_badges(self._label_badges(display))

    def show_plot(self, frame_plot: FramePlot | None) -> None:
        """Draw a summary plot, or explain its absence."""
        if frame_plot is None:
            self.plot_view.clear()
            self._set_badges([("no plot for this frame", "info")])
            return
        if frame_plot.failed:
            self.plot_view.clear()
            self._set_badges([(f"could not read plot: {frame_plot.error}", "error")])
            return
        self.plot_view.set_array(frame_plot.plot)
        self._set_badges([])

    def show_loading(self) -> None:
        """Say that the frame is still being prepared."""
        self._set_badges([("loading...", "info")])

    def show_empty(self) -> None:
        """Clear the panel when there is no frame at all."""
        self.pair.clear()
        self.plot_view.clear()
        self._set_badges([])
        self.status.setText("")

    def set_status(self, text: str) -> None:
        """Show the line describing where the user is."""
        self.status.setText(text)

    def _label_badges(self, display) -> list[tuple[str, str]]:
        """Return the badges describing a frame's label."""
        badges: list[tuple[str, str]] = []
        text, level = LABEL_BADGES[display.label_status]
        if text:
            badges.append((text, level))
        if display.label_resized:
            badges.append(("label resized to match image", "warning"))
        return badges

    def _set_badges(self, badges: list[tuple[str, str]]) -> None:
        if not badges:
            self.badges.setText("")
            return
        parts = [
            f'<span style="{BADGE_STYLES[level]}">{text}</span>' for text, level in badges
        ]
        self.badges.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts))
