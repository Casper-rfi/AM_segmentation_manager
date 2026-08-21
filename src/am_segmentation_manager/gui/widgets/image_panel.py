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
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
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
from .common import LABEL_BADGES, class_legend_html
from .image_view import ImageView, LinkedImageViews
from .theme import BADGE_STYLES

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)

class ImagePanel(QWidget):
    """The frame view, its mode controls, legend and badges."""

    #: Emitted with the view the user selected.
    mode_changed = Signal(str)
    #: Emitted when the overlay is switched on or off.
    overlay_toggled = Signal(bool)
    #: Emitted with the new label opacity, 0.0-1.0.
    alpha_changed = Signal(float)

    #: Emitted when the acquisition under the label is shown or hidden.
    image_toggled = Signal(bool)
    #: Emitted with the imaging condition to display; empty string means base.
    variant_changed = Signal(str)
    #: Emitted when the user asks for the displayed file's path, or to open its
    #: folder. The panel does not know which file it is showing -- it is handed
    #: a prepared image -- so the window answers these from the controller.
    path_requested = Signal()
    reveal_requested = Signal()

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        self.config = config
        self._alpha = DEFAULT_ALPHA

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_path_menu)

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

        # -- imaging condition ---------------------------------------------
        # A dropdown rather than buttons: variant names run long
        # (``8_fi_16``) and a stage can hold five of them. Hidden entirely
        # unless the current modality has a choice, which is the usual case.
        controls.addSpacing(16)
        self.variant_label = QLabel("Variant")
        self.variant_combo = QComboBox()
        self.variant_combo.setMinimumWidth(110)
        self.variant_combo.setToolTip(
            "Which imaging condition to display. This is a view choice only: a "
            "state change follows the label, so images sharing a label always "
            "change together whichever one you are looking at."
        )
        self.variant_label.setToolTip(self.variant_combo.toolTip())
        self.variant_combo.currentIndexChanged.connect(self._variant_chosen)
        controls.addWidget(self.variant_label)
        controls.addWidget(self.variant_combo)
        self._show_variant_controls(False)

        controls.addSpacing(20)
        self.overlay_check = QCheckBox("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip(
            "Draw the label over the right-hand image\nShortcut: Space"
        )
        self.overlay_check.toggled.connect(self.overlay_toggled.emit)
        controls.addWidget(self.overlay_check)

        self.image_check = QCheckBox("Show image")
        self.image_check.setChecked(True)
        self.image_check.setToolTip(
            "Draw the label over the acquisition, or over black.\n"
            "Over black is how an annotation of a few pixels is found; the "
            "left pane still shows the image."
        )
        self.image_check.toggled.connect(self.image_toggled.emit)
        controls.addWidget(self.image_check)

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

        controls.addSpacing(20)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Scale the view to the pane\nShortcut: G")
        self.fit_button.clicked.connect(self.fit)
        controls.addWidget(self.fit_button)

        controls.addStretch()
        layout.addLayout(controls)

        # -- the views -----------------------------------------------------
        self.pair = LinkedImageViews("image", "image + label")
        self.plot_view = ImageView()
        # Plots are held as rendered, which is several times the pane, so the
        # view is always minifying one. Acquisitions keep nearest neighbour.
        self.plot_view.set_smooth_scaling(True)
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

    def _show_path_menu(self, point) -> None:
        """Offer the displayed file's path on right-click.

        Which file that is depends on the mode and the variant dropdown, so the
        panel only reports that it was asked; the window resolves it.
        """
        menu = QMenu(self)
        copy = menu.addAction("Copy image path")
        reveal = menu.addAction("Reveal image in file manager")
        chosen = menu.exec_(self.mapToGlobal(point))
        if chosen is copy:
            self.path_requested.emit()
        elif chosen is reveal:
            self.reveal_requested.emit()

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

    # -- imaging conditions ------------------------------------------------

    def _variant_chosen(self, index: int) -> None:
        """Emit the chosen condition, using an empty string for the base."""
        if index < 0:
            return
        value = self.variant_combo.itemData(index)
        self.variant_changed.emit(value or "")

    def _show_variant_controls(self, visible: bool) -> None:
        self.variant_label.setVisible(visible)
        self.variant_combo.setVisible(visible)

    def set_variants(
        self,
        variants: list[str | None],
        current: str | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Offer the imaging conditions available for the current view.

        The control is hidden unless there is a genuine choice, so a project
        with one condition per modality -- every Autolamella project so far --
        shows nothing extra.

        Args:
            variants: Conditions present, base first. ``None`` is the base.
            current: The condition on screen.
            descriptions: Human-readable names from the project metadata, keyed
                by variant. A variant with no description shows its raw name.
        """
        if len(variants) < 2:
            self._show_variant_controls(False)
            return

        described = descriptions or {}
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        for variant in variants:
            if variant is None:
                caption = "base"
            else:
                caption = described.get(variant, variant)
            self.variant_combo.addItem(caption, variant)
        position = variants.index(current) if current in variants else 0
        self.variant_combo.setCurrentIndex(position)
        self.variant_combo.blockSignals(False)
        self._show_variant_controls(True)

    def _build_legend(self) -> None:
        """Render a colour swatch and name for each foreground class."""
        self.legend.setText(class_legend_html(self.config))

    def fit(self) -> None:
        """Fit whichever view is on screen.

        The panel holds two of them and only one is ever visible, so a caller
        that reaches for a particular view fits the one the user is not looking
        at half the time.
        """
        if self.stack.currentWidget() is self.plot_view:
            self.plot_view.fit()
        else:
            self.pair.fit()

    # -- rendering ---------------------------------------------------------

    def show_image(
        self,
        frame_image: FrameImage | None,
        show_overlay: bool,
        with_image: bool = True,
    ) -> None:
        """Draw an acquisition, or explain why there is nothing to draw.

        *with_image* governs the overlay pane only. The left pane is the
        acquisition as recorded, which is the whole reason there are two.
        """
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
                display.image, display.label, self.config,
                alpha=self._alpha, hide_image=not with_image,
            )
            self.pair.right.set_array(overlay)
        else:
            self.pair.right.set_array(display.image)

        self._set_badges(self._label_badges(display))

    def show_plot(self, frame_plot: FramePlot | None, *, detailed: bool = False) -> None:
        """Draw a summary plot, or explain its absence.

        Args:
            frame_plot: What to draw.
            detailed: This is the same figure already on screen, read whole.
                Swapped in without moving the view, so it arrives as the
                picture becoming legible rather than as the pane jumping.
        """
        if frame_plot is None:
            self.plot_view.clear()
            self._set_badges([("no plot for this frame", "info")])
            return
        if frame_plot.failed:
            self.plot_view.clear()
            self._set_badges([(f"could not read plot: {frame_plot.error}", "error")])
            return
        # Fitted on arrival, and left where it is when the full-resolution
        # copy replaces it.
        self.plot_view.set_array(frame_plot.plot, same_picture=detailed)
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
