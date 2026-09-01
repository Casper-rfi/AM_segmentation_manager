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
from ...layout import SERIES_LABELS
from ..cache import FrameImage, FramePlot
from ..controller import DisplayMode
from .common import (
    LABEL_BADGES,
    ToggleSwitch,
    class_legend_html,
    fill_choice_combo,
)
from .image_view import ImageView, LinkedImageViews
from .theme import BADGE_STYLES, SWITCH_RUN_GAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)

#: How far a figure must be magnified past the pane-sized copy before the copy
#: read whole is worth drawing instead. Not exactly 1, because returning to fit
#: lands a floating-point hair either side of it -- and landing on the wrong
#: side leaves the fitted view resampled, which is the one view that has to be
#: exact. Nothing flaps at this margin: a wheel step is 25%.
PLOT_DETAIL_MAGNIFICATION = 1.01

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
    #: Emitted with the series whose timeline to show.
    series_changed = Signal(str)
    #: Emitted when the user asks for the displayed file's path, or to open its
    #: folder. The panel does not know which file it is showing -- it is handed
    #: a prepared image -- so the window answers these from the controller.
    path_requested = Signal()
    reveal_requested = Signal()

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        self.config = config
        self._alpha = DEFAULT_ALPHA
        #: Whether the figure on show describes the whole stage; see
        #: :meth:`set_plot_scope`.
        self._plot_is_shared = False

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

        # -- series --------------------------------------------------------
        # Beside the modality buttons because it is the same kind of choice --
        # which images the timeline is made of -- and before the variant,
        # because it decides which frames exist where the variant only decides
        # which file of a frame is drawn. One at a time rather than a set: a
        # timeline mixing alignment runs with polishing frames is not something
        # anyone proofreads.
        controls.addSpacing(16)
        self.series_label = QLabel("Series")
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(140)
        self.series_combo.setToolTip(
            "Which images the timeline is made of. Autolamella writes "
            "reference and alignment images as well as the Adaptive Polishing "
            "ones; each is its own timeline, shown one at a time."
        )
        self.series_label.setToolTip(self.series_combo.toolTip())
        self.series_combo.currentIndexChanged.connect(self._series_chosen)
        controls.addWidget(self.series_label)
        controls.addWidget(self.series_combo)

        # -- imaging condition ---------------------------------------------
        # A dropdown rather than buttons: variant names run long
        # (``8_fi_16``) and a stage can hold five of them.
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

        controls.addSpacing(20)
        self.overlay_check = ToggleSwitch("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip(
            "Draw the label over the right-hand image\nShortcut: V"
        )
        self.overlay_check.toggled.connect(self.overlay_toggled.emit)
        controls.addWidget(self.overlay_check)

        # A switch ends at its caption, where a tick box carried padding past
        # it. Without this the two switches and the slider's label ran together
        # into one phrase -- "Show label Show image Opacity" -- and nothing said
        # which caption belonged to which control.
        controls.addSpacing(SWITCH_RUN_GAP)
        self.image_check = ToggleSwitch("Show image")
        self.image_check.setChecked(True)
        self.image_check.setToolTip(
            "Draw the label over the acquisition, or over black.\n"
            "Over black is how an annotation of a few pixels is found; the "
            "left pane still shows the image."
        )
        self.image_check.toggled.connect(self.image_toggled.emit)
        controls.addWidget(self.image_check)

        controls.addSpacing(SWITCH_RUN_GAP)
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
        # Only ever minified when the whole copy is on screen; the pane-sized
        # one is drawn 1:1. Acquisitions keep nearest neighbour.
        self.plot_view.set_smooth_scaling(True)
        # The two copies of the figure on screen: prepared for this pane, and
        # read whole where it has been read.
        self._plot_copies: tuple = (None, None)
        self._swapping = False
        self.plot_view.zoomed.connect(self._plot_zoomed)
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

    def _series_chosen(self, index: int) -> None:
        """Emit the chosen series."""
        if index < 0:
            return
        self.series_changed.emit(self.series_combo.itemData(index))

    def set_series(self, series: list[str], current: str) -> None:
        """Offer the series the current lamella holds.

        **Always on screen, greyed where there is nothing to choose.** Hiding
        it moved every control beside it each time the reviewer stepped onto a
        lamella of one series, and a lamella holding none of the series being
        shown left no way back to the one it does hold.

        Args:
            series: The series present, in display order.
            current: The one on screen.
        """
        fill_choice_combo(
            self.series_combo,
            [(SERIES_LABELS.get(name, name), name) for name in series],
            current,
        )

    def _variant_chosen(self, index: int) -> None:
        """Emit the chosen condition, using an empty string for the base."""
        if index < 0:
            return
        value = self.variant_combo.itemData(index)
        self.variant_changed.emit(value or "")

    def set_variants(
        self,
        variants: list[str | None],
        current: str | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Offer the imaging conditions available for the current view.

        Greyed rather than hidden where there is nothing to choose, for the
        reason :meth:`set_series` gives: every Autolamella project so far has
        one condition per modality, so hiding it moved this row on nearly every
        step between projects.

        Args:
            variants: Conditions present, base first. ``None`` is the base.
            current: The condition on screen.
            descriptions: Human-readable names from the project metadata, keyed
                by variant. A variant with no description shows its raw name.
        """
        described = descriptions or {}
        fill_choice_combo(
            self.variant_combo,
            [
                ("base" if variant is None else described.get(variant, variant), variant)
                for variant in variants
            ],
            current,
        )

    def _build_legend(self) -> None:
        """Render a colour swatch and name for each foreground class."""
        self.legend.setText(class_legend_html(self.config))

    def _plot_zoomed(self) -> None:
        """Reconsider which copy to draw after the figure is zoomed.

        Guarded because drawing one emits this again: the swap is a zoom in
        every sense Qt understands.
        """
        if self._swapping or self._plot_copies[0] is None:
            return
        self._swapping = True
        try:
            self._draw_plot()
        finally:
            self._swapping = False

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

    def show_plot(
        self, frame_plot: FramePlot | None, whole: FramePlot | None = None
    ) -> None:
        """Draw a summary plot, or explain its absence.

        Args:
            frame_plot: The copy prepared for this pane, which is what is drawn
                until the figure is magnified past its own resolution.
            whole: The same figure read whole, where it has been read. Used
                only where it would show more than *frame_plot* can.
        """
        if frame_plot is None:
            self.plot_view.clear()
            self._plot_copies = (None, None)
            self._set_badges([("no plot for this frame", "info")])
            return
        if frame_plot.failed:
            self.plot_view.clear()
            self._plot_copies = (None, None)
            self._set_badges([(f"could not read plot: {frame_plot.error}", "error")])
            return
        stepping = frame_plot.plot
        # A figure of another size is another picture and is fitted; every
        # plot of one lamella is normally the same size, which is what makes
        # stepping leave the view alone.
        drawn = self._plot_copies[0]
        same = drawn is not None and drawn.shape[:2] == stepping.shape[:2]
        self._plot_copies = (
            stepping,
            whole.plot if whole is not None and not whole.failed else None,
        )
        self._draw_plot(same_picture=same)
        self._set_badges(
            [("summarises the whole stage", "info")] if self._plot_is_shared else []
        )

    def set_plot_scope(self, shared: bool) -> None:
        """Say whether the figure on show describes the stage or one frame.

        An alignment run writes one figure for the several images it took to
        converge, so the same picture is shown on each of its frames. Read as
        that frame's own it would be badly misleading, and nothing about the
        figure itself says otherwise.
        """
        self._plot_is_shared = shared

    def _draw_plot(self, *, same_picture: bool = True) -> None:
        """Draw whichever copy of the figure shows the most at this zoom.

        The pane-sized copy is drawn at exactly one array pixel per physical
        pixel, which is the only scale that does not resample it -- so at fit,
        where a plot is nearly always looked at, nothing is filtered and the
        figure is as sharp as the file. Past that the pane copy has nothing
        left to show and would only be stretched, so the whole one takes over.
        """
        stepping, whole = self._plot_copies
        if stepping is None:
            return
        ratio = self._plot_pixel_ratio(stepping)
        if whole is not None and self._plot_magnification() > PLOT_DETAIL_MAGNIFICATION:
            self.plot_view.set_array(whole, same_picture=same_picture)
            return
        self.plot_view.set_array(
            stepping, device_pixel_ratio=ratio, native=True, same_picture=same_picture
        )

    def _plot_pixel_ratio(self, stepping) -> float:
        """Return the physical pixels per logical pixel this copy holds.

        Read off the copy and the pane it is drawn in rather than asked of the
        screen. The copy was reduced to fit the pane measured in physical
        pixels, so whichever of its edges touches that box gives the ratio back
        exactly -- and a copy prepared for a pane of another size still fills
        the one it is drawn in, where trusting the screen would leave it at
        half or twice the size until the copy was prepared again.

        Never below 1: a figure smaller than the pane is drawn at its own
        pixels rather than stretched to fill it.
        """
        viewport = self.plot_view.viewport().rect()
        if not viewport.width() or not viewport.height():
            return 1.0
        return max(
            1.0,
            stepping.shape[1] / viewport.width(),
            stepping.shape[0] / viewport.height(),
        )

    def _plot_magnification(self) -> float:
        """Return how far the figure is drawn past the pane copy's resolution.

        Measured against that copy rather than read off the transform, so it
        means the same thing whichever copy is on screen -- which is what stops
        the two swapping back and forth around the threshold.
        """
        stepping, _ = self._plot_copies
        item = self.plot_view._item.boundingRect()
        if stepping is None or not item.width():
            return 0.0
        drawn = item.width() * self.plot_view.transform().m11()
        return drawn / (stepping.shape[1] / self._plot_pixel_ratio(stepping))

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
