"""Middle panel: the frame itself, and the grid it was milled from.

Shows the acquisition beside the same acquisition under its label, or the
summary plot, or the project's grid overview with its lamellae marked,
according to the selected view. The last is the one view that is not of the
frame at all, which is why it outlives stepping through a timeline. Also
carries the class legend and the badges that explain what is *not* being shown
-- a missing label, an auto-generated empty one, a resampled one, a series with
no labels at all, a file that would not read, or an approximate position.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig
from ...display import (
    DEFAULT_ALPHA,
    DEFAULT_INTENSITY_RANGE,
    apply_intensity_range,
    compose_overlay,
)
from ...layout import SERIES_LABELS
from ...overview import LamellaMarker
from ..cache import FrameImage, FramePlot
from ..controller import DisplayMode
from .common import (
    LABEL_BADGES,
    VIEW_ONLY_BADGE,
    badge_markup,
    choice_count_badge,
    class_legend_html,
    fill_choice_combo,
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
)
from .grid_view import GridView
from .image_view import ImageView, LinkedImageViews
from .theme import BADGE_STYLES, GROUP_GAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)

#: How far a figure must be magnified past the pane-sized copy before the copy
#: read whole is worth drawing instead. Not exactly 1, because returning to fit
#: lands a floating-point hair either side of it -- and landing on the wrong
#: side leaves the fitted view resampled, which is the one view that has to be
#: exact. Nothing flaps at this margin: a wheel step is 25%.
PLOT_DETAIL_MAGNIFICATION = 1.01

#: What the shared slot beside the series is for, in each view. The frame
#: views choose an imaging condition there; the Grid view has none, and a
#: project can hold several overviews, so it chooses between those instead.
VARIANT_HINT = (
    "Which imaging condition to display. This is a view choice only: a "
    "state change follows the label, so images sharing a label always "
    "change together whichever one you are looking at."
)
#: What a view that cannot be chosen says on hover, instead of what it shows.
UNAVAILABLE_VIEWS = {
    "sem": "No SEM image of this frame",
    "fib": "No FIB image of this frame",
    "plot": "No plot for this frame",
    "grid": "No grid overview for this project",
}
OVERVIEW_HINT = (
    "Which of the project's grid overviews to show. The earliest that can "
    "place the lamellae opens; a later one may show them milled or coated."
)

class ImagePanel(QWidget):
    """The frame view, its mode controls, legend and badges."""

    #: Emitted with the view the user selected.
    mode_changed = Signal(str)
    #: Emitted when the overlay is switched on or off.
    overlay_toggled = Signal(bool)
    #: Emitted with the new label opacity, 0.0-1.0.
    alpha_changed = Signal(float)
    #: Emitted with the intensity limits asked for, as ``(low, high)``.
    intensity_range_changed = Signal(int, int)

    #: Emitted when the acquisition under the label is shown or hidden.
    image_toggled = Signal(bool)
    #: Emitted with the imaging condition to display; empty string means base.
    variant_changed = Signal(str)
    #: Emitted with the series whose timeline to show.
    series_changed = Signal(str)
    #: Emitted with the position of the grid overview to show.
    overview_chosen = Signal(int)
    #: Emitted when the user asks for the displayed file's path, or to open its
    #: folder. The panel does not know which file it is showing -- it is handed
    #: a prepared image -- so the window answers these from the controller.
    path_requested = Signal()
    reveal_requested = Signal()

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        self.config = config
        self._alpha = DEFAULT_ALPHA
        #: The intensities shown as black and as white. A property of the
        #: *view*, held here rather than in the controller and never reaching
        #: an export -- see :func:`~...display.apply_intensity_range`. Kept
        #: across frames on purpose: comparing one frame with the next is what
        #: it is for, and a setting lost on every step could not do that.
        self._intensity_range = DEFAULT_INTENSITY_RANGE
        #: Whether the figure on show describes the whole stage; see
        #: :meth:`set_plot_scope`.
        self._plot_is_shared = False
        #: Whether the slot beside the series is choosing between overviews
        #: rather than imaging conditions; see :meth:`set_overviews`.
        self._choosing_overview = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(3)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_path_menu)

        # -- view controls -------------------------------------------------
        # Wrapped, not a row: these already needed more width than the centre
        # pane opens at, so the Fit button and part of the opacity slider were
        # off the edge before the intensity control was added to them. A row
        # that wraps costs a second line only on the panes too narrow to hold
        # one, where the alternative is controls nobody can reach.
        self.controls = FlowArea()
        controls = FlowLayout(spacing=6)
        self.controls.setLayout(controls)
        # One widget rather than three loose buttons: in a flow layout the run
        # could wrap apart, leaving what is plainly one choice reading as two.
        self.mode_control = SegmentedControl([
            (mode, caption, f"Show the {caption} view\nShortcut: {key}")
            for mode, caption, key in (
                ("sem", "SEM", "1"), ("fib", "FIB", "2"), ("plot", "Plots", "3"),
                ("grid", "Grid", "4"),
            )
        ])
        self.mode_control.chosen.connect(self.mode_changed.emit)
        #: The segments, so a caller can enable and disable them one at a time.
        self.mode_buttons = self.mode_control.buttons
        #: What each says while it can be chosen, put back when it can again.
        self._mode_hints = {
            mode: button.toolTip() for mode, button in self.mode_buttons.items()
        }
        self.mode_buttons["sem"].setChecked(True)

        # -- series --------------------------------------------------------
        # Beside the modality buttons because it is the same kind of choice --
        # which images the timeline is made of -- and before the variant,
        # because it decides which frames exist where the variant only decides
        # which file of a frame is drawn. One at a time rather than a set: a
        # timeline mixing alignment runs with polishing frames is not something
        # anyone proofreads.
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Scale the view to the pane\nShortcut: G")
        self.fit_button.clicked.connect(self.fit)
        # Which view, and then fit that view to the pane: the only two controls
        # here that act on the pane itself rather than on what is drawn in it.
        # Left in its own width, or a lone button stretches to the segments
        # above it and reads as one more of them.
        controls.addWidget(keep_paired(
            (self.mode_control,), (self.fit_button,),
            align={(1, 0): Qt.AlignmentFlag.AlignLeft},
        ))

        controls.addSpacing(GROUP_GAP)
        self.series_label = QLabel("Series")
        self.series_combo = ElidingCombo()
        self.series_combo.setMinimumWidth(110)
        self.series_combo.setToolTip(
            "Which images the timeline is made of. Autolamella writes "
            "reference and alignment images as well as the Adaptive Polishing "
            "ones; each is its own timeline, shown one at a time."
        )
        self.series_label.setToolTip(self.series_combo.toolTip())
        self.series_combo.currentIndexChanged.connect(self._series_chosen)

        # -- imaging condition ---------------------------------------------
        # A dropdown rather than buttons: variant names run long
        # (``8_fi_16``) and a stage can hold five of them.
        self.variant_label = QLabel("Variant")
        self.variant_combo = ElidingCombo()
        self.variant_combo.setMinimumWidth(110)
        self.variant_combo.setToolTip(VARIANT_HINT)
        self.variant_label.setToolTip(self.variant_combo.toolTip())
        self.variant_combo.currentIndexChanged.connect(self._variant_chosen)
        # Which images the timeline is made of, above which file of one is
        # drawn: the narrower question under the wider one.
        controls.addWidget(keep_paired(
            (self.series_label, self.series_combo, choice_count_badge(self.series_combo)),
            (self.variant_label, self.variant_combo, choice_count_badge(self.variant_combo)),
        ))

        controls.addSpacing(GROUP_GAP)
        self.overlay_check = ToggleSwitch("Show label")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip(
            "Draw the label over the right-hand image\nShortcut: V"
        )
        self.overlay_check.toggled.connect(self.overlay_toggled.emit)

        self.image_check = ToggleSwitch("Show image")
        self.image_check.setChecked(True)
        self.image_check.setToolTip(
            "Draw the label over the acquisition, or over a flat dark "
            "ground.\nThat ground is how a few-pixel annotation is found, and "
            "its edge is the frame's; the left pane still shows the image."
        )
        self.image_check.toggled.connect(self.image_toggled.emit)
        # What is drawn, one switch above the other: they answer the same
        # question of the two layers, and stacked they cost one caption's width
        # rather than two.
        controls.addWidget(keep_paired((self.overlay_check,), (self.image_check,)))

        # A switch ends at its caption, where a tick box carried padding past
        # it, so a switch beside a caption reads as one phrase without a gap.
        # The pairing above is what separates the groups now; this only has to
        # be wider than the flow's own spacing.
        controls.addSpacing(GROUP_GAP)
        opacity_label = QLabel("Opacity")
        self.alpha_slider = ValueSlider()
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(DEFAULT_ALPHA * 100))
        self.alpha_slider.setFixedWidth(SLIDER_WIDTH)
        self.alpha_slider.setToolTip("How strongly the label colour is drawn over the image")
        opacity_label.setToolTip(self.alpha_slider.toolTip())
        self.alpha_slider.valueChanged.connect(self._alpha_moved)

        low, high = DEFAULT_INTENSITY_RANGE
        hint = (
            "The intensities shown as black and as white; outside them is "
            "clipped.\nChanges nothing on disk and reaches no export. "
            "Double-click a handle to put it back."
        )
        # Fiji's words for these two, which is the vocabulary the people using
        # this already have. "Black / white" named the *result* and left the
        # reader to work out which end each slider was. Abbreviated because
        # this caption sets the width of the paired block it sits in, and the
        # words spelled out cost the row a third line on a 1440px screen.
        range_label = QLabel("Min / max")
        range_label.setToolTip(hint)
        self.intensity_slider = RangeSlider(0, 255, low, high)
        self.intensity_slider.setToolTip(hint)
        self.intensity_slider.range_changed.connect(self._range_moved)
        # How the label is drawn, above how the image under it is: two sliders,
        # aligned, and the long caption no longer sets the row's width alone.
        controls.addWidget(keep_paired(
            (opacity_label, self.alpha_slider),
            (range_label, self.intensity_slider),
        ))

        layout.addWidget(self.controls)

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
        self.grid_view = GridView()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.pair)
        self.stack.addWidget(self.plot_view)
        self.stack.addWidget(self.grid_view)
        layout.addWidget(self.stack, stretch=1)

        # -- badges and legend ---------------------------------------------
        # Wrapped and elided, never full width: a line that cannot shorten is a
        # floor under the whole tab, and the splitter takes that width from the
        # lamella list. The caption runs to a project path and a file name.
        self.badges = QLabel("")
        self.badges.setTextFormat(Qt.TextFormat.RichText)
        self.badges.setWordWrap(True)
        layout.addWidget(self.badges)

        self.legend = QLabel("")
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self._build_legend()
        layout.addWidget(self.legend)

        self.status = ElidingLabel()
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

    def _range_moved(self, low: int, high: int) -> None:
        """Take the new limits and pass them on.

        The two are held apart by the control rather than here: a reader who
        drags one into the other should see it refuse, not watch the picture
        turn into a threshold and wonder what they did.
        """
        self._intensity_range = (low, high)
        self.intensity_range_changed.emit(low, high)

    def reset_intensity_range(self) -> None:
        """Show the whole range again."""
        self.intensity_slider.reset()

    #: Which page of the stack each view is drawn on. The two acquisition
    #: views share the linked pair, so they are one page.
    _PAGES: dict[str, int] = {"sem": 0, "fib": 0, "plot": 1, "grid": 2}

    def set_mode(self, mode: DisplayMode) -> None:
        """Reflect the current view without re-emitting a change."""
        self.mode_control.set_current(mode)
        self.stack.setCurrentIndex(self._PAGES.get(mode, 0))
        if mode != "grid":
            self._offer_overviews(False)

    def set_available_modes(self, modes: Sequence[str]) -> None:
        """Disable the views this frame has nothing to show for, saying why."""
        for mode, button in self.mode_buttons.items():
            available = mode in modes
            button.setEnabled(available)
            button.setToolTip(
                self._mode_hints[mode] if available else UNAVAILABLE_VIEWS[mode]
            )

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
        """Emit the chosen condition, using an empty string for the base.

        Or the chosen overview, while the slot is offering those instead.
        """
        if index < 0:
            return
        if self._choosing_overview:
            self.overview_chosen.emit(int(self.variant_combo.itemData(index)))
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
        self._offer_overviews(False)
        described = descriptions or {}
        fill_choice_combo(
            self.variant_combo,
            [
                ("base" if variant is None else described.get(variant, variant), variant)
                for variant in variants
            ],
            current,
        )

    def set_overviews(self, captions: list[str], current: int) -> None:
        """Offer the project's grid overviews in the imaging condition's place.

        The same slot rather than a control of its own, which would appear and
        move the row each time the Grid view is chosen. Leaving the Grid view
        hands the slot back -- see :meth:`set_mode`.

        Args:
            captions: One per overview, in the order the controller holds them.
            current: The position of the one shown.
        """
        self._offer_overviews(True)
        fill_choice_combo(
            self.variant_combo,
            [(caption, index) for index, caption in enumerate(captions)],
            current,
        )

    def _offer_overviews(self, offering: bool) -> None:
        """Name the shared slot for what it is choosing between."""
        if offering == self._choosing_overview:
            return
        self._choosing_overview = offering
        self.variant_label.setText("Overview" if offering else "Variant")
        hint = OVERVIEW_HINT if offering else VARIANT_HINT
        self.variant_combo.setToolTip(hint)
        self.variant_label.setToolTip(hint)
        # Emptied either way: an overview's caption left behind reads as a
        # condition, and a condition as an overview.
        fill_choice_combo(self.variant_combo, [], None)

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

    def apply_defaults(self, values) -> None:
        """Open the controls at what the settings say, before anything is drawn.

        Called once, at construction: these decide what a control *opens* at,
        so applying them to a control somebody is already using would undo a
        choice they made this session.
        """
        self.alpha_slider.setValue(int(values["label_opacity"]))

    def show_image(
        self,
        frame_image: FrameImage | None,
        show_overlay: bool,
        with_image: bool = True,
        labelled: bool = True,
    ) -> None:
        """Draw an acquisition, or explain why there is nothing to draw.

        *with_image* governs the overlay pane only. The left pane is the
        acquisition as recorded, which is the whole reason there are two.
        *labelled* is False for a series with no labels, whose badge then says
        the frame is view-only rather than that its label is missing.
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
        assert display is not None, "a frame that did not fail holds its display"
        # Once, for both panes: the two must not disagree about how the same
        # acquisition looks, and at neutral this hands back the cached array
        # itself rather than a copy of it.
        image = apply_intensity_range(display.image, *self._intensity_range)
        self.pair.left.set_array(image)

        if display.label is not None and show_overlay:
            overlay = compose_overlay(
                image, display.label, self.config,
                alpha=self._alpha, hide_image=not with_image,
            )
            self.pair.right.set_array(overlay)
        else:
            self.pair.right.set_array(image)

        self._set_badges(
            intensity_badge(self._intensity_range) + self._label_badges(display, labelled)
        )

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
        assert stepping is not None, "a plot that did not fail holds its array"
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

    def show_grid(
        self,
        image,
        markers: tuple[LamellaMarker, ...] = (),
        current: str | None = None,
        error: str | None = None,
        unplaced: bool = False,
        reason: str | None = None,
    ) -> None:
        """Draw the project's grid overview with its lamellae marked.

        Args:
            image: The prepared overview, or None where there is none to draw.
            markers: Where each lamella falls, as fractions of the picture.
            current: The lamella on screen, drawn heavier and named.
            error: Why the overview could not be read, where that is the reason
                there is nothing to draw.
            unplaced: The lamella on screen is a real one that this overview
                does not mark, so the reviewer is told rather than left looking
                for a ring that is not there.
            reason: Why nothing is placed on this overview, said in place of
                the approximation, there being nothing approximate on screen.

        **The badge says the marks are approximate**, every time rather than
        once: a mark is where a lamella is only to within what
        ``overview.reproject`` ignores, and a reader arriving at this view for
        the first time has no other way to know that.
        """
        if image is None:
            self.grid_view.clear_markers()
            self.grid_view.clear()
            self._set_badges(
                [(f"overview could not be read: {error}", "error")] if error
                else [("no grid overview for this project", "info")]
            )
            return
        self.grid_view.set_array(image)
        height, width = image.shape[:2]
        self.grid_view.set_markers(markers, current, (width, height))

        badges = (
            [(reason, "info")] if reason is not None
            else [("positions are approximate", "warning")]
        )
        if unplaced:
            badges.append(("this lamella has no recorded position", "warning"))
        off = sum(1 for marker in markers if not marker.inside)
        if off:
            badges.append((f"{off} lamella(e) fall outside this overview", "info"))
        self._set_badges(badges)

    def show_loading(self) -> None:
        """Say that the frame is still being prepared."""
        self._set_badges([("loading...", "info")])

    def show_empty(self) -> None:
        """Clear the panel when there is no frame at all."""
        self.pair.clear()
        self.plot_view.clear()
        self.grid_view.clear_markers()
        self.grid_view.clear()
        self._set_badges([])
        self.status.setText("")

    def set_status(self, text: str) -> None:
        """Show the line describing where the user is."""
        self.status.setText(text)

    def _label_badges(self, display, labelled: bool = True) -> list[tuple[str, str]]:
        """Return the badges describing a frame's label."""
        if not labelled:
            return [VIEW_ONLY_BADGE]
        badges: list[tuple[str, str]] = []
        text, level = LABEL_BADGES[display.label_status]
        if text:
            badges.append((text, level))
        if display.label_resized:
            badges.append(("label resized to match image", "warning"))
        return badges

    def _set_badges(self, badges: list[tuple[str, str]]) -> None:
        """Show what is worth saying about the frame's label."""
        self.badges.setText(badge_markup(badges))
