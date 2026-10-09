"""A chart, the bar of what can be done to it, and the canvas swapping redrawing needs.

The only place Qt and matplotlib meet. :mod:`am_segmentation_manager.plots`
builds figures and knows nothing about widgets; this attaches a canvas to one.
**Matplotlib's own toolbar is never shown**: its Subplots and Customize change a
figure bound for a paper with nothing recording it. A hidden one still drives
pan and zoom, which are matplotlib's to do, and the bar offers those, a reset,
and the ways out -- a file, the clipboard, or the counts behind the chart. The
wheel zooms about the pointer and Space held pans, as on the Label canvas. A
chart of curves can say which point the pointer is on.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QImage
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...plots import CurvePoint, Limits, nearest_curve_point, view_limits, zoom_about
from .controls import FlowArea, FlowLayout, SegmentedControl
from .paths import choose_save_file

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: The formats a figure can be saved in, and the resolution a raster copy of it
#: is written at -- enough for a slide or a printed page.
FIGURE_FILTERS = ("PNG image (*.png)", "SVG image (*.svg)", "PDF document (*.pdf)")
FIGURE_DPI = 200

#: How much one notch of the wheel zooms.
WHEEL_ZOOM = 1.2

#: What matplotlib's toolbar calls its two modes.
PAN_MODE = "pan/zoom"
ZOOM_MODE = "zoom rect"

#: How a grid can be shaded, as the segments offering it. In the order of
#: :data:`~am_segmentation_manager.plots.SHADINGS`.
SHADING_SEGMENTS = (
    ("count", "Count", "Shade each cell by how many images it holds."),
    (
        "row", "Share of row",
        "Shade each cell by its share of its own row, so one large row stops "
        "hiding the pattern in the small ones.",
    ),
    ("column", "Share of column", "Shade each cell by its share of its own column."),
)


def canvas_for(
    figure: Figure, parent: QWidget | None = None
) -> tuple[FigureCanvasQTAgg, NavigationToolbar2QT]:
    """Return a Qt canvas drawing *figure*, and the hidden toolbar that drives it.

    Imported here rather than at module level: the backend pulls in Qt through
    matplotlib's own binding lookup, and a package installed without the plots
    extra must still import this module to report that it cannot chart.
    """
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )

    canvas = FigureCanvasQTAgg(figure)
    toolbar = NavigationToolbar2QT(canvas, parent)
    toolbar.hide()
    return canvas, toolbar


class ChartView(QWidget):
    """One chart, under a bar of how it is shaded and what can be done to it.

    Shows a message instead when there is nothing to draw, so a caller never
    has to decide between a chart and a dialog explaining its absence.
    """

    #: Emitted when the user asks for the counts behind the chart as a CSV.
    export_table_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build: Callable[[str], Figure] | None = None
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None
        self._hover = False
        self._note = None
        #: Whether the view the chart was drawn at is on the toolbar's stack,
        #: which is what Reset view goes back to.
        self._home_kept = False
        #: The view each panel was drawn at, which the wheel never zooms out beyond.
        self._drawn_at: dict[Axes, Limits] = {}
        #: The toolbar's mode when Space was pressed, or None while it is not held.
        self._mode_before_space: object | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Wrapped, not a row: a grid adds the shading to six buttons, and a row
        # that cannot wrap is a floor under the window wider than the screen.
        self.bar = FlowArea()
        bar = FlowLayout(spacing=6)
        self.bar.setLayout(bar)
        self.shading = SegmentedControl(SHADING_SEGMENTS)
        self.shading.set_current("count")
        self.shading.chosen.connect(lambda _shading: self._draw())
        self.shading.setVisible(False)
        bar.addWidget(self.shading)

        self.reset_button = self._action(
            "Reset view", "Undo any panning and zooming\nShortcut: G", self.reset_view
        )
        self.pan_button = self._action(
            "Pan",
            "Drag to move the chart; drag with the right button to stretch it.\n"
            "Holding Space pans without this.",
            self._pan, checkable=True,
        )
        self.zoom_button = self._action(
            "Zoom",
            "Drag a rectangle to zoom into it.\nThe scroll wheel zooms about the pointer without this.",
            self._zoom, checkable=True,
        )
        self.save_button = self._action(
            "Save figure...", "Write the chart as a PNG, SVG or PDF file", self._save
        )
        self.copy_button = self._action(
            "Copy figure", "Copy the chart as an image, to paste into a slide", self._copy
        )
        self.export_button = self._action(
            "Export table as CSV...",
            "Write the counts behind the chart as a CSV, as am-seg stats --output does",
            self.export_table_requested.emit,
        )
        for button in self._actions():
            bar.addWidget(button)
        self._layout.addWidget(self.bar)

        self.message = QLabel()
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        self._layout.addWidget(self.message, stretch=1)
        self._set_actions_enabled(False)

    @staticmethod
    def _action(caption: str, hint: str, slot, *, checkable: bool = False) -> QToolButton:
        button = QToolButton()
        button.setText(caption)
        button.setToolTip(hint)
        button.setCheckable(checkable)
        button.clicked.connect(lambda _checked=False: slot())
        return button

    def _actions(self) -> tuple[QToolButton, ...]:
        return (
            self.reset_button, self.pan_button, self.zoom_button,
            self.save_button, self.copy_button, self.export_button,
        )

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in self._actions():
            button.setEnabled(enabled)

    def show_message(self, text: str) -> None:
        """Show *text* in place of a chart, and take down any chart showing."""
        self._build = None
        self._take_down()
        self.shading.setVisible(False)
        self.message.setText(text)
        self.message.setVisible(True)
        self._set_actions_enabled(False)

    def show_chart(
        self, build: Callable[[str], Figure], *, shadeable: bool = False, hover: bool = False
    ) -> None:
        """Draw the figure *build* returns.

        Args:
            build: Called as ``build(shading)``, with one of
                :data:`~am_segmentation_manager.plots.SHADINGS`, and returns a
                figure. A builder rather than a figure, so choosing a shading
                draws a different chart rather than repainting this one.
            shadeable: Whether the shading applies. Only a grid has rows and
                columns to take a share of.
            hover: Whether the curve point under the pointer is named beside it.
        """
        self._build = build
        self._hover = hover
        self.message.setVisible(False)
        self.shading.setVisible(shadeable)
        self._draw()

    def _take_down(self) -> None:
        """Remove the current canvas and its toolbar.

        Taken down rather than hidden: an old canvas left in the layout keeps
        its figure alive, and a session of charts would accumulate every one.
        """
        for widget in (self._toolbar, self._canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._figure = self._canvas = self._toolbar = self._note = None
        self._home_kept = False
        self._drawn_at = {}
        self._mode_before_space = None

    def _draw(self) -> None:
        """Replace the canvas with a chart drawn for the shading chosen."""
        if self._build is None:
            return
        shading = "count" if self.shading.isHidden() else (self.shading.current() or "count")
        figure = self._build(shading)
        canvas, toolbar = canvas_for(figure, self)

        self._take_down()
        self._figure, self._canvas, self._toolbar = figure, canvas, toolbar
        self._drawn_at = view_limits(figure)
        self._layout.addWidget(canvas, stretch=1)
        # Space reaches the canvas only while it has the keyboard, which the
        # pointer coming over it gives it.
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        canvas.mpl_connect("scroll_event", lambda event: self.wheel(event.x, event.y, event.step))
        if self._hover:
            canvas.mpl_connect("motion_notify_event", lambda event: self.point_at(event.x, event.y))
            canvas.mpl_connect("figure_leave_event", lambda _event: self._say(None))
        self._set_actions_enabled(True)
        self._sync_modes()
        canvas.draw_idle()

    def point_at(self, x: float, y: float) -> None:
        """Name the curve point at canvas pixel (*x*, *y*), or take the name down.

        Nothing is named while panning or zooming, where the pointer is doing
        something else.
        """
        if self._figure is None:
            return
        busy = self._toolbar is not None and bool(self._toolbar.mode)
        self._say(None if busy else nearest_curve_point(self._figure, x, y))

    def hovered(self) -> str:
        """Return what is said of the point under the pointer, or "" where nothing is."""
        return self._note.get_text() if self._note is not None else ""

    def _say(self, point: CurvePoint | None) -> None:
        """Put the note on *point*, or take it down; a saved figure never carries one left behind."""
        if self._note is None and point is None:
            return
        if self._note is not None:
            self._note.remove()
            self._note = None
        if point is not None:
            # Towards the middle of the panel, so the note is never cut off at its edge.
            low, high = point.axes.get_xlim()
            left = point.x > (low + high) / 2
            self._note = point.axes.annotate(
                point.describe(), (point.x, point.y),
                xytext=(-10 if left else 10, 10), textcoords="offset points",
                ha="right" if left else "left", fontsize=8, annotation_clip=False,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#888888"},
                arrowprops={"arrowstyle": "-", "color": "#888888"},
            )
        if self._canvas is not None:
            self._canvas.draw_idle()

    def wheel(self, x: float, y: float, steps: float) -> None:
        """Zoom about canvas pixel (*x*, *y*) by *steps* notches, in for positive.

        Never out beyond the view the chart was drawn at, which Reset view goes back to.
        """
        if self._figure is None or self._toolbar is None or not steps:
            return
        if not self._home_kept:
            self._toolbar.push_current()
            self._home_kept = True
        zoomed = zoom_about(self._figure, x, y, WHEEL_ZOOM ** steps, self._drawn_at)
        if zoomed and self._canvas is not None:
            self._canvas.draw_idle()

    def hold_space(self, held: bool) -> None:
        """Pan for as long as Space is held, then give back the mode there was."""
        if self._toolbar is None or held == (self._mode_before_space is not None):
            return
        if held:
            self._mode_before_space = self._toolbar.mode
            if self._toolbar.mode != PAN_MODE:
                self._toolbar.pan()
        else:
            before, self._mode_before_space = self._mode_before_space, None
            if before == ZOOM_MODE:
                self._toolbar.zoom()
            elif before != PAN_MODE:
                self._toolbar.pan()
        self._sync_modes()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        """Take Space on the canvas for panning, and let go of it when the keyboard leaves."""
        canvas = self._canvas
        if canvas is None or watched is not canvas:
            return super().eventFilter(watched, event)
        kind = event.type()
        if kind == QEvent.Type.Enter:
            canvas.setFocus(Qt.FocusReason.MouseFocusReason)
        elif kind == QEvent.Type.FocusOut:
            self.hold_space(False)
        elif kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.hold_space(kind == QEvent.Type.KeyPress)
            return True
        return super().eventFilter(watched, event)

    def reset_view(self) -> None:
        """Undo any panning and zooming; nothing where no chart is drawn."""
        if self._toolbar is not None:
            self._toolbar.home()

    def _pan(self) -> None:
        if self._toolbar is not None:
            self._toolbar.pan()
        self._sync_modes()

    def _zoom(self) -> None:
        if self._toolbar is not None:
            self._toolbar.zoom()
        self._sync_modes()

    def _sync_modes(self) -> None:
        """Check Pan or Zoom as matplotlib has it, a click on one letting go of the other."""
        mode = self._toolbar.mode if self._toolbar is not None else ""
        for button, active in ((self.pan_button, PAN_MODE), (self.zoom_button, ZOOM_MODE)):
            button.blockSignals(True)
            button.setChecked(mode == active)
            button.blockSignals(False)

    def _save(self) -> None:
        """Write the chart to a file named by the user, in the format its suffix names."""
        if self._figure is None:
            return
        path = choose_save_file(
            self, "Save figure", filters=FIGURE_FILTERS, default_suffix="png"
        )
        if path is None:
            return
        try:
            self._figure.savefig(path, dpi=FIGURE_DPI)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not save the figure", str(error))
            return
        logger.info("Saved the chart to %s", path)

    def _copy(self) -> None:
        """Put the chart on the clipboard, drawn as a saved PNG would be."""
        if self._figure is None:
            return
        buffer = io.BytesIO()
        self._figure.savefig(buffer, format="png", dpi=FIGURE_DPI)
        # No format argument: PySide6 refuses one passed as bytes, the type its stubs declare.
        QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))
