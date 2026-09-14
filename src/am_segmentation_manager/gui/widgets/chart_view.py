"""A chart, the bar of what can be done to it, and the canvas swapping redrawing needs.

The only place Qt and matplotlib meet. :mod:`am_segmentation_manager.plots`
builds figures and knows nothing about widgets; this attaches a canvas to one.
**Matplotlib's own toolbar is never shown**: its Subplots and Customize change a
figure bound for a paper with nothing recording it. A hidden one still drives
pan and zoom, which are matplotlib's to do, and the bar offers those, a reset,
and the ways out -- a file, the clipboard, or the counts behind the chart.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QImage
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .controls import SegmentedControl
from .paths import choose_save_file

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: The formats a figure can be saved in, and the resolution a raster copy of it
#: is written at -- enough for a slide or a printed page.
FIGURE_FILTERS = ("PNG image (*.png)", "SVG image (*.svg)", "PDF document (*.pdf)")
FIGURE_DPI = 200

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

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.shading = SegmentedControl(SHADING_SEGMENTS)
        self.shading.set_current("count")
        self.shading.chosen.connect(lambda _shading: self._draw())
        self.shading.setVisible(False)
        bar.addWidget(self.shading)
        bar.addStretch(1)

        self.reset_button = self._action("Reset view", "Undo any panning and zooming", self._reset)
        self.pan_button = self._action(
            "Pan", "Drag to move the chart; drag with the right button to stretch it",
            self._pan, checkable=True,
        )
        self.zoom_button = self._action(
            "Zoom", "Drag a rectangle to zoom into it", self._zoom, checkable=True
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
        self._layout.addLayout(bar)

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
        self, build: Callable[[str], Figure], *, shadeable: bool = False
    ) -> None:
        """Draw the figure *build* returns.

        Args:
            build: Called as ``build(shading)``, with one of
                :data:`~am_segmentation_manager.plots.SHADINGS`, and returns a
                figure. A builder rather than a figure, so choosing a shading
                draws a different chart rather than repainting this one.
            shadeable: Whether the shading applies. Only a grid has rows and
                columns to take a share of.
        """
        self._build = build
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
        self._figure = self._canvas = self._toolbar = None

    def _draw(self) -> None:
        """Replace the canvas with a chart drawn for the shading chosen."""
        if self._build is None:
            return
        shading = "count" if self.shading.isHidden() else (self.shading.current() or "count")
        figure = self._build(shading)
        canvas, toolbar = canvas_for(figure, self)

        self._take_down()
        self._figure, self._canvas, self._toolbar = figure, canvas, toolbar
        self._layout.addWidget(canvas, stretch=1)
        self._set_actions_enabled(True)
        self._sync_modes()
        canvas.draw_idle()

    def _reset(self) -> None:
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
        for button, active in ((self.pan_button, "pan/zoom"), (self.zoom_button, "zoom rect")):
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
