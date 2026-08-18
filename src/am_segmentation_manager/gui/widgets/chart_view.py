"""A chart, and the canvas swapping that redrawing one needs.

The only place Qt and matplotlib meet. :mod:`am_segmentation_manager.plots`
builds figures and knows nothing about widgets; this attaches a canvas to one
and gives it matplotlib's own toolbar, which is where saving and zooming come
from rather than being rebuilt here.

Used twice, which is why it is a widget rather than part of the dialog: inline
in the Export tab, where the breakdown can be read as a table or a chart, and in
:mod:`chart_dialog`, which detaches one so two breakdowns can be compared side
by side.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def canvas_for(figure, parent: QWidget | None = None):
    """Return a Qt canvas drawing *figure*, and the toolbar that drives it.

    Imported here rather than at module level: the backend pulls in Qt through
    matplotlib's own binding lookup, and a package installed without the plots
    extra must still import this module to report that it cannot chart.
    """
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )

    canvas = FigureCanvasQTAgg(figure)
    return canvas, NavigationToolbar2QT(canvas, parent)


class ChartView(QWidget):
    """One chart, redrawn when the user changes how it is shaded.

    Shows a message instead when there is nothing to draw, so a caller never
    has to decide between a chart and a dialog explaining its absence.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build: Callable[[bool], object] | None = None
        self._canvas = None
        self._toolbar = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.message = QLabel()
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self._layout.addWidget(self.message)

        self.normalise = QCheckBox("Shade by share of each row")
        self.normalise.setToolTip(
            "Colour each cell by its share of its own row instead of by the "
            "count, so one large row stops hiding the pattern in the small ones."
        )
        self.normalise.setVisible(False)
        self.normalise.toggled.connect(self._draw)

        self._controls = QHBoxLayout()
        self._controls.addWidget(self.normalise)
        self._controls.addStretch(1)
        self._layout.addLayout(self._controls)

    def show_message(self, text: str) -> None:
        """Show *text* in place of a chart, and take down any chart showing."""
        self._build = None
        self._take_down()
        self.normalise.setVisible(False)
        self.message.setText(text)
        self.message.setVisible(True)

    def show_chart(
        self, build: Callable[[bool], object], *, normalisable: bool = False
    ) -> None:
        """Draw the figure *build* returns.

        Args:
            build: Called as ``build(normalise)`` and returns a figure. Taking a
                builder rather than a figure is what lets the shading toggle
                redraw: the chart is a different chart, not a repainted one.
            normalisable: Whether the shading toggle applies. Only a grid has a
                row to take a share of.
        """
        self._build = build
        self.message.setVisible(False)
        self.normalise.setVisible(normalisable)
        self._draw()

    def _take_down(self) -> None:
        """Remove the current canvas and toolbar.

        Taken down rather than hidden: an old canvas left in the layout keeps
        its figure alive, and a session of charts would accumulate every one.
        """
        for widget in (self._toolbar, self._canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._canvas = self._toolbar = None

    def _draw(self) -> None:
        """Replace the canvas with a chart drawn for the current toggle."""
        if self._build is None:
            return
        figure = self._build(self.normalise.isChecked())
        canvas, toolbar = canvas_for(figure, self)

        self._take_down()
        self._layout.removeItem(self._controls)

        self._canvas, self._toolbar = canvas, toolbar
        self._layout.addWidget(toolbar)
        self._layout.addWidget(canvas, stretch=1)
        self._layout.addLayout(self._controls)
        canvas.draw_idle()
