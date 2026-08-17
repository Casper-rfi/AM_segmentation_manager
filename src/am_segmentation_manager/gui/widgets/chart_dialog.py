"""A window showing one chart of the current breakdown.

The only place Qt and matplotlib meet. :mod:`am_segmentation_manager.plots`
builds figures and knows nothing about widgets; this attaches a canvas to one
and gives it matplotlib's own toolbar, which is where saving and zooming come
from rather than being rebuilt here.

Modeless, and several can be open at once: comparing two breakdowns means
having both on screen, and a modal chart would make that impossible.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
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


class ChartDialog(QDialog):
    """One chart, redrawn when the user changes how it is shaded.

    Args:
        title: The window title.
        build: Called as ``build(normalise)`` and returns a figure. Taking a
            builder rather than a figure is what lets the shading toggle
            redraw: the chart is a different chart, not a repainted one.
        normalisable: Whether the shading toggle applies. Only a grid has a
            row to take a share of.
    """

    def __init__(
        self,
        title: str,
        build: Callable[[bool], object],
        *,
        normalisable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self._build = build

        self._layout = QVBoxLayout(self)
        self._canvas = None
        self._toolbar = None

        self.normalise = QCheckBox("Shade by share of each row")
        self.normalise.setToolTip(
            "Colour each cell by its share of its own row instead of by the "
            "count. One row dwarfing the others hides the pattern inside the "
            "small ones; this asks whether the mix is the same across them."
        )
        self.normalise.setVisible(normalisable)
        self.normalise.toggled.connect(self._draw)

        self._controls = QHBoxLayout()
        self._controls.addWidget(self.normalise)
        self._controls.addStretch(1)

        self._draw()

    def _draw(self) -> None:
        """Replace the canvas with a chart drawn for the current toggle."""
        figure = self._build(self.normalise.isChecked())
        canvas, toolbar = canvas_for(figure, self)

        # Taken down rather than hidden: an old canvas left in the layout keeps
        # its figure alive, and a session of charts would accumulate every one.
        for widget in (self._toolbar, self._canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._layout.removeItem(self._controls)

        self._canvas, self._toolbar = canvas, toolbar
        self._layout.addWidget(toolbar)
        self._layout.addWidget(canvas, stretch=1)
        self._layout.addLayout(self._controls)
        canvas.draw_idle()
