"""A window showing one chart of the current breakdown.

Modeless, and several can be open at once: comparing two breakdowns means
having both on screen, and a modal chart would make that impossible. The Export
tab can also show a chart inline, which is the same :class:`ChartView` -- this
is the one you detach when a second is wanted beside it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from qtpy.QtWidgets import QDialog, QVBoxLayout, QWidget

from .chart_view import ChartView, canvas_for

logger = logging.getLogger(__name__)

__all__ = ["ChartDialog", "canvas_for"]


class ChartDialog(QDialog):
    """One chart in a window of its own.

    Args:
        title: The window title.
        build: Called as ``build(normalise)`` and returns a figure.
        normalisable: Whether the shading toggle applies. Only a grid has a row
            to take a share of.
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

        self.view = ChartView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view)
        self.view.show_chart(build, normalisable=normalisable)

    @property
    def normalise(self):
        """The shading toggle, which lives on the view this dialog wraps."""
        return self.view.normalise

    @property
    def _canvas(self):
        """The current canvas, for tests asserting that a redraw replaced it."""
        return self.view._canvas
