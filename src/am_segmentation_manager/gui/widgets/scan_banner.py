"""The panel shown over the window while the collection is being read.

A scan holds the interface thread, so for as long as it runs the window cannot
answer, repaint or say anything -- and on a large collection that is fifteen
seconds of an empty project list above two black panes, which reads as a hang
rather than as work. This says what is happening and how far it has got.

**Nothing here processes events**, for the reason `busy_dialog.py` sets out.
The banner is repainted synchronously by the scan that is blocking, one widget
at a time, so no queued signal can be delivered part-way through a scan.

An overlay parented to the window rather than a dialog: a modal dialog needs
the event loop to appear at all, and the event loop is exactly what is not
running.
"""
from __future__ import annotations

from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QColor, QFont, QPainter
from qtpy.QtWidgets import QWidget

from .branding import PRODUCT_NAME, logo_pixmap
from .theme import ACCENT, BORDER_LIGHT, HEADING, MUTED, WINDOW

#: The mark's side, in logical pixels. Large enough to be the thing the eye
#: lands on: this is the one moment the window has nothing else to show.
LOGO = 96

#: The progress bar's size and how far it sits below the text.
BAR_WIDTH = 320
BAR_HEIGHT = 6
BAR_RADIUS = 3

#: Gaps down the stack, in logical pixels.
GAP_AFTER_LOGO = 18
GAP_AFTER_NAME = 6
GAP_AFTER_PHASE = 18


class ScanBanner(QWidget):
    """Covers the window while a scan runs, and says how far it has got.

    Painted rather than laid out, so that one :meth:`QWidget.repaint` puts the
    whole of it on screen: a layout of child widgets would need each of them
    repainted in turn, and any of them missed is a banner that reports a
    position it no longer holds.
    """

    def __init__(self, parent: QWidget) -> None:
        """Build the banner over *parent*, hidden until :meth:`begin`."""
        super().__init__(parent)
        self._logo = logo_pixmap(LOGO)
        self._phase = ""
        self._done = 0
        self._total = 0
        self.hide()

    # -- what it says ------------------------------------------------------

    def begin(self, phase: str) -> None:
        """Show the banner over the whole of its parent, saying *phase*.

        Shown and painted here rather than left to the next turn of the event
        loop, there being no next turn until the scan this announces returns.
        """
        self._phase = phase
        self._done = 0
        self._total = 0
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.repaint()

    def set_phase(self, phase: str) -> None:
        """Say what is happening now, keeping whatever count is showing."""
        self._phase = phase
        self._repaint_if_visible()

    def set_progress(self, done: int, total: int) -> None:
        """Report *done* of *total*, which turns the bar into a count.

        Matches the ``on_progress(done, total)`` contract the worker thread
        uses, so a scan reports the same way wherever it is run from.
        """
        self._done = done
        self._total = total
        self._repaint_if_visible()

    def end(self) -> None:
        """Take the banner away."""
        self.hide()

    def _repaint_if_visible(self) -> None:
        """Repaint now, unless nothing is showing.

        ``isHidden`` rather than ``isVisible``: offscreen nothing is ever
        visible, and a banner that refused to update there would be one the
        tests could not read.
        """
        if not self.isHidden():
            self.repaint()

    # -- what it looks like ------------------------------------------------

    @property
    def message(self) -> str:
        """Return the line under the name, count and all.

        Separate from the painting so what it says can be asserted without
        reading pixels; that it is *drawn* is a separate question.
        """
        if self._total <= 0:
            return self._phase
        return f"{self._phase}  {self._done} of {self._total}"

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        """Draw the mark, the name, what is happening, and how far it has got."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.fillRect(self.rect(), QColor(WINDOW))
            self._paint_stack(painter)
        finally:
            # Always: a QPainter outliving its paint device crashes the process
            # rather than reporting an error.
            painter.end()

    def _paint_stack(self, painter: QPainter) -> None:
        """Draw the mark and everything under it, centred as one block."""
        centre = self.rect().center().x()
        top = self._stack_top()

        if not self._logo.isNull():
            painter.drawPixmap(
                centre - self._logo.width() // 2, top, self._logo
            )
            top += self._logo.height() + GAP_AFTER_LOGO

        name = QFont(self.font())
        name.setPointSizeF(name.pointSizeF() * 1.5)
        name.setBold(True)
        painter.setFont(name)
        painter.setPen(QColor(HEADING))
        top = self._paint_line(painter, PRODUCT_NAME, top)
        top += GAP_AFTER_NAME

        painter.setFont(self.font())
        painter.setPen(QColor(MUTED))
        top = self._paint_line(painter, self.message, top)
        self._paint_bar(painter, centre, top + GAP_AFTER_PHASE)

    def _paint_line(self, painter: QPainter, text: str, top: int) -> int:
        """Draw one centred line at *top*, returning where the next one starts."""
        height = painter.fontMetrics().height()
        painter.drawText(
            QRect(0, top, self.width(), height), Qt.AlignHCenter | Qt.AlignTop, text
        )
        return top + height

    def _paint_bar(self, painter: QPainter, centre: int, top: int) -> None:
        """Draw the track, and the filled part of it once there is a count.

        No bar at all until the scan can say how much work there is: an
        indeterminate bar has to animate to mean anything, and animation needs
        the event loop this is drawn instead of.
        """
        if self._total <= 0:
            return
        painter.setPen(Qt.NoPen)
        track = QRect(centre - BAR_WIDTH // 2, top, BAR_WIDTH, BAR_HEIGHT)
        painter.setBrush(QColor(BORDER_LIGHT))
        painter.drawRoundedRect(track, BAR_RADIUS, BAR_RADIUS)
        filled = min(max(self._done, 0), self._total) / self._total
        if filled <= 0:
            return
        painter.setBrush(QColor(ACCENT))
        painter.drawRoundedRect(
            QRect(track.left(), track.top(), round(track.width() * filled), BAR_HEIGHT),
            BAR_RADIUS,
            BAR_RADIUS,
        )

    def _stack_top(self) -> int:
        """Return where the block starts, so the whole of it sits centred."""
        metrics = self.fontMetrics()
        height = (
            (self._logo.height() + GAP_AFTER_LOGO if not self._logo.isNull() else 0)
            + round(metrics.height() * 1.5) + GAP_AFTER_NAME
            + metrics.height() + GAP_AFTER_PHASE
            + BAR_HEIGHT
        )
        return max(0, (self.height() - height) // 2)
