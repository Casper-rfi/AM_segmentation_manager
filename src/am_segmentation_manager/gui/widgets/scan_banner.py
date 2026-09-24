"""The panel shown over the window while the collection is being read.

A scan holds the interface thread, so this says what is happening and how far it
has got. **Each update turns the event loop once**, a repaint reaching the screen
only when the main thread returns to the loop, which a blocking scan never does.
**So input is stopped by this widget**: ``ExcludeUserInputEvents`` delivers a press
already posted, so the banner filters the application and swallows every press,
key and wheel meant for the window while it is up, and every request to close it
-- what makes turning the loop safe where ``busy_dialog.py`` says it is not.
Timers and queued signals still run, so a modal dialog can open during a scan,
and it alone is let through. An overlay rather than a dialog, a modal dialog
needing the event loop to appear at all.
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QEventLoop, QRect, Qt
from qtpy.QtGui import QColor, QFont, QPainter
from qtpy.QtWidgets import QApplication, QWidget

from ...identity import PRODUCT_NAME
from .branding import logo_pixmap
from .theme import BORDER_LIGHT, HEADING, MUTED, WINDOW, active_accent

#: What the banner swallows while it is up. Everything a person can do: the
#: shortcut events as well as the key ones, or F1 would still open the list
#: over a window mid-scan.
BLOCKED_EVENTS = frozenset({
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseButtonRelease,
    QEvent.Type.MouseButtonDblClick,
    QEvent.Type.KeyPress,
    QEvent.Type.KeyRelease,
    QEvent.Type.Wheel,
    QEvent.Type.Shortcut,
    QEvent.Type.ShortcutOverride,
})

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


def _inside_modal_dialog(watched) -> bool:
    """Whether *watched* is, or is part of, the modal dialog now open."""
    modal = QApplication.activeModalWidget()
    if modal is None:
        return False
    if isinstance(watched, QWidget):
        return watched is modal or modal.isAncestorOf(watched)
    return watched is modal.windowHandle()


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
        #: Whether the application filter is installed, so that begin() twice
        #: over does not stack two of them and end() can take the right one off.
        self._filtering = False
        self.hide()

    # -- what it says ------------------------------------------------------

    def begin(self, phase: str) -> None:
        """Show the banner over the whole of its parent, saying *phase*.

        Put on the screen here rather than left to the next turn of the event
        loop, there being no next turn until the scan this announces returns.
        """
        self._phase = phase
        self._done = 0
        self._total = 0
        parent = self.parentWidget()
        assert parent is not None, "the banner covers the widget it was made on"
        self.setGeometry(parent.rect())
        self.show()
        self.raise_()
        application = QApplication.instance()
        if application is not None and not self._filtering:
            application.installEventFilter(self)
            self._filtering = True
        self._present()

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
        """Take the banner away, and let input through again.

        Called from a ``finally``, so a scan that raised does not leave the
        window unable to be typed into.
        """
        application = QApplication.instance()
        if application is not None and self._filtering:
            application.removeEventFilter(self)
        self._filtering = False
        self.hide()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        """Swallow anything a person does to the window while the banner is up.

        A modal dialog open meanwhile, a failure being reported say, still
        answers: it opened inside the scan, which waits until it is dismissed.
        A close request is marked ignored as well as swallowed, Qt taking a
        close event it was never told about as accepted.
        """
        kind = event.type()
        if kind == QEvent.Type.Close and watched is self.window():
            event.ignore()
            return True
        if kind in BLOCKED_EVENTS and not _inside_modal_dialog(watched):
            return True
        return super().eventFilter(watched, event)

    def _repaint_if_visible(self) -> None:
        """Put the new state on the screen, unless nothing is showing.

        ``isHidden`` rather than ``isVisible``: offscreen nothing is ever
        visible, and a banner that refused to update there would be one the
        tests could not read.
        """
        if not self.isHidden():
            self._present()

    def _present(self) -> None:
        """Paint, and let the platform put the frame on the screen.

        The one pass of the event loop this costs is what the module docstring
        is about. Input is asked to stay out and, where it does not, is
        swallowed by :meth:`eventFilter`.
        """
        self.repaint()
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

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
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
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
            QRect(0, top, self.width(), height), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text
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
        painter.setPen(Qt.PenStyle.NoPen)
        track = QRect(centre - BAR_WIDTH // 2, top, BAR_WIDTH, BAR_HEIGHT)
        painter.setBrush(QColor(BORDER_LIGHT))
        painter.drawRoundedRect(track, BAR_RADIUS, BAR_RADIUS)
        filled = min(max(self._done, 0), self._total) / self._total
        if filled <= 0:
            return
        painter.setBrush(QColor(active_accent().colour))
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
