"""The strip listing the versions of the label being proofread.

Sits under the proofreading canvas and is **closed unless somebody opens it**, and
whoever owns it decides whether it shows. It offers what is worth doing with a
version -- read the list, look at it, put it back -- a prediction being a version
converted rather than restored. A comparison is drawn on the *same* canvas,
segmentation being compared by flicking between two pictures in one place.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from .controls import ToggleSwitch

logger = logging.getLogger(__name__)

__all__ = ["COMPARE_KEY", "COMPARE_KEY_CODE", "DIFFERENCE_CLASS", "VersionsBar"]

#: The class value a difference mask is drawn with. Deliberately outside any
#: configured class, so it renders in the colour reserved for one nothing
#: defines: what is on screen then is plainly not a segmentation class.
DIFFERENCE_CLASS: int = 255

#: What the compare control says, and the key that does it. **Not V**, which
#: already shows and hides the label in this tab and in Curate. B for "before",
#: under the hand not holding the stylus.
COMPARE_KEY: str = "B"
COMPARE_KEY_CODE = Qt.Key.Key_B


class VersionsBar(QWidget):
    """Lists kept versions and asks for one to be compared against or restored.

    Signals:
        selected: A version was chosen, by its index in the list, or -1 for
            none.
        comparing: The view should show the chosen version rather than the
            label being proofread.
        differences: The view should mark where the two disagree.
        restore_requested: The chosen version should be put back.
    """

    selected = Signal(int)
    comparing = Signal(bool)
    differences = Signal(bool)
    restore_requested = Signal()

    def __init__(self) -> None:
        """Build the strip, hidden until a label has versions to show."""
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.caption = QLabel("Earlier versions")
        row.addWidget(self.caption)

        self.chooser = QComboBox()
        self.chooser.setMinimumWidth(320)
        self.chooser.setToolTip(
            "Versions of this segmentation, newest first: those kept when the "
            "label was replaced, and predictions, shown as converted.\n"
            "The percentage is how much a kept version differs from the one "
            "before it."
        )
        self.chooser.currentIndexChanged.connect(self._chosen)
        row.addWidget(self.chooser, stretch=1)

        self.compare_button = QPushButton(f"Hold to compare ({COMPARE_KEY})")
        self.compare_button.setCheckable(True)
        self.compare_button.setToolTip(
            "Show the chosen version in place of the label, so the two can be "
            f"flicked between. Hold {COMPARE_KEY} to do the same from the canvas."
        )
        self.compare_button.toggled.connect(self.comparing.emit)
        row.addWidget(self.compare_button)

        # A switch rather than a tick box: it is on or off by itself, which is
        # the distinction every other standalone control here already makes.
        self.difference_check = ToggleSwitch("Show differences")
        self.difference_check.setToolTip(
            "Mark every pixel where the chosen version and the label disagree."
        )
        self.difference_check.toggled.connect(self.differences.emit)
        row.addWidget(self.difference_check)

        self.restore_button = QPushButton("Restore…")
        self.restore_button.setToolTip(
            "Put the chosen version back. What is here now is kept first, so "
            "this can itself be undone."
        )
        self.restore_button.clicked.connect(self.restore_requested.emit)
        row.addWidget(self.restore_button)

        for widget in self.findChildren(QWidget):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.set_rows(())
        self.setVisible(False)

    def set_rows(self, rows, current: int = 0) -> None:
        """List *rows*, choosing the one at *current*.

        Whether the strip shows is left to :meth:`set_open`.
        """
        blocked = self.chooser.blockSignals(True)
        self.chooser.clear()
        for row in rows:
            self.chooser.addItem(row.describe())
            if not row.available:
                index = self.chooser.count() - 1
                self.chooser.setItemData(
                    index,
                    "The record names this version, but its pixels are not "
                    "here: they were kept in a copy that did not bring its "
                    "archive along. It can be neither shown nor put back.",
                    Qt.ItemDataRole.ToolTipRole,
                )
        self.chooser.setCurrentIndex(current if 0 <= current < len(rows) else -1)
        self.chooser.blockSignals(blocked)
        self._chosen(self.chooser.currentIndex())

    def set_open(self, open_: bool) -> None:
        """Show or hide the strip, stopping any comparison on the way out.

        Hiding while a version is on the canvas would leave the wrong picture
        up with nothing left on screen to say so.
        """
        if not open_:
            self.stop_comparing()
            self.difference_check.setChecked(False)
        self.setVisible(open_)

    def set_available(self, available: bool, *, restorable: bool = True) -> None:
        """Say whether the chosen version can be looked at, and put back.

        Args:
            available: Whether its pixels are here to be drawn.
            restorable: Whether putting it back is the right way to use it. A
                prediction is not: it becomes a label by being converted, which
                records which model made it.
        """
        self.compare_button.setEnabled(available)
        self.difference_check.setEnabled(available)
        self.restore_button.setEnabled(available and restorable)
        self.restore_button.setToolTip(
            "Put the chosen version back. What is here now is kept first, so "
            "this can itself be undone."
            if restorable else
            "A prediction is not put back but converted, which writes a label "
            "from it and records the model. Convert from the Curate tab."
        )
        if not available:
            self.stop_comparing()
            self.difference_check.setChecked(False)

    def stop_comparing(self) -> None:
        """Go back to showing the label being proofread."""
        if self.compare_button.isChecked():
            self.compare_button.setChecked(False)

    def set_comparing(self, comparing: bool) -> None:
        """Show the chosen version, or the label, without asking again."""
        if self.compare_button.isChecked() != comparing:
            self.compare_button.setChecked(comparing)

    @property
    def chosen(self) -> int:
        """Return which version is chosen, or -1 for none."""
        return self.chooser.currentIndex()

    def _chosen(self, index: int) -> None:
        """Pass on a choice, having stopped showing the last one."""
        self.stop_comparing()
        self.difference_check.setChecked(False)
        self.selected.emit(index)
