"""The strip listing the kept versions of the label being proofread.

Sits under the proofreading canvas and costs nothing when a label has no kept
versions, which is the ordinary case: the whole strip hides itself. When there
are versions it offers the three things worth doing with one -- read the list,
look at it, and put it back.

The comparison is drawn on the *same* canvas rather than beside it. Segmentation
is compared by flicking between two pictures in one place, not across a gap that
halves the resolution of both; holding the compare key does exactly that.
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

#: What the compare control says, and the key that does it. One letter, on the
#: left of the keyboard, so it falls under the hand not holding the stylus.
COMPARE_KEY: str = "V"
COMPARE_KEY_CODE = Qt.Key_V


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
            "Versions kept when this label was replaced, newest first. The "
            "percentage is how much each one differs from the version before it."
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
            widget.setFocusPolicy(Qt.NoFocus)
        self.set_rows(())

    def set_rows(self, rows) -> None:
        """Show *rows*, or hide the strip where there are none.

        A label nobody has replaced has no history to offer, and a control that
        is always there and usually empty is one more thing to read past.
        """
        self.setVisible(bool(rows))
        blocked = self.chooser.blockSignals(True)
        self.chooser.clear()
        for row in rows:
            self.chooser.addItem(row.describe())
            if not row.available:
                index = self.chooser.count() - 1
                self.chooser.setItemData(
                    index,
                    "This version's pixels are not in this copy. Export again "
                    "with --with-archive to bring them.",
                    Qt.ToolTipRole,
                )
        self.chooser.setCurrentIndex(0 if rows else -1)
        self.chooser.blockSignals(blocked)
        self._chosen(self.chooser.currentIndex())

    def set_available(self, available: bool) -> None:
        """Say whether the chosen version can be looked at and put back."""
        self.compare_button.setEnabled(available)
        self.difference_check.setEnabled(available)
        self.restore_button.setEnabled(available)
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
