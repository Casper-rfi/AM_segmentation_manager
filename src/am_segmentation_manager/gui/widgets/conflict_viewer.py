"""Looking at the two versions of a label a merge cannot decide between.

Both are drawn over the same acquisition in the same place, and **B** swaps
between them. Side by side halves the resolution of each, and a crack is a
fraction of a percent of a frame: the difference that decides this is not one
a smaller picture shows. The mask answers *where* they differ, holding B
answers *what* each holds there.

The acquisition comes from this collection rather than from the copy. A merge
never moves an image, so the two are the same file -- and this side is the one
less likely to have been touched since the copy was taken.

Nothing here writes. The choice made on it is carried back to the row that
opened it, so a decision made while looking is not one to remember afterwards.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from ...config import SegmentationConfig
from ...display import DEFAULT_ALPHA, compose_overlay, to_display_8bit
from ...io.image import read_array, to_2d
from .image_view import ImageView
from .theme import BADGE_STYLES, MUTED

logger = logging.getLogger(__name__)

__all__ = ["ConflictViewer"]

#: The class a difference is drawn with. Outside anything a configuration
#: defines, so it renders in the colour reserved for an unconfigured class and
#: what is on screen is plainly not a segmentation. The same value the version
#: strip uses, one meaning for one mark.
DIFFERENCE_CLASS: int = 255

#: What the key does, and which key. **Not V**, which shows and hides the
#: label elsewhere; B for "both", under the hand not holding the stylus.
COMPARE_KEY: str = "B"


class ConflictViewer(QDialog):
    """Shows both versions of one label, and takes the decision about them."""

    def __init__(self, parent, row, config: SegmentationConfig) -> None:
        """Build the viewer for *row*, reading its two labels and the image.

        Args:
            parent: The window it belongs to.
            row: A :class:`~..report.ConflictRow` with labels.
            config: Supplies the colour of each class, shared with the rest of
                the window so a class is one colour everywhere.
        """
        super().__init__(parent)
        self.setWindowTitle(row.title.rsplit("/", 1)[-1])
        self.resize(900, 760)
        self._config = config
        self._showing_theirs = False
        self._decision: str | None = None

        layout = QVBoxLayout(self)
        heading = QLabel(row.title)
        heading.setWordWrap(True)
        heading.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(heading)

        self._ours, self._theirs, self._image, problems = _read(row)

        self.view = ImageView()
        layout.addWidget(self.view, stretch=1)

        self.badge = QLabel("")
        self.badge.setStyleSheet(BADGE_STYLES["warning"])
        layout.addWidget(self.badge)

        for problem in problems:
            note = QLabel(problem)
            note.setWordWrap(True)
            note.setStyleSheet(BADGE_STYLES["warning"])
            layout.addWidget(note)

        controls = QHBoxLayout()
        self.differences = QCheckBox("Mark where they differ")
        self.differences.setEnabled(self._comparable())
        self.differences.setToolTip(
            "Draw every pixel the two disagree on, in the colour no class uses."
            if self._comparable() else
            "The two are different sizes, so there is no pixel to pixel "
            "comparison to draw."
        )
        self.differences.toggled.connect(lambda _on: self._draw())
        controls.addWidget(self.differences)
        controls.addStretch(1)
        held = QLabel(f"Hold {COMPARE_KEY} to see the copy's")
        held.setStyleSheet(f"color: {MUTED};")
        controls.addWidget(held)
        layout.addLayout(controls)

        buttons = QDialogButtonBox()
        self.keep = buttons.addButton("Keep this one", QDialogButtonBox.AcceptRole)
        self.take = buttons.addButton("Take the copy's", QDialogButtonBox.AcceptRole)
        cancel = buttons.addButton(QDialogButtonBox.Cancel)
        # Neither side is the default, and Qt makes the first accepting button
        # one unless told otherwise. A default here is a rule about which tree
        # wins, drawn in the accent -- and a conflict exists precisely because
        # there is no such rule. Cancel takes it, as in every dialog before a
        # write: Return must not be able to decide one.
        for button in (self.keep, self.take):
            button.setDefault(False)
            button.setAutoDefault(False)
        cancel.setDefault(True)
        self.keep.clicked.connect(lambda: self._decide("ours"))
        self.take.clicked.connect(lambda: self._decide("theirs"))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

        self._draw()
        self.view.fit()

    # -- what is on screen -------------------------------------------------

    def _comparable(self) -> bool:
        """Return whether the two labels can be compared pixel for pixel."""
        return (
            self._ours is not None
            and self._theirs is not None
            and to_2d(self._ours).shape == to_2d(self._theirs).shape
        )

    def _draw(self) -> None:
        """Put the version now being shown on the canvas."""
        label = self._theirs if self._showing_theirs else self._ours
        if label is None:
            self.view.clear()
            return
        label = to_2d(label)
        if self.differences.isChecked() and self._comparable():
            other = to_2d(self._theirs if not self._showing_theirs else self._ours)
            label = np.where(label != other, DIFFERENCE_CLASS, label)
        image = self._image
        if image is None or image.shape[:2] != label.shape[:2]:
            # No acquisition, or one the label does not match: the label is
            # still worth seeing, and drawn on black it is obvious that what
            # is under it is missing rather than dark.
            image = np.zeros(label.shape[:2], dtype=np.uint8)
        self.view.set_array(
            compose_overlay(image, label, self._config, alpha=DEFAULT_ALPHA)
        )
        self.badge.setText(
            "Showing the copy's label" if self._showing_theirs else ""
        )

    # -- the key -----------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Swap to the copy's label while the compare key is down."""
        if event.key() == Qt.Key_B and not event.isAutoRepeat():
            self._showing_theirs = True
            self._draw()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Put this collection's label back when the key comes up."""
        if event.key() == Qt.Key_B and not event.isAutoRepeat():
            self._showing_theirs = False
            self._draw()
            return
        super().keyReleaseEvent(event)

    # -- the answer --------------------------------------------------------

    def _decide(self, side: str) -> None:
        self._decision = side
        self.accept()

    @property
    def decision(self) -> str | None:
        """Return the side chosen, or None where the viewer was closed."""
        return self._decision


def _read(row) -> tuple:
    """Return both labels, the acquisition and anything that could not be read.

    A file that will not open is reported rather than raised: the other side
    is still worth looking at, and which of the two is missing is itself
    something to decide on.
    """
    problems: list[str] = []

    def opened(path: Path | None, what: str):
        if path is None or not Path(path).is_file():
            problems.append(f"There is no {what} to show at {path}.")
            return None
        try:
            return read_array(Path(path))
        except Exception as error:  # noqa: BLE001 - reported, not raised
            logger.warning("Could not read %s: %s", path, error)
            problems.append(f"The {what} could not be read: {error}")
            return None

    ours = opened(row.ours, "label here")
    theirs = opened(row.theirs, "label in the copy")
    image = opened(row.image, "acquisition")
    if image is not None:
        image = to_display_8bit(to_2d(image))
    return ours, theirs, image, tuple(problems)
