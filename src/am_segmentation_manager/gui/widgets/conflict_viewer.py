"""Looking at the two versions of a label a merge cannot decide between.

Both are drawn over the same acquisition in the same place, and holding **B**
swaps between them. Side by side halves the resolution of each, and a crack
is a fraction of a percent of a frame: the difference that decides this is not
one a smaller picture shows. The mask answers *where* they differ, holding B
answers *what* each holds there. The acquisition is this collection's -- a
merge never moves an image -- and nothing here writes or decides.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...config import SegmentationConfig
from ...display import DEFAULT_ALPHA, compose_overlay, to_display_8bit
from ...io.image import read_array, to_2d
from .image_view import ImageView
from .theme import BADGE_STYLES, MUTED
from .versions_bar import COMPARE_KEY, DIFFERENCE_CLASS

logger = logging.getLogger(__name__)

__all__ = ["ConflictView"]


class ConflictView(QWidget):
    """Shows the two versions of one conflicting label over its acquisition."""

    def __init__(self, config: SegmentationConfig, parent: QWidget | None = None) -> None:
        """Build an empty view; :meth:`show_row` puts a conflict on it.

        Args:
            config: Supplies the colour of each class, shared with the rest of
                the window so a class is one colour everywhere.
        """
        super().__init__(parent)
        self._config = config
        self._ours: np.ndarray | None = None
        self._theirs: np.ndarray | None = None
        self._image: np.ndarray | None = None
        self._showing_theirs = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView()
        layout.addWidget(self.view, stretch=1)

        self.badge = QLabel("")
        self.badge.setStyleSheet(BADGE_STYLES["warning"])
        layout.addWidget(self.badge)

        self.problems = QLabel("")
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet(BADGE_STYLES["warning"])
        self.problems.setHidden(True)
        layout.addWidget(self.problems)

        controls = QHBoxLayout()
        self.differences = QCheckBox("Mark where they differ")
        self.differences.toggled.connect(lambda _on: self._draw())
        controls.addWidget(self.differences)
        controls.addStretch(1)
        held = QLabel(f"Hold {COMPARE_KEY} to see the copy's")
        held.setStyleSheet(f"color: {MUTED};")
        controls.addWidget(held)
        layout.addLayout(controls)
        self._offer_differences()

    # -- what is on screen -------------------------------------------------

    def show_row(self, row) -> None:
        """Read *row*'s two labels and its acquisition, and draw this side's.

        Args:
            row: A :class:`~..report.ConflictRow`, or None to show nothing.
        """
        self._showing_theirs = False
        if row is None:
            self._ours = self._theirs = self._image = None
            problems: tuple[str, ...] = ()
        else:
            self._ours, self._theirs, self._image, problems = _read(row)
        self.problems.setText("\n".join(problems))
        self.problems.setHidden(not problems)
        self._offer_differences()
        self._draw()

    def set_comparing(self, held: bool) -> None:
        """Show the copy's label while *held*, and this collection's otherwise."""
        if held == self._showing_theirs:
            return
        self._showing_theirs = held
        self._draw()

    @property
    def showing_theirs(self) -> bool:
        """Return whether the copy's label is the one on screen."""
        return self._showing_theirs

    def _comparable(self) -> bool:
        """Return whether the two labels can be compared pixel for pixel."""
        return (
            self._ours is not None
            and self._theirs is not None
            and to_2d(self._ours).shape == to_2d(self._theirs).shape
        )

    def _offer_differences(self) -> None:
        """Offer the difference mask only where there is a pairing to draw."""
        comparable = self._comparable()
        self.differences.setEnabled(comparable)
        if comparable:
            hint = "Draw every pixel the two disagree on, in the colour no class uses."
        elif self._ours is not None and self._theirs is not None:
            hint = (
                "The two are different sizes, so there is no pixel to pixel "
                "comparison to draw."
            )
        else:
            hint = "There are not two labels here to compare."
        self.differences.setToolTip(hint)

    def _draw(self) -> None:
        """Put the version now being shown on the canvas."""
        label = self._theirs if self._showing_theirs else self._ours
        image = self._image
        self.badge.setText(
            "Showing the copy's label" if self._showing_theirs else ""
        )
        if label is None and image is None:
            self.view.clear()
            return
        if label is None:
            self.view.set_array(np.ascontiguousarray(np.stack((image,) * 3, axis=-1)))
            return
        label = to_2d(label)
        if self.differences.isChecked() and self._comparable():
            other = to_2d(self._ours if self._showing_theirs else self._theirs)
            label = np.where(label != other, DIFFERENCE_CLASS, label)
        if image is None or image.shape[:2] != label.shape[:2]:
            # No acquisition, or one the label does not match: the label is
            # still worth seeing, and drawn on black it is obvious that what
            # is under it is missing rather than dark.
            image = np.zeros(label.shape[:2], dtype=np.uint8)
        self.view.set_array(
            compose_overlay(image, label, self._config, alpha=DEFAULT_ALPHA)
        )


def _read(row) -> tuple:
    """Return both labels, the acquisition and anything that could not be read.

    A file that will not open is reported rather than raised: the other side
    is still worth looking at, and which of the two is missing is itself
    something to decide on. A state conflict has no labels to miss.
    """
    problems: list[str] = []

    def opened(path: Path | None, what: str, *, expected: bool = True):
        if path is None or not Path(path).is_file():
            if expected:
                where = f" at {path}" if path is not None else ""
                problems.append(f"There is no {what} to show{where}.")
            return None
        try:
            return read_array(Path(path))
        except Exception as error:  # noqa: BLE001 - reported, not raised
            logger.warning("Could not read %s: %s", path, error)
            problems.append(f"The {what} could not be read: {error}")
            return None

    ours = opened(row.ours, "label here", expected=row.is_label)
    theirs = opened(row.theirs, "label in the copy", expected=row.is_label)
    image_path = row.image or (row.record.image_path if row.record is not None else None)
    image = opened(image_path, "acquisition")
    if image is not None:
        image = to_display_8bit(to_2d(image))
    return ours, theirs, image, tuple(problems)
