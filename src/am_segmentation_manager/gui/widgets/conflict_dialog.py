"""Deciding what a merge cannot decide for itself.

A conflict is one thing both sides changed differently since the copy was
taken. Nothing here guesses: timestamps decide nothing, and neither tree is
authoritative, so each is answered by name or the merge does not happen.

**One row per group, one choice on it.** A label's pixels and its workflow
state are resolved under a single key, so taking the copy's label takes its
state with it -- two controls over one key would be a window lying about what
a click did. Taking their pixels while keeping a ``corrected`` claim made here
would assert that somebody proofread pixels nobody here has seen.

Nothing is chosen in advance and nothing is applied until every row is
answered: a default would be the rule about which tree wins that the merge
refuses to have.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .controls import SegmentedControl
from .theme import BADGE_STYLES, MUTED

__all__ = ["ConflictDialog", "KEEP_OURS", "TAKE_THEIRS"]

#: What a decision is called where a merge reads it.
KEEP_OURS = "ours"
TAKE_THEIRS = "theirs"

#: The two sides, as the control offers them. "Here" and "the copy" rather
#: than "ours" and "theirs": one machine's copy is not another person, and the
#: reader is looking at one of the two trees.
CHOICES: tuple[tuple[str, str, str], ...] = (
    (KEEP_OURS, "Keep this one", "Keep what this collection holds now."),
    (TAKE_THEIRS, "Take the copy's", "Replace it with what the copy holds."),
)


class ConflictDialog(QDialog):
    """Asks which version to keep, for everything a merge cannot decide."""

    def __init__(self, parent, rows: Sequence) -> None:
        """Build one row per conflict.

        Args:
            parent: The window it belongs to.
            rows: :class:`~..operations_controller.ConflictRow` values.
        """
        super().__init__(parent)
        self.setWindowTitle("Decide what to keep")
        self.resize(760, 520)
        layout = QVBoxLayout(self)

        intro = QLabel(
            f"{len(rows)} thing(s) were changed both here and in the copy "
            f"since it was taken. Choose which to keep. Nothing is merged "
            f"until every one is answered, and the version not chosen stays "
            f"in the archive either way."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        inner = QWidget()
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 0, 0)
        self._controls: dict[str, SegmentedControl] = {}
        for row in rows:
            column.addWidget(self._build_row(row))
        column.addStretch(1)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(inner)
        layout.addWidget(area, stretch=1)

        self.outstanding = QLabel()
        self.outstanding.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.outstanding)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Merge")
        buttons.button(QDialogButtonBox.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons
        self._retest()

    def _build_row(self, row) -> QWidget:
        """Return one conflict as a block: what it is, then the choice."""
        block = QFrame()
        block.setFrameShape(QFrame.NoFrame)
        stack = QVBoxLayout(block)
        stack.setContentsMargins(0, 6, 0, 10)

        heading = QLabel(row.title)
        heading.setWordWrap(True)
        heading.setTextInteractionFlags(Qt.TextSelectableByMouse)
        stack.addWidget(heading)

        if row.corrected:
            # Said on the row rather than counted in the summary: which of
            # these is proofread work decides how the choice is made, and a
            # count at the top does not say which row to look at.
            warning = QLabel("Proofread work here would be replaced")
            warning.setStyleSheet(BADGE_STYLES["warning"])
            stack.addWidget(warning)

        for line in row.details:
            detail = QLabel(line)
            detail.setWordWrap(True)
            detail.setStyleSheet(f"color: {MUTED};")
            stack.addWidget(detail)

        # Nothing is chosen in advance: a default here would be a rule about
        # which tree wins, and a conflict exists precisely because there is
        # not one.
        control = SegmentedControl(CHOICES)
        control.chosen.connect(self._retest)
        stack.addWidget(control)

        self._controls[row.key] = control
        stack.addWidget(_rule())
        return block

    def _retest(self) -> None:
        """Offer the merge only once every row has been answered."""
        undecided = sum(
            1 for control in self._controls.values() if control.current() is None
        )
        self.outstanding.setText(
            "" if not undecided else f"{undecided} still to decide"
        )
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not undecided)

    @property
    def resolutions(self) -> dict[str, str]:
        """Return what was decided, keyed as a merge reads it."""
        return {
            key: side
            for key, control in self._controls.items()
            if (side := control.current()) is not None
        }


def _rule() -> QFrame:
    """Return a hairline between rows, so two do not read as one."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    return line
