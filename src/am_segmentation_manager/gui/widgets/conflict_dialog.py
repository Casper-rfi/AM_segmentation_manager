"""Deciding what a merge cannot decide for itself.

A conflict is one thing both sides changed differently since the copy was
taken. Nothing here guesses and nothing is chosen in advance: the metadata
fields come first, then the labels, and *Merge* is offered only once every
conflict on both pages is answered -- a default would be the rule about which
tree wins that the merge refuses to have. Decisions are held by a
:class:`~..conflict_resolution.ConflictResolution`; the pages only show them.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig
from ...operations.merge import OURS, THEIRS
from ..conflict_resolution import ConflictResolution
from ..preferences import DEFAULT_ADVANCE
from ..report import ConflictRow
from .conflict_label_page import LabelConflictPage
from .theme import BADGE_STYLES, MUTED

__all__ = ["ConflictDialog", "FieldConflictPage"]

#: The two sides, as a row's buttons offer them. "Here" and "the copy" rather
#: than "ours" and "theirs": one machine's copy is not another person, and the
#: reader is looking at one of the two trees.
CHOICES: tuple[tuple[str, str, str], ...] = (
    (OURS, "Keep this one", "Keep what this collection holds now."),
    (THEIRS, "Take the copy's", "Replace it with what the copy holds."),
)


class ConflictDialog(QDialog):
    """Asks which version to keep, for everything a merge cannot decide."""

    def __init__(
        self,
        parent,
        rows: Sequence[ConflictRow],
        config: SegmentationConfig,
        *,
        advance: bool = DEFAULT_ADVANCE,
    ) -> None:
        """Build a page for the metadata conflicts and one for the labels.

        A page with nothing on it is left out.

        Args:
            parent: The window it belongs to.
            rows: The conflicts, as a merge report carries them.
            config: Supplies the class colours the label page draws with.
            advance: Whether the label page opens moving on after a decision.
        """
        super().__init__(parent)
        self.setWindowTitle("Decide what to keep")
        self.resize(1240, 820)
        self.resolution = ConflictResolution(rows)
        layout = QVBoxLayout(self)

        self.pages = QStackedWidget()
        self.field_page: FieldConflictPage | None = None
        self.label_page: LabelConflictPage | None = None
        if self.resolution.fields:
            self.field_page = FieldConflictPage(self.resolution)
            self.field_page.changed.connect(self._update)
            self.pages.addWidget(self.field_page)
        if self.resolution.labels:
            self.label_page = LabelConflictPage(
                self.resolution, config, advance=advance
            )
            self.label_page.changed.connect(self._update)
            self.pages.addWidget(self.label_page)
        layout.addWidget(self.pages, stretch=1)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.status)

        buttons = QDialogButtonBox()
        self.back = buttons.addButton("Back", QDialogButtonBox.ActionRole)
        self.next = buttons.addButton("Next", QDialogButtonBox.ActionRole)
        self.merge = buttons.addButton("Merge", QDialogButtonBox.AcceptRole)
        cancel = buttons.addButton(QDialogButtonBox.Cancel)
        # Qt makes the first accepting button the default and draws it in the
        # accent. Cancel takes it, as before every other write: Return must
        # not be able to merge -- and it asks before losing any decision.
        for button in (self.back, self.next, self.merge):
            button.setAutoDefault(False)
            button.setDefault(False)
        cancel.setDefault(True)
        self.back.clicked.connect(lambda: self._turn(-1))
        self.next.clicked.connect(lambda: self._turn(1))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons
        self._update()

    @property
    def resolutions(self) -> dict[str, str]:
        """Return what was decided, keyed as a merge reads it."""
        return self.resolution.resolutions

    def _nodes_of(self, page: QWidget | None):
        """Return the conflicts *page* decides."""
        if page is self.field_page:
            return self.resolution.fields
        return self.resolution.labels

    def _turn(self, step: int) -> None:
        """Move *step* pages on or back."""
        index = self.pages.currentIndex() + step
        if 0 <= index < self.pages.count():
            self.pages.setCurrentIndex(index)
        self._update()

    def _update(self) -> None:
        """Offer the next page once this one is decided, and Merge once all are."""
        index = self.pages.currentIndex()
        last = index >= self.pages.count() - 1
        self.back.setVisible(index > 0)
        self.next.setVisible(not last)
        self.merge.setVisible(last)
        page_left = self.resolution.outstanding(self._nodes_of(self.pages.currentWidget()))
        self.next.setEnabled(page_left == 0)
        left = self.resolution.outstanding()
        self.merge.setEnabled(left == 0)
        names = [
            name for name, page in (
                ("metadata fields", self.field_page), ("labels", self.label_page)
            )
            if page is not None
        ]
        step = f"Step {index + 1} of {len(names)}: {names[index]}. " if names else ""
        self.status.setText(
            step + (f"{left} still to decide." if left else "Every conflict is decided.")
        )

    def reject(self) -> None:
        """Close without merging, asking first where decisions would be lost."""
        decided = len(self.resolution.resolutions)
        if decided and QMessageBox.question(
            self,
            "Discard these decisions?",
            f"{decided} conflict(s) have been decided. Close without merging, "
            f"and lose those decisions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        super().reject()


class FieldConflictPage(QWidget):
    """Decides metadata conflicts, one field to a row."""

    #: Emitted whenever a decision changes.
    changed = Signal()

    def __init__(self, resolution: ConflictResolution, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.resolution = resolution
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        keys = [key for node in resolution.fields for key in node.keys]

        intro = QLabel(
            f"{len(keys)} project metadata field(s) were changed differently "
            f"here and in the copy since it was taken. Choose which to keep; "
            f"the history keeps both values either way."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        inner = QWidget()
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 0, 0)
        self._choices: dict[str, dict[str, QPushButton]] = {}
        for key in keys:
            column.addWidget(self._build_row(key))
        column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(inner)
        layout.addWidget(area, stretch=1)

        row = QHBoxLayout()
        for caption, side, hint in (
            ("Keep all here", OURS, "Keep this collection's value of every field"),
            ("Take all the copy's", THEIRS, "Take the copy's value of every field"),
            ("Clear", None, "Undecide every field"),
        ):
            button = QPushButton(caption)
            button.setAutoDefault(False)
            button.setToolTip(hint)
            button.clicked.connect(
                lambda _checked=False, s=side: self.decide_everything(s)
            )
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self._refresh()

    def _build_row(self, key: str) -> QWidget:
        """Return one field as a block: what it is, what each side holds, the choice."""
        node = self.resolution.conflict(key)
        block = QFrame()
        stack = QVBoxLayout(block)
        stack.setContentsMargins(0, 6, 0, 10)
        heading = QLabel(f"{node.row.project} — {node.caption}")
        heading.setWordWrap(True)
        heading.setTextInteractionFlags(Qt.TextSelectableByMouse)
        stack.addWidget(heading)
        for line in node.row.details:
            detail = QLabel(line)
            detail.setWordWrap(True)
            detail.setStyleSheet(f"color: {MUTED};")
            stack.addWidget(detail)
        if node.row.corrected:
            warning = QLabel("Proofread work here would be replaced")
            warning.setStyleSheet(BADGE_STYLES["warning"])
            stack.addWidget(warning)
        choose = QHBoxLayout()
        buttons: dict[str, QPushButton] = {}
        for side, caption, hint in CHOICES:
            button = QPushButton(caption)
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setToolTip(hint)
            button.clicked.connect(
                lambda checked, k=key, s=side: self._decide(k, s if checked else None)
            )
            choose.addWidget(button)
            buttons[side] = button
        choose.addStretch(1)
        stack.addLayout(choose)
        self._choices[key] = buttons
        return block

    def decide_everything(self, side: str | None) -> None:
        """Decide every field one way, or undecide them all with None."""
        self.resolution.decide_all(self.resolution.fields, side)
        self._refresh()
        self.changed.emit()

    def _decide(self, key: str, side: str | None) -> None:
        self.resolution.decide(self.resolution.conflict(key), side)
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        """Light each row's button for the side it holds."""
        for key, buttons in self._choices.items():
            chosen = self.resolution.side(key)
            for side, button in buttons.items():
                button.setChecked(side == chosen)
