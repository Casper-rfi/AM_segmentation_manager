"""Deciding what a merge cannot decide for itself.

A conflict is one thing both sides changed differently since the copy was
taken. The metadata fields come first, then the labels, and *Merge* is
offered only once every conflict on both pages is answered; nothing is chosen
in advance. Decisions are held by a
:class:`~..conflict_resolution.ConflictResolution`; the pages only show them.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...config import SegmentationConfig
from ..conflict_resolution import ConflictResolution
from ..preferences import DEFAULT_ADVANCE
from ..report import ConflictRow
from .conflict_field_page import FieldConflictPage
from .conflict_label_page import LabelConflictPage
from .theme import MUTED

__all__ = ["ConflictDialog"]


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
