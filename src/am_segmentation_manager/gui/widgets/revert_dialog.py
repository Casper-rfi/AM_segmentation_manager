"""Naming what a revert puts the selection back to.

Before the last conversion, before the last merge, or the state a savepoint named:
one control, because the three differ only in which recorded version each label
goes back to. The savepoint row is offered only where savepoints exist and says
what each still reaches, a round nothing can be recovered from being one to know
about before choosing it.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

__all__ = ["RevertDialog"]

#: What each option puts labels back to, in the order they are reached for: a
#: conversion is the one this exists for, a merge is the same shape of change,
#: and a savepoint is the whole round rather than one operation.
UNDO_LABELS: tuple[tuple[str, str], ...] = (
    ("conversion", "before the last conversion"),
    ("merge", "before the last merge"),
)


class RevertDialog(QDialog):
    """Asks what to put the selection back to."""

    def __init__(
        self,
        parent,
        scope: str,
        savepoints: Sequence = (),
        *,
        operations: bool = True,
    ) -> None:
        """Build the dialog.

        Args:
            parent: The window it belongs to.
            scope: What this would act on, in the words the window uses
                elsewhere, so it says so rather than leaving the reviewer to
                remember.
            savepoints: The savepoints here, newest first. Empty hides that
                option rather than offering one with nothing behind it.
            operations: Offer undoing the last conversion or merge as well.
                False leaves only the savepoints, which is the whole of what
                restoring one over a collection means -- there is no "the last
                conversion" across a hundred projects.
        """
        super().__init__(parent)
        self.setWindowTitle("Revert labels" if operations else "Restore savepoint")
        layout = QVBoxLayout(self)

        heading = QLabel(
            "Put these labels back to:" if operations
            else "Put every label back to the state named:"
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self.choices = QButtonGroup(self)
        self._keys: list[str] = []
        if operations:
            for key, caption in UNDO_LABELS:
                button = QRadioButton(caption)
                self.choices.addButton(button, len(self._keys))
                self._keys.append(key)
                layout.addWidget(button)

        self.savepoint_box = QComboBox()
        self.savepoint_button = QRadioButton("the state named")
        for point in savepoints:
            self.savepoint_box.addItem(_describe(point), point.id)
        if savepoints and operations:
            self.choices.addButton(self.savepoint_button, len(self._keys))
            self._keys.append("")
            row = QHBoxLayout()
            row.addWidget(self.savepoint_button)
            row.addWidget(self.savepoint_box, stretch=1)
            layout.addLayout(row)
            self.savepoint_button.toggled.connect(self.savepoint_box.setEnabled)
            self.savepoint_box.setEnabled(False)
        elif savepoints:
            self._keys.append("")
            layout.addWidget(self.savepoint_box)

        if self.choices.buttons():
            self.choices.button(0).setChecked(True)

        reach = QLabel(scope)
        reach.setWordWrap(True)
        layout.addWidget(reach)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Preview")
        # Cancel is the default, as in every dialog that precedes a write.
        buttons.button(QDialogButtonBox.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

    @property
    def undoing(self) -> str:
        """Return the operation to undo, or empty where a savepoint is chosen."""
        chosen = self.choices.checkedId()
        return self._keys[chosen] if chosen >= 0 else ""

    @property
    def savepoint(self) -> str:
        """Return the chosen savepoint's id, or empty where none is chosen."""
        if self.undoing:
            return ""
        return self.savepoint_box.currentData() or ""


def _describe(point) -> str:
    """Return one savepoint as a row: what it is called and what it reaches."""
    name = point.title or point.id[:8]
    return f"{name} — {point.reachable}/{point.targets} labels reachable"
