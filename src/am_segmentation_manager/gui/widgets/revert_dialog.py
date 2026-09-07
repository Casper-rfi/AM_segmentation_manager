"""Naming what a revert puts the selection back to.

Three ways of saying it -- before the last conversion, before the last merge,
or the state a savepoint named -- and one operation behind all of them. They
are one control rather than three buttons because they differ only in which
recorded version each label goes back to; a window that split them by which
argument was passed would be harder to learn than the command it replaces.

The savepoint row is offered only where savepoints exist, and carries what each
one still reaches: a round nothing can be recovered from is one to know about
before choosing it, not after.
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

    def __init__(self, parent, scope: str, savepoints: Sequence = ()) -> None:
        """Build the dialog.

        Args:
            parent: The window it belongs to.
            scope: What the selection reaches, in the words the window uses
                elsewhere, so this says what it would act on rather than
                leaving the reviewer to remember.
            savepoints: The savepoints here, newest first. Empty hides that
                option rather than offering one with nothing behind it.
        """
        super().__init__(parent)
        self.setWindowTitle("Revert labels")
        layout = QVBoxLayout(self)

        heading = QLabel("Put these labels back to:")
        layout.addWidget(heading)

        self.choices = QButtonGroup(self)
        self._keys: list[str] = []
        for key, caption in UNDO_LABELS:
            button = QRadioButton(caption)
            self.choices.addButton(button, len(self._keys))
            self._keys.append(key)
            layout.addWidget(button)

        self.savepoint_box = QComboBox()
        self.savepoint_button = QRadioButton("the state named")
        if savepoints:
            self.choices.addButton(self.savepoint_button, len(self._keys))
            self._keys.append("")
            for point in savepoints:
                self.savepoint_box.addItem(_describe(point), point.id)
            row = QHBoxLayout()
            row.addWidget(self.savepoint_button)
            row.addWidget(self.savepoint_box, stretch=1)
            layout.addLayout(row)
            self.savepoint_button.toggled.connect(self.savepoint_box.setEnabled)
            self.savepoint_box.setEnabled(False)

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
        return self._keys[self.choices.checkedId()]

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
