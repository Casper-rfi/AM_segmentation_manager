"""Naming a state of the collection, before anything is written.

A savepoint marks a round of model improvement, which somebody decides they have
finished, so this collects what only they can supply -- the round's name and why
-- and the one choice that costs disk, then hands over to the ordinary preview.
The name is required and nothing suggests one: a savepoint is found by its title
months later.
"""
from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from .controls import ToggleSwitch

__all__ = ["SavepointDialog"]


class SavepointDialog(QDialog):
    """Asks what to call a savepoint, and whether to close its gap."""

    def __init__(self, parent, projects: int) -> None:
        """Build the dialog.

        Args:
            parent: The window it belongs to.
            projects: How many projects are in the scan, said out loud because
                a savepoint covers all of them and nothing else on this tab
                works that way.
        """
        super().__init__(parent)
        self.setWindowTitle("Savepoint")
        layout = QVBoxLayout(self)

        intro = QLabel(
            f"Record what every label in all {projects} project(s) under this "
            f"folder holds now, under a name. Revert can put them back to it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("round 3")
        self.title_edit.setToolTip(
            "What this state is called. Used to find it again and to put a "
            "whole round back by name."
        )
        form.addRow("Name", self.title_edit)

        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("as proofread through August")
        self.description_edit.setToolTip("Why it was taken. Optional.")
        form.addRow("Note", self.description_edit)
        layout.addLayout(form)

        self.close_gap = ToggleSwitch("Also keep labels the model made")
        self.close_gap.setToolTip(
            "Keep a copy of the labels a model wrote that nobody has edited.\n"
            "Without this they are left out, because a model can make them\n"
            "again -- but then the next conversion overwrites them and this\n"
            "savepoint can no longer reach what they held."
        )
        layout.addWidget(self.close_gap)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Preview")
        # Cancel is the default, as in every dialog that precedes a write:
        # Return must not be able to start one.
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

        self.title_edit.textChanged.connect(self._retest)
        self._retest()

    def _retest(self) -> None:
        """Offer the preview only once there is a name to give it."""
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self.title_edit.text().strip())
        )

    @property
    def title(self) -> str:
        """Return the name, stripped."""
        return self.title_edit.text().strip()

    @property
    def description(self) -> str:
        """Return the note, stripped."""
        return self.description_edit.text().strip()

    @property
    def closing_gap(self) -> bool:
        """Return whether at-risk versions should be kept now."""
        return self.close_gap.isChecked()
