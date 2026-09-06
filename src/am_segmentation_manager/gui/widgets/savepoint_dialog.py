"""Naming a state of the collection, before anything is written.

A savepoint marks a round of model improvement, which is a thing somebody
decides they have finished rather than something the software can notice. So
this collects the two things only they can supply -- what to call the round and
why -- and the one choice that costs disk, and then hands over to the ordinary
preview.

The name is required and nothing suggests one: a savepoint found by its title
months later is worth the sentence it takes to write, and a generated name
would be one more timestamp in a collection already full of them.
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
            f"Name the state of all {projects} project(s) in this folder, so a "
            f"round of work can be compared against it or put back by name."
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

        self.close_gap = ToggleSwitch("Keep what a conversion would destroy")
        self.close_gap.setToolTip(
            "Copy into the archive the labels holding untouched model output.\n"
            "The archive skips those as regenerable, so without this the next\n"
            "conversion makes them unreachable from this savepoint."
        )
        layout.addWidget(self.close_gap)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Preview")
        # Cancel is the default, as in every dialog that precedes a write:
        # Return must not be able to start one.
        buttons.button(QDialogButtonBox.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

        self.title_edit.textChanged.connect(self._retest)
        self._retest()

    def _retest(self) -> None:
        """Offer the preview only once there is a name to give it."""
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
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
