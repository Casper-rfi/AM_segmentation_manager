"""What this window remembers about the person using it, rather than the data.

One dialog, opened with F2, for the settings that belong to this machine and
not to any collection. It holds one thing so far: the name changes are recorded
under, which lives in the user's home directory because who is at the keyboard
is a property of the machine.

**Nothing here is asked for.** A name is stated or it is not, and a collection
where nobody stated one deposits no names at all -- so the field opens empty
and clearing it is a supported answer rather than an omission.
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

from ...history import resolve_author

__all__ = ["SettingsDialog", "author_summary"]


def author_summary(name: str | None, *, asked: bool = False) -> str:
    """Return the line saying who changes are being recorded under.

    Shown beside the mark, where it is visible on every tab: a name reaches
    every entry written from anywhere in the window, so no one tab is the
    place to say it and no one action is the place to warn.

    Args:
        name: The name in effect, or None where nobody stated one. Required,
            None being an answer here rather than an absent argument.
        asked: Whether to spell out what having no name means, which is worth
            a clause in a dialog and too long for the tab row.
    """
    if not asked:
        return f"Recording as {name}" if name else "Recording without a name"
    if name:
        return f"Changes are recorded as {name}."
    return (
        "Changes are recorded without a name. Entries say what happened and "
        "when, and carry nobody."
    )


class SettingsDialog(QDialog):
    """The settings belonging to this machine rather than to a collection."""

    def __init__(self, parent) -> None:
        """Build the dialog, filled from what is set now."""
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)

        current, _source = resolve_author()

        form = QFormLayout()
        self.author_edit = QLineEdit(current or "")
        self.author_edit.setPlaceholderText("nobody")
        self.author_edit.setToolTip(
            "Who changes are recorded under, on this machine.\n"
            "Leave empty to record no name at all."
        )
        form.addRow("Record changes as", self.author_edit)
        layout.addLayout(form)

        note = QLabel(
            "Every change to a label, a workflow state or a project's metadata "
            "is recorded, from any tab. A name is written into each entry, and "
            "the record lives inside the project -- so depositing a collection "
            "deposits it. Leave this empty and entries carry no name."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

    @property
    def author(self) -> str:
        """Return the name typed, trimmed. Empty means record no name."""
        return self.author_edit.text().strip()
