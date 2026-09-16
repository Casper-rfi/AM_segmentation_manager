"""What this window remembers about the person using it, rather than the data.

One dialog (F2, or the gear beside the mark) for settings belonging to this
machine: the name changes are recorded under, kept in the home directory, and the
accent and values the window opens on, declared by ``gui/preferences.py``. **Nothing here is
asked for**: the name field opens empty and clearing it is a supported answer. The
fields below it are built from :data:`~...preferences.PREFERENCES`, so adding a
preference is a row there.
"""
from __future__ import annotations

from typing import TypeAlias

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from ...history import resolve_author
from .. import preferences
from .controls import SegmentedControl, ToggleSwitch, WrappingLabel

__all__ = ["SettingsDialog", "author_summary"]

#: How wide the dialog opens, in logical pixels. Wide enough that the notes
#: wrap to a few lines rather than a column, and that a spin box beside its
#: caption is a control rather than a sliver.
DIALOG_WIDTH = 420

#: How wide a number's box may get. A form stretches its field column, and a
#: two-digit answer in a box the width of the dialog reads as an empty field
#: with something small in the corner of it.
SPIN_WIDTH = 110

#: The control a preference is edited with: a switch, a number, or a run of
#: segments for one of a closed set.
Editor: TypeAlias = "ToggleSwitch | QSpinBox | SegmentedControl"

#: Said under the preference boxes. A default decides what a control opens on,
#: so changing one while that control is on screen changes nothing there -- and
#: a dialog that left the reader to discover this is one they would think was
#: broken.
APPLIED_NOTE = (
    "These decide what the window and each control open at. The window applies "
    "them when it next opens, so anything already on screen keeps what it is set to."
)


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

    def __init__(self, parent, values=None) -> None:
        """Build the dialog, filled from what is set now.

        Args:
            values: The preferences as resolved by
                :func:`~...preferences.read`. None fills every control with
                its packaged fallback, which is what a caller with no store
                would want.
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(DIALOG_WIDTH)
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

        note = WrappingLabel(
            "Every change to a label, a workflow state or a project's metadata "
            "is recorded, from any tab. A name is written into each entry, and "
            "the record lives inside the project -- so depositing a collection "
            "deposits it. Leave this empty and entries carry no name."
        )
        layout.addWidget(note)

        held = dict(values or {entry.key: entry.fallback for entry in preferences.PREFERENCES})
        #: The control for each preference, so :attr:`values` can read them
        #: back without a second table saying which widget went where.
        self._editors: dict[str, Editor] = {}
        for title, entries in preferences.groups():
            layout.addWidget(self._build_group(title, entries, held))

        layout.addWidget(WrappingLabel(APPLIED_NOTE))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

    def _build_group(self, title: str, entries, held) -> QGroupBox:
        """Return one box of preferences, a row per entry."""
        box = QGroupBox(title)
        form = QFormLayout(box)
        for entry in entries:
            editor = self._build_editor(entry, held.get(entry.key, entry.fallback))
            editor.setToolTip(entry.hint)
            self._editors[entry.key] = editor
            if entry.is_flag:
                # A switch carries its own caption, so a second one beside it
                # would say the same thing twice.
                form.addRow(editor)
            else:
                caption = QLabel(entry.label)
                caption.setToolTip(entry.hint)
                form.addRow(caption, editor)
        return box

    @staticmethod
    def _build_editor(entry: preferences.Preference, value: preferences.Value) -> Editor:
        """Return the control for one preference, filled with *value*."""
        if entry.is_choice:
            run = SegmentedControl(
                [(key, caption, entry.hint) for key, caption in entry.choices]
            )
            run.set_current(str(value))
            # A form stretches its field column, which pulls the segments apart
            # into buttons that no longer read as one control.
            run.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            return run
        if entry.is_flag:
            switch = ToggleSwitch(entry.label)
            switch.setChecked(bool(value))
            return switch
        assert entry.low is not None and entry.high is not None, f"{entry.key} has no range"
        box = QSpinBox()
        box.setRange(entry.low, entry.high)
        box.setSuffix(entry.suffix)
        box.setValue(int(value))
        box.setMaximumWidth(SPIN_WIDTH)
        return box

    @property
    def author(self) -> str:
        """Return the name typed, trimmed. Empty means record no name."""
        return self.author_edit.text().strip()

    @property
    def values(self) -> dict[str, preferences.Value]:
        """Return every preference as the dialog now holds it."""
        return {key: _held(editor) for key, editor in self._editors.items()}


def _held(editor: Editor) -> preferences.Value:
    """Return the value one preference's control holds."""
    if isinstance(editor, SegmentedControl):
        chosen = editor.current()
        assert chosen is not None, "every run is built with a segment chosen"
        return chosen
    if isinstance(editor, ToggleSwitch):
        return editor.isChecked()
    return editor.value()
