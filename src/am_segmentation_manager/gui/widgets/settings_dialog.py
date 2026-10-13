"""What this window remembers about the person using it, rather than the data.

One dialog (F2, or the gear beside the mark) for settings belonging to this
machine: the name changes are recorded under, kept in the home directory, and the
accent and values the window opens on, declared by ``gui/preferences.py``. **Nothing here is
asked for**: the name field opens empty and clearing it is a supported answer. The
fields below it are built from :data:`~...preferences.PREFERENCES`, so adding a
preference is a row there. They scroll above buttons that stay put, the dialog
being capped to the screen it opens on. A viewer's asks no name, recording
nothing, and offers only what its one view opens on. **Training defaults that
could not be trained with are not kept**: OK says why and stays.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...history import resolve_author
from .. import preferences
from . import theme
from .common import fit_dialog_to_screen
from .controls import SegmentedControl, ToggleSwitch, WrappingLabel
from .theme import ERROR

__all__ = ["SettingsDialog", "author_summary"]

#: How wide the dialog opens, in logical pixels. Wide enough that the notes
#: wrap to a few lines rather than a column, and that a spin box beside its
#: caption is a control rather than a sliver.
DIALOG_WIDTH = 420

#: Kept clear of the screen's edges when the dialog is capped to fit it.
SCREEN_MARGIN = 64

#: How wide a number's box may get. A form stretches its field column, and a
#: two-digit answer in a box the width of the dialog reads as an empty field
#: with something small in the corner of it.
SPIN_WIDTH = 110

#: The control a preference is edited with: a switch, a whole number, a real
#: one, or a run of segments for one of a closed set.
Editor: TypeAlias = "ToggleSwitch | QSpinBox | RealEdit | SegmentedControl"

#: Said under the preference boxes. A default decides what a control opens on,
#: so changing one while that control is on screen changes nothing there -- and
#: a dialog that left the reader to discover this is one they would think was
#: broken.
APPLIED_NOTE = (
    "The accent changes as soon as it is chosen, and Cancel puts it back. The "
    "rest decide what each control opens at: the window applies them when it "
    "next opens, so anything already on screen keeps what it is set to. The "
    "training defaults are also what My defaults on the Train tab gives."
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


class RealEdit(QLineEdit):
    """A real number typed as it is written, ``2e-06`` as readily as ``0.5``.

    A spin box steps by a fixed amount, which no learning rate does.
    """

    def __init__(self, entry: preferences.Preference, value: float) -> None:
        super().__init__()
        self._entry = entry
        self._held = value
        self.setValue(value)
        self.editingFinished.connect(lambda: self.setValue(self.value()))

    def value(self) -> float:
        """Return the number typed, held to the preference's range, or the one last held."""
        low, high = self._entry.low, self._entry.high
        assert low is not None and high is not None, f"{self._entry.key} has no range"
        try:
            typed = float(self.text())
        except ValueError:
            return self._held
        if typed != typed:
            return self._held
        return max(float(low), min(float(high), typed))

    def setValue(self, value: float) -> None:  # noqa: N802 - a spin box's spelling
        """Hold and show *value*."""
        self._held = float(value)
        self.setText(f"{self._held:g}")


class SettingsDialog(QDialog):
    """The settings belonging to this machine rather than to a collection."""

    def __init__(
        self,
        parent,
        values=None,
        *,
        preview_accent: Callable[[str], None] | None = None,
        viewer: bool = False,
    ) -> None:
        """Build the dialog, filled from what is set now.

        Args:
            values: The preferences as resolved by
                :func:`~...preferences.read`. None fills every control with
                its packaged fallback, which is what a caller with no store
                would want.
            preview_accent: Called with an accent's key when one is chosen, and
                with the one the dialog opened on when it is cancelled after
                that, so the window shows a choice before it is kept.
            viewer: Build a viewer's: no name, and only the preferences
                :attr:`~...preferences.Preference.viewer` marks.
        """
        super().__init__(parent)
        self._preview_accent = preview_accent
        self.setWindowTitle("Settings")
        self.setMinimumWidth(DIALOG_WIDTH)
        outer = QVBoxLayout(self)
        #: Everything but the buttons, which scrolls where the screen is short.
        self._body = QWidget()
        layout = QVBoxLayout(self._body)
        layout.setContentsMargins(0, 0, 0, 0)

        #: The name field, or None in a viewer, which records nothing.
        self.author_edit: QLineEdit | None = None
        if not viewer:
            self.author_edit = self._build_author(layout)

        held = dict(values or {entry.key: entry.fallback for entry in preferences.PREFERENCES})
        #: The accent on screen when the dialog opened, which Cancel puts back.
        self._opened_accent = str(held.get(preferences.ACCENT, preferences.DEFAULT_ACCENT))
        #: The control for each preference, so :attr:`values` can read them
        #: back without a second table saying which widget went where.
        self._editors: dict[str, Editor] = {}
        #: The button putting a training box back to the published settings, by box.
        self.restore_buttons: dict[str, QPushButton] = {}
        for title, entries in preferences.groups(viewer=viewer):
            layout.addWidget(self._build_group(title, entries, held))

        layout.addWidget(WrappingLabel(APPLIED_NOTE))
        #: Why OK did not close the dialog, shown only then.
        self.problem_label = WrappingLabel("")
        self.problem_label.setStyleSheet(f"color: {ERROR};")
        self.problem_label.setHidden(True)
        layout.addWidget(self.problem_label)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._body)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.buttons = buttons
        self.fit_to_screen()

    def fit_to_screen(self) -> None:
        """Size to every setting, or to the screen where that is smaller.

        Measured from the body rather than the dialog, a scroll area reporting
        a small default hint whatever it holds; and its height at the width it
        will have, the notes being word-wrapped.
        """
        outer = self.layout()
        assert outer is not None, "the dialog is laid out when it is built"
        margins = outer.contentsMargins()
        wide = max(DIALOG_WIDTH, self._body.sizeHint().width() + margins.left() + margins.right())
        body = self._body.layout()
        assert body is not None, "the body is laid out when it is built"
        inner = wide - margins.left() - margins.right()
        tall = (
            (body.heightForWidth(inner) if body.hasHeightForWidth() else body.sizeHint().height())
            + margins.top() + margins.bottom() + outer.spacing()
            + self.buttons.sizeHint().height()
        )
        fit_dialog_to_screen(self, QSize(wide, tall), self._scroll, margin=SCREEN_MARGIN)

    @staticmethod
    def _build_author(layout: QVBoxLayout) -> QLineEdit:
        """Add the name changes are recorded under, and what it is for."""
        current, _source = resolve_author()
        form = QFormLayout()
        edit = QLineEdit(current or "")
        edit.setPlaceholderText("nobody")
        edit.setToolTip(
            "Name under which changes are recorded on this machine.\nLeave empty to record no name."
        )
        form.addRow("Record changes as", edit)
        layout.addLayout(form)
        layout.addWidget(WrappingLabel(
            "Every change to a label, a workflow state or a project's metadata "
            "is recorded, from any tab. A name is written into each entry, and "
            "the record lives inside the project -- so depositing a collection "
            "deposits it. Leave this empty and entries carry no name."
        ))
        return edit

    def _build_group(self, title: str, entries, held) -> QGroupBox:
        """Return one box of preferences, a row per entry."""
        box = QGroupBox(title)
        form = QFormLayout(box)
        for entry in entries:
            editor = self._build_editor(entry, held.get(entry.key, entry.fallback))
            editor.setToolTip(entry.hint)
            if entry.key == preferences.ACCENT and isinstance(editor, SegmentedControl):
                self._offer_accents(editor)
            self._editors[entry.key] = editor
            if entry.is_flag:
                # A switch carries its own caption, so a second one beside it
                # would say the same thing twice.
                form.addRow(editor)
            else:
                caption = QLabel(entry.label)
                caption.setToolTip(entry.hint)
                form.addRow(caption, editor)
        if title in preferences.TRAIN_GROUPS.values():
            restore = QPushButton("Restore recommended")
            restore.setToolTip("Reset these to the values the published models were trained with.")
            restore.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            restore.clicked.connect(lambda _checked=False, group=title: self.restore(group))
            form.addRow(restore)
            self.restore_buttons[title] = restore
        return box

    def restore(self, group: str) -> None:
        """Put every preference of *group* back to what the package ships, without keeping it yet."""
        for key, value in preferences.recommended(group).items():
            editor = self._editors[key]
            assert isinstance(editor, QSpinBox | RealEdit), f"{key} is not a number"
            editor.setValue(value)  # type: ignore[arg-type]

    def accept(self) -> None:
        """Close and keep what is set, unless the training defaults could not be trained with."""
        problems = preferences.training_problems(self.values)
        if problems:
            self.problem_label.setText("\n".join(problems))
            self.problem_label.setHidden(False)
            self._scroll.ensureWidgetVisible(self.problem_label)
            return
        super().accept()

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
        if entry.is_real:
            edit = RealEdit(entry, float(value))
            edit.setMaximumWidth(SPIN_WIDTH)
            return edit
        box = QSpinBox()
        box.setRange(int(entry.low), int(entry.high))
        box.setSuffix(entry.suffix)
        box.setValue(int(value))
        box.setMaximumWidth(SPIN_WIDTH)
        return box

    def _offer_accents(self, run: SegmentedControl) -> None:
        """Show each accent beside its name, and preview one as it is chosen.

        A dot rather than the chosen fill alone, which is drawn in the accent
        already on screen and so shows the colour of neither answer until one
        is previewed.
        """
        for key, button in run.buttons.items():
            button.setIcon(theme.accent_swatch(theme.ACCENTS[key]))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        if self._preview_accent is not None:
            run.chosen.connect(self._preview_accent)

    def reject(self) -> None:
        """Close without keeping anything, the accent on screen included."""
        editor = self._editors.get(preferences.ACCENT)
        chosen = _held(editor) if editor is not None else self._opened_accent
        if self._preview_accent is not None and chosen != self._opened_accent:
            self._preview_accent(self._opened_accent)
        super().reject()

    @property
    def author(self) -> str | None:
        """Return the name typed, trimmed: empty to record no name, None where not asked."""
        if self.author_edit is None:
            return None
        return self.author_edit.text().strip()

    @property
    def values(self) -> dict[str, preferences.Value]:
        """Return every preference the dialog offers, as it now holds it."""
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
