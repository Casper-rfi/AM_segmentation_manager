"""Right panel: which images to change, to which state, and how far that reaches.

*Which rows* a change covers is chosen on the left; this panel answers the rest
-- which modality of them, and which state -- and reports how far it reaches
before it is made. The modality sits here rather than beside the ticks because
it is read together with the reach line it changes, and because both are next
to the buttons that act on them. Everything that *writes* stays disabled until
editing is switched on, so reviewing data cannot change it by a stray click;
choosing what a change would cover writes nothing and stays usable.

The states are mutually exclusive because a group has exactly one entry in its
stage's state file. A frame whose acquisitions disagree therefore shows no
option chosen.
"""
from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ...operations.selection import RecordState
from ..controller import MIXED
from .frame_strip import STATE_COLOURS
from .theme import SHARED_NOTE_STYLE

#: Workflow order, which is also the order of the rows in the timeline: data is
#: excluded, or queued for proofreading, or finished.
STATES: tuple[RecordState, ...] = ("deselected", "selected", "corrected")

#: Key that applies each state, shown in its tooltip.
STATE_KEYS = {"deselected": "D", "selected": "S", "corrected": "C"}

STATE_HELP = {
    "deselected": "Exclude from proofreading. No label is created or measured "
                  "for these, and a training export leaves them out.",
    "selected": "Queue for proofreading. This is what an image starts as.",
    "corrected": "Mark proofreading finished, which a training export trusts. "
                 "Leaving this state again asks for confirmation.",
}

SWATCH_SIZE = 12


class StatePanel(QWidget):
    """The editing guard, the reach of the next change, and the state controls."""

    #: Emitted when the editing guard is switched.
    editing_toggled = Signal(bool)
    #: Emitted with the state the user asked for.
    state_requested = Signal(str)
    #: Emitted with the modality a state change applies to.
    target_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # -- the guard -----------------------------------------------------
        from qtpy.QtWidgets import QCheckBox

        self.enable_check = QCheckBox("Enable editing")
        self.enable_check.setToolTip(
            "Unlock the controls below. Off by default so that reviewing data "
            "cannot change it by accident."
        )
        self.enable_check.toggled.connect(self._editing_toggled)
        layout.addWidget(self.enable_check)

        # -- what is selected ------------------------------------------------
        # The ticks and the timeline band are on the other side of the window,
        # and a wide change is made from here -- so this says in words what
        # they say in marks, next to the button that acts on it.
        self.applies_to = QLabel("")
        self.applies_to.setWordWrap(True)
        self.applies_to.setToolTip(
            "What the state buttons below will move. Tick projects or lamellae "
            "on the left, or Ctrl-click frames on the timeline; with nothing "
            "picked out it is the frame on screen."
        )
        layout.addWidget(self.applies_to)

        # -- reach -----------------------------------------------------------
        # Above the state controls, because it describes what is about to
        # happen; below them it would read as a report of what already did.
        # Always shown, even when it says "Covers 1 image, 1 label, 1 lamella":
        # a line that appears only sometimes is one nobody has learned to read.
        self.reach = QLabel("")
        self.reach.setWordWrap(True)
        self.reach.setToolTip(
            "How far this state change reaches. Images that share a label change "
            "together, so this can be more than the images you selected."
        )
        layout.addWidget(self.reach)

        self.reach_shared = QLabel("")
        self.reach_shared.setWordWrap(True)
        self.reach_shared.setStyleSheet(SHARED_NOTE_STYLE)
        self.reach_shared.setToolTip(self.reach.toolTip())
        layout.addWidget(self.reach_shared)

        layout.addWidget(self._build_target())

        # -- state ---------------------------------------------------------
        self.state_box = QGroupBox("State")
        state_layout = QVBoxLayout(self.state_box)
        self.state_buttons: dict[str, QRadioButton] = {}
        self.state_group = QButtonGroup(self)
        # Not exclusive: a frame whose SEM and FIB differ must be able to show
        # no option chosen, which an exclusive group cannot represent.
        self.state_group.setExclusive(False)
        for state in STATES:
            button = QRadioButton(state)
            button.setIcon(_swatch(state))
            button.setIconSize(QSize(SWATCH_SIZE, SWATCH_SIZE))
            button.setToolTip(f"{STATE_HELP[state]}\nShortcut: {STATE_KEYS[state]}")
            button.clicked.connect(lambda _checked, s=state: self.state_requested.emit(s))
            self.state_group.addButton(button)
            state_layout.addWidget(button)
            self.state_buttons[state] = button
        layout.addWidget(self.state_box)

        # -- readout -------------------------------------------------------
        self.readout = QLabel("")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        layout.addStretch(1)

        self.set_editing_enabled(False)

    def _build_target(self) -> QWidget:
        """Return the modality control.

        Above the state buttons, in the order the two are read: which images,
        then which state. It stays usable with editing off, because choosing
        what *would* move changes nothing on disk and the reach line above
        recounts as you go.
        """
        box = QGroupBox("Apply to")
        box.setToolTip(
            "Which images a state change moves. The two modalities can be in "
            "different states."
        )
        row = QHBoxLayout(box)
        self.target_buttons: dict[str, QRadioButton] = {}
        group = QButtonGroup(self)
        for target, caption, hint in (
            ("SEM", "SEM", "Change the SEM image only"),
            ("FIB", "FIB", "Change the FIB image only"),
            ("both", "Both", "Change both images together"),
        ):
            button = QRadioButton(caption)
            button.setToolTip(hint)
            button.toggled.connect(
                lambda checked, t=target: checked and self.target_changed.emit(t)
            )
            group.addButton(button)
            row.addWidget(button)
            self.target_buttons[target] = button
        self.target_buttons["SEM"].setChecked(True)
        return box

    def set_target(self, target: str) -> None:
        """Reflect the current modality without re-emitting a change."""
        button = self.target_buttons[target]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)

    def _editing_toggled(self, enabled: bool) -> None:
        self.set_editing_enabled(enabled)
        self.editing_toggled.emit(enabled)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock or unlock the controls that change data on disk.

        Only the state buttons: choosing what a change *would* apply to writes
        nothing, so the ticks and the modality control stay usable with the
        guard on, and the reach line recounts as they change.
        """
        self.state_box.setEnabled(enabled)
        if self.enable_check.isChecked() != enabled:
            self.enable_check.blockSignals(True)
            self.enable_check.setChecked(enabled)
            self.enable_check.blockSignals(False)

    def show_state(self, target_state, sem_state, fib_state) -> None:
        """Reflect the current state of the frame.

        A frame whose acquisitions disagree shows no option chosen, rather than
        implying a state it is not in.
        """
        for state, button in self.state_buttons.items():
            button.blockSignals(True)
            button.setChecked(state == target_state)
            button.blockSignals(False)

        parts = [
            f"SEM: {sem_state or 'not present'}",
            f"FIB: {fib_state or 'not present'}",
        ]
        if target_state == MIXED:
            parts.append("(modalities differ)")
        self.readout.setText("\n".join(parts))

    def show_reach(self, summary: str, shared_note: str = "") -> None:
        """Show how far the next state change would reach.

        Args:
            summary: The always-present line, e.g. ``"Covers 4 images + 1 label"``.
            shared_note: The extra line naming images pulled in by a shared
                label, or empty when nothing is shared.
        """
        self.reach.setText(summary)
        self.reach_shared.setText(shared_note)
        self.reach_shared.setVisible(bool(shared_note))

    def show_selection(self, description: str) -> None:
        """Say in words what the state buttons will move."""
        self.applies_to.setText(f"Applies to: {description}")


def _swatch(state: str) -> QIcon:
    """Return a solid colour chip matching the timeline's colour for *state*.

    Putting the colour next to the control that sets it is what makes the
    timeline readable without a separate key.
    """
    pixmap = QPixmap(SWATCH_SIZE, SWATCH_SIZE)
    pixmap.fill(STATE_COLOURS.get(state, Qt.transparent))
    return QIcon(pixmap)
