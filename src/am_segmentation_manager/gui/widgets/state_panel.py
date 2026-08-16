"""Right panel: which state to move to, and how far that reaches.

*What* a change applies to is chosen on the left; this panel answers the other
half, and reports how much would move before it is made. Everything that writes
stays disabled until editing is switched on, so reviewing data cannot change it
by a stray click.

The states are mutually exclusive because the filesystem makes them so -- an
image lives in exactly one of the three locations. A frame whose acquisitions
disagree therefore shows no option chosen.
"""
from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
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
    "deselected": "Exclude from proofreading. Moves files into deselected/, "
                  "which stops the segmentation software generating labels for them.",
    "selected": "Queue for proofreading. Files sit in the normal folder.",
    "corrected": "Mark proofreading finished. Moves files into corrected/; "
                 "moving them back out asks for confirmation.",
}

SWATCH_SIZE = 12


class StatePanel(QWidget):
    """The editing guard, the reach of the next change, and the state controls."""

    #: Emitted when the editing guard is switched.
    editing_toggled = Signal(bool)
    #: Emitted with the state the user asked for.
    state_requested = Signal(str)

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
        # Always shown, even when it says "Moves 1 image + 1 label": a line
        # that appears only sometimes is a line nobody has learned to read.
        self.reach = QLabel("")
        self.reach.setWordWrap(True)
        self.reach.setToolTip(
            "How much this state change moves. Images that share a label move "
            "together, so this can be more than the images you selected."
        )
        layout.addWidget(self.reach)

        self.reach_shared = QLabel("")
        self.reach_shared.setWordWrap(True)
        self.reach_shared.setStyleSheet(SHARED_NOTE_STYLE)
        self.reach_shared.setToolTip(self.reach.toolTip())
        layout.addWidget(self.reach_shared)

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
        """Show how much the next state change would move.

        Args:
            summary: The always-present line, e.g. ``"Moves 4 images + 1 label"``.
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
