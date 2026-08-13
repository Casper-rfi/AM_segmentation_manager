"""Right panel: changing the workflow state of the current frame.

Everything here is disabled until editing is explicitly switched on, so that
reviewing data cannot change it by a stray click.

State is presented as a set of mutually exclusive choices rather than
independent checkboxes, because that is what the filesystem enforces: an image
lives in exactly one of the selected, deselected and corrected locations. When
a frame's two acquisitions disagree, no option is shown as chosen.
"""
from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ...operations.selection import RecordState
from ..controller import MIXED
from .frame_strip import STATE_COLOURS

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
    """The editing guard, the edit target, and the state controls."""

    #: Emitted when the editing guard is switched.
    editing_toggled = Signal(bool)
    #: Emitted with the new edit target: 'SEM', 'FIB' or 'both'.
    target_changed = Signal(str)
    #: Emitted with the new edit scope: 'frame', 'lamella' or 'project'.
    scope_changed = Signal(str)
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

        # -- target --------------------------------------------------------
        self.target_box = QGroupBox("Apply to")
        target_layout = QVBoxLayout(self.target_box)
        self.target_box.setToolTip(
            "Which images a state change moves. The two modalities can be in "
            "different states."
        )
        self.target_buttons: dict[str, QRadioButton] = {}
        target_group = QButtonGroup(self)
        for target, caption, hint in (
            ("SEM", "SEM", "Change the SEM image only"),
            ("FIB", "FIB", "Change the FIB image only"),
            ("both", "Both", "Change both images of this frame together"),
        ):
            button = QRadioButton(caption)
            button.setToolTip(hint)
            button.toggled.connect(
                lambda checked, t=target: checked and self.target_changed.emit(t)
            )
            target_group.addButton(button)
            target_layout.addWidget(button)
            self.target_buttons[target] = button
        self.target_buttons["SEM"].setChecked(True)
        layout.addWidget(self.target_box)

        # -- scope ---------------------------------------------------------
        self.scope_box = QGroupBox("Across")
        scope_layout = QVBoxLayout(self.scope_box)
        self.scope_box.setToolTip(
            "How many images a state change covers. Anything wider than the "
            "current frame is confirmed before any file is moved."
        )
        self.scope_buttons: dict[str, QRadioButton] = {}
        scope_group = QButtonGroup(self)
        for scope, caption, hint in (
            ("frame", "This frame", "Only the frame on screen"),
            ("lamella", "This lamella", "Every frame of the current lamella"),
            ("project", "This project", "Every frame of every lamella in the project"),
        ):
            button = QRadioButton(caption)
            button.setToolTip(hint)
            button.toggled.connect(
                lambda checked, s=scope: checked and self.scope_changed.emit(s)
            )
            scope_group.addButton(button)
            scope_layout.addWidget(button)
            self.scope_buttons[scope] = button
        self.scope_buttons["frame"].setChecked(True)
        layout.addWidget(self.scope_box)

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

        log_box = QGroupBox("Changes")
        log_layout = QVBoxLayout(log_box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        log_layout.addWidget(self.log)
        layout.addWidget(log_box, stretch=1)

        self.set_editing_enabled(False)

    def _editing_toggled(self, enabled: bool) -> None:
        self.set_editing_enabled(enabled)
        self.editing_toggled.emit(enabled)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock or unlock the controls that change data on disk."""
        self.target_box.setEnabled(enabled)
        self.scope_box.setEnabled(enabled)
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

    def set_target(self, target: str) -> None:
        """Reflect the current edit target without re-emitting a change."""
        button = self.target_buttons[target]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)

    def set_scope(self, scope: str) -> None:
        """Reflect the current edit scope without re-emitting a change."""
        button = self.scope_buttons[scope]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)

    def append_log(self, message: str) -> None:
        """Record a change, so what happened stays visible."""
        self.log.appendPlainText(message)


def _swatch(state: str) -> QIcon:
    """Return a solid colour chip matching the timeline's colour for *state*.

    Putting the colour next to the control that sets it is what makes the
    timeline readable without a separate key.
    """
    pixmap = QPixmap(SWATCH_SIZE, SWATCH_SIZE)
    pixmap.fill(STATE_COLOURS.get(state, Qt.transparent))
    return QIcon(pixmap)
