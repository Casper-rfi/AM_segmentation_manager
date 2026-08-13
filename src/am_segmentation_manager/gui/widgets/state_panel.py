"""Right panel: changing the workflow state of the current frame.

Everything here is disabled until editing is explicitly switched on, so that
reviewing data cannot change it by a stray click.

State is presented as a set of mutually exclusive choices rather than
independent checkboxes, because that is what the filesystem enforces: an image
lives in exactly one of the selected, deselected and corrected locations. When
a frame's two acquisitions disagree, no option is shown as chosen.
"""
from __future__ import annotations

from qtpy.QtCore import Signal
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

STATES: tuple[RecordState, ...] = ("selected", "deselected", "corrected")


class StatePanel(QWidget):
    """The editing guard, the edit target, and the state controls."""

    #: Emitted when the editing guard is switched.
    editing_toggled = Signal(bool)
    #: Emitted with the new edit target: 'SEM', 'FIB' or 'both'.
    target_changed = Signal(str)
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
        self.target_buttons: dict[str, QRadioButton] = {}
        target_group = QButtonGroup(self)
        for target, caption in (("SEM", "SEM"), ("FIB", "FIB"), ("both", "Both")):
            button = QRadioButton(caption)
            button.toggled.connect(
                lambda checked, t=target: checked and self.target_changed.emit(t)
            )
            target_group.addButton(button)
            target_layout.addWidget(button)
            self.target_buttons[target] = button
        self.target_buttons["both"].setChecked(True)
        layout.addWidget(self.target_box)

        # -- state ---------------------------------------------------------
        self.state_box = QGroupBox("State")
        state_layout = QVBoxLayout(self.state_box)
        self.state_buttons: dict[str, QRadioButton] = {}
        self.state_group = QButtonGroup(self)
        self.state_group.setExclusive(False)
        for state in STATES:
            button = QRadioButton(state)
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

    def append_log(self, message: str) -> None:
        """Record a change, so what happened stays visible."""
        self.log.appendPlainText(message)
