"""Right panel: which images to change, to which state, and how far that reaches.

*Which rows* are chosen on the left; this panel chooses the modality, the family
where there are several, and the state, and reports the reach before a change is
made, beside the buttons that act.
Everything that *writes* stays disabled until editing is on, and a series whose
images carry no labels stays disabled regardless, the two reasons held apart so
lifting one cannot lift the other. States are mutually exclusive, a group having
one entry in its stage's state file, so a frame whose images disagree shows no
option chosen.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...operations.selection import RecordState
from ..controller import MIXED
from .common import MAX_LISTED_FILES, breakable
from .controls import SegmentedControl, ToggleSwitch
from .frame_strip import STATE_COLOURS
from .theme import BADGE_STYLES

#: Workflow order, which is also the order of the rows in the timeline: data is
#: excluded, or queued for proofreading, or finished.
STATES: tuple[RecordState, ...] = ("deselected", "selected", "corrected")

#: Key that applies each state, shown in its tooltip.
STATE_KEYS = {"deselected": "D", "selected": "S", "corrected": "C"}

STATE_HELP = {
    "deselected": "Exclude from proofreading and model training.",
    "selected": "Queue for proofreading, or use as uncorrected pseudo-labels.",
    "corrected": "Mark proofreading finished. Leaving this state again asks for confirmation.",
}

SWATCH_SIZE = 12


class StatePanel(QWidget):
    """The editing guard, the reach of the next change, and the state controls.

    The guard is the window's to place, above the State and Families tabs: it
    unlocks both, and one switch per tab could disagree.
    """

    #: Emitted when the editing guard is switched.
    editing_toggled = Signal(bool)
    #: Emitted with the state the user asked for.
    state_requested = Signal(str)
    #: Emitted with the modality a state change applies to.
    target_changed = Signal(str)
    #: Emitted with ``"this"`` or ``"every"``: which families a change reaches.
    family_scope_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        #: The two independent reasons the state buttons may be locked, kept
        #: apart so that switching editing back on cannot unlock a series that
        #: has no labels to change. Recomputed together, as the operations
        #: panel recomputes its own pair.
        self._editing_enabled = False
        self._series_is_editable = True

        # -- the guard -----------------------------------------------------
        self.enable_check = ToggleSwitch("Enable editing")
        self.enable_check.setToolTip(
            "Unlocks the controls below"
        )
        self.enable_check.toggled.connect(self._editing_toggled)

        # -- what is selected ------------------------------------------------
        # The ticks and the timeline band are on the other side of the window,
        # and a wide change is made from here -- so this says in words what
        # they say in marks, next to the button that acts on it.
        self.applies_to = QLabel("")
        self.applies_to.setWordWrap(True)
        self.applies_to.setToolTip(
            "Selected images for which the state would change. Tick projects or lamellae on the "
            "left, or Ctrl-click frames on the timeline."
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
        # The warning badge rather than tinted body text: it reports a
        # consequence nobody asked for, which the line above cannot show.
        self.reach_shared.setStyleSheet(BADGE_STYLES["warning"])
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

        # -- what was last written -------------------------------------------
        # The one thing on this tab that changes files without a dialog to
        # report it is a state change from these buttons, so the answer to
        # "what did that just do" belongs beside them. Everything else --
        # converting, compressing, saving labels -- reports in its own dialog
        # and is repeated here only so the last thing written is on screen.
        self.last_change = QLabel("")
        self.last_change.setWordWrap(True)
        # Never a floor under the panel's width, which is the splitter's to set:
        # a name that cannot break is cut at the edge rather than widening it.
        policy = self.last_change.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.last_change.setSizePolicy(policy)
        self.last_change.setStyleSheet(BADGE_STYLES["info"])
        self.last_change.setToolTip(
            "The last change written to disk this session."
        )
        layout.addWidget(self.last_change)
        layout.addStretch(1)

        self.set_editing_enabled(False)

    def _build_target(self) -> QWidget:
        """Return the modality control, and the family control beneath it.

        Above the state buttons, in the order the two are read: which images,
        then which state. It stays usable with editing off, because choosing
        what *would* move changes nothing on disk and the reach line above
        recounts as you go.

        The family control is shown only where the scope holds several
        families of a beam: every other scope has nothing for it to choose.
        """
        box = QGroupBox("Apply to")
        box.setToolTip(
            "Which images a state change moves. The state for FIB and SEM is set independently."
        )
        rows = QVBoxLayout(box)
        row = QHBoxLayout()
        self.target_control = SegmentedControl([
            ("SEM", "SEM", "Change the SEM image only"),
            ("FIB", "FIB", "Change the FIB image only"),
            ("both", "Both", "Change both images together"),
        ])
        self.target_control.chosen.connect(self.target_changed.emit)
        #: The segments, so a caller can reach one at a time.
        self.target_buttons = self.target_control.buttons
        row.addWidget(self.target_control)
        row.addStretch()
        rows.addLayout(row)
        self.target_buttons["SEM"].setChecked(True)

        self.family_control = SegmentedControl([
            (
                "this", "This family",
                "Change only the family on screen. All variants within one family share one label "
                "and selection state together.",
            ),
            ("every", "Every family", "Change every family of the beam"),
        ])
        self.family_control.chosen.connect(self.family_scope_changed.emit)
        self.family_control.set_current("this")
        self.family_control.setVisible(False)
        rows.addWidget(self.family_control)
        return box

    def set_target(self, target: str) -> None:
        """Reflect the current modality without re-emitting a change."""
        self.target_control.set_current(target)

    def show_family_choice(self, offered: bool, scope: str) -> None:
        """Show the family control where there is a choice, reflecting *scope*."""
        self.family_control.setVisible(offered)
        self.family_control.set_current(scope)

    def _editing_toggled(self, enabled: bool) -> None:
        self.set_editing_enabled(enabled)
        self.editing_toggled.emit(enabled)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock or unlock the controls that change data on disk.

        Only the state buttons: choosing what a change *would* apply to writes
        nothing, so the ticks and the modality control stay usable with the
        guard on, and the reach line recounts as they change.
        """
        self._editing_enabled = bool(enabled)
        self._apply_lock()
        if self.enable_check.isChecked() != enabled:
            self.enable_check.blockSignals(True)
            self.enable_check.setChecked(enabled)
            self.enable_check.blockSignals(False)

    def set_series_editable(self, editable: bool) -> None:
        """Lock the state buttons for a series whose images have no labels.

        The reviewer's own toggle is left alone: this is not a guard they can
        switch off, and clearing their setting to express it would then be
        restored the moment they moved to a lamella that does have labels. The
        two reasons are held apart and applied together.
        """
        self._series_is_editable = bool(editable)
        self._apply_lock()

    def _apply_lock(self) -> None:
        """Enable the state buttons only where both reasons allow it."""
        allowed = self._editing_enabled and self._series_is_editable
        self.state_box.setEnabled(allowed)
        self.state_box.setToolTip(
            ""
            if self._series_is_editable
            else "These images have no labels, so there is no state to declare."
        )

    def show_state(self, target_state, sem_state, fib_state) -> None:
        """Reflect the current state of the frame.

        A frame whose images disagree -- its two beams, or under *Every
        family* two families -- shows no option chosen, rather than implying a
        state it is not in.
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
            parts.append("(the images it covers differ)")
        self.readout.setText("\n".join(parts))

    def show_last_change(self, description: str) -> None:
        """Say what was last written, or clear the line with an empty string."""
        self.last_change.setText(breakable(description))
        self.last_change.setVisible(bool(description))

    def show_reach(
        self, summary: str, shared_note: str = "", shared_images: Sequence[str] = ()
    ) -> None:
        """Show how far the next state change would reach.

        Args:
            summary: The always-present line, e.g. ``"Covers 4 images + 1 label"``.
            shared_note: The extra line saying how many images a shared label
                pulls in, or empty when nothing is shared.
            shared_images: Those images, named on hover: the count says a change
                reaches further, and only the names say what it reaches.
        """
        self.reach.setText(summary)
        self.reach_shared.setText(shared_note)
        self.reach_shared.setVisible(bool(shared_note))
        listed = list(shared_images[:MAX_LISTED_FILES])
        if len(shared_images) > MAX_LISTED_FILES:
            listed.append(f"... and {len(shared_images) - MAX_LISTED_FILES} more")
        self.reach_shared.setToolTip(
            "Also changed, through the label they share:\n" + "\n".join(listed)
            if listed else self.reach.toolTip()
        )

    def show_selection(self, description: str) -> None:
        """Say in words what the state buttons will move."""
        self.applies_to.setText(f"Applies to: {description}")


def _swatch(state: str) -> QIcon:
    """Return a solid colour chip matching the timeline's colour for *state*.

    Putting the colour next to the control that sets it is what makes the
    timeline readable without a separate key.
    """
    pixmap = QPixmap(SWATCH_SIZE, SWATCH_SIZE)
    pixmap.fill(STATE_COLOURS.get(state, Qt.GlobalColor.transparent))
    return QIcon(pixmap)
