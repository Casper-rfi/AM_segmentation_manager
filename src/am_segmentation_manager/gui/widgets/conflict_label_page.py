"""The conflict resolver's label page: the tree beside the two versions.

The tree shows what is decided, and decides a whole folder, project or lamella
at a click. The viewer shows one conflict at a time, stepped through with the
arrow keys as Curate steps through frames. **The keys decide the conflict on
screen, never a heading.** Decisions are held by the resolution it is given.
"""
from __future__ import annotations

import html

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:  # Qt 6 moved QShortcut out of QtWidgets.
    from qtpy.QtGui import QShortcut
except ImportError:  # pragma: no cover - Qt 5
    from qtpy.QtWidgets import QShortcut

from ...config import SegmentationConfig
from ...operations.merge import OURS, THEIRS
from ..conflict_resolution import ConflictNode, ConflictResolution
from .conflict_tree import ConflictTree, decide_all_buttons
from .conflict_viewer import ConflictView
from .controls import GripSplitter
from .theme import BADGE_STYLES, MUTED
from .versions_bar import COMPARE_KEY_CODE

__all__ = ["KEYS", "LabelConflictPage"]

#: The keys this page steps and decides with: the sequence bound, how the hint
#: under the viewer writes it, and what it does. The hint is built from these,
#: so it cannot list a key that is not bound. Holding B to compare belongs to
#: the viewer, which says so itself.
KEYS: tuple[tuple[str, str, str], ...] = (
    ("K", "K", "keep this one"),
    ("T", "T", "take the copy's"),
    ("Backspace", "Backspace", "undecide"),
    ("Left", "←", "back"),
    ("Right", "→", "on through the lamella"),
    ("Up", "↑", "previous lamella"),
    ("Down", "↓", "next lamella"),
)

#: The keys the tree hands on rather than using to move or search.
PASSED_KEYS: tuple[int, ...] = (
    Qt.Key.Key_K, Qt.Key.Key_T, Qt.Key.Key_Backspace,
    Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
    COMPARE_KEY_CODE,
)


class LabelConflictPage(QWidget):
    """Decides label conflicts: the tree on the left, one conflict on the right."""

    #: Emitted whenever a decision changes, so the dialog can offer the next step.
    changed = Signal()

    def __init__(
        self,
        resolution: ConflictResolution,
        config: SegmentationConfig,
        *,
        advance: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """Build the page over *resolution*'s label conflicts.

        Args:
            config: Supplies the class colours the viewer draws with.
            advance: Whether deciding a conflict moves on to the next
                undecided one. A switch on the page moves it for this merge.
        """
        super().__init__(parent)
        self.resolution = resolution
        self._key: str | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panes = GripSplitter(Qt.Orientation.Horizontal)
        self.panes.addWidget(self._build_tree())
        self.panes.addWidget(self._build_viewer(config, advance))
        self.panes.setSizes([380, 820])
        layout.addWidget(self.panes)

        actions = {
            "K": lambda: self.decide_current(OURS),
            "T": lambda: self.decide_current(THEIRS),
            "Backspace": lambda: self.decide_current(None),
            "Left": lambda: self.step_within(-1),
            "Right": lambda: self.step_within(1),
            "Up": lambda: self.step_across(-1),
            "Down": lambda: self.step_across(1),
        }
        self._shortcuts = [
            QShortcut(QKeySequence(key), self, activated=actions[key])
            for key, _shown, _does in KEYS
        ]

        self.tree.set_nodes(resolution.labels, resolution)
        keys = [key for node in resolution.labels for key in node.keys]
        first = resolution.first_undecided(resolution.labels)
        self.show_conflict(first or (keys[0] if keys else None))
        self._decided(announce=False)

    def _build_tree(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.tree = ConflictTree(passed_keys=PASSED_KEYS)
        self.tree.setToolTip(
            "All labels that were changed differently on both sides.\n"
            "Click Here or Copy on a row to decide everything under it; click again to undecide."
        )
        self.tree.row_selected.connect(self._row_selected)
        self.tree.side_requested.connect(self._side_requested)
        column.addWidget(self.tree, stretch=1)

        column.addLayout(decide_all_buttons("label", self.decide_everything))

        self.outstanding = QLabel("")
        self.outstanding.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.outstanding)
        return box

    def _build_viewer(self, config: SegmentationConfig, advance: bool) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel("")
        self.heading.setTextFormat(Qt.TextFormat.RichText)
        self.heading.setWordWrap(True)
        self.heading.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.heading)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setStyleSheet(f"color: {MUTED};")
        column.addWidget(self.details)
        self.proofread = QLabel("Corrected label here: taking the copy's replaces it.")
        self.proofread.setStyleSheet(BADGE_STYLES["warning"])
        self.proofread.setHidden(True)
        column.addWidget(self.proofread)

        self.viewer = ConflictView(config)
        column.addWidget(self.viewer, stretch=1)

        choose = QHBoxLayout()
        self.keep = self._choice(
            "Keep this one (K)", OURS, "Keep the label of this collection."
        )
        self.take = self._choice(
            "Take the copy's (T)", THEIRS, "Replace it with the label from the copy."
        )
        choose.addWidget(self.keep)
        choose.addWidget(self.take)
        choose.addStretch(1)
        self.advance = QCheckBox("Move on after deciding")
        self.advance.setToolTip(
            "Go to the next undecided conflict after deciding one."
        )
        self.advance.setChecked(advance)
        choose.addWidget(self.advance)
        column.addLayout(choose)

        hint = QLabel(" · ".join(f"{shown} {does}" for _key, shown, does in KEYS))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED};")
        column.addWidget(hint)
        return box

    def _choice(self, caption: str, side: str, hint: str) -> QPushButton:
        """Return a button deciding the conflict on screen, lit while it holds *side*.

        Neither is a default, and Qt makes a dialog's buttons defaults unless
        told otherwise: a default here is a rule about which tree wins.
        """
        button = QPushButton(caption)
        button.setCheckable(True)
        button.setAutoDefault(False)
        button.setDefault(False)
        button.setToolTip(hint)
        button.clicked.connect(
            lambda checked, s=side: self.decide_current(s if checked else None)
        )
        return button

    # -- what is on screen -------------------------------------------------

    @property
    def current_key(self) -> str | None:
        """Return the conflict on screen, which is what the keys decide."""
        return self._key

    def show_conflict(self, key: str | None, *, follow_tree: bool = True) -> None:
        """Put the conflict *key* on screen, moving the tree to it unless told not to."""
        self._key = key
        if key is None:
            self.heading.setText("")
            self.details.setText("")
            self.proofread.setHidden(True)
            self.viewer.show_row(None)
            self._show_choice()
            return
        node = self.resolution.conflict(key)
        row = node.row
        assert row is not None, "a conflict key names a conflict, which carries its row"
        if follow_tree:
            self.tree.show_current(node)
        where = " — ".join(part for part in (row.project, row.lamella) if part)
        self.heading.setText(
            f"<b>{html.escape(node.caption)}</b><br>{html.escape(where)}"
        )
        self.details.setText("\n".join(row.details))
        self.proofread.setHidden(not row.corrected)
        self.viewer.show_row(row)
        self._show_choice()

    def step_within(self, offset: int) -> None:
        """Move *offset* conflicts through the lamella on screen, stopping at its ends."""
        if self._key is not None:
            self._move_to(self.resolution.step_within(self._key, offset))

    def step_across(self, offset: int) -> None:
        """Move to the first conflict of the lamella *offset* lamellae away."""
        if self._key is not None:
            self._move_to(self.resolution.step_across(self._key, offset))

    def _move_to(self, key: str | None) -> None:
        if key is not None:
            self.show_conflict(key)

    def _row_selected(self, node: ConflictNode) -> None:
        """Show what the reader moved to; a heading shows its first conflict."""
        if node.keys:
            self.show_conflict(node.keys[0], follow_tree=node.is_conflict)

    # -- deciding ------------------------------------------------------------

    def decide_current(self, side: str | None) -> None:
        """Decide the conflict on screen, or undecide it with None."""
        if self._key is None:
            return
        self.resolution.decide(self.resolution.conflict(self._key), side)
        self._decided()
        if side is not None and self.advance.isChecked():
            self._move_to(self.resolution.next_undecided(self._key))

    def decide_everything(self, side: str | None) -> None:
        """Decide every label conflict one way, or undecide them all with None."""
        self.resolution.decide_all(self.resolution.labels, side)
        self._decided()

    def _side_requested(self, node: ConflictNode, side: str) -> None:
        self.resolution.toggle(node, side)
        self._decided()

    def _decided(self, *, announce: bool = True) -> None:
        """Bring every mark, button and count into step with the decisions."""
        self.tree.show_decisions(self.resolution)
        self._show_choice()
        left = self.resolution.outstanding(self.resolution.labels)
        total = sum(len(node.keys) for node in self.resolution.labels)
        self.outstanding.setText(
            f"{left} of {total} still to decide" if left
            else f"All {total} decided"
        )
        if announce:
            self.changed.emit()

    def _show_choice(self) -> None:
        """Light the button of the side the conflict on screen holds, if any."""
        side = self.resolution.side(self._key) if self._key is not None else None
        self.keep.setChecked(side == OURS)
        self.take.setChecked(side == THEIRS)
        self.keep.setEnabled(self._key is not None)
        self.take.setEnabled(self._key is not None)

    # -- the keys ----------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Swap to the copy's label while the compare key is down."""
        if event.key() == COMPARE_KEY_CODE and not event.isAutoRepeat():
            self.viewer.set_comparing(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Put this collection's label back when the key comes up."""
        if event.key() == COMPARE_KEY_CODE and not event.isAutoRepeat():
            self.viewer.set_comparing(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Switch the keys on, and put the keyboard where they reach."""
        super().showEvent(event)
        for shortcut in self._shortcuts:
            shortcut.setEnabled(True)
        self.tree.setFocus()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Switch the keys off while the other page is in front."""
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
        self.viewer.set_comparing(False)
        super().hideEvent(event)
