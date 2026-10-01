"""The Families tab beside the Curate view: a project's families of imaging conditions.

Beside the images because whether one label fits every condition of a family, and
which condition to predict from, are judged by looking at them: choosing a
condition in the tree shows it in the view, and choosing one there selects it
here. The tree lists the whole project of the lamella in view and greys the
conditions this lamella does not hold; *Applies to* says whether a change is the
project's, reaching every lamella, or this lamella's own. Every rule is
:mod:`..families_controller`'s or the operation's, and the editing switch above
the tabs locks it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...families import StaleFamilies
from ...models import Lamella, Project
from ...operations.families_edit import FamiliesValidationError
from ...state import StaleState
from ..families_controller import FamiliesController, FamilyNode, leaving_corrected
from . import theme
from .common import no_primary_star, primary_star
from .controls import ElidingLabel

logger = logging.getLogger(__name__)

#: The Family dropdown's last entry, which asks for a name rather than being one.
NEW_FAMILY = "\x00new family"

#: The item role holding a condition row's beam, which choosing it passes on.
_MODALITY_ROLE = Qt.ItemDataRole.UserRole + 1

#: How tall the message under the buttons may grow before it scrolls: a preview
#: lists a line per change, and the operations below must stay on screen.
_MESSAGE_HEIGHT = 140

#: How far each level of the tree is indented: beam, family, condition in a narrow column.
_INDENT = 12


class FamiliesPanel(QWidget):
    """The tree of a project's families, and the controls that change them."""

    #: Emitted after a families change was written, or refused as stale: the
    #: index the window holds describes families that are no longer so.
    families_changed = Signal()
    #: Emitted with a beam and a condition chosen in the tree, to be shown.
    condition_chosen = Signal(str, str)

    def __init__(self, families: FamiliesController) -> None:
        super().__init__()
        self.families = families
        #: Asked before a change is written, with its project; the window
        #: answers whether label edits waiting there may be put aside.
        self.may_change_families: Callable[[Project], bool] = lambda project: True
        #: Every condition's row, as the tree was last built.
        self._rows: list[QTreeWidgetItem] = []

        column = QVBoxLayout(self)
        # Shortened rather than widening the column, a project's name being long.
        self.title = ElidingLabel("Families")
        self.title.setObjectName(theme.SECTION_HEADING)
        self.title.setToolTip(
            "Which imaging conditions share a label. A condition belongs to the "
            "family its folder name gives, unless AM_families.yaml moves it."
        )
        column.addWidget(self.title)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        column.addWidget(self.status)

        self.applies_row = QWidget()
        applies = QHBoxLayout(self.applies_row)
        applies.setContentsMargins(0, 0, 0, 0)
        self.applies_label = QLabel("Applies to")
        self.applies_combo = QComboBox()
        # A lamella's name is long, and the column is narrow: shortened, not widened.
        self.applies_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.applies_combo.setMinimumContentsLength(10)
        self.applies_combo.activated.connect(self._target_chosen)
        applies.addWidget(self.applies_label)
        applies.addWidget(self.applies_combo, stretch=1)
        column.addWidget(self.applies_row)

        # One column, its notes after the name: the column is narrow, and a
        # second one scrolled sideways.
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(_INDENT)
        self.tree.currentItemChanged.connect(self._selection_changed)
        column.addWidget(self.tree, stretch=1)
        column.addWidget(self._build_controls())

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message_area = QScrollArea()
        self.message_area.setWidget(self.message)
        self.message_area.setWidgetResizable(True)
        self.message_area.setFrameShape(QFrame.Shape.NoFrame)
        self.message_area.setMaximumHeight(_MESSAGE_HEIGHT)
        self.message_area.setVisible(False)
        column.addWidget(self.message_area)
        self._update_controls()

    def _build_controls(self) -> QWidget:
        """Return the controls under the tree: one row for the condition, one for the changes."""
        holder = QWidget()
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, 0, 0)

        moving = QHBoxLayout()
        self.family_label = QLabel("Family")
        self.family_combo = QComboBox()
        self.family_combo.setToolTip(
            "Move the selected condition into another family, or a new one. "
            "Its folder is not renamed: the move is kept in AM_families.yaml."
        )
        self.family_label.setToolTip(self.family_combo.toolTip())
        self.family_combo.activated.connect(self._family_chosen)
        moving.addWidget(self.family_label)
        moving.addWidget(self.family_combo, stretch=1)
        rows.addLayout(moving)

        primary = QHBoxLayout()
        self.primary_button = QPushButton("Set primary")
        self.primary_button.setToolTip(
            "Predict, measure and export the family from the selected condition, "
            "and open frames on it"
        )
        self.primary_button.clicked.connect(self._make_primary)
        self.default_primary_button = QPushButton("Reset primary")
        self.default_primary_button.setToolTip(
            "Stand the family on its base condition again, or its first variant"
        )
        self.default_primary_button.clicked.connect(self._default_primary)
        primary.addWidget(self.primary_button)
        primary.addWidget(self.default_primary_button)
        rows.addLayout(primary)

        changes = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip("Say what saving the changes would do. Nothing is written.")
        self.preview_button.clicked.connect(self._preview)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Write the changes to AM_families.yaml")
        self.save_button.clicked.connect(self._save)
        self.discard_button = QPushButton("Discard")
        self.discard_button.setToolTip("Put the families back to what is saved. Nothing is written.")
        self.discard_button.clicked.connect(self._discard)
        for button in (self.preview_button, self.save_button, self.discard_button):
            changes.addWidget(button)
        rows.addLayout(changes)
        return holder

    # -- what is shown -------------------------------------------------------

    def show_scope(self, projects: Sequence[Project], lamella: Lamella | None) -> None:
        """Show the families of the project in view, marking what *lamella* holds."""
        self.families.set_scope(projects, lamella)
        self.refresh()

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock or unlock the controls that change the families file."""
        self.families.editing_enabled = enabled
        self._update_controls()

    def select_condition(self, condition: str | None) -> None:
        """Select *condition*'s row, as when the Variant control chose it, showing nothing new."""
        wanted = (condition or "").lower()
        if (self._selected_condition() or "").lower() == wanted:
            return
        for item in self._rows:
            if str(item.data(0, Qt.ItemDataRole.UserRole)).lower() == wanted:
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(item)
                self.tree.blockSignals(False)
                self._update_controls()
                return

    def refresh(self) -> None:
        """Rebuild the tree, beam then family then condition, keeping the row selected."""
        project = self.families.project
        self.title.setText(
            f"Families: {project.display_name}" if project is not None else "Families"
        )
        status = self.families.status()
        self.status.setText(status)
        self.status.setVisible(bool(status))
        selected = self._selected_condition()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._rows = []
        beams: dict[str, QTreeWidgetItem] = {}
        for node in self.families.tree():
            beam = beams.get(node.modality)
            if beam is None:
                beam = beams[node.modality] = QTreeWidgetItem(self.tree, [node.modality])
            family = QTreeWidgetItem(beam, [node.family])
            several = len(node.conditions) > 1
            for row in node.conditions:
                item = QTreeWidgetItem(
                    family, [f"{row.name} ({row.marks})" if row.marks else row.name]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, row.name)
                item.setData(0, _MODALITY_ROLE, node.modality)
                self._rows.append(item)
                if several:
                    item.setIcon(0, primary_star() if row.primary else no_primary_star())
                stages = f"In {row.stages} stage(s) of the project"
                item.setToolTip(0, f"{stages}; {row.note}" if row.note else stages)
                if not row.here:
                    item.setForeground(0, QColor(theme.MUTED))
                if row.unsaved:
                    font = item.font(0)
                    font.setItalic(True)
                    item.setFont(0, font)
                if row.name == selected:
                    self.tree.setCurrentItem(item)
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._update_controls()

    def _selected_condition(self) -> str | None:
        """Return the condition selected in the tree, or None for a heading or nothing."""
        item = self.tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None

    def _selected_family(self) -> FamilyNode | None:
        """Return the family the selected condition is in, as the tree shows it."""
        condition = self._selected_condition()
        return next(
            (
                node for node in self.families.tree()
                if any(row.name == condition for row in node.conditions)
            ),
            None,
        )

    def _selection_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        """Offer what can be done to the row chosen, and show it if it is a condition."""
        self._update_controls()
        if current is None:
            return
        condition = current.data(0, Qt.ItemDataRole.UserRole)
        modality = current.data(0, _MODALITY_ROLE)
        if condition and modality:
            self.condition_chosen.emit(modality, condition)

    def _show_target(self) -> None:
        """Offer the project or the lamella in view, naming the lamella changes are held for."""
        lamella = self.families.target or self.families.lamella_choice
        self.applies_row.setVisible(lamella is not None)
        self.applies_combo.blockSignals(True)
        self.applies_combo.clear()
        if lamella is not None:
            self.applies_combo.addItem("Whole project", False)
            self.applies_combo.addItem(f"{lamella.name} only", True)
            self.applies_combo.setCurrentIndex(1 if self.families.target is not None else 0)
        self.applies_combo.blockSignals(False)
        held = self.families.has_changes
        self.applies_combo.setEnabled(
            self.families.editing_enabled and self.families.editable and not held
        )
        hint = (
            "Whether a change is the project's, reaching every lamella, or this "
            "lamella's own, replacing the project's here alone."
        )
        if held:
            hint += "\nSave or discard the changes first: they belong to this choice."
        self.applies_combo.setToolTip(hint)
        self.applies_label.setToolTip(hint)

    def _target_chosen(self, index: int) -> None:
        """Make changes the project's or this lamella's own, and show the families as they stand there."""
        try:
            self.families.set_target(bool(self.applies_combo.itemData(index)))
        except ValueError as error:
            QMessageBox.warning(self, "Cannot change it now", str(error))
        self.refresh()

    def _update_controls(self) -> None:
        """Offer what can be done to the selected condition, and to the changes held."""
        self._show_target()
        condition = self._selected_condition()
        node = self._selected_family()
        editing = self.families.editing_enabled and self.families.editable
        chosen = editing and condition is not None and node is not None
        self.family_combo.blockSignals(True)
        self.family_combo.clear()
        if condition is not None and node is not None:
            for caption, family in self.families.family_options(condition):
                self.family_combo.addItem(caption, family)
            self.family_combo.addItem("New family...", NEW_FAMILY)
            current = self.families.family_of(condition).lower()
            values = [
                str(self.family_combo.itemData(row)).lower()
                for row in range(self.family_combo.count())
            ]
            self.family_combo.setCurrentIndex(values.index(current) if current in values else 0)
        self.family_combo.blockSignals(False)
        self.family_combo.setEnabled(chosen)
        row = next((row for row in node.conditions if row.name == condition), None) if node else None
        self.primary_button.setEnabled(chosen and row is not None and not row.named_primary)
        self.default_primary_button.setEnabled(
            chosen and node is not None and any(r.named_primary for r in node.conditions)
        )
        changed = self.families.has_changes
        self.preview_button.setEnabled(changed)
        self.save_button.setEnabled(editing and changed)
        self.discard_button.setEnabled(changed)

    def _say(self, text: str) -> None:
        """Show *text* under the buttons, the space it takes given back while there is none."""
        self.message.setText(text)
        self.message_area.setVisible(bool(text))

    # -- changing them ----------------------------------------------------------

    def _family_chosen(self, index: int) -> None:
        """Move the selected condition into the family chosen, or a new one named now."""
        condition = self._selected_condition()
        if condition is None or index < 0:
            return
        family = self.family_combo.itemData(index)
        if family == NEW_FAMILY:
            name, accepted = QInputDialog.getText(
                self, "New family",
                f"Name the family {condition} should form or join. It starts with "
                "sem or fib and holds letters, digits, '-' and '.'.",
            )
            if not accepted or not name.strip():
                self._update_controls()
                return
            family = name.strip()
        try:
            self.families.set_family(condition, family)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot move it there", str(error))
        self._say("")
        self.refresh()

    def _make_primary(self) -> None:
        """Name the selected condition as its family's primary, held until saved."""
        node = self._selected_family()
        condition = self._selected_condition()
        if node is not None and condition is not None:
            self.families.set_primary(node.family, condition)
            self.refresh()

    def _default_primary(self) -> None:
        """Return the selected condition's family to its default primary, held until saved."""
        node = self._selected_family()
        if node is not None:
            self.families.set_primary(node.family, None)
            self.refresh()

    def _discard(self) -> None:
        """Forget the changes held, writing nothing."""
        self.families.discard()
        self._say("")
        self.refresh()

    def _preview(self) -> None:
        """Say what saving the changes would do, or why it cannot."""
        try:
            change = self.families.plan()
        except (FamiliesValidationError, ValueError) as error:
            self._say(_refusal(error))
            return
        self._say("\n".join(change.describe()))

    def _save(self) -> None:
        """Write the changes, asking first where proofreading is at stake.

        Planned again rather than taken from a preview, which may be stale.
        Images leaving a corrected group lose the claim, so that is asked;
        label edits waiting in the project are the window's to settle, a move
        changing which label they belong to.
        """
        project = self.families.project
        if project is None or not self.families.has_changes:
            self._say("No changes")
            return
        try:
            change = self.families.plan()
        except (FamiliesValidationError, ValueError) as error:
            self._say(_refusal(error))
            QMessageBox.warning(self, "Cannot save", _refusal(error))
            return
        if not self.may_change_families(project):
            self._say("Not saved")
            return
        leaving = leaving_corrected(change)
        if leaving and QMessageBox.question(
            self,
            "Leave a corrected group",
            f"{leaving} image(s) would leave a label group declared corrected, "
            "and lose that claim. The proofreading stays in the label itself.\n\n"
            "Save the change?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            self._say("Not saved")
            return
        try:
            applied = self.families.apply(change)
        except (StaleFamilies, StaleState) as error:
            QMessageBox.warning(
                self, "Somebody else changed this project",
                f"{error}\n\nNothing was changed. Reading the folder again now.",
            )
            self._say("Not saved")
            self.families_changed.emit()
            return
        except PermissionError as error:
            self._say(f"Not saved: {error}")
            return
        self._say("Saved.\n" + "\n".join(applied.describe()))
        self.families_changed.emit()


def _refusal(error: Exception) -> str:
    """Return why a families change cannot be made, one reason a line."""
    problems = getattr(error, "problems", None) or [str(error)]
    return "Cannot save:\n" + "\n".join(problems)
