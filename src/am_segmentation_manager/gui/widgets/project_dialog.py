"""Choosing which directories become projects.

The one place in the window where a person, not the code, decides what a project
is: which directory of an unmarked dataset is the project is a fact about the
acquisition, so this proposes and a person confirms. Two lists for two decisions:
projects already indexed need only the descriptive file and start ticked;
unindexed directories are proposals, each saying what it would become, one
gathering several datasets left unticked with the reason, and *Look inside*
replacing it with the directories below.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ...operations.init_project import project_placement_problem
from ..operations_controller import OperationsController
from .paths import choose_directory

logger = logging.getLogger(__name__)

#: Marks a row that stands for a directory rather than a heading. Headings are
#: rows too, so that one list can carry both groups without a second widget.
ROW_PATH = Qt.ItemDataRole.UserRole
ROW_KIND = Qt.ItemDataRole.UserRole + 1


class AddProjectsDialog(QDialog):
    """Proposes directories to make projects, and writes the ones ticked."""

    def __init__(self, parent, controller: OperationsController, root: Path) -> None:
        """Build the dialog and scan *root*.

        Args:
            parent: The window it belongs to.
            controller: Supplies the proposals and writes the files. Holds no
                Qt, so the whole decision is testable without this dialog.
            root: The folder to scan first. Usually the one already open.
        """
        super().__init__(parent)
        self.controller = controller
        self.root = Path(root)
        self.setWindowTitle("Add projects")
        self.resize(760, 520)

        layout = QVBoxLayout(self)

        explanation = QLabel(
            "Autolamella projects are found by their experiment.yaml and need nothing here. Tick "
            "the other folders that should become projects."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        where = QHBoxLayout()
        self.path_box = QLineEdit(str(self.root))
        self.path_box.setReadOnly(True)
        self.choose_button = QPushButton("Choose folder...")
        self.choose_button.clicked.connect(self._choose)
        where.addWidget(self.path_box, stretch=1)
        where.addWidget(self.choose_button)
        layout.addLayout(where)

        self.rows = QListWidget()
        self.rows.setAlternatingRowColors(True)
        self.rows.currentItemChanged.connect(self._update_buttons)
        # Ticking a row is the other thing that changes whether there is
        # anything to write, and the common one: a list whose only proposal
        # gathers datasets opens with nothing ticked.
        self.rows.itemChanged.connect(self._update_buttons)
        layout.addWidget(self.rows, stretch=1)

        actions = QHBoxLayout()
        self.descend_button = QPushButton("Look inside")
        self.descend_button.setToolTip(
            "Replace the selected row with the folders inside it. Use for a folder that holds "
            "several projects."
        )
        self.descend_button.clicked.connect(self._descend)
        self.itself_button = QPushButton("Add this folder itself")
        self.itself_button.setToolTip(
            "Add the selected folder as one project. Use for data without sem or fib folders, "
            "which the scan cannot recognise."
        )
        self.itself_button.clicked.connect(self._add_itself)
        actions.addWidget(self.descend_button)
        actions.addWidget(self.itself_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Write metadata files")
        # Cancel is the default, as in every dialog here that writes: Return
        # should not be able to create projects by accident.
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons

        self.scan()

    # -- filling the list --------------------------------------------------

    def scan(self) -> None:
        """Read the chosen folder and rebuild the two lists."""
        self.rows.clear()
        undescribed = self.controller.undescribed_projects(self.root)
        proposals = self.controller.propose_projects(self.root)

        if undescribed:
            self._add_heading(f"Projects without a metadata file ({len(undescribed)})")
            for path in undescribed:
                self.rows.addItem(
                    self._row(
                        path,
                        self.controller.relative(path),
                        "indexed already; the file adds the descriptive fields",
                        ticked=True,
                        kind="project",
                    )
                )

        if proposals:
            self._add_heading(f"Not projects yet ({len(proposals)})")
            for candidate in proposals:
                self.rows.addItem(self._proposal_item(candidate))

        if not undescribed and not proposals:
            self._add_heading("Nothing found that needs a metadata file")
        self._update_buttons()

    def _add_heading(self, text: str) -> None:
        item = QListWidgetItem(text)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setData(ROW_KIND, "heading")
        self.rows.addItem(item)

    def _proposal_item(self, candidate) -> QListWidgetItem:
        """Return one proposal, unticked where it looks like a gathering folder."""
        heading, detail = self.controller.describe_proposal(candidate)
        return self._row(
            candidate.path,
            heading,
            detail,
            ticked=not candidate.holds_several_datasets,
            kind="candidate",
        )

    def _row(
        self, path: Path, heading: str, detail: str, *, ticked: bool, kind: str
    ) -> QListWidgetItem:
        item = QListWidgetItem(f"{heading}\n      {detail}")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
        item.setData(ROW_PATH, str(path))
        item.setData(ROW_KIND, kind)
        return item

    # -- the two extra actions ---------------------------------------------

    def _choose(self) -> None:
        """Point the dialog at another folder and scan it."""
        chosen = choose_directory(self, "Folder to look in", self.root)
        if chosen is None:
            return
        self.root = chosen
        self.path_box.setText(str(chosen))
        self.scan()

    def _descend(self) -> None:
        """Replace the selected proposal with what is one level inside it."""
        item = self.rows.currentItem()
        if item is None or item.data(ROW_KIND) != "candidate":
            return
        directory = Path(item.data(ROW_PATH))
        inside = self.controller.propose_projects_below(directory)
        if not inside:
            self.status.setText(
                f"No folder inside {directory.name} looks like a project."
            )
            return
        row = self.rows.row(item)
        self.rows.takeItem(row)
        for offset, candidate in enumerate(inside):
            self.rows.insertItem(row + offset, self._proposal_item(candidate))
        self.status.setText(
            f"{directory.name} replaced by the {len(inside)} folder(s) inside it."
        )
        self._update_buttons()

    def _add_itself(self) -> None:
        """Offer the chosen folder as a project, whatever the scan found.

        Refused where the folder is a collection: the escape hatch exists for
        data the scan cannot recognise, not for overriding the one check that
        stops several samples landing under a single metadata record.
        """

        problem = project_placement_problem(self.root)
        if problem is not None:
            QMessageBox.warning(self, "Cannot add this folder", problem)
            return
        if any(
            self.rows.item(row).data(ROW_PATH) == str(self.root)
            for row in range(self.rows.count())
        ):
            self.status.setText(f"{self.root.name} is already listed.")
            return
        self._add_heading("Chosen by hand (1)")
        self.rows.addItem(
            self._row(
                self.root,
                self.controller.relative(self.root),
                "the folder itself, whatever the scan found in it",
                ticked=True,
                kind="candidate",
            )
        )
        self._update_buttons()

    def _update_buttons(self, *_args) -> None:
        """Offer *Look inside* only for a proposal, and OK only with a tick."""
        item = self.rows.currentItem()
        self.descend_button.setEnabled(
            item is not None and item.data(ROW_KIND) == "candidate"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.chosen()))

    # -- what was decided --------------------------------------------------

    def chosen(self) -> list[Path]:
        """Return the directories ticked, in the order they are listed."""
        return [
            Path(item.data(ROW_PATH))
            for row in range(self.rows.count())
            if (item := self.rows.item(row)).data(ROW_KIND) != "heading"
            and item.checkState() == Qt.CheckState.Checked
        ]
