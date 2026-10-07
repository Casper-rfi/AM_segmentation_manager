"""The proofreading checklist beside the Label tab's canvas, and each step's help.

Draws what :class:`~..proofreading.Checklist` holds for the label on screen and
reports ticks back; it decides nothing. The steps are rows of a list, ticked as
every list here is; under them, the step selected says what to do in it, and its
help explains how.
"""
from __future__ import annotations

import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...config import default_config_path, load_segmentation_config
from ...display import compose_overlay
from ..proofreading import Step, read_example
from .controls import SegmentedControl, WrappingLabel
from .image_view import to_qimage
from .lists import CheckList
from .theme import MUTED

logger = logging.getLogger(__name__)

#: The width an example is drawn at, or near: a narrower crop is magnified to it.
EXAMPLE_WIDTH = 384

def magnification(width: int) -> int:
    """Return the whole factor an example *width* pixels wide is drawn larger by.

    Whole, so every pixel of the label stays a sharp square: a crop narrowed to
    show a thin edge is shown bigger rather than lost in the window.
    """
    return max(1, EXAMPLE_WIDTH // max(1, width))


#: The panel's width: room for the longest step's title on two lines.
PANEL_WIDTH = 270

#: What the panel says where there is no label to proofread through.
NOTHING_TO_CHECK = "No label to proofread on this frame."


class StepHelp(QDialog):
    """A step's explanation and its example, in a window that stays open while you work.

    The example is drawn here from its image and its two labels, by the code
    that draws every overlay, in the packaged classes' colours: the indices
    were painted with those classes, whatever the project in view uses.
    """

    def __init__(self, parent) -> None:
        """Build the window, empty until :meth:`show_step` fills it."""
        super().__init__(parent)
        self.setMinimumWidth(EXAMPLE_WIDTH + 24)
        self._pictures: dict[str, tuple[QPixmap, QPixmap] | None] = {}
        layout = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold;")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.instructions = WrappingLabel()
        self.instructions.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.instructions)
        self.explanation = WrappingLabel()
        layout.addWidget(self.explanation)

        self.example = QLabel()
        self.example.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.example)
        self.example_view = SegmentedControl([
            ("before", "Before", "The label before this step"),
            ("after", "After", "The label once this step is done"),
        ])
        self.example_view.chosen.connect(self._show_picture)
        layout.addWidget(self.example_view, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._step: Step | None = None

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def show_step(self, number: int, step: Step) -> None:
        """Show *step*, numbered *number* from 1, and bring the window forward."""
        self._step = step
        self.setWindowTitle(f"Step {number}: {step.title}")
        self.title.setText(f"{number}. {step.title}")
        self.instructions.setText(
            "<ul style='margin-left: -20px;'>"
            + "".join(f"<li>{line}</li>" for line in step.instructions)
            + "</ul>"
        )
        self.explanation.setText(step.explanation)
        has_example = self._pictures_of(step) is not None
        self.example.setVisible(has_example)
        self.example_view.setVisible(has_example)
        if has_example:
            self.example_view.set_current("before")
            self._show_picture("before")
        self.adjustSize()
        self.show()
        self.raise_()

    def _show_picture(self, which: str) -> None:
        pictures = self._pictures_of(self._step) if self._step is not None else None
        if pictures is not None:
            self.example.setPixmap(pictures[0] if which == "before" else pictures[1])

    def _pictures_of(self, step: Step) -> tuple[QPixmap, QPixmap] | None:
        """Return the step's example drawn before and after, or None without one.

        Read once and kept: a window that stays open is asked again on every
        step anyone opens. An example that will not read is logged and left
        out, the explanation being help enough without it.
        """
        if step.example not in self._pictures:
            self._pictures[step.example] = None
            try:
                example = read_example(step.example)
            except (OSError, RuntimeError, ValueError):
                logger.exception("Could not read the example %r", step.example)
                example = None
            if example is not None:
                config = load_segmentation_config(default_config_path())
                factor = magnification(example.image.shape[1])
                height, width = example.image.shape[:2]

                def drawn(label) -> QPixmap:
                    return QPixmap.fromImage(
                        to_qimage(compose_overlay(example.image, label, config)).scaled(
                            factor * width,
                            factor * height,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.FastTransformation,
                        )
                    )

                self._pictures[step.example] = (drawn(example.before), drawn(example.after))
        return self._pictures[step.example]


class ChecklistPanel(QGroupBox):
    """The steps of proofreading the label on screen, ticked off as they are done.

    The save step's row cannot be ticked by hand: it ticks itself when the
    label is saved and marked corrected, and saying otherwise would be a claim
    about the file that nothing had checked.
    """

    #: A step's row was ticked or unticked: its index, and whether it is now ticked.
    step_toggled = Signal(int, bool)

    def __init__(self, steps: tuple[Step, ...]) -> None:
        """Build a row for each of *steps*."""
        super().__init__("Checklist")
        self.steps = steps
        self.setFixedWidth(PANEL_WIDTH)
        self.help = StepHelp(self)
        self._current: int | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        self.nothing = WrappingLabel(NOTHING_TO_CHECK)
        self.nothing.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.nothing)

        self.list = CheckList()
        # Wrapped rather than elided: a step's title is the instruction.
        self.list.setWordWrap(True)
        self.list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.set_values(
            [f"{index + 1}. {step.title}" for index, step in enumerate(steps)],
            tooltips=[" ".join(step.instructions) for step in steps],
        )
        last = self.list.item(len(steps) - 1)
        last.setFlags(last.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        last.setToolTip("Ticks itself once this label is saved and marked corrected.")
        self.list.row_ticked.connect(self.step_toggled.emit)
        self.list.currentRowChanged.connect(self._describe)
        layout.addWidget(self.list, stretch=1)

        self.instructions = WrappingLabel()
        self.instructions.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.instructions)
        self.help_button = QPushButton("How to do this step...")
        self.help_button.clicked.connect(
            lambda: self.show_help(max(0, self.list.currentRow()))
        )
        layout.addWidget(self.help_button)

        self.hint = WrappingLabel("C: next step  ·  Shift+C: back")
        self.hint.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.hint)
        self._describe(0)

    def show_progress(self, ticked: frozenset[int] | None, current: int) -> None:
        """Show which steps are ticked, and select the current one.

        Args:
            ticked: The steps ticked, or None where there is no label to check.
            current: The first step not ticked.
        """
        checking = ticked is not None
        self.nothing.setVisible(not checking)
        for widget in (self.list, self.instructions, self.help_button):
            widget.setEnabled(checking)
        self.list.show_ticks(
            [ticked is not None and step in ticked for step in range(len(self.steps))]
        )
        self._current = min(current, len(self.steps) - 1) if checking else None
        for row in range(self.list.count()):
            item = self.list.item(row)
            font = item.font()
            font.setBold(row == self._current)
            item.setFont(font)
        if self._current is not None:
            self.list.setCurrentRow(self._current)

    @property
    def current(self) -> int | None:
        """Return the step shown as current, or None where nothing is checked."""
        return self._current

    def is_ticked(self, step: int) -> bool:
        """Return whether *step*'s row is ticked."""
        return step in self.list.ticked_rows()

    def _describe(self, step: int) -> None:
        """Say what to do in *step*, under the list."""
        if not 0 <= step < len(self.steps):
            return
        self.instructions.setText(
            f"<b>Step {step + 1}</b><br>"
            + "<br>".join(f"&bull; {line}" for line in self.steps[step].instructions)
        )

    def show_help(self, index: int) -> None:
        """Open the explanation of step *index*."""
        self.help.show_step(index + 1, self.steps[index])
