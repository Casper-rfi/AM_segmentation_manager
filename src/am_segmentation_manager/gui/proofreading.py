"""The proofreading checklist: the steps, and how far each label has got through them.

An aid against forgetting a step, held in memory only: nothing here is written
to disk or recorded, and closing the window forgets it. Progress is kept per
label, keyed by the path the label is written to, so imaging conditions sharing
a label share their progress as they share the label. The last step is saving,
which ticks itself when the label is saved and marked corrected.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """One step of proofreading a label.

    Attributes:
        title: What the step is, as the checklist lists it.
        instructions: What to do in it, a line each.
        explanation: How to do it, for the step's help.
    """

    title: str
    instructions: tuple[str, ...]
    explanation: str


#: The steps, in the order a label is proofread. The last is saving.
STEPS: tuple[Step, ...] = (
    Step(
        "Remove false positives outside the lamella",
        ("Use the speck finder (X) to make them easier to see.",),
        "Look for label on material that is not the lamella, the GIS layer or "
        "the vacuum beside them, and paint it over with background. The speck "
        "finder (X) hides the image and draws the label at full strength, so a "
        "stray patch stands out against the background.",
    ),
    Step(
        "Correct the sides of the lamella",
        (
            "Add lamella label where it is missing.",
            "Remove lamella label where it does not belong.",
        ),
        "Follow each side of the lamella from top to bottom. Paint lamella where "
        "the label stops short of the edge, and the neighbouring class where it "
        "runs past it. Locking a class with its padlock keeps the brush from "
        "painting over it, so an edge can be pushed without disturbing the "
        "class beyond it.",
    ),
    Step(
        "Add, remove and correct cracks and holes",
        (
            "Hide the label (V) and look for cracks and holes that are not "
            "annotated, and for crack/hole label where there is none.",
            "Outline each crack or hole with the brush (D), then fill it (F).",
            "Remove any crack/hole label that covers the lamella surface.",
        ),
        "Cracks and holes are easy to miss under the label, so hide it (V) and "
        "compare. To annotate one, trace its outline with crack/hole using the "
        "brush (D), then click inside the outline with the fill tool (F). Where "
        "crack/hole label lies on intact lamella surface, paint it back to "
        "lamella.",
    ),
    Step(
        "Correct the GIS-lamella boundary",
        (
            "Remove GIS label that is actually lamella.",
            "Add GIS label to any GIS that was missed.",
            "Correct the sides of the GIS layer.",
            "If there are many errors, consider erasing the whole boundary area "
            "and drawing it by hand instead.",
        ),
        "Follow the edge between the GIS layer and the lamella across its width. "
        "Paint lamella where GIS label covers lamella, and GIS where the layer "
        "was missed, then check the layer's two ends. Where the edge is wrong "
        "in many places, painting the whole region along it with one class and "
        "drawing the edge again is quicker than mending it.",
    ),
    Step(
        "Correct the GIS-vacuum boundary",
        (
            "Add missing GIS label.",
            "Change GIS label on vacuum to vacuum.",
        ),
        "Follow the outer edge of the GIS layer, where it meets the vacuum. "
        "Paint GIS where the layer was missed, and vacuum where GIS label "
        "reaches into the vacuum.",
    ),
    Step(
        "Correct the lamella-vacuum boundary",
        (
            "Add missing lamella label.",
            "Change lamella label on vacuum to vacuum.",
        ),
        "Follow the edge of the lamella where it meets the vacuum. Paint lamella "
        "where the label stops short of it, and vacuum where lamella label "
        "reaches into the vacuum.",
    ),
    Step(
        "Correct the vacuum-background boundaries",
        (
            "Add missing vacuum label.",
            "Remove vacuum label on other material.",
            "Check that the top and bottom edges are completely filled.",
        ),
        "Follow the edges of the vacuum. Paint vacuum where it was missed, and "
        "the right class where vacuum label lies on other material. The vacuum "
        "should reach the top and bottom edges of the image without a gap: "
        "switching the Image off makes a gap along an edge easy to see.",
    ),
    Step(
        "Find stray pixels with the speck finder",
        ("Turn on the speck finder (X), and step through the specks with N.",),
        "Painting leaves small stray pieces behind. With the speck finder on (X), "
        "N moves to the next speck; paint over those that are mistakes. Some "
        "specks are real, so it removes nothing itself.",
    ),
    Step(
        "Save and mark corrected",
        ("Press C or Ctrl+S, and choose Save and mark corrected.",),
        "Saving writes the label, and marking it corrected keeps every automated "
        "operation from overwriting it. Marking needs Enable editing switched on, "
        "on the Curate tab. This step ticks itself once the label is saved and "
        "marked corrected.",
    ),
)

class Checklist:
    """How far through :data:`STEPS` each label has got, in memory."""

    def __init__(self, steps: tuple[Step, ...] = STEPS) -> None:
        """Start with nothing ticked."""
        self.steps = steps
        self._ticked: dict[Path, set[int]] = {}

    @property
    def save_step(self) -> int:
        """Return the step a save ticks: the last."""
        return len(self.steps) - 1

    def ticked(self, key: Path) -> frozenset[int]:
        """Return the steps ticked for the label at *key*."""
        return frozenset(self._ticked.get(key, ()))

    def current(self, key: Path) -> int:
        """Return the first step not ticked, or ``len(steps)`` where every one is."""
        ticked = self._ticked.get(key, set())
        return next(
            (step for step in range(len(self.steps)) if step not in ticked), len(self.steps)
        )

    def set_ticked(self, key: Path, step: int, ticked: bool) -> None:
        """Tick or untick *step* for the label at *key*.

        Raises:
            IndexError: If there is no such step.
        """
        if not 0 <= step < len(self.steps):
            raise IndexError(f"no step {step}")
        steps = self._ticked.setdefault(key, set())
        if ticked:
            steps.add(step)
        else:
            steps.discard(step)
        if not steps:
            del self._ticked[key]

    def advance(self, key: Path) -> bool:
        """Tick the current step; return whether one was ticked.

        False at the save step, which a save ticks, and once every step is.
        """
        step = self.current(key)
        if step >= self.save_step:
            return False
        self.set_ticked(key, step, True)
        return True

    def retreat(self, key: Path) -> bool:
        """Untick the step before the current one; return whether one was."""
        step = self.current(key)
        if step == 0:
            return False
        self.set_ticked(key, step - 1, False)
        return True

    def is_started(self, key: Path) -> bool:
        """Return whether any step is ticked for the label at *key*."""
        return key in self._ticked

    def is_checked(self, key: Path) -> bool:
        """Return whether every step before saving is ticked for *key*."""
        return all(step in self._ticked.get(key, ()) for step in range(self.save_step))

    def mark_saved(self, keys: Iterable[Path]) -> None:
        """Tick the save step for each of *keys*, saved and marked corrected."""
        for key in keys:
            self.set_ticked(key, self.save_step, True)

    def forget(self, keys: Iterable[Path]) -> None:
        """Untick every step for each of *keys*."""
        for key in keys:
            self._ticked.pop(key, None)
