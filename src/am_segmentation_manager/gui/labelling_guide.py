"""The labelling guide: the steps, and how far each label has got through them.

An aid against forgetting a step, held in memory only: nothing here is written
to disk or recorded, and closing the window forgets it. Progress is kept per
label, keyed by the path the label is written to, so imaging conditions sharing
a label share their progress as they share the label. The last step is saving,
which ticks itself when the label is saved and marked corrected. A step's
example is a crop of a real frame and its label before and after the step,
shipped in ``data/labelling_guide/``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.image import read_array

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

#: Where the examples are kept: a folder for each, named by :attr:`Step.example`.
EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "labelling_guide"

#: The files of an example, cropped alike: the image, and its label before and
#: after the step, as class indices of the packaged segmentation config.
EXAMPLE_FILES = ("image.png", "before.png", "after.png")


@dataclass(frozen=True)
class Step:
    """One step of proofreading a label.

    Attributes:
        title: What the step is, as the labelling guide lists it.
        instructions: What to do in it, a line each.
        explanation: How to do it, for the step's help.
        example: The folder under :data:`EXAMPLES` showing it, or empty.
    """

    title: str
    instructions: tuple[str, ...]
    explanation: str
    example: str = ""


@dataclass(frozen=True)
class Example:
    """A crop of a frame, and its label before and after one step."""

    image: np.ndarray
    before: np.ndarray
    after: np.ndarray


def read_example(name: str, root: Path = EXAMPLES) -> Example | None:
    """Return the example called *name*, or None where it is not there.

    Raises:
        ValueError: If its three files are not all the same size.
    """
    folder = root / name
    if not name or not all((folder / file).is_file() for file in EXAMPLE_FILES):
        return None
    image, before, after = (read_array(folder / file) for file in EXAMPLE_FILES)
    if not image.shape[:2] == before.shape[:2] == after.shape[:2]:
        raise ValueError(f"the files of example {name!r} are not cropped alike")
    return Example(image=image, before=before, after=after)


#: The steps, in the order a label is proofread. The last is saving.
STEPS: tuple[Step, ...] = (
    Step(
        "Remove false positives outside the lamella",
        ("Use the speck finder (X) to make them easier to see.",),
        "Remove any label not on the lamella, GIS layer or the vacuum above or "
        "below it. Nearby correct labels, such as the vacuum in this example, can "
        "be locked (Shift+T), and the false positive labels can be removed with a "
        "large brush set to background. Use the speck finder (X) to make it easier "
        "to see them.",
        example="false_positives_outside",
    ),
    Step(
        "Correct the sides of the lamella",
        (
            "Add lamella label where it is missing.",
            "Remove lamella label where it does not belong.",
        ),
        "Follow each side of the lamella from top to bottom, and paint lamella "
        "where the label stops short of the edge, or remove it where it runs "
        "past it. Nearby classes can be locked (the padlock icon) so they are "
        "not painted over.",
        example="lamella_sides",
    ),
    Step(
        "Correct cracks and holes",
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
        example="cracks_and_holes",
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
        "Follow the edge between the GIS layer and the lamella across its width, "
        "and paint lamella where GIS covers the lamella. Do the same for GIS "
        "where GIS is covered with lamella. Hide the overlay (V) when painting, "
        "and use the brush outline to see what will be painted. In some cases, "
        "it can be quicker and more accurate to erase the whole boundary area, "
        "draw the boundary with GIS, and use the fill tool (F) to fill in the "
        "remaining areas after enclosing them. Correct the lamella sides to make "
        "sure they cover the full width of the lamella.",
        example="gis_lamella_boundary",
    ),
    Step(
        "Correct the GIS-vacuum boundary",
        (
            "Add missing GIS label.",
            "Change GIS label on vacuum to vacuum.",
        ),
        "Follow the outer edge of the GIS layer where it meets the vacuum, along "
        "the full width. First add GIS where missed, in particular on the sides. "
        "Change GIS to vacuum where GIS label reaches into the vacuum. Hide the "
        "labels (V) when painting.",
        example="gis_vacuum_boundary",
    ),
    Step(
        "Correct the lamella-vacuum boundary",
        (
            "Add missing lamella label.",
            "Change lamella label on vacuum to vacuum.",
        ),
        "Follow the top edge of the lamella where it meets the vacuum, along the "
        "full width. First add lamella where missed. Change lamella to vacuum "
        "where lamella label reaches into the vacuum. Hide the labels (V) when "
        "painting.",
        example="lamella_vacuum_boundary",
    ),
    Step(
        "Correct the vacuum-background boundaries",
        (
            "Add missing vacuum label.",
            "Remove vacuum label on other material.",
            "Check that the top and bottom edges are completely filled.",
        ),
        "Follow the edges of the vacuum and paint vacuum where it was missed, and "
        "background where vacuum label covers other materials. The vacuum usually "
        "extends to the top and bottom of the image. If this is the case, make "
        "sure the label covers up to the top and bottom as well. Use the speck "
        "finder (X) to make this easier to see.",
        example="vacuum_background_boundary",
    ),
    Step(
        "Find stray pixels with the speck finder",
        ("Turn on the speck finder (X), and step through the specks with N.",),
        "Remove any stray pixels and pieces left behind, which can be found using "
        "the speck finder (X). N moves to the next speck, to paint over it. Some "
        "objects found by the speck finder are real (e.g. small holes), so check "
        "before removing any.",
        example="stray_pixels",
    ),
    Step(
        "Save and mark corrected",
        ("Press C or Ctrl+S, and choose Save and mark corrected.",),
        "Save the label to write it to disk. Marking a label as corrected keeps "
        "every automated operation from overwriting it. To mark labels as "
        "corrected when saving, enable editing first on the Curate tab (top "
        "right). This step ticks itself once the label is saved and marked "
        "corrected.",
    ),
)

class LabellingGuide:
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
