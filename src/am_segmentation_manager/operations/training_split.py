"""Dividing a training export into the sets a model is trained and scored on.

What a split may keep together, and the arithmetic over items that already
carry a group: nothing here builds a row, reads a label or touches the
filesystem, so a split can be tested on strings. **A split never cuts a group**,
and **one seed gives one split** -- an export's manifest records the seed so the
same files can be made again, which holds only while this arithmetic does not.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from ..models import ImageRecord

#: An item being split, whatever its group is.
T = TypeVar("T")

#: What share of the exportable pairs goes to training where nobody says.
#: One spelling for the window, the command line and these functions, which
#: had four between them and could have drifted apart silently.
DEFAULT_TRAIN_RATIO = 0.8


#: What a split can keep on one side, as ``(key, caption)``, finest first. The
#: finest holds together the imaging conditions and both beams of one
#: acquisition, which no split may divide; the coarser two keep a whole lamella,
#: or a whole Autolamella project directory, out of the other side.
KEEP_TOGETHER: tuple[tuple[str, str], ...] = (
    ("acquisition", "Acquisition"),
    ("lamella", "Lamella"),
    ("project", "Project"),
)

#: The level whose groups a row builder knows without being told.
ACQUISITION = "acquisition"

#: What the window and the command line keep together where nobody says.
#: Successive frames of one lamella are near-duplicates, so a finer split
#: scores a model on lamellae it was trained on.
DEFAULT_KEEP_TOGETHER = "lamella"


@dataclass(frozen=True)
class Grouping:
    """What a split keeps on one side, by name and by record.

    One value rather than a name and a function passed apart, so the level a
    manifest records cannot differ from the one the rows were grouped by.

    Attributes:
        level: A key of :data:`KEEP_TOGETHER`.
        key: Returns the group a record belongs to at *level*; None at
            :data:`ACQUISITION`, which the row builder already knows.

    Raises:
        ValueError: If *level* is unknown, or *key* is missing or given where
            the level says otherwise.
    """

    level: str = ACQUISITION
    key: Callable[[ImageRecord], Hashable] | None = None

    def __post_init__(self) -> None:
        if self.level not in dict(KEEP_TOGETHER):
            raise ValueError(f"Unknown keep-together level: {self.level!r}")
        if (self.key is None) != (self.level == ACQUISITION):
            raise ValueError(
                f"A {self.level!r} grouping "
                f"{'takes no' if self.level == ACQUISITION else 'needs a'} key"
            )


def split_grouped(
    items: Sequence[T],
    key: Callable[[T], str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
) -> tuple[list[T], list[T]]:
    """Split *items* into training and validation, keeping a group together.

    The general form of :func:`split_pairs`: what a group *is* differs between
    a single-beam export, where it is the label, and a paired one, where it is
    the acquisition -- but the arithmetic of filling to the nearest side of the
    target without ever cutting a group is the same.

    Args:
        items: What to split.
        key: Returns the group an item belongs to.
        train_ratio: Share of items assigned to training.
        seed: Seed for a reproducible shuffle, or None to leave the order.

    Returns:
        ``(training, validation)``.
    """
    if train_ratio >= 1.0:
        return list(items), []
    if train_ratio <= 0.0:
        return [], list(items)

    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)

    groups = list(grouped.values())
    if seed is not None:
        random.Random(seed).shuffle(groups)

    target = len(items) * train_ratio
    training: list = []
    validation: list = []
    for group in groups:
        # Whole groups only. Filling to the nearest side of the target keeps
        # the ratio as close as the grouping allows without ever splitting one.
        if len(training) + len(group) / 2 <= target:
            training.extend(group)
        else:
            validation.extend(group)
    return training, validation


def split_pairs(
    pairs: list[tuple[str, str]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split image/label pairs into training and validation sets.

    Pairs are grouped by **label** before splitting, so every image sharing a
    label lands on the same side. Several imaging conditions of one scene share
    one label; splitting them across the two sets would put near-identical
    images in both, and the validation score would then measure how well the
    model recognises images it has effectively already seen.

    With one image per label -- every project that has no imaging-condition
    variants -- each group holds a single pair and this behaves exactly as a
    plain shuffle-and-split.

    Reports nothing: the export view runs this on every filter change, and a log
    line per keystroke saying how a split came out reads as an export happening.
    :func:`~am_segmentation_manager.operations.csv.export_training_csv` reports
    it, because it is the one that writes.

    Args:
        pairs: ``(image, label)`` relative paths.
        train_ratio: Share of pairs assigned to training.
        seed: Seed for a reproducible shuffle, or None to leave the order.

    Returns:
        ``(train_pairs, val_pairs)``.
    """
    return split_grouped(pairs, lambda pair: pair[1], train_ratio, seed)
