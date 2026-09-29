"""Dividing a training export into the sets a model is trained and scored on.

Pure arithmetic over items that already carry a group: nothing here builds a
row, reads a label or touches the filesystem, so a split can be tested on
strings. **A split never cuts a group**, and **one seed gives one split** -- an
export's manifest records the seed so the same files can be made again, which
holds only while this arithmetic does not change.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

#: An item being split, whatever its group is.
T = TypeVar("T")

#: What share of the exportable pairs goes to training where nobody says.
#: One spelling for the window, the command line and these functions, which
#: had four between them and could have drifted apart silently.
DEFAULT_TRAIN_RATIO = 0.8


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
