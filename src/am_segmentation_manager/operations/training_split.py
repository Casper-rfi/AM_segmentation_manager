"""Dividing a training export into the sets a model is trained and scored on.

What a split may keep together, and the arithmetic over items that already carry
a group -- one train/validation split, or K folds for cross-validation. Nothing
here builds a row, reads a label or touches the filesystem, so a split can be
tested on strings. **A split never cuts a group**, and **one seed gives one
split** -- an export's manifest records the seed so the same files can be made
again, which holds only while this arithmetic does not.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Mapping, Sequence
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


@dataclass(frozen=True)
class Balance:
    """What the validation side holds a share of, by name and by record.

    One value for the reason :class:`Grouping` is one: the properties a
    manifest records cannot differ from those the rows were balanced by.

    Attributes:
        names: The properties balanced on, as an export's manifest lists them.
        key: Returns a record's value of each, which is its stratum.
    """

    names: tuple[str, ...]
    key: Callable[[ImageRecord], Hashable]


def split_balanced(
    items: Sequence[T],
    key: Callable[[T], str],
    stratum: Callable[[T], Hashable],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
) -> tuple[list[T], list[T]]:
    """Split *items* so each stratum gives the validation side its own share.

    A group belongs to the stratum most of its items are in, and each stratum
    is split by :func:`split_grouped`, in the order the strata are first met.
    **A stratum of two or more groups always validates on one**, which the
    ratio alone would deny a small one; a stratum of one group trains.

    Returns:
        ``(training, validation)``.
    """
    of = _strata(items, key, stratum)
    strata: dict[Hashable, list[T]] = {}
    for value in of.values():
        strata.setdefault(value, [])
    for item in items:
        strata[of[key(item)]].append(item)

    training: list[T] = []
    validation: list[T] = []
    for held in strata.values():
        train, validate = split_grouped(held, key, train_ratio, seed)
        if not validate and 0.0 < train_ratio < 1.0 and len({key(item) for item in held}) > 1:
            last = key(train[-1])
            validate = [item for item in train if key(item) == last]
            train = [item for item in train if key(item) != last]
        training.extend(train)
        validation.extend(validate)
    return training, validation


def _strata(
    items: Sequence[T], key: Callable[[T], str], stratum: Callable[[T], Hashable]
) -> dict[str, Hashable]:
    """Return the stratum of each group: the one most of its items are in."""
    counts: dict[str, dict[Hashable, int]] = {}
    for item in items:
        within = counts.setdefault(key(item), {})
        within[stratum(item)] = within.get(stratum(item), 0) + 1
    # The first of the commonest, so a tie is decided by the order the items
    # came in and one seed gives one split.
    return {
        group: max(within, key=lambda value: within[value]) for group, within in counts.items()
    }


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


def split_validating(
    items: Sequence[T],
    key: Callable[[T], str],
    validates: Callable[[T], bool],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
    stratum: Callable[[T], Hashable] | None = None,
) -> tuple[list[T], list[T], list[T]]:
    """Split *items* so that only those *validates* accepts are scored on.

    The items that may validate are split as :func:`split_grouped` splits them,
    so the validation side is the one a split of those alone gives. Every other
    item then trains, **unless its group is on the validation side**: training
    on it would score a model on a group it has seen, so it is left out.

    Args:
        stratum: What to balance the validation side by, as
            :func:`split_balanced` does; unbalanced where omitted.

    Returns:
        ``(training, validation, left_out)``.
    """
    eligible = [item for item in items if validates(item)]
    if stratum is None:
        training, validation = split_grouped(eligible, key, train_ratio, seed)
    else:
        training, validation = split_balanced(eligible, key, stratum, train_ratio, seed)
    validated = {key(item) for item in validation}
    left_out: list[T] = []
    for item in items:
        if not validates(item):
            (left_out if key(item) in validated else training).append(item)
    return training, validation, left_out


#: The two sides of a split, as an export's images file names them, and what a
#: row on neither is called.
TRAINING, VALIDATION, LEFT_OUT = "train", "validation", "left out"


class DividedGroup(ValueError):
    """A group this split keeps whole was on both sides of the split it is pinned to."""


def split_pinned(
    items: Sequence[T],
    key: Callable[[T], str],
    pinned: Callable[[T], str | None],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
    validates: Callable[[T], bool] | None = None,
    stratum: Callable[[T], Hashable] | None = None,
) -> tuple[list[T], list[T], list[T]]:
    """Split *items*, keeping every group on the side an earlier split put it.

    A group any of whose items was validated on validates again, with whatever
    of it *validates* accepts now; one that trained trains again. The groups the
    earlier split never saw are split among themselves, as
    :func:`split_validating` splits them -- or, **balanced, by where each
    stratum stands with the pinned groups counted**: a new group validates
    where that leaves its stratum's validation share nearer the target, so a
    kind arriving one group at a time is not sent to training every time.

    Args:
        pinned: Returns :data:`TRAINING` or :data:`VALIDATION` for an item the
            earlier split held, None for one it did not.
        validates: Whether an item may be scored on; every item where omitted.
        stratum: What to balance by, the pinned groups counting towards each
            stratum's standing.

    Returns:
        ``(training, validation, left_out)``.

    Raises:
        DividedGroup: If the earlier split put one group on both sides, as one
            keeping a finer level together can.
    """
    accepts = validates or (lambda item: True)
    sides: dict[str, set[str]] = {}
    for item in items:
        side = pinned(item)
        if side is not None:
            sides.setdefault(key(item), set()).add(side)
    divided = sorted(group for group, found in sides.items() if len(found) > 1)
    if divided:
        raise DividedGroup(
            f"{len(divided)} group(s) were on both sides of the earlier split, "
            f"starting with {divided[0]}"
        )

    if stratum is None:
        training, validation, left_out = split_validating(
            [item for item in items if key(item) not in sides], key, accepts, train_ratio, seed
        )
        validating = {group for group, side in sides.items() if side == {VALIDATION}}
    else:
        training, validation, left_out = [], [], []
        validating = {group for group, side in sides.items() if side == {VALIDATION}} | (
            _new_groups_to_validate(
                [item for item in items if accepts(item)], key, stratum, sides, train_ratio, seed
            )
        )
    for item in items:
        group = key(item)
        if group in validating:
            (validation if accepts(item) else left_out).append(item)
        elif stratum is not None or group in sides:
            training.append(item)
    return training, validation, left_out


def _new_groups_to_validate(
    eligible: Sequence[T],
    key: Callable[[T], str],
    stratum: Callable[[T], Hashable],
    sides: Mapping[str, set[str]],
    train_ratio: float,
    seed: int | None,
) -> set[str]:
    """Return the groups no earlier split held that should validate, stratum by stratum.

    Each stratum's target is its share of *all* its items, pinned ones
    included; new groups are dealt in shuffled order, each to validation where
    that is the nearer side of the target. A stratum of two or more groups that
    would validate on none gives up its last new one, as :func:`split_balanced` does.
    """
    of = _strata(eligible, key, stratum)
    size: dict[str, int] = {}
    for item in eligible:
        size[key(item)] = size.get(key(item), 0) + 1
    total: dict[Hashable, int] = {}
    validated: dict[Hashable, int] = {}
    groups: dict[Hashable, int] = {}
    for group, value in of.items():
        total[value] = total.get(value, 0) + size[group]
        groups[value] = groups.get(value, 0) + 1
        if sides.get(group) == {VALIDATION}:
            validated[value] = validated.get(value, 0) + size[group]

    new = [group for group in of if group not in sides]
    if seed is not None:
        random.Random(seed).shuffle(new)
    chosen: set[str] = set()
    if train_ratio >= 1.0:
        return chosen
    if train_ratio <= 0.0:
        return set(new)
    for group in new:
        value = of[group]
        if validated.get(value, 0) + size[group] / 2 <= total[value] * (1 - train_ratio):
            chosen.add(group)
            validated[value] = validated.get(value, 0) + size[group]
    for value, count in groups.items():
        last = [group for group in new if of[group] == value][-1:]
        if count > 1 and not validated.get(value) and last:
            chosen.add(last[0])
    return chosen


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


def assign_folds(
    items: Sequence[T],
    key: Callable[[T], str],
    folds: int,
    seed: int | None = 42,
) -> list[list[T]]:
    """Divide *items* into *folds* validation sets, keeping a group together.

    Groups are shuffled, then each goes to the fold holding fewest items so far,
    so the folds end at most one group apart in size and a different seed still
    gives a different partition.

    Returns:
        One list per fold, the items that fold validates on. A fold trains on
        every other fold's; see :func:`fold_training`.

    Raises:
        ValueError: If *folds* is below 2, or above the number of groups.
    """
    if folds < 2:
        raise ValueError(f"Cross-validation needs at least 2 folds, not {folds}")
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)
    if len(grouped) < folds:
        raise ValueError(
            f"{len(grouped)} group(s) cannot fill {folds} folds without dividing one"
        )

    groups = list(grouped.values())
    if seed is not None:
        random.Random(seed).shuffle(groups)

    assigned: list[list] = [[] for _ in range(folds)]
    for group in groups:
        # The first of the smallest, so a tie is broken by fold number and one
        # seed always gives one partition.
        smallest = min(range(folds), key=lambda index: len(assigned[index]))
        assigned[smallest].extend(group)
    return assigned


def fold_training(validation_sets: Sequence[Sequence[T]], fold: int) -> list[T]:
    """Return what fold *fold* (from 0) trains on: every other fold's validation set, in order."""
    return [
        item
        for index, validation in enumerate(validation_sets)
        if index != fold
        for item in validation
    ]
