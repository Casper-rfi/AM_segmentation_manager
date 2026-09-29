"""The arithmetic that divides a training export, held to the splits it has made.

A manifest records the seed an export was split with, so that the same files can
be made again. That promise is kept by the arithmetic, not by the seed: these
tests fix which group lands where for a known input, and fail if a change to
:func:`split_grouped` or :func:`assign_folds` would make an old manifest's seed
give a different split.
"""
from __future__ import annotations

import pytest

from am_segmentation_manager.operations.training_split import (
    assign_folds,
    fold_training,
    split_grouped,
)

#: Groups of uneven size, so the fill-to-nearest rule decides where each lands.
SIZES = (1, 4, 2, 1, 3, 1, 5, 2, 1, 1, 2, 3)
ITEMS = [f"g{group:02d}-{i}" for group, size in enumerate(SIZES) for i in range(size)]


def _group(item: str) -> str:
    return item.split("-")[0]


def _groups_in_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(_group(item) for item in items))


@pytest.mark.parametrize(
    ("seed", "training", "validation"),
    [
        (
            42,
            ["g07", "g05", "g02", "g08", "g09", "g06", "g11", "g03", "g04", "g00"],
            ["g01", "g10"],
        ),
        (
            7,
            ["g07", "g11", "g03", "g10", "g08", "g04", "g09", "g01", "g00", "g06"],
            ["g02", "g05"],
        ),
    ],
)
def test_a_recorded_seed_still_gives_the_split_it_gave(seed, training, validation):
    train, val = split_grouped(ITEMS, _group, train_ratio=0.8, seed=seed)

    assert _groups_in_order(train) == training
    assert _groups_in_order(val) == validation


def test_a_group_keeps_its_items_in_the_order_they_came():
    train, val = split_grouped(ITEMS, _group, train_ratio=0.8, seed=42)

    for side in (train, val):
        for group in _groups_in_order(side):
            assert [item for item in side if _group(item) == group] == [
                item for item in ITEMS if _group(item) == group
            ]


def test_no_seed_leaves_the_groups_in_the_order_they_came():
    train, val = split_grouped(ITEMS, _group, train_ratio=0.5, seed=None)

    for side in (train, val):
        order = _groups_in_order(side)
        assert order == sorted(order)
    assert sorted(train + val) == ITEMS


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("seed", "folds"),
    [
        (42, [["g07", "g09", "g03", "g04", "g10"], ["g05", "g08", "g06"],
              ["g02", "g11", "g00", "g01"]]),
        (7, [["g07", "g08", "g04", "g02"], ["g11", "g09", "g00", "g06"],
             ["g03", "g10", "g01", "g05"]]),
    ],
)
def test_a_recorded_seed_still_gives_the_folds_it_gave(seed, folds):
    assigned = assign_folds(ITEMS, _group, 3, seed=seed)

    assert [_groups_in_order(fold) for fold in assigned] == folds


def test_every_item_validates_in_exactly_one_fold():
    assigned = assign_folds(ITEMS, _group, 4, seed=42)

    assert sorted(item for fold in assigned for item in fold) == sorted(ITEMS)


def test_no_group_is_in_two_folds():
    assigned = assign_folds(ITEMS, _group, 4, seed=42)

    seen = [set(_groups_in_order(fold)) for fold in assigned]
    assert sum(len(groups) for groups in seen) == len(set().union(*seen)) == len(SIZES)


@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_folds_end_within_one_group_of_each_other(seed):
    sizes = [len(fold) for fold in assign_folds(ITEMS, _group, 4, seed=seed)]

    assert max(sizes) - min(sizes) <= max(SIZES)


def test_a_fold_trains_on_every_other_fold():
    assigned = assign_folds(ITEMS, _group, 3, seed=42)

    for index, validation in enumerate(assigned):
        training = fold_training(assigned, index)
        assert sorted(training + validation) == sorted(ITEMS)
        assert not set(training) & set(validation)


def test_another_seed_can_give_other_folds():
    assert assign_folds(ITEMS, _group, 3, seed=42) == assign_folds(ITEMS, _group, 3, seed=42)
    assert assign_folds(ITEMS, _group, 3, seed=42) != assign_folds(ITEMS, _group, 3, seed=7)


@pytest.mark.parametrize("folds", [0, 1, len(SIZES) + 1])
def test_folds_that_cannot_be_filled_are_refused(folds):
    with pytest.raises(ValueError):
        assign_folds(ITEMS, _group, folds)
