"""The arithmetic that divides a training export, held to the splits it has made.

A manifest records the seed an export was split with, so that the same files can
be made again. That promise is kept by the arithmetic, not by the seed: these
tests fix which group lands where for a known input, and fail if a change to
:func:`split_grouped` would make an old manifest's seed give a different split.
"""
from __future__ import annotations

import pytest

from am_segmentation_manager.operations.training_split import split_grouped

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
