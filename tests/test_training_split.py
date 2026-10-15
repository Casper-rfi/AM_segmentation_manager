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
    DividedGroup,
    assign_folds,
    fold_training,
    split_balanced,
    split_grouped,
    split_pinned,
    split_validating,
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


def _pairs(conditions: int) -> list[tuple[str, str]]:
    """Ten labels, each shared by *conditions* images of one scene."""
    return [(f"sem_{c}/img_{n}.tif", f"labels/img_{n}.tif")
            for n in range(10) for c in range(conditions)]


def _label(pair: tuple[str, str]) -> str:
    return pair[1]


def test_images_sharing_a_label_stay_on_the_same_side():
    """Splitting them would put near-identical images in train and validation."""
    train, val = split_grouped(_pairs(3), _label, train_ratio=0.8, seed=42)

    assert {_label(pair) for pair in train}.isdisjoint(_label(pair) for pair in val)


def test_every_item_is_placed_exactly_once():
    pairs = _pairs(3)

    train, val = split_grouped(pairs, _label, train_ratio=0.8, seed=42)

    assert sorted(train + val) == sorted(pairs)


@pytest.mark.parametrize(("conditions", "lowest", "highest"), [(1, 0.8, 0.8), (3, 0.7, 0.9)])
def test_the_ratio_is_kept_as_closely_as_whole_groups_allow(conditions, lowest, highest):
    pairs = _pairs(conditions)

    train, _ = split_grouped(pairs, _label, train_ratio=0.8, seed=42)

    assert lowest <= len(train) / len(pairs) <= highest


@pytest.mark.parametrize(("ratio", "kept_for_training"), [(1.0, True), (0.0, False)])
def test_a_ratio_of_one_or_zero_keeps_everything_on_one_side(ratio, kept_for_training):
    pairs = _pairs(3)

    train, val = split_grouped(pairs, _label, train_ratio=ratio)

    assert (train, val) == ((pairs, []) if kept_for_training else ([], pairs))


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


def _proofread(item: str) -> bool:
    """Whether an item may be validated on: the even-numbered of each group, here."""
    return int(item.split("-")[1]) % 2 == 0


def test_adding_items_that_cannot_validate_leaves_the_validation_side_as_it_was():
    proofread = [item for item in ITEMS if _proofread(item)]
    _, alone = split_grouped(proofread, _group, train_ratio=0.8, seed=42)

    _, validation, _ = split_validating(ITEMS, _group, _proofread, train_ratio=0.8, seed=42)

    assert validation == alone
    assert validation, "or the test compares nothing with nothing"


def test_an_item_that_cannot_validate_never_trains_beside_its_group_s_validation():
    training, validation, left_out = split_validating(
        ITEMS, _group, _proofread, train_ratio=0.8, seed=42
    )

    validated = {_group(item) for item in validation}
    assert not validated & {_group(item) for item in training}
    assert left_out == [
        item for item in ITEMS if _group(item) in validated and not _proofread(item)
    ]
    assert sorted(training + validation + left_out) == sorted(ITEMS)


def test_a_group_with_nothing_to_validate_on_always_trains():
    items = [*ITEMS, "g99-1", "g99-3"]

    for seed in range(20):
        training, _, _ = split_validating(items, _group, _proofread, train_ratio=0.5, seed=seed)
        assert {"g99-1", "g99-3"} <= set(training)


def _pin(training: list[str], validation: list[str]):
    """Return what says which side an earlier split put an item on."""
    sides = {**dict.fromkeys(training, "train"), **dict.fromkeys(validation, "validation")}
    return sides.get


def test_no_group_changes_side_when_a_split_is_kept_to():
    earlier_train, earlier_val = split_grouped(ITEMS, _group, train_ratio=0.8, seed=42)
    grown = [*ITEMS, *(f"n{group:02d}-{i}" for group in range(10) for i in range(2))]

    for seed in (1, 2, 3):
        training, validation, left_out = split_pinned(
            grown, _group, _pin(earlier_train, earlier_val), train_ratio=0.5, seed=seed
        )

        assert set(earlier_train) <= set(training)
        assert set(earlier_val) <= set(validation)
        assert not left_out
        new = {_group(item) for item in grown if item.startswith("n")}
        assert new & {_group(item) for item in training}, "the new groups fill both sides"
        assert new & {_group(item) for item in validation}


def test_what_is_proofread_since_in_a_validated_group_validates_too():
    proofread = [item for item in ITEMS if _proofread(item)]
    earlier_train, earlier_val, _ = split_validating(ITEMS, _group, _proofread, 0.8, 42)

    training, validation, left_out = split_pinned(
        ITEMS, _group, _pin(earlier_train, earlier_val), 0.8, 7, validates=lambda item: True
    )

    validated = {_group(item) for item in earlier_val}
    assert validation and {_group(item) for item in validation} == validated
    assert len(validation) > len(earlier_val), "its groups hold more than was proofread then"
    assert not left_out
    assert not validated & {_group(item) for item in training}
    assert proofread


def test_a_group_the_earlier_split_divided_is_refused():
    with pytest.raises(DividedGroup, match="g01"):
        split_pinned(ITEMS, _group, _pin(["g01-0"], ["g01-1"]))


#: Forty common groups and three rare ones, two items each: what a random
#: split by group leaves the rare kind out of more often than not.
KINDS = {f"c{n:02d}": "common" for n in range(40)} | {f"r{n}": "rare" for n in range(3)}
KIND_ITEMS = [f"{group}-{i}" for group in KINDS for i in range(2)]


def _kind(item: str) -> str:
    return KINDS[_group(item)]


def test_every_kind_two_groups_hold_validates_whatever_the_seed():
    for seed in range(30):
        training, validation = split_balanced(KIND_ITEMS, _group, _kind, 0.8, seed)

        assert {_kind(item) for item in validation} == {"common", "rare"}
        assert not {_group(item) for item in training} & {_group(item) for item in validation}
        assert sorted(training + validation) == sorted(KIND_ITEMS)
        common = [item for item in validation if _kind(item) == "common"]
        assert len(common) == 16, "a fifth of the eighty common items"


def test_a_kind_one_group_holds_trains():
    items = [*KIND_ITEMS, "only-0", "only-1"]
    kinds = {**KINDS, "only": "single"}

    training, validation = split_balanced(
        items, _group, lambda item: kinds[_group(item)], 0.8, 42
    )

    assert {"only-0", "only-1"} <= set(training)
    assert "single" not in {kinds[_group(item)] for item in validation}


def test_a_group_of_two_kinds_counts_as_the_one_most_of_it_is():
    mixed = {"m-0": "rare", "m-1": "common", "m-2": "common"}
    items = [*KIND_ITEMS, *mixed]

    training, validation = split_balanced(
        items, _group, lambda item: mixed.get(item) or _kind(item), 0.8, 42
    )

    side = training if "m-0" in training else validation
    assert set(mixed) <= set(side), "a group is never divided to balance it"


def test_balancing_applies_to_what_may_validate_and_to_nothing_else():
    training, validation, left_out = split_validating(
        KIND_ITEMS, _group, _proofread, 0.8, 42, stratum=_kind
    )

    assert {_kind(item) for item in validation} == {"common", "rare"}
    assert all(_proofread(item) for item in validation)
    assert {_group(item) for item in left_out} == {_group(item) for item in validation}


def test_a_rare_kind_arriving_one_group_at_a_time_reaches_validation():
    """Split among themselves, a kind's only new group trains every round."""
    held = [item for item in KIND_ITEMS if _kind(item) == "common"]
    training, validation = split_balanced(held, _group, _kind, 0.8, 42)
    arrived: list[str] = []
    kinds = dict(KINDS)

    sides_seen = []
    for round_number in range(6):
        group = f"new{round_number}"
        kinds[group] = "rare"
        arrived += [f"{group}-0", f"{group}-1"]
        items = [*held, *arrived]
        training, validation, _ = split_pinned(
            items, _group, _pin(training, validation), 0.8, round_number,
            stratum=lambda item: kinds[_group(item)],
        )
        sides_seen.append("validation" if f"{group}-0" in validation else "train")
        assert sorted(training + validation) == sorted(items)

    assert sides_seen.count("validation") in (1, 2), sides_seen
    assert sides_seen[1] == "validation", "the second of a kind, where the first trained"


def test_a_pinned_group_stays_where_it_was_however_its_kind_stands():
    earlier_train, earlier_val = split_grouped(KIND_ITEMS, _group, 0.8, 42)

    training, validation, _ = split_pinned(
        KIND_ITEMS, _group, _pin(earlier_train, earlier_val), 0.8, 7, stratum=_kind
    )

    assert sorted(training) == sorted(earlier_train)
    assert sorted(validation) == sorted(earlier_val)


def test_new_groups_fill_the_side_their_kind_is_short_on():
    """A kind wholly in training before: its new groups validate until it has its share."""
    common = [item for item in KIND_ITEMS if _kind(item) == "common"]
    new = [f"n{n:02d}-{i}" for n in range(10) for i in range(2)]

    training, validation, _ = split_pinned(
        [*common, *new], _group, _pin(common, []), 0.8, 42, stratum=lambda item: "one kind"
    )

    assert set(common) <= set(training)
    assert len(validation) == 20, "all ten new groups, a fifth of the hundred items"
