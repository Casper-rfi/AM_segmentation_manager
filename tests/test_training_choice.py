"""Tests for the epoch a person marks as a run's best: one file, one reader, two changes."""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.training.choice import (
    CHOICE_FILENAME,
    choose_best,
    clear_chosen_best,
    read_chosen_best,
)
from am_segmentation_manager.training.runs import read_training_run
from test_training_runs import FIXTURE


def test_a_run_nobody_chose_for_has_no_choice(tmp_path: Path):
    assert read_chosen_best(tmp_path) is None


def test_a_choice_is_read_back_with_who_made_it_and_when(tmp_path: Path):
    chosen = choose_best(tmp_path, read_training_run(FIXTURE), 3, user="Casper")

    assert read_chosen_best(tmp_path) == chosen
    assert (chosen.epoch, chosen.chosen_by) == (3, "Casper")
    assert chosen.chosen_at.endswith("Z")


def test_a_later_choice_replaces_the_earlier_one(tmp_path: Path):
    run = read_training_run(FIXTURE)
    choose_best(tmp_path, run, 3, user="Casper")

    choose_best(tmp_path, run, 1, user="Casper")

    assert read_chosen_best(tmp_path).epoch == 1


def test_an_epoch_the_run_did_not_validate_is_refused_and_nothing_is_written(tmp_path: Path):
    with pytest.raises(ValueError, match="Epoch 9 was not validated"):
        choose_best(tmp_path, read_training_run(FIXTURE), 9)

    assert not (tmp_path / CHOICE_FILENAME).exists()


def test_clearing_says_whether_there_was_a_choice(tmp_path: Path):
    choose_best(tmp_path, read_training_run(FIXTURE), 2)

    assert clear_chosen_best(tmp_path) is True
    assert clear_chosen_best(tmp_path) is False
    assert read_chosen_best(tmp_path) is None


def test_choosing_touches_nothing_the_run_wrote(tmp_path: Path):
    import shutil

    copy = tmp_path / FIXTURE.name
    shutil.copytree(FIXTURE, copy)
    before = {path.name: path.read_bytes() for path in copy.iterdir()}

    choose_best(copy, read_training_run(copy), 2)

    after = {path.name: path.read_bytes() for path in copy.iterdir()}
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == {CHOICE_FILENAME}


@pytest.mark.parametrize("body", ["version: 1\n", "version: 1\nepoch: best\n", "version: 99\nepoch: 2\n"])
def test_a_choice_that_cannot_be_understood_is_said_rather_than_read_as_none(tmp_path: Path, body: str):
    (tmp_path / CHOICE_FILENAME).write_text(body, encoding="utf-8")

    with pytest.raises(ValueError):
        read_chosen_best(tmp_path)
