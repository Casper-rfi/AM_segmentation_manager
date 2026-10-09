"""Tests for which runs a folder holds and what each is called, by the window and the command line alike."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.cli import main
from am_segmentation_manager.gui.training_controller import TrainingController
from am_segmentation_manager.operations.training_choice import (
    CHOICE_FILENAME,
    chosen_best_or_none,
)
from am_segmentation_manager.operations.training_run import RUN_RECORD_FILENAME
from am_segmentation_manager.training_compare import EVERY_SET, offered_sets
from am_segmentation_manager.training_listing import list_runs
from test_training_process import _prepare
from test_training_runs import _epoch, _run

SCORED = {"train": [_epoch(1, 0.1, 0.1)], "val": [_epoch(1, 0.5, 0.5)]}


def _names(root: Path) -> list[str]:
    return sorted(run.name for run in list_runs(root))


def test_two_runs_written_into_one_folder_are_named_apart(tmp_path: Path):
    _run(tmp_path, "training_261001_120000")
    _run(tmp_path, "training_261002_120000")
    _run(tmp_path / "shared", "training_261003_120000")
    _run(tmp_path / "shared", "training_261004_120000")

    assert _names(tmp_path) == [
        "shared/training_261003_120000", "shared/training_261004_120000",
        "training_261001_120000", "training_261002_120000",
    ]


def test_a_run_in_a_folder_of_its_own_is_called_by_that_folder(tmp_path: Path):
    written = _run(tmp_path / "round" / "first")

    assert [(run.name, run.home, run.written) for run in list_runs(tmp_path)] == [
        ("round/first", written, written)
    ]
    assert _names(written) == ["first"]


def test_a_prepared_run_is_listed_once_and_not_again_as_what_it_wrote(tmp_path: Path):
    home = _prepare(tmp_path)
    assert [(run.name, run.written) for run in list_runs(tmp_path / "runs")] == [("round01", None)]

    written = _run(home)

    assert list_runs(tmp_path / "runs") == list_runs(home)
    assert [(run.name, run.home, run.written) for run in list_runs(home)] == [
        ("round01", home, written)
    ]


def test_a_run_whose_record_cannot_be_read_is_still_listed(tmp_path: Path):
    home = _prepare(tmp_path)
    (home / RUN_RECORD_FILENAME).write_text("version: 99\n", encoding="utf-8")

    assert _names(tmp_path / "runs") == ["round01"]


def test_the_command_line_and_the_window_call_a_run_the_same(tmp_path: Path, capsys):
    runs = tmp_path / "runs"
    _run(_prepare(tmp_path), **SCORED)
    _run(runs, "training_261001_120000", **SCORED)
    _run(runs, "training_261002_120000", **SCORED)
    _run(runs / "theirs", "training_261003_120000", **SCORED)
    controller = TrainingController(stack_present=lambda: True)
    controller.training_directory = runs

    assert main(["compare-training", str(runs)]) == 0
    printed = sorted(line.split()[0] for line in capsys.readouterr().out.splitlines()[1:])

    assert printed == sorted(run.name for run in controller.runs()) == [
        "round01", "theirs", "training_261001_120000", "training_261002_120000",
    ]


def test_the_runs_own_validation_is_offered_unless_one_is_named():
    sets = [{"held": ()}, {"held": (), "other": ()}]

    assert offered_sets("", sets) == offered_sets(EVERY_SET, sets) == ("validation", "held", "other")
    assert offered_sets("held", sets) == ("held", "other")


def test_a_choice_that_cannot_be_read_is_none_to_a_listing_and_says_so(tmp_path: Path, caplog):
    assert chosen_best_or_none(tmp_path) is None
    (tmp_path / CHOICE_FILENAME).write_text("version: 99\nepoch: 1\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert chosen_best_or_none(tmp_path) is None

    assert CHOICE_FILENAME in caplog.text
