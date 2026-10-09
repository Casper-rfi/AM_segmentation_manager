"""Tests for reading what an AM-model-training run has written.

``data/amt_run`` is a real run's directory: the first three epochs of each metrics file
and its ``training_parameters.json``, written by AM-model-training 0.1.1.dev3, with the
cluster path in ``output_path`` replaced. It is what holds the reader to the format
rather than to a format this suite imagines. Everything else builds the smallest
directory that shows one behaviour.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.plots import has_plot_support
from am_segmentation_manager.training_runs import (
    DEFAULT_KEY_METRICS,
    EpochMetrics,
    best_epoch,
    find_training_runs,
    improving_epochs,
    parse_epochs,
    parse_started,
    read_training_run,
)

FIXTURE = Path(__file__).parent / "data" / "amt_run" / "training_260927_093830"


def _epoch(epoch: int, iou: float, dice: float, loss: float = 1.0) -> dict:
    """Return one metrics-file entry, as AM-model-training writes it."""
    return {
        "epoch": epoch, "loss": loss,
        "confusion_matrix": [[1, 0], [0, 1]],
        "iou": [iou, iou], "mean_iou": iou, "weighted_average_iou": iou,
        "dice": [dice, dice], "mean_dice": dice, "weighted_average_dice": dice,
    }


def _run(root: Path, name: str = "training_261001_120000", *, train=(), val=(),
         checkpoints=(), parameters=None, run_id: str = "set") -> Path:
    """Write a run directory holding exactly what is given."""
    directory = root / name
    directory.mkdir(parents=True)
    if train:
        (directory / f"{run_id}_train_metrics.json").write_text(json.dumps(list(train)))
    if val:
        (directory / f"{run_id}_val_metrics.json").write_text(json.dumps(list(val)))
    for epoch in checkpoints:
        (directory / f"{run_id}_epoch{epoch:03}.pth").write_bytes(b"")
    if parameters is not None:
        (directory / "training_parameters.json").write_text(json.dumps(parameters))
    return directory


# -- the real run -------------------------------------------------------------


def test_a_real_run_reads_as_its_files_say():
    run = read_training_run(FIXTURE)

    assert run.run_id == "train_arctis_val_all"
    assert run.started == datetime(2026, 9, 27, 9, 38, 30, tzinfo=UTC)
    assert run.finished
    assert [e.epoch for e in run.train] == [1, 2, 3]
    assert [e.epoch for e in run.val] == [1, 2, 3]
    assert run.label_names == ("background", "gis", "lamella", "crack", "vacuum")
    assert run.frozen_epochs == 25
    assert run.max_epochs == 100
    assert run.unreadable == ()


def test_a_real_run_s_epochs_carry_every_metric_per_class():
    first = read_training_run(FIXTURE).val[0]

    assert first.loss == pytest.approx(1.0549114393403431)
    for metric in ("iou", "dice", "f1", "accuracy", "precision", "recall"):
        assert len(first.per_class[metric]) == 5
        assert f"weighted_average_{metric}" in first.scalars
    assert "confusion_matrix" not in first.per_class


def test_the_best_epoch_is_ranked_as_the_run_ranks_it():
    """The mean of the key metrics, which the files do not store: epoch 2 scores
    0.304 against 0.277 and 0.300."""
    run = read_training_run(FIXTURE)

    assert run.key_metrics == DEFAULT_KEY_METRICS
    assert run.best().epoch == 2


# -- parsing ------------------------------------------------------------------


def test_the_earliest_of_a_tie_is_best_as_the_run_decides():
    epochs = parse_epochs([_epoch(1, 0.5, 0.5), _epoch(2, 0.6, 0.6), _epoch(3, 0.6, 0.6)])

    assert best_epoch(epochs, DEFAULT_KEY_METRICS).epoch == 2


def test_no_validated_epoch_means_no_best():
    assert best_epoch((), DEFAULT_KEY_METRICS) is None


def test_epochs_are_put_in_order():
    epochs = parse_epochs([_epoch(3, 0.1, 0.1), _epoch(1, 0.1, 0.1)])

    assert [e.epoch for e in epochs] == [1, 3]


def test_an_entry_of_another_format_is_refused():
    with pytest.raises(ValueError, match="epoch"):
        parse_epochs([{"iou": [0.5]}])


def test_score_is_the_mean_of_the_keys():
    epoch = EpochMetrics(1, 0.1, {"a": 0.2, "b": 0.4})

    assert epoch.score(("a", "b")) == pytest.approx(0.3)


def test_a_directory_name_gives_the_start_in_utc():
    assert parse_started("training_261008_235959") == datetime(2026, 10, 8, 23, 59, 59, tzinfo=UTC)
    assert parse_started("wide_seed2") is None


# -- a run in progress --------------------------------------------------------


def test_a_run_without_parameters_has_not_finished(tmp_path: Path):
    run = read_training_run(_run(tmp_path, train=[_epoch(1, 0.1, 0.1)]))

    assert not run.finished
    assert run.frozen_epochs is None
    assert run.max_epochs is None
    assert run.label_names == ("background", "gis", "lamella", "crack", "vacuum")


def test_a_file_caught_mid_write_is_reported_not_raised(tmp_path: Path):
    """AM-model-training truncates and rewrites each metrics file after every epoch,
    so a read can meet it empty or half-written."""
    directory = _run(tmp_path, train=[_epoch(1, 0.1, 0.1)], val=[_epoch(1, 0.1, 0.1)])
    (directory / "set_val_metrics.json").write_text('[{"epoch": 1, "lo')

    run = read_training_run(directory)

    assert run.unreadable == (directory / "set_val_metrics.json",)
    assert [e.epoch for e in run.train] == [1]
    assert run.val == ()


def test_checkpoints_are_found_by_epoch(tmp_path: Path):
    directory = _run(tmp_path, val=[_epoch(1, 0.1, 0.1), _epoch(2, 0.2, 0.2)],
                     checkpoints=[1, 2])

    run = read_training_run(directory)

    assert [kept.epoch for kept in run.checkpoints] == [1, 2]
    assert run.checkpoint_for(run.best().epoch).path == directory / "set_epoch002.pth"
    assert run.checkpoint_for(7) is None


def test_a_checkpoint_cut_short_is_not_whole(tmp_path: Path):
    """A run stopped while saving leaves the archive without its directory."""
    import zipfile

    directory = _run(tmp_path, checkpoints=[1, 2])
    with zipfile.ZipFile(directory / "set_epoch001.pth", "w") as archive:
        archive.writestr("set_epoch001/data.pkl", b"weights")
    (directory / "set_epoch002.pth").write_bytes(b"PK\x03\x04 cut short")

    first, second = read_training_run(directory).checkpoints

    assert first.whole()
    assert not second.whole()


def test_show_training_marks_a_checkpoint_cut_short(tmp_path: Path, capsys):
    _run(tmp_path, train=[_epoch(1, 0.1, 0.1)], checkpoints=[1])

    main(["show-training", str(tmp_path)])

    assert "checkpoints kept after epoch(s): 1 (incomplete)" in capsys.readouterr().out


def test_the_run_id_comes_from_a_checkpoint_when_nothing_else_names_it(tmp_path: Path):
    run = read_training_run(_run(tmp_path, checkpoints=[4], run_id="arctis"))

    assert run.run_id == "arctis"


def test_a_run_that_has_written_nothing_has_no_id(tmp_path: Path):
    run = read_training_run(_run(tmp_path))

    assert run.run_id is None
    assert run.best() is None


# -- finding runs -------------------------------------------------------------


def test_a_run_directory_is_found_as_itself(tmp_path: Path):
    directory = _run(tmp_path)

    assert find_training_runs(directory) == [directory]


def test_runs_are_found_two_levels_down_oldest_first(tmp_path: Path):
    """A training directory, a directory per run, then AM-model-training's own."""
    later = _run(tmp_path / "b", "training_261002_000000")
    earlier = _run(tmp_path / "a" / "seed0", "training_261001_000000")
    _run(tmp_path / "a" / "b" / "c", "training_260101_000000")

    assert find_training_runs(tmp_path) == [earlier, later]


def test_a_run_is_not_looked_inside(tmp_path: Path):
    outer = _run(tmp_path)
    _run(outer, "training_261009_000000")

    assert find_training_runs(tmp_path) == [outer]


def test_a_directory_merely_named_like_a_run_prefix_is_not_one(tmp_path: Path):
    (tmp_path / "training_notes").mkdir()

    assert find_training_runs(tmp_path) == []


# -- the command --------------------------------------------------------------


def test_show_training_reports_the_best_epoch(tmp_path: Path, capsys):
    shutil.copytree(FIXTURE, tmp_path / FIXTURE.name)

    assert main(["show-training", str(tmp_path)]) == 0

    said = capsys.readouterr().out
    assert "train_arctis_val_all" in said
    assert "finished after 3 of at most 100 epochs" in said
    assert "best validation epoch 2" in said
    assert "not kept in this directory" in said


def test_show_training_says_an_unfinished_run_may_be_running(tmp_path: Path, capsys):
    _run(tmp_path, train=[_epoch(1, 0.1, 0.1)])

    assert main(["show-training", str(tmp_path)]) == 0

    assert "not finished (training, or stopped): 1 epoch(s) so far" in capsys.readouterr().out


def test_show_training_refuses_a_place_with_no_run(tmp_path: Path):
    assert main(["show-training", str(tmp_path)]) == 2


@pytest.mark.skipif(not has_plot_support(), reason="matplotlib not installed")
def test_show_training_draws_one_run(tmp_path: Path):
    output = tmp_path / "curves.png"

    assert main(["show-training", str(FIXTURE), "--plot", str(output)]) == 0
    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.skipif(not has_plot_support(), reason="matplotlib not installed")
def test_show_training_draws_the_classes_named_and_refuses_one_it_does_not_know(tmp_path: Path):
    output = tmp_path / "curves.svg"

    assert main(["show-training", str(FIXTURE), "--plot", str(output), "--classes", "gis", "crack"]) == 0
    drawn = output.read_text(encoding="utf-8")
    assert "crack" in drawn and "vacuum" not in drawn

    assert main(["show-training", str(FIXTURE), "--plot", str(tmp_path / "no.svg"), "--classes", "ice"]) == 2
    assert not (tmp_path / "no.svg").exists()


def test_show_training_will_not_draw_several_runs_at_once(tmp_path: Path):
    _run(tmp_path, "training_261001_000000", train=[_epoch(1, 0.1, 0.1)])
    _run(tmp_path, "training_261002_000000", train=[_epoch(1, 0.1, 0.1)])

    assert main(["show-training", str(tmp_path), "--plot", str(tmp_path / "c.png")]) == 2
    assert not (tmp_path / "c.png").exists()


def test_show_training_will_not_replace_a_figure(tmp_path: Path):
    output = tmp_path / "curves.png"
    output.write_bytes(b"mine")

    assert main(["show-training", str(FIXTURE), "--plot", str(output)]) == 2
    assert output.read_bytes() == b"mine"


# -- which epochs a run saved -------------------------------------------------


def _scored(*scores: float) -> list[dict]:
    return [_epoch(number, score, score) for number, score in enumerate(scores, 1)]


def test_an_epoch_is_saved_when_it_improves_on_training_or_on_validation(tmp_path: Path):
    """As AM-model-training keeps them: a new best on either side, the first of a tie."""
    directory = _run(
        tmp_path, train=_scored(0.1, 0.2, 0.2, 0.3, 0.3), val=_scored(0.5, 0.4, 0.4, 0.4, 0.6)
    )

    run = read_training_run(directory)

    assert improving_epochs(run.train, run.key_train_metrics) == (1, 2, 4)
    assert improving_epochs(run.val, run.key_metrics) == (1, 5)
    assert run.saved_epochs(save_all=False) == (1, 2, 4, 5)


def test_a_run_told_to_keep_every_epoch_saved_every_one(tmp_path: Path):
    directory = _run(tmp_path, train=_scored(0.3, 0.2, 0.1), val=_scored(0.3, 0.2, 0.1))

    assert read_training_run(directory).saved_epochs(save_all=True) == (1, 2, 3)


def test_weights_for_an_epoch_that_improved_nothing_say_every_epoch_was_kept(tmp_path: Path):
    """Nothing in a run AM-model-training was started on directly says what it was told."""
    scores = _scored(0.3, 0.2, 0.1)
    told = _run(tmp_path, "training_261001_120000", train=scores, val=scores, checkpoints=(2,))
    untold = _run(tmp_path, "training_261002_120000", train=scores, val=scores, checkpoints=(1,))

    assert read_training_run(told).saved_epochs() == (1, 2, 3)
    assert read_training_run(untold).saved_epochs() == (1,)


def test_the_saved_epochs_of_a_real_run_follow_its_metrics():
    """The real run improved on training at every epoch, and on validation at two."""
    run = read_training_run(FIXTURE)

    assert improving_epochs(run.val, run.key_metrics) == (1, 2)

    assert run.saved_epochs(save_all=False) == (1, 2, 3)


# -- marking an epoch as best from the command line ---------------------------


def test_choose_best_marks_an_epoch_and_show_training_reports_it(tmp_path: Path, capsys):
    directory = _run(tmp_path, train=_scored(0.1, 0.2), val=_scored(0.2, 0.1), checkpoints=(1,))

    assert main(["--user", "Casper", "choose-best", str(directory), "--epoch", "2"]) == 0
    capsys.readouterr()
    main(["show-training", str(directory)])
    said = capsys.readouterr().out

    assert "best validation epoch 1" in said
    assert "marked as best: epoch 2\n    weights: not kept in this directory" in said


def test_choose_best_refuses_an_epoch_never_validated_and_clears_on_request(tmp_path: Path, capsys):
    directory = _run(tmp_path, train=_scored(0.1), val=_scored(0.2))

    assert main(["choose-best", str(directory), "--epoch", "5"]) == 2
    assert main(["choose-best", str(directory), "--epoch", "1"]) == 0
    assert main(["choose-best", str(directory), "--clear"]) == 0
    assert "Cleared" in capsys.readouterr().out
    assert main(["choose-best", str(tmp_path / "nowhere"), "--epoch", "1"]) == 2


# -- runs side by side from the command line ----------------------------------


def test_compare_training_prints_a_row_per_run_and_writes_the_csv(tmp_path: Path, capsys):
    _run(tmp_path / "a", train=_scored(0.1, 0.2), val=_scored(0.25, 0.5))
    _run(tmp_path / "b", train=_scored(0.1), val=_scored(0.75))
    output = tmp_path / "table.csv"

    assert main(["compare-training", str(tmp_path), "--output", str(output)]) == 0
    said = capsys.readouterr().out.splitlines()

    assert said[0].split()[:2] == ["run", "epochs"]
    assert [line.split()[:3] for line in said[1:]] == [["a", "2", "0.5000"], ["b", "1", "0.7500"]]
    assert output.read_text(encoding="utf-8").splitlines()[1].startswith("a,2,0.5000")
    assert main(["compare-training", str(tmp_path), "--output", str(output)]) == 2


def test_compare_training_takes_the_epoch_marked_and_a_range(tmp_path: Path, capsys):
    directory = _run(tmp_path / "a", train=_scored(0.1, 0.2, 0.3), val=_scored(0.25, 0.5, 0.75))
    main(["choose-best", str(directory), "--epoch", "1"])
    capsys.readouterr()

    main(["compare-training", str(tmp_path), "--value", "marked"])
    assert capsys.readouterr().out.splitlines()[1].split()[:4] == ["a", "1", "(marked)", "0.2500"]

    main(["compare-training", str(tmp_path), "--value", "mean", "--epochs", "2", "3"])
    assert capsys.readouterr().out.splitlines()[1].split()[:4] == ["a", "2-3", "(2)", "0.6250"]


def test_compare_training_with_no_run_says_so(tmp_path: Path):
    assert main(["compare-training", str(tmp_path)]) == 2


def test_compare_training_shades_its_plot_between_the_values_asked_for(tmp_path: Path, monkeypatch):
    """From 0 to 1 unless asked, so a plot made before the option is the plot made now."""
    pytest.importorskip("matplotlib")
    from am_segmentation_manager.cli import training as command

    _run(tmp_path / "a", train=_scored(0.1, 0.2), val=_scored(0.25, 0.5))
    _run(tmp_path / "b", train=_scored(0.1), val=_scored(0.75))
    shaded = []
    monkeypatch.setattr(
        command, "plot_comparison",
        lambda table, output, *, overwrite, colour_range: shaded.append(colour_range),
    )
    plot = str(tmp_path / "table.png")

    assert main(["compare-training", str(tmp_path), "--plot", plot]) == 0
    assert main(["compare-training", str(tmp_path), "--plot", plot, "--colour-range", "0.4,0.9"]) == 0
    assert main(["compare-training", str(tmp_path), "--plot", plot, "--colour-range", "auto"]) == 0

    assert shaded[:2] == [(0.0, 1.0), (0.4, 0.9)]
    low, high = shaded[2]
    assert 0.0 < low < high < 1.0


@pytest.mark.parametrize("asked", ["0.9,0.4", "0.5", "low,high", "0.2,1.5"])
def test_a_colour_range_that_is_not_one_is_refused_at_the_command_line(tmp_path: Path, asked):
    with pytest.raises(SystemExit):
        main(["compare-training", str(tmp_path), "--colour-range", asked])



def test_a_folder_that_cannot_be_listed_is_passed_over_and_said_so(tmp_path: Path, monkeypatch, caplog):
    from am_segmentation_manager.training_runs import find_directories

    wanted = _run(tmp_path / "open")
    (tmp_path / "shut").mkdir()
    (tmp_path / ".hidden" / "training_261001_120000").mkdir(parents=True)
    listing = Path.iterdir

    def iterdir(directory: Path):
        if directory.name == "shut":
            raise PermissionError("not yours")
        return listing(directory)

    monkeypatch.setattr(Path, "iterdir", iterdir)

    with caplog.at_level("WARNING"):
        found = find_directories(tmp_path, 1, lambda path: path.name.startswith("training_"))

    assert found == [wanted]
    assert "shut" in caplog.text
