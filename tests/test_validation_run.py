"""Tests for preparing a validation inside its run, and for the process that scores it.

Nothing here loads a model: the scoring is stood in for, since what is under test is
what is checked before anything is written, where things are put, and that each epoch
scored is kept as it is done.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.io.tables import write_csv
from am_segmentation_manager.operations import training_process
from am_segmentation_manager.operations.training_process import run_child, run_status
from am_segmentation_manager.operations.training_run import TrainingRunRefused
from am_segmentation_manager.operations.validation_run import (
    apply_validation,
    find_validations,
    plan_validation,
    read_images,
    read_validation_record,
)
from am_segmentation_manager.validation_scores import find_scores, metrics_for, read_scores
from test_training_process import export_of_placeholders
from test_training_runs import _epoch


def _run(tmp_path: Path, epochs=(1, 2, 3), cut_short=()) -> Path:
    """Return a run AM-model-training wrote, with whole weights for *epochs*."""
    written = tmp_path / "runs" / "wide" / "training_261001_120000"
    written.mkdir(parents=True)
    entries = [_epoch(n, 0.1 * n, 0.1 * n) for n in range(1, max(epochs) + 1)]
    (written / "set_train_metrics.json").write_text(json.dumps(entries))
    (written / "set_val_metrics.json").write_text(json.dumps(entries))
    for epoch in epochs:
        path = written / f"set_epoch{epoch:03}.pth"
        if epoch in cut_short:
            path.write_bytes(b"cut short")
        else:
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data.pkl", "weights")
    return written


def _plan(tmp_path: Path, run: Path | None = None, **asked):
    asked.setdefault("inspect_label", lambda path, classes: None)
    return plan_validation(
        run or _run(tmp_path), asked.pop("manifest", None) or export_of_placeholders(tmp_path),
        asked.pop("name", "held"), **asked,
    )


def _refusal(tmp_path: Path, **asked) -> str:
    with pytest.raises(TrainingRunRefused) as refused:
        _plan(tmp_path, **asked)
    return str(refused.value)


# -- preparing ----------------------------------------------------------------


def test_a_validation_scores_the_export_s_validation_list_on_every_whole_checkpoint(tmp_path: Path):
    plan = _plan(tmp_path)

    assert [checkpoint.epoch for checkpoint in plan.checkpoints] == [1, 2, 3]
    assert [image.name for image, _ in plan.pairs] == ["b.tif"]
    assert plan.images == (("b.tif",),)
    assert plan.directory == plan.run_directory / "am_validations" / "held"
    assert "epoch(s) 1-3" in "\n".join(plan.describe())


def test_every_row_can_be_asked_for(tmp_path: Path):
    assert [image.name for image, _ in _plan(tmp_path, rows="all").pairs] == ["a.tif", "b.tif"]


def test_planning_writes_nothing(tmp_path: Path):
    run = _run(tmp_path)
    before = sorted(run.rglob("*"))

    _plan(tmp_path, run)

    assert sorted(run.rglob("*")) == before


def test_a_checkpoint_cut_short_is_left_out_and_refused_when_asked_for(tmp_path: Path):
    run = _run(tmp_path, cut_short=(2,))

    assert [c.epoch for c in _plan(tmp_path, run).checkpoints] == [1, 3]
    with pytest.raises(TrainingRunRefused, match=r"No whole checkpoint for epoch\(s\) 2, 9"):
        _plan(tmp_path, run, name="again", epochs=[1, 2, 9], manifest=tmp_path / "export" / "set_manifest.yaml")


def test_a_run_with_no_checkpoint_cannot_be_validated(tmp_path: Path):
    empty = tmp_path / "runs" / "training_261002_000000"
    empty.mkdir(parents=True)

    assert "holds no checkpoint to score" in _refusal(tmp_path, run=empty)


def test_a_label_that_cannot_be_scored_against_stops_the_validation(tmp_path: Path):
    said = _refusal(tmp_path, inspect_label=lambda path, classes: "could not be read")

    assert "1 label(s) cannot be scored against" in said
    assert not (tmp_path / "runs" / "wide" / "training_261001_120000" / "am_validations").exists()


def test_a_name_that_cannot_be_one_and_rows_that_are_none_are_refused_together(tmp_path: Path):
    said = _refusal(tmp_path, name="held out", rows="some")

    assert "cannot name a validation" in said and "Unknown rows" in said


def test_an_export_with_no_validation_list_says_to_score_every_row(tmp_path: Path):
    manifest = export_of_placeholders(tmp_path)
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("validation: set_val.csv", "validation: null"),
        encoding="utf-8",
    )

    run = _run(tmp_path)

    assert "score every row of it instead" in _refusal(tmp_path, run=run, manifest=manifest)
    assert len(_plan(tmp_path, run, manifest=manifest, rows="all").pairs) == 1


# -- writing ------------------------------------------------------------------


def test_a_validation_is_written_inside_its_run_and_read_back(tmp_path: Path):
    plan = _plan(tmp_path)

    directory = apply_validation(plan, user="Casper")

    record = read_validation_record(directory)
    assert (record.name, record.epochs, record.created_by) == ("held", (1, 2, 3), "Casper")
    assert record.run_directory == plan.run_directory
    assert find_validations(plan.run_directory) == [directory]
    assert read_images(record.images_file) == [{"image": "b.tif"}]
    assert run_status(directory).state == "prepared"


def test_a_validation_directory_is_never_reused(tmp_path: Path):
    plan = _plan(tmp_path)
    apply_validation(plan)

    with pytest.raises(FileExistsError):
        apply_validation(plan)
    assert "already exists" in _refusal(
        tmp_path, run=plan.run_directory, manifest=plan.manifest.path
    )


def test_the_metadata_the_export_wrote_travels_with_the_rows_scored(tmp_path: Path):
    manifest = export_of_placeholders(tmp_path)
    write_csv(
        manifest.parent / "set_images.csv", ("split", "image", "instrument"),
        [("train", "a.tif", "Arctis"), ("validation", "b.tif", "Hydra")],
    )
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "validation: set_val.csv", "validation: set_val.csv\n  images: set_images.csv"
        ),
        encoding="utf-8",
    )

    directory = apply_validation(_plan(tmp_path, manifest=manifest, rows="all"))

    assert read_images(directory / "images.csv") == [
        {"image": "a.tif", "split": "train", "instrument": "Arctis"},
        {"image": "b.tif", "split": "validation", "instrument": "Hydra"},
    ]


@pytest.mark.parametrize(
    ("asked", "said"),
    [
        ({"start_path": "."}, "is inside the collection at"),
        ({"gone": "b.tif"}, "1 file(s) the lists name are not under"),
        ({"inspect_label": lambda path, classes: "could not be read"}, "could not be read"),
    ],
)
def test_a_run_and_a_validation_refuse_an_export_for_the_same_reasons(tmp_path: Path, asked, said):
    from am_segmentation_manager.operations.training_run import NEW_MODEL, plan_training_run

    manifest = export_of_placeholders(tmp_path)
    run = _run(tmp_path)
    asked = dict(asked)
    asked.setdefault("inspect_label", lambda path, classes: None)
    if "start_path" in asked:
        asked["start_path"] = tmp_path
    if "gone" in asked:
        (tmp_path / "collection" / asked.pop("gone")).unlink()

    with pytest.raises(TrainingRunRefused) as training:
        plan_training_run(manifest, tmp_path / "runs", "round01", NEW_MODEL, **asked)
    with pytest.raises(TrainingRunRefused) as validation:
        plan_validation(run, manifest, "held", **asked)

    assert said in str(training.value) and said in str(validation.value)


# -- scoring ------------------------------------------------------------------


def _stand_in(monkeypatch, fail_after: int | None = None) -> list[Path]:
    """Stand in for the model: every image scores a perfect two-class matrix."""
    from am_segmentation_manager.ml import scoring

    scored: list[Path] = []

    def score_checkpoints(checkpoints, list_file, *, device="auto", workers=None, on_device=None):
        images = len(list_file.read_text(encoding="utf-8").splitlines()) - 1
        for weights in checkpoints:
            if fail_after is not None and len(scored) >= fail_after:
                raise RuntimeError("out of memory")
            scored.append(weights)
            yield [[[4, 0], [0, 2]]] * images, [0.25] * images

    monkeypatch.setattr(scoring, "score_checkpoints", score_checkpoints)
    return scored


def test_each_epoch_s_scores_are_kept_by_the_names_the_export_gave(tmp_path: Path, monkeypatch):
    scored = _stand_in(monkeypatch)
    directory = apply_validation(_plan(tmp_path, rows="all"))

    assert run_child(directory) == 0

    assert [path.name for path in scored] == [f"set_epoch00{n}.pth" for n in (1, 2, 3)]
    assert run_status(directory).state == "finished"
    files = find_scores(directory)
    assert [path.name for path in files] == [f"scores_epoch00{n}.json" for n in (1, 2, 3)]
    scores = read_scores(files[1])
    assert (scores.epoch, scores.images) == (2, ("a.tif", "b.tif"))
    assert metrics_for(scores, {"b.tif"}).per_class["iou"] == (1.0, 1.0)


def test_the_weights_a_scores_file_records_are_the_ones_the_loss_used(tmp_path: Path, monkeypatch):
    from am_segmentation_manager.ml.scoring import evaluation_arguments
    from am_segmentation_manager.validation_scores import CLASS_WEIGHTS

    _stand_in(monkeypatch)
    directory = apply_validation(_plan(tmp_path))
    run_child(directory)

    asked = evaluation_arguments(Path("set_epoch001.pth"), Path("list.csv"), True, 0, None)
    assert asked["loss_kwargs"]["weights"] is CLASS_WEIGHTS
    assert read_scores(find_scores(directory)[0]).weights == CLASS_WEIGHTS


def test_a_validation_that_fails_part_way_keeps_the_epochs_it_finished(tmp_path: Path, monkeypatch):
    _stand_in(monkeypatch, fail_after=2)
    directory = apply_validation(_plan(tmp_path))

    assert run_child(directory) == 1

    status = run_status(directory)
    assert status.state == "failed" and "out of memory" in status.error
    assert len(find_scores(directory)) == 2


def test_weights_deleted_since_a_validation_was_prepared_fail_it_before_anything_is_scored(
    tmp_path: Path, monkeypatch
):
    scored = _stand_in(monkeypatch)
    plan = _plan(tmp_path)
    directory = apply_validation(plan)
    plan.checkpoints[1].path.unlink()

    assert run_child(directory) == 1

    assert scored == [] and "epoch 2 have gone" in run_status(directory).error


def test_a_validation_is_started_once_like_a_run(tmp_path: Path, monkeypatch):
    _stand_in(monkeypatch)
    directory = apply_validation(_plan(tmp_path))
    run_child(directory)

    with pytest.raises(training_process.TrainingNotStarted, match="already started"):
        training_process.start_training(directory, ["true"])


# -- the command line ---------------------------------------------------------


def test_validate_says_what_it_would_do_and_a_dry_run_writes_nothing(tmp_path: Path, capsys, monkeypatch):
    from am_segmentation_manager.operations import training_run

    monkeypatch.setattr(training_run, "label_report", lambda path, classes: (None, None))
    run = _run(tmp_path)
    manifest = export_of_placeholders(tmp_path)

    assert main(["validate", str(run), str(manifest), "--name", "held", "--dry-run"]) == 0

    said = capsys.readouterr().out
    assert "1 image(s), the validation list" in said and "nothing was written" in said
    assert find_validations(run) == []


def test_validate_gives_every_reason_it_cannot(tmp_path: Path, capsys):
    run = tmp_path / "nowhere"

    assert main(["validate", str(run), str(tmp_path / "none_manifest.yaml"), "--name", "a b"]) == 2

    said = capsys.readouterr().out
    assert "cannot name a validation" in said and "holds no checkpoint" in said


def test_show_training_reports_a_run_s_validations(tmp_path: Path, capsys, monkeypatch):
    _stand_in(monkeypatch)
    plan = _plan(tmp_path)
    run_child(apply_validation(plan))
    capsys.readouterr()

    main(["show-training", str(plan.run_directory)])

    said = capsys.readouterr().out
    assert "validation held: finished, 3 of 3 epoch(s) scored" in said
    assert "epoch 3, iou per class: 1.000, 1.000" in said


# -- a validation scored, for the tests of what is made of one ----------------


def scored_validation(tmp_path: Path, monkeypatch, run: Path | None = None, name: str = "held") -> Path:
    """Return a run with a finished validation of two images, one of each of two instruments."""
    from am_segmentation_manager.operations import training_run

    monkeypatch.setattr(training_run, "label_report", lambda path, classes: (None, None))
    _stand_in(monkeypatch)
    run = run or _run(tmp_path)
    manifest = tmp_path / "export" / "set_manifest.yaml"
    if not manifest.exists():
        manifest = export_of_placeholders(tmp_path)
    if not (manifest.parent / "set_images.csv").exists():
        write_csv(
            manifest.parent / "set_images.csv", ("split", "image", "instrument"),
            [("train", "a.tif", "Arctis"), ("validation", "b.tif", "Hydra")],
        )
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "validation: set_val.csv", "validation: set_val.csv\n  images: set_images.csv"
            ),
            encoding="utf-8",
        )
    run_child(apply_validation(plan_validation(run, manifest, name, rows="all")))
    return run


def test_compare_training_gives_a_column_to_each_part_of_a_validation(tmp_path: Path, capsys, monkeypatch):
    run = scored_validation(tmp_path, monkeypatch)

    assert main([
        "compare-training", str(run), "--validated-on", "held", "--split-by", "instrument",
        "--classes", "gis",
    ]) == 0

    header, row = (line.split() for line in capsys.readouterr().out.splitlines())
    assert header == ["run", "epochs", "gis", "Arctis", "iou", "gis", "Hydra", "iou"]
    assert row[-2:] == ["1.0000", "1.0000"]


def test_compare_training_sets_every_validation_beside_the_run_s_own(tmp_path: Path, capsys, monkeypatch):
    run = scored_validation(tmp_path, monkeypatch)

    assert main(["compare-training", str(run), "--validated-on", "all", "--classes", "gis"]) == 0
    assert "gis validation iou  gis held iou" in capsys.readouterr().out

    assert main(["compare-training", str(run), "--split-by", "instrument"]) == 2
