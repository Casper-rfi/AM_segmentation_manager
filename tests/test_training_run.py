"""Tests for preparing a training run's directory from a training export.

Each starts from a real export -- ``export_training_csv`` over real label files --
so the manifest and lists a run reads are the ones an export writes, not ones this
suite imagines. What is pinned is the contract: every reason for a refusal at once,
nothing written before the checks pass, and a directory another program can train from.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.csv import export_kfold_csv, export_training_csv
from am_segmentation_manager.operations.training_manifest import read_training_manifest
from am_segmentation_manager.operations.training_run import (
    NEW_MODEL,
    REFINE,
    RUN_RECORD_FILENAME,
    TrainingRunRefused,
    apply_training_run,
    default_settings,
    label_problem,
    plan_training_run,
    read_run_record,
)
from conftest import write_label


def _records(root: Path, count: int = 10) -> list[ImageRecord]:
    """Return *count* SEM images, one per lamella, each with an annotated label."""
    records = []
    for i in range(count):
        stage = root / f"{i:02d}-lamella" / "Adaptive_polish_1"
        image = stage / "sem" / f"AP_img_{i:03d}_SEM.tif"
        image.parent.mkdir(parents=True)
        image.write_text("img")
        label = write_label(stage / "labels" / image.name, "annotated")
        records.append(
            ImageRecord(image_path=image, labels=[label], state="corrected", modality="SEM")
        )
    return records


@pytest.fixture
def export(tmp_path: Path) -> Path:
    """Return the manifest of a ten-image export, made from an absolute collection path."""
    root = tmp_path / "collection"
    export_training_csv(_records(root), root, tmp_path / "export", prefix="set")
    return tmp_path / "export" / "set_manifest.yaml"


def _plan(export: Path, **changes):
    arguments = {
        "manifest_file": export,
        "training_directory": export.parent.parent / "runs",
        "name": "round01",
        "settings": NEW_MODEL,
    }
    arguments.update(changes)
    return plan_training_run(**arguments)


def _refusal(export: Path, **changes) -> str:
    with pytest.raises(TrainingRunRefused) as refused:
        _plan(export, **changes)
    return str(refused.value)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# -- the published defaults ---------------------------------------------------


def test_a_new_model_defaults_to_the_published_settings():
    assert (NEW_MODEL.max_epochs, NEW_MODEL.frozen_epochs) == (100, 25)
    assert (NEW_MODEL.initial_learning_rate, NEW_MODEL.max_learning_rate) == (1e-5, 1e-3)
    assert default_settings("new") is NEW_MODEL


def test_refining_defaults_to_the_published_settings():
    assert (REFINE.max_epochs, REFINE.frozen_epochs) == (50, 0)
    assert (REFINE.initial_learning_rate, REFINE.max_learning_rate) == (2e-6, 3e-4)
    assert default_settings("refine") is REFINE


def test_neither_default_saves_every_epoch():
    assert not NEW_MODEL.save_all and not REFINE.save_all


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"max_epochs": 0}, "at least 1"),
        ({"frozen_epochs": 100}, "fewer epochs than are trained"),
        ({"frozen_epochs": -1}, "fewer epochs than are trained"),
        ({"initial_learning_rate": 0.0}, "above 0"),
        ({"initial_learning_rate": 1.0}, "no higher than the maximum"),
        ({"training_batch_size": 0}, "at least 1 image"),
        ({"mode": "retrain"}, "Unknown mode"),
    ],
)
def test_settings_that_cannot_train_are_refused(export: Path, change, said):
    assert said in _refusal(export, settings=replace(NEW_MODEL, **change))


# -- planning -----------------------------------------------------------------


def test_a_plan_holds_every_pair_of_the_export_as_absolute_paths(export: Path):
    plan = _plan(export)

    assert len(plan.train) == 8 and len(plan.validation) == 2
    assert all(path.is_absolute() and path.is_file() for pair in plan.train for path in pair)
    assert plan.run_directory == export.parent.parent / "runs" / "round01"


def test_planning_writes_nothing(export: Path):
    _plan(export)

    assert not (export.parent.parent / "runs").exists()


def test_a_plan_says_what_it_would_do(export: Path):
    said = "\n".join(_plan(export).describe())

    assert "8 training and 2 validation pair(s)" in said
    assert "encoder frozen for the first 25" in said
    assert "1e-05 to 0.001" in said
    assert "ImageNet weights, which are downloaded" in said


def test_every_reason_is_given_at_once(export: Path):
    said = _refusal(export, name="has space", settings=replace(REFINE, max_epochs=0))

    assert "cannot name a run" in said
    assert "at least 1" in said
    assert "needs the model weights" in said


def test_an_existing_run_directory_is_never_reused(export: Path):
    existing = export.parent.parent / "runs" / "round01"
    existing.mkdir(parents=True)

    assert "already exists" in _refusal(export)
    assert list(existing.iterdir()) == []


@pytest.mark.parametrize("name", ["", "has space", "a/b", "../up", ".hidden"])
def test_a_name_that_is_not_a_plain_file_name_is_refused(export: Path, name: str):
    assert "cannot name a run" in _refusal(export, name=name)


def test_a_run_inside_the_collection_is_refused(export: Path):
    """The scanner would walk it, and a collection is data, not models."""
    inside = export.parent.parent / "collection" / "runs"

    assert "inside the collection" in _refusal(export, training_directory=inside)


def test_refining_needs_weights_that_exist(export: Path, tmp_path: Path):
    assert "needs the model weights" in _refusal(export, settings=REFINE)
    assert "do not exist" in _refusal(
        export, settings=REFINE, base_weights=tmp_path / "absent.pth"
    )


def test_a_new_model_takes_no_weights(export: Path, tmp_path: Path):
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"weights")

    assert "starts from no weights" in _refusal(export, base_weights=weights)


def test_a_cross_validation_export_is_refused(tmp_path: Path):
    root = tmp_path / "collection"
    export_kfold_csv(_records(root), root, tmp_path / "export", folds=2, prefix="set")

    assert "cross-validation export of 2 folds" in _refusal(
        tmp_path / "export" / "set_manifest.yaml"
    )


def test_an_export_of_both_beams_on_a_row_is_refused(tmp_path: Path):
    root = tmp_path / "collection"
    export_training_csv(_records(root), root, tmp_path / "export", prefix="set", paired=True)

    assert "one beam's image,label pairs" in _refusal(tmp_path / "export" / "set_manifest.yaml")


def test_an_export_with_no_validation_list_is_refused(tmp_path: Path):
    """AM-model-training would split one list at random, cutting through lamellae."""
    root = tmp_path / "collection"
    export_training_csv(_records(root), root, tmp_path / "export", prefix="set", train_ratio=1.0)

    assert "no validation list" in _refusal(tmp_path / "export" / "set_manifest.yaml")


def test_a_manifest_recording_a_relative_folder_needs_the_folder_named(
    tmp_path: Path, monkeypatch
):
    """The command line's default start path is ".", which says nothing later."""
    root = tmp_path / "collection"
    records = _records(root)
    monkeypatch.chdir(root)
    export_training_csv(records, Path("."), tmp_path / "export", prefix="set")
    manifest = tmp_path / "export" / "set_manifest.yaml"
    monkeypatch.chdir(tmp_path)

    assert "does not say where the collection is" in _refusal(manifest)
    assert len(_plan(manifest, start_path=root).train) == 8


def test_a_file_that_has_gone_is_refused_and_named(export: Path):
    gone = export.parent.parent / "collection" / "00-lamella" / "Adaptive_polish_1" / "sem"
    (gone / "AP_img_000_SEM.tif").unlink()

    said = _refusal(export)

    assert "1 file(s) the lists name are not under" in said
    assert "AP_img_000_SEM.tif" in said


def test_many_missing_files_are_counted_not_all_listed(export: Path):
    for image in (export.parent.parent / "collection").rglob("sem/*.tif"):
        image.unlink()

    assert "and 5 more" in _refusal(export)


# -- labels are looked inside again -------------------------------------------


def _first_label(export: Path) -> Path:
    stage = export.parent.parent / "collection" / "00-lamella" / "Adaptive_polish_1"
    return stage / "labels" / "AP_img_000_SEM.tif"


def test_a_label_emptied_since_the_export_is_refused(export: Path):
    """An export leaves empty labels out, and a label can be emptied afterwards."""
    write_label(_first_label(export), "empty")

    said = _refusal(export)

    assert "1 label(s) cannot be trained on" in said
    assert "holds only background" in said


def test_a_label_that_no_longer_opens_is_refused(export: Path):
    _first_label(export).write_bytes(b"not a tiff")

    assert "could not be read" in _refusal(export)


def test_a_label_holding_a_class_the_model_does_not_predict_is_refused(export: Path):
    import numpy as np
    import tifffile

    array = np.zeros((16, 16), dtype=np.uint8)
    array[2:6, 2:6] = 5
    tifffile.imwrite(str(_first_label(export)), array)

    assert "holds class 5, and the model predicts classes 0 to 4" in _refusal(export)


def test_an_export_told_to_include_empty_labels_may_train_on_them(tmp_path: Path):
    root = tmp_path / "collection"
    records = _records(root)
    write_label(records[0].labels[0], "empty")
    export_training_csv(
        records, root, tmp_path / "export", prefix="set", include_empty_labels=True
    )

    plan = _plan(tmp_path / "export" / "set_manifest.yaml")

    assert len(plan.train) + len(plan.validation) == 10


def test_an_annotated_label_within_the_classes_has_no_problem(export: Path):
    assert label_problem(_first_label(export), 5) is None


# -- writing ------------------------------------------------------------------


def test_applying_writes_the_run_directory_whole(export: Path):
    directory = apply_training_run(_plan(export))

    assert sorted(path.name for path in directory.iterdir()) == [
        RUN_RECORD_FILENAME, "round01.csv", "set_manifest.yaml", "val.csv",
    ]


def test_the_lists_are_ones_another_program_resolves_from_anywhere(export: Path):
    """AM-model-training resolves a relative path against the list's own folder."""
    directory = apply_training_run(_plan(export))

    train, validation = _rows(directory / "round01.csv"), _rows(directory / "val.csv")

    assert len(train) == 8 and len(validation) == 2
    assert list(train[0]) == ["image", "label"]
    for row in train + validation:
        assert Path(row["image"]).is_absolute() and Path(row["image"]).is_file()
        assert Path(row["label"]).is_absolute() and Path(row["label"]).is_file()


def test_the_split_is_the_export_s_and_not_remade(export: Path):
    directory = apply_training_run(_plan(export))

    def names(path: Path) -> list[str]:
        return [Path(row["image"]).name for row in _rows(path)]

    assert names(directory / "round01.csv") == names(export.parent / "set_train.csv")
    assert names(directory / "val.csv") == names(export.parent / "set_val.csv")


def test_the_export_s_manifest_is_copied_unchanged(export: Path):
    directory = apply_training_run(_plan(export))

    assert (directory / "set_manifest.yaml").read_bytes() == export.read_bytes()


def test_the_record_reads_back_as_what_was_asked(export: Path):
    settings = replace(NEW_MODEL, max_epochs=12, frozen_epochs=3, save_all=True)
    directory = apply_training_run(_plan(export, settings=settings, device="cuda:1"), user="Ada")

    record = read_run_record(directory)

    assert record.name == "round01"
    assert record.settings == settings
    assert record.base is None
    assert record.device == "cuda:1"
    assert record.created_by == "Ada"
    assert record.train_list == directory / "round01.csv"
    assert record.validation_list == directory / "val.csv"


def test_a_refine_records_what_identifies_its_starting_weights(export: Path, tmp_path: Path):
    import hashlib

    weights = tmp_path / "AM_SEM_All_V01.pth"
    weights.write_bytes(b"weights")
    directory = apply_training_run(_plan(export, settings=REFINE, base_weights=weights))

    base = read_run_record(directory).base

    assert base["file"] == str(weights.resolve())
    assert base["size"] == 7
    assert base["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert base["run"] is None, "not a torch archive, so no run is recorded inside it"


def test_the_record_says_where_the_set_came_from(export: Path):
    directory = apply_training_run(_plan(export))

    document = yaml.safe_load((directory / RUN_RECORD_FILENAME).read_text(encoding="utf-8"))

    assert document["training_set"] == {
        "manifest": "set_manifest.yaml",
        "source": str((export.parent.parent / "collection").resolve()),
        "train": 8,
        "validation": 2,
    }


def test_a_directory_that_appeared_since_the_plan_is_not_written_into(export: Path):
    plan = _plan(export)
    plan.run_directory.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        apply_training_run(plan)
    assert list(plan.run_directory.iterdir()) == []


def test_a_directory_without_a_record_has_none(tmp_path: Path):
    assert read_run_record(tmp_path) is None


def test_a_record_from_a_later_am_seg_is_refused(export: Path):
    directory = apply_training_run(_plan(export))
    record = directory / RUN_RECORD_FILENAME
    record.write_text(
        record.read_text(encoding="utf-8").replace("version: 1", "version: 2"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="later version"):
        read_run_record(directory)


def test_a_file_that_is_not_a_record_is_refused(tmp_path: Path):
    (tmp_path / RUN_RECORD_FILENAME).write_text("version: 1\nname: x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a training run's record"):
        read_run_record(tmp_path)


# -- the manifest reader ------------------------------------------------------


def test_a_manifest_reads_back_as_the_export_wrote_it(export: Path):
    manifest = read_training_manifest(export)

    assert manifest.train == export.parent / "set_train.csv"
    assert manifest.validation == export.parent / "set_val.csv"
    assert manifest.columns == ("image", "label")
    assert manifest.folds == 0
    assert not manifest.include_empty_labels
    assert Path(manifest.source).is_absolute()


def test_a_missing_or_foreign_manifest_is_refused(tmp_path: Path):
    other = tmp_path / "notes.yaml"
    other.write_text("title: notes\n", encoding="utf-8")

    for path in (tmp_path / "absent_manifest.yaml", other):
        with pytest.raises(ValueError, match="not a training export's manifest"):
            read_training_manifest(path)


def test_a_manifest_from_a_later_am_seg_is_refused(export: Path):
    export.write_text(
        export.read_text(encoding="utf-8").replace("version: 1", "version: 2"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="later version"):
        read_training_manifest(export)
