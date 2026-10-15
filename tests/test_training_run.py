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
from types import SimpleNamespace

import pytest
import yaml

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.training_export import export_kfold_csv, export_training_csv
from am_segmentation_manager.operations.training_manifest import read_training_manifest
from am_segmentation_manager.operations.training_run import (
    NEW_MODEL,
    REFINE,
    RUN_RECORD_FILENAME,
    TrainingRunRefused,
    apply_training_run,
    check_pairs,
    default_settings,
    find_run_directories,
    label_report,
    plan_training_run,
    read_run_record,
    relocate_pairs,
    resized_shape,
    shape_problem,
)
from am_segmentation_manager.progress import Cancellation, OperationCancelled
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
        ({"frozen_epochs": 100}, "fewer than the maximum number of epochs"),
        ({"frozen_epochs": -1}, "fewer than the maximum number of epochs"),
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


def test_a_manifest_names_no_folder_of_this_machine(tmp_path: Path, monkeypatch):
    """Whether the export was given the collection in full or as ".", from wherever it was run."""
    root = tmp_path / "collection"
    records = _records(root)
    export_training_csv(records, root, tmp_path / "full", prefix="set")
    monkeypatch.chdir(root)
    export_training_csv(records, Path("."), tmp_path / "typed", prefix="set")
    monkeypatch.chdir(tmp_path)

    for export in ("full", "typed"):
        folder = tmp_path / export
        written = "".join(path.read_text(encoding="utf-8") for path in folder.iterdir())
        assert str(tmp_path) not in written and str(tmp_path.resolve()) not in written
        assert "collection: ../collection" in (folder / "set_manifest.yaml").read_text(encoding="utf-8")
        assert len(_plan(folder / "set_manifest.yaml").train) == 8


def test_an_export_moved_with_its_collection_still_finds_it(export: Path, tmp_path: Path):
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for name in ("collection", "export"):
        (tmp_path / name).rename(moved / name)

    plan = _plan(moved / "export" / "set_manifest.yaml", training_directory=tmp_path / "runs")

    assert plan.source == (moved / "collection").resolve()


def test_an_export_moved_away_from_its_collection_needs_it_named(export: Path, tmp_path: Path):
    apart = tmp_path / "sent" / "deeper"
    apart.mkdir(parents=True)
    for file in list(export.parent.iterdir()):
        file.rename(apart / file.name)
    manifest = apart / export.name

    assert "are not under" in _refusal(manifest, training_directory=tmp_path / "runs")
    named = _plan(manifest, training_directory=tmp_path / "runs", start_path=tmp_path / "collection")
    assert len(named.train) == 8


def test_the_folder_named_is_taken_before_the_one_the_manifest_says(export: Path, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()

    assert "are not under" in _refusal(export, start_path=other)


def _older(export: Path, source: str) -> Path:
    """Rewrite a manifest as one written before it said where the collection is from itself."""
    document = yaml.safe_load(export.read_text(encoding="utf-8"))
    del document["collection"]
    document["source"] = source
    export.write_text(yaml.safe_dump(document), encoding="utf-8")
    return export


def test_an_older_manifest_recording_the_folder_in_full_still_prepares_a_run(
    export: Path, tmp_path: Path
):
    older = _older(export, str(tmp_path / "collection"))

    assert read_training_manifest(older).collection == tmp_path / "collection"
    assert len(_plan(older).train) == 8


def test_an_older_manifest_recording_the_folder_as_typed_needs_it_named(
    export: Path, tmp_path: Path
):
    """A "." said where the command was run, which nothing recorded."""
    older = _older(export, ".")

    assert "does not say where the collection is" in _refusal(older)
    assert len(_plan(older, start_path=tmp_path / "collection").train) == 8


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
    assert label_report(_first_label(export), 5)[0] is None


def test_checking_reports_each_label_once_and_stops_before_the_next(tmp_path: Path):
    """Opening every label of a large set takes minutes, so the window shows a bar and Stop."""
    images = [tmp_path / f"{name}.tif" for name in ("a_base", "a_other", "b")]
    shared, own = tmp_path / "a_label.tif", tmp_path / "b_label.tif"
    for path in (*images, shared, own):
        path.write_text("x")
    # Two imaging conditions naming one label: it is opened, and counted, once.
    pairs = [(images[0], shared), (images[1], shared), (images[2], own)]
    manifest = SimpleNamespace(include_empty_labels=False)
    reported: list[tuple[int, int]] = []
    stop = Cancellation()

    def opened(done: int, total: int) -> None:
        reported.append((done, total))
        if stop_after_first:
            stop.request()

    stop_after_first = False
    check_pairs(
        pairs, tmp_path, manifest, purpose="trained on",
        inspect_label=lambda path, classes: None, on_progress=opened, cancellation=stop,
    )
    assert reported == [(1, 2), (2, 2)]

    reported.clear()
    stop_after_first = True
    with pytest.raises(OperationCancelled):
        check_pairs(
            pairs, tmp_path, manifest, purpose="trained on",
            inspect_label=lambda path, classes: None, on_progress=opened, cancellation=stop,
        )
    assert reported == [(1, 2)]


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


def test_the_record_reads_back_how_many_pairs_the_run_was_prepared_with(export: Path):
    directory = apply_training_run(_plan(export))

    record = read_run_record(directory)

    assert (record.train_count, record.validation_count) == (8, 2)


def test_a_record_that_does_not_say_how_many_pairs_reads_back_as_not_knowing(export: Path):
    """Nothing is counted in its place: the lists may have been edited since."""
    directory = apply_training_run(_plan(export))
    path = directory / RUN_RECORD_FILENAME
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    del document["training_set"]
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    record = read_run_record(directory)

    assert (record.train_count, record.validation_count) == (None, None)


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
    assert manifest.collection == export.parent / "../collection"


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


# -- finding prepared runs ----------------------------------------------------


def test_a_prepared_run_is_found_as_itself_and_from_its_training_directory(export: Path):
    directory = apply_training_run(_plan(export))
    (directory.parent / "notes").mkdir()

    assert find_run_directories(directory) == [directory]
    assert find_run_directories(directory.parent) == [directory]
    assert find_run_directories(export.parent) == []


def test_prepared_runs_are_found_through_folders_only_as_deep_as_asked(export: Path):
    directory = apply_training_run(_plan(export))
    above = directory.parent.parent

    assert find_run_directories(above) == []
    assert find_run_directories(above, depth=1) == [directory]


# -- one shape to a batch -----------------------------------------------------

def _oblong_label(path: Path) -> None:
    """Rewrite the label at *path* annotated and three wide to two high."""
    import numpy as np
    import tifffile

    array = np.zeros((16, 24), dtype=np.uint8)
    array[4:12, 4:12] = 2
    tifffile.imwrite(str(path), array)


def _a_training_label(export: Path) -> Path:
    """Return a label the export's training list names, whichever the split put there."""
    row = _rows(export.parent / "set_train.csv")[0]
    return export.parent.parent / "collection" / row["label"]


def test_an_image_is_scaled_by_its_longer_side_keeping_its_proportions():
    assert resized_shape((1024, 1536)) == (1024, 1536)
    assert resized_shape((2048, 3072)) == (1024, 1536)
    assert resized_shape((1000, 1000)) == (1536, 1536)


def test_images_of_two_proportions_are_refused_before_a_run_is_prepared(export: Path):
    """A batch is one tensor, so a square image among oblong ones stops training part way."""
    label = _a_training_label(export)
    _oblong_label(label)

    said = _refusal(export)

    assert "The training list holds images of 2 proportions" in said
    assert "7 that become 1536 x 1536, 1 that become 1536 x 1024" in said
    assert label.name in said
    assert not (export.parent.parent / "runs").exists()


def test_a_batch_of_one_holds_images_of_any_proportions(export: Path):
    _oblong_label(_a_training_label(export))

    plan = _plan(export, settings=replace(NEW_MODEL, training_batch_size=1))

    assert len(plan.train) + len(plan.validation) == 10


def test_images_of_one_proportion_at_two_sizes_are_one_shape():
    pairs = [(Path("a.tif"), Path("a_label.tif")), (Path("b.tif"), Path("b_label.tif"))]
    shapes = {Path("a_label.tif"): (1024, 1536), Path("b_label.tif"): (2048, 3072)}

    assert shape_problem("training", pairs, shapes, batch_size=6) is None


# -- what the validation images are of ------------------------------------------


def _described_export(tmp_path: Path, written: bool = True) -> Path:
    """Return a two-pair export's manifest naming an images file, written or not."""
    from am_segmentation_manager.io.tables import write_csv
    from test_training_process import export_of_placeholders

    manifest = export_of_placeholders(tmp_path)
    if written:
        write_csv(
            manifest.parent / "set_images.csv", ("split", "image", "microscope_model"),
            [("train", "a.tif", "Arctis"), ("validation", "b.tif", "Hydra")],
        )
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "validation: set_val.csv", "validation: set_val.csv\n  images: set_images.csv"
        ),
        encoding="utf-8",
    )
    return manifest


def _prepared(tmp_path: Path, manifest: Path) -> Path:
    return apply_training_run(plan_training_run(
        manifest, tmp_path / "runs", "round01", NEW_MODEL,
        inspect_label=lambda path, classes: None,
    ))


def test_what_the_validation_images_are_of_is_kept_with_the_run(tmp_path: Path):
    """A run outlives the export it was made from, and is looked at long after."""
    run = _prepared(tmp_path, _described_export(tmp_path))

    record = read_run_record(run)
    with record.validation_images.open(newline="", encoding="utf-8") as handle:
        described = list(csv.DictReader(handle))
    with record.validation_list.open(newline="", encoding="utf-8") as handle:
        listed = list(csv.DictReader(handle))

    assert described == [{"image": "b.tif", "split": "validation", "microscope_model": "Hydra"}]
    assert len(described) == len(listed)


def test_an_export_that_described_no_images_leaves_a_run_without_them(tmp_path: Path):
    from test_training_process import export_of_placeholders

    run = _prepared(tmp_path, export_of_placeholders(tmp_path))

    assert read_run_record(run).validation_images is None
    assert not (run / "val_images.csv").exists()


def test_an_images_file_that_has_gone_does_not_stop_a_run_being_prepared(tmp_path: Path):
    run = _prepared(tmp_path, _described_export(tmp_path, written=False))

    assert read_run_record(run).validation_images is None


VM = "/shared/data/FIBSEM/labelled"
VM_ROW = (f"{VM}/proj/lam_005/SEM/img_0000.tif", f"{VM}/proj/lam_005/sem_labels/img_0000.tif")


def test_a_run_s_lists_are_found_under_the_scanned_folder_where_theirs_is_gone(tmp_path: Path):
    """Nothing is on disk here: where a row goes is decided on its text alone."""
    here = tmp_path / "collection"

    assert relocate_pairs([VM_ROW], Path(VM), here) == (
        (here / "proj/lam_005/SEM/img_0000.tif", here / "proj/lam_005/sem_labels/img_0000.tif"),
    )


def test_a_path_outside_the_recorded_collection_is_left_as_written(tmp_path: Path):
    """A neighbour whose name only starts as the collection's does is outside it too."""
    elsewhere = ("/scratch/other/img.tif", f"{VM}_old/proj/img.tif")

    assert relocate_pairs([elsewhere], VM, tmp_path) == (
        (Path(elsewhere[0]), Path(elsewhere[1])),
    )


@pytest.mark.parametrize(
    ("recorded", "row"),
    [
        (r"C:\Data\labelled", (r"C:\Data\labelled\proj\SEM\a.tif", r"C:\Data\labelled\proj\sem_labels\a.tif")),
        # As Windows reads the record of a run prepared on Linux, and its list.
        (r"\shared\labelled", ("/shared/labelled/proj/SEM/a.tif", "/shared/labelled/proj/sem_labels/a.tif")),
        ("/shared/labelled", ("/shared/labelled/proj/SEM/a.tif", "/shared/labelled/proj/sem_labels/a.tif")),
    ],
)
def test_a_list_written_on_windows_is_found_on_a_mac_and_the_other_way(
    tmp_path: Path, recorded: str, row: tuple[str, str]
):
    assert relocate_pairs([row], recorded, tmp_path) == (
        (tmp_path / "proj" / "SEM" / "a.tif", tmp_path / "proj" / "sem_labels" / "a.tif"),
    )


def test_a_posix_folder_differing_only_in_case_is_another_folder(tmp_path: Path):
    row = ("/shared/Labelled/proj/a.tif", "/shared/Labelled/proj/b.tif")

    assert relocate_pairs([row], "/shared/labelled", tmp_path) == ((Path(row[0]), Path(row[1])),)
