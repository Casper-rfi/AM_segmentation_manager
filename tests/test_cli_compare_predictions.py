"""``am-seg compare-predictions``: a sheet of several runs' predictions, nothing else written."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.io.tables import write_csv
from am_segmentation_manager.operations import preview_process
from am_segmentation_manager.operations.preview_process import PreviewOutcome
from am_segmentation_manager.operations.training_manifest import write_training_manifest
from am_segmentation_manager.training.preparation import (
    NEW_MODEL,
    apply_training_run,
    plan_training_run,
)

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("matplotlib")
pytest.importorskip("cv2")

#: Each image as ``(name, side, microscope)``: two to train on, three to validate on.
IMAGES = (
    ("t1", "train", "Arctis"), ("t2", "train", "Hydra"),
    ("v1", "validation", "Arctis"), ("v2", "validation", "Hydra"), ("v3", "validation", "Hydra"),
)


def _export(tmp_path: Path) -> Path:
    """Return the manifest of an export of real images, each described by its microscope."""
    collection, export = tmp_path / "collection", tmp_path / "export"
    collection.mkdir()
    export.mkdir()
    for name, _, _ in IMAGES:
        tifffile.imwrite(collection / f"{name}.tif", np.full((40, 60), 90, dtype=np.uint8))
        label = np.zeros((40, 60), dtype=np.uint8)
        label[10:20, 10:30] = 2
        tifffile.imwrite(collection / f"{name}_label.tif", label)
    for side, file in (("train", "set_train.csv"), ("validation", "set_val.csv")):
        write_csv(export / file, ("image", "label"), [
            (f"{name}.tif", f"{name}_label.tif") for name, where, _ in IMAGES if where == side
        ])
    write_csv(export / "set_images.csv", ("split", "image", "microscope_model"), [
        (side, f"{name}.tif", microscope) for name, side, microscope in IMAGES
    ])
    return write_training_manifest(
        export / "set_manifest.yaml", source=collection, scope="",
        files={
            "train": export / "set_train.csv", "validation": export / "set_val.csv",
            "images": export / "set_images.csv",
        },
        columns=("image", "label"), train_ratio=0.5, seed=1, keep_together="lamella",
        conditions="primary", include_empty_labels=False, counts={},
    )


def _run(tmp_path: Path, manifest: Path, name: str, weights: bool = True) -> Path:
    """Return a prepared run that has validated two epochs and kept the second's weights."""
    from test_training_controller import _write_epochs

    run = apply_training_run(plan_training_run(
        manifest, tmp_path / "runs", name, NEW_MODEL, inspect_label=lambda path, classes: None,
    ))
    written = _write_epochs(run, 2, run_id=f"{name}_val")
    if weights:
        with zipfile.ZipFile(written / f"{name}_val_epoch002.pth", "w") as archive:
            archive.writestr("weights/data.pkl", b"weights")
    return run


@pytest.fixture
def predicted(monkeypatch):
    """Stand in for the child: every model predicts class 1 everywhere, at its own resolution."""
    asked: list[tuple[list[Path], list[Path], str]] = []

    def preview(images, weights, device="auto", **_):
        asked.append((list(images), list(weights), device))
        return PreviewOutcome({
            (model, image): np.ones((16, 24), dtype=np.uint8) for model in weights for image in images
        })

    monkeypatch.setattr(preview_process, "preview_out_of_process", preview)
    return asked


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    manifest = _export(tmp_path)
    _run(tmp_path, manifest, "first")
    _run(tmp_path, manifest, "second")
    return tmp_path / "runs"


def _files(folder: Path) -> set[Path]:
    return {path for path in folder.rglob("*") if path.is_file()}


def test_a_sheet_is_written_and_nothing_else_anywhere(runs: Path, tmp_path: Path, predicted):
    before = _files(tmp_path)
    sheet = tmp_path / "sheet.png"

    assert main(["compare-predictions", str(runs), "--output", str(sheet), "--count", "2"]) == 0

    assert sheet.read_bytes().startswith(b"\x89PNG")
    assert _files(tmp_path) - before == {sheet}


def test_every_run_s_best_checkpoint_predicts_the_same_validation_images(
    runs: Path, tmp_path: Path, predicted
):
    main(["compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png")])

    images, weights, _ = predicted[0]
    assert {image.name for image in images} == {"v1.tif", "v2.tif", "v3.tif"}
    assert sorted(model.name for model in weights) == ["first_val_epoch002.pth", "second_val_epoch002.pth"]


def test_images_are_chosen_by_what_the_export_said_they_are_of(runs: Path, tmp_path: Path, predicted):
    main([
        "compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png"),
        "--microscope", "Hydra", "--count", "4",
    ])

    assert {image.name for image in predicted[0][0]} == {"v2.tif", "v3.tif"}


def test_a_value_no_image_has_is_refused_with_those_there_are(
    runs: Path, tmp_path: Path, predicted, capsys
):
    status = main([
        "compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png"),
        "--microscope", "Krios",
    ])

    assert status == 2 and not predicted
    assert "there is: Arctis, Hydra" in capsys.readouterr().out


def test_more_images_than_can_be_judged_side_by_side_are_refused(
    runs: Path, tmp_path: Path, predicted, capsys
):
    status = main([
        "compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png"), "--count", "5",
    ])

    assert status == 2 and not predicted
    assert "--count is from 1 to 4" in capsys.readouterr().out


def test_the_same_seed_draws_the_same_images(runs: Path, tmp_path: Path, predicted):
    for name in ("one.png", "two.png"):
        main([
            "compare-predictions", str(runs), "--output", str(tmp_path / name),
            "--count", "1", "--seed", "5",
        ])

    assert predicted[0][0] == predicted[1][0]


def test_a_run_with_no_weights_is_said_and_the_others_still_drawn(
    runs: Path, tmp_path: Path, predicted, capsys
):
    _run(tmp_path, tmp_path / "export" / "set_manifest.yaml", "third", weights=False)

    status = main(["compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png")])

    assert status == 0
    assert "third: epoch 2, run's best: its weights are not on disk" in capsys.readouterr().out
    assert len(predicted[0][1]) == 2


def test_an_existing_sheet_is_not_replaced_unless_asked(runs: Path, tmp_path: Path, predicted):
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"mine")

    assert main(["compare-predictions", str(runs), "--output", str(sheet)]) == 2
    assert sheet.read_bytes() == b"mine"
    assert main(["compare-predictions", str(runs), "--output", str(sheet), "--overwrite"]) == 0


def test_a_child_that_could_not_predict_writes_no_sheet(runs: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        preview_process, "preview_out_of_process",
        lambda *_, **__: PreviewOutcome(error="RuntimeError: the ml extra is not installed"),
    )
    sheet = tmp_path / "sheet.png"

    assert main(["compare-predictions", str(runs), "--output", str(sheet)]) == 1
    assert not sheet.exists()


def test_the_label_can_be_given_a_column_of_its_own(runs: Path, tmp_path: Path, predicted):
    sheet = tmp_path / "sheet.png"

    assert main(["compare-predictions", str(runs), "--output", str(sheet), "--with-label"]) == 0



def test_runs_trained_on_another_machine_are_drawn_from_the_collection_named(
    runs: Path, tmp_path: Path, predicted
):
    here = tmp_path / "moved"
    (tmp_path / "collection").rename(here)
    sheet = tmp_path / "sheet.png"

    assert main(["compare-predictions", str(runs), "--output", str(sheet)]) == 1
    assert predicted == [] and not sheet.exists()
    assert main([
        "compare-predictions", str(runs), "--output", str(sheet), "--start-path", str(here),
    ]) == 0

    images, _, _ = predicted[-1]
    assert images and all(image.is_relative_to(here) and image.is_file() for image in images)


def test_a_collection_that_is_no_folder_is_refused(runs: Path, tmp_path: Path, predicted):
    status = main([
        "compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png"),
        "--start-path", str(tmp_path / "nowhere"),
    ])

    assert status == 2 and not predicted


def test_images_not_on_this_machine_are_said_once_before_anything_predicts(
    runs: Path, tmp_path: Path, predicted, capsys
):
    (tmp_path / "collection").rename(tmp_path / "moved")

    main(["compare-predictions", str(runs), "--output", str(tmp_path / "sheet.png")])

    said = capsys.readouterr().out
    assert said.count("on this machine") == 1 and ".tif" not in said
    assert "--start-path" in said and str(tmp_path / "collection") in said
    assert predicted == []
