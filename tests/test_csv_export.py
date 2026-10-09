"""Tests for CSV training data export functionality."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.csv import export_training_csv, find_matching_label
from am_segmentation_manager.state import read_state, write_state
from conftest import write_label


def _declare_corrected(stage: Path, names: list[str]) -> None:
    """Declare a stage's groups corrected, as the window would."""
    write_state(read_state(stage), {f"labels/{name}": "corrected" for name in names})


def test_find_matching_label(tmp_path: Path):
    """A group has one label in one place, so there is nothing to choose."""
    stage = tmp_path / "stage"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    img = sem / "img1.tif"
    label = labels / "img1.tif"
    label.write_text("lbl")

    record = ImageRecord(image_path=img, labels=[label], state="corrected")
    assert find_matching_label(record) == label

    absent = ImageRecord(image_path=img, labels=[labels / "gone.tif"])
    assert find_matching_label(absent) is None


def test_export_training_csv_full_split(tmp_path: Path):
    root = tmp_path / "dataset"
    stage = root / "01-lamella" / "Adaptive_polish_1"
    sem = stage / "sem" / "corrected"
    labels = stage / "labels" / "corrected"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    records = []
    for i in range(10):
        img = sem / f"AP_img_{i:03d}_SEM.tif"
        lbl = labels / f"AP_img_{i:03d}_SEM.tif"
        img.write_text("img")
        write_label(lbl, "annotated")
        records.append(
            ImageRecord(
                image_path=img,
                labels=[lbl],
                state="corrected",
                modality="SEM",
            )
        )

    out_dir = tmp_path / "output"
    train_path, val_path = export_training_csv(
        records,
        start_path=root,
        output_dir=out_dir,
        prefix="test_dataset",
        train_ratio=0.8,
        seed=42,
    )

    assert train_path.exists()
    assert val_path is not None and val_path.exists()

    # Verify CSV format and row count (8 train, 2 val)
    with train_path.open("r", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
        assert reader[0] == ["image", "label"]
        assert len(reader) == 9  # 1 header + 8 rows
        # Check relative paths format
        assert reader[1][0].startswith("01-lamella/Adaptive_polish_1/sem/")

    with val_path.open("r", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
        assert reader[0] == ["image", "label"]
        assert len(reader) == 3  # 1 header + 2 rows


def test_export_training_csv_reproducibility(tmp_path: Path):
    root = tmp_path / "dataset"
    stage = root / "01-lamella" / "Adaptive_polish_1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    records = []
    for i in range(20):
        img = sem / f"AP_img_{i:03d}_SEM.tif"
        lbl = labels / f"AP_img_{i:03d}_SEM.tif"
        img.write_text("img")
        write_label(lbl, "annotated")
        records.append(
            ImageRecord(
                image_path=img,
                labels=[lbl],
                state="selected",
                modality="SEM",
            )
        )

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    train1, val1 = export_training_csv(records, root, out1, seed=123)
    train2, val2 = export_training_csv(records, root, out2, seed=123)

    assert train1.read_text() == train2.read_text()
    assert val1.read_text() == val2.read_text()


def test_cli_create_training_csv_integration(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    stage = project / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    for i in range(5):
        (sem / f"img{i}.tif").write_text("img")
        write_label(labels / f"img{i}.tif", "annotated")
    _declare_corrected(stage, [f"img{i}.tif" for i in range(5)])

    out_dir = tmp_path / "csv_output"

    rc = main([
        "create-training-csv",
        "--start-path", str(tmp_path),
        "--project", "projX",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--state", "corrected",
        # This project has SEM only, so a paired export would leave every FIB
        # cell blank; the shape under test here is the single-beam one.
        "--modality", "SEM",
        "-o", str(out_dir),
        "--output-prefix", "test_run",
    ])

    assert rc == 0
    train_csv = out_dir / "test_run_train.csv"
    val_csv = out_dir / "test_run_val.csv"
    assert train_csv.exists()
    assert val_csv.exists()

    content_train = train_csv.read_text()
    content_val = val_csv.read_text()
    assert "image,label" in content_train
    assert "image,label" in content_val
    combined = content_train + content_val
    assert "projX/L1/stage1/sem/img0.tif" in combined


def test_cli_create_training_csv_positional_scan_dir(tmp_path: Path):
    scan_dir = tmp_path / "20260728_CB_AM_bitmap_overnight"
    project = scan_dir / "AutoLamella-2026-07-28-10-31"
    stage = project / "01-ruling-ibex" / "Adaptive_polish_1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: AutoLamella-2026-07-28-10-31")

    name = "01-ruling-ibex_AP_img_013_SEM.tif"
    (sem / name).write_text("img")
    write_label(labels / name, "annotated")
    _declare_corrected(stage, [name])

    out_dir = tmp_path / "csv_out"

    rc = main([
        "create-training-csv",
        str(scan_dir),
        "--project", "all",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--state", "corrected",
        "-o", str(out_dir),
        "--split", "1.0",
    ])

    assert rc == 0
    train_csv = out_dir / "dataset_train.csv"
    assert train_csv.exists()
    content = train_csv.read_text()
    # Paths MUST start with AutoLamella-2026-07-28-10-31 relative to the scan directory 20260728_CB_AM_bitmap_overnight
    assert "AutoLamella-2026-07-28-10-31/01-ruling-ibex/Adaptive_polish_1/sem/01-ruling-ibex_AP_img_013_SEM.tif" in content
    assert "AutoLamella-2026-07-28-10-31/01-ruling-ibex/Adaptive_polish_1/labels/01-ruling-ibex_AP_img_013_SEM.tif" in content


def test_cli_create_training_csv_requires_explicit_state(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    # Call CLI without --state flag -> should fail (argparse SystemExit)
    with pytest.raises(SystemExit):
        main([
            "create-training-csv",
            "--start-path", str(tmp_path),
            "--project", "projX",
            "--lamella", "all",
            "--stage", "all",
            "--image", "all",
        ])


# ---------------------------------------------------------------------------
# Replacing an earlier export, and the manifest
# ---------------------------------------------------------------------------

def _labelled_records(root: Path, count: int = 10) -> list[ImageRecord]:
    """Return *count* SEM images of one stage, each with a label of its own."""
    stage = root / "01-lamella" / "Adaptive_polish_1"
    (stage / "sem").mkdir(parents=True)
    (stage / "labels").mkdir(parents=True)
    records = []
    for i in range(count):
        image = stage / "sem" / f"AP_img_{i:03d}_SEM.tif"
        label = stage / "labels" / f"AP_img_{i:03d}_SEM.tif"
        image.write_text("img")
        write_label(label, "annotated")
        records.append(
            ImageRecord(image_path=image, labels=[label], state="corrected", modality="SEM")
        )
    return records


def _manifest(out_dir: Path, prefix: str = "dataset") -> dict:
    import yaml

    return yaml.safe_load((out_dir / f"{prefix}_manifest.yaml").read_text())


def test_an_existing_validation_file_stops_the_export_before_anything_is_written(
    tmp_path: Path,
):
    """A split replaced by accident is gone with the seed that made it, so the
    refusal comes before a label is opened, not after the train file."""
    root = tmp_path / "dataset"
    records = _labelled_records(root)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "dataset_val.csv").write_text("earlier")
    opened = []

    with pytest.raises(FileExistsError, match="dataset_val.csv"):
        export_training_csv(
            records, root, out_dir, is_empty=lambda path: opened.append(path) or False
        )

    assert opened == []
    assert (out_dir / "dataset_val.csv").read_text() == "earlier"
    assert sorted(path.name for path in out_dir.iterdir()) == ["dataset_val.csv"]


def test_overwriting_replaces_an_earlier_training_set(tmp_path: Path):
    root = tmp_path / "dataset"
    records = _labelled_records(root)
    out_dir = tmp_path / "output"
    export_training_csv(records, root, out_dir, seed=1)

    export_training_csv(records, root, out_dir, seed=2, overwrite=True)

    assert _manifest(out_dir)["split"]["seed"] == 2


def test_the_manifest_counts_what_the_files_hold(tmp_path: Path):
    root = tmp_path / "dataset"
    records = _labelled_records(root)
    out_dir = tmp_path / "output"

    train, val = export_training_csv(
        records, root, out_dir, prefix="run", train_ratio=0.8, seed=42,
        scope="modality SEM", user="Ada Lovelace",
    )

    manifest = _manifest(out_dir, "run")
    rows = {
        "train": len(train.read_text().splitlines()) - 1,
        "validation": len(val.read_text().splitlines()) - 1,
    }
    assert manifest["files"] == {"train": "run_train.csv", "validation": "run_val.csv"}
    assert manifest["counts"]["train"] == rows["train"] == 8
    assert manifest["counts"]["validation"] == rows["validation"] == 2
    assert manifest["counts"]["rows"] == manifest["counts"]["images"] == 10
    assert manifest["counts"]["images_left_out"] == 0
    assert manifest["split"] == {"train_ratio": 0.8, "seed": 42, "keep_together": "acquisition"}
    assert manifest["columns"] == ["image", "label"]
    assert manifest["scope"] == "modality SEM"
    assert manifest["written_by"] == "Ada Lovelace"
    assert manifest["collection"] == "../dataset"
    assert "source" not in manifest


def test_an_unseeded_split_records_no_seed(tmp_path: Path):
    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"

    export_training_csv(_labelled_records(root), root, out_dir, seed=None)

    assert _manifest(out_dir)["split"]["seed"] is None


def test_a_full_training_share_is_not_stopped_by_a_validation_file_it_would_not_write(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "dataset_val.csv").write_text("earlier")

    _, val = export_training_csv(_labelled_records(root), root, out_dir, train_ratio=1.0)

    assert val is None
    assert _manifest(out_dir)["files"]["validation"] is None
    assert (out_dir / "dataset_val.csv").read_text() == "earlier"


def test_cli_refuses_to_replace_a_training_set_without_overwrite(tmp_path: Path):
    project = tmp_path / "projX"
    stage = project / "L1" / "stage1"
    (stage / "sem").mkdir(parents=True)
    (stage / "labels").mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: projX")
    for i in range(5):
        (stage / "sem" / f"img{i}.tif").write_text("img")
        write_label(stage / "labels" / f"img{i}.tif", "annotated")
    _declare_corrected(stage, [f"img{i}.tif" for i in range(5)])
    out_dir = tmp_path / "csv_output"
    command = [
        "create-training-csv", "--start-path", str(tmp_path), "--project", "projX",
        "--lamella", "all", "--stage", "all", "--image", "all",
        "--state", "corrected", "--modality", "SEM", "-o", str(out_dir),
    ]

    assert main(command) == 0
    first = (out_dir / "dataset_train.csv").read_text()

    assert main([*command, "--seed", "7"]) == 2
    assert (out_dir / "dataset_train.csv").read_text() == first

    assert main([*command, "--seed", "7", "--overwrite"]) == 0
    manifest = _manifest(out_dir)
    assert manifest["split"]["seed"] == 7
    assert "--state corrected" in manifest["scope"]
