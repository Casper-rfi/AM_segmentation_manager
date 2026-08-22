"""Tests for CSV training data export functionality."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.csv import export_training_csv, find_matching_label
from am_segmentation_manager.state import read_state, write_state


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
        lbl.write_text("lbl")
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
        lbl.write_text("lbl")
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
        (labels / f"img{i}.tif").write_text("lbl")
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
    (labels / name).write_text("lbl")
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
