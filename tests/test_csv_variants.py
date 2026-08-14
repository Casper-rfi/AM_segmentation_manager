"""Exporting training data from images that share a label.

Several imaging conditions of one acquisition are near-identical images sharing
one label. That makes two things matter for a training export: how many of them
to include by default, and -- if several are included -- that they never end up
on opposite sides of the train/validation split, which would score the model on
scenes it has effectively already seen.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.cli import main
from am_segmentation_manager.operations.csv import split_pairs

SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


def make_project(root: Path, acquisitions: int = 1) -> Path:
    """Three SEM conditions sharing one label per acquisition, plus a FIB image."""
    project = root / "project"
    lamella = project / "lam_010"
    for directory in (*SEM_VARIANTS, "FIB", "labels", "labels_FIB"):
        (lamella / directory).mkdir(parents=True, exist_ok=True)
        for number in range(acquisitions):
            (lamella / directory / f"img_{number:04d}.tif").write_bytes(b"")
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
    return project


def export(tmp_path: Path, *extra: str, split: str = "1.0") -> list[dict]:
    output = tmp_path / "out"
    assert main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "-o", str(output), "--split", split, "--include-empty-labels", *extra,
    ]) == 0
    with (output / "dataset_train.csv").open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Which conditions are exported
# ---------------------------------------------------------------------------

def test_one_condition_per_scene_is_exported_by_default(tmp_path: Path):
    """Otherwise a dataset quietly gains several near-identical copies."""
    make_project(tmp_path)

    rows = export(tmp_path)

    assert len(rows) == 2  # one SEM, one FIB
    assert not any("fi_1" in row["image"] or "fi_4" in row["image"] for row in rows)


def test_the_default_works_where_no_unsuffixed_directory_exists(tmp_path: Path):
    """The oldest layout has no plain sem/: every SEM set is a variant.

    Filtering for "the base directory" would select no SEM image at all, which
    is the layout this whole exercise exists to support.
    """
    make_project(tmp_path)

    rows = export(tmp_path)

    sem_rows = [row for row in rows if "SEM" in row["image"]]
    assert len(sem_rows) == 1
    assert "SEM_8/" in sem_rows[0]["image"]


def test_every_condition_can_be_exported_deliberately(tmp_path: Path):
    """Five conditions of one scene are five valid training pairs."""
    make_project(tmp_path)

    rows = export(tmp_path, "--conditions", "all")

    assert len(rows) == 4  # three SEM conditions, one FIB
    assert sum("SEM_8" in row["image"] for row in rows) == 3


def test_a_single_named_condition_can_be_exported(tmp_path: Path):
    """The command default must not be appended to an explicit choice."""
    make_project(tmp_path)

    rows = export(tmp_path, "--variant", "8_fi_1")

    assert len(rows) == 1
    assert "SEM_8_fi_1" in rows[0]["image"]


def test_state_changes_still_cover_every_condition_by_default(tmp_path: Path):
    """The export default must not leak into commands that follow the label."""
    project = make_project(tmp_path)

    assert main([
        "deselect", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--modality", "SEM",
    ]) == 0

    for name in SEM_VARIANTS:
        assert (project / "lam_010" / name / "deselected" / "img_0000.tif").exists()


# ---------------------------------------------------------------------------
# The train/validation split
# ---------------------------------------------------------------------------

def test_images_sharing_a_label_stay_on_the_same_side(tmp_path: Path):
    """Splitting them would put near-identical images in train and validation."""
    pairs = [(f"sem_{condition}/img_{n}.tif", f"labels/img_{n}.tif")
             for n in range(10) for condition in range(3)]

    train, val = split_pairs(pairs, train_ratio=0.8, seed=42)

    train_labels = {label for _, label in train}
    val_labels = {label for _, label in val}
    assert train_labels & val_labels == set()


def test_every_pair_is_placed_exactly_once(tmp_path: Path):
    pairs = [(f"sem_{c}/img_{n}.tif", f"labels/img_{n}.tif")
             for n in range(10) for c in range(3)]

    train, val = split_pairs(pairs, train_ratio=0.8, seed=42)

    assert sorted(train + val) == sorted(pairs)


def test_the_split_ratio_is_respected_within_a_group(tmp_path: Path):
    pairs = [(f"sem_{c}/img_{n}.tif", f"labels/img_{n}.tif")
             for n in range(10) for c in range(3)]

    train, val = split_pairs(pairs, train_ratio=0.8, seed=42)

    assert 0.7 <= len(train) / len(pairs) <= 0.9


def test_unshared_labels_split_exactly_as_before(tmp_path: Path):
    """One image per label: grouping must change nothing."""
    pairs = [(f"sem/img_{n}.tif", f"labels/img_{n}.tif") for n in range(10)]

    train, val = split_pairs(pairs, train_ratio=0.8, seed=42)

    assert len(train) == 8
    assert len(val) == 2


def test_a_full_training_split_keeps_everything(tmp_path: Path):
    pairs = [(f"sem/img_{n}.tif", f"labels/img_{n}.tif") for n in range(5)]
    assert split_pairs(pairs, train_ratio=1.0) == (pairs, [])


def test_a_full_validation_split_keeps_everything(tmp_path: Path):
    pairs = [(f"sem/img_{n}.tif", f"labels/img_{n}.tif") for n in range(5)]
    assert split_pairs(pairs, train_ratio=0.0) == ([], pairs)


def test_the_split_is_reproducible(tmp_path: Path):
    pairs = [(f"sem_{c}/img_{n}.tif", f"labels/img_{n}.tif")
             for n in range(10) for c in range(3)]

    assert split_pairs(pairs, seed=7) == split_pairs(pairs, seed=7)


def test_an_end_to_end_export_does_not_leak_a_scene_across_the_split(tmp_path: Path):
    make_project(tmp_path, acquisitions=10)
    output = tmp_path / "out"

    assert main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--conditions", "all", "--include-empty-labels",
        "-o", str(output), "--split", "0.8",
    ]) == 0

    def labels(name: str) -> set[str]:
        with (output / name).open() as handle:
            return {row["label"] for row in csv.DictReader(handle)}

    assert labels("dataset_train.csv") & labels("dataset_val.csv") == set()
