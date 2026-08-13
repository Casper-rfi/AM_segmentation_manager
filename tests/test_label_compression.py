"""Tests for recompressing existing on-disk label files."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

import numpy as np
import tifffile

from am_segmentation_manager.cli import main
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.label_compression import compress_labels


def _sparse_label(shape=(512, 512)) -> "np.ndarray":
    """A label mask that compresses well: mostly one value, one block of another."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[100:400, 100:400] = 2
    return arr


def test_compress_labels_shrinks_selected_label(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    label_path = labels_dir / "img1.tif"
    tifffile.imwrite(str(label_path), _sparse_label())
    original_size = label_path.stat().st_size

    record = ImageRecord(image_path=tmp_path / "img1.tif", labels=[label_path])
    results = compress_labels([record])

    assert len(results) == 1
    result = results[0]
    assert result.label_path == label_path
    # Pixel values must be unchanged even though the file shrank.
    assert np.array_equal(tifffile.imread(str(label_path)), _sparse_label())
    if result.applied:
        assert result.new_size < original_size
        assert label_path.stat().st_size == result.new_size


def test_compress_labels_dry_run_does_not_modify_file(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    label_path = labels_dir / "img1.tif"
    tifffile.imwrite(str(label_path), _sparse_label())
    original_bytes = label_path.read_bytes()

    record = ImageRecord(image_path=tmp_path / "img1.tif", labels=[label_path])
    results = compress_labels([record], dry_run=True)

    assert results[0].applied is False
    assert label_path.read_bytes() == original_bytes


def test_compress_labels_refuses_corrected_without_confirmation(tmp_path: Path):
    labels_dir = tmp_path / "labels" / "corrected"
    labels_dir.mkdir(parents=True)
    label_path = labels_dir / "img1.tif"
    tifffile.imwrite(str(label_path), _sparse_label())

    record = ImageRecord(image_path=tmp_path / "img1.tif", labels=[label_path], corrected=True)

    with pytest.raises(PermissionError):
        compress_labels([record])

    # File must be untouched after the refusal.
    assert np.array_equal(tifffile.imread(str(label_path)), _sparse_label())


def test_compress_labels_allows_corrected_with_confirmation(tmp_path: Path):
    labels_dir = tmp_path / "labels" / "corrected"
    labels_dir.mkdir(parents=True)
    label_path = labels_dir / "img1.tif"
    tifffile.imwrite(str(label_path), _sparse_label())

    record = ImageRecord(image_path=tmp_path / "img1.tif", labels=[label_path], corrected=True)
    results = compress_labels([record], allow_corrected=True)

    assert len(results) == 1
    assert np.array_equal(tifffile.imread(str(label_path)), _sparse_label())


def test_compress_labels_deduplicates_shared_labels(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    label_path = labels_dir / "img1.tif"
    tifffile.imwrite(str(label_path), _sparse_label())

    record_a = ImageRecord(image_path=tmp_path / "a.tif", labels=[label_path])
    record_b = ImageRecord(image_path=tmp_path / "b.tif", labels=[label_path])
    results = compress_labels([record_a, record_b])

    assert len(results) == 1


def test_cli_compress_labels_integration(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    stage = project / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    (sem / "img1.tif").write_text("img")
    tifffile.imwrite(str(labels / "img1.tif"), _sparse_label())
    original_size = (labels / "img1.tif").stat().st_size

    rc = main([
        "compress-labels",
        "--start-path", str(tmp_path),
        "--project", "projX",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--modality", "SEM",
    ])

    assert rc == 0
    assert np.array_equal(tifffile.imread(str(labels / "img1.tif")), _sparse_label())


def test_cli_compress_labels_dry_run_leaves_file_untouched(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    stage = project / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    (sem / "img1.tif").write_text("img")
    tifffile.imwrite(str(labels / "img1.tif"), _sparse_label())
    original_bytes = (labels / "img1.tif").read_bytes()

    rc = main([
        "compress-labels",
        "--start-path", str(tmp_path),
        "--project", "projX",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--modality", "SEM",
        "--dry-run",
    ])

    assert rc == 0
    assert (labels / "img1.tif").read_bytes() == original_bytes


def test_cli_compress_labels_refuses_corrected_without_flag(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    stage = project / "L1" / "stage1"
    sem_corr = stage / "sem" / "corrected"
    labels_corr = stage / "labels" / "corrected"
    sem_corr.mkdir(parents=True)
    labels_corr.mkdir(parents=True)

    (sem_corr / "img1.tif").write_text("img")
    tifffile.imwrite(str(labels_corr / "img1.tif"), _sparse_label())

    rc = main([
        "compress-labels",
        "--start-path", str(tmp_path),
        "--project", "projX",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--modality", "SEM",
        "--state", "corrected",
    ])

    assert rc == 2
