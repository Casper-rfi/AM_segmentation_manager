"""Downstream operations on data where several images share one label.

Compression and CSV re-checking were written before shared labels existed. The
tests here pin that they behave correctly on such data rather than assuming the
phase 1 refactor carried them across: a shared label must be processed once,
not once per image that references it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from am_segmentation_manager.operations.selection import deselect_records
from am_segmentation_manager.operations.training_export_check import check_csv_paths
from am_segmentation_manager.scanner import Scanner

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile", "cv2")
)

SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


def make_project(root: Path, annotated: bool = True) -> Path:
    """Three SEM conditions sharing one label, plus an independent FIB image."""
    project = root / "project"
    lamella = project / "lam_010"
    for directory in (*SEM_VARIANTS, "FIB"):
        (lamella / directory).mkdir(parents=True)
        _write(lamella / directory / "img_0004.tif", annotated=False)
    for directory in ("labels", "labels_FIB"):
        (lamella / directory).mkdir(parents=True)
        _write(lamella / directory / "img_0004.tif", annotated=annotated)
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
    return project


def _write(path: Path, annotated: bool) -> None:
    if not HAS_IMAGE_STACK:
        path.write_bytes(b"")
        return
    import numpy as np
    import tifffile

    array = np.zeros((32, 48), dtype=np.uint8)
    if annotated:
        array[:10] = 2
    tifffile.imwrite(str(path), array)


def scan(root: Path) -> list:
    index = Scanner(root).build_index()
    return [
        record
        for project in index.all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]


# ---------------------------------------------------------------------------
# Label compression
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_IMAGE_STACK, reason="optional image libs not installed")
def test_a_shared_label_is_compressed_once_not_once_per_image(tmp_path: Path):
    """Three records reference one file; compressing it three times is waste."""
    from am_segmentation_manager.operations.label_compression import compress_labels

    make_project(tmp_path)
    records = [r for r in scan(tmp_path) if r.modality == "SEM"]
    assert len(records) == len(SEM_VARIANTS)

    results = compress_labels(records, dry_run=True)

    assert len(results) == 1
    assert results[0].label_path.parent.name == "labels"


@pytest.mark.skipif(not HAS_IMAGE_STACK, reason="optional image libs not installed")
def test_compression_still_covers_every_distinct_label(tmp_path: Path):
    from am_segmentation_manager.operations.label_compression import compress_labels

    make_project(tmp_path)

    results = compress_labels(scan(tmp_path), dry_run=True)

    assert {r.label_path.parent.name for r in results} == {"labels", "labels_FIB"}


# ---------------------------------------------------------------------------
# Re-checking an exported CSV
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    lines = ["image,label", *(f"{image},{label}" for image, label in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_an_exported_row_survives_a_state_change(tmp_path: Path):
    """The reason state stopped being a folder: curating an image used to
    rewrite the very path a training CSV had already recorded."""
    make_project(tmp_path)
    _write_csv(tmp_path / "dataset.csv", [
        ("project/lam_010/SEM_8/img_0004.tif", "project/lam_010/labels/img_0004.tif"),
    ])
    csv_path = tmp_path / "dataset.csv"

    records = [r for r in scan(tmp_path) if r.image_path.parent.name == "SEM_8"]
    deselect_records(records)

    results = check_csv_paths(csv_path, tmp_path)

    assert len(results) == 1
    assert results[0].ok
    assert results[0].columns["image"].status == "ok"
    assert results[0].columns["label"].status == "ok"


def test_a_row_naming_a_pulled_in_variant_survives_too(tmp_path: Path):
    """The whole group changed state, and not one of its files moved."""
    make_project(tmp_path)
    _write_csv(tmp_path / "dataset.csv", [
        ("project/lam_010/SEM_8_fi_4/img_0004.tif", "project/lam_010/labels/img_0004.tif"),
    ])
    csv_path = tmp_path / "dataset.csv"

    deselect_records([r for r in scan(tmp_path) if r.image_path.parent.name == "SEM_8"])

    assert check_csv_paths(csv_path, tmp_path)[0].ok


def test_csv_check_reports_a_genuinely_missing_file(tmp_path: Path):
    make_project(tmp_path)
    csv_path = _write_csv(tmp_path / "dataset.csv", [
        ("project/lam_010/SEM_8/absent.tif", "project/lam_010/labels/absent.tif"),
    ])

    results = check_csv_paths(csv_path, tmp_path)

    assert results[0].columns["image"].status == "missing"
    assert results[0].columns["label"].status == "missing"


def test_csv_check_passes_an_untouched_export(tmp_path: Path):
    make_project(tmp_path)
    csv_path = _write_csv(tmp_path / "dataset.csv", [
        ("project/lam_010/SEM_8/img_0004.tif", "project/lam_010/labels/img_0004.tif"),
    ])

    results = check_csv_paths(csv_path, tmp_path)

    assert results[0].ok is True
