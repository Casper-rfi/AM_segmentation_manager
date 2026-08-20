"""Tests for stage and image discovery inside Scanner.

These tests create small synthetic project structures under pytest's tmp_path
and assert the scanner builds the expected Project -> Lamella -> Stage -> ImageRecord
hierarchy. The tests do not require heavy image dependencies.
"""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.scanner import Scanner


def _write_file(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_stage_discovery_detects_sem_and_fib(tmp_path: Path):
    # create a minimal project structure with one lamella and one stage
    base = tmp_path / "workspace"
    project_root = base / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")

    lamella = project_root / "01-test-lamella"
    stage = lamella / "Adaptive_polish_1"

    sem = stage / "sem"
    fib = stage / "fib"
    _write_file(sem / "img_sem.tif")
    _write_file(fib / "img_fib.tif")

    scanner = Scanner(base)
    idx = scanner.build_index()

    projects = idx.all_projects()
    assert len(projects) == 1
    project = projects[0]
    assert project.name == project_root.name
    assert len(project.lamellae) == 1
    lam = project.lamellae[0]
    # should discover the stage folder
    assert any(s.path == stage for s in lam.stages)

    # find stage and verify images
    found_stage = next(s for s in lam.stages if s.path == stage)
    img_paths = {r.image_path.name: r for r in found_stage.images}
    assert "img_sem.tif" in img_paths
    assert "img_fib.tif" in img_paths
    assert img_paths["img_sem.tif"].modality == "SEM"
    assert img_paths["img_fib.tif"].modality == "FIB"
    assert img_paths["img_sem.tif"].project == project.name
    assert img_paths["img_sem.tif"].lamella == lam.name
    assert img_paths["img_sem.tif"].stage == stage.name
    assert img_paths["img_sem.tif"].state == "selected"


def test_image_record_pred_label_matching(tmp_path: Path):
    base = tmp_path / "workspace"
    project_root = base / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")

    lamella = project_root / "01-test-lamella"
    stage = lamella / "Adaptive_polish_1"

    sem = stage / "sem"
    _write_file(sem / "img_sem.tif")

    # add prediction and label files matching the sem image name
    preds_clean = stage / "predictions" / "clean"
    labels = stage / "labels"
    _write_file(preds_clean / "img_sem.tif")
    _write_file(labels / "img_sem.tif")

    scanner = Scanner(base)
    idx = scanner.build_index()
    project = idx.all_projects()[0]
    lam = project.lamellae[0]
    found_stage = next(s for s in lam.stages if s.path == stage)

    rec = next(r for r in found_stage.images if r.image_path.name == "img_sem.tif")
    # predictions and labels should be matched by filename
    assert any(p.name == "img_sem.tif" for p in rec.predictions)
    assert any(label.name == "img_sem.tif" for label in rec.labels)


def test_lamella_discovery_does_not_require_numeric_prefix(tmp_path: Path):
    base = tmp_path / "workspace"
    project_root = base / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")

    lamella = project_root / "test-lamella"
    stage = lamella / "Adaptive_polish_1"
    sem = stage / "sem"
    _write_file(sem / "img_sem.tif")

    scanner = Scanner(base)
    idx = scanner.build_index()

    projects = idx.all_projects()
    assert len(projects) == 1
    project = projects[0]
    assert len(project.lamellae) == 1
    assert project.lamellae[0].name == lamella.name


def test_scanner_assigns_numbered_lamella_and_stage_labels(tmp_path: Path):
    base = tmp_path / "workspace"
    project_root = base / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")

    lamella = project_root / "01-test-lamella"
    first_stage = lamella / "stage_a"
    second_stage = lamella / "stage_b"
    _write_file(first_stage / "sem" / "img_a.tif")
    _write_file(second_stage / "sem" / "img_b.tif")

    scanner = Scanner(base)
    idx = scanner.build_index()
    project = idx.all_projects()[0]
    lam = project.lamellae[0]

    assert lam.number == "01"
    assert [stage.number for stage in lam.stages] == ["01", "02"]
    assert lam.display_name == "01-test-lamella"
    assert lam.stages[0].display_name == "01: stage_a"
    assert lam.stages[1].display_name == "02: stage_b"


def test_scanner_numbers_projects_by_sorted_scan_order(tmp_path: Path):
    base = tmp_path / "workspace"
    for name in ("z-project", "a-project"):
        root = base / name
        root.mkdir(parents=True)
        (root / "experiment.yaml").write_text("project: test")

    projects = Scanner(base).build_index().all_projects()

    assert [(project.number, project.display_name) for project in projects] == [
        ("001", "001: a-project"),
        ("002", "002: z-project"),
    ]


def test_scanner_orders_stages_by_timestamp_in_folder_name(tmp_path: Path):
    base = tmp_path / "workspace"
    project_root = base / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")
    lamella = project_root / "01-test-lamella"
    later_stage = lamella / "Adaptive_polish_2026-08-07-15-30-00"
    earlier_stage = lamella / "Adaptive_polish_2026-08-07-14-20-00"
    _write_file(later_stage / "sem" / "image.tif")
    _write_file(earlier_stage / "sem" / "image.tif")

    stages = Scanner(base).build_index().all_projects()[0].lamellae[0].stages

    assert [stage.name for stage in stages] == [earlier_stage.name, later_stage.name]
    assert [stage.number for stage in stages] == ["01", "02"]


