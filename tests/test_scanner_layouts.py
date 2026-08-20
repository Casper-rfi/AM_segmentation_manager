"""Scanner behaviour across label naming conventions.

Phase 1 routed the scanner's directory names through :mod:`layout`. These tests
pin what that must not change: existing projects using the pre-convention
``labels`` / ``labels_fib`` names index exactly as before, and projects using
the current ``<image-dir>_labels`` convention index too.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.scanner import Scanner


def _build(tmp_path: Path, sem_labels: str, fib_labels: str) -> Path:
    """Build a one-stage project using the given label directory names."""
    project = tmp_path / "project"
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    for directory, filename in (
        ("sem", "img_000_SEM.tif"),
        ("fib", "img_000_FIB.tif"),
        (sem_labels, "img_000_SEM.tif"),
        (fib_labels, "img_000_FIB.tif"),
    ):
        (stage / directory).mkdir(parents=True, exist_ok=True)
        (stage / directory / filename).write_bytes(b"")
    project.mkdir(exist_ok=True)
    (project / "experiment.yaml").write_text("name: project\n", encoding="utf-8")
    return project


def _records(tmp_path: Path):
    index = Scanner(tmp_path).build_index()
    project = index.all_projects()[0]
    return [
        record
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]


@pytest.mark.parametrize(
    "sem_labels, fib_labels",
    [
        ("labels", "labels_fib"),   # pre-convention, as every existing project uses
        ("labels", "labels_FIB"),   # pre-convention, as written by the oldest datasets
        ("sem_labels", "fib_labels"),  # current convention
    ],
)
def test_labels_are_matched_under_every_naming_convention(tmp_path, sem_labels, fib_labels):
    _build(tmp_path, sem_labels, fib_labels)
    records = _records(tmp_path)

    assert len(records) == 2
    for record in records:
        assert len(record.labels) == 1, f"{record.modality} label not matched"
        assert record.labels[0].name == record.image_path.name


def test_a_nested_file_is_not_indexed_at_all(tmp_path: Path):
    """It belongs to a dataset that has not been migrated. Indexing it would
    report a deselected image as selected; inspect-layout says so instead."""
    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    for directory in ("sem", "sem_labels"):
        (stage / directory / "deselected").mkdir()
        (stage / directory / "img_000_SEM.tif").rename(
            stage / directory / "deselected" / "img_000_SEM.tif"
        )

    records = {record.modality: record for record in _records(tmp_path)}

    assert "SEM" not in records
    assert records["FIB"].state == "selected"


def test_state_comes_from_the_stage_file(tmp_path: Path):
    from am_segmentation_manager.state import read_state, write_state

    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    write_state(read_state(stage), {"sem_labels/img_000_SEM.tif": "corrected"})

    records = {record.modality: record for record in _records(tmp_path)}

    assert records["SEM"].state == "corrected"
    assert records["FIB"].state == "selected"


def test_predictions_are_matched_from_raw_and_clean(tmp_path: Path):
    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    for kind in ("raw", "clean"):
        (stage / "predictions" / kind).mkdir(parents=True)
        (stage / "predictions" / kind / "img_000_SEM.tif").write_bytes(b"")

    records = {record.modality: record for record in _records(tmp_path)}

    assert len(records["SEM"].predictions) == 2
    assert {path.parent.name for path in records["SEM"].predictions} == {"raw", "clean"}
    assert records["FIB"].predictions == []


def test_variant_directories_are_indexed_with_their_variant_recorded(tmp_path: Path):
    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem_5kv").mkdir()
    (stage / "sem_5kv" / "img_000_SEM.tif").write_bytes(b"")

    records = {record.image_path.parent.name: record for record in _records(tmp_path)}

    assert set(records) == {"sem", "fib", "sem_5kv"}
    assert records["sem_5kv"].variant == "5kv"
    assert records["sem"].variant is None


def test_a_variant_sharing_the_modality_label_directory_is_marked_shared(tmp_path: Path):
    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem_5kv").mkdir()
    (stage / "sem_5kv" / "img_000_SEM.tif").write_bytes(b"")

    records = {record.image_path.parent.name: record for record in _records(tmp_path)}

    # Both SEM sets resolve to sem_labels/, so both reference the same label.
    assert records["sem_5kv"].labels == records["sem"].labels
    assert records["sem_5kv"].label_shared is True
    assert records["sem"].label_shared is True
    # The FIB set has that directory to itself.
    assert records["fib"].label_shared is False


def test_a_variant_with_its_own_labels_is_not_marked_shared(tmp_path: Path):
    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    for directory in ("sem_5kv", "sem_5kv_labels"):
        (stage / directory).mkdir()
        (stage / directory / "img_000_SEM.tif").write_bytes(b"")

    records = {record.image_path.parent.name: record for record in _records(tmp_path)}

    assert records["sem_5kv"].label_shared is False
    assert records["sem_5kv"].labels[0].parent.name == "sem_5kv_labels"
    assert records["sem"].labels[0].parent.name == "sem_labels"


def test_the_stage_carries_its_image_sets(tmp_path: Path):
    project = _build(tmp_path, "sem_labels", "fib_labels")
    stage_path = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage_path / "sem_5kv").mkdir()
    (stage_path / "sem_5kv" / "img_000_SEM.tif").write_bytes(b"")

    index = Scanner(tmp_path).build_index()
    stage = index.all_projects()[0].lamellae[0].stages[0]

    assert [image_set.name for image_set in stage.image_sets] == ["sem", "sem_5kv", "fib"]


def test_labels_are_matched_per_image_set_not_per_stage(tmp_path: Path):
    """Two image directories can hold the same filename.

    The SEM and FIB images here share a filename, which older filename-keyed
    matching would have crossed over between modalities.
    """
    project = tmp_path / "project"
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    for directory in ("sem", "fib", "sem_labels", "fib_labels"):
        (stage / directory).mkdir(parents=True)
        (stage / directory / "img_0004.tif").write_bytes(b"")
    project.mkdir(exist_ok=True)
    (project / "experiment.yaml").write_text("name: project\n", encoding="utf-8")

    records = {record.modality: record for record in _records(tmp_path)}

    assert records["SEM"].labels == [stage / "sem_labels" / "img_0004.tif"]
    assert records["FIB"].labels == [stage / "fib_labels" / "img_0004.tif"]
