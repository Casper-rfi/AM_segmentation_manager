"""Tests for reading ``AM_families.yaml`` and for its reaching label resolution.

The file is hand-editable, so the reader is tested against every way it can be
wrong: a damaged file must be refused, never read as no corrections, which would
send labels to another directory. The rest pins that a scan, a label group and
a bare image path all resolve under the same corrections.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from am_segmentation_manager import families as families_module
from am_segmentation_manager.families import (
    AM_FAMILIES_FILENAME,
    FamiliesFileError,
    families_for,
    read_families,
)
from am_segmentation_manager.layout import NO_CORRECTIONS
from am_segmentation_manager.operations.label_groups import build_group, build_groups
from am_segmentation_manager.operations.selection import record_from_image_path
from am_segmentation_manager.scanner import Scanner


def _write(project: Path, text: str) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / AM_FAMILIES_FILENAME).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_a_project_without_the_file_has_no_corrections(tmp_path: Path):
    assert read_families(tmp_path) is NO_CORRECTIONS


def test_an_empty_file_has_no_corrections(tmp_path: Path):
    _write(tmp_path, "# nothing yet\n")

    assert read_families(tmp_path) is NO_CORRECTIONS


def test_the_file_is_read_into_corrections(tmp_path: Path):
    _write(
        tmp_path,
        "version: 1\n"
        "families:\n"
        "  sem_mag2: sem-mag2\n"
        "  sem2kVb: sem-2kV\n"
        "primary:\n"
        "  SEM: SEM_8_fi_16\n",
    )

    corrections = read_families(tmp_path)

    assert corrections.family_of("SEM_MAG2") == "sem-mag2"
    assert corrections.family_of("sem2kvb") == "sem-2kV"
    assert corrections.family_of("sem_fi8") is None
    assert corrections.primary_of("sem") == "SEM_8_fi_16"


@pytest.mark.parametrize(
    "text, problem",
    [
        ("families:\n  sem_mag2: sem-a\n  sem_mag2: sem-b\n", "Duplicate entry"),
        ("families: {}\nvariants: {}\n", "unknown section variants"),
        ("version: 2\n", "later version"),
        ("version: one\n", "whole number"),
        ("families:\n  - sem_mag2\n", "must be a mapping"),
        ("families:\n  sem_mag2: 5\n", "not a name mapped to a name"),
        ("families:\n  sem_mag2: sem_other\n", "is not a family name"),
        ("- a list\n", "top level"),
        ("families: [unclosed\n", "Failed to parse"),
    ],
)
def test_a_file_that_is_wrong_is_refused_and_named(tmp_path: Path, text: str, problem: str):
    _write(tmp_path, text)

    with pytest.raises(FamiliesFileError, match=problem) as raised:
        read_families(tmp_path)
    assert AM_FAMILIES_FILENAME in str(raised.value)


def test_an_image_path_finds_its_projects_corrections(tmp_path: Path):
    project = tmp_path / "project"
    _write(project, "families:\n  sem_mag2: sem-mag2\n")
    (project / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    image = project / "lam_000" / "adaptive_polish" / "sem_mag2" / "img.tif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")

    assert families_for(image).family_of("sem_mag2") == "sem-mag2"
    assert families_for(tmp_path) is NO_CORRECTIONS


def test_a_batch_reads_each_projects_file_once(tmp_path: Path, monkeypatch):
    (tmp_path / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    stage = tmp_path / "lam_000"
    stage.mkdir()
    reads: list[Path] = []
    real = families_module.read_families
    monkeypatch.setattr(
        families_module, "read_families", lambda root: reads.append(root) or real(root)
    )
    cache: dict = {}

    families_for(stage, cache)
    families_for(stage, cache)

    assert reads == [tmp_path]


# ---------------------------------------------------------------------------
# Reaching resolution
# ---------------------------------------------------------------------------

CONDITIONS = ("sem", "sem_fi8", "sem_mag2")


def _project(root: Path, families: str | None) -> Path:
    """Build a one-stage project whose three SEM conditions share ``sem_labels``."""
    project = root / "project"
    stage = project / "lam_000" / "adaptive_polish_1"
    for directory in (*CONDITIONS, "sem_labels"):
        (stage / directory).mkdir(parents=True)
        (stage / directory / "img_0000.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    if families is not None:
        _write(project, families)
    return stage


def _scanned(root: Path) -> dict[str, object]:
    project = Scanner(root).build_index().all_projects()[0]
    return {record.image_path.parent.name: record for record in project.image_records}


def test_a_scan_resolves_under_the_projects_corrections(tmp_path: Path):
    stage = _project(tmp_path, "families:\n  sem_mag2: sem-mag2\n")

    records = _scanned(tmp_path)

    moved = records["sem_mag2"]
    assert (moved.family, moved.variant) == ("sem-mag2", "mag2")
    assert moved.labels == []
    assert moved.labels_directory == stage / "sem-mag2_labels"
    assert moved.predictions_directory == stage / "sem-mag2_predictions"
    assert records["sem_fi8"].labels == [stage / "sem_labels" / "img_0000.tif"]
    assert records["sem_fi8"].label_shared is True


def test_a_scan_of_a_damaged_file_indexes_by_names_alone(tmp_path: Path, caplog):
    stage = _project(tmp_path, "families:\n  sem_mag2: [\n")

    with caplog.at_level(logging.ERROR):
        records = _scanned(tmp_path)

    assert records["sem_mag2"].labels_directory == stage / "sem_labels"
    assert AM_FAMILIES_FILENAME in caplog.text


def test_a_label_group_holds_only_the_conditions_of_its_corrected_family(tmp_path: Path):
    _project(tmp_path, "families:\n  sem_mag2: sem-mag2\n")
    records = _scanned(tmp_path)

    group = build_group(records["sem"])

    assert {record.image_path.parent.name for record in group.records} == {"sem", "sem_fi8"}
    assert len(build_groups(records.values())) == 2


def test_a_label_group_is_refused_while_the_file_is_damaged(tmp_path: Path):
    _project(tmp_path, "families:\n  sem_mag2: [\n")
    records = _scanned(tmp_path)

    with pytest.raises(FamiliesFileError):
        build_group(records["sem"])


def test_an_image_path_alone_resolves_under_the_corrections(tmp_path: Path):
    stage = _project(tmp_path, "families:\n  sem_mag2: sem-mag2\n")

    record = record_from_image_path(stage / "sem_mag2" / "img_0000.tif")

    assert (record.family, record.variant) == ("sem-mag2", "mag2")
    assert record.labels == []
    assert record.labels_directory == stage / "sem-mag2_labels"
    assert record.predictions_directory == stage / "sem-mag2_predictions"
