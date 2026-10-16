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
    FamiliesLookup,
    families_for,
    read_families,
)
from am_segmentation_manager.layout import (
    NO_CORRECTIONS,
    NO_PROJECT_FAMILIES,
    FamilyCorrections,
    ProjectFamilies,
)
from am_segmentation_manager.operations.label_groups import (
    build_group,
    build_groups,
    record_from_image_path,
)
from am_segmentation_manager.scanner import Scanner


def _write(project: Path, text: str) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / AM_FAMILIES_FILENAME).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_a_project_without_the_file_has_no_corrections(tmp_path: Path):
    assert read_families(tmp_path) is NO_PROJECT_FAMILIES


def test_an_empty_file_has_no_corrections(tmp_path: Path):
    _write(tmp_path, "# nothing yet\n")

    assert read_families(tmp_path) is NO_PROJECT_FAMILIES


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

    corrections = read_families(tmp_path).project

    assert corrections.family_of("SEM_MAG2") == "sem-mag2"
    assert corrections.family_of("sem2kvb") == "sem-2kV"
    assert corrections.family_of("sem_fi8") is None
    assert corrections.primary_of("sem") == "SEM_8_fi_16"


@pytest.mark.parametrize(
    "text, problem",
    [
        ("families:\n  sem_mag2: sem-a\n  sem_mag2: sem-b\n", "Duplicate entry"),
        ("families: {}\nvariants: {}\n", "unknown section variants"),
        ("version: 3\n", "later version"),
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
    families = FamiliesLookup()

    families.at(stage)
    families.at(stage / "sem")
    families.at(tmp_path)

    assert reads == [tmp_path]


def test_a_project_given_corrections_is_never_read(tmp_path: Path, monkeypatch):
    """Whichever directory below its root is asked about, as a merge needs."""
    _write(tmp_path, "families:\n  sem_mag2: [\n")
    (tmp_path / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    image = tmp_path / "lam_000" / "adaptive_polish" / "sem_mag2" / "img.tif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")
    given = ProjectFamilies(project=FamilyCorrections(families={"sem_mag2": "sem-mag2"}))
    families = FamiliesLookup({tmp_path: given})

    assert families.at(image).family_of("sem_mag2") == "sem-mag2"
    assert families.at(image.parents[1]).family_of("sem_mag2") == "sem-mag2"


# ---------------------------------------------------------------------------
# Reaching resolution
# ---------------------------------------------------------------------------

CONDITIONS = ("sem", "sem_fi8", "sem_mag2")


LAMELLA_FILE = (
    "version: 2\n"
    "families:\n  sem_mag2: sem-mag2\n"
    "lamellae:\n"
    "  lam_001:\n"
    "    families:\n      sem_mag2: sem\n"
    "    primary:\n      sem: sem_fi8\n"
)


def test_a_lamella_section_is_read_over_the_projects(tmp_path: Path):
    _write(tmp_path, LAMELLA_FILE)

    families = read_families(tmp_path)

    assert families.at("lam_000").family_of("sem_mag2") == "sem-mag2"
    assert families.at("lam_001/adaptive_polish_1").family_of("sem_mag2") == "sem"
    assert families.at("lam_001").primary_of("sem") == "sem_fi8"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("lamellae: [lam_001]\n", "lamellae must be a mapping"),
        ("lamellae:\n  lam_001:\n    moves: {}\n", "unknown section moves"),
        ("lamellae:\n  lam_001:\n    families:\n      sem_x: fib-x\n", "lam_001: sem_x is a SEM"),
    ],
)
def test_a_lamella_section_that_is_wrong_is_refused_and_named(
    tmp_path: Path, text: str, problem: str
):
    _write(tmp_path, text)

    with pytest.raises(FamiliesFileError, match=problem):
        read_families(tmp_path)


def test_a_file_without_lamella_sections_stays_version_1(tmp_path: Path):
    """An AM-seg that knows no lamella sections still reads it."""
    from am_segmentation_manager.families import write_families
    from am_segmentation_manager.layout import FamilyCorrections, ProjectFamilies

    write_families(
        tmp_path, ProjectFamilies(project=FamilyCorrections(families={"sem_mag2": "sem-mag2"})), None
    )

    assert "version: 1" in (tmp_path / AM_FAMILIES_FILENAME).read_text(encoding="utf-8")


def test_lamella_sections_are_written_as_version_2_and_read_back(tmp_path: Path):
    from am_segmentation_manager.families import write_families

    _write(tmp_path, LAMELLA_FILE)
    families = read_families(tmp_path)
    (tmp_path / AM_FAMILIES_FILENAME).unlink()

    write_families(tmp_path, families, None)

    assert "version: 2" in (tmp_path / AM_FAMILIES_FILENAME).read_text(encoding="utf-8")
    assert read_families(tmp_path) == families


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


def _two_lamellae(root: Path, families: str) -> tuple[Path, Path]:
    """Build lam_000 and lam_001, each one stage of CONDITIONS sharing sem_labels."""
    stages = []
    for lamella in ("lam_000", "lam_001"):
        stage = root / "project" / lamella / "adaptive_polish_1"
        for directory in (*CONDITIONS, "sem_labels"):
            (stage / directory).mkdir(parents=True)
            (stage / directory / "img_0000.tif").write_bytes(b"")
        stages.append(stage)
    (root / "project" / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    _write(root / "project", families)
    return stages[0], stages[1]


def test_a_scan_resolves_each_lamella_under_its_own_corrections(tmp_path: Path):
    first, second = _two_lamellae(tmp_path, LAMELLA_FILE)

    project = Scanner(tmp_path).build_index().all_projects()[0]
    by_lamella = {
        (record.image_path.parents[2].name, record.image_path.parent.name): record
        for record in project.image_records
    }

    assert by_lamella[("lam_000", "sem_mag2")].labels_directory == first / "sem-mag2_labels"
    assert by_lamella[("lam_001", "sem_mag2")].labels_directory == second / "sem_labels"


def test_a_lamella_carries_its_corrections_to_its_frames(tmp_path: Path):
    """The timeline asks the lamella, not its project, which family a condition is in."""
    from am_segmentation_manager.frames import build_lamella_frames

    _two_lamellae(tmp_path, LAMELLA_FILE)
    first, second = Scanner(tmp_path).build_index().all_projects()[0].lamellae

    assert first.families.family_of("sem_mag2") == "sem-mag2"
    assert second.families.family_of("sem_mag2") == "sem", "its section undoes the move"
    assert [
        {(choice.family, choice.condition) for choice in build_lamella_frames(own)[0].condition_choices("SEM")}
        & {("sem-mag2", "sem_mag2"), ("sem", "sem_mag2")}
        for own in (first, second)
    ] == [{("sem-mag2", "sem_mag2")}, {("sem", "sem_mag2")}]


def test_an_image_path_alone_finds_its_lamellas_corrections(tmp_path: Path):
    first, second = _two_lamellae(tmp_path, LAMELLA_FILE)

    assert families_for(first / "sem_mag2").family_of("sem_mag2") == "sem-mag2"
    assert families_for(second / "sem_mag2").family_of("sem_mag2") == "sem"
    record = record_from_image_path(second / "sem_mag2" / "img_0000.tif")
    assert record.labels_directory == second / "sem_labels"
