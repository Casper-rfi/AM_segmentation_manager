"""Tests for indexing one record per image where a lamella holds copies.

Autolamella copies the image an alignment aligns against into every run that
uses it, so one acquisition appears as up to six files. What makes recognising
that safe is the filename: two images taken inside one second are ordinary --
an alignment run takes several -- so time alone would merge pictures that
differ.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from am_segmentation_manager.layout import image_files_of
from am_segmentation_manager.scanner import Scanner

TAKEN = datetime(2026, 7, 28, 10, 39, 35)


def touch(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """Two alignment runs holding the same reference image, and their own.

    The header is stood in for, because these are empty files: the reference
    image is one acquisition and each run's own images are their own.
    """
    root = tmp_path / "AutoLamella-2026-07-28-10-31"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")
    alignment = root / "01-ruling-ibex" / "Alignment"

    # Each run's own images are its own, minutes apart, because two runs on one
    # lamella cannot overlap. Within a run the iterations land in one second,
    # which is what the filename has to separate. The reference image is one
    # acquisition, copied into both.
    times: dict[Path, datetime] = {}
    for run, minute in (("01-ruling-ibex - Mill Fiducial-10-35-07", 35),
                        ("01-ruling-ibex - Polishing-11-33-36", 55)):
        directory = touch(
            alignment / run,
            "reference_image.tif", "new_image_00.tif",
            "new_image_01.tif", "final_image.tif",
        )
        times[directory / "reference_image.tif"] = TAKEN
        times[directory / "new_image_00.tif"] = datetime(2026, 7, 28, 10, minute, 43)
        times[directory / "new_image_01.tif"] = datetime(2026, 7, 28, 10, minute, 43)
        times[directory / "final_image.tif"] = datetime(2026, 7, 28, 10, minute, 44)

    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", times.get(path))),
    )
    return root


def records(project: Path):
    return Scanner(project.parent).build_index().find_by_name("bitmap").image_records


def paths_of(project: Path, name: str) -> list[Path]:
    return [r.image_path for r in records(project) if r.image_path.name == name]


# ---------------------------------------------------------------------------
# A copy is indexed once
# ---------------------------------------------------------------------------

def test_a_reference_image_copied_into_two_runs_is_indexed_once(project: Path):
    assert len(paths_of(project, "reference_image.tif")) == 1


def test_the_copy_kept_is_the_one_in_the_earliest_run(project: Path):
    """Stages are already in order when copies are dropped."""
    kept = paths_of(project, "reference_image.tif")[0]

    assert kept.parent.name == "01-ruling-ibex - Mill Fiducial-10-35-07"


def test_the_image_set_stops_naming_the_copy(project: Path):
    """Otherwise anything reading the set would disagree with the records."""
    index = Scanner(project.parent).build_index()

    for lamella in index.find_by_name("bitmap").lamellae:
        for stage in lamella.stages:
            from_sets = {p for s in stage.image_sets for p in image_files_of(s).values()}
            assert from_sets == {r.image_path for r in stage.images}


# ---------------------------------------------------------------------------
# What must not be merged
# ---------------------------------------------------------------------------

def test_two_images_taken_in_one_second_are_both_kept(project: Path):
    """``new_image_00`` and ``new_image_01`` of a run are different pictures."""
    for run in ("01-ruling-ibex - Mill Fiducial-10-35-07",
                "01-ruling-ibex - Polishing-11-33-36"):
        held = {
            r.image_path.name
            for r in records(project)
            if r.image_path.parent.name == run
        }
        assert {"new_image_00.tif", "new_image_01.tif", "final_image.tif"} <= held


def test_each_run_keeps_its_own_images(project: Path):
    assert len(paths_of(project, "final_image.tif")) == 2
    assert len(paths_of(project, "new_image_00.tif")) == 2


def test_two_runs_are_told_apart_by_when_they_ran(project: Path):
    """The one case the filename cannot separate.

    Two runs' ``final_image.tif`` differ in nothing but their acquisition
    time, so a lamella whose runs somehow shared an instant would have one
    dropped. Two runs on one lamella cannot overlap, and across the reference
    collection the rule merged 34 groups, none holding two different images.
    """
    stamps = {
        r.acquired_at
        for r in records(project)
        if r.image_path.name == "final_image.tif"
    }

    assert len(stamps) == 2


def test_an_image_with_no_acquisition_time_is_never_treated_as_a_copy(
    project: Path, monkeypatch
):
    """Nothing establishes that two files are the same picture without one."""
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", None)),
    )

    assert len(paths_of(project, "reference_image.tif")) == 2


def test_adaptive_images_are_never_treated_as_copies(tmp_path: Path):
    """They carry no acquisition time, and their numbers already separate them."""
    root = tmp_path / "AutoLamella-2026"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    for stamp in ("2026-07-28-11-15-36AM", "2026-07-28-12-05-03PM"):
        touch(root / "01-ape" / f"adaptive_polish_{stamp}" / "sem", "img_000_SEM.tif")

    kept = [
        r for r in Scanner(root.parent).build_index().find_by_name("p").image_records
    ]

    assert len(kept) == 2
