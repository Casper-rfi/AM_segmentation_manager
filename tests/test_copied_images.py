"""Tests for indexing one record per image where a lamella holds copies.

Autolamella duplicates the image an alignment aligns against in two ways: it
copies it into every run that uses it, and it saves it outside the runs under
a name of its own. Two rules therefore, and each carries its own guard against
merging pictures that merely coincide -- the filename for the first, the
containing directory for the second -- because two images taken inside one
second are ordinary, an alignment run taking several.
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
# The original outside the runs
# ---------------------------------------------------------------------------

def test_the_original_beside_the_runs_wins_over_the_run_copies(
    tmp_path: Path, monkeypatch
):
    """Autolamella saves it once outside and copies it into each run.

    The run copies go: the original is the one file, its name says what it is,
    and it carries the milling task the runs' generic names do not.
    """
    root = tmp_path / "AutoLamella-2026-08-06-15-53"
    task = root / "01-guided-fish" / "Milling" / "Polishing"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: localised\n", encoding="utf-8")
    original = "Polishing_alignment_reference_20-12-30_ib.tif"
    touch(task, original, "Polishing_pre-task_20-12-33_ib.tif")
    times = {
        task / original: TAKEN,
        task / "Polishing_pre-task_20-12-33_ib.tif": datetime(2026, 8, 6, 20, 12, 33),
    }
    for run in ("Fine 1-20-13-12", "Fine 2-20-22-02"):
        directory = touch(task / "Alignment" / run, "reference_image.tif")
        times[directory / "reference_image.tif"] = TAKEN
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", times.get(path))),
    )

    held = {r.image_path for r in Scanner(root.parent).build_index().find_by_name(
        "localised"
    ).image_records}

    assert task / original in held
    assert not any(path.name == "reference_image.tif" for path in held)


def test_a_run_reference_never_saved_outside_is_kept(tmp_path: Path, monkeypatch):
    """15 of the 51 in the reference collection are of this kind.

    Nothing outside the runs holds that acquisition, so the run's copy is the
    only record of it and dropping it would lose the image.
    """
    root = tmp_path / "AutoLamella-2026-08-06-15-53"
    task = root / "01-guided-fish" / "Milling" / "Polishing"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: localised\n", encoding="utf-8")
    touch(task, "Polishing_pre-task_20-12-33_ib.tif")
    directory = touch(task / "Alignment" / "Fine 1-20-13-12", "reference_image.tif")
    times = {
        task / "Polishing_pre-task_20-12-33_ib.tif": datetime(2026, 8, 6, 20, 12, 33),
        directory / "reference_image.tif": TAKEN,
    }
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", times.get(path))),
    )

    held = {r.image_path for r in Scanner(root.parent).build_index().find_by_name(
        "localised"
    ).image_records}

    assert directory / "reference_image.tif" in held


def test_a_same_time_image_in_another_directory_does_not_drop_the_run_copy(
    tmp_path: Path, monkeypatch
):
    """The directory is the second signal, and it has to be doing work.

    The original is where the runs' ``Alignment/`` folder sits, never anywhere
    else in the lamella; matching on the acquisition time alone would let an
    unrelated image elsewhere claim a run's only reference.
    """
    root = tmp_path / "AutoLamella-2026-08-06-15-53"
    lamella = root / "01-guided-fish"
    task = lamella / "Milling" / "Polishing"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: localised\n", encoding="utf-8")
    touch(lamella, "ref_Rough Milling_start_ib.tif")  # elsewhere, same instant
    directory = touch(task / "Alignment" / "Fine 1-20-13-12", "reference_image.tif")
    times = {
        lamella / "ref_Rough Milling_start_ib.tif": TAKEN,
        directory / "reference_image.tif": TAKEN,
    }
    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", times.get(path))),
    )

    held = {r.image_path for r in Scanner(root.parent).build_index().find_by_name(
        "localised"
    ).image_records}

    assert directory / "reference_image.tif" in held
    assert lamella / "ref_Rough Milling_start_ib.tif" in held


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


# ---------------------------------------------------------------------------
# A copy the project holds, rather than one lamella
# ---------------------------------------------------------------------------

@pytest.fixture
def across_lamellae(tmp_path: Path, monkeypatch) -> Path:
    """Three lamellae, two of them handed one lamella's setup image.

    The shape the script-based prototype produced: one acquisition written into
    several lamellae as each one's own starting reference. Nothing inside a
    lamella shows it, so the rule scoped to one lamella cannot see it.
    """
    root = tmp_path / "AutoLamella-2024-06-26-17-12"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")

    times: dict[Path, datetime] = {}
    for index, name in enumerate(("01-enough-bear", "02-game-boar", "03-trusty-osprey")):
        lamella = touch(root / name, "ref_SetupLamella.tif", "ref_MillRoughCut_start_ib.tif")
        # The setup image is one acquisition, handed to all three.
        times[lamella / "ref_SetupLamella.tif"] = TAKEN
        # Each lamella's own milling reference is its own, minutes apart.
        times[lamella / "ref_MillRoughCut_start_ib.tif"] = datetime(
            2024, 6, 26, 17, 40 + index, 12
        )

    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", times.get(path))),
    )
    return root


def test_an_image_handed_to_several_lamellae_is_indexed_once(across_lamellae: Path):
    assert len(paths_of(across_lamellae, "ref_SetupLamella.tif")) == 1


def test_the_lamella_that_took_it_keeps_it(across_lamellae: Path):
    """Lamellae are built in path order, so the earliest keeps the image."""
    kept = paths_of(across_lamellae, "ref_SetupLamella.tif")[0]
    assert kept.parent.name == "01-enough-bear"


def test_each_lamella_keeps_its_own_images(across_lamellae: Path):
    assert len(paths_of(across_lamellae, "ref_MillRoughCut_start_ib.tif")) == 3


def test_the_image_set_stops_naming_the_copy_across_lamellae(across_lamellae: Path):
    for record in records(across_lamellae):
        if record.image_path.parent.name == "01-enough-bear":
            continue
        assert record.image_path.name != "ref_SetupLamella.tif"


def test_one_second_shared_by_two_lamellae_needs_the_filename_too(
    tmp_path: Path, monkeypatch
):
    """The key keeps the filename, so a coincidence of time is not a merge."""
    root = tmp_path / "AutoLamella-2024-06-26-17-12"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")
    times: dict[Path, datetime] = {}
    for name, image in (("01-enough-bear", "ref_SetupLamella.tif"),
                        ("02-game-boar", "ref_ReadyLamella_start_ib.tif")):
        times[touch(root / name, image) / image] = TAKEN

    monkeypatch.setattr(
        "am_segmentation_manager.scanner.header_reader",
        lambda: (lambda path: ("FIB", times.get(path))),
    )
    assert len(records(root)) == 2
