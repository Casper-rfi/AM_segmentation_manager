"""Tests for finding the stages whose images sit loose in a directory.

The tree these build is the 2026 Autolamella layout, which nests reference
images above the Adaptive Polishing stages they surround: a lamella directory
holds reference images and its milling tasks, and each task holds reference
images and its Adaptive Polishing stages. That nesting is the whole reason
:func:`find_stage_directories` treats the two shapes of stage differently, so
it is what these tests build rather than a flat fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.discovery import (
    discover_stages,
    find_dataset_candidates,
    find_stage_directories,
    holds_images,
    holds_loose_images,
)


def mark_project(project: Path) -> Path:
    """Make *project* a project root. Only the marker's existence matters here."""
    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text("name: project\n", encoding="utf-8")
    return project


def touch(directory: Path, *names: str) -> Path:
    """Create *names* as empty files in *directory*, which is created too."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


@pytest.fixture
def modern_project(tmp_path: Path) -> Path:
    """A 2026 project: reference images above the stages they surround."""
    project = tmp_path / "AutoLamella-2026-07-28-10-31"
    mark_project(project)
    lamella = touch(
        project / "01-ruling-ibex",
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
    )
    touch(
        lamella / "Alignment" / "01-ruling-ibex - Polishing-10-55-07",
        "reference_image.tif",
        "new_image_00.tif",
        "final_image.tif",
    )
    task = touch(
        lamella / "Milling" / "Polishing",
        "Polishing_pre-task_10-55-29_eb.tif",
        "Polishing_pre-task_10-55-29_ib.tif",
    )
    touch(
        task / "Alignment" / "Fine 1-10-56-09",
        "reference_image.tif",
        "final_image.tif",
    )
    stage = task / "adaptive_polish_2026-07-28-11-34-24AM"
    touch(stage / "sem", "01-ruling-ibex_AP_img_000_SEM.tif")
    touch(stage / "fib", "01-ruling-ibex_AP_img_000_FIB.tif")
    return project


# ---------------------------------------------------------------------------
# The invariant: a loose-image stage must not end the search
# ---------------------------------------------------------------------------

def test_reference_images_above_a_stage_do_not_hide_it(modern_project: Path):
    """The regression this whole phase exists to prevent.

    Both the lamella and the milling task hold reference images. If either
    ended the search the way an image directory does, the Adaptive Polishing
    stage below would vanish from the index.
    """
    found = find_stage_directories(modern_project)

    adaptive = (
        modern_project
        / "01-ruling-ibex"
        / "Milling"
        / "Polishing"
        / "adaptive_polish_2026-07-28-11-34-24AM"
    )
    assert adaptive in found


def test_every_directory_holding_images_is_a_stage(modern_project: Path):
    lamella = modern_project / "01-ruling-ibex"
    task = lamella / "Milling" / "Polishing"

    found = set(find_stage_directories(modern_project))

    assert found == {
        lamella,
        lamella / "Alignment" / "01-ruling-ibex - Polishing-10-55-07",
        task,
        task / "Alignment" / "Fine 1-10-56-09",
        task / "adaptive_polish_2026-07-28-11-34-24AM",
    }


def test_a_directory_holding_no_images_is_not_a_stage(modern_project: Path):
    """``Milling/`` only ever holds task directories."""
    assert modern_project / "01-ruling-ibex" / "Milling" not in find_stage_directories(
        modern_project
    )


def test_an_image_directory_still_ends_the_search(tmp_path: Path):
    """Nothing sits below ``sem/``, so descending into one is wasted work."""
    project = tmp_path / "project"
    mark_project(project)
    stage = project / "01-neat-ape" / "adaptive_polish_2026-01-01-01-01-01AM"
    touch(stage / "sem", "img_000_SEM.tif")
    # A directory that would be a loose stage, were anything to look in it.
    touch(stage / "sem" / "nested", "ref_Polishing_start_eb.tif")

    found = find_stage_directories(project)

    assert found == [stage]


def test_a_directory_holding_both_shapes_is_returned_once(tmp_path: Path):
    """A collapsed dataset could hold both. The scanner indexes both either way."""
    project = tmp_path / "project"
    mark_project(project)
    lamella = touch(project / "lam_003", "ref_MillRough_start_eb.tif")
    touch(lamella / "SEM_8", "img_0000.tif")

    assert find_stage_directories(project) == [lamella]


# ---------------------------------------------------------------------------
# What a discovered stage says about itself
# ---------------------------------------------------------------------------

def test_a_lamella_standing_in_for_a_stage_says_so(modern_project: Path):
    lamella = modern_project / "01-ruling-ibex"

    stage = next(s for s in discover_stages(modern_project) if s.path == lamella)

    assert stage.stage_is_synthetic
    assert not stage.lamella_is_synthetic
    assert stage.lamella_path == lamella


def test_a_milling_task_is_a_stage_of_its_lamella(modern_project: Path):
    lamella = modern_project / "01-ruling-ibex"
    task = lamella / "Milling" / "Polishing"

    stage = next(s for s in discover_stages(modern_project) if s.path == task)

    assert stage.lamella_path == lamella
    assert not stage.stage_is_synthetic
    assert not stage.in_alignment_run


@pytest.mark.parametrize(
    "parts",
    [
        ("01-ruling-ibex", "Alignment", "01-ruling-ibex - Polishing-10-55-07"),
        ("01-ruling-ibex", "Milling", "Polishing", "Alignment", "Fine 1-10-56-09"),
    ],
)
def test_an_alignment_run_is_marked_at_either_level(modern_project: Path, parts):
    run = modern_project.joinpath(*parts)

    stage = next(s for s in discover_stages(modern_project) if s.path == run)

    assert stage.in_alignment_run
    assert stage.lamella_path == modern_project / "01-ruling-ibex"


# ---------------------------------------------------------------------------
# holds_loose_images
# ---------------------------------------------------------------------------

def test_holds_loose_images_does_not_need_a_rule_to_recognise_the_name(tmp_path):
    """A name no rule reaches is indexed as `other`, so the directory holding it
    has to become a stage -- a set cannot be built in one that never does."""
    run = touch(tmp_path / "Alignment" / "Fine 1-10-56-09", "reference_image.tif")

    assert holds_loose_images(run)
    assert holds_loose_images(touch(tmp_path / "d", "something_new_eb.tif"))


def test_holds_loose_images_ignores_what_this_package_omits_on_purpose(tmp_path):
    assert not holds_loose_images(touch(tmp_path / "e", "01-ape_centring_SEM.tif"))
    assert not holds_loose_images(touch(tmp_path / "f", "notes.txt"))


# ---------------------------------------------------------------------------
# What loose images must NOT change
# ---------------------------------------------------------------------------

def test_loose_images_alone_do_not_make_a_directory_hold_images(tmp_path: Path):
    lamella = touch(tmp_path / "01-ruling-ibex", "ref_Polishing_start_eb.tif")

    assert not holds_images(lamella)


def test_a_directory_of_reference_images_is_not_proposed_as_a_new_project(tmp_path):
    """``init-project`` marks datasets worth curating, which these are not."""
    touch(tmp_path / "somewhere" / "01-ruling-ibex", "ref_Polishing_start_eb.tif")

    assert find_dataset_candidates(tmp_path) == []


def test_a_dataset_of_adaptive_images_is_still_proposed(tmp_path: Path):
    """The outermost unmarked directory holding images, as documented."""
    touch(tmp_path / "somewhere" / "lam_003" / "SEM_8", "img_0000.tif")

    assert find_dataset_candidates(tmp_path) == [tmp_path]
