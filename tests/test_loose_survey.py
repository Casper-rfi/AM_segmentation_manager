"""Tests for what ``inspect-layout`` reports about loose images.

The survey is the command that has to be honest about an unanticipated layout,
so these check both halves of that: the reference and alignment images it now
counts, and the findings it must *not* raise for them -- 5,237 images arriving
with no labels is not 460 things to act on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.survey import survey_project


def touch(directory: Path, *names: str) -> Path:
    """Create *names* as empty files in *directory*, which is created too."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


@pytest.fixture
def resolved_beams(monkeypatch):
    """Answer for the images whose filename states no beam.

    These fixtures write empty files, which have no header to read. Every such
    image in the reference collection is an ion-beam image, so standing in for
    the header with FIB is what makes the fixture resemble real data rather
    than a collection of unreadable files. A test about an *unresolvable* beam
    overrides this.
    """
    monkeypatch.setattr(
        "am_segmentation_manager.survey.header_reader",
        lambda: (lambda path: ("FIB", None)),
    )


@pytest.fixture
def project(tmp_path: Path, resolved_beams) -> Path:
    """A 2026 project: reference and alignment images around one stage."""
    root = tmp_path / "AutoLamella-2026-07-28-10-31"
    root.mkdir(parents=True)
    (root / "experiment.yaml").write_text("name: bitmap\n", encoding="utf-8")
    lamella = touch(
        root / "01-ruling-ibex",
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_start_ib.tif",
        "ref_alignment_ib.tif",
    )
    touch(
        lamella / "Alignment" / "01-ruling-ibex - Polishing-10-55-07",
        "reference_image.tif",
        "new_image_00.tif",
    )
    task = touch(
        lamella / "Milling" / "Polishing", "Polishing_pre-task_10-55-29_eb.tif"
    )
    stage = task / "adaptive_polish_2026-07-28-11-34-24AM"
    touch(stage / "sem", "01-ruling-ibex_AP_img_000_SEM.tif")
    return root


def categories(survey) -> set[str]:
    return {finding.category for finding in survey.findings}


def finding(survey, category: str):
    return next(f for f in survey.findings if f.category == category)


# ---------------------------------------------------------------------------
# What the survey now counts
# ---------------------------------------------------------------------------

def test_loose_images_are_counted(project: Path):
    survey = survey_project(project)

    assert survey.image_count == 7
    assert survey.images_by_series == {
        "adaptive": 1,
        "reference": 3,
        "alignment": 3,
    }


def test_a_project_of_adaptive_images_alone_reports_one_series(tmp_path: Path):
    """So an unchanged project's summary reads exactly as it used to."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    touch(root / "01-ape" / "adaptive_polish_2026-01-01-01-01-01AM" / "sem", "a_SEM.tif")

    assert survey_project(root).images_by_series == {"adaptive": 1}


# ---------------------------------------------------------------------------
# Findings the survey must not raise
# ---------------------------------------------------------------------------

def test_a_series_nobody_labels_yet_is_not_reported_as_unlabelled(project: Path):
    """5,237 images arriving unlabelled is not 460 things to act on."""
    survey = survey_project(project)

    unlabelled = [f for f in survey.findings if f.category == "no-labels-yet"]

    assert [f.path.name for f in unlabelled] == ["sem"]


def test_starting_to_label_a_series_reports_it_normally(project: Path):
    """The suppression is about a series nobody has started, not a series."""
    lamella = project / "01-ruling-ibex"
    touch(lamella / "reference_sem_labels")

    survey = survey_project(project)

    assert finding(survey, "images-without-labels").path == lamella


def test_only_a_stage_of_image_directories_is_expected_to_be_named_for_one(project):
    """A lamella or alignment run was never going to be adaptive_polish*."""
    survey = survey_project(project)

    assert "unexpected-stage-name" not in categories(survey)


def test_a_directory_holding_a_stage_is_not_reported_as_ignored(project: Path):
    """``Alignment/`` and ``Milling/`` are descended into, not ignored."""
    survey = survey_project(project)

    ignored = [f for f in survey.findings if f.category == "unrecognised-directory"]

    assert ignored == []


def test_a_directory_holding_nothing_is_still_reported_as_ignored(project: Path):
    stray = project / "01-ruling-ibex" / "Scratch"
    stray.mkdir()

    survey = survey_project(project)

    assert finding(survey, "unrecognised-directory").path == stray


# ---------------------------------------------------------------------------
# Findings the survey must raise
# ---------------------------------------------------------------------------

def test_an_image_matching_no_rule_is_reported(project: Path):
    touch(project / "01-ruling-ibex", "something_new_eb.tif")

    message = finding(survey_project(project), "unrecognised-image").message

    assert "something_new_eb.tif" in message


def test_an_image_whose_beam_nothing_can_state_is_reported(project: Path, monkeypatch):
    monkeypatch.setattr(
        "am_segmentation_manager.survey.header_reader",
        lambda: (lambda path: (None, None)),
    )
    touch(project / "01-ruling-ibex", "ref_PositionReady.tif")

    message = finding(survey_project(project), "unresolved-beam").message

    assert "ref_PositionReady.tif" in message


def test_images_no_image_set_claims_are_reported(project: Path):
    """The check that catches a layout nobody anticipated."""
    touch(project, "overview-image-2026-07-28_10-31-00.tif")

    message = finding(survey_project(project), "unindexed-images").message

    assert "overview-image-2026-07-28_10-31-00.tif" in message


def test_deliberately_omitted_images_are_not_reported_as_unindexed(project: Path):
    """Not indexed *and* not reported is the point of ignoring them."""
    stage = (
        project / "01-ruling-ibex" / "Milling" / "Polishing"
        / "adaptive_polish_2026-07-28-11-34-24AM"
    )
    touch(stage, "01-ruling-ibex_centring_SEM.tif")

    assert "unindexed-images" not in categories(survey_project(project))


def test_labels_are_not_reported_as_unindexed_images(project: Path):
    """Otherwise annotating a project would report thousands of them."""
    stage = (
        project / "01-ruling-ibex" / "Milling" / "Polishing"
        / "adaptive_polish_2026-07-28-11-34-24AM"
    )
    touch(stage / "sem_labels", "01-ruling-ibex_AP_img_000_SEM.tif")

    assert "unindexed-images" not in categories(survey_project(project))
