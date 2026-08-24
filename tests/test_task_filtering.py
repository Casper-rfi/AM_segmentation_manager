"""Tests for resolving, counting and selecting images by milling task.

The task is registered once, as a breakdown, which is what gives it the count,
a ``--task`` selector and a filter row in the Export tab without three
declarations that could disagree. These check that registration, the
vocabulary each lamella's names are matched against, and where a task is
resolved from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from am_segmentation_manager.aggregate import BREAKDOWNS, MISSING, by_task, count_by
from am_segmentation_manager.cli import build_parser
from am_segmentation_manager.filtering import NO_TASK, matches_task
from am_segmentation_manager.layout import normalise_task_name, task_name_runs
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.tasks import (
    NO_TASKS,
    build_vocabulary,
    read_vocabularies,
)

# The two eras' shapes, transcribed from real experiment.yaml files.
MODERN_POSITION = {
    "petname": "01-guided-fish",
    "task_history": [
        {"name": "Setup Lamella Position", "task_type": "SELECT_MILLING_POSITION"},
        {"name": "Mill Fiducial", "task_type": "MILL_FIDUCIAL"},
        {"name": "Rough Milling", "task_type": "MILL_ROUGH"},
        {"name": "Polishing", "task_type": "MILL_POLISHING"},
    ],
    "task_config": {
        "Mill Fiducial": {
            "milling": {"fiducial": {"name": "Fiducial", "stages": [{"name": "Fiducial"}]}}
        },
        "Rough Milling": {
            "milling": {
                "mill_rough": {
                    "name": "Rough Milling",
                    "stages": [{"name": "Rough 1"}, {"name": "Rough 2"}],
                }
            }
        },
        "Polishing": {
            "milling": {
                "mill_polishing": {
                    "name": "Polishing",
                    "stages": [{"name": "Fine 1"}, {"name": "Localised Adaptive Polishing"}],
                }
            }
        },
    },
}

EARLIER_POSITION = {
    "petname": "02-crisp-squid",
    "states": {"Created": {}, "MillRough": {}, "MillPolishing": {}, "SetupLamella": {}},
    "protocol": {
        "mill_rough": [{"name": "Rough 1"}, {"name": "Rough 2"}],
        "mill_polishing": [{"name": "Fine 1"}, {"name": "Fine 2"}],
    },
}


def record(task: str | None, name: str = "img.tif") -> ImageRecord:
    return ImageRecord(image_path=Path("/p") / name, task=task, modality="SEM")


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "left, right",
    [
        ("Rough 1", "Rough-1"),
        ("mill_polishing", "MillPolishing"),
        ("Rough Milling", "Rough-Milling"),
        ("Setup Lamella Position", "setup lamella position"),
    ],
)
def test_spellings_of_one_task_normalise_to_one_name(left, right):
    """A task name is whatever the protocol's author typed, in either place."""
    assert normalise_task_name(left) == normalise_task_name(right)


def test_runs_are_offered_longest_first_then_leftmost():
    """Longest first is what makes a composite resolve through the task."""
    runs = task_name_runs("Rough-Milling-Rough-2_finished", max_words=2)

    assert runs.index("roughmilling") < runs.index("rough2")


def test_no_run_is_longer_than_the_vocabulary_could_match():
    assert all(len(run) <= len("roughmillingrough") for run in
               task_name_runs("Rough-Milling-Rough-2_finished", max_words=3))


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

def test_a_stage_resolves_to_the_task_that_ran_it():
    """Task only: stages roll up rather than forming a second axis."""
    vocabulary = build_vocabulary(MODERN_POSITION)

    assert vocabulary.resolve("Rough-Milling-Rough-2_finished_20-21-21_eb") == "Rough Milling"
    assert vocabulary.resolve("Polishing-Fine-1_finished_20-21-21_eb") == "Polishing"


def test_a_task_spelt_differently_in_a_directory_still_resolves():
    """Autolamella writes Milling/Fiducial/ for the task called Mill Fiducial."""
    assert build_vocabulary(MODERN_POSITION).resolve("Fiducial") == "Mill Fiducial"


def test_a_name_carrying_no_task_resolves_to_nothing():
    """Beam-shift alignment and an alignment run's own images name none."""
    vocabulary = build_vocabulary(MODERN_POSITION)

    assert vocabulary.resolve("beam_shift_alignment_1778239301_650222_ib") is None
    assert vocabulary.resolve("new_image_00") is None
    assert vocabulary.resolve("reference_image") is None


def test_the_earlier_era_reports_the_spelling_its_filenames_use():
    """A state's MillPolishing wins over the protocol key it is stored under."""
    vocabulary = build_vocabulary(EARLIER_POSITION)

    assert vocabulary.resolve("ref_MillPolishing_start_eb") == "MillPolishing"
    assert vocabulary.resolve("ref_milling_Fine-2_finished_1778244865_eb") == "MillPolishing"
    assert vocabulary.resolve("ref_milling_Rough-1_finished_1778239428_eb") == "MillRough"


def test_a_name_two_tasks_both_claim_is_dropped_rather_than_guessed():
    """Reporting a task an image may not belong to is worse than none."""
    vocabulary = build_vocabulary(
        {
            "protocol": {
                "fiducial": [{"name": "Fiducial"}],
                "microexpansion": [{"name": "Fiducial"}],
            }
        }
    )

    assert vocabulary.resolve("ref_Fiducial_initial_alignment_1778237717_ib") is None


def test_a_project_recording_no_tasks_gives_no_vocabulary():
    """Every dataset predating experiment.yaml, and any unlisted lamella."""
    assert read_vocabularies(None) == {}
    assert read_vocabularies({"name": "project"}) == {}
    assert NO_TASKS.resolve("Polishing") is None


def test_vocabularies_are_keyed_by_the_directory_the_images_are_in():
    vocabularies = read_vocabularies(
        {"positions": [MODERN_POSITION, EARLIER_POSITION]}
    )

    assert set(vocabularies) == {"01-guided-fish", "02-crisp-squid"}


# ---------------------------------------------------------------------------
# Where the scanner resolves a task from
# ---------------------------------------------------------------------------

def test_the_directory_answers_for_an_adaptive_polishing_image(tmp_path):
    """Its filename names no task; Milling/<Task>/ above it does."""
    from am_segmentation_manager.scanner import Scanner

    lamella = tmp_path / "01-guided-fish"
    image = lamella / "Milling" / "Polishing" / "adaptive_polish_2026" / "sem" / "a_SEM.tif"

    assert Scanner._resolve_task(image, lamella, build_vocabulary(MODERN_POSITION)) == "Polishing"


def test_the_filename_answers_where_the_directory_cannot(tmp_path):
    """A lamella directory holds every task's reference images at once."""
    from am_segmentation_manager.scanner import Scanner

    lamella = tmp_path / "01-guided-fish"
    image = lamella / "ref_Rough Milling_start_eb.tif"

    assert Scanner._resolve_task(image, lamella, build_vocabulary(MODERN_POSITION)) == "Rough Milling"


def test_the_lamella_directory_itself_is_never_read_as_a_task(tmp_path):
    """A petname sharing a word with a task would claim every image in it."""
    from am_segmentation_manager.scanner import Scanner

    lamella = tmp_path / "03-polishing-crab"
    image = lamella / "beam_shift_alignment_1778239301_650222_ib.tif"

    assert Scanner._resolve_task(image, lamella, build_vocabulary(MODERN_POSITION)) is None


# ---------------------------------------------------------------------------
# Counting and selecting
# ---------------------------------------------------------------------------

RECORDS = [
    record("Polishing", "ref_Polishing_start_eb.tif"),
    record("Polishing", "Polishing-Fine-1_finished_eb.tif"),
    record("Rough Milling", "ref_Rough Milling_start_eb.tif"),
    record(None, "beam_shift_alignment_ib.tif"),
]


def test_counting_by_task_puts_an_unresolved_image_in_its_own_row():
    assert count_by(RECORDS, by_task) == {
        "Polishing": 2, "Rough Milling": 1, MISSING: 1,
    }


@pytest.mark.parametrize(
    "selectors, expected",
    [
        (None, ["Polishing", "Polishing", "Rough Milling", None]),
        (["all"], ["Polishing", "Polishing", "Rough Milling", None]),
        (["Polishing"], ["Polishing", "Polishing"]),
        (["polishing"], ["Polishing", "Polishing"]),
        (["rough-milling"], ["Rough Milling"]),
        (["Polishing,Rough Milling"], ["Polishing", "Polishing", "Rough Milling"]),
        ([NO_TASK], [None]),
        (["MillPolishing"], []),
    ],
)
def test_task_selectors_match_what_they_name(selectors, expected):
    """Separators and case are ignored; an unused name matches nothing."""
    assert [r.task for r in RECORDS if matches_task(r, selectors)] == expected


def test_an_unused_task_name_is_not_an_error():
    """Open vocabulary, unlike a series: it may be another project's task."""
    assert matches_task(RECORDS[0], ["Nonexistent"]) is False


def test_task_is_registered_as_a_breakdown_of_record_kind():
    """Kind 'record' is what keeps it out of the header and label passes."""
    entry = BREAKDOWNS["task"]

    assert entry.kind == "record"
    assert entry.factory is by_task


def test_task_is_offered_as_an_export_filter():
    """One declaration gives the count, the selector and the filter row."""
    from am_segmentation_manager.gui.export_controller import (
        DEDICATED_CONTROLS,
        UNFILTERABLE,
    )

    assert "task" not in DEDICATED_CONTROLS
    assert "task" not in UNFILTERABLE


def test_every_command_with_a_scope_offers_the_task_selector():
    parser = build_parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    for name, sub in action.choices.items():
        options = {option for item in sub._actions for option in item.option_strings}
        if "--series" in options:
            assert "--task" in options, name
