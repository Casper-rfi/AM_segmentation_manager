"""Where a prediction gets written, when there is more than one candidate.

Prediction directories resolve exactly as label directories do, including the
legacy names and the fallback to a directory shared by every condition of one
scene. Sharing is safe here for a reason that lives elsewhere: only the primary
image of a label group is ever predicted (``label_groups.primary_record``), so
two conditions never claim one filename -- and predicting the others would waste
the work on noisier images that share the same label anyway.

The paths here are written out literally rather than asked of layout.py, for
the reason conftest.py gives: a test that derives its expectation from the code
under test agrees with it by construction.
"""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.layout import (
    resolve_labels_directory,
    resolve_predictions_directory,
    subdirectories_by_name,
)


def stage(tmp_path: Path, *directories: str) -> Path:
    """Create a stage holding *directories* and return it."""
    for name in directories:
        (tmp_path / name).mkdir(parents=True)
    return tmp_path


def resolve(stage_path: Path, image_directory: str, modality: str):
    return resolve_predictions_directory(
        stage_path / image_directory, modality, subdirectories_by_name(stage_path)
    )


# --- nothing exists yet: the private name is where one would go ------------

def test_a_new_sem_prediction_directory_is_named_after_its_image_directory(tmp_path: Path):
    directory, exists, legacy = resolve(stage(tmp_path, "sem"), "sem", "SEM")

    assert directory == tmp_path / "sem_predictions"
    assert (exists, legacy) == (False, False)


def test_a_new_fib_prediction_directory_mirrors_the_sem_one(tmp_path: Path):
    """The question this file was written for: FIB predictions have a name
    already, and it is the same rule as everything else."""
    directory, exists, legacy = resolve(stage(tmp_path, "fib"), "fib", "FIB")

    assert directory == tmp_path / "fib_predictions"
    assert (exists, legacy) == (False, False)


def test_a_variant_with_nothing_to_share_gets_the_private_name(tmp_path: Path):
    directory, _, _ = resolve(stage(tmp_path, "fib", "fib_lowmag"), "fib_lowmag", "FIB")

    assert directory == tmp_path / "fib_lowmag_predictions"


# --- existing directories are reused, including the legacy names -----------

def test_an_existing_directory_is_used_rather_than_a_second_naming(tmp_path: Path):
    """The label rule applied here: a project must not end up holding both
    namings because this package wrote the newer one beside the older."""
    stage_path = stage(tmp_path, "sem", "predictions")

    directory, exists, legacy = resolve(stage_path, "sem", "SEM")

    assert directory == tmp_path / "predictions"
    assert (exists, legacy) == (True, True)


def test_the_legacy_fib_name_is_recognised_when_it_is_there(tmp_path: Path):
    stage_path = stage(tmp_path, "fib", "predictions_fib")

    directory, exists, legacy = resolve(stage_path, "fib", "FIB")

    assert directory == tmp_path / "predictions_fib"
    assert (exists, legacy) == (True, True)


def test_the_private_directory_wins_over_a_legacy_one(tmp_path: Path):
    stage_path = stage(tmp_path, "sem", "sem_predictions", "predictions")

    directory, _, legacy = resolve(stage_path, "sem", "SEM")

    assert directory == tmp_path / "sem_predictions"
    assert legacy is False


# --- conditions of one scene -----------------------------------------------

def test_conditions_share_a_prediction_directory_as_they_share_a_label(tmp_path: Path):
    """Four conditions of one acquisition resolve to one directory, for
    predictions and labels alike. Only the primary one is ever predicted, so
    the filename they have in common is written once."""
    stage_path = stage(
        tmp_path, "SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "sem_predictions", "sem_labels"
    )
    siblings = subdirectories_by_name(stage_path)
    conditions = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")

    predictions = {resolve(stage_path, name, "SEM")[0] for name in conditions}
    labels = {
        resolve_labels_directory(stage_path / name, "SEM", siblings)[0]
        for name in conditions
    }

    assert predictions == {tmp_path / "sem_predictions"}
    assert labels == {tmp_path / "sem_labels"}
