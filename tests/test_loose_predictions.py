"""Predicting and converting an image that sits loose in its directory.

The round trip is the point: a prediction written for a reference image has to
land where the next scan looks for it. It did not, because the destination was
re-derived from the image's path -- which for a loose image is the lamella
directory, resolving to a ``<lamella>_predictions`` outside the lamella that
the scanner never read. So every prediction written there came back as
"no prediction yet".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.layout import (
    resolve_loose_labels_directory,
    resolve_loose_predictions_directory,
    subdirectories_by_name,
)
from am_segmentation_manager.operations.predict import prediction_destination
from am_segmentation_manager.operations.predictions import plan_conversions
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset, write_marker


def build_lamella_with_reference_images(root: Path) -> Path:
    """Build a 2026-shaped project and return its lamella directory."""
    dataset = build_dataset(
        root,
        lamella_name="01-neat-ape",
        stage_parts=("Milling", "Polishing", "adaptive_polish_2026-07-09-11-23-46AM"),
    )
    write_marker(dataset.project, "experiment", name="project")
    for name in (
        "ref_Polishing_start_eb.tif",
        "ref_Polishing_final_res_01_eb.tif",
    ):
        (dataset.lamella / name).write_bytes(b"")
    return dataset.lamella


def reference_records(root: Path) -> list:
    index = Scanner(root).build_index()
    return [
        record
        for project in index.all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
        if record.series == "reference"
    ]


# ---------------------------------------------------------------------------
# Where a loose set's predictions resolve to
# ---------------------------------------------------------------------------

def test_a_loose_set_resolves_to_a_name_of_its_own(tmp_path: Path):
    """Named for its series and beam, like its labels, and beside them."""
    directory = tmp_path / "01-neat-ape"
    directory.mkdir()

    predictions, exists = resolve_loose_predictions_directory(
        directory, "reference", "SEM", subdirectories_by_name(directory)
    )
    labels, _ = resolve_loose_labels_directory(
        directory, "reference", "SEM", subdirectories_by_name(directory)
    )

    assert predictions == directory / "reference_sem_predictions"
    assert predictions.parent == labels.parent
    assert exists is False


def test_a_loose_set_never_falls_back_to_the_shared_directory(tmp_path: Path):
    """That directory holds the pipeline's Adaptive Polishing output.

    A dataset collapsing the stage level keeps both in the lamella, so falling
    back would pair every reference image with a prediction of another scene.
    """
    directory = tmp_path / "lam_010"
    (directory / "sem_predictions").mkdir(parents=True)

    resolved, exists = resolve_loose_predictions_directory(
        directory, "reference", "SEM", subdirectories_by_name(directory)
    )

    assert resolved == directory / "reference_sem_predictions"
    assert exists is False


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------

def test_a_reference_prediction_is_written_inside_its_lamella(tmp_path: Path):
    """Not into a <lamella>_predictions beside it, where nothing reads."""
    lamella = build_lamella_with_reference_images(tmp_path)
    record = reference_records(tmp_path)[0]

    destination = prediction_destination(record)

    assert lamella in destination.parents
    assert destination.parent == lamella / "reference_sem_predictions" / "am_seg"


def test_the_scanner_finds_a_prediction_written_where_predict_puts_it(tmp_path: Path):
    """The round trip: written by one rule, found by the same one."""
    build_lamella_with_reference_images(tmp_path)
    before = reference_records(tmp_path)
    assert [record.predictions for record in before] == [[], []]

    for record in before:
        destination = prediction_destination(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"")

    after = reference_records(tmp_path)
    assert all(record.predictions for record in after)


def test_convert_predictions_then_has_something_to_convert(tmp_path: Path):
    """The symptom: 'no prediction yet' for every reference image."""
    build_lamella_with_reference_images(tmp_path)
    for record in reference_records(tmp_path):
        destination = prediction_destination(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"")

    plan = plan_conversions(reference_records(tmp_path))

    assert plan.count_skipped("no-prediction") == 0
    assert len(plan.conversions) == 2
    assert all(
        conversion.destination.parent.name == "reference_sem_labels"
        for conversion in plan.conversions
    )


def test_an_adaptive_prediction_still_goes_where_it_always_did(tmp_path: Path):
    """The fallback chain is unchanged: a shared directory is still accepted."""
    dataset = build_dataset(tmp_path, predictions=True)
    index = Scanner(dataset.root).build_index()
    record = next(
        record
        for project in index.all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
        if record.series == "adaptive" and record.modality == "SEM"
    )

    destination = prediction_destination(record)

    assert destination.parent == dataset.stage / "predictions" / "am_seg"
    assert record.predictions, "the pipeline's own raw/clean output stays indexed"


def test_a_record_that_does_not_say_is_refused_rather_than_guessed():
    """Better than writing somewhere plausible that nothing will read."""
    from am_segmentation_manager.models import ImageRecord

    record = ImageRecord(image_path=Path("/p/img.tif"), modality="SEM")

    with pytest.raises(ValueError, match="does not record where its predictions live"):
        prediction_destination(record)
