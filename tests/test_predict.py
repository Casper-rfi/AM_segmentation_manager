"""Planning and running a prediction batch.

None of this needs torch: the model arrives as a callable, which is what lets
the parts most worth pinning -- one prediction per label group, resumption,
provenance, and the refusal to predict the wrong modality -- be tested in an
environment that has no inference stack at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.operations.predict import (
    PredictionSource,
    plan_predictions,
    prediction_destination,
    read_provenance,
    run_predictions,
    write_provenance,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

np = pytest.importorskip("numpy")

MODEL = PredictionSource(name="AM_SEM_All_V01.pth", run_name="run_epoch086", generation="1.0")


def write_real_images(records) -> None:
    """Replace the builder's placeholder image files with readable TIFFs.

    build_dataset writes empty files, which is all a test of paths or counts
    needs. Prediction actually opens the image, so these tests need pixels; kept
    here rather than added to the shared builder until a second file wants it.
    """
    import tifffile

    for record in records:
        tifffile.imwrite(
            str(record.image_path),
            np.full((64, 96), 7, dtype=np.uint8),
        )


def records_of(root: Path, modality: str = "SEM"):
    index = Scanner(root).build_index()
    return [
        record
        for project in index.all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
        if record.modality == modality
    ]


def fake_predict(image: np.ndarray) -> np.ndarray:
    """Stand in for a model: half-size output, as a real one produces."""
    height, width = image.shape[:2]
    return np.ones((height // 2, width // 2), dtype=np.uint8)


# --- where a prediction goes ------------------------------------------------

def test_a_prediction_goes_under_the_directory_this_package_owns(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    record = records_of(dataset.root)[0]

    destination = prediction_destination(record)

    assert destination.parent.name == "am_seg"
    assert destination.name == record.image_path.name
    assert destination.parent.parent.name.endswith("predictions")


def test_planning_creates_nothing(tmp_path: Path):
    """A plan is a question, not a change."""
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)

    plan = plan_predictions(records, MODEL)

    assert plan.predictions
    assert not any(item.destination.parent.exists() for item in plan.predictions)


# --- one prediction per label group -----------------------------------------

def test_conditions_sharing_a_label_are_predicted_once(tmp_path: Path):
    """The reason this is not one prediction per image: the other conditions
    share the label this one becomes, so predicting them is wasted work on
    noisier images."""
    dataset = build_dataset(tmp_path, sem_variants=("sem_fi_1", "sem_fi_4"))
    records = records_of(dataset.root)

    plan = plan_predictions(records, MODEL)

    assert len(plan.predictions) * 3 == len(records)
    assert plan.represented == len(records) - len(plan.predictions)


def test_the_primary_condition_is_the_one_predicted(tmp_path: Path):
    dataset = build_dataset(tmp_path, sem_variants=("sem_fi_1",))

    plan = plan_predictions(records_of(dataset.root), MODEL)

    predicted = {item.record.variant for item in plan.predictions}
    assert predicted == {None}


def test_a_selection_of_variants_alone_is_predicted_from_the_primary(tmp_path: Path):
    """The primary's record is built from disk, not scanned, and must say where
    its prediction goes."""
    dataset = build_dataset(tmp_path, sem_variants=("sem_fi_1",))
    variants = [record for record in records_of(dataset.root) if record.variant == "fi_1"]

    plan = plan_predictions(variants, MODEL)

    assert {item.record.variant for item in plan.predictions} == {None}
    assert all(item.destination.parent.name == "am_seg" for item in plan.predictions)


def test_a_family_predicts_from_the_primary_its_project_names(tmp_path: Path):
    dataset = build_dataset(tmp_path, sem_variants=("sem_fi_1",))
    (dataset.project / "AM_families.yaml").write_text(
        "primary:\n  sem: sem_fi_1\n", encoding="utf-8"
    )

    plan = plan_predictions(records_of(dataset.root), MODEL)

    assert {item.record.variant for item in plan.predictions} == {"fi_1"}


def test_a_family_in_conflict_is_predicted_once_per_destination(tmp_path: Path):
    """Two label directories make two groups, but one predictions directory
    holds both: two writes there would leave only the second."""
    stage = tmp_path / "project" / "lam_000" / "adaptive_polish_1"
    for directory in ("sem", "sem_fi1", "sem_fi2", "sem_labels", "sem_fi1_labels"):
        (stage / directory).mkdir(parents=True)
        (stage / directory / "img_0000.tif").write_bytes(b"")
    (tmp_path / "project" / "experiment.yaml").write_text("name: p\n", encoding="utf-8")
    # sem_fi1's group first, so the family's primary must win rather than the order.
    records = sorted(records_of(tmp_path), key=lambda record: record.condition != "sem_fi1")

    plan = plan_predictions(records, MODEL)

    [planned] = plan.predictions
    assert planned.record.condition == "sem"
    assert planned.destination == stage / "sem_predictions" / "am_seg" / "img_0000.tif"
    assert plan.represented == 2
    assert planned.shared_with == 2


# --- one modality per run ---------------------------------------------------

def test_a_selection_spanning_both_modalities_is_refused(tmp_path: Path):
    """A model is trained on one modality and nothing here can tell which, so
    the selection is the only statement of it -- and one naming both says
    nothing. An SEM model over a FIB image returns a confident, entirely wrong
    prediction, so it has to be a choice somebody made."""
    dataset = build_dataset(tmp_path)
    both = records_of(dataset.root) + records_of(dataset.root, modality="FIB")

    plan = plan_predictions(both, MODEL)

    assert plan.is_refused
    assert not plan.predictions
    assert plan.modalities == ("SEM", "FIB")
    assert plan.modality is None


def test_either_modality_alone_predicts_and_names_itself(tmp_path: Path):
    """Predicting FIB needs no setting and no model description: select FIB."""
    dataset = build_dataset(tmp_path)

    for modality in ("SEM", "FIB"):
        plan = plan_predictions(records_of(dataset.root, modality=modality), MODEL)

        assert not plan.is_refused
        assert plan.modality == modality
        assert {item.record.modality for item in plan.predictions} == {modality}


def test_a_refused_plan_resolves_no_destinations(tmp_path: Path):
    """Refused whole and early: reading every stage directory for a run that
    will not happen is work for nothing."""
    dataset = build_dataset(tmp_path)
    both = records_of(dataset.root) + records_of(dataset.root, modality="FIB")

    plan = plan_predictions(both, MODEL)

    assert plan.represented == 0


def test_running_a_refused_plan_raises_rather_than_writing(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    both = records_of(dataset.root) + records_of(dataset.root, modality="FIB")
    plan = plan_predictions(both, MODEL)

    with pytest.raises(RuntimeError, match="Refusing to predict"):
        run_predictions(plan, fake_predict, MODEL)


# --- writing, provenance and resumption -------------------------------------

def test_a_run_writes_one_file_per_planned_prediction(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)
    write_real_images(records)
    plan = plan_predictions(records, MODEL)

    written = run_predictions(plan, fake_predict, MODEL)

    assert written == len(plan.predictions)
    assert all(item.destination.exists() for item in plan.predictions)


def test_the_prediction_keeps_the_model_resolution(tmp_path: Path):
    """Not resized to the image: convert-predictions does that, and storing it
    upsampled would only be undone."""
    import tifffile

    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)
    write_real_images(records)
    plan = plan_predictions(records, MODEL)
    run_predictions(plan, fake_predict, MODEL)

    item = plan.predictions[0]
    image = tifffile.imread(item.record.image_path)
    prediction = tifffile.imread(item.destination)
    assert prediction.shape == (image.shape[0] // 2, image.shape[1] // 2)


def test_every_prediction_records_which_model_made_it(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)
    write_real_images(records)
    plan = plan_predictions(records, MODEL)
    run_predictions(plan, fake_predict, MODEL)

    item = plan.predictions[0]
    entry = read_provenance(item.destination.parent)[item.destination.name]

    assert entry["model"] == "AM_SEM_All_V01.pth"
    assert entry["run"] == "run_epoch086"
    assert entry["generation"] == "1.0"
    assert entry["predicted_at"]
    assert entry["image"] == item.record.image_path.name
    assert entry["condition"] == item.record.image_path.parent.name


def test_a_prediction_recorded_before_generations_is_still_up_to_date(tmp_path: Path):
    """The generation is recorded, not matched on: one checkpoint is of one
    generation, and every prediction made before it was recorded was made as
    generation 1, so none of them is predicted again."""
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)
    write_real_images(records)
    run_predictions(plan_predictions(records, MODEL), fake_predict, MODEL)
    replanned = plan_predictions(records, MODEL, overwrite=True)
    directories = {item.destination.parent for item in replanned.predictions}
    for directory in directories:
        entries = read_provenance(directory)
        for entry in entries.values():
            del entry["generation"]
        write_provenance(directory, entries)

    plan = plan_predictions(records, MODEL)

    assert not plan.predictions
    assert plan.up_to_date


def test_a_second_run_of_the_same_model_predicts_nothing_again(tmp_path: Path):
    """What makes an interrupted run cheap to resume."""
    dataset = build_dataset(tmp_path)
    write_real_images(records_of(dataset.root))
    first = plan_predictions(records_of(dataset.root), MODEL)
    run_predictions(first, fake_predict, MODEL)

    second = plan_predictions(records_of(dataset.root), MODEL)

    assert not second.predictions
    assert len(second.up_to_date) == len(first.predictions)


def test_overwrite_predicts_them_again(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    write_real_images(records_of(dataset.root))
    first = plan_predictions(records_of(dataset.root), MODEL)
    run_predictions(first, fake_predict, MODEL)

    second = plan_predictions(records_of(dataset.root), MODEL, overwrite=True)

    assert len(second.predictions) == len(first.predictions)
    assert all(item.outcome == "replace" for item in second.predictions)


def test_a_different_model_predicts_again_without_being_asked(tmp_path: Path):
    """Resumption keys on the model, not on the file existing: a prediction from
    another model is not this model's work already done.

    This is also how an accidental run is undone -- predicting the same images
    with the right model replaces them, no --overwrite needed."""
    dataset = build_dataset(tmp_path)
    write_real_images(records_of(dataset.root))
    run_predictions(plan_predictions(records_of(dataset.root), MODEL), fake_predict, MODEL)
    other = PredictionSource(name="AM_SEM_Arctis_V01.pth", run_name="other")

    plan = plan_predictions(records_of(dataset.root), other)

    assert plan.predictions
    assert all(item.outcome == "replace" for item in plan.predictions)


def test_an_unaccounted_prediction_is_made_again(tmp_path: Path):
    """A file with no provenance cannot be claimed as this model's work."""
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)
    write_real_images(records)
    plan = plan_predictions(records, MODEL)
    run_predictions(plan, fake_predict, MODEL)
    (plan.predictions[0].destination.parent / "am_seg_predictions.yaml").unlink()

    again = plan_predictions(records_of(dataset.root), MODEL)

    assert len(again.predictions) == len(plan.predictions)


def test_progress_is_reported_for_each_prediction(tmp_path: Path):
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)
    write_real_images(records)
    plan = plan_predictions(records, MODEL)
    seen: list[int] = []

    run_predictions(plan, fake_predict, MODEL, on_progress=lambda n, _: seen.append(n))

    assert seen == list(range(1, len(plan.predictions) + 1))
