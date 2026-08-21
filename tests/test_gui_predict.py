"""Predicting from the window: the preview, the refusals, and stopping.

The controller needs no display and no torch. The model is loaded only when a
run actually applies, so everything up to that point -- the counts, the wrong
modality, the missing weights -- is asserted here with no inference stack
present, which is also the state most developer machines are in.

Stopping is what these mostly exist for. A run over a project is an hour, and
the property that makes cancelling safe is that it lands between whole images:
what was written stays written, and is recorded as this model's work so the next
run does not repeat it.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from am_segmentation_manager.gui.operations_controller import OperationsController
from am_segmentation_manager.progress import Cancellation, OperationCancelled
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset


@pytest.fixture
def controller(tmp_path: Path) -> OperationsController:
    operations = OperationsController()
    operations.set_start_path(tmp_path)
    return operations


def weights_file(tmp_path: Path) -> Path:
    """Write something shaped like a checkpoint.

    Never loaded: every test here either previews, or substitutes the loading.
    What matters is that it is a zip with one root, so the run name is read the
    way it is from a real one. Its name says SEM and means nothing -- which
    images are predicted is decided by the selection, never by this.
    """
    weights = tmp_path / "AM_SEM_All_V01.pth"
    with zipfile.ZipFile(weights, "w") as archive:
        archive.writestr("training_run_epoch086/data.pkl", b"")
    return weights


def records_of(root: Path, modality: str = "SEM"):
    return [
        record
        for project in Scanner(root).build_index().all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
        if record.modality == modality
    ]


def real_images(records):
    """Give the builder's placeholder images readable pixels."""
    for record in records:
        tifffile.imwrite(str(record.image_path), np.full((64, 96), 7, dtype=np.uint8))
    return records


def loaded_with(controller, monkeypatch, predict):
    """Substitute model loading, so no torch is needed to exercise a real run."""
    class FakeModel:
        pass

    fake = FakeModel()
    fake.predict = predict
    monkeypatch.setattr(
        "am_segmentation_manager.ml.model.load_model",
        lambda weights, description, device="cpu": fake,
    )
    return fake


def halving(image: np.ndarray) -> np.ndarray:
    return np.ones((image.shape[0] // 2, image.shape[1] // 2), dtype=np.uint8)


# --- previewing -------------------------------------------------------------

def test_a_preview_writes_nothing_and_loads_no_model(controller, tmp_path: Path):
    """Whether there is anything to predict is a question about the filesystem.
    Answering it must not cost the seconds a model load does."""
    dataset = build_dataset(tmp_path)

    report = controller.predict(
        records_of(dataset.root), weights_file(tmp_path), dry_run=True
    )

    assert report.count
    assert not report.applied
    assert "Would predict" in report.headline


def test_a_missing_model_is_reported_with_nothing_to_apply(controller, tmp_path: Path):
    dataset = build_dataset(tmp_path)

    report = controller.predict(
        records_of(dataset.root), tmp_path / "absent.pth", dry_run=True
    )

    assert report.count == 0
    assert "absent.pth" in report.headline


def test_a_description_with_a_typo_is_reported_before_anything_loads(
    controller, tmp_path: Path
):
    dataset = build_dataset(tmp_path)
    weights = weights_file(tmp_path)
    weights.with_name(weights.name + ".yaml").write_text("image_sze: 1024\n")

    report = controller.predict(records_of(dataset.root), weights, dry_run=True)

    assert report.count == 0
    assert "image_sze" in report.headline


def test_the_preview_says_what_a_shared_label_covers(controller, tmp_path: Path):
    dataset = build_dataset(tmp_path, sem_variants=("sem_fi_1", "sem_fi_4"))

    report = controller.predict(
        records_of(dataset.root), weights_file(tmp_path), dry_run=True
    )

    assert any("share a label" in line for line in report.lines)


# --- refusing ---------------------------------------------------------------

def test_a_selection_spanning_both_modalities_says_to_pick_one(
    controller, tmp_path: Path
):
    """Not offered as a confirmation: nothing here knows which modality the
    model is for, so there is nothing for the user to allow -- only a selection
    for them to narrow."""
    dataset = build_dataset(tmp_path)
    both = records_of(dataset.root) + records_of(dataset.root, modality="FIB")

    report = controller.predict(both, weights_file(tmp_path), dry_run=True)

    assert report.count == 0
    assert not report.blocked
    assert "both SEM and FIB" in report.headline
    assert "set the edit target" in report.headline


def test_the_headline_says_which_modality_is_being_predicted(
    controller, tmp_path: Path
):
    """The one thing the confirmation must not leave the reader to infer from a
    list of paths: which images these are."""
    dataset = build_dataset(tmp_path)

    for modality in ("SEM", "FIB"):
        report = controller.predict(
            records_of(dataset.root, modality=modality), weights_file(tmp_path),
            dry_run=True,
        )

        assert f"{modality} image(s)" in report.headline


def test_the_device_is_resolved_before_it_reaches_torch(
    controller, tmp_path: Path, monkeypatch
):
    """'auto' is this package's word, not torch's. Passed through it reaches
    model.to('auto'), which fails naming every device except that one."""
    dataset = build_dataset(tmp_path)
    records = real_images(records_of(dataset.root))
    asked: list[str] = []

    class FakeModel:
        predict = staticmethod(halving)

    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.available_devices", lambda: ("cpu",)
    )
    monkeypatch.setattr(
        "am_segmentation_manager.ml.model.load_model",
        lambda weights, description, device="cpu": (
            asked.append(device) or FakeModel()
        ),
    )

    controller.predict(records, weights_file(tmp_path), device="auto")

    assert asked == ["cpu"]


# --- running and stopping ---------------------------------------------------

def test_a_run_writes_and_reports_what_it_wrote(controller, tmp_path: Path, monkeypatch):
    dataset = build_dataset(tmp_path)
    records = real_images(records_of(dataset.root))
    loaded_with(controller, monkeypatch, halving)

    report = controller.predict(records, weights_file(tmp_path), device="cpu")

    assert report.applied
    assert report.count == len(records)
    assert "Predicted" in report.headline


def test_stopping_keeps_what_was_written_and_says_so(
    controller, tmp_path: Path, monkeypatch
):
    """The property that makes a Stop button safe rather than merely
    responsive: cancelling lands between whole images."""
    dataset = build_dataset(tmp_path, acquisitions=4)
    records = real_images(records_of(dataset.root))
    loaded_with(controller, monkeypatch, halving)
    cancellation = Cancellation()

    # Ask to stop as soon as the first image is done.
    report = controller.predict(
        records, weights_file(tmp_path), device="cpu",
        cancellation=cancellation,
        on_progress=lambda done, total: cancellation.request(),
    )

    assert report.applied
    assert report.count == 1
    assert "Stopped after 1 of 4" in report.headline
    assert "kept" in report.headline


def test_what_a_stopped_run_wrote_is_not_predicted_again(
    controller, tmp_path: Path, monkeypatch
):
    """Provenance is flushed per image, so resuming costs only the remainder."""
    dataset = build_dataset(tmp_path, acquisitions=4)
    records = real_images(records_of(dataset.root))
    loaded_with(controller, monkeypatch, halving)
    cancellation = Cancellation()
    controller.predict(
        records, weights_file(tmp_path), device="cpu",
        cancellation=cancellation,
        on_progress=lambda done, total: cancellation.request(),
    )

    resumed = controller.predict(records, weights_file(tmp_path), dry_run=True)

    assert resumed.count == 3
    assert any("already predicted by this model" in line for line in resumed.lines)


def test_progress_counts_towards_the_total(controller, tmp_path: Path, monkeypatch):
    dataset = build_dataset(tmp_path, acquisitions=3)
    records = real_images(records_of(dataset.root))
    loaded_with(controller, monkeypatch, halving)
    seen: list[tuple[int, int]] = []

    controller.predict(
        records, weights_file(tmp_path), device="cpu",
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_cancellation_not_asked_for_does_not_stop_anything():
    """The flag is only read where it is polled, so an unused one is inert."""
    cancellation = Cancellation()

    assert not cancellation.requested
    cancellation.check()

    cancellation.request()
    assert cancellation.requested
    with pytest.raises(OperationCancelled):
        cancellation.check()


# --- what a run leaves in the log -------------------------------------------
#
# The window has crashed outright after a completed run. A run whose steps are
# not marked cannot say which one it reached, so these assert the marks are
# there rather than anything about what they say.

def test_a_run_marks_loading_predicting_and_releasing(
    controller, tmp_path: Path, monkeypatch, caplog
):
    dataset = build_dataset(tmp_path)
    records = real_images(records_of(dataset.root))
    loaded_with(controller, monkeypatch, halving)

    with caplog.at_level("INFO", logger="am_segmentation_manager.gui.operations_controller"):
        controller.predict(records, weights_file(tmp_path), device="cpu")

    messages = [record.getMessage() for record in caplog.records]
    assert any("Predicting" in message for message in messages)
    assert any("Model loaded" in message for message in messages)
    assert any("Releasing the model" in message for message in messages)
    assert any("Model released" in message for message in messages)


def test_a_stopped_run_still_marks_the_release(
    controller, tmp_path: Path, monkeypatch, caplog
):
    """The release is what a crash would land in, so a run that ended early
    must mark it too -- otherwise the log only covers the runs that finished."""
    dataset = build_dataset(tmp_path)
    records = real_images(records_of(dataset.root))
    loaded_with(controller, monkeypatch, halving)
    cancellation = Cancellation()

    with caplog.at_level("INFO", logger="am_segmentation_manager.gui.operations_controller"):
        report = controller.predict(
            records, weights_file(tmp_path), device="cpu",
            cancellation=cancellation,
            on_progress=lambda done, total: cancellation.request(),
        )

    assert "Stopped after" in report.headline
    messages = [record.getMessage() for record in caplog.records]
    assert any("Model released" in message for message in messages)


def test_a_model_that_never_loaded_is_not_reported_as_released(
    controller, tmp_path: Path, monkeypatch, caplog
):
    """A release line with nothing loaded before it would send a reader
    looking for a crash in a step that never ran."""
    dataset = build_dataset(tmp_path)
    records = real_images(records_of(dataset.root))
    monkeypatch.setattr(
        "am_segmentation_manager.ml.model.load_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no torch")),
    )

    with caplog.at_level("INFO", logger="am_segmentation_manager.gui.operations_controller"):
        with pytest.raises(RuntimeError):
            controller.predict(records, weights_file(tmp_path), device="cpu")

    messages = [record.getMessage() for record in caplog.records]
    assert not any("Model released" in message for message in messages)


def test_peak_memory_is_reported_or_says_it_is_unknown():
    """Read from the standard library, so it must not raise where the platform
    does not answer."""
    from am_segmentation_manager.gui.operations_controller import peak_memory

    reported = peak_memory()
    assert reported == "unknown" or reported[-1] in "B"
