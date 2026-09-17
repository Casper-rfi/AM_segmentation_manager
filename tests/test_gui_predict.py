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

import pytest

from am_segmentation_manager.gui.operations_controller import OperationsController
from am_segmentation_manager.operations.predict_process import ProcessOutcome
from am_segmentation_manager.progress import Cancellation, OperationCancelled
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")


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


def ran(outcome: ProcessOutcome, seen: list | None = None):
    """Return a runner standing in for the child process.

    The applying run happens in a process of its own, so a test that wants to
    exercise the reporting substitutes what comes back rather than the model
    that would have produced it. What the child itself does is covered in
    test_predict.py, against the same planning code it calls.
    """
    def run(records, weights, device, *, on_progress=None, cancellation=None):
        if seen is not None:
            seen.append((list(records), Path(weights), device))
        if on_progress is not None:
            for index in range(1, outcome.total + 1):
                on_progress(index, outcome.total)
        return outcome

    return run


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


def test_the_child_resolves_the_device_before_it_reaches_torch(
    tmp_path: Path, monkeypatch
):
    """'auto' is this package's word, not torch's. Passed through it reaches
    model.to('auto'), which fails naming every device except that one.

    Driven directly rather than through a spawned process: it is a plain
    function, and running it here is what lets it be checked without torch.
    """
    from am_segmentation_manager.ml import environment
    from am_segmentation_manager.ml import model as model_module
    from am_segmentation_manager.operations import predict_process

    dataset = build_dataset(tmp_path)
    records = real_images(records_of(dataset.root, modality="SEM"))
    asked: list[str] = []

    class FakeModel:
        @staticmethod
        def predict(image):
            return np.ones(
                (image.shape[0] // 2, image.shape[1] // 2), dtype=np.uint8
            )

    monkeypatch.setattr(environment, "available_devices", lambda: ("cpu",))
    monkeypatch.setattr(
        model_module, "load_model",
        lambda weights, description, device="cpu": (
            asked.append(device) or FakeModel()
        ),
    )
    messages: list = []

    class Collect:
        def put(self, message):
            messages.append(message)

    monkeypatch.setattr(predict_process, "_forward_logs", lambda queue: None)
    predict_process._child(records, str(weights_file(tmp_path)), "auto", Collect())

    assert asked == ["cpu"]
    outcome = next(m[1] for m in messages if m[0] == "done")
    assert outcome.written == len(records)
    assert outcome.error == ""


# --- running, out of process ------------------------------------------------
#
# Inference happens in a child, so these pin what the window makes of what it
# reports rather than the inference itself.

def test_a_run_reports_what_the_child_wrote(controller, tmp_path: Path):
    report = controller.predict(
        real_images(records_of(build_dataset(tmp_path).root)),
        weights_file(tmp_path),
        run=ran(ProcessOutcome(
            written=3, total=3, modality="SEM",
            destinations=("a.tif", "b.tif", "c.tif"),
        )),
    )

    assert report.applied
    assert report.count == 3
    assert "Predicted 3 SEM image(s)" in report.headline


def test_a_stopped_run_keeps_what_it_wrote_and_says_so(controller, tmp_path: Path):
    """The property that makes a Stop button safe rather than merely
    responsive: cancelling lands between whole images, and each is written by
    rename before the next begins."""
    report = controller.predict(
        real_images(records_of(build_dataset(tmp_path).root)),
        weights_file(tmp_path),
        run=ran(ProcessOutcome(
            written=1, total=3, modality="SEM", stopped=True,
            destinations=("a.tif", "b.tif", "c.tif"),
        )),
    )

    assert report.applied
    assert report.count == 1
    assert "Stopped after 1 of 3" in report.headline
    assert "run it again to continue" in report.headline
    assert len(report.lines) == 1, "only what was actually written is listed"


def test_a_child_that_could_not_start_is_reported(controller, tmp_path: Path):
    """A missing inference stack now surfaces from the process that has to
    import it, since this one deliberately never does."""
    report = controller.predict(
        real_images(records_of(build_dataset(tmp_path).root)),
        weights_file(tmp_path),
        run=ran(ProcessOutcome(error="RuntimeError: no torch")),
    )

    assert not report.applied
    assert report.count == 0
    assert "no torch" in report.headline


def test_a_selection_of_both_modalities_is_refused_by_the_child_too(
    controller, tmp_path: Path
):
    """The planning happens in the child on the applying run, so the refusal
    has to come back rather than be decided twice."""
    report = controller.predict(
        real_images(records_of(build_dataset(tmp_path).root)),
        weights_file(tmp_path),
        run=ran(ProcessOutcome(modalities=("SEM", "FIB"))),
    )

    assert not report.count
    assert "predict one at a time" in report.headline


def test_the_device_reaches_the_child(controller, tmp_path: Path):
    """'auto' resolves in the child, which is the only process that can see
    what the machine has."""
    seen: list = []
    controller.predict(
        real_images(records_of(build_dataset(tmp_path).root)),
        weights_file(tmp_path),
        device="cuda:1",
        run=ran(ProcessOutcome(written=1, total=1, modality="SEM"), seen),
    )

    assert seen[0][2] == "cuda:1"


def test_progress_counts_towards_the_total(controller, tmp_path: Path):
    seen: list[tuple[int, int]] = []
    controller.predict(
        real_images(records_of(build_dataset(tmp_path).root)),
        weights_file(tmp_path),
        on_progress=lambda done, total: seen.append((done, total)),
        run=ran(ProcessOutcome(written=3, total=3, modality="SEM")),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_preview_still_costs_nothing(controller, tmp_path: Path):
    """It starts no process and loads no model: what a run would write is a
    question about the filesystem."""
    dataset = build_dataset(tmp_path)
    records = records_of(dataset.root)

    report = controller.predict(
        records, weights_file(tmp_path), dry_run=True,
        run=lambda *a, **k: pytest.fail("a preview must start no process"),
    )

    assert not report.applied
    assert report.count


def test_a_cancellation_not_asked_for_does_not_stop_anything():
    """The flag is only read where it is polled, so an unused one is inert."""
    cancellation = Cancellation()

    assert not cancellation.requested
    cancellation.check()

    cancellation.request()
    assert cancellation.requested
    with pytest.raises(OperationCancelled):
        cancellation.check()


# --- the child process itself -----------------------------------------------

def test_the_device_probe_really_starts_a_process():
    """It has to: this process never imports torch, which is the whole point.
    Without the inference stack the child says nothing rather than raising."""
    from am_segmentation_manager.operations.predict_process import probe_devices

    probe = probe_devices()
    devices = probe.devices

    assert isinstance(devices, tuple)
    assert all(isinstance(name, str) for name in devices)
    assert isinstance(probe.problem, str)
