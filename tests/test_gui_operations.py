"""The operations the window runs, and the preview-then-apply flow around them.

The controller half needs no display: it is where the safety properties are
asserted -- that a preview writes nothing, that corrected labels stop an
operation, and that a report says what actually happened.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("numpy")

import numpy as np
import tifffile

from am_segmentation_manager.gui.operations_controller import (
    OperationsController,
    format_bytes,
)
from am_segmentation_manager.models import ImageRecord

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None


def _sparse_label(shape=(256, 256)) -> np.ndarray:
    """A mask that compresses well: mostly one value, one block of another."""
    array = np.zeros(shape, dtype=np.uint8)
    array[50:200, 50:200] = 2
    return array


@pytest.fixture
def controller(tmp_path: Path) -> OperationsController:
    operations = OperationsController()
    operations.set_start_path(tmp_path)
    return operations


# ---------------------------------------------------------------------------
# Converting predictions
# ---------------------------------------------------------------------------

def _project_with_prediction(tmp_path: Path) -> tuple[Path, Path]:
    """Return the stage directory and the SEM image with a clean prediction."""
    project = tmp_path / "project"
    stage = project / "01-lamella" / "stage"
    (project / "experiment.yaml").parent.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: project")
    image = stage / "sem" / "img.tif"
    prediction = stage / "predictions" / "clean" / "img.tif"
    for path in (image, prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), _sparse_label())
    return stage, image


def test_a_conversion_preview_writes_nothing(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "clean" / "img.tif"],
    )

    report = controller.convert_predictions([record], "clean", dry_run=True)

    assert report.count == 1
    assert report.applied is False
    assert not (stage / "sem_labels").exists()
    assert any("sem_labels" in line for line in report.lines)


def test_converting_writes_the_labels_the_preview_listed(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "clean" / "img.tif"],
    )
    preview = controller.convert_predictions([record], "clean", dry_run=True)

    report = controller.convert_predictions([record], "clean")

    assert report.applied is True
    assert report.count == 1
    assert report.lines == preview.lines
    assert (stage / "sem_labels" / "img.tif").exists()


def test_an_image_that_already_has_a_label_is_not_converted(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    label = stage / "sem_labels" / "img.tif"
    label.parent.mkdir(parents=True)
    tifffile.imwrite(str(label), np.zeros((4, 4), dtype=np.uint8))
    original = label.read_bytes()
    record = ImageRecord(
        image_path=image, modality="SEM", labels=[label],
        predictions=[stage / "predictions" / "clean" / "img.tif"],
    )

    report = controller.convert_predictions([record], "clean", dry_run=True)

    assert report.count == 0
    assert "no matching prediction" in report.headline
    assert label.read_bytes() == original


# ---------------------------------------------------------------------------
# Compressing labels
# ---------------------------------------------------------------------------

def _record_with_label(tmp_path: Path, folder: str = "labels") -> ImageRecord:
    label = tmp_path / folder / "img.tif"
    label.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(label), _sparse_label())
    return ImageRecord(image_path=tmp_path / "img.tif", labels=[label])


def test_a_compression_preview_does_not_touch_the_files(controller, tmp_path: Path):
    """The preview counts files; it must not re-encode them to find a saving.

    Measuring the saving means doing the whole operation, which is minutes on a
    real project, so the preview reports size on disk instead.
    """
    record = _record_with_label(tmp_path)
    original = record.labels[0].read_bytes()

    report = controller.compress_labels([record], dry_run=True)

    assert report.count == 1
    assert report.applied is False
    assert record.labels[0].read_bytes() == original


def test_compressing_reports_what_it_saved(controller, tmp_path: Path):
    record = _record_with_label(tmp_path)
    before = record.labels[0].stat().st_size

    report = controller.compress_labels([record])

    assert report.applied is True
    assert np.array_equal(tifffile.imread(str(record.labels[0])), _sparse_label())
    assert record.labels[0].stat().st_size <= before


def test_corrected_labels_block_compression_and_are_named(controller, tmp_path: Path):
    record = _record_with_label(tmp_path, folder="labels/corrected")
    original = record.labels[0].read_bytes()

    report = controller.compress_labels([record], dry_run=True)

    assert report.blocked == "corrected"
    assert any("corrected" in line for line in report.lines)
    assert record.labels[0].read_bytes() == original


def test_confirming_lets_a_corrected_label_be_compressed(controller, tmp_path: Path):
    record = _record_with_label(tmp_path, folder="labels/corrected")

    report = controller.compress_labels([record], allow_corrected=True)

    assert report.blocked == ""
    assert report.applied is True
    assert np.array_equal(tifffile.imread(str(record.labels[0])), _sparse_label())


def test_a_selection_with_no_labels_has_nothing_to_compress(controller, tmp_path: Path):
    record = ImageRecord(image_path=tmp_path / "img.tif")

    report = controller.compress_labels([record], dry_run=True)

    assert report.count == 0
    assert "no labels" in report.headline


# ---------------------------------------------------------------------------
# Repairing state
# ---------------------------------------------------------------------------

def _inconsistent_group(tmp_path: Path) -> ImageRecord:
    """An image left in the normal folder while its label sits in deselected/."""
    stage = tmp_path / "project" / "01" / "stage"
    image = stage / "sem" / "img.tif"
    label = stage / "sem_labels" / "deselected" / "img.tif"
    for path in (image, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))
    return ImageRecord(
        image_path=image, labels=[stage / "sem_labels" / "img.tif"], modality="SEM"
    )


def test_a_repair_preview_moves_nothing(controller, tmp_path: Path):
    record = _inconsistent_group(tmp_path)

    report = controller.repair_state([record], dry_run=True)

    assert report.count == 1
    assert report.applied is False
    assert record.image_path.exists()


def test_repairing_moves_the_image_to_follow_its_label(controller, tmp_path: Path):
    record = _inconsistent_group(tmp_path)
    original = record.image_path

    report = controller.repair_state([record])

    assert report.applied is True
    assert not original.exists()
    assert (original.parent / "deselected" / "img.tif").exists()


def test_a_consistent_selection_reports_nothing_to_repair(controller, tmp_path: Path):
    record = _record_with_label(tmp_path)

    report = controller.repair_state([record], dry_run=True)

    assert report.count == 0
    assert "Nothing to repair" in report.headline


def test_repairing_out_of_corrected_is_refused_until_confirmed(controller, tmp_path: Path):
    """The label is merely selected, so repair would move a corrected image out."""
    stage = tmp_path / "project" / "01" / "stage"
    image = stage / "sem" / "corrected" / "img.tif"
    label = stage / "sem_labels" / "img.tif"
    for path in (image, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))
    record = ImageRecord(image_path=image, labels=[label], modality="SEM")

    report = controller.repair_state([record], dry_run=True)

    assert report.blocked == "corrected"
    assert image.exists()


# ---------------------------------------------------------------------------
# Read-only reports, and the shared shape they come back in
# ---------------------------------------------------------------------------

def test_a_survey_reports_the_projects_it_found(controller, tmp_path: Path):
    _project_with_prediction(tmp_path)

    report = controller.survey(tmp_path)

    assert report.count == 1
    assert any("Layout survey" in line for line in report.lines)


def test_adding_metadata_files_previews_before_writing(controller, tmp_path: Path):
    _project_with_prediction(tmp_path)

    preview = controller.add_metadata_files(tmp_path, dry_run=True)
    assert preview.count == 1
    assert not (tmp_path / "project" / "AM_metadata.yaml").exists()

    applied = controller.add_metadata_files(tmp_path, dry_run=False)
    assert applied.applied is True
    assert (tmp_path / "project" / "AM_metadata.yaml").exists()


def test_a_project_that_already_has_metadata_keeps_it(controller, tmp_path: Path):
    _project_with_prediction(tmp_path)
    existing = tmp_path / "project" / "AM_metadata.yaml"
    existing.write_text("sample: HeLa\n")

    report = controller.add_metadata_files(tmp_path, dry_run=False)

    assert report.count == 0
    assert existing.read_text() == "sample: HeLa\n"
    assert any(line.startswith("kept") for line in report.lines)


def test_checking_a_csv_reports_the_rows_that_moved(controller, tmp_path: Path):
    (tmp_path / "sem").mkdir()
    (tmp_path / "sem" / "img.tif").write_bytes(b"i")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("image,label\nsem/img.tif,sem_labels/gone.tif\n")

    report = controller.check_csv(csv_path, tmp_path)

    assert report.count == 1
    assert any("MISSING" in line for line in report.lines)


def test_a_report_reads_headline_first(controller, tmp_path: Path):
    record = _record_with_label(tmp_path)

    report = controller.compress_labels([record], dry_run=True)

    assert report.text()[0] == report.headline
    assert len(report.text()) == 1 + len(report.lines)


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (900, "900 B"), (2048, "2.0 kB"), (5 * 1024**2, "5.0 MB")],
)
def test_sizes_are_reported_in_a_readable_unit(size, expected):
    assert format_bytes(size) == expected


# ---------------------------------------------------------------------------
# The window's flow around them
# ---------------------------------------------------------------------------

pytest_qt = pytest.mark.skipif(not HAS_QT, reason="Qt binding not installed")


@pytest.fixture(scope="session")
def qt_app():
    """One QApplication for the whole session; Qt dislikes several."""
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest_qt
def test_the_worker_runs_off_the_interface_thread(qt_app):
    """Submitted work must not run where the widgets are drawn.

    Asserted on the thread the callable actually ran on rather than on the
    result coming back, because a queued connection that silently degraded to a
    direct call would still deliver a result -- and would freeze the window.
    """
    import threading

    from qtpy.QtCore import QObject, Signal

    from am_segmentation_manager.gui.worker import BackgroundOperation

    class Submitter(QObject):
        submit = Signal(object)

    operation = BackgroundOperation()
    submitter = Submitter()
    submitter.submit.connect(operation.worker.run)
    results: list = []
    operation.worker.finished.connect(results.append)
    operation.worker.failed.connect(results.append)

    try:
        submitter.submit.emit(lambda: threading.current_thread().name)
        submitter.submit.emit(lambda: 1 / 0)
        _spin(qt_app, lambda: len(results) == 2)
    finally:
        operation.stop()

    assert results[0] != threading.current_thread().name
    assert isinstance(results[1], ZeroDivisionError)


def _spin(app, condition, timeout: float = 10.0) -> None:
    """Run the event loop until *condition* holds, or fail."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    pytest.fail("Timed out waiting for the worker")
