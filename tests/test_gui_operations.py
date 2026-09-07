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
# Finding specks across a selection
# ---------------------------------------------------------------------------

def _speck_label(path: Path, specks: int) -> Path:
    """Write a label holding an annotation and *specks* single pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((64, 96), dtype=np.uint8)
    array[10:40, 10:80] = 2
    for n in range(specks):
        array[50, 2 + n * 3] = 1
    tifffile.imwrite(str(path), array)
    return path


def _speck_record(tmp_path: Path, project: str, name: str, specks: int) -> ImageRecord:
    label = _speck_label(tmp_path / project / "labels" / f"{name}.tif", specks)
    return ImageRecord(
        image_path=tmp_path / project / "sem" / f"{name}.tif",
        modality="SEM", project=project, labels=[label],
    )


def _rows(report) -> list[list[str]]:
    """Return the report's table as rows of cells, heading and rule dropped."""
    return [line.split() for line in report.lines[2:]]


def test_the_report_is_a_table_under_a_heading(controller, tmp_path: Path):
    """The dialog is monospace, so padding the columns lines them up -- which
    is the point: three numbers a row read as a list otherwise."""
    records = [_speck_record(tmp_path, "tidy", "a", 1)]

    report = controller.find_specks(records, at_most=64)

    heading, rule = report.lines[0], report.lines[1]
    assert heading.split() == ["Project", "Specks", "Labels", "Per", "label"]
    assert set(rule) == {"-"} and len(rule) == len(heading)
    assert len(report.lines[2]) == len(heading)


def test_the_projects_come_back_worst_first_by_specks_a_label(
    controller, tmp_path: Path
):
    """Forty across forty labels is a stray pixel here and there; forty across
    three is three labels somebody should open."""
    records = [
        _speck_record(tmp_path, "thin", "a", 2),
        _speck_record(tmp_path, "thin", "b", 2),
        _speck_record(tmp_path, "thin", "c", 2),
        _speck_record(tmp_path, "concentrated", "d", 5),
    ]

    report = controller.find_specks(records, at_most=64)

    assert [row[0] for row in _rows(report)] == ["concentrated", "thin"]
    assert _rows(report)[0][1:] == ["5", "1", "5.0"]
    assert _rows(report)[1][1:] == ["6", "3", "2.0"]


def test_a_clean_sweep_says_what_it_covered(controller, tmp_path: Path):
    """"Nothing in 248 labels" is a different statement from "nothing"."""
    records = [_speck_record(tmp_path, "tidy", f"img_{n}", 0) for n in range(3)]

    report = controller.find_specks(records, at_most=64)

    assert report.lines == ()
    assert "3 label(s)" in report.headline


def test_a_shared_label_is_counted_once(controller, tmp_path: Path):
    """The label is the file somebody has to open, and the conditions of one
    scene share it."""
    shared = _speck_label(tmp_path / "p" / "labels" / "img.tif", 5)
    records = [
        ImageRecord(image_path=tmp_path / "p" / directory / "img.tif",
                    modality="SEM", project="p", labels=[shared])
        for directory in ("sem", "sem_5kv")
    ]

    report = controller.find_specks(records, at_most=64)

    assert _rows(report) == [["p", "5", "1", "5.0"]]
    assert "of 1 label(s)" in report.headline


def test_the_threshold_is_what_decides(controller, tmp_path: Path):
    records = [_speck_record(tmp_path, "p", "img", 3)]

    assert controller.find_specks(records, at_most=64).lines
    assert controller.find_specks(records, at_most=0).lines == ()


def test_nothing_selected_is_said_rather_than_swept(controller):
    report = controller.find_specks([], at_most=64)

    assert report.count == 0
    assert "Nothing selected" in report.headline


def test_the_sweep_can_be_stopped(controller, tmp_path: Path):
    """Read-only, so stopping costs nothing: what it leaves is a smaller answer
    rather than a half-finished anything."""
    from am_segmentation_manager.progress import Cancellation

    records = [_speck_record(tmp_path, "p", f"img_{n}", 2) for n in range(4)]
    cancellation = Cancellation()
    cancellation.request()

    with pytest.raises(Exception):
        controller.find_specks(records, at_most=64, cancellation=cancellation)


def test_the_sweep_counts_in_labels_not_records(controller, tmp_path: Path):
    """Conditions of one scene cost one read between them, so a counter over
    records would sit still for several at a time."""
    shared = _speck_label(tmp_path / "p" / "labels" / "img.tif", 1)
    records = [
        ImageRecord(image_path=tmp_path / "p" / directory / "img.tif",
                    modality="SEM", project="p", labels=[shared])
        for directory in ("sem", "sem_5kv", "sem_8")
    ]
    seen = []

    controller.find_specks(records, at_most=64, on_progress=lambda d, t: seen.append((d, t)))

    assert seen == [(1, 1)]


# ---------------------------------------------------------------------------
# Converting predictions
# ---------------------------------------------------------------------------

def _project_with_prediction(tmp_path: Path) -> tuple[Path, Path]:
    """Return the stage directory and the SEM image with a raw prediction."""
    project = tmp_path / "project"
    stage = project / "01-lamella" / "stage"
    (project / "experiment.yaml").parent.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: project")
    image = stage / "sem" / "img.tif"
    prediction = stage / "predictions" / "raw" / "img.tif"
    for path in (image, prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), _sparse_label())
    return stage, image


def test_a_conversion_preview_writes_nothing(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "raw" / "img.tif"],
        labels_directory=stage / "sem_labels",
    )

    report = controller.convert_predictions([record], dry_run=True)

    assert report.count == 1
    assert report.applied is False
    assert not (stage / "sem_labels").exists()
    assert any("sem_labels" in line for line in report.lines)


def test_converting_writes_the_labels_the_preview_listed(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "raw" / "img.tif"],
        labels_directory=stage / "sem_labels",
    )
    preview = controller.convert_predictions([record], dry_run=True)

    report = controller.convert_predictions([record])

    assert report.applied is True
    assert report.count == 1
    assert report.lines == preview.lines
    assert (stage / "sem_labels" / "img.tif").exists()


def test_converting_reports_where_it_put_each_label(controller, tmp_path: Path):
    """The window names these on the records instead of rescanning to find
    them, so a report that did not carry them would cost a walk of the
    collection after every conversion."""
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "raw" / "img.tif"],
        labels_directory=stage / "sem_labels",
    )

    report = controller.convert_predictions([record])

    assert report.written == (stage / "sem_labels" / "img.tif",)


def test_a_conversion_preview_reports_nothing_written(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "raw" / "img.tif"],
        labels_directory=stage / "sem_labels",
    )

    assert controller.convert_predictions([record], dry_run=True).written == ()


def _outcome(**fields):
    """Return a child-process outcome with the fields a report reads."""
    from am_segmentation_manager.operations.predict_process import ProcessOutcome

    return ProcessOutcome(**fields)


def test_a_prediction_reports_where_each_one_went(controller, tmp_path: Path):
    outcome = _outcome(
        written=2, total=2, modality="SEM",
        destinations=(str(tmp_path / "a.tif"), str(tmp_path / "b.tif")),
    )

    report = controller._report_outcome(outcome, tmp_path / "model.pth")

    assert report.written == (tmp_path / "a.tif", tmp_path / "b.tif")


def test_a_stopped_run_reports_only_what_it_reached(controller, tmp_path: Path):
    """`destinations` is filled from the plan before the run, so a stopped one
    names files it never wrote. A record claiming a prediction that is not
    there offers to convert nothing, and says so only when pressed."""
    outcome = _outcome(
        written=1, total=3, modality="SEM", stopped=True,
        destinations=(
            str(tmp_path / "a.tif"),
            str(tmp_path / "b.tif"),
            str(tmp_path / "c.tif"),
        ),
    )

    report = controller._report_outcome(outcome, tmp_path / "model.pth")

    assert report.written == (tmp_path / "a.tif",)


def _record(
    stage: Path,
    kind: str | None = "raw",
    label: Path | None = None,
    *,
    corrected: bool = False,
) -> ImageRecord:
    """Return a record whose prediction is of *kind*, with an optional label.

    Nothing is written: the summary line is counted from the scanned index, and
    a test that built files would not notice if it started reading them.
    """
    directory = stage / "predictions"
    return ImageRecord(
        image_path=stage / "sem" / "img.tif", modality="SEM",
        predictions=[(directory / kind if kind else directory) / "img.tif"] if kind else [],
        labels=[label] if label else [],
        state="corrected" if corrected else "selected",
    )


def test_the_summary_counts_what_is_ready_without_reading_the_disk(tmp_path: Path):
    controller = OperationsController()
    stage = tmp_path / "stage"

    summary = controller.describe_predictions([_record(stage), _record(stage)])

    assert summary == "2 of 2 image(s) ready to convert."


def test_the_summary_separates_the_reasons_nothing_is_ready(tmp_path: Path):
    """The three reasons need different actions, so one count cannot serve.

    Post-processed means re-run the pipeline; missing means wait for it; a
    label already there means neither.
    """
    controller = OperationsController()
    stage = tmp_path / "stage"

    summary = controller.describe_predictions([
        _record(stage, "clean"), _record(stage, None),
        _record(stage, "raw", stage / "sem_labels" / "img.tif", corrected=True),
    ])

    assert "Nothing to convert" in summary
    assert "post-processed" in summary
    assert "no prediction yet" in summary
    assert "proofread" in summary


def test_a_replaceable_label_is_not_reported_as_nothing_to_convert(tmp_path: Path):
    """There is work available; it just replaces something and must be confirmed."""
    controller = OperationsController()
    stage = tmp_path / "stage"

    summary = controller.describe_predictions(
        [_record(stage, "raw", stage / "sem_labels" / "img.tif")]
    )

    assert summary == "Nothing new to convert. 1 with a replaceable label."


def test_an_empty_selection_reports_nothing_at_all(tmp_path: Path):
    """"0 of 0" reads as a fault; with nothing selected the line is hidden."""
    assert OperationsController().describe_predictions([]) == ""


def _labelled_record(stage: Path, image: Path, label_state: str, content) -> ImageRecord:
    """Return a record declared to be in *label_state*, its label holding *content*."""
    directory = stage / "sem_labels"
    directory.mkdir(parents=True, exist_ok=True)
    label = directory / "img.tif"
    tifffile.imwrite(str(label), content)
    return ImageRecord(
        image_path=image, modality="SEM", labels=[label],
        predictions=[stage / "predictions" / "raw" / "img.tif"],
        state=label_state,
    )


def test_an_existing_label_blocks_until_replacing_is_confirmed(controller, tmp_path: Path):
    """A label that is already there is not silently skipped, nor silently lost.

    Silently skipping is what made a re-run look broken; overwriting without
    asking is what the guard exists to prevent. So it blocks, and says which
    files and what is in them.
    """
    stage, image = _project_with_prediction(tmp_path)
    record = _labelled_record(stage, image, "selected", _sparse_label())
    original = record.labels[0].read_bytes()

    report = controller.convert_predictions([record], dry_run=True)

    assert report.blocked == "existing"
    assert report.applied is False
    assert any("replace (holds annotation)" in line for line in report.lines)
    assert record.labels[0].read_bytes() == original


def test_an_empty_placeholder_is_named_as_costing_nothing(controller, tmp_path: Path):
    """The segmentation software writes an empty label for every input image.

    Replacing one loses nothing, and the preview has to say so -- otherwise the
    only safe-looking answer to the prompt is to cancel.
    """
    stage, image = _project_with_prediction(tmp_path)
    record = _labelled_record(stage, image, "selected", np.zeros((4, 4), dtype=np.uint8))

    report = controller.convert_predictions([record], dry_run=True)

    assert report.blocked == "existing"
    assert any("replace (empty placeholder)" in line for line in report.lines)


def test_confirming_replaces_the_label_in_place(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = _labelled_record(stage, image, "selected", np.zeros((4, 4), dtype=np.uint8))
    label = record.labels[0]

    report = controller.convert_predictions([record], allow_overwrite=True)

    assert report.applied is True
    assert report.count == 1
    # Rewritten where it lay, not duplicated into a second file.
    assert len(list((stage / "sem_labels").glob("*.tif"))) == 1
    assert tifffile.imread(str(label)).shape == _sparse_label().shape


def test_a_proofread_label_is_refused_even_when_replacing_is_confirmed(
    controller, tmp_path: Path
):
    """The confirmable tier stops short of proofread work.

    Compression can be confirmed for corrected labels because it preserves pixel
    values. Conversion replaces them with a machine guess, so there is no
    setting that reaches one -- the report names them and says what to do instead.
    """
    stage, image = _project_with_prediction(tmp_path)
    record = _labelled_record(stage, image, "corrected", _sparse_label())
    original = record.labels[0].read_bytes()

    report = controller.convert_predictions([record], allow_overwrite=True)

    assert report.count == 0
    assert record.labels[0].read_bytes() == original
    assert any("refused" in line for line in report.lines)
    assert any("back to selected" in line for line in report.lines)


# ---------------------------------------------------------------------------
# Compressing labels
# ---------------------------------------------------------------------------

def _record_with_label(
    tmp_path: Path,
    folder: str = "labels",
    *,
    name: str = "img.tif",
    corrected: bool = False,
) -> ImageRecord:
    label = tmp_path / folder / name
    label.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(label), _sparse_label())
    return ImageRecord(
        image_path=tmp_path / name,
        labels=[label],
        state="corrected" if corrected else "selected",
    )


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


def test_compression_says_how_far_it_has_got(controller, tmp_path: Path):
    """A whole collection is minutes, and a dialog that only spins says the
    window is busy without saying it is advancing."""
    records = [
        _record_with_label(tmp_path, name=f"label_{index}.tif") for index in range(3)
    ]
    seen: list[tuple[int, int]] = []

    controller.compress_labels(
        records, on_progress=lambda done, total: seen.append((done, total))
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_stopped_compression_keeps_what_it_finished(controller, tmp_path: Path):
    """Safe to stop for a reason the other batch operations do not share: a
    recompressed label holds the same pixels, so a half-done run differs from a
    finished one only in how much disk it takes."""
    from am_segmentation_manager.progress import Cancellation

    records = [
        _record_with_label(tmp_path, name=f"label_{index}.tif") for index in range(4)
    ]
    cancellation = Cancellation()

    report = controller.compress_labels(
        records,
        on_progress=lambda done, total: done == 2 and cancellation.request(),
        cancellation=cancellation,
    )

    assert "Stopped after 2 of 4" in report.headline
    assert "run it again to finish" in report.headline
    assert report.applied is True


def test_an_uninterrupted_compression_does_not_say_it_stopped(controller, tmp_path: Path):
    records = [
        _record_with_label(tmp_path, name=f"label_{index}.tif") for index in range(2)
    ]

    report = controller.compress_labels(records)

    assert "Stopped" not in report.headline
    assert "Compressed" in report.headline


def test_a_compression_preview_reports_no_progress(controller, tmp_path: Path):
    """It reads file sizes and answers at once; a bar that filled instantly
    would be reporting on the wrong run."""
    record = _record_with_label(tmp_path)
    seen: list = []

    controller.compress_labels([record], dry_run=True, on_progress=seen.append)

    assert seen == []


def test_corrected_labels_block_compression_and_are_named(controller, tmp_path: Path):
    record = _record_with_label(tmp_path, corrected=True)
    original = record.labels[0].read_bytes()

    report = controller.compress_labels([record], dry_run=True)

    assert report.blocked == "corrected"
    assert any(record.labels[0].name in line for line in report.lines), (
        "the labels at stake have to be named, not just counted"
    )
    assert record.labels[0].read_bytes() == original


def test_confirming_lets_a_corrected_label_be_compressed(controller, tmp_path: Path):
    record = _record_with_label(tmp_path, corrected=True)

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
def test_a_survey_reports_the_projects_it_found(controller, tmp_path: Path):
    _project_with_prediction(tmp_path)

    report = controller.survey(tmp_path)

    assert report.count == 1
    assert any("Layout survey" in line for line in report.lines)


def test_creating_projects_previews_before_writing(controller, tmp_path: Path):
    _project_with_prediction(tmp_path)
    chosen = [tmp_path / "project"]

    preview = controller.create_projects(chosen, dry_run=True)
    assert preview.count == 1
    assert not (tmp_path / "project" / "AM_metadata.yaml").exists()

    applied = controller.create_projects(chosen, dry_run=False)
    assert applied.applied is True
    assert (tmp_path / "project" / "AM_metadata.yaml").exists()


def test_a_project_that_already_has_metadata_keeps_it(controller, tmp_path: Path):
    _project_with_prediction(tmp_path)
    existing = tmp_path / "project" / "AM_metadata.yaml"
    existing.write_text("sample: HeLa\n")

    report = controller.create_projects([tmp_path / "project"], dry_run=False)

    assert report.count == 0
    assert existing.read_text() == "sample: HeLa\n"
    assert any(line.startswith("kept") for line in report.lines)


def test_a_project_already_indexed_but_undescribed_is_offered(controller, tmp_path: Path):
    """Autolamella writes experiment.yaml and never AM_metadata.yaml, so these
    are projects that simply have no descriptive fields yet."""
    _project_with_prediction(tmp_path)

    assert controller.undescribed_projects(tmp_path) == [tmp_path / "project"]

    controller.create_projects([tmp_path / "project"], dry_run=False)
    assert controller.undescribed_projects(tmp_path) == []


def _gathered_datasets(tmp_path: Path) -> Path:
    """Build a folder holding two datasets of different layout eras.

    With a marked project beside it, so the scan root is not itself the
    outermost unmarked directory holding images -- which is what it would be
    in a tree holding nothing else, making it the only proposal.
    """
    anchor = tmp_path / "indexed" / "lam_001" / "sem"
    anchor.mkdir(parents=True)
    (anchor / "img.tif").write_bytes(b"i")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")

    gathered = tmp_path / "to sort"
    collapsed = gathered / "2024_dataset" / "lam_001" / "sem"
    collapsed.mkdir(parents=True)
    (collapsed / "img.tif").write_bytes(b"i")
    nested = gathered / "2026_dataset" / "lam_001" / "stage" / "sem"
    nested.mkdir(parents=True)
    (nested / "img.tif").write_bytes(b"i")
    return gathered


def test_a_gathering_directory_is_proposed_unticked_and_can_be_opened(
    controller, tmp_path: Path
):
    """The failure this exists to stop: one marker over several acquisitions.

    Proposing the gathering folder would make each dataset in it a lamella of
    a single project, putting two samples under one metadata record.
    """
    gathered = _gathered_datasets(tmp_path)

    proposals = controller.propose_projects(tmp_path)

    assert [item.path for item in proposals] == [gathered]
    assert proposals[0].holds_several_datasets is True

    inside = controller.propose_projects_below(gathered)
    assert [item.path.name for item in inside] == ["2024_dataset", "2026_dataset"]
    assert all(not item.holds_several_datasets for item in inside)


def test_creating_a_project_in_a_gathering_directory_writes_nothing(
    controller, tmp_path: Path
):
    """Batch-validated: one bad path in the set stops the whole set."""
    gathered = _gathered_datasets(tmp_path)

    report = controller.create_projects(
        [gathered / "2024_dataset", gathered], dry_run=False
    )

    assert report.blocked == "placement"
    assert report.count == 0
    assert not (gathered / "AM_metadata.yaml").exists()
    assert not (gathered / "2024_dataset" / "AM_metadata.yaml").exists()


def test_a_proposal_says_what_it_would_become(controller, tmp_path: Path):
    """Lamella names that read like dataset names are what tells a reader the
    level is wrong, and no structural test says that."""
    gathered = _gathered_datasets(tmp_path)

    proposal = controller.propose_projects(tmp_path)[0]
    heading, detail = controller.describe_proposal(proposal)

    assert "2 lamella(e), 2 image(s)" in heading
    assert "2024_dataset" in detail and "2026_dataset" in detail
    assert "gathers several datasets" in detail

    inside = controller.propose_projects_below(gathered)
    _heading, detail = controller.describe_proposal(inside[0])
    assert detail == "would become lamellae: lam_001"


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


# ---------------------------------------------------------------------------
# Reverting: three ways of naming what to go back to, one operation behind them
# ---------------------------------------------------------------------------

def _edit_through_the_package(label: Path) -> None:
    """Replace *label* the way the window does, so the old version is archived.

    A direct write leaves nothing to go back to, which is the whole reason
    saving goes through one path: what it replaces is kept on the way past.
    """
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.label_save import LabelToSave, save_labels

    save_labels(
        [
            LabelToSave(
                destination=label,
                label=_sparse_label() + 1,
                fingerprint=fingerprint_of(label),
            )
        ],
        recorder=Recorder.for_run(),
    )


def _recorded_collection(tmp_path: Path):
    """A scanned collection whose project records, with a savepoint taken."""
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.operations.init_history import (
        apply_baseline,
        plan_baseline,
    )
    from am_segmentation_manager.scanner import Scanner
    from conftest import build_dataset

    root = tmp_path / "collection"
    root.mkdir()
    build_dataset(root, lamellae=1, acquisitions=2, labels="annotated")
    projects = Scanner(root).build_index().all_projects()
    apply_baseline(plan_baseline(projects), recorder=Recorder.for_run())
    return root, projects


def test_a_savepoint_can_be_returned_to_by_name(tmp_path: Path):
    """The round trip the whole feature exists for, at controller level:
    name a state, change a label, put the round back."""
    controller = OperationsController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    controller.take_savepoint(projects, "round 1")

    records = [record for project in projects for record in project.image_records]
    label = next(label for record in records for label in record.labels)
    before = tifffile.imread(str(label))
    _edit_through_the_package(label)
    assert not np.array_equal(tifffile.imread(str(label)), before)

    report = controller.revert_labels(
        records, savepoint="round 1", projects=projects,
        allow_corrected=True, dry_run=True,
    )
    assert report.count, report.headline
    assert not report.applied

    controller.revert_labels(
        records, savepoint="round 1", projects=projects, allow_corrected=True,
    )
    # Pixels, never bytes: a restore goes through the ordinary save and
    # re-encodes, which is why a digest here is over the array.
    assert np.array_equal(tifffile.imread(str(label)), before), "it did not go back"


def test_reverting_refuses_proofread_work_until_it_is_named(tmp_path: Path):
    """The rule CLAUDE.md states, checked before anything is written."""
    from am_segmentation_manager.operations.selection import transition_records

    controller = OperationsController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    controller.take_savepoint(projects, "round 1")

    records = [record for project in projects for record in project.image_records]
    transition_records(records, "corrected")
    label = next(label for record in records for label in record.labels)
    _edit_through_the_package(label)

    report = controller.revert_labels(
        records, savepoint="round 1", projects=projects, dry_run=True
    )

    assert report.blocked == "corrected"
    assert not report.applied
    assert "proofread work" in report.headline


def test_a_savepoint_that_is_not_here_is_reported_rather_than_raised(tmp_path: Path):
    controller = OperationsController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)

    report = controller.revert_labels(
        [record for project in projects for record in project.image_records],
        savepoint="round 9", projects=projects, dry_run=True,
    )

    assert report.blocked == "refused"
    assert not report.applied


def test_a_savepoint_is_refused_inside_a_copy(tmp_path: Path):
    """A copy holds part of a project while carrying that project's log whole,
    so a savepoint written there names labels the copy never had."""
    from am_segmentation_manager.operations.export_record import (
        ExportRecord,
        write_export_record,
    )

    controller = OperationsController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    write_export_record(
        root, ExportRecord(exported_at="2026-09-07T00:00:00Z", exported_by="", source="x")
    )

    report = controller.take_savepoint(projects, "round 1", dry_run=True)

    assert report.blocked == "refused"
    assert "copy" in report.headline
    assert not report.applied
