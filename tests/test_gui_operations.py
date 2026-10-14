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

from am_segmentation_manager.gui.history_controller import HistoryController
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

    report = controller._report_outcome(outcome, tmp_path / "model.pth", "Adaptive Polishing")

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

    report = controller._report_outcome(outcome, tmp_path / "model.pth", "Adaptive Polishing")

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

    summary = controller.describe_predictions(
        [_record(stage), _record(stage)], "microscope"
    )

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
    assert "corrected, never replaced" in summary


def test_a_replaceable_label_is_not_reported_as_nothing_to_convert(tmp_path: Path):
    """There is work available; it just replaces something and must be confirmed."""
    controller = OperationsController()
    stage = tmp_path / "stage"

    summary = controller.describe_predictions(
        [_record(stage, "raw", stage / "sem_labels" / "img.tif")], "microscope"
    )

    assert summary == "Nothing new to convert. 1 with a replaceable label."


def test_the_summary_counts_images_that_would_use_the_other_source(tmp_path: Path):
    """Which predictions exist is asked before pressing, not found out after."""
    controller = OperationsController()
    stage = tmp_path / "stage"

    summary = controller.describe_predictions(
        [_record(stage, "raw"), _record(stage, "am_seg")]
    )

    assert summary == (
        "2 of 2 image(s) ready to convert. "
        "1 from microscope predictions, having no am-seg one."
    )


def _with_am_seg_prediction(stage: Path, image: Path) -> ImageRecord:
    """Return a record holding both an am-seg and a microscope prediction."""
    am_seg = stage / "predictions" / "am_seg" / "img.tif"
    am_seg.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(am_seg), _sparse_label())
    return ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "raw" / "img.tif", am_seg],
        labels_directory=stage / "sem_labels",
    )


def test_the_preview_names_the_source_it_converts_from(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = _with_am_seg_prediction(stage, image)

    chosen = controller.convert_predictions([record], dry_run=True, prefer="microscope")
    default = controller.convert_predictions([record], dry_run=True)

    assert chosen.headline == "Would write 1 label(s) from microscope predictions."
    assert chosen.lines == (
        "project/01-lamella/stage/sem_labels/img.tif: write, from microscope",
    )
    assert default.headline == "Would write 1 label(s) from am-seg predictions."


def test_the_preview_says_how_many_images_use_the_other_source(controller, tmp_path: Path):
    stage, image = _project_with_prediction(tmp_path)
    record = ImageRecord(
        image_path=image, modality="SEM",
        predictions=[stage / "predictions" / "raw" / "img.tif"],
        labels_directory=stage / "sem_labels",
    )

    report = controller.convert_predictions([record], dry_run=True)

    assert "1 image(s) have no am-seg prediction and use the microscope one." in report.lines


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
    assert any("label holding annotation replaced" in line for line in report.lines)
    assert "  project/01-lamella/stage/sem_labels/img.tif" in report.lines
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
    assert "    1  empty placeholder replaced, nothing lost" in report.lines


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
    assert any("left unchanged" in line for line in report.lines)
    assert any("back to selected" in line for line in report.lines)


def _mixed_selection(tmp_path: Path) -> list[ImageRecord]:
    """One image for each thing a conversion can find where its label goes.

    Without a label; with an empty placeholder; with annotation nobody marked
    corrected; with an earlier conversion never edited; and corrected.
    """
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.operations.label_io import label_fields
    from conftest import start_history

    project = tmp_path / "project"
    stage = project / "01-lamella" / "stage"
    start_history(project)
    records = []
    contents = {
        "new": None,
        "placeholder": np.zeros((4, 4), dtype=np.uint8),
        "annotated": _sparse_label(),
        "converted": _sparse_label() * 2,
        "corrected": _sparse_label(),
    }
    for name, content in contents.items():
        image = stage / "sem" / f"{name}.tif"
        prediction = stage / "predictions" / "raw" / f"{name}.tif"
        for path in (image, prediction):
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(path), _sparse_label())
        labels = []
        if content is not None:
            label = stage / "sem_labels" / f"{name}.tif"
            label.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(label), content)
            labels = [label]
        if name == "converted":
            fields = label_fields(content)
            Recorder(by="casper", by_source="flag").record(
                label, "label-converted", **fields,
                model="fpn", prediction_kind="raw",
            )
        records.append(ImageRecord(
            image_path=image, modality="SEM", labels=labels,
            labels_directory=stage / "sem_labels",
            predictions=[prediction],
            state="corrected" if name == "corrected" else "selected",
        ))
    return records


def test_replacing_says_what_each_label_there_comes_to(controller, tmp_path: Path):
    """A large selection mixes what costs nothing with what costs annotation, and
    the dialog has to separate them before anyone presses the button."""
    report = controller.convert_predictions(_mixed_selection(tmp_path), dry_run=True)

    assert report.headline == (
        "Converting replaces 3 existing label(s). "
        "1 corrected label(s) are left untouched."
    )
    counts = [line for line in report.lines if line[:5].strip().isdigit()]
    assert counts == [
        "    1  new label, none there yet",
        "    1  empty placeholder replaced, nothing lost",
        "    1  label holding annotation replaced, not marked corrected: "
        "kept in AM_archive/, restorable",
        "    1  earlier conversion replaced, never edited: not kept, its "
        "prediction converts it again",
        "    1  corrected label left untouched: converting never replaces one",
    ]


def test_the_button_says_how_many_are_replaced_and_kept(controller, tmp_path: Path):
    """"Replace them" read as reaching the corrected labels too."""
    report = controller.convert_predictions(_mixed_selection(tmp_path), dry_run=True)

    assert report.action == "Replace 3, keep 1 corrected"


def test_each_kind_of_replaced_label_is_named_under_its_heading(controller, tmp_path: Path):
    report = controller.convert_predictions(_mixed_selection(tmp_path), dry_run=True)
    lines = list(report.lines)

    def under(heading: str) -> str:
        start = next(i for i, line in enumerate(lines) if line.startswith(heading))
        return lines[start + 1].strip()

    assert under("With annotation").endswith("annotated.tif")
    assert under("Earlier conversions").endswith("converted.tif")
    assert under("Corrected, left unchanged").endswith("corrected.tif")
    assert not any(line.strip().endswith("placeholder.tif") for line in lines), (
        "a placeholder costs nothing, so it is counted and not listed"
    )


def test_what_the_preview_promises_kept_is_kept(controller, tmp_path: Path):
    """The preview decides as the write does, so 'kept' is not a hope."""
    records = _mixed_selection(tmp_path)

    controller.convert_predictions(records, allow_overwrite=True)

    archived = sorted(
        path.name.split("__")[0]
        for path in (tmp_path / "project" / "AM_archive").rglob("*.tif")
    )
    assert archived == ["annotated"]


def test_a_long_list_of_labels_to_replace_is_cut_short(controller, tmp_path: Path):
    from am_segmentation_manager.gui.report import MAX_REPORTED_FILES

    stage, _image = _project_with_prediction(tmp_path)
    records = []
    for index in range(MAX_REPORTED_FILES + 5):
        image = stage / "sem" / f"img{index:02d}.tif"
        tifffile.imwrite(str(image), _sparse_label())
        label = stage / "sem_labels" / f"img{index:02d}.tif"
        label.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(label), _sparse_label())
        records.append(ImageRecord(
            image_path=image, modality="SEM", labels=[label],
            predictions=[stage / "predictions" / "raw" / "img.tif"],
        ))

    report = controller.convert_predictions(records, dry_run=True)

    assert "... and 5 more" in report.lines


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
    assert "Run it again to finish" in report.headline
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


def test_the_summary_counts_the_directories_beside_the_projects_and_their_images(
    controller, tmp_path: Path
):
    """The project list is silent about data nothing has indexed, which is the
    state a collection is in before any of this has been done."""
    _gathered_datasets(tmp_path)

    assert controller.unindexed_summary(tmp_path) == (1, 2)


def test_a_fully_indexed_folder_has_nothing_unindexed(controller, tmp_path: Path):
    from conftest import build_dataset

    build_dataset(tmp_path)

    assert controller.unindexed_summary(tmp_path) == (0, 0)


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


@pytest_qt
def test_the_worker_runs_off_the_interface_thread(qt_app):
    """Submitted work must not run where the widgets are drawn.

    Asserted on the thread the callable actually ran on rather than on the
    result coming back, because a queued connection that silently degraded to a
    direct call would still deliver a result -- and would freeze the window.
    """
    import threading

    from am_segmentation_manager.gui.worker import BackgroundOperation

    operation = BackgroundOperation()
    results: list = []
    take = lambda: results.append(operation.worker.take_outcome())  # noqa: E731
    operation.worker.finished.connect(take)
    operation.worker.failed.connect(take)

    try:
        operation.submit(lambda: threading.current_thread().name)
        operation.submit(lambda: 1 / 0)
        _spin(qt_app, lambda: len(results) == 2)
    finally:
        operation.stop()

    assert results[0] != threading.current_thread().name
    assert isinstance(results[1], ZeroDivisionError)


def _signals(worker) -> dict[str, list[str]]:
    """Return each signal *worker*'s class declares, with its parameter types."""
    from qtpy.QtCore import QMetaMethod

    meta = worker.metaObject()
    return {
        bytes(method.name()).decode(): [bytes(t).decode() for t in method.parameterTypes()]
        for method in (meta.method(i) for i in range(meta.methodOffset(), meta.methodCount()))
        if method.methodType() == QMetaMethod.MethodType.Signal
    }


@pytest_qt
def test_no_signal_from_a_worker_thread_carries_a_python_object(qt_app):
    """A Python object carried by a queued signal is released by the binding's
    C++ event destructor on whichever thread the event reached, racing the
    interface thread handling it -- so the signals of both workers carry
    numbers, text or nothing, and anything larger travels through a queue."""
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.loader import FrameLoadWorker
    from am_segmentation_manager.gui.worker import OperationWorker

    assert _signals(OperationWorker()) == {
        "finished": [], "failed": [], "progressed": ["int", "int"],
    }
    assert _signals(FrameLoadWorker(FrameCache())) == {
        "loaded": ["int", "int"],
        "batch_finished": ["int"],
        "counted": ["int", "QString", "int"],
    }


@pytest_qt
def test_the_loader_runs_what_it_is_given_off_the_interface_thread(qt_app, tmp_path):
    """Handed over through its queue, in the order it was asked, and reported
    from the loader's own thread."""
    import threading

    from qtpy.QtCore import Qt

    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.loader import BackgroundLoader

    loader = BackgroundLoader(FrameCache())
    seen: list[tuple[str, int, str]] = []
    loader.worker.counted.connect(
        lambda token, key, count: seen.append((key, count, threading.current_thread().name)),
        Qt.ConnectionType.DirectConnection,
    )
    token = loader.next_token()
    try:
        loader.count_specks([(tmp_path / "a.tif", tmp_path / "a.tif")], token, 4)
        loader.count_specks([(tmp_path / "b.tif", tmp_path / "b.tif")], token, 4)
        _spin(qt_app, lambda: len(seen) == 2)
    finally:
        loader.stop()

    here = threading.current_thread().name
    assert [(key, count) for key, count, _thread in seen] == [
        (str(tmp_path / "a.tif"), 0), (str(tmp_path / "b.tif"), 0),
    ]
    assert all(thread != here for _key, _count, thread in seen)


@pytest_qt
def test_the_work_and_its_result_are_freed_on_the_submitting_thread(qt_app):
    """The worker lets go of both before it signals, so the submitter's
    reference is the last one and whatever the closure captured is freed where
    it was made -- the window's thread, not the operations thread."""
    import threading
    import weakref

    from am_segmentation_manager.gui.worker import BackgroundOperation

    class Watched:
        pass

    freed: dict[str, str] = {}

    def watched(name: str) -> Watched:
        thing = Watched()
        weakref.finalize(
            thing, lambda: freed.setdefault(name, threading.current_thread().name)
        )
        return thing

    def make_work():
        captured = watched("captured")

        def work() -> Watched:
            assert captured is not None
            return watched("result")

        return work

    work = make_work()
    operation = BackgroundOperation()
    done: list[bool] = []
    operation.worker.finished.connect(lambda: done.append(True))
    try:
        operation.submit(work)
        _spin(qt_app, lambda: done)
        result = operation.worker.take_outcome()
        assert freed == {}, "the worker must not free what it was handed"
        del work, result
    finally:
        operation.stop()

    here = threading.current_thread().name
    assert freed == {"captured": here, "result": here}


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

def _edit_through_the_package(label: Path, shift: int = 1) -> None:
    """Replace *label* the way the window does, so the old version is archived.

    *shift* makes the new pixels differ from another edit's.

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
                label=_sparse_label() + shift,
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


def _records_of(projects) -> list[ImageRecord]:
    """Every image record of *projects*, in scan order."""
    return [record for project in projects for record in project.image_records]


def test_a_savepoint_can_be_returned_to_by_name(tmp_path: Path):
    """The round trip the whole feature exists for, at controller level:
    name a state, change a label, put the round back."""
    controller = HistoryController()
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

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    controller.take_savepoint(projects, "round 1")

    records = [record for project in projects for record in project.image_records]
    label = next(label for record in records for label in record.labels)
    # Edited before it is marked: a save over a corrected label needs consent.
    _edit_through_the_package(label)
    transition_records(records, "corrected")

    report = controller.revert_labels(
        records, savepoint="round 1", projects=projects, dry_run=True
    )

    assert report.blocked == "corrected"
    assert not report.applied
    assert "are marked corrected" in report.headline


def test_a_savepoint_that_is_not_here_is_reported_rather_than_raised(tmp_path: Path):
    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)

    report = controller.revert_labels(
        [record for project in projects for record in project.image_records],
        savepoint="round 9", projects=projects, dry_run=True,
    )

    assert not report.count and not report.blocked
    assert not report.applied


def test_reverting_what_was_never_converted_says_which_labels_cannot_be_reached(
    tmp_path: Path,
):
    """Nothing here has been converted, so nothing has a version to go back to
    -- and naming each label and why beats a count of zero. With no count the
    window has nothing to offer to apply."""
    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    records = _records_of(projects)

    report = controller.revert_labels(records, projects=projects)

    assert report.headline.startswith(
        f"Nothing to restore: none of these {len(records)} label(s)"
    )
    assert not report.count and not report.blocked and not report.applied
    names = [label.name for record in records for label in record.labels]
    assert [line.split(": ")[0].rsplit("/", 1)[-1] for line in report.lines] == names


@pytest.mark.parametrize(("dry_run", "starts"), [(False, True), (True, False)])
def test_a_savepoint_starts_the_record_it_writes_into_but_a_preview_does_not(
    tmp_path: Path, dry_run: bool, starts: bool
):
    """A savepoint is a write like any other, so a project keeping no history
    joins the record first and is then named; the preview only reads."""
    from am_segmentation_manager.history import history_file_path
    from am_segmentation_manager.operations.savepoint import list_savepoints
    from am_segmentation_manager.scanner import Scanner
    from conftest import build_dataset

    root = tmp_path / "collection"
    root.mkdir()
    build_dataset(root, lamellae=1, acquisitions=2, labels="annotated")
    projects = Scanner(root).build_index().all_projects()
    controller = HistoryController()
    controller.set_start_path(root)

    controller.take_savepoint(projects, "round 1", dry_run=dry_run)

    assert all(
        history_file_path(project.path).is_file() is starts for project in projects
    )
    if starts:
        (point,) = list_savepoints(projects)
        assert point.found == len(projects), "every project is named by it"


def test_reclaiming_keeps_everything_while_no_savepoint_has_closed_a_round(
    tmp_path: Path,
):
    """The savepoints are the whole policy, so with none taken the archive is
    one open round and nothing goes. Saying why beats reporting zero."""
    from am_segmentation_manager.operations.archive import archived_versions

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    label = next(label for record in _records_of(projects) for label in record.labels)
    for shift in (1, 2, 3):
        _edit_through_the_package(label, shift)
    kept = [
        version.path
        for project in projects
        for version in archived_versions(project.path)
    ]
    assert len(kept) == 3, "each edit left the version it replaced behind"

    report = controller.reclaim_archive(projects)

    assert report.headline.startswith("Nothing to free yet.")
    assert f"All {len(kept)} kept version(s) are kept" in report.headline
    assert not report.count and not report.applied
    assert all(path.is_file() for path in kept), "nothing was removed"


def test_a_savepoint_is_refused_inside_a_copy(tmp_path: Path):
    """A copy holds part of a project while carrying that project's log whole,
    so a savepoint written there names labels the copy never had."""
    from am_segmentation_manager.operations.export_record import (
        ExportRecord,
        write_export_record,
    )

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    write_export_record(
        root, ExportRecord(exported_at="2026-09-07T00:00:00Z", exported_by="", source="x")
    )

    report = controller.take_savepoint(projects, "round 1", dry_run=True)

    # A refusal carries no count and cannot be confirmed past: what it
    # refuses is not the user's to overrule.
    assert not report.count and not report.blocked
    assert "copy" in report.headline
    assert not report.applied


# ---------------------------------------------------------------------------
# Copying a selection out, which writes somewhere the collection is not
# ---------------------------------------------------------------------------

def test_a_copy_carries_what_was_selected_its_label_and_the_project_file(
    controller, tmp_path: Path
):
    """The filters decide what, so only the records handed over are copied --
    with the label each is drawn with and the file that makes the copy
    indexable."""
    root, projects = _recorded_collection(tmp_path)
    first, other, *_ = _records_of(projects)
    destination = tmp_path / "deposit"

    report = controller.copy_files(
        [first], destination, projects=projects, start_path=root
    )

    assert report.applied and report.count
    assert (destination / first.image_path.relative_to(root)).is_file()
    assert (destination / first.labels[0].relative_to(root)).is_file()
    assert not (destination / other.image_path.relative_to(root)).exists()
    project_file = projects[0].path / "experiment.yaml"
    assert (destination / project_file.relative_to(root)).is_file()


def test_a_copy_preview_says_what_it_would_write_and_writes_nothing(
    controller, tmp_path: Path
):
    """Gigabytes being the usual size of this, refusing the preview has to
    leave the folder as it was."""
    root, projects = _recorded_collection(tmp_path)
    destination = tmp_path / "deposit"

    report = controller.copy_files(
        _records_of(projects), destination, projects=projects, start_path=root,
        dry_run=True,
    )

    assert report.headline.startswith("Would copy")
    assert report.count and not report.applied
    assert not destination.exists()


def test_a_copy_made_here_can_be_merged_back(controller, tmp_path: Path):
    """A copy carrying no fork point is refused by `merge` outright, so an
    export that wrote one only from the command line meant the window could
    make copies it could never take back."""
    from am_segmentation_manager.operations.export_record import read_export_record
    from am_segmentation_manager.operations.merge import plan_merge

    root, projects = _recorded_collection(tmp_path)
    destination = tmp_path / "laptop"

    report = controller.copy_files(
        _records_of(projects), destination, projects=projects, start_path=root
    )

    assert read_export_record(destination) is not None
    assert plan_merge(destination, root) is not None
    assert report.lines[0] == "Exported with a fork point, so it can be merged back."


def test_a_copy_records_what_it_was_narrowed_to(controller, tmp_path: Path):
    """In the caller's words, because writing it as command-line flags would
    describe a run nobody made."""
    from am_segmentation_manager.operations.export_record import read_export_record

    root, projects = _recorded_collection(tmp_path)
    destination = tmp_path / "subset"

    controller.copy_files(
        _records_of(projects), destination, projects=projects, start_path=root,
        scope="SEM, annotated",
    )

    assert read_export_record(destination).scope == "SEM, annotated"


def test_a_copy_that_was_stopped_is_not_given_a_fork_point(
    controller, tmp_path: Path
):
    """What was copied is kept, but a fork point describes the whole tree the
    copy was meant to be, and the report says to run it again before merging it back."""
    from am_segmentation_manager.operations.export_record import read_export_record
    from am_segmentation_manager.progress import Cancellation

    root, projects = _recorded_collection(tmp_path)
    destination = tmp_path / "interrupted"
    cancellation = Cancellation()

    report = controller.copy_files(
        _records_of(projects), destination, projects=projects, start_path=root,
        cancellation=cancellation,
        on_progress=lambda done, total: cancellation.request(),
    )

    assert report.applied and "Stopped early" in report.headline
    assert read_export_record(destination) is None
    assert "before merging it back" in report.lines[0]


@pytest.mark.parametrize("dry_run", [True, False])
def test_copying_into_the_collection_is_refused_in_the_preview_and_the_run(
    controller, tmp_path: Path, dry_run: bool
):
    """It would be walking into its own output, and the copy would be scanned
    as part of the collection it came from. A refusal carries no count and
    cannot be confirmed past: the answer is another folder."""
    root, projects = _recorded_collection(tmp_path)
    inside = root / "export"

    report = controller.copy_files(
        _records_of(projects), inside, projects=projects, start_path=root,
        dry_run=dry_run,
    )

    assert report.headline.startswith("Choose a folder outside the collection")
    assert not report.count and not report.blocked and not report.applied
    assert not inside.exists()


# ---------------------------------------------------------------------------
# Merging a copy back, which is the other half of the export beside it
# ---------------------------------------------------------------------------

def _exported_copy(tmp_path: Path, root: Path, projects):
    """Copy *root* out the way the window's export does, fork point and all."""
    from am_segmentation_manager.operations.copy_export import (
        copy_export,
        plan_copy,
        record_fork_point,
    )

    destination = tmp_path / "copy"
    records = [record for project in projects for record in project.image_records]
    copy_export(plan_copy(records, destination, root, projects=projects))
    record_fork_point(destination, records, projects, root)
    return destination


def test_a_copy_worked_on_elsewhere_comes_home(tmp_path: Path):
    """The round trip the Export tab's two buttons are: out, worked on, back."""
    from am_segmentation_manager.scanner import Scanner

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy = _exported_copy(tmp_path, root, projects)

    theirs = Scanner(copy).build_index().all_projects()
    label = next(
        label
        for project in theirs
        for record in project.image_records
        for label in record.labels
    )
    _edit_through_the_package(label)
    changed = tifffile.imread(str(label))

    report = controller.merge_copy(copy, projects, dry_run=True)
    assert report.count, report.headline
    assert not report.applied

    applied = controller.merge_copy(copy, projects)
    assert applied.applied, applied.headline

    ours = root / label.relative_to(copy)
    assert np.array_equal(tifffile.imread(str(ours)), changed), (
        "the work done in the copy did not arrive"
    )


def test_a_folder_that_is_not_a_copy_is_refused_by_name(tmp_path: Path):
    """Only a copy an export made carries the fork point a merge needs."""
    controller = HistoryController()
    root, _projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    stranger = tmp_path / "stranger"
    stranger.mkdir()

    report = controller.merge_copy(stranger)

    assert not report.count and not report.blocked
    assert "fork point" in report.headline
    assert not report.applied


def test_merging_over_proofread_work_is_refused_until_it_is_named(tmp_path: Path):
    """CLAUDE.md's rule, and the one thing a merge was once found not doing."""
    from am_segmentation_manager.operations.selection import transition_records
    from am_segmentation_manager.scanner import Scanner

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy = _exported_copy(tmp_path, root, projects)

    theirs = Scanner(copy).build_index().all_projects()
    label = next(
        label
        for project in theirs
        for record in project.image_records
        for label in record.labels
    )
    _edit_through_the_package(label)
    transition_records(
        [record for project in projects for record in project.image_records],
        "corrected",
    )

    report = controller.merge_copy(copy, projects, dry_run=True)

    assert report.blocked == "corrected"
    assert not report.applied
    assert "corrected item(s)" in report.headline


# ---------------------------------------------------------------------------
# Deciding a conflict
# ---------------------------------------------------------------------------

def _conflicting_copy(tmp_path: Path, root: Path, projects):
    """Return a copy where one label was changed differently on both sides."""
    from am_segmentation_manager.scanner import Scanner

    copy = _exported_copy(tmp_path, root, projects)
    theirs = Scanner(copy).build_index().all_projects()
    their_label = next(
        label
        for project in theirs
        for record in project.image_records
        for label in record.labels
    )
    our_label = root / their_label.relative_to(copy)
    _edit_through_the_package(their_label)
    save_two(our_label)
    return copy, our_label, their_label


def save_two(label: Path) -> None:
    """Write a different edit here, so both sides have moved."""
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.io.atomic import fingerprint_of
    from am_segmentation_manager.operations.label_save import LabelToSave, save_labels

    save_labels(
        [
            LabelToSave(
                destination=label,
                label=_sparse_label() + 2,
                fingerprint=fingerprint_of(label),
            )
        ],
        allow_corrected=True,
        recorder=Recorder.for_run(),
    )


def test_a_conflict_is_reported_as_something_to_decide(tmp_path: Path):
    """Not as a refusal: a person can answer this, and the report carries what
    they need to answer it with."""
    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy, ours, theirs = _conflicting_copy(tmp_path, root, projects)

    report = controller.merge_copy(copy, projects, dry_run=True)

    assert report.blocked == "conflicts"
    assert len(report.conflicts) == 1
    row = report.conflicts[0]
    assert row.ours == ours and row.theirs == theirs
    assert any("label" in line for line in row.details)


def test_a_report_says_how_many_lines_it_left_out():
    """A list that stops without saying so reads as the whole of it."""
    from am_segmentation_manager.gui.report import MAX_REPORTED_FILES, capped

    lines = [f"line {n}" for n in range(MAX_REPORTED_FILES + 7)]

    shown = capped(lines)

    assert shown[:-1] == tuple(lines[:MAX_REPORTED_FILES])
    assert shown[-1] == "... and 7 more"
    assert capped(lines[:MAX_REPORTED_FILES]) == tuple(lines[:MAX_REPORTED_FILES])


def test_conflicts_past_the_listed_ones_are_counted_and_still_decided(
    tmp_path: Path, monkeypatch
):
    """The report lists only so many, and says so; the dialog gets every one."""
    from am_segmentation_manager.gui import report as report_module
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.operations.metadata_edit import (
        update_project_metadata,
    )

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy = _exported_copy(tmp_path, root, projects)
    for tree, value in ((root, "here"), (copy, "there")):
        update_project_metadata(
            tree / projects[0].key,
            {"description": value, "sample": value},
            recorder=Recorder.for_run(),
        )
    monkeypatch.setattr(report_module, "MAX_REPORTED_FILES", 1)

    report = controller.merge_copy(copy, projects, dry_run=True)

    assert report.blocked == "conflicts"
    assert len(report.conflicts) == 2
    assert report.lines[-1] == "... and 1 more"


def test_a_conflict_says_where_it_sits(tmp_path: Path):
    """What the resolver arranges conflicts by -- project, lamella, the image
    for acquisition order, the field -- comes with each one."""
    from am_segmentation_manager.history import Recorder
    from am_segmentation_manager.operations.metadata_edit import (
        update_project_metadata,
    )

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy, _ours, _theirs = _conflicting_copy(tmp_path, root, projects)
    for tree, value in ((root, "here"), (copy, "there")):
        update_project_metadata(
            tree / projects[0].key, {"description": value},
            recorder=Recorder.for_run(),
        )

    report = controller.merge_copy(copy, projects, dry_run=True)

    group = next(row for row in report.conflicts if not row.is_field)
    field = next(row for row in report.conflicts if row.is_field)
    assert group.project == projects[0].key
    assert group.record is not None
    assert group.lamella and group.lamella == group.record.lamella
    assert (field.project, field.metadata_field) == (projects[0].key, "description")
    assert (field.our_value, field.their_value) == ("here", "there")


def test_what_the_resolver_decides_is_what_the_merge_reads(tmp_path: Path):
    from am_segmentation_manager.gui.conflict_resolution import ConflictResolution
    from am_segmentation_manager.operations.merge import THEIRS

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy, ours, theirs = _conflicting_copy(tmp_path, root, projects)
    wanted = tifffile.imread(str(theirs))

    blocked = controller.merge_copy(copy, projects, dry_run=True)
    resolution = ConflictResolution(blocked.conflicts)
    resolution.decide_all(resolution.labels, THEIRS)
    applied = controller.merge_copy(
        copy, projects, resolutions=resolution.resolutions, allow_corrected=True
    )

    assert applied.applied, applied.headline
    assert np.array_equal(tifffile.imread(str(ours)), wanted)


def test_a_label_and_its_state_are_one_decision(tmp_path: Path):
    """They resolve under one key, so two controls over one key would be a
    window lying about what a click did."""
    from am_segmentation_manager.operations.selection import transition_records
    from am_segmentation_manager.scanner import Scanner

    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy, ours, _theirs = _conflicting_copy(tmp_path, root, projects)

    # Move the state differently on each side as well, on the same group.
    theirs_projects = Scanner(copy).build_index().all_projects()
    transition_records(
        [r for p in theirs_projects for r in p.image_records], "deselected"
    )
    transition_records(
        [r for p in projects for r in p.image_records], "corrected"
    )

    report = controller.merge_copy(copy, projects, dry_run=True)

    keyed = [row for row in report.conflicts if row.key.endswith(ours.name)]
    assert len(keyed) == 1, "the label and its state came out as two rows"


def test_deciding_lets_the_merge_go_ahead(tmp_path: Path):
    """The decision goes into the plan rather than onto one already made."""
    controller = HistoryController()
    root, projects = _recorded_collection(tmp_path)
    controller.set_start_path(root)
    copy, ours, theirs = _conflicting_copy(tmp_path, root, projects)
    wanted = tifffile.imread(str(theirs))

    blocked = controller.merge_copy(copy, projects, dry_run=True)
    decided = {row.key: "theirs" for row in blocked.conflicts}

    applied = controller.merge_copy(
        copy, projects, resolutions=decided, allow_corrected=True
    )

    assert applied.applied, applied.headline
    assert np.array_equal(tifffile.imread(str(ours)), wanted)
