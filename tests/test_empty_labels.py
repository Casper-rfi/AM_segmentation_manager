"""Telling an annotation from an empty label placeholder.

The external segmentation software writes a label for every image in its input
folder, so a label file existing is not evidence that anyone annotated it. An
empty label is worse than a missing one for training: it pairs an image with a
blank mask, teaching a model that the structure is background.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from am_segmentation_manager import parallel
from am_segmentation_manager.cli import main
from am_segmentation_manager.operations import label_io
from am_segmentation_manager.operations.label_io import (
    ANNOTATED,
    EMPTY,
    MISSING,
    UNREADABLE,
    classify_labels,
    is_label_file_empty,
    read_emptiness,
)
from am_segmentation_manager.state import read_state
from am_segmentation_manager.survey import survey_collection

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile", "cv2")
)
pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="optional image libs not installed"
)

# Imported only when present: these are optional extras, and the whole module
# is skipped above when they are missing.
if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile


def _write_label(path: Path, annotated: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((32, 48), dtype=np.uint8)
    if annotated:
        array[:10] = 2
    tifffile.imwrite(str(path), array)
    return path


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.full((32, 48), 100, dtype=np.uint8))
    return path


def make_project(root: Path) -> Path:
    """A project whose SEM label is annotated and whose FIB label is empty.

    Modelled on the real case: labels were added for both modalities, but the
    FIB one is the placeholder the segmentation software wrote.
    """
    project = root / "project"
    stage = project / "01-lamella" / "adaptive_polish"
    _write_image(stage / "sem" / "img_000_electron.tif")
    _write_image(stage / "fib" / "img_000_ion.tif")
    _write_label(stage / "sem_labels" / "img_000_electron.tif", annotated=True)
    _write_label(stage / "fib_labels" / "img_000_ion.tif", annotated=False)
    (project / "experiment.yaml").write_text("name: project\n", encoding="utf-8")
    return project


# ---------------------------------------------------------------------------
# Reading a label file
# ---------------------------------------------------------------------------


def test_classify_labels_keeps_the_three_findings_apart():
    """Each needs a different action -- annotate it, find where the label went,
    or work out why the file will not open -- so they are not one count."""
    from am_segmentation_manager.models import ImageRecord

    states = {"a": MISSING, "b": EMPTY, "c": UNREADABLE, "d": ANNOTATED}
    records = [ImageRecord(image_path=Path(name)) for name in states]

    found = classify_labels(records, lambda r: states[r.image_path.name])

    assert [r.image_path.name for r in found.missing] == ["a"]
    assert [r.image_path.name for r in found.empty] == ["b"]
    assert [r.image_path.name for r in found.unreadable] == ["c"]


def test_an_annotated_set_is_falsy():
    """The caller asks "is anything wrong here" before building a dialog."""
    from am_segmentation_manager.models import ImageRecord

    records = [ImageRecord(image_path=Path("a"))]

    assert not classify_labels(records, lambda r: ANNOTATED)


def test_the_summary_counts_only_what_was_found():
    """A dialog saying "0 with an unreadable label" invites the reader to work
    out that it means nothing."""
    from am_segmentation_manager.models import ImageRecord

    records = [ImageRecord(image_path=Path("a")), ImageRecord(image_path=Path("b"))]

    summary = classify_labels(records, lambda r: MISSING).summary()

    assert summary == "2 with no label"


def test_classify_labels_reads_the_labels_when_given_no_lookup(tmp_path: Path):
    """The default has to be the real check: a caller with no cache still gets
    an answer off the disk rather than an empty result."""
    from am_segmentation_manager.models import ImageRecord

    image = _write_image(tmp_path / "sem" / "img.tif")
    label = _write_label(tmp_path / "sem_labels" / "img.tif", annotated=False)

    found = classify_labels([ImageRecord(image_path=image, labels=[label])])

    assert [r.image_path for r in found.empty] == [image]


def test_classify_labels_reads_each_label_once_and_together(tmp_path: Path, label_batches):
    """Read one record at a time, a share's labels cost a round trip each in turn,
    and a label shared by several conditions is read once per condition."""
    from am_segmentation_manager.models import ImageRecord

    shared = _write_label(tmp_path / "sem_labels" / "scene.tif", annotated=True)
    other = _write_label(tmp_path / "sem_labels" / "other.tif", annotated=False)
    records = [
        ImageRecord(image_path=tmp_path / "sem" / name, labels=[label])
        for name, label in (("a.tif", shared), ("b.tif", shared), ("c.tif", other))
    ]
    found = classify_labels(records)

    assert label_batches == [[other, shared]]
    assert [record.image_path.name for record in found.empty] == ["c.tif"]


def test_labels_read_in_worker_processes_answer_as_one_read_here(tmp_path: Path, monkeypatch):
    """A label missing or damaged is still None there, never an error that stops the batch."""
    paths = [
        _write_label(tmp_path / "annotated.tif", annotated=True),
        _write_label(tmp_path / "empty.tif", annotated=False),
        tmp_path / "missing.tif",
    ]
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"II*\x00 not a label any reader can decode")
    paths.append(corrupt)
    expected = {path: is_label_file_empty(path) for path in paths}
    parallel.shutdown_pool()
    monkeypatch.setattr(label_io, "LABEL_POOL_FILES", 0)
    monkeypatch.setattr(parallel, "POOL_WORKERS", 2)
    try:
        read = read_emptiness(paths)
        assert parallel._pool is not None
    finally:
        parallel.shutdown_pool()

    assert read == expected


def test_an_all_background_label_is_empty(tmp_path: Path):
    path = _write_label(tmp_path / "label.tif", annotated=False)
    assert is_label_file_empty(path) is True


def test_an_annotated_label_is_not_empty(tmp_path: Path):
    path = _write_label(tmp_path / "label.tif", annotated=True)
    assert is_label_file_empty(path) is False


def test_a_missing_label_is_unknown_not_empty(tmp_path: Path):
    """None and True must stay distinct: unreadable is not the same as blank."""
    assert is_label_file_empty(tmp_path / "absent.tif") is None


def test_an_unreadable_label_is_unknown(tmp_path: Path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a tiff")
    assert is_label_file_empty(path) is None


@pytest.mark.parametrize(
    ("where", "expected"),
    [("second page", None), ("green channel", None), ("first page only", False)],
)
def test_a_label_annotated_beyond_its_first_plane_is_unreadable(tmp_path: Path, where, expected):
    """Only the first plane was read, so annotation on another was judged empty."""
    import numpy as np
    import tifffile

    path = tmp_path / "label.tif"
    if where == "green channel":
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[4:8, 4:8, 1] = 2
    else:
        pixels = np.zeros((2, 16, 16), dtype=np.uint8)
        pixels[1 if where == "second page" else 0, 4:8, 4:8] = 2
    tifffile.imwrite(path, pixels)

    assert is_label_file_empty(path) is expected


# ---------------------------------------------------------------------------
# The survey
# ---------------------------------------------------------------------------

def test_the_survey_reports_empty_labels(tmp_path: Path):
    make_project(tmp_path)

    survey = survey_collection(tmp_path)
    findings = [f for f in survey.all_findings if f.category == "empty-labels"]

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "fib_labels" in str(findings[0].path)


def test_the_survey_reads_a_stage_s_labels_together(tmp_path: Path, label_batches):
    """Read one at a time, a share's labels cost a round trip each in turn, and
    one image set is often too few to share among the worker processes."""
    make_project(tmp_path)

    survey_collection(tmp_path)

    assert [sorted(path.name for path in batch) for batch in label_batches] == [
        ["img_000_electron.tif", "img_000_ion.tif"]
    ]


def test_the_survey_counts_annotated_labels_separately(tmp_path: Path):
    make_project(tmp_path)

    survey = survey_collection(tmp_path)
    by_name = {e.image_set.name: e for e in survey.projects[0].image_sets}

    assert by_name["sem"].label_count == 1
    assert by_name["sem"].empty_labels == 0
    assert by_name["sem"].annotated_labels == 1
    assert by_name["fib"].label_count == 1
    assert by_name["fib"].empty_labels == 1
    assert by_name["fib"].annotated_labels == 0


def test_the_label_check_can_be_skipped(tmp_path: Path):
    """Structure-only survey of a very large collection."""
    make_project(tmp_path)

    survey = survey_collection(tmp_path, check_labels=False)

    assert [f for f in survey.all_findings if f.category == "empty-labels"] == []
    assert all(e.empty_labels is None for e in survey.projects[0].image_sets)
    assert all(e.annotated_labels is None for e in survey.projects[0].image_sets)


def test_cli_inspect_layout_can_skip_the_label_check(tmp_path: Path, capsys):
    make_project(tmp_path)

    assert main(["inspect-layout", str(tmp_path), "--no-label-check"]) == 0
    assert "empty-labels" not in capsys.readouterr().out


def test_cli_inspect_layout_reports_empty_labels_by_default(tmp_path: Path, capsys):
    make_project(tmp_path)

    assert main(["inspect-layout", str(tmp_path)]) == 0
    assert "empty-labels" in capsys.readouterr().out


def _correct(root: Path, modality: str) -> list[str]:
    """The whole collection, which this command makes you say explicitly."""
    return [
        "correct", "--start-path", str(root), "--modality", modality,
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
    ]


def test_cli_correct_warns_about_an_empty_label_but_goes_ahead(tmp_path: Path, capsys):
    """The command line is where a deliberate exception is made, so this
    reports rather than refuses -- but it must not be silent, or a training set
    ends up short with nothing having said so."""
    make_project(tmp_path)

    assert main(_correct(tmp_path, "FIB")) == 0

    output = capsys.readouterr().out
    assert "1 with an empty label" in output
    assert "img_000_ion.tif" in output
    stage = tmp_path / "project/01-lamella/adaptive_polish"
    assert read_state(stage).states == {"fib_labels/img_000_ion.tif": "corrected"}


def test_cli_correct_says_nothing_about_an_annotated_label(tmp_path: Path, capsys):
    make_project(tmp_path)

    assert main(_correct(tmp_path, "SEM")) == 0

    assert "empty label" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Training export
# ---------------------------------------------------------------------------

def _export(tmp_path: Path, *extra: str, modality: str = "all") -> list[dict] | None:
    """Return the rows exported, or None where the export was refused."""
    import csv as csv_module

    output = tmp_path / "out"
    code = main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--modality", modality, "-o", str(output), "--split", "1.0",
        # Called more than once per test into one folder, deliberately.
        "--overwrite", *extra,
    ])
    if code == 2:
        return None
    assert code == 0
    with (output / "dataset_train.csv").open(encoding="utf-8") as handle:
        return list(csv_module.DictReader(handle))


@pytest.mark.parametrize("command", [
    ["create-training-csv", "-o", "{out}", "--split", "1.0"],
    ["export-metadata-csv", "-o", "{out}/images.csv", "--no-headers"],
    ["stats", "--by", "label"],
], ids=lambda command: command[0])
def test_a_command_reads_each_label_once_and_together(
    tmp_path: Path, monkeypatch, label_batches, command
):
    """Read one at a time, a share's labels cost a round trip each in turn; and
    a training export asks of each label twice, for its rows and its images file."""
    make_project(tmp_path)
    reads: list[Path] = []
    real = label_io.is_label_file_empty

    def counted(path: Path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(label_io, "is_label_file_empty", counted)
    (tmp_path / "out").mkdir()

    assert main([
        *(part.format(out=tmp_path / "out") for part in command),
        "--start-path", str(tmp_path), "--project", "all", "--lamella", "all",
        "--stage", "all", "--image", "all", "--state", "all", "--modality", "all",
    ]) == 0

    assert [sorted(path.name for path in batch) for batch in label_batches] == [
        ["img_000_electron.tif", "img_000_ion.tif"]
    ]
    assert sorted(reads) == sorted({path for batch in label_batches for path in batch})


def test_an_empty_label_is_not_paired_with_its_image(tmp_path: Path):
    """A blank mask actively teaches the model the structure is background. In
    a paired row that shows as an empty cell: the acquisition is still exported
    for the beam that was annotated."""
    (rows,) = _export(tmp_path) if make_project(tmp_path) else []

    assert "electron" in rows["sem_image"]
    assert "electron" in rows["sem_label"]
    assert "ion" in rows["fib_image"], "the image is still recorded"
    assert rows["fib_label"] == "", "the blank label is not"


def test_an_empty_label_is_excluded_from_a_single_beam_export(tmp_path: Path):
    """With only the beam whose label is blank, there is nothing left to write."""
    make_project(tmp_path)

    assert _export(tmp_path, modality="FIB") is None
    assert len(_export(tmp_path, modality="SEM") or []) == 1


def test_an_empty_label_can_be_exported_deliberately(tmp_path: Path):
    make_project(tmp_path)

    (row,) = _export(tmp_path, "--include-empty-labels")

    assert row["fib_label"] != ""


def test_the_exclusion_is_reported_not_silent(tmp_path: Path, capsys):
    make_project(tmp_path)

    _export(tmp_path)

    output = capsys.readouterr().out
    assert "only background" in output
    assert "--include-empty-labels" in output


def _make_fib_label_unreadable(root: Path) -> None:
    label = root / "project" / "01-lamella" / "adaptive_polish" / "fib_labels" / "img_000_ion.tif"
    label.write_bytes(b"II*\x00 not a label any reader can decode")


def test_a_label_that_cannot_be_read_is_not_paired_even_when_blank_ones_are_asked_for(
    tmp_path: Path, capsys
):
    """Nobody can say whether it is blank, so it is not a known negative either."""
    make_project(tmp_path)
    _make_fib_label_unreadable(tmp_path)

    (row,) = _export(tmp_path, "--include-empty-labels")

    assert row["fib_label"] == ""
    assert "electron" in row["sem_label"]
    assert "could not be read" in capsys.readouterr().out


def test_the_manifest_counts_the_labels_that_could_not_be_read(tmp_path: Path):
    from am_segmentation_manager.io.yaml import read_yaml

    make_project(tmp_path)
    _make_fib_label_unreadable(tmp_path)
    _export(tmp_path)

    manifest = read_yaml(tmp_path / "out" / "dataset_manifest.yaml")
    assert manifest["counts"]["unreadable_labels_left_out"] == 1


def test_without_the_image_stack_no_label_is_exported_and_the_export_says_why(
    tmp_path: Path, monkeypatch, capsys
):
    """A core install cannot open a label, so cannot tell a placeholder from
    annotation; exporting them all would train on every blank mask."""
    from am_segmentation_manager.operations import label_io, training_export

    make_project(tmp_path)
    monkeypatch.setattr(label_io, "has_image_stack_support", lambda: False)
    monkeypatch.setattr(training_export, "has_image_stack_support", lambda: False)

    assert _export(tmp_path, modality="SEM") is None
    output = capsys.readouterr().out
    assert "could not be read" in output
    assert "'gui' extra" in output


def test_the_export_preview_names_labels_that_could_not_be_read():
    from am_segmentation_manager.gui.export_controller import ExportPreview

    summary = ExportPreview(matched=3, annotated=2, unreadable_labels=1, exportable=2).summary()

    assert "1 with an unreadable label" in summary


def test_a_damaged_label_tifffile_reads_as_no_pixels_is_unknown_not_empty(tmp_path: Path):
    """tifffile answers some damaged files with an empty array rather than an
    error; counted as a blank placeholder, one would be replaced unasked."""
    label = tmp_path / "sem_labels" / "img.tif"
    label.parent.mkdir()
    label.write_bytes(b"II*\x00 not a label any reader can decode")

    assert is_label_file_empty(label) is None
