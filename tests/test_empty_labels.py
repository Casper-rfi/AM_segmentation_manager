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

from am_segmentation_manager.cli import main
from am_segmentation_manager.operations.label_io import (
    ANNOTATED,
    EMPTY,
    MISSING,
    UNREADABLE,
    classify_labels,
    is_label_file_empty,
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

def _export(tmp_path: Path, *extra: str) -> list[str]:
    output = tmp_path / "out"
    assert main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--modality", "all", "-o", str(output), "--split", "1.0", *extra,
    ]) == 0
    rows = (output / "dataset_train.csv").read_text().strip().splitlines()
    return rows[1:]


def test_an_empty_label_is_excluded_from_training_export(tmp_path: Path):
    """A blank mask actively teaches the model the structure is background."""
    make_project(tmp_path)

    rows = _export(tmp_path)

    assert len(rows) == 1
    assert "electron" in rows[0]
    assert "ion" not in rows[0]


def test_an_empty_label_can_be_exported_deliberately(tmp_path: Path):
    make_project(tmp_path)

    rows = _export(tmp_path, "--include-empty-labels")

    assert len(rows) == 2


def test_the_exclusion_is_reported_not_silent(tmp_path: Path, capsys):
    make_project(tmp_path)

    _export(tmp_path)

    output = capsys.readouterr().out
    assert "only background" in output
    assert "--include-empty-labels" in output
