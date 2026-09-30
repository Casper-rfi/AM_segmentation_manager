"""Tests for ``scripts/remove_duplicate_condition_labels.py``.

What it must not get wrong is removing work: anything not provably what the
model wrote stops the run with nothing removed. The trees are small stages of
one family split across private label and predictions directories.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from remove_duplicate_condition_labels import main, plan_repairs

from am_segmentation_manager.history import Recorder, read_log
from am_segmentation_manager.layout import build_image_sets
from am_segmentation_manager.operations.label_io import label_fields
from am_segmentation_manager.state import read_state, write_state
from conftest import start_history

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

IMAGES = ("img_0000.tif", "img_0001.tif")


@pytest.fixture
def recorder() -> Recorder:
    return Recorder(by="casper", by_source="flag")


def write_label(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((16, 24), dtype=np.uint8)
    pixels[2:6, 3:9] = value
    tifffile.imwrite(path, pixels)
    return path


def convert(path: Path, value: int, recorder: Recorder) -> Path:
    """Write a label as a conversion does, and record it."""
    write_label(path, value)
    fields = label_fields(tifffile.imread(path))
    recorder.record(path, "label-converted", model="m.pth", prediction_kind="am_seg", **fields)
    return path


def predict(directory: Path, names=IMAGES) -> None:
    """Write am-seg predictions and the record naming them."""
    own = directory / "am_seg"
    for name in names:
        write_label(own / name, 1)
    lines = ["predictions:"] + [f"  {name}:\n    image: {name}\n    model: m.pth" for name in names]
    (own / "am_seg_predictions.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def split(tmp_path: Path, recorder: Recorder) -> Path:
    """A family of two conditions, each predicted and converted into its own directories.

    ``SEM_8_labels/img_0000.tif`` has since been proofread.
    """
    project = tmp_path / "project"
    start_history(project)
    stage = project / "lam_000"
    for condition, value in (("SEM_8", 2), ("SEM_8_fi_1", 3)):
        for name in IMAGES:
            write_label(stage / condition / name, 50)
            convert(stage / f"{condition}_labels" / name, value, recorder)
        predict(stage / f"{condition}_predictions")
    proofread = stage / "SEM_8_labels" / IMAGES[0]
    write_label(proofread, 4)
    recorder.record(proofread, "label-saved", **label_fields(tifffile.imread(proofread)))
    return tmp_path


def stage_of(root: Path) -> Path:
    return root / "project" / "lam_000"


def test_the_directory_holding_proofread_work_is_kept(split: Path):
    (repair,) = plan_repairs(split)

    assert not repair.problems
    assert repair.kept_labels == stage_of(split) / "SEM_8_labels"
    assert repair.kept_predictions == stage_of(split) / "SEM_8_predictions"
    removed = {r.path.relative_to(stage_of(split)).as_posix() for r in repair.removals}
    assert "SEM_8_fi_1_labels/img_0000.tif" in removed
    assert "SEM_8_fi_1_predictions/am_seg/img_0001.tif" in removed
    assert "SEM_8_fi_1_predictions/am_seg/am_seg_predictions.yaml" in removed
    assert not any(path.startswith("SEM_8_labels") for path in removed)


def test_with_no_proofread_work_the_primary_condition_is_kept(tmp_path: Path, recorder):
    project = tmp_path / "project"
    start_history(project)
    stage = project / "lam_000"
    for condition in ("SEM_8", "SEM_TLD"):
        write_label(stage / condition / IMAGES[0], 50)
        convert(stage / f"{condition}_labels" / IMAGES[0], 2, recorder)

    (repair,) = plan_repairs(tmp_path)

    assert repair.kept_labels == stage / "SEM_8_labels"


def test_the_family_directory_is_kept_over_a_private_one(tmp_path: Path, recorder):
    project = tmp_path / "project"
    start_history(project)
    stage = project / "lam_000"
    for condition in ("SEM", "SEM_8"):
        write_label(stage / condition / IMAGES[0], 50)
        convert(stage / f"{condition}_labels" / IMAGES[0], 2, recorder)

    (repair,) = plan_repairs(tmp_path)

    assert repair.kept_labels == stage / "SEM_labels"


def test_a_label_edited_in_both_directories_stops_the_run(split: Path, recorder):
    edited = stage_of(split) / "SEM_8_fi_1_labels" / IMAGES[1]
    write_label(edited, 9)
    recorder.record(edited, "label-saved", **label_fields(tifffile.imread(edited)))

    (repair,) = plan_repairs(split)

    assert any("more than one" in problem for problem in repair.problems)


def test_a_state_change_after_the_conversion_leaves_it_untouched(split: Path, recorder):
    """A deselection records no pixels; the state itself is checked on its own."""
    stage = stage_of(split)
    for directory in ("SEM_8_labels", "SEM_8_fi_1_labels"):
        recorder.record(stage / directory / IMAGES[1], "state-changed", old="selected", new="deselected")
    write_state(read_state(stage), {
        "SEM_8_labels/img_0001.tif": "deselected",
        "SEM_8_fi_1_labels/img_0001.tif": "deselected",
    })

    (repair,) = plan_repairs(split)

    assert not repair.problems


def test_a_label_changed_without_a_record_is_not_removed(split: Path):
    write_label(stage_of(split) / "SEM_8_fi_1_labels" / IMAGES[1], 7)

    (repair,) = plan_repairs(split)

    assert repair.problems


def test_a_removed_label_declared_corrected_stops_the_run(split: Path):
    stage = stage_of(split)
    write_state(read_state(stage), {"SEM_8_fi_1_labels/img_0001.tif": "corrected"})

    (repair,) = plan_repairs(split)

    assert repair.problems


def test_a_state_the_kept_label_does_not_share_stops_the_run(split: Path):
    stage = stage_of(split)
    write_state(read_state(stage), {"SEM_8_fi_1_labels/img_0001.tif": "deselected"})

    (repair,) = plan_repairs(split)

    assert any("declared deselected" in problem for problem in repair.problems)


def test_a_label_only_the_removed_directory_holds_stops_the_run(split: Path):
    (stage_of(split) / "SEM_8_labels" / IMAGES[1]).unlink()

    (repair,) = plan_repairs(split)

    assert any("holds no label for its image" in problem for problem in repair.problems)


def test_a_prediction_its_record_does_not_name_stops_the_run(split: Path):
    write_label(stage_of(split) / "SEM_8_fi_1_predictions" / "am_seg" / "stray.tif", 1)

    (repair,) = plan_repairs(split)

    assert any("stray.tif" in problem for problem in repair.problems)


def test_pipeline_output_in_a_removed_directory_stops_the_run(split: Path):
    write_label(stage_of(split) / "SEM_8_fi_1_predictions" / "raw" / IMAGES[0], 1)

    (repair,) = plan_repairs(split)

    assert any("raw" in problem for problem in repair.problems)


def test_a_preview_removes_nothing(split: Path):
    before = sorted(split.rglob("*"))

    assert main([str(split)]) == 0
    assert sorted(split.rglob("*")) == before


def test_a_problem_anywhere_removes_nothing_anywhere(split: Path, tmp_path: Path):
    write_label(stage_of(split) / "SEM_8_fi_1_labels" / IMAGES[1], 7)
    before = sorted(split.rglob("*"))

    assert main([str(split), "--apply", "--manifest", str(tmp_path / "m.csv")]) == 2
    assert sorted(split.rglob("*")) == before


def test_apply_leaves_one_directory_the_whole_family_reads(split: Path, tmp_path: Path):
    stage = stage_of(split)
    write_state(read_state(stage), {
        "SEM_8_labels/img_0001.tif": "deselected",
        "SEM_8_fi_1_labels/img_0001.tif": "deselected",
    })
    manifest = tmp_path / "manifest.csv"

    assert main([str(split), "--apply", "--manifest", str(manifest)]) == 0

    assert not (stage / "SEM_8_fi_1_labels").exists()
    assert not (stage / "SEM_8_fi_1_predictions").exists()
    assert {image_set.labels_directory for image_set in build_image_sets(stage)} == {
        stage / "SEM_8_labels"
    }
    assert read_state(stage).states == {"SEM_8_labels/img_0001.tif": "deselected"}
    with manifest.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 5  # two labels, two predictions, a record


def test_every_removed_label_is_recorded_as_removed(split: Path, tmp_path: Path):
    assert main([str(split), "--apply", "--manifest", str(tmp_path / "m.csv")]) == 0

    removals = [
        event for event in read_log(split / "project").events if event.kind == "label-removed"
    ]
    assert sorted(event.target for event in removals) == [
        "lam_000/SEM_8_fi_1_labels/img_0000.tif", "lam_000/SEM_8_fi_1_labels/img_0001.tif",
    ]
    assert all("SEM_8_labels/" in event.data["reason"] for event in removals)
