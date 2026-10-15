"""Exporting training data from images that share a label.

Several imaging conditions of one acquisition are near-identical images sharing
one label. That makes two things matter for a training export: how many of them
to include by default, and -- if several are included -- that they never end up
on opposite sides of the train/validation split, which would score the model on
scenes it has effectively already seen.
"""
from __future__ import annotations

import csv
from pathlib import Path

from am_segmentation_manager.cli import main
from am_segmentation_manager.state import read_state
from conftest import write_label

SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


def make_project(root: Path, acquisitions: int = 1) -> Path:
    """Three SEM conditions sharing one label per acquisition, plus a FIB image."""
    project = root / "project"
    lamella = project / "lam_010"
    for directory in (*SEM_VARIANTS, "FIB"):
        (lamella / directory).mkdir(parents=True, exist_ok=True)
        for number in range(acquisitions):
            (lamella / directory / f"img_{number:04d}.tif").write_bytes(b"")
    # Annotated, because an export pairs only a label it can see is not blank.
    for directory in ("labels", "labels_FIB"):
        for number in range(acquisitions):
            write_label(lamella / directory / f"img_{number:04d}.tif", "annotated")
    (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
    return project


def export(
    tmp_path: Path, *extra: str, split: str = "1.0", modality: str = "SEM"
) -> list[dict]:
    """Export and read back the training file.

    One modality by default: which conditions of a scene are exported is a
    question about one beam, and a paired export answers it in the sem_ columns
    where a plain ``image`` column reads more clearly. The paired shape has its
    own test below.
    """
    output = tmp_path / "out"
    assert main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--modality", modality,
        "-o", str(output), "--split", split, *extra,
    ]) == 0
    with (output / "dataset_train.csv").open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Which conditions are exported
# ---------------------------------------------------------------------------

def test_one_condition_per_scene_is_exported_by_default(tmp_path: Path):
    """Otherwise a dataset quietly gains several near-identical copies."""
    make_project(tmp_path)

    rows = export(tmp_path)

    assert len(rows) == 1
    assert not any("fi_1" in row["image"] or "fi_4" in row["image"] for row in rows)


def test_both_modalities_put_the_two_beams_on_one_row(tmp_path: Path):
    """A file mixing SEM and FIB rows under one ``image`` column is not
    something anyone trains on; side by side, either beam can be trained from
    the one file, or the two together."""
    make_project(tmp_path)

    rows = export(tmp_path, modality="all")

    assert len(rows) == 1
    assert list(rows[0]) == ["sem_image", "sem_label", "fib_image", "fib_label"]
    assert "SEM_8" in rows[0]["sem_image"]
    assert "FIB" in rows[0]["fib_image"]


def test_every_condition_repeats_the_other_beam(tmp_path: Path):
    """The cross product. Three SEM conditions against one FIB image are three
    rows repeating the FIB path -- which "one image per label" is there to
    collapse where it is not wanted."""
    make_project(tmp_path)

    rows = export(tmp_path, "--conditions", "all", modality="all")

    assert len(rows) == len(SEM_VARIANTS)
    assert len({row["sem_image"] for row in rows}) == len(SEM_VARIANTS)
    assert len({row["fib_image"] for row in rows}) == 1


def test_the_default_works_where_no_unsuffixed_directory_exists(tmp_path: Path):
    """The oldest layout has no plain sem/: every SEM set is a variant.

    Filtering for "the base directory" would select no SEM image at all, which
    is the layout this whole exercise exists to support.
    """
    make_project(tmp_path)

    rows = export(tmp_path)

    sem_rows = [row for row in rows if "SEM" in row["image"]]
    assert len(sem_rows) == 1
    assert "SEM_8/" in sem_rows[0]["image"]


def test_the_primary_a_project_names_is_the_condition_exported(tmp_path: Path):
    project = make_project(tmp_path)
    (project / "AM_families.yaml").write_text(
        "primary:\n  SEM: SEM_8_fi_4\n", encoding="utf-8"
    )

    rows = export(tmp_path)

    assert [Path(row["image"]).parent.name for row in rows] == ["SEM_8_fi_4"]


def test_every_condition_can_be_exported_deliberately(tmp_path: Path):
    """Five conditions of one scene are five valid training pairs."""
    make_project(tmp_path)

    rows = export(tmp_path, "--conditions", "all")

    assert len(rows) == len(SEM_VARIANTS)
    assert sum("SEM_8" in row["image"] for row in rows) == 3


def test_a_single_named_condition_can_be_exported(tmp_path: Path):
    """The command default must not be appended to an explicit choice."""
    make_project(tmp_path)

    rows = export(tmp_path, "--variant", "8_fi_1")

    assert len(rows) == 1
    assert "SEM_8_fi_1" in rows[0]["image"]


def test_state_changes_still_cover_every_condition_by_default(tmp_path: Path):
    """The export default must not leak into commands that follow the label."""
    project = make_project(tmp_path)

    assert main([
        "deselect", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--modality", "SEM",
    ]) == 0

    declared = read_state(project / "lam_010").states
    assert declared == {"labels/img_0000.tif": "deselected"}


# ---------------------------------------------------------------------------
# The train/validation split
# ---------------------------------------------------------------------------

def test_an_end_to_end_export_does_not_leak_a_scene_across_the_split(tmp_path: Path):
    make_project(tmp_path, acquisitions=10)
    output = tmp_path / "out"

    assert main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--conditions", "all", "--include-empty-labels", "--modality", "SEM",
        "-o", str(output), "--split", "0.8",
    ]) == 0

    def labels(name: str) -> set[str]:
        with (output / name).open() as handle:
            return {row["label"] for row in csv.DictReader(handle)}

    assert labels("dataset_train.csv") & labels("dataset_val.csv") == set()
