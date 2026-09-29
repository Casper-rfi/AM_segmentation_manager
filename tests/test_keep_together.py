"""What a training split keeps on one side: an acquisition, a lamella or a project.

Successive images of one lamella are near-duplicates, so a split that divides a
lamella scores a model on lamellae it was trained on. The level is chosen, the
lamella by default, and whatever it is no group lands on both sides.
"""
from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath

import pytest
import yaml

from am_segmentation_manager.aggregate import BREAKDOWNS, project_lookup, split_grouping
from am_segmentation_manager.cli import main
from am_segmentation_manager.cli.main import build_parser
from am_segmentation_manager.frames import group_by_acquisition
from am_segmentation_manager.operations.csv import build_training_rows
from am_segmentation_manager.operations.training_split import (
    DEFAULT_KEEP_TOGETHER,
    Grouping,
)
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

PROJECTS = ("alpha", "beta", "gamma")


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """Three projects of three lamellae each, their lamellae named alike."""
    pytest.importorskip("tifffile")
    root = tmp_path / "collection"
    for name in PROJECTS:
        build_dataset(
            root, name=name, lamellae=3, acquisitions=4, labels="annotated",
            modalities=("SEM", "FIB"),
        )
    return root


def _export(root: Path, output: Path, *extra: str, modality: str = "SEM") -> Path:
    assert main([
        "create-training-csv", "--start-path", str(root), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--modality", modality, "-o", str(output), *extra,
    ]) == 0
    return output


def _sides(output: Path, depth: int, column: str = "image") -> tuple[set, set]:
    """Return the groups each side holds, a group being the path's first *depth* parts."""

    def groups(name: str) -> set[tuple[str, ...]]:
        with (output / name).open(newline="") as handle:
            return {
                PurePosixPath(row[column].replace("\\", "/")).parts[:depth]
                for row in csv.DictReader(handle)
                if row[column]
            }

    return groups("dataset_train.csv"), groups("dataset_val.csv")


def _manifest(output: Path) -> dict:
    return yaml.safe_load((output / "dataset_manifest.yaml").read_text())


# ---------------------------------------------------------------------------
# No group on both sides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("level", "depth"), [("lamella", 2), ("project", 1)])
@pytest.mark.parametrize("modality", ["SEM", "all"])
def test_no_group_is_on_both_sides_of_the_split(collection, tmp_path, level, depth, modality):
    column = "sem_image" if modality == "all" else "image"
    output = _export(
        collection, tmp_path / "out", "--keep-together", level, "--split", "0.6",
        modality=modality,
    )

    train, validation = _sides(output, depth, column)

    assert train and validation
    assert train & validation == set()
    assert _manifest(output)["split"]["keep_together"] == level


def test_a_split_keeps_lamellae_together_unless_told_otherwise(collection, tmp_path):
    output = _export(collection, tmp_path / "out", "--split", "0.6")

    train, validation = _sides(output, 2)

    assert train & validation == set()
    assert _manifest(output)["split"]["keep_together"] == DEFAULT_KEEP_TOGETHER == "lamella"


def test_keeping_acquisitions_together_divides_lamellae(collection, tmp_path):
    """The finest level is what every export did before the choice existed."""
    output = _export(
        collection, tmp_path / "out", "--keep-together", "acquisition", "--split", "0.6"
    )

    train, validation = _sides(output, 2)

    assert train & validation
    assert _manifest(output)["split"]["keep_together"] == "acquisition"


# ---------------------------------------------------------------------------
# What a group is
# ---------------------------------------------------------------------------

def test_lamellae_of_one_name_in_two_projects_are_two_groups(collection):
    """Almost every project has an ``01-``; merged, they would be one group."""
    projects = Scanner(collection).build_index().all_projects()
    records = [record for project in projects for record in project.image_records]
    key = split_grouping("lamella", project_lookup(projects)).key
    assert key is not None

    lamellae = {(record.project, record.lamella) for record in records}
    groups = {key(record) for record in records}

    assert len({lamella for _, lamella in lamellae}) == 3
    assert len(groups) == len(lamellae) == 9


def test_a_split_groups_by_what_the_breakdown_counts(collection):
    """So the table counted by lamella shows the groups the split keeps whole."""
    projects = Scanner(collection).build_index().all_projects()
    lookup = project_lookup(projects)
    records = [record for project in projects for record in project.image_records]

    for level, breakdown in (("lamella", "lamella"), ("project", "autolamella-project")):
        key = split_grouping(level, lookup).key
        counted = BREAKDOWNS[breakdown].factory(lookup)
        assert key is not None
        assert all(key(record) == counted(record) for record in records)


def test_the_finest_level_groups_as_the_row_builder_always_has(collection):
    """A label in a single-beam export, an acquisition in a paired one."""
    projects = Scanner(collection).build_index().all_projects()
    records = [record for project in projects for record in project.image_records]
    sem = [record for record in records if record.modality == "SEM"]

    single = build_training_rows(sem, collection, paired=False, is_empty=lambda _: False)
    paired = build_training_rows(records, collection, paired=True, is_empty=lambda _: False)

    assert [row.group for row in single] == [row.values[1] for row in single]
    assert {row.group for row in paired} == {
        acquisition.identity for acquisition in group_by_acquisition(records)
    }


@pytest.mark.parametrize(
    ("level", "key"),
    [("glacier", None), ("lamella", None), ("acquisition", str)],
)
def test_a_grouping_that_cannot_say_what_a_group_is_is_refused(level, key):
    with pytest.raises(ValueError):
        Grouping(level, key)


# ---------------------------------------------------------------------------
# Saying what came of it
# ---------------------------------------------------------------------------

def test_one_group_holding_every_row_is_warned_about(collection, tmp_path, capsys):
    """A split by project over one project cannot fill both sides."""
    output = tmp_path / "out"
    assert main([
        "create-training-csv", "--start-path", str(collection), "--project", "alpha",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--modality", "SEM", "-o", str(output), "--keep-together", "project",
    ]) == 0

    assert "Every row is in one project" in capsys.readouterr().out
    assert _manifest(output)["counts"]["validation"] == 0


def test_the_command_line_and_the_window_keep_the_same_default():
    from am_segmentation_manager.gui.export_controller import TrainingExport
    from am_segmentation_manager.gui.preferences import BY_KEY

    args = build_parser().parse_args(["create-training-csv", "--state", "all"])

    assert args.keep_together == TrainingExport().keep_together == DEFAULT_KEEP_TOGETHER
    assert BY_KEY["keep_together"].fallback == DEFAULT_KEEP_TOGETHER
