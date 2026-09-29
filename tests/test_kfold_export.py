"""Writing K train/validation pairs for cross-validation.

Every row validates in exactly one fold and trains in the others, no group kept
together is in two folds, and the manifest says what each fold holds. Fold files
an earlier export left, and this one would not replace, stop it: a run reading
every fold would take them for this export's.
"""
from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath

import pytest
import yaml

from am_segmentation_manager.cli import main
from am_segmentation_manager.cli.main import build_parser
from conftest import build_dataset

PROJECTS = ("alpha", "beta", "gamma")


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """Three projects of three lamellae each."""
    pytest.importorskip("tifffile")
    root = tmp_path / "collection"
    for name in PROJECTS:
        build_dataset(
            root, name=name, lamellae=3, acquisitions=4, labels="annotated",
            modalities=("SEM",),
        )
    return root


def _export(root: Path, output: Path, *extra: str) -> int:
    return main([
        "create-training-csv", "--start-path", str(root), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--state", "all",
        "--modality", "SEM", "-o", str(output), *extra,
    ])


def _rows(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        return [(row["image"], row["label"]) for row in csv.DictReader(handle)]


def _lamella(row: tuple[str, str]) -> tuple[str, ...]:
    return PurePosixPath(row[0].replace("\\", "/")).parts[:2]


def _manifest(output: Path) -> dict:
    return yaml.safe_load((output / "dataset_manifest.yaml").read_text())


def test_every_row_validates_in_exactly_one_fold_and_trains_in_the_rest(collection, tmp_path):
    output = tmp_path / "out"
    assert _export(collection, output, "--folds", "3") == 0

    validation = [_rows(output / f"dataset_fold{k}_val.csv") for k in (1, 2, 3)]
    everything = sorted(row for fold in validation for row in fold)

    assert len(everything) == len(set(everything)) == 36
    for k, validating in enumerate(validation, start=1):
        training = _rows(output / f"dataset_fold{k}_train.csv")
        assert sorted(training + validating) == everything
        assert not set(training) & set(validating)
    assert not (output / "dataset_train.csv").exists()


def test_no_lamella_is_in_two_folds(collection, tmp_path):
    output = tmp_path / "out"
    assert _export(collection, output, "--folds", "4") == 0

    lamellae = [
        {_lamella(row) for row in _rows(output / f"dataset_fold{k}_val.csv")}
        for k in range(1, 5)
    ]

    assert sum(len(fold) for fold in lamellae) == len(set().union(*lamellae)) == 9


def test_the_manifest_says_what_each_fold_holds(collection, tmp_path):
    output = tmp_path / "out"
    assert _export(collection, output, "--folds", "3", "--keep-together", "project") == 0

    manifest = _manifest(output)

    assert manifest["split"] == {
        "train_ratio": pytest.approx(2 / 3), "seed": 42,
        "keep_together": "project", "folds": 3,
    }
    assert manifest["files"]["fold2_validation"] == "dataset_fold2_val.csv"
    assert len(manifest["files"]) == 6
    assert [fold["fold"] for fold in manifest["folds"]] == [1, 2, 3]
    assert sorted(name for fold in manifest["folds"] for name in fold["validates_on"]) == [
        "alpha", "beta", "gamma",
    ]
    for fold in manifest["folds"]:
        rows = len(_rows(output / f"dataset_fold{fold['fold']}_val.csv"))
        assert fold["validation"] == rows
        assert fold["train"] == manifest["counts"]["rows"] - rows


def test_more_folds_than_groups_is_refused_before_anything_is_written(
    collection, tmp_path, capsys
):
    output = tmp_path / "out"

    assert _export(collection, output, "--folds", "4", "--keep-together", "project") == 2

    assert "cannot fill 4 folds" in capsys.readouterr().out
    assert not output.exists() or not any(output.iterdir())


def test_folds_an_earlier_export_left_are_refused_even_with_overwrite(
    collection, tmp_path, capsys
):
    """Fold 5 would outlive the manifest naming it, and be read as this export's."""
    output = tmp_path / "out"
    assert _export(collection, output, "--folds", "5") == 0
    manifest = (output / "dataset_manifest.yaml").read_text()
    capsys.readouterr()

    assert _export(collection, output, "--folds", "3", "--overwrite") == 2

    reported = capsys.readouterr().out
    assert "dataset_fold4_train.csv" in reported and "--overwrite" not in reported
    assert (output / "dataset_manifest.yaml").read_text() == manifest


def test_a_single_split_is_refused_beside_an_earlier_exports_folds(collection, tmp_path):
    output = tmp_path / "out"
    assert _export(collection, output, "--folds", "2") == 0

    assert _export(collection, output, "--overwrite") == 2
    assert not (output / "dataset_train.csv").exists()


def test_the_same_folds_are_replaced_with_overwrite(collection, tmp_path):
    output = tmp_path / "out"
    assert _export(collection, output, "--folds", "3") == 0
    assert _export(collection, output, "--folds", "3") == 2   # in the way

    assert _export(collection, output, "--folds", "3", "--overwrite", "--seed", "7") == 0
    assert _manifest(output)["split"]["seed"] == 7


@pytest.mark.parametrize("extra", [["--folds", "1"], ["--folds", "three"],
                                   ["--folds", "3", "--split", "0.7"]])
def test_folds_the_command_cannot_honour_are_refused_by_the_parser(extra):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["create-training-csv", "--state", "all", *extra])
