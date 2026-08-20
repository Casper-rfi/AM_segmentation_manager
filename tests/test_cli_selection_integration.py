"""Command-line curation, end to end.

State is declared in the stage's file, so what these check is what the file
says afterwards and which records the scope actually reached -- not which
folder anything sits in.
"""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.cli import main
from am_segmentation_manager.state import read_state
from conftest import build_dataset


def declared(stage: Path) -> dict[str, str]:
    """Return what a stage's state file declares."""
    return dict(read_state(stage).states)


def scope(root: Path, *extra: str) -> list[str]:
    """The scope arguments every curation command needs."""
    return [
        "--start-path", str(root), "--project", "all", "--lamella", "all",
        "--stage", "all", "--image", "all", "--modality", "SEM", *extra,
    ]


def test_deselect_then_select_declares_and_clears(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, modalities=("SEM",), acquisitions=1)
    name = dataset.filenames["SEM"][0]

    assert main(["deselect", *scope(tmp_path)]) == 0
    assert declared(dataset.stage) == {
        f"{dataset.label_dir('SEM').name}/{name}": "deselected"
    }

    assert main(["select", *scope(tmp_path)]) == 0
    assert declared(dataset.stage) == {}


def test_nothing_moves(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, modalities=("SEM",), acquisitions=1)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.tif"))

    assert main(["deselect", *scope(tmp_path)]) == 0

    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.tif")) == before
    assert declared(dataset.stage)


def test_numeric_scope_values_reach_only_the_named_images(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, modalities=("SEM",), acquisitions=3)
    first, second, third = dataset.filenames["SEM"]
    labels = dataset.label_dir("SEM").name

    assert main([
        "deselect", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "01", "--stage", "01", "--image", "000-001",
        "--modality", "SEM",
    ]) == 0

    assert declared(dataset.stage) == {
        f"{labels}/{first}": "deselected",
        f"{labels}/{second}": "deselected",
    }
    assert f"{labels}/{third}" not in declared(dataset.stage)


def test_an_unscoped_command_refuses_and_changes_nothing(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, modalities=("SEM",), acquisitions=2)

    assert main(["deselect", "--start-path", str(tmp_path), "--modality", "SEM"]) == 2

    assert declared(dataset.stage) == {}


def test_only_the_named_modality_changes(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path, acquisitions=1)

    assert main(["deselect", *scope(tmp_path, "--modality", "FIB")]) == 0

    entries = declared(dataset.stage)
    assert len(entries) == 1
    assert next(iter(entries)).startswith(dataset.label_dir("FIB").name)


def test_the_state_filter_leaves_corrected_images_alone(tmp_path: Path) -> None:
    """``--state selected`` restricts the scope to one workflow state, so an
    image already marked corrected is not swept up by a bulk deselect."""
    dataset = build_dataset(tmp_path, modalities=("SEM",), acquisitions=2)
    first, second = dataset.filenames["SEM"]
    labels = dataset.label_dir("SEM").name

    assert main([
        "correct", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "000",
        "--modality", "SEM",
    ]) == 0

    assert main(["deselect", *scope(tmp_path, "--state", "selected")]) == 0

    assert declared(dataset.stage) == {
        f"{labels}/{first}": "corrected",
        f"{labels}/{second}": "deselected",
    }


def test_a_project_is_reachable_by_name_and_by_number(tmp_path: Path) -> None:
    """The project directory carries a number; experiment.yaml carries a name."""
    project = tmp_path / "001-project-folder"
    stage = project / "01-lamella" / "stage"
    sem = stage / "sem"
    sem.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: Human project name")
    (sem / "img1.tif").write_text("image")

    arguments = [
        "--start-path", str(tmp_path), "--lamella", "01", "--stage", "01",
        "--image", "all", "--modality", "SEM",
    ]
    assert main(["deselect", "--project", "Human project name", *arguments]) == 0
    assert declared(stage) == {"sem_labels/img1.tif": "deselected"}

    assert main(["select", "--project", "001", *arguments]) == 0
    assert declared(stage) == {}
