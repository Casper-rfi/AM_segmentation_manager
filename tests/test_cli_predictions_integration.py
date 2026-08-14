"""Integration coverage for selector-based prediction conversion."""
from pathlib import Path

from am_segmentation_manager.cli import main


def _build_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal project with one SEM and one FIB prediction."""
    project = tmp_path / "project"
    stage = project / "01-lamella" / "stage"
    (project / "experiment.yaml").parent.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: project")
    sem_image = stage / "sem" / "sem.tif"
    fib_image = stage / "fib" / "fib.tif"
    sem_prediction = stage / "predictions" / "clean" / sem_image.name
    fib_prediction = stage / "predictions_fib" / "clean" / fib_image.name
    for path in (sem_image, fib_image, sem_prediction, fib_prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    return stage, sem_image, fib_image


def _convert(tmp_path: Path, modality: str) -> int:
    return main([
        "convert-predictions", "--start-path", str(tmp_path), "--project", "project",
        "--lamella", "01", "--stage", "01", "--image", "all",
        "--modality", modality, "--kind", "clean",
    ])


def test_cli_converts_only_selected_modality(tmp_path: Path) -> None:
    stage, sem_image, fib_image = _build_project(tmp_path)

    assert _convert(tmp_path, "FIB") == 0

    assert not (stage / "sem_labels" / sem_image.name).exists()
    # A project with no label directory yet gets one in the current naming
    # convention: <image directory>_labels beside the images.
    assert (stage / "fib_labels" / fib_image.name).exists()


def test_cli_writes_into_an_existing_legacy_label_directory(tmp_path: Path) -> None:
    """Pre-convention directories are used as they are, not duplicated.

    Creating ``fib_labels/`` next to an existing ``labels_fib/`` would split one
    project's labels across two directories, so an existing legacy directory
    wins over the name a new project would be given.
    """
    stage, _, fib_image = _build_project(tmp_path)
    (stage / "labels_fib").mkdir()

    assert _convert(tmp_path, "FIB") == 0

    assert (stage / "labels_fib" / fib_image.name).exists()
    assert not (stage / "fib_labels").exists()
