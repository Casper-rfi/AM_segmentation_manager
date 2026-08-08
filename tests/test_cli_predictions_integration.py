"""Integration coverage for selector-based prediction conversion."""
from pathlib import Path

from am_segmentation_manager.cli import main


def test_cli_converts_only_selected_modality(tmp_path: Path) -> None:
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

    result = main([
        "convert-predictions", "--start-path", str(tmp_path), "--project", "project",
        "--lamella", "01", "--stage", "01", "--image", "all", "--modality", "FIB", "--kind", "clean",
    ])

    assert result == 0
    assert not (stage / "labels" / sem_image.name).exists()
    assert (stage / "labels_fib" / fib_image.name).exists()
