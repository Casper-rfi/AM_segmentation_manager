"""Integration coverage for selector-based prediction conversion."""
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from conftest import write_prediction


def _build_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal project with one SEM and one FIB prediction."""
    project = tmp_path / "project"
    stage = project / "01-lamella" / "stage"
    (project / "experiment.yaml").parent.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: project")
    sem_image = stage / "sem" / "sem.tif"
    fib_image = stage / "fib" / "fib.tif"
    sem_prediction = stage / "predictions" / "raw" / sem_image.name
    fib_prediction = stage / "predictions_fib" / "raw" / fib_image.name
    for path in (sem_image, fib_image):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    for path in (sem_prediction, fib_prediction):
        write_prediction(path, 1)
    return stage, sem_image, fib_image


def _convert(tmp_path: Path, modality: str, *extra: str) -> int:
    return main([
        "convert-predictions", "--start-path", str(tmp_path), "--project", "project",
        "--lamella", "01", "--stage", "01", "--image", "all",
        "--modality", modality, *extra,
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


def test_cli_keeps_an_existing_label_without_the_overwrite_flag(tmp_path: Path) -> None:
    """Overwriting is opt-in on the command line as it is in the window."""
    stage, _, fib_image = _build_project(tmp_path)
    label = stage / "fib_labels" / fib_image.name
    label.parent.mkdir(parents=True)
    label.write_bytes(b"existing label")

    _convert(tmp_path, "FIB")

    assert label.read_bytes() == b"existing label"

    _convert(tmp_path, "FIB", "--overwrite")

    assert label.read_bytes() != b"existing label"


def test_cli_refuses_a_proofread_label_even_with_overwrite(tmp_path: Path) -> None:
    """`--overwrite` reaches pseudo-labels, never proofread ones."""
    from am_segmentation_manager.state import read_state, write_state

    stage, _, fib_image = _build_project(tmp_path)
    label = stage / "fib_labels" / fib_image.name
    label.parent.mkdir(parents=True)
    label.write_bytes(b"proofread")
    write_state(read_state(stage), {f"fib_labels/{fib_image.name}": "corrected"})

    _convert(tmp_path, "FIB", "--overwrite")

    assert label.read_bytes() == b"proofread"


def test_cli_never_converts_post_processed_predictions(tmp_path: Path) -> None:
    """A prediction under ``clean/`` is not a label source, however it is reached.

    Post-processing rewrites the intensity values, so the result no longer
    corresponds to the segmentation classes. An image whose only prediction is
    post-processed has nothing convertible, which the command reports as finding
    nothing rather than by writing an unusable label.
    """
    stage, _, fib_image = _build_project(tmp_path)
    (stage / "predictions_fib" / "raw" / fib_image.name).unlink()
    clean = stage / "predictions_fib" / "clean" / fib_image.name
    clean.parent.mkdir(parents=True, exist_ok=True)
    clean.write_bytes(b"image")

    _convert(tmp_path, "FIB")

    assert not (stage / "fib_labels" / fib_image.name).exists()


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Every command that moves files offers a dry run, this one included."""
    stage, _, fib_image = _build_project(tmp_path)

    assert _convert(tmp_path, "FIB", "--dry-run") == 0

    assert not (stage / "fib_labels" / fib_image.name).exists()

    assert _convert(tmp_path, "FIB") == 0

    assert (stage / "fib_labels" / fib_image.name).exists()


def test_cli_converts_the_prediction_source_asked_for(tmp_path: Path) -> None:
    """am-seg's prediction by default, the microscope's with --prefer microscope."""
    stage, _, fib_image = _build_project(tmp_path)
    am_seg = write_prediction(stage / "predictions_fib" / "am_seg" / fib_image.name, 2)
    microscope = stage / "predictions_fib" / "raw" / fib_image.name
    label = stage / "fib_labels" / fib_image.name

    assert _convert(tmp_path, "FIB", "--prefer", "microscope") == 0
    assert label.read_bytes() == microscope.read_bytes()

    assert _convert(tmp_path, "FIB", "--overwrite") == 0
    assert label.read_bytes() == am_seg.read_bytes()


def _annotated_fib_label(tmp_path: Path) -> Path:
    """A project with no history yet, whose FIB image has a label holding annotation."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    stage, _, fib_image = _build_project(tmp_path)
    blank = np.zeros((8, 8), dtype=np.uint8)
    for path in (fib_image, stage / "predictions_fib" / "raw" / fib_image.name):
        tifffile.imwrite(path, blank)
    label = stage / "fib_labels" / fib_image.name
    label.parent.mkdir(parents=True)
    annotated = blank.copy()
    annotated[2:4, 2:4] = 2
    tifffile.imwrite(label, annotated)
    return label


def test_cli_keeps_a_replaced_label_where_the_project_had_no_history(tmp_path: Path) -> None:
    """A conversion brings its projects into the record before writing, as the
    window does, so what it replaces reaches the archive."""
    label = _annotated_fib_label(tmp_path)
    before = label.read_bytes()

    assert _convert(tmp_path, "FIB", "--overwrite") == 0

    archived = list((tmp_path / "project" / "AM_archive").rglob("*.tif"))
    assert len(archived) == 1 and archived[0].read_bytes() == before


def test_cli_says_what_a_replaced_label_comes_to(tmp_path: Path, capsys) -> None:
    _annotated_fib_label(tmp_path)

    _convert(tmp_path, "FIB")

    assert (
        "1 label(s) already there were left as they are (1 holding annotation not "
        "marked corrected, kept in AM_archive/); pass --overwrite to replace them"
    ) in capsys.readouterr().out
