import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.cli import main


def test_cli_deselect_select_integration(tmp_path: Path) -> None:
    """Integration test: create a minimal project/indexed stage and run the
    CLI deselect/select commands to ensure files are moved as expected.
    """
    project = tmp_path / "projX"
    project.mkdir()
    # experiment.yaml is used by Scanner to discover projects
    (project / "experiment.yaml").write_text("name: projX")

    lam = project / "L1"
    stage = lam / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"

    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    img = sem / "img2.tif"
    lbl = labels / "img2.tif"
    img.write_text("img")
    lbl.write_text("lbl")

    # deselect via CLI using index scoping
    rc = main(["deselect", "--start-path", str(tmp_path), "--project", project.name, "--lamella", lam.name, "--stage", stage.name, "--image", "all", "--modality", "SEM"])
    assert rc == 0
    assert (sem / "deselected" / "img2.tif").exists()
    assert (labels / "deselected" / "img2.tif").exists()

    # select via CLI
    rc = main(["select", "--start-path", str(tmp_path), "--project", project.name, "--lamella", lam.name, "--stage", stage.name, "--image", "all", "--modality", "SEM"])
    assert rc == 0
    assert (sem / "img2.tif").exists()
    assert (labels / "img2.tif").exists()


def test_cli_selection_accepts_numeric_scope_values(tmp_path: Path) -> None:
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    lam = project / "01-lamella"
    stage = lam / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    img0 = sem / "01-lamella_AP_img_000_SEM.tif"
    img1 = sem / "01-lamella_AP_img_001_SEM.tif"
    img2 = sem / "01-lamella_AP_img_002_SEM.tif"
    for path in (img0, img1, img2):
        path.write_text("img")
    (labels / img0.name).write_text("lbl")
    (labels / img1.name).write_text("lbl")
    (labels / img2.name).write_text("lbl")

    rc = main([
        "deselect",
        "--start-path",
        str(tmp_path),
        "--project",
        project.name,
        "--lamella",
        "01",
        "--stage",
        "01",
        "--image",
        "000-001",
        "--modality",
        "SEM",
    ])
    assert rc == 0
    assert (sem / "deselected" / img0.name).exists()
    assert (sem / "deselected" / img1.name).exists()
    assert not (sem / "deselected" / img2.name).exists()
    assert (labels / "deselected" / img0.name).exists()


def test_cli_selection_requires_explicit_scope_values(tmp_path: Path) -> None:
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    lam = project / "01-lamella"
    stage = lam / "stage1"
    sem = stage / "sem"
    sem.mkdir(parents=True)

    img0 = sem / "01-lamella_AP_img_000_SEM.tif"
    img1 = sem / "01-lamella_AP_img_001_SEM.tif"
    for path in (img0, img1):
        path.write_text("img")

    rc = main(["deselect", "--start-path", str(tmp_path), "--modality", "SEM"])
    assert rc == 2
    assert img0.exists()
    assert img1.exists()


def test_cli_selection_accepts_project_name_and_number(tmp_path: Path) -> None:
    project = tmp_path / "001-project-folder"
    stage = project / "01-lamella" / "stage"
    sem = stage / "sem"
    sem.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: Human project name")
    image = sem / "img1.tif"
    image.write_text("image")

    arguments = ["--start-path", str(tmp_path), "--lamella", "01", "--stage", "01", "--image", "all", "--modality", "SEM"]
    assert main(["deselect", "--project", "Human project name", *arguments]) == 0
    assert (sem / "deselected" / image.name).exists()
    assert main(["select", "--project", "001", *arguments]) == 0
    assert image.exists()


def test_cli_selection_changes_only_the_explicit_modality(tmp_path: Path) -> None:
    project = tmp_path / "projX"
    stage = project / "01-lamella" / "stage"
    sem_image = stage / "sem" / "sem.tif"
    fib_image = stage / "fib" / "fib.tif"
    (project / "experiment.yaml").parent.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: projX")
    sem_image.parent.mkdir(parents=True)
    fib_image.parent.mkdir(parents=True)
    sem_image.write_text("sem")
    fib_image.write_text("fib")

    result = main([
        "deselect", "--start-path", str(tmp_path), "--project", "projX",
        "--lamella", "01", "--stage", "01", "--image", "all", "--modality", "FIB",
    ])

    assert result == 0
    assert sem_image.exists()
    assert (stage / "fib" / "deselected" / fib_image.name).exists()


def test_cli_deselect_state_filter_skips_already_corrected(tmp_path: Path) -> None:
    """--state restricts the shared scope filter to a single workflow state.

    A corrected image should be left untouched when --state selected is
    given, since it no longer counts as "selected".
    """
    project = tmp_path / "projX"
    stage = project / "01-lamella" / "stage"
    (project / "experiment.yaml").parent.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: projX")

    selected_image = stage / "sem" / "selected.tif"
    corrected_image = stage / "sem" / "corrected" / "corrected.tif"
    corrected_label = stage / "labels" / "corrected" / "corrected.tif"
    selected_image.parent.mkdir(parents=True)
    corrected_image.parent.mkdir(parents=True)
    corrected_label.parent.mkdir(parents=True)
    selected_image.write_text("sem")
    corrected_image.write_text("sem")
    corrected_label.write_text("lbl")

    result = main([
        "deselect", "--start-path", str(tmp_path), "--project", "projX",
        "--lamella", "01", "--stage", "01", "--image", "all",
        "--modality", "SEM", "--state", "selected",
    ])

    assert result == 0
    # Only the selected image moved; the corrected image/label are untouched.
    assert (stage / "sem" / "deselected" / "selected.tif").exists()
    assert corrected_image.exists()
    assert corrected_label.exists()
