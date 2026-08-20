from pathlib import Path

from am_segmentation_manager.cli import main
from am_segmentation_manager.state import read_state


def test_cli_correct_and_select_integration(tmp_path: Path) -> None:
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

    # correct via CLI using index scoping
    rc = main(["correct", "--start-path", str(tmp_path), "--project", project.name, "--lamella", lam.name, "--stage", stage.name, "--image", "all", "--modality", "SEM"])
    assert rc == 0
    assert read_state(stage).states == {"labels/img2.tif": "corrected"}
    assert img.exists() and lbl.exists()

    # Select restores a corrected image to the normal state.
    rc = main(["select", "--start-path", str(tmp_path), "--project", project.name, "--lamella", lam.name, "--stage", stage.name, "--image", "all", "--modality", "SEM", "--confirm-corrected"])
    assert rc == 0
    assert read_state(stage).states == {}
    assert img.exists() and lbl.exists()
