from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.selection import deselect_records, select_records


def test_deselect_and_select_records(tmp_path: Path) -> None:
    """Ensure deselect_records moves image and label files into 'deselected' and
    select_records moves them back and updates the ImageRecord.selected flag.
    """
    project = tmp_path / "project1"
    lamella = project / "LAM1"
    stage = lamella / "stageA"
    sem = stage / "sem"
    labels_dir = stage / "labels"

    sem.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    img = sem / "img1.tif"
    lbl = labels_dir / "img1.tif"

    img.write_text("image")
    lbl.write_text("label")

    rec = ImageRecord(
        image_path=img,
        labels=[lbl],
        project="project1",
        lamella="LAM1",
        stage="stageA",
        modality="SEM",
        selected=True,
    )

    # Deselect: moves files into deselected/ subfolders and marks record.selected False
    deselect_records([rec])

    assert rec.selected is False
    assert not img.exists()
    assert (sem / "deselected" / "img1.tif").exists()
    assert (labels_dir / "deselected" / "img1.tif").exists()

    # Select: should move files back and set selected True
    select_records([rec])

    assert rec.selected is True
    assert img.exists()
    assert (sem / "deselected" / "img1.tif").exists() is False
    assert (labels_dir / "deselected" / "img1.tif").exists() is False


def test_transition_validates_the_entire_batch_before_moving(tmp_path: Path) -> None:
    sem = tmp_path / "sem"
    safe_image = sem / "safe.tif"
    duplicate_image = sem / "duplicate.tif"
    safe_image.parent.mkdir()
    safe_image.write_text("safe")
    duplicate_image.write_text("normal copy")
    (sem / "deselected").mkdir()
    (sem / "deselected" / duplicate_image.name).write_text("deselected copy")

    safe_record = ImageRecord(image_path=safe_image)
    conflicting_record = ImageRecord(image_path=duplicate_image)

    with pytest.raises(FileExistsError, match="Multiple state copies"):
        deselect_records([safe_record, conflicting_record])

    # The safe record is untouched because validation completed before moves.
    assert safe_image.exists()
    assert not (sem / "deselected" / safe_image.name).exists()
