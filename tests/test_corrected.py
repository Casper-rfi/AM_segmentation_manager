from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.selection import mark_corrected_records, select_records


def test_mark_corrected_records_and_restore_by_selecting(tmp_path: Path) -> None:
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

    # Mark corrected: moves files into corrected/ subfolders and marks record.corrected True
    mark_corrected_records([rec])

    assert rec.corrected is True
    assert rec.selected is False
    assert not img.exists()
    assert (sem / "corrected" / "img1.tif").exists()
    assert (labels_dir / "corrected" / "img1.tif").exists()

    # Corrected artifacts require an explicit confirmation before moving.
    with pytest.raises(PermissionError):
        select_records([rec])
    assert (sem / "corrected" / "img1.tif").exists()
    assert (labels_dir / "corrected" / "img1.tif").exists()

    # Selecting with confirmation restores files and clears corrected state.
    select_records([rec], allow_move_from_corrected=True)

    assert rec.corrected is False
    assert rec.selected is True
    assert img.exists()
    assert (sem / "corrected" / "img1.tif").exists() is False
    assert (labels_dir / "corrected" / "img1.tif").exists() is False
