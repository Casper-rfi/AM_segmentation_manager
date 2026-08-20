from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.selection import mark_corrected_records, select_records
from am_segmentation_manager.state import read_state


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

    # Mark corrected: declares the group corrected and leaves the files alone.
    mark_corrected_records([rec])

    assert rec.corrected is True
    assert rec.selected is False
    assert img.exists()
    assert lbl.exists()
    assert list(read_state(stage).states.values()) == ["corrected"]

    # Corrected artifacts require an explicit confirmation before changing.
    with pytest.raises(PermissionError):
        select_records([rec])
    assert list(read_state(stage).states.values()) == ["corrected"]

    # Selecting with confirmation clears the corrected state.
    select_records([rec], allow_move_from_corrected=True)

    assert rec.corrected is False
    assert rec.selected is True
    assert img.exists()
    assert read_state(stage).states == {}
