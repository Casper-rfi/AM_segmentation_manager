from pathlib import Path

import pytest

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.selection import deselect_records, select_records
from am_segmentation_manager.state import read_state


def test_deselect_and_select_records(tmp_path: Path) -> None:
    """Deselecting declares the group deselected and selecting clears it, in
    both cases updating the record and leaving every file where it is.
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

    # Deselect: declares the group deselected and marks record.selected False
    deselect_records([rec])

    assert rec.selected is False
    assert img.exists()
    assert lbl.exists()
    assert list(read_state(stage).states.values()) == ["deselected"]

    # Select: clears the declaration and sets selected True
    select_records([rec])

    assert rec.selected is True
    assert img.exists()
    assert read_state(stage).states == {}


def test_transition_validates_the_entire_batch_before_writing(tmp_path: Path) -> None:
    """One record that cannot be resolved stops the whole batch."""
    stage = tmp_path / "stageA"
    sem = stage / "sem"
    safe_image = sem / "safe.tif"
    sem.mkdir(parents=True)
    safe_image.write_text("safe")

    safe_record = ImageRecord(image_path=safe_image)
    missing_record = ImageRecord(image_path=sem / "absent.tif")

    with pytest.raises(FileNotFoundError):
        deselect_records([safe_record, missing_record])

    # The safe record is untouched because validation completed before writing.
    assert safe_record.selected is True
    assert read_state(stage).states == {}
