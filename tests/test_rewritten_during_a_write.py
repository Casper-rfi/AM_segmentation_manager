"""A file rewritten while a write to it is under way is not written over.

Each writer checks its destination when the batch is planned and again just
before its rename, the second being what these exercise: something else
rewrites the file after the first check, while the new version is being
encoded, and the write must leave that other version in place.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

import numpy as np
import tifffile

from am_segmentation_manager import state as state_module
from am_segmentation_manager.io.atomic import fingerprint_of
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations import label_save, predictions
from am_segmentation_manager.operations.label_save import (
    LabelToSave,
    SaveStopped,
    StaleLabels,
    save_labels,
)
from am_segmentation_manager.operations.predictions import (
    ConversionStopped,
    convert_record_predictions,
)
from am_segmentation_manager.state import StaleState, read_state, state_file_path, write_state

#: Another program's version, at another size: the same size, rewritten in place
#: within a clock step, is the one change a fingerprint cannot see.
ELSEWHERE = np.full((32, 32), 7, dtype=np.uint8)


def _tiff(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.full((64, 64), value, dtype=np.uint8))
    return path


def _rewritten_while_encoding(monkeypatch, module, destination: Path) -> None:
    """Have another program write *destination* while *module* encodes a label."""
    real = module.save_label_image

    def encode(mask, path, dtype):
        tifffile.imwrite(str(destination), ELSEWHERE)
        real(mask, path, dtype)

    monkeypatch.setattr(module, "save_label_image", encode)


def test_a_save_leaves_a_label_rewritten_while_it_ran(tmp_path: Path, monkeypatch):
    label = _tiff(tmp_path / "sem_labels" / "img.tif", 1)
    _rewritten_while_encoding(monkeypatch, label_save, label)

    with pytest.raises(SaveStopped) as stopped:
        save_labels([
            LabelToSave(
                destination=label,
                label=np.full((64, 64), 2, dtype=np.uint8),
                fingerprint=fingerprint_of(label),
            )
        ])

    assert isinstance(stopped.value.__cause__, StaleLabels)
    assert stopped.value.written == ()
    assert (tifffile.imread(str(label)) == ELSEWHERE).all()


def test_a_conversion_leaves_a_label_that_appeared_while_it_ran(tmp_path: Path, monkeypatch):
    image = _tiff(tmp_path / "sem" / "img.tif", 0)
    prediction = _tiff(tmp_path / "sem_predictions" / "img.tif", 1)
    label = tmp_path / "sem_labels" / "img.tif"
    record = ImageRecord(
        image_path=image, predictions=[prediction], modality="SEM",
        labels_directory=label.parent,
    )
    label.parent.mkdir()
    _rewritten_while_encoding(monkeypatch, predictions, label)

    with pytest.raises(ConversionStopped) as stopped:
        convert_record_predictions([record])

    assert isinstance(stopped.value.__cause__, StaleLabels)
    assert (tifffile.imread(str(label)) == ELSEWHERE).all()


def test_a_state_file_rewritten_while_it_was_written_is_left(tmp_path: Path, monkeypatch):
    previous = read_state(tmp_path)
    real = state_module.write_yaml

    def written_elsewhere_first(path, *args, **kwargs):
        state_file_path(tmp_path).write_text("state:\n  sem_labels/b.tif: corrected\n")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(state_module, "write_yaml", written_elsewhere_first)

    with pytest.raises(StaleState):
        write_state(previous, {"sem_labels/a.tif": "deselected"})

    assert read_state(tmp_path).states == {"sem_labels/b.tif": "corrected"}
