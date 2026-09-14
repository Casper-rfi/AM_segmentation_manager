"""Tests for what each label holds, as the window's views share it.

Opening a label costs milliseconds and every view asks about every frame, so an
answer is remembered -- which makes forgetting the part that matters: a label
written by one view must be read again by all of them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import am_segmentation_manager.gui.known_labels as known_labels_module
from am_segmentation_manager.gui.known_labels import KnownLabels
from am_segmentation_manager.models import ImageRecord

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")


def image_with_label(root: Path, name: str, label: Path) -> ImageRecord:
    """Return a record whose label is *label*, whether or not it exists yet."""
    image = root / "sem" / name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    return ImageRecord(image_path=image, labels=[label], modality="SEM")


def write_label(path: Path, value: int) -> Path:
    """Write a small label holding *value* in one corner, or nothing at zero."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8), dtype=np.uint8)
    array[:2, :2] = value
    tifffile.imwrite(str(path), array)
    return path


@pytest.fixture
def reads(monkeypatch) -> list[Path]:
    """Every label opened, in order."""
    opened: list[Path] = []
    original = known_labels_module.is_label_file_empty

    def counted(path: Path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(known_labels_module, "is_label_file_empty", counted)
    return opened


def test_a_label_shared_by_two_images_is_opened_once(tmp_path: Path, reads):
    label = write_label(tmp_path / "sem_labels" / "a.tif", 2)
    first = image_with_label(tmp_path, "a.tif", label)
    second = image_with_label(tmp_path, "a_5kv.tif", label)
    states = KnownLabels()

    assert states.state(first) == "annotated"
    assert states.state(second) == "annotated"
    assert states.state(first) == "annotated"
    assert reads == [label]


def test_a_missing_label_is_reported_without_opening_anything(tmp_path: Path, reads):
    record = image_with_label(tmp_path, "a.tif", tmp_path / "sem_labels" / "a.tif")

    assert KnownLabels().state(record) == "missing"
    assert reads == []


def test_forgetting_a_label_reads_it_again_for_every_image_sharing_it(tmp_path: Path):
    label = write_label(tmp_path / "sem_labels" / "a.tif", 0)
    first = image_with_label(tmp_path, "a.tif", label)
    second = image_with_label(tmp_path, "a_5kv.tif", label)
    states = KnownLabels()
    assert states.state(first) == states.state(second) == "empty"

    write_label(label, 3)
    assert states.forget(labels=[label]) == 1

    assert states.state(first) == states.state(second) == "annotated"


def test_a_record_that_gained_a_label_is_read_afresh(tmp_path: Path):
    label = tmp_path / "sem_labels" / "a.tif"
    record = image_with_label(tmp_path, "a.tif", label)
    states = KnownLabels()
    assert states.state(record) == "missing"

    write_label(label, 2)
    assert states.state(record) == "missing", "remembered until it is forgotten"
    states.forget(records=[record])

    assert states.state(record) == "annotated"


def test_what_a_pass_already_read_is_taken_without_opening_the_label(tmp_path: Path, reads):
    label = write_label(tmp_path / "sem_labels" / "a.tif", 2)
    record = image_with_label(tmp_path, "a.tif", label)
    states = KnownLabels()

    states.remember(record.image_path, label, False)

    assert states.knows(label)
    assert states.state(record) == "annotated"
    assert reads == []


def test_clearing_forgets_everything(tmp_path: Path, reads):
    label = write_label(tmp_path / "sem_labels" / "a.tif", 2)
    record = image_with_label(tmp_path, "a.tif", label)
    states = KnownLabels()
    states.state(record)

    states.clear()

    assert not states.knows(label)
    states.state(record)
    assert reads == [label, label]
