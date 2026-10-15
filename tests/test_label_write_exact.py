"""A label is written with exactly the pixels asked for, or not at all.

:func:`save_label_image` is what every label write ends in -- a save, a
conversion, a prediction, a recompression -- and it writes into a temporary
that is then renamed over the label. Anything short of the exact pixels must
therefore raise, or the rename puts it over proofread work: OpenCV reports a
failed write by returning False, and a cast to 8 bits wraps 300 round to 44.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

import numpy as np
import tifffile

from am_segmentation_manager.io.atomic import fingerprint_of, write_atomically
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.label_compression import compress_labels
from am_segmentation_manager.operations.label_io import save_label_image
from am_segmentation_manager.operations.label_save import LabelToSave, SaveStopped, save_labels
from am_segmentation_manager.operations.pseudo_labels import prediction_to_label


def _existing(path: Path, value: int = 3) -> Path:
    """Write an uncompressed label holding *value* everywhere, and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.full((64, 64), value, dtype=np.uint8))
    return path


@pytest.fixture
def imwrite_fails(monkeypatch):
    """OpenCV declining to write, as it does on Windows for a non-ASCII path."""
    monkeypatch.setattr("cv2.imwrite", lambda *args, **kwargs: False)


# ---------------------------------------------------------------------------
# The writer itself
# ---------------------------------------------------------------------------

def test_a_label_round_trips_exactly(tmp_path: Path):
    mask = np.zeros((32, 48), dtype=np.uint16)
    mask[4:9, 5:20] = 1000
    save_label_image(mask, tmp_path / "label.tif", mask.dtype)

    written = tifffile.imread(str(tmp_path / "label.tif"))
    assert written.dtype == np.uint16
    assert np.array_equal(written, mask)


def test_small_class_indices_of_a_wide_type_are_written_as_8_bit(tmp_path: Path):
    """What a model returns: int64 indices that fit in a byte."""
    mask = np.zeros((16, 16), dtype=np.int64)
    mask[2:5, 2:5] = 4
    save_label_image(mask, tmp_path / "label.tif", mask.dtype)

    written = tifffile.imread(str(tmp_path / "label.tif"))
    assert written.dtype == np.uint8
    assert np.array_equal(written, mask)


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(np.full((16, 16), 300, dtype=np.int32), id="wraps-past-255"),
        pytest.param(np.full((16, 16), -1, dtype=np.int16), id="negative"),
        pytest.param(np.full((16, 16), 2.5, dtype=np.float32), id="fractional"),
        pytest.param(np.full((16, 16), np.nan, dtype=np.float32), id="nan"),
        pytest.param(np.zeros((16, 16, 3), dtype=np.uint8), id="rgb"),
        pytest.param(np.zeros((3, 16, 16), dtype=np.uint8), id="stack"),
    ],
)
def test_a_mask_that_cannot_be_stored_exactly_is_refused(tmp_path: Path, mask):
    with pytest.raises(ValueError):
        save_label_image(mask, tmp_path / "label.tif", mask.dtype)
    assert not (tmp_path / "label.tif").exists()


def test_a_write_opencv_declines_raises(tmp_path: Path, imwrite_fails):
    with pytest.raises(OSError, match="could not write"):
        save_label_image(np.zeros((8, 8), np.uint8), tmp_path / "label.tif", np.uint8)


def test_a_file_that_does_not_read_back_as_written_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tifffile.imread", lambda *args, **kwargs: np.ones((8, 8), np.uint8))
    with pytest.raises(OSError, match="did not read back"):
        save_label_image(np.zeros((8, 8), np.uint8), tmp_path / "label.tif", np.uint8)


def test_a_failed_write_through_a_temporary_leaves_the_label_as_it_was(
    tmp_path: Path, imwrite_fails
):
    label = _existing(tmp_path / "labels" / "img.tif")
    before = label.read_bytes()

    with pytest.raises(OSError):
        write_atomically(
            label, lambda temporary: save_label_image(np.ones((64, 64), np.uint8), temporary, np.uint8)
        )

    assert label.read_bytes() == before
    assert list(label.parent.iterdir()) == [label], "a temporary was left"


# ---------------------------------------------------------------------------
# The writers built on it
# ---------------------------------------------------------------------------

def test_a_save_opencv_declines_leaves_the_label_as_it_was(tmp_path: Path, imwrite_fails):
    label = _existing(tmp_path / "labels" / "img.tif")
    before = label.read_bytes()

    with pytest.raises(SaveStopped, match="could not write"):
        save_labels([
            LabelToSave(
                destination=label,
                label=np.ones((64, 64), np.uint8),
                fingerprint=fingerprint_of(label),
            )
        ])

    assert label.read_bytes() == before


def test_a_conversion_opencv_declines_leaves_the_label_as_it_was(
    tmp_path: Path, imwrite_fails
):
    label = _existing(tmp_path / "labels" / "img.tif")
    prediction = _existing(tmp_path / "predictions" / "img.tif", value=1)
    image = _existing(tmp_path / "sem" / "img.tif", value=0)
    before = label.read_bytes()

    with pytest.raises(OSError):
        prediction_to_label(prediction, label, target_image_path=image)

    assert label.read_bytes() == before


# ---------------------------------------------------------------------------
# Recompression
# ---------------------------------------------------------------------------

def _record(label: Path) -> ImageRecord:
    return ImageRecord(image_path=label.parent.parent / "sem" / label.name, labels=[label])


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.full((64, 64), 300, dtype=np.int32), id="wraps-past-255"),
        pytest.param(np.full((64, 64), 2.5, dtype=np.float32), id="fractional"),
        pytest.param(np.full((64, 64, 3), (10, 20, 30), dtype=np.uint8), id="rgb"),
        pytest.param(np.zeros((3, 64, 64), dtype=np.uint8), id="stack"),
    ],
)
def test_recompression_leaves_alone_a_label_it_cannot_reencode_exactly(
    tmp_path: Path, array
):
    label = tmp_path / "labels" / "img.tif"
    label.parent.mkdir()
    tifffile.imwrite(str(label), array, photometric="rgb" if array.shape[-1] == 3 else "minisblack")
    before = label.read_bytes()

    [result] = compress_labels([_record(label)])

    assert not result.applied
    assert result.problem
    assert label.read_bytes() == before


def test_recompression_leaves_alone_a_label_it_cannot_read(tmp_path: Path):
    label = tmp_path / "labels" / "img.tif"
    label.parent.mkdir()
    label.write_bytes(b"not a tiff")

    [result] = compress_labels([_record(label)])

    assert not result.applied
    assert "could not be read" in result.problem
    assert label.read_bytes() == b"not a tiff"


def test_recompression_goes_on_past_a_label_it_leaves_alone(tmp_path: Path):
    unsafe = tmp_path / "labels" / "a.tif"
    unsafe.parent.mkdir()
    tifffile.imwrite(str(unsafe), np.full((256, 256), 300, dtype=np.int32))
    fine = tmp_path / "labels" / "b.tif"
    sparse = np.zeros((256, 256), dtype=np.uint8)
    sparse[50:150, 50:150] = 2
    tifffile.imwrite(str(fine), sparse)

    results = compress_labels([_record(unsafe), _record(fine)])

    assert [bool(result.problem) for result in results] == [True, False]
    assert results[1].applied
    assert np.array_equal(tifffile.imread(str(fine)), sparse)


def test_recompression_opencv_declines_leaves_the_label_as_it_was(
    tmp_path: Path, imwrite_fails
):
    """A failed write leaves an empty temporary, which is smaller than any label."""
    label = _existing(tmp_path / "labels" / "img.tif")
    before = label.read_bytes()

    [result] = compress_labels([_record(label)])

    assert not result.applied
    assert "could not write" in result.problem
    assert label.read_bytes() == before


# ---------------------------------------------------------------------------
# A conversion stopped partway
# ---------------------------------------------------------------------------

def _convertible(root: Path, name: str) -> ImageRecord:
    image = _existing(root / "sem" / name, value=0)
    prediction = _existing(root / "sem_predictions" / name, value=1)
    return ImageRecord(
        image_path=image, predictions=[prediction], modality="SEM",
        labels_directory=root / "sem_labels",
    )


def test_a_conversion_stopped_partway_says_what_it_wrote(tmp_path: Path, monkeypatch):
    from am_segmentation_manager.operations import label_io
    from am_segmentation_manager.operations.pseudo_labels import (
        ConversionStopped,
        convert_record_predictions,
    )

    records = [_convertible(tmp_path, "a.tif"), _convertible(tmp_path, "b.tif")]
    real, calls = label_io.save_label_image, []

    def second_write_fails(mask, path, dtype):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        real(mask, path, dtype)

    monkeypatch.setattr(
        "am_segmentation_manager.operations.pseudo_labels.save_label_image", second_write_fails
    )
    with pytest.raises(ConversionStopped, match="disk full") as stopped:
        convert_record_predictions(records)

    assert stopped.value.written == (tmp_path / "sem_labels" / "a.tif",)
    assert (tmp_path / "sem_labels" / "a.tif").exists()
    assert not (tmp_path / "sem_labels" / "b.tif").exists()


def test_the_command_line_reports_a_conversion_stopped_partway(tmp_path: Path, monkeypatch, capsys):
    from am_segmentation_manager.cli import main
    from am_segmentation_manager.operations.pseudo_labels import ConversionStopped
    from conftest import build_dataset

    build_dataset(tmp_path, predictions=True, labels="none")

    def stopped(*args, **kwargs):
        raise ConversionStopped("Converted 1 of 2 label(s), then stopped: disk full", [tmp_path])

    monkeypatch.setattr(
        "am_segmentation_manager.cli.curating.convert_record_predictions", stopped
    )
    code = main([
        "convert-predictions", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--modality", "SEM",
    ])

    assert code == 1
    output = capsys.readouterr().out
    assert "disk full" in output
    assert "re-run to convert the rest" in output
