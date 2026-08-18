"""Tests for writing proofread labels back to disk.

The one operation in this feature that destroys data if it is wrong, so the
emphasis is on what it refuses: a file something else has rewritten, and a label
already marked corrected.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "tifffile", "cv2")
)

pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="Optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile

    from am_segmentation_manager.operations.label_save import (
        LabelToSave,
        StaleLabels,
        fingerprint_of,
        plan_saves,
        save_labels,
    )


def annotated(value: int = 2):
    """A small label holding an annotation."""
    label = np.zeros((16, 24), dtype=np.uint8)
    label[2:8, 3:12] = value
    return label


def existing(path: Path, value: int = 1) -> Path:
    """Write a label file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.full((16, 24), value, dtype=np.uint8))
    return path


def item(path: Path, label=None, **kwargs) -> LabelToSave:
    """A save entry whose fingerprint matches what is on disk now."""
    return LabelToSave(
        destination=path,
        label=annotated() if label is None else label,
        fingerprint=kwargs.pop("fingerprint", fingerprint_of(path)),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_an_edited_label_is_written(tmp_path: Path):
    destination = existing(tmp_path / "labels" / "img.tif")

    report = save_labels([item(destination)])

    np.testing.assert_array_equal(tifffile.imread(str(destination)), annotated())
    assert report.written == (destination,)


def test_a_new_label_is_created_with_its_directory(tmp_path: Path):
    destination = tmp_path / "labels" / "img.tif"

    report = save_labels([item(destination, created=True)])

    assert destination.exists()
    assert report.created == (destination,)
    assert "1 new" in report.headline()


def test_a_saved_label_is_compressed(tmp_path: Path):
    """What makes compress-labels unnecessary for anything saved here."""
    big = np.zeros((512, 512), dtype=np.uint8)
    big[100:400, 100:400] = 2
    destination = tmp_path / "labels" / "img.tif"

    save_labels([item(destination, label=big, created=True)])

    assert destination.stat().st_size < big.nbytes // 4


def test_the_dtype_of_the_edited_array_is_kept(tmp_path: Path):
    label = annotated().astype(np.uint16)
    destination = tmp_path / "labels" / "img.tif"

    save_labels([item(destination, label=label, created=True)])

    assert tifffile.imread(str(destination)).dtype == np.uint16


def test_saving_nothing_is_not_an_error(tmp_path: Path):
    report = save_labels([])

    assert report.count == 0
    assert report.headline() == "nothing to save"


# ---------------------------------------------------------------------------
# A file that changed underneath
# ---------------------------------------------------------------------------

def test_a_label_rewritten_since_it_was_opened_is_refused(tmp_path: Path):
    destination = existing(tmp_path / "labels" / "img.tif")
    entry = item(destination)
    existing(destination, value=7)  # something else wrote it

    with pytest.raises(StaleLabels, match="changed on disk"):
        save_labels([entry])

    assert (tifffile.imread(str(destination)) == 7).all(), "the file was overwritten"


def test_one_stale_label_stops_the_whole_batch(tmp_path: Path):
    good = existing(tmp_path / "labels" / "good.tif")
    stale = existing(tmp_path / "labels" / "stale.tif")
    entries = [item(good), item(stale)]
    existing(stale, value=7)

    with pytest.raises(StaleLabels):
        save_labels(entries)

    assert (tifffile.imread(str(good)) == 1).all(), "a batch must not write partway"


def test_a_label_that_appeared_since_it_was_opened_is_refused(tmp_path: Path):
    destination = tmp_path / "labels" / "img.tif"
    entry = item(destination, created=True)
    existing(destination, value=7)

    with pytest.raises(StaleLabels):
        save_labels([entry])


def test_the_plan_names_what_changed(tmp_path: Path):
    destination = existing(tmp_path / "labels" / "img.tif")
    entry = item(destination)
    existing(destination, value=7)

    plan = plan_saves([entry])

    assert plan.stale == (entry,)
    assert plan.count == 0
    assert "changed on disk" in plan.summary()


# ---------------------------------------------------------------------------
# Corrected labels
# ---------------------------------------------------------------------------

def test_a_corrected_label_is_not_replaced_without_consent(tmp_path: Path):
    destination = existing(tmp_path / "labels" / "corrected" / "img.tif")

    with pytest.raises(PermissionError, match="proofread"):
        save_labels([item(destination)])

    assert (tifffile.imread(str(destination)) == 1).all()


def test_a_corrected_label_is_replaced_once_confirmed(tmp_path: Path):
    destination = existing(tmp_path / "labels" / "corrected" / "img.tif")

    save_labels([item(destination)], allow_corrected=True)

    np.testing.assert_array_equal(tifffile.imread(str(destination)), annotated())


def test_one_corrected_label_holds_up_the_whole_batch(tmp_path: Path):
    plain = existing(tmp_path / "labels" / "img.tif")
    proofread = existing(tmp_path / "labels" / "corrected" / "img.tif")

    with pytest.raises(PermissionError):
        save_labels([item(plain), item(proofread)])

    assert (tifffile.imread(str(plain)) == 1).all(), "a batch must not write partway"


def test_the_plan_separates_the_corrected_ones(tmp_path: Path):
    plain = existing(tmp_path / "labels" / "img.tif")
    proofread = existing(tmp_path / "labels" / "corrected" / "img.tif")

    plan = plan_saves([item(plain), item(proofread)])

    assert len(plan.writes) == 1
    assert len(plan.corrected) == 1
    assert plan.count == 2
    assert "1 already corrected" in plan.summary()


# ---------------------------------------------------------------------------
# Whole or absent
# ---------------------------------------------------------------------------

def test_a_failed_write_leaves_the_previous_label_intact(tmp_path: Path, monkeypatch):
    destination = existing(tmp_path / "labels" / "img.tif")

    def explode(mask, path, dtype):
        raise OSError("disk full")

    monkeypatch.setattr(
        "am_segmentation_manager.operations.label_save.save_label_image", explode
    )

    with pytest.raises(OSError, match="disk full"):
        save_labels([item(destination)])

    assert (tifffile.imread(str(destination)) == 1).all()
    assert list(destination.parent.iterdir()) == [destination], "a temporary was left"


def test_a_fingerprint_of_a_missing_file_is_nothing(tmp_path: Path):
    assert fingerprint_of(tmp_path / "absent.tif") is None
