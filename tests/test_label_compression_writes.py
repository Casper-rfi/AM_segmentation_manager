"""How a compression run writes: permissions, staleness and the dry run.

Separate from ``test_label_compression.py``, which checks that the encoding
shrinks a label without altering a pixel. These check what happens around the
encoding, so the encoder is stubbed -- which is the only way to provoke the
case that matters most: something else rewriting the label while it is being
re-encoded. A real encoder gives no window to do that in.

Stubbing also keeps the permission checks honest about what they are testing.
The mode is lost or kept by the rename, not by the encoder, and cv2 truncating
rather than recreating is what exposes it.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("numpy")

import numpy as np
import tifffile

from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations import label_compression
from am_segmentation_manager.operations.label_compression import compress_labels


class StubEncoder:
    """Stands in for LZW re-encoding, writing a deliberately tiny file.

    ``payload`` is what the encoding produces -- small by default, so a run
    commits. ``during`` is called with the destination while the encoding is
    in progress, which is how a test makes something else rewrite the label
    underneath the run.
    """

    def __init__(self) -> None:
        self.payload = b"small"
        self.during: list = []

    def __call__(self, mask, output_path: Path, original_dtype) -> None:
        for hook in self.during:
            hook()
        output_path.write_bytes(self.payload)


@pytest.fixture
def stub_encoder(monkeypatch):
    encoder = StubEncoder()
    monkeypatch.setattr(label_compression, "save_label_image", encoder)
    monkeypatch.setattr(label_compression, "has_image_stack_support", lambda: True)
    return encoder


def _label(tmp_path: Path, name: str = "img1.tif") -> Path:
    """Write an uncompressed label big enough that 'small' beats it."""
    directory = tmp_path / "sem_labels"
    directory.mkdir(exist_ok=True)
    path = directory / name
    tifffile.imwrite(str(path), np.zeros((64, 64), dtype=np.uint8))
    return path


def _record(tmp_path: Path, label: Path) -> ImageRecord:
    return ImageRecord(image_path=tmp_path / "sem" / label.name, labels=[label])


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_a_compressed_label_keeps_the_permissions_it_had(tmp_path: Path, stub_encoder):
    """A label in a shared archive that only its author can read is a label
    nobody else can use, so recompressing must not narrow it."""
    label = _label(tmp_path)
    os.chmod(label, 0o644)

    results = compress_labels([_record(tmp_path, label)])

    assert results[0].applied is True
    assert stat.S_IMODE(label.stat().st_mode) == 0o644


def test_a_compressed_label_is_never_left_owner_only(tmp_path: Path, stub_encoder):
    label = _label(tmp_path)
    os.chmod(label, 0o664)

    compress_labels([_record(tmp_path, label)])

    assert stat.S_IMODE(label.stat().st_mode) & stat.S_IRGRP


# ---------------------------------------------------------------------------
# Changed underneath the run
# ---------------------------------------------------------------------------

def test_a_label_rewritten_mid_run_is_left_alone(tmp_path: Path, stub_encoder):
    """Recompressing pixels that have since been replaced would put the old
    ones back over somebody else's write."""
    label = _label(tmp_path)
    replacement = b"somebody else's proofreading, written while we encoded"
    stub_encoder.during.append(lambda: label.write_bytes(replacement))

    results = compress_labels([_record(tmp_path, label)])

    assert results[0].applied is False
    assert results[0].stale is True
    assert label.read_bytes() == replacement


def test_a_stale_label_does_not_stop_the_rest_of_the_run(tmp_path: Path, stub_encoder):
    """Compression is optional maintenance over a whole collection, so one file
    moving under it skips that file rather than abandoning the run."""
    changing = _label(tmp_path, "changes.tif")
    steady = _label(tmp_path, "steady.tif")
    stub_encoder.during.append(lambda: changing.write_bytes(b"x" * 999))

    results = compress_labels(
        [_record(tmp_path, changing), _record(tmp_path, steady)]
    )

    by_name = {result.label_path.name: result for result in results}
    assert by_name["changes.tif"].stale is True
    assert by_name["changes.tif"].applied is False
    assert by_name["steady.tif"].applied is True


def test_an_untouched_label_is_not_reported_stale(tmp_path: Path, stub_encoder):
    label = _label(tmp_path)

    assert compress_labels([_record(tmp_path, label)])[0].stale is False


# ---------------------------------------------------------------------------
# When there is nothing to gain
# ---------------------------------------------------------------------------

def test_a_label_that_would_grow_is_left_alone(tmp_path: Path, stub_encoder):
    """Re-encoding is only worth committing if it actually saves something."""
    label = _label(tmp_path)
    before = label.read_bytes()
    stub_encoder.payload = b"y" * 100_000

    results = compress_labels([_record(tmp_path, label)])

    assert results[0].applied is False
    assert results[0].stale is False
    assert label.read_bytes() == before


def test_a_dry_run_measures_without_writing(tmp_path: Path, stub_encoder):
    """The saving cannot be known without encoding, so a dry run encodes and
    then declines to rename."""
    label = _label(tmp_path)
    before = label.read_bytes()

    results = compress_labels([_record(tmp_path, label)], dry_run=True)

    assert results[0].applied is False
    assert results[0].new_size < results[0].original_size
    assert label.read_bytes() == before


def test_a_dry_run_leaves_no_temporary_behind(tmp_path: Path, stub_encoder):
    label = _label(tmp_path)

    compress_labels([_record(tmp_path, label)], dry_run=True)

    assert [path.name for path in label.parent.iterdir()] == [label.name]


def test_a_dry_run_does_not_report_staleness(tmp_path: Path, stub_encoder):
    """Nothing was going to be written, so nothing was nearly overwritten."""
    label = _label(tmp_path)
    stub_encoder.during.append(lambda: label.write_bytes(b"changed"))

    assert compress_labels([_record(tmp_path, label)], dry_run=True)[0].stale is False


# ---------------------------------------------------------------------------
# The corrected guard still fires
# ---------------------------------------------------------------------------

def test_a_corrected_label_is_refused_before_anything_is_written(
    tmp_path: Path, stub_encoder
):
    label = _label(tmp_path)
    before = label.read_bytes()
    record = _record(tmp_path, label)
    record.state = "corrected"

    with pytest.raises(PermissionError):
        compress_labels([record])

    assert label.read_bytes() == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_a_confirmed_corrected_label_keeps_its_permissions(tmp_path: Path, stub_encoder):
    """The confirmed path is the one a proofread label actually travels."""
    label = _label(tmp_path)
    os.chmod(label, 0o644)
    record = _record(tmp_path, label)
    record.state = "corrected"

    results = compress_labels([record], allow_corrected=True)

    assert results[0].applied is True
    assert stat.S_IMODE(label.stat().st_mode) == 0o644
