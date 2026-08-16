"""Recompress existing on-disk label files to reduce storage size.

The proofreading software this package feeds into saves corrected labels
using the same uncompressed encoding as the source SEM/FIB image, so a
label file ends up taking as much disk space as the image it annotates.
This module re-encodes discrete label masks with :func:`save_label_image`,
which typically produces a much smaller file for the same pixel values --
pixel dimensions and label values are never altered.
"""
from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..layout import state_of_path
from ..models import ImageRecord
from .label_io import has_image_stack_support, save_label_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelCompressionResult:
    """Outcome of attempting to recompress a single label file.

    Attributes:
        label_path: Path to the label file that was checked.
        original_size: On-disk size in bytes before recompression.
        new_size: Size the recompressed encoding would take (or does take,
            if applied) in bytes.
        applied: Whether the file on disk was actually replaced. False for
            dry runs, and also False when recompression would not have made
            the file smaller.
    """

    label_path: Path
    original_size: int
    new_size: int
    applied: bool


def existing_labels(records: Iterable[ImageRecord]) -> list[Path]:
    """Return the deduplicated, existing label files referenced by *records*.

    Public so a caller can say how much work a compression run is before
    starting one, which measuring the saving would otherwise require doing the
    whole run twice.
    """
    seen: set[Path] = set()
    labels: list[Path] = []
    for record in records:
        for label in record.labels:
            if label.exists() and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def compress_labels(
    records: Iterable[ImageRecord],
    *,
    allow_corrected: bool = False,
    dry_run: bool = False,
) -> list[LabelCompressionResult]:
    """Recompress every on-disk label referenced by *records*.

    Validates the whole batch before changing anything: if any label lives
    under a ``corrected`` folder, the whole call is refused unless
    *allow_corrected* is True, since corrected labels are user-generated
    data that must never be modified without explicit confirmation.

    Each file is re-encoded into a temporary file in the same directory
    first, so a crash mid-write can never leave a truncated label; the
    temporary file only replaces the original once it has been written
    successfully. When *dry_run* is True, or when recompression would not
    shrink the file, the original is left untouched.

    Args:
        records: Records whose labels should be checked (as produced by the
            scanner / filtering module).
        allow_corrected: Set True to permit recompressing labels that live
            in a ``corrected`` folder.
        dry_run: If True, measure potential savings without modifying any
            files.

    Returns:
        One :class:`LabelCompressionResult` per existing label file.
    """
    if not has_image_stack_support():
        raise RuntimeError(
            "tifffile/numpy/cv2 are required to compress labels: install the 'all' extra"
        )
    import tifffile

    labels = existing_labels(records)
    if not allow_corrected:
        corrected = [label for label in labels if state_of_path(label) == "corrected"]
        if corrected:
            raise PermissionError(
                "Refusing to recompress corrected labels without explicit "
                f"confirmation: {corrected}"
            )

    results: list[LabelCompressionResult] = []
    for label_path in labels:
        original_size = label_path.stat().st_size
        array = tifffile.imread(str(label_path))

        fd, tmp_name = tempfile.mkstemp(dir=label_path.parent, suffix=label_path.suffix)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            save_label_image(array, tmp_path, array.dtype)
            new_size = tmp_path.stat().st_size
            applied = not dry_run and new_size < original_size
            if applied:
                os.replace(tmp_path, label_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        results.append(LabelCompressionResult(label_path, original_size, new_size, applied))
        if applied:
            logger.info("Compressed %s: %d -> %d bytes", label_path, original_size, new_size)

    return results
