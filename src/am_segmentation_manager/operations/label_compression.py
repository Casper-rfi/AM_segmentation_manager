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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io.atomic import fingerprint_of, write_atomically
from ..models import ImageRecord
from .label_io import has_image_stack_support, save_label_image
from .selection import current_state

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
            the file smaller, or when the file changed underneath the run.
        stale: Whether something else rewrote the label while it was being
            recompressed, so it was left alone. Distinguished from the other
            reasons for not applying: those mean there was nothing to gain,
            this one means somebody else's write was nearly overwritten.
    """

    label_path: Path
    original_size: int
    new_size: int
    applied: bool
    stale: bool = False


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

    Validates the whole batch before changing anything: if any label belongs to
    a group declared corrected, the whole call is refused unless
    *allow_corrected* is True, since corrected labels are user-generated
    data that must never be modified without explicit confirmation.

    Each file is written through :func:`~..io.atomic.write_atomically`, so a
    crash mid-write can never leave a truncated label, and the label keeps the
    permissions it had rather than the owner-only ones a temporary file is
    created with. The original is left untouched when *dry_run* is True, when
    recompression would not shrink the file, and when the label changed on disk
    while it was being re-encoded -- recompressing pixels that have since been
    replaced would put the old ones back.

    Args:
        records: Records whose labels should be checked (as produced by the
            scanner / filtering module).
        allow_corrected: Set True to permit recompressing labels whose group
            is declared corrected.
        dry_run: If True, measure potential savings without modifying any
            files.

    Returns:
        One :class:`LabelCompressionResult` per existing label file.
    """
    if not has_image_stack_support():
        raise RuntimeError(
            "tifffile/numpy/cv2 are required to compress labels: install the 'all' extra"
        )

    records = list(records)
    labels = existing_labels(records)
    if not allow_corrected:
        declared_corrected = {
            label
            for record in records
            if current_state(record) == "corrected"
            for label in record.labels
        }
        corrected = [label for label in labels if label in declared_corrected]
        if corrected:
            raise PermissionError(
                "Refusing to recompress corrected labels without explicit "
                f"confirmation: {corrected}"
            )

    return [_recompress(label_path, dry_run=dry_run) for label_path in labels]


def _recompress(label_path: Path, *, dry_run: bool) -> LabelCompressionResult:
    """Re-encode one label, keeping the result only if it is worth keeping.

    The encoding always happens -- there is no way to know what it would save
    without doing it -- so a dry run is the same work as a real run, minus the
    rename. Three things stop the rename: the new encoding is no smaller, the
    caller only asked what would happen, or the label changed on disk while it
    was being re-encoded. That last one would put the old pixels back over
    somebody else's write, so it is reported rather than passed over.
    """
    import tifffile

    original_size = label_path.stat().st_size
    before = fingerprint_of(label_path)
    array = tifffile.imread(str(label_path))

    new_size = original_size
    stale = False

    def commit(temporary: Path) -> bool:
        nonlocal new_size, stale
        new_size = temporary.stat().st_size
        if new_size >= original_size:
            return False
        if dry_run:
            return False
        if fingerprint_of(label_path) != before:
            logger.warning(
                "%s changed on disk while it was being recompressed; "
                "leaving it as it now is", label_path,
            )
            stale = True
            return False
        return True

    applied = write_atomically(
        label_path,
        lambda temporary: save_label_image(array, temporary, array.dtype),
        commit_if=commit,
    )
    if applied:
        logger.info("Compressed %s: %d -> %d bytes", label_path, original_size, new_size)
    return LabelCompressionResult(label_path, original_size, new_size, applied, stale=stale)
