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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..history import Recorder
from ..io.atomic import fingerprint_of, write_atomically
from ..io.image import read_array
from ..io.long_paths import refuse_unreachable
from ..models import ImageRecord
from ..progress import OperationCancelled
from .label_io import has_image_stack_support, label_fields, save_label_image
from .selection import current_state, declared_states

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
        problem: Why the label could not be read or re-encoded without
            altering a pixel, so it was left alone; empty where it could.
    """

    label_path: Path
    original_size: int
    new_size: int
    applied: bool
    stale: bool = False
    problem: str = ""


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


def corrected_labels(records: Iterable[ImageRecord], labels: Iterable[Path]) -> list[Path]:
    """Return those of *labels* whose group is corrected, by its record or its state file now.

    One answer for the preview and the run, so the two cannot disagree.

    Raises:
        StateFileError: If a stage's state file cannot be read.
    """
    labels = list(labels)
    remembered = {
        label
        for record in records
        if current_state(record) == "corrected"
        for label in record.labels
    }
    declared = declared_states(labels)
    return [
        label for label in labels
        if label in remembered or declared[label] == "corrected"
    ]


def compress_labels(
    records: Iterable[ImageRecord],
    *,
    allow_corrected: bool = False,
    dry_run: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    recorder: Recorder | None = None,
) -> list[LabelCompressionResult]:
    """Recompress every on-disk label referenced by *records*.

    Validates the whole batch before changing anything: if any label belongs to
    a group declared corrected -- by its record, or its state file now -- the whole call is refused unless
    *allow_corrected* is True, since corrected labels are user-generated
    data that must never be modified without explicit confirmation.

    Each file is written through :func:`~..io.atomic.write_atomically`, so a
    crash mid-write can never leave a truncated label, and the label keeps the
    permissions it had rather than the owner-only ones a temporary file is
    created with. The original is left untouched when *dry_run* is True, when
    recompression would not shrink the file, when the label changed on disk
    while it was being re-encoded -- recompressing pixels that have since been
    replaced would put the old ones back -- and when it will not read, or will
    not re-encode without altering a pixel, which its result's ``problem`` says.

    Args:
        records: Records whose labels should be checked (as produced by the
            scanner / filtering module).
        allow_corrected: Set True to permit recompressing labels whose group
            is declared corrected.
        dry_run: If True, measure potential savings without modifying any
            files.
        on_progress: Called as ``on_progress(done, total)`` after each label,
            counted in labels rather than records because a shared label is one
            piece of work however many images reference it. Raising
            :class:`~..progress.OperationCancelled` from it stops the run.

    Returns:
        One :class:`LabelCompressionResult` per label file examined, in the
        order they were examined. Shorter than :func:`existing_labels` when the
        run was stopped through *on_progress*.

    Raises:
        PathTooLong: If Windows cannot reach a label's temporary copy here.
            Checked before anything is written.
        StateFileError: If a stage's state file cannot be read, so whether a
            label there is corrected cannot be told.
    """
    if not has_image_stack_support():
        raise RuntimeError(
            "tifffile/numpy/cv2 are required to compress labels: install the 'gui' extra"
        )

    records = list(records)
    labels = existing_labels(records)
    if not allow_corrected:
        corrected = corrected_labels(records, labels)
        if corrected:
            raise PermissionError(
                "Refusing to recompress corrected labels without explicit "
                f"confirmation: {corrected}"
            )

    if not dry_run:
        refuse_unreachable(labels)
    results = []
    try:
        for done, label_path in enumerate(labels, start=1):
            result = _recompress(label_path, dry_run=dry_run)
            results.append(result)
            if recorder is not None and result.applied:
                _record_compression(recorder, result)
            if on_progress is not None:
                on_progress(done, len(labels))
    except OperationCancelled:
        # Returned rather than raised: stopping lands between whole files, so
        # what is in *results* is finished work and the rest were never opened.
        # A caller tells a stopped run from a complete one by comparing this
        # against :func:`existing_labels`, which is why that is public.
        logger.info("Compression stopped after %d of %d label(s)", len(results), len(labels))
    return results


def _record_compression(recorder: Recorder, result: LabelCompressionResult) -> None:
    """Record a recompression, which changed the file but not the segmentation.

    The digest before and after are the same by construction -- that is what
    makes this a re-encoding rather than an edit -- and both are written so a
    reader can check the claim rather than infer it from an absence.
    """

    try:
        digest = label_fields(read_array(result.label_path))["digest"]
    except Exception as error:  # noqa: BLE001 - a damaged file raises whatever its decoder does
        logger.warning("Not recording %s: %s", result.label_path, error)
        return
    recorder.record(
        result.label_path, "label-compressed",
        digest=digest, parent_digest=digest,
        bytes_before=result.original_size, bytes_after=result.new_size,
    )


def _recompress(label_path: Path, *, dry_run: bool) -> LabelCompressionResult:
    """Re-encode one label, keeping the result only if it is worth keeping.

    The encoding always happens -- there is no way to know what it would save
    without doing it -- so a dry run is the same work as a real run, minus the
    rename. Three things stop the rename: the new encoding is no smaller, the
    caller only asked what would happen, or the label changed on disk while it
    was being re-encoded. That last one would put the old pixels back over
    somebody else's write, so it is reported rather than passed over. A label
    that will not read, or not re-encode without altering a pixel, is left
    alone with the reason, and the batch goes on.
    """
    import tifffile

    # Before the file is opened, so a run that stalls names the file it stalled
    # on rather than the last one it finished.
    logger.debug("Recompressing %s", label_path)
    original_size = label_path.stat().st_size
    before = fingerprint_of(label_path)
    try:
        array = tifffile.imread(str(label_path))
    except Exception as error:  # noqa: BLE001 - one unreadable label must not stop the batch
        return _left_alone(label_path, original_size, f"could not be read: {error}")

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

    try:
        applied = write_atomically(
            label_path,
            lambda temporary: save_label_image(array, temporary, array.dtype),
            commit_if=commit,
        )
    except (ValueError, OSError) as error:
        return _left_alone(label_path, original_size, str(error))
    if applied:
        # Debug, not info: this runs once per label over a whole collection,
        # and both front ends already say the same thing in a form that suits
        # them -- the command line prints its own line per result, the window
        # lists every file in the report. A third copy is thousands of writes
        # to stdout from a worker thread, which is a flood at best and back
        # pressure on that thread at worst.
        logger.debug("Compressed %s: %d -> %d bytes", label_path, original_size, new_size)
    return LabelCompressionResult(label_path, original_size, new_size, applied, stale=stale)


def _left_alone(label_path: Path, original_size: int, problem: str) -> LabelCompressionResult:
    """Return the result for a label this run could not safely re-encode."""
    logger.warning("Left %s alone: %s", label_path, problem)
    return LabelCompressionResult(
        label_path, original_size, original_size, applied=False, problem=problem
    )
