"""Turning predictions into the pseudo-labels that are then proofread.

A prediction becomes a label, resized by nearest neighbour where its shape
differs from its image's, or copied with a warning without the image stack.
Only predictions the pipeline has not post-processed are eligible; where an
image has both this package's and the microscope's, the caller says which to
take and the other stands in where it is missing. An existing label is replaced
only when the caller asks, and a proofread one never.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from ..history import Recorder
from ..io.atomic import write_atomically
from ..io.image import to_2d
from ..io.long_paths import refuse_unreachable
from ..layout import (
    DEFAULT_PREDICTION_ORIGIN,
    MODALITIES,
    PredictionOrigin,
    is_convertible_prediction,
    prediction_origin,
    prediction_precedence,
)
from ..models import ImageRecord
from .archive import Archivist, archive_paths_for, archive_replaced
from .label_io import (
    has_image_stack_support,
    is_label_file_empty,
    label_destination,
    label_fields,
    read_label,
    replacement_fields,
    save_label_image,
)
from .predict import read_provenance
from .selection import current_state

logger = logging.getLogger(__name__)


def prediction_to_label(prediction_path: Path, label_path: Path, target_image_path: Path | None = None) -> None:
    """Convert a prediction file into a label file, whole or not at all.

    Resized to *target_image_path* by nearest neighbour where their dimensions
    differ, which preserves the discrete label values; copied unchanged, with a
    warning, when the optional image stack is absent.

    Args:
        prediction_path: Path to the prediction image file.
        label_path: Destination path for the label image. Written by rename, so
            an interrupted run cannot truncate a label already there.
        target_image_path: Optional path to the SEM/FIB image whose shape should
            be matched.
    """
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

    write_atomically(
        label_path,
        lambda temporary: _write_label(prediction_path, temporary, target_image_path),
    )


def _write_label(prediction_path: Path, label_path: Path, target_image_path: Path | None) -> None:
    """Write the converted prediction to *label_path*, resizing where needed.

    Separate from :func:`prediction_to_label` only so that every path through
    the conversion -- copy, matched shapes, resized -- lands in one temporary
    file that the caller then moves into place.
    """
    if target_image_path is None:
        # nothing to match, do a direct copy
        shutil.copy2(str(prediction_path), str(label_path))
        logger.info("Copied prediction %s -> %s (no target provided)", prediction_path, label_path)
        return

    if not has_image_stack_support():
        logger.warning("Image processing libs (tifffile/numpy/cv2) not available; copying prediction without resizing")
        shutil.copy2(str(prediction_path), str(label_path))
        return

    # perform resizing using tifffile + numpy + cv2
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import tifffile  # type: ignore

    # read images
    try:
        pred = tifffile.imread(str(prediction_path))
        target = tifffile.imread(str(target_image_path))
    except Exception as exc:
        logger.warning(
            "Could not read TIFFs for resizing (%s); copying prediction instead: %s",
            exc,
            prediction_path,
        )
        shutil.copy2(str(prediction_path), str(label_path))
        return

    # handle multi-page or channel dimensions: reduce to 2D if necessary
    pred2 = to_2d(pred).astype(np.uint8)
    target2 = to_2d(target)

    if pred2.shape == target2.shape:
        # shapes already match; write prediction directly
        save_label_image(pred2, label_path, pred2.dtype)
        logger.info("Wrote label %s (shapes matched)", label_path)
        return

    # resize prediction to match target shape (nearest neighbour)
    target_h, target_w = target2.shape[:2]
    resized = cv2.resize(pred2, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    # ensure integer type for labels
    if not np.issubdtype(resized.dtype, np.integer):
        resized = resized.astype(np.uint8)

    save_label_image(resized, label_path, resized.dtype)
    logger.info("Resized prediction %s (%s) -> %s (%s)", prediction_path, pred2.shape, label_path, resized.shape)


#: What is already at a conversion's destination. The three that exist differ in
#: what replacing them costs: nothing, nothing, and irreplaceable work.
ExistingLabel = Literal["none", "placeholder", "annotated", "corrected"]

EXISTING_DESCRIPTIONS: Final[dict[ExistingLabel, str]] = {
    "none": "write",
    "placeholder": "replace (empty placeholder)",
    "annotated": "replace (holds annotation)",
    "corrected": "refused (corrected)",
}


@dataclass(frozen=True)
class PlannedConversion:
    """One prediction and the label file it would become.

    Attributes:
        record: The image the prediction belongs to.
        prediction: The prediction file to convert.
        destination: Where the label would be written. An existing label's own
            path when there is one, so replacing it overwrites that file rather
            than writing a second label elsewhere.
        existing: What is at the destination now.
        fallback: Whether the prediction is from the source not asked for,
            the image having none from the one that was.
    """

    record: ImageRecord
    prediction: Path
    destination: Path
    existing: ExistingLabel = "none"
    fallback: bool = False

    @property
    def origin(self) -> PredictionOrigin:
        """Return which of the two sources the prediction came from."""
        return prediction_origin(self.prediction)

    def describe(self) -> str:
        """Return what this conversion would do to the destination, and from what."""
        return f"{EXISTING_DESCRIPTIONS[self.existing]}, from {self.origin}"


#: Why a record produces no conversion. Reported rather than logged, because a
#: run that converts nothing is the case most in need of an explanation.
SkipReason = Literal["no-prediction", "post-processed"]

SKIP_DESCRIPTIONS: Final[dict[SkipReason, str]] = {
    "no-prediction": "no prediction file",
    "post-processed": "only post-processed predictions, whose values do not "
                      "match the segmentation classes",
}


@dataclass(frozen=True)
class SkippedConversion:
    """One record that will not be converted, and why.

    Attributes:
        record: The image passed over.
        reason: Which of :data:`SKIP_DESCRIPTIONS` applies.
    """

    record: ImageRecord
    reason: SkipReason


@dataclass(frozen=True)
class ConversionPlan:
    """Everything a conversion run would and would not do.

    Split four ways because each needs a different answer: run it, confirm the
    loss, set the group back to selected first, or wait for a prediction.

    Attributes:
        conversions: The labels that would be written now, in record order.
        replaceable: Conversions held back because a label is already there and
            overwriting was not asked for. Empty when it was.
        refused: Conversions that will not run at any setting, because a
            proofread label is at the destination.
        skipped: Records with nothing to convert, each with its reason.
    """

    conversions: tuple[PlannedConversion, ...] = ()
    replaceable: tuple[PlannedConversion, ...] = ()
    refused: tuple[PlannedConversion, ...] = ()
    skipped: tuple[SkippedConversion, ...] = ()

    def count_skipped(self, reason: SkipReason) -> int:
        """Return how many records were passed over for *reason*."""
        return sum(1 for item in self.skipped if item.reason == reason)

    def count_fallbacks(self) -> int:
        """Return how many conversions, run or held back, use the other source."""
        return sum(
            1 for item in (*self.conversions, *self.replaceable) if item.fallback
        )


@dataclass(frozen=True)
class ConversionCounts:
    """How a set of records divides up, counted from the index alone.

    Opens no file and walks no directory, so it can be recomputed whenever the
    selection changes; this counts images, while :func:`plan_conversions` is the
    authority on how many label files a run would write.

    Attributes:
        total: Images considered.
        ready: Images with a convertible prediction and no label yet.
        labelled: Images whose existing label could be replaced on confirmation.
        corrected: Images whose label is proofread, which nothing will replace.
        post_processed: Images whose only predictions are the unusable kind.
        no_prediction: Images the prediction pipeline has not reached.
        fallback: Of the ready and labelled images, those with no prediction
            from the preferred source, whose other one would be used.
    """

    total: int = 0
    ready: int = 0
    labelled: int = 0
    corrected: int = 0
    post_processed: int = 0
    no_prediction: int = 0
    fallback: int = 0


def count_convertible(
    records: Iterable[ImageRecord],
    *,
    prefer: PredictionOrigin = DEFAULT_PREDICTION_ORIGIN,
) -> ConversionCounts:
    """Return how *records* divide up, without touching the filesystem."""
    counts: dict[str, int] = {}
    total = 0
    for record in records:
        if record.modality not in MODALITIES:
            continue
        total += 1
        sources = {
            prediction_origin(path)
            for path in record.predictions if is_convertible_prediction(path)
        }
        if not sources:
            key = "post_processed" if record.predictions else "no_prediction"
        elif not record.labels:
            key = "ready"
        elif current_state(record) == "corrected":
            key = "corrected"
        else:
            key = "labelled"
        counts[key] = counts.get(key, 0) + 1
        if key in ("ready", "labelled") and prefer not in sources:
            counts["fallback"] = counts.get("fallback", 0) + 1
    return ConversionCounts(total=total, **counts)


def classify_destination(
    label_path: Path | None, *, corrected: bool = False
) -> ExistingLabel:
    """Return what replacing *label_path* would cost.

    A label that cannot be read counts as annotated, being the last thing to
    overwrite on an assumption.

    Args:
        corrected: Whether the group owning this label is declared corrected.
            Passed in rather than read from the path, which no longer says.
    """
    if label_path is None or not label_path.exists():
        return "none"
    if corrected:
        return "corrected"
    return "placeholder" if is_label_file_empty(label_path) is True else "annotated"


def plan_conversions(
    records: Iterable[ImageRecord],
    *,
    overwrite: bool = False,
    prefer: PredictionOrigin = DEFAULT_PREDICTION_ORIGIN,
) -> ConversionPlan:
    """Return the conversions that would run for *records*, and what they do not.

    Sources are whatever :func:`~..layout.is_convertible_prediction` allows. An
    image with predictions from both sources converts the *prefer* one; an image
    with only the other converts that, marked as a fallback. An existing
    label is replaced only when *overwrite* is set, and a proofread one not even
    then. The whole set is resolved before any of it is written, and no
    destination appears twice -- conditions of one scene share a label file, and
    listing it twice would have the second write silently replace the first.
    """
    planned: list[PlannedConversion] = []
    replaceable: list[PlannedConversion] = []
    refused: list[PlannedConversion] = []
    skipped: list[SkippedConversion] = []
    claimed: set[Path] = set()
    for record in records:
        if record.modality not in MODALITIES:
            continue
        convertible = sorted(
            (
                prediction for prediction in record.predictions
                if is_convertible_prediction(prediction)
            ),
            key=lambda prediction: prediction_precedence(prediction, prefer),
        )
        if not convertible:
            # Distinguished so the report can say which of the two it is: one
            # means the pipeline has not run, the other that it ran and its
            # output cannot be used.
            reason: SkipReason = "post-processed" if record.predictions else "no-prediction"
            skipped.append(SkippedConversion(record, reason))
            continue

        destination = label_destination(record)
        if destination in claimed:
            continue
        claimed.add(destination)

        existing = classify_destination(
            destination, corrected=current_state(record) == "corrected"
        )
        conversion = PlannedConversion(
            record, convertible[0], destination, existing,
            fallback=prediction_origin(convertible[0]) != prefer,
        )
        if existing == "corrected":
            refused.append(conversion)
        elif existing == "none" or overwrite:
            planned.append(conversion)
        else:
            replaceable.append(conversion)
    return ConversionPlan(
        tuple(planned), tuple(replaceable), tuple(refused), tuple(skipped)
    )


def convert_record_predictions(
    records: Iterable[ImageRecord],
    *,
    overwrite: bool = False,
    recorder: Recorder | None = None,
    prefer: PredictionOrigin = DEFAULT_PREDICTION_ORIGIN,
) -> tuple[Path, ...]:
    """Convert predictions for explicitly selected image records.

    Corrected labels are never overwritten, whatever *overwrite* is set to. A
    label is written to the directory its image set resolves to, which is also
    where an existing one is replaced.

    Args:
        records: The records to convert predictions for.
        overwrite: Replace an existing label that is not proofread.
        recorder: Where to record what was written, or None to record nothing.
        prefer: The source to convert where an image has predictions from both.

    Returns:
        Where each label was written. The paths rather than a count, so a
        caller can name the new labels on the records that share them instead
        of re-reading the collection to find them; ``len`` gives the count.

    Raises:
        PathTooLong: If Windows cannot reach a label or the archived copy of
            what it replaces here. Checked before anything is written.
        ArchiveFailed: If the version a label replaces could not be kept. That
            label and every one after it are left as they were.
    """
    plan = plan_conversions(records, overwrite=overwrite, prefer=prefer)
    destinations = [conversion.destination for conversion in plan.conversions]
    refuse_unreachable([*destinations, *archive_paths_for(recorder, destinations)])
    archivist = None
    if recorder is not None:

        archivist = Archivist(recorder)

    for conversion in plan.conversions:
        if conversion.existing != "none":
            logger.info(
                "Replacing %s label %s from %s",
                conversion.existing, conversion.destination, conversion.prediction,
            )
        previous = (
            read_label(conversion.destination)
            if recorder is not None and conversion.existing != "none"
            else None
        )
        archived = archive_replaced(archivist, conversion.destination, previous)
        prediction_to_label(
            conversion.prediction,
            conversion.destination,
            target_image_path=conversion.record.image_path,
        )
        if recorder is not None:
            _record_conversion(recorder, conversion, previous, archived)
    return tuple(conversion.destination for conversion in plan.conversions)


def _record_conversion(
    recorder: Recorder,
    conversion: PlannedConversion,
    previous,
    archived: str | None = None,
) -> None:
    """Record one label made from a prediction, and which model made it.

    The model is read from the predictions directory's own provenance file,
    which is the only account of what filled it; a prediction the pipeline
    wrote rather than this package has none, and says so.
    """

    written = read_label(conversion.destination)
    if written is None:
        return
    try:
        fields = label_fields(written)
    except ValueError as error:
        logger.warning("Not recording %s: %s", conversion.destination, error)
        return

    fields["prediction_kind"] = conversion.prediction.parent.name
    fields["model"] = _recorded_model(conversion.prediction)
    fields.update(
        replacement_fields(conversion.destination, previous, written, archived)
    )
    recorder.record(conversion.destination, "label-converted", **fields)


def _recorded_model(prediction: Path) -> str:
    """Return the model named for *prediction*, or that nothing names one."""

    entry = read_provenance(prediction.parent).get(prediction.name)
    if isinstance(entry, dict):
        model = entry.get("model")
        if isinstance(model, str) and model:
            return model
    return "unrecorded"
