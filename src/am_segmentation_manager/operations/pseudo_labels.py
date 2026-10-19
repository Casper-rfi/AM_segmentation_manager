"""Turning predictions into the pseudo-labels that are then proofread.

A prediction becomes a label, resized by nearest neighbour where its shape
differs from its image's, or copied with a warning without the image stack.
Only predictions the pipeline has not post-processed are eligible; where an
image has both this package's and the microscope's, the caller says which to
take and the other stands in where it is missing. An existing label is replaced
only when the caller asks, and a proofread one never. Every prediction of a
batch is read and checked before its first label is written.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal

from ..history import Recorder, containing_project
from ..io.atomic import Fingerprint, fingerprint_of, write_atomically
from ..io.image import to_2d
from ..io.long_paths import refuse_unreachable
from ..layout import (
    DEFAULT_PREDICTION_ORIGIN,
    MODALITIES,
    PREDICTION_ORIGIN_NAMES,
    PredictionOrigin,
    is_convertible_prediction,
    prediction_origin,
    prediction_precedence,
)
from ..models import ImageRecord
from ..progress import StoppedPartway
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
from .label_save import StaleLabels
from .predict import read_provenance
from .selection import current_state, declared_states

logger = logging.getLogger(__name__)


def prediction_to_label(
    prediction_path: Path,
    label_path: Path,
    target_image_path: Path | None = None,
    *,
    commit_if: Callable[[Path], bool] | None = None,
) -> bool:
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
        commit_if: As :func:`~..io.atomic.write_atomically`: asked just before
            the rename whether to go ahead.

    Returns:
        Whether *label_path* was written.
    """
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

    return write_atomically(
        label_path,
        lambda temporary: _write_label(prediction_path, temporary, target_image_path),
        commit_if=commit_if,
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

    import tifffile  # type: ignore

    # read images
    try:
        pred = tifffile.imread(str(prediction_path))
        target = tifffile.imread(str(target_image_path))
    except Exception as exc:  # noqa: BLE001 - a damaged file raises whatever its decoder does
        logger.warning(
            "Could not read TIFFs for resizing (%s); copying prediction instead: %s",
            exc,
            prediction_path,
        )
        shutil.copy2(str(prediction_path), str(label_path))
        return

    # handle multi-page or channel dimensions: reduce to 2D if necessary
    pred2 = to_2d(pred)
    target2 = to_2d(target)
    label = as_converted(pred2, target2.shape[:2], prediction_path)
    save_label_image(label, label_path, label.dtype)
    if pred2.shape == target2.shape:
        logger.info("Wrote label %s (shapes matched)", label_path)
    else:
        logger.info("Resized prediction %s (%s) -> %s (%s)", prediction_path, pred2.shape, label_path, label.shape)


def as_converted(prediction, shape: tuple[int, int], prediction_path: Path):
    """Return *prediction* as the label converting it to *shape* would write.

    Class indices, resized by nearest neighbour where the shapes differ. The one
    resize a conversion and a comparison against a prediction share, so what is
    compared is pixel for pixel what converting would write.

    Raises:
        ValueError: As :func:`_class_indices`, for values that are not class
            indices.
    """
    import cv2  # type: ignore

    indices = _class_indices(to_2d(prediction), prediction_path)
    if indices.shape == tuple(shape):
        return indices
    height, width = shape
    return cv2.resize(indices, (width, height), interpolation=cv2.INTER_NEAREST)


def _class_indices(prediction, prediction_path: Path):
    """Return *prediction* as 8-bit class indices, refusing any value that would change.

    Raises:
        ValueError: For a value that is not a whole number from 0 to 255 -- a
            probability map, or a class map past 255 -- which the cast would
            silently turn into another class, or into background.
    """
    import numpy as np  # type: ignore

    with np.errstate(invalid="ignore", over="ignore"):
        indices = prediction.astype(np.uint8)
    if not np.array_equal(indices, prediction):
        raise ValueError(
            f"{prediction_path.name} holds {prediction.dtype} values from "
            f"{prediction.min()} to {prediction.max()}, not class indices from 0 to "
            f"255; converting it would change them"
        )
    return indices


class ConversionStopped(StoppedPartway):
    """Raised when a conversion stopped partway, having written what it carries."""


class UnconvertiblePredictions(ValueError):
    """Raised before the first write, when a prediction will not read or is not class indices.

    Attributes:
        refused: Each such prediction, with why.
    """

    def __init__(self, refused: Iterable[tuple[Path, str]]) -> None:
        self.refused: tuple[tuple[Path, str], ...] = tuple(refused)
        first = self.refused[0][1]
        more = len(self.refused) - 1
        super().__init__(
            f"{len(self.refused)} prediction(s) cannot be converted, so nothing was converted. "
            + first + (f" (and {more} more, listed in the log)" if more else "")
        )


def unconvertible_predictions(predictions: Iterable[Path]) -> list[tuple[Path, str]]:
    """Return each of *predictions* a conversion would refuse, with why, reading every one.

    Empty without the image stack, where a conversion copies each prediction unread.
    """
    if not has_image_stack_support():
        return []
    import tifffile  # type: ignore

    refused = []
    for prediction in predictions:
        try:
            pixels = tifffile.imread(str(prediction))
        except Exception as error:  # noqa: BLE001 - a damaged file raises whatever its decoder does
            refused.append((prediction, f"{prediction.name} could not be read: {error}"))
            continue
        if pixels.size == 0:
            refused.append((prediction, f"{prediction.name} holds no pixels"))
            continue
        try:
            _class_indices(to_2d(pixels), prediction)
        except ValueError as error:
            refused.append((prediction, str(error)))
    return refused


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
        fingerprint: The destination as it was when *existing* was decided,
            None where there was nothing. A conversion writes only over that.
    """

    record: ImageRecord
    prediction: Path
    destination: Path
    existing: ExistingLabel = "none"
    fallback: bool = False
    fingerprint: Fingerprint | None = None

    @property
    def origin(self) -> PredictionOrigin:
        """Return which of the two sources the prediction came from."""
        return prediction_origin(self.prediction)

    def describe(self) -> str:
        """Return what this conversion would do to the destination, and from what."""
        return f"{EXISTING_DESCRIPTIONS[self.existing]}, from {PREDICTION_ORIGIN_NAMES[self.origin]}"


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
    then -- proofread as the stage's state file declares it now, not as the
    record remembers it. The whole set is resolved before any of it is written,
    and no destination appears twice -- conditions of one scene share a label
    file, and listing it twice would have the second write silently replace the
    first.

    Raises:
        StateFileError: If a stage's state file cannot be read, so whether a
            label there is proofread cannot be told.
    """
    planned: list[PlannedConversion] = []
    replaceable: list[PlannedConversion] = []
    refused: list[PlannedConversion] = []
    skipped: list[SkippedConversion] = []
    claimed: set[Path] = set()
    candidates: list[tuple[ImageRecord, Path, Path]] = []
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
        candidates.append((record, convertible[0], destination))

    declared = declared_states(destination for _, _, destination in candidates)
    for record, prediction, destination in candidates:
        corrected = "corrected" in (current_state(record), declared[destination])
        fingerprint = fingerprint_of(destination)
        existing = classify_destination(destination, corrected=corrected)
        conversion = PlannedConversion(
            record, prediction, destination, existing,
            fallback=prediction_origin(prediction) != prefer,
            fingerprint=fingerprint,
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


#: What a conversion leaves of a label it replaces: nothing worth keeping, a
#: copy in the archive, or nothing kept because its prediction can make it again.
Replaced = Literal["placeholder", "kept", "reconvertible"]


def what_replacing_leaves(
    conversions: Iterable[PlannedConversion], recorder: Recorder
) -> dict[Path, Replaced]:
    """Return what replacing each existing label in *conversions* leaves of it.

    Decided as the run's :class:`~.archive.Archivist` decides at the write, so a
    preview cannot promise a copy the run will not keep. Assumes the run
    records, as every writing run does through ``writing_recorder``; a label
    that will not open is kept as its bytes, so it counts as kept.
    """
    archivist = Archivist(recorder)
    found: dict[Path, Replaced] = {}
    for conversion in conversions:
        if conversion.existing == "placeholder":
            found[conversion.destination] = "placeholder"
        elif conversion.existing == "annotated":
            kept = _would_keep(archivist, conversion.destination)
            found[conversion.destination] = "kept" if kept else "reconvertible"
    return found


def _would_keep(archivist: Archivist, destination: Path) -> bool:
    """Return whether replacing the label at *destination* would archive it."""
    root = containing_project(destination)
    previous = read_label(destination)
    if root is None or previous is None:
        return True
    try:
        target = destination.resolve().relative_to(root).as_posix()
        digest = str(label_fields(previous)["digest"])
    except ValueError:
        return True
    keep, _reason = archivist.decision(root, target, digest, False)
    return keep


def convert_record_predictions(
    records: Iterable[ImageRecord],
    *,
    overwrite: bool = False,
    recorder: Recorder | None = None,
    prefer: PredictionOrigin = DEFAULT_PREDICTION_ORIGIN,
    on_progress: Callable[[int, int], None] | None = None,
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
        on_progress: Called as ``on_progress(done, total)`` once before the
            first label and after each label is written and recorded.

    Returns:
        Where each label was written. The paths rather than a count, so a
        caller can name the new labels on the records that share them instead
        of re-reading the collection to find them; ``len`` gives the count.

    Raises:
        PathTooLong: If Windows cannot reach a label or the archived copy of
            what it replaces here. Checked before anything is written.
        UnconvertiblePredictions: If a prediction will not read or is not
            class indices. Every one is read before anything is written.
        ConversionStopped: If a label could not be written, or the version it
            replaces could not be kept. That label and every one after it are
            left as they were; the error carries what was written before it.
    """
    plan = plan_conversions(records, overwrite=overwrite, prefer=prefer)
    destinations = [conversion.destination for conversion in plan.conversions]
    refuse_unreachable([*destinations, *archive_paths_for(recorder, destinations)])
    refused = unconvertible_predictions(
        conversion.prediction for conversion in plan.conversions
    )
    for _prediction, reason in refused:
        logger.warning("Cannot convert: %s", reason)
    if refused:
        raise UnconvertiblePredictions(refused)
    archivist = None
    if recorder is not None:

        archivist = Archivist(recorder)

    written: list[Path] = []
    total = len(plan.conversions)
    if on_progress is not None:
        # Before the first label as well, so a bar a previous phase filled
        # starts again at nothing rather than sitting full through this one.
        on_progress(0, total)
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
        try:
            archived = archive_replaced(archivist, conversion.destination, previous)
            # Asked again just before the rename: a label that appeared, or
            # changed, since the plan looked is not what the plan decided about.
            if not prediction_to_label(
                conversion.prediction,
                conversion.destination,
                target_image_path=conversion.record.image_path,
                commit_if=partial(_unchanged, conversion),
            ):
                raise StaleLabels(
                    f"{conversion.destination.name} changed on disk while the "
                    f"conversion was running, so it was not written over"
                )
        except Exception as error:
            raise ConversionStopped(
                f"Converted {len(written)} of {total} label(s), then "
                f"stopped at {conversion.destination.name}: {error}",
                written,
            ) from error
        written.append(conversion.destination)
        if recorder is not None:
            _record_conversion(recorder, conversion, previous, archived)
        if on_progress is not None:
            on_progress(len(written), total)
    return tuple(written)


def _unchanged(conversion: PlannedConversion, _temporary: Path) -> bool:
    """Return whether *conversion*'s destination is as it was when planned."""
    return fingerprint_of(conversion.destination) == conversion.fingerprint


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
