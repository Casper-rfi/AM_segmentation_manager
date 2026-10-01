"""Producing predictions for selected images with a trained model.

Nothing here imports torch: the model arrives as a callable and a
:class:`PredictionSource`, so planning, resumption, provenance and writing test
without the inference stack, :mod:`..ml.model` supplying them. One prediction is
made per label group, from its primary condition, and recorded with it, and
predictions are written only to
the directory this package owns, so a run never touches the pipeline's output.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from ..families import FamiliesLookup
from ..io.atomic import write_atomically
from ..io.image import read_array, to_2d
from ..io.long_paths import refuse_unreachable
from ..io.yaml import read_yaml, write_yaml
from ..layout import (
    MODALITIES,
    OWN_PREDICTION_KIND,
    PREDICTION_PROVENANCE_FILENAME,
)
from ..models import ImageRecord
from .label_groups import build_groups, primary_record
from .label_io import save_label_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionSource:
    """Which model is making the predictions, for provenance and for checking.

    Attributes:
        name: The weights file's name, as it would be typed to re-run this.
        run_name: The training run recorded inside the weights, or ``None``.
            Two checkpoints can share a filename; the run is what identifies
            what actually produced a prediction.
        generation: How the model is run, recorded but not matched on: one
            checkpoint is of one generation, so the model and run already say it.
    """

    name: str
    run_name: str | None = None
    generation: str | None = None

    def matches(self, entry: Mapping[str, Any]) -> bool:
        """Return whether a provenance entry records this same model."""
        return entry.get("model") == self.name and entry.get("run") == self.run_name


#: What running a prediction would do at its destination.
PredictionOutcome = Literal["write", "replace", "up-to-date"]

OUTCOME_DESCRIPTIONS: Final[dict[PredictionOutcome, str]] = {
    "write": "write",
    "replace": "replace (predicted before)",
    "up-to-date": "skip (already predicted by this model)",
}


@dataclass(frozen=True)
class PlannedPrediction:
    """One image and the prediction file it would produce.

    Attributes:
        record: The image to run the model over.
        destination: Where the prediction would be written.
        outcome: What running it would do to that destination.
        shared_with: How many other images this prediction serves without
            their being predicted themselves.
    """

    record: ImageRecord
    destination: Path
    outcome: PredictionOutcome
    shared_with: int = 0

    def describe(self) -> str:
        """Return what this prediction would do to the destination."""
        return OUTCOME_DESCRIPTIONS[self.outcome]


@dataclass(frozen=True)
class PredictionPlan:
    """Everything a prediction run would and would not do.

    Attributes:
        predictions: What would be written, in record order.
        up_to_date: Images this model has already predicted, held back so a
            re-run costs only what an interrupted one did not reach.
        modalities: The modalities the selection holds, in display order. More
            than one refuses the run: nothing here knows which of them the
            model was trained for, so the selection has to say, and a selection
            saying both says nothing.
        represented: Images not predicted because another image's prediction
            serves them: one of their label group, or of their family where its
            labels are split. Counted rather than listed, being the normal case.
    """

    predictions: tuple[PlannedPrediction, ...] = ()
    up_to_date: tuple[PlannedPrediction, ...] = ()
    modalities: tuple[str, ...] = ()
    represented: int = 0

    @property
    def modality(self) -> str | None:
        """Return the one modality this run covers, or None if it is not one."""
        return self.modalities[0] if len(self.modalities) == 1 else None

    @property
    def conditions(self) -> tuple[tuple[str, int], ...]:
        """Return each imaging condition predicted from and how many times, most first.

        What says which image of a family the model sees; an image sitting
        directly in its stage has no condition and is not counted.
        """
        counts: dict[str, int] = {}
        for item in self.predictions:
            if item.record.condition is not None:
                counts[item.record.condition] = counts.get(item.record.condition, 0) + 1
        return tuple(sorted(counts.items(), key=lambda entry: (-entry[1], entry[0].lower())))

    @property
    def is_refused(self) -> bool:
        """Return whether the run must not proceed: it spans both modalities."""
        return len(self.modalities) > 1


def prediction_destination(record: ImageRecord) -> Path:
    """Return where *record*'s prediction belongs.

    **Read off the record, never re-derived from its path.** How far a
    predictions directory sits from an image depends on the shape of its image
    set, exactly as a label directory does: an Adaptive Polishing image sits one
    level below its stage in a directory naming a modality, while a reference
    image sits directly in a directory naming a lamella. Deriving from
    ``image_path.parent`` therefore resolved a reference image's predictions to
    a ``<lamella>_predictions`` outside the lamella -- somewhere the scanner
    never looks, so every prediction written there was reported as missing.

    Creates no directory and touches no file.

    Raises:
        ValueError: If the record does not say where its predictions live,
            which means it was built by something that has not been taught to.
    """
    if record.predictions_directory is None:
        raise ValueError(
            f"{record.image_path} does not record where its predictions live"
        )
    return record.predictions_directory / OWN_PREDICTION_KIND / record.image_path.name


def read_provenance(directory: Path) -> dict[str, Any]:
    """Return what is recorded about the predictions in *directory*.

    A missing or unreadable file reads as nothing recorded, which makes every
    prediction there look unaccounted for and so re-runnable -- the safe way for
    this to fail, since the alternative is skipping work on a claim that cannot
    be checked.
    """
    try:
        data = read_yaml(directory / PREDICTION_PROVENANCE_FILENAME)
    except RuntimeError as error:
        logger.warning("Could not read prediction provenance in %s: %s", directory, error)
        return {}
    recorded = data.get("predictions")
    return recorded if isinstance(recorded, dict) else {}


def write_provenance(directory: Path, entries: Mapping[str, Any]) -> None:
    """Record how each prediction in *directory* was made, rewriting the record whole."""
    write_yaml(
        directory / PREDICTION_PROVENANCE_FILENAME,
        {"predictions": dict(entries)},
        sort_keys=True,
    )


def plan_predictions(
    records: Iterable[ImageRecord],
    source: PredictionSource,
    *,
    overwrite: bool = False,
) -> PredictionPlan:
    """Return what predicting *records* with *source* would do.

    Expands to label groups and keeps each group's primary image, so a scene
    captured under several conditions is predicted once, from the condition
    :func:`~.label_groups.primary_record` chooses. **No destination is planned
    twice**, so groups meeting at one file are predicted once, from their
    primary (``test_a_family_in_conflict_is_predicted_once_per_destination``).
    The whole set is resolved before any of it runs.

    A selection spanning both modalities is refused rather than predicted.
    Nothing here knows which one the model was trained for -- a checkpoint does
    not say, and its filename is not evidence -- so the selection is the only
    statement of it, and one naming both states nothing. An SEM model over a FIB
    image returns a confident, entirely wrong prediction, so that has to be a
    choice somebody made rather than one nobody noticed.
    """
    predictions: list[PlannedPrediction] = []
    up_to_date: list[PlannedPrediction] = []
    represented = 0
    provenance: dict[Path, dict[str, Any]] = {}

    candidates = [record for record in records if record.modality in MODALITIES]
    modalities = tuple(
        modality for modality in MODALITIES
        if any(record.modality == modality for record in candidates)
    )
    if len(modalities) > 1:
        # Refused whole: resolving destinations for a run that will not happen
        # reads every stage directory for nothing.
        return PredictionPlan(modalities=modalities)

    families = FamiliesLookup()
    competing: dict[Path, list[tuple[ImageRecord, int]]] = {}
    for group in build_groups(candidates):
        chosen = primary_record(group.records, families)
        competing.setdefault(prediction_destination(chosen), []).append(
            (chosen, len(group.records))
        )

    for destination, groups in competing.items():
        # Several groups meet at one file only where a family's labels are
        # split across directories; one write serves them all.
        chosen = primary_record([record for record, _ in groups], families)
        covered = sum(size for _, size in groups)
        represented += covered - 1

        if destination.parent not in provenance:
            provenance[destination.parent] = read_provenance(destination.parent)
        recorded = provenance[destination.parent].get(destination.name)

        if not destination.exists():
            outcome: PredictionOutcome = "write"
        elif isinstance(recorded, dict) and source.matches(recorded) and not overwrite:
            outcome = "up-to-date"
        else:
            outcome = "replace"

        planned = PlannedPrediction(chosen, destination, outcome, shared_with=covered - 1)
        (up_to_date if outcome == "up-to-date" else predictions).append(planned)

    return PredictionPlan(
        tuple(predictions), tuple(up_to_date), modalities, represented
    )


def describe_conditions(conditions: Iterable[tuple[str, int]]) -> str:
    """Return ``" from sem"``, or ``" from sem (412) and sem-2kV (38)"``, or ``""`` for none.

    Counted only where there are several, one condition needing no count.
    """
    listed = list(conditions)
    if not listed:
        return ""
    if len(listed) == 1:
        return f" from {listed[0][0]}"
    named = [f"{condition} ({count})" for condition, count in listed]
    return f" from {', '.join(named[:-1])} and {named[-1]}"


def run_predictions(
    plan: PredictionPlan,
    predict: Callable[[np.ndarray], np.ndarray],
    source: PredictionSource,
    *,
    on_progress: Callable[[int, PlannedPrediction], None] | None = None,
) -> int:
    """Run *plan*, writing each prediction and recording how it was made.

    Provenance is flushed after every write rather than at the end, so a run
    stopped part way leaves an accurate account of what it did and the next run
    resumes from there.

    Args:
        plan: What to run, from :func:`plan_predictions`.
        predict: Turns one image array into a 2D array of class indices.
        source: The model doing it, recorded against every file written.
        on_progress: Called with the 1-based index and the prediction just
            written. Raising :class:`~..progress.OperationCancelled` from it
            stops the run between images, which is why there is no separate
            cancellation argument.

    Returns:
        How many predictions were written.

    Raises:
        RuntimeError: If the plan was refused, which the caller should have
            reported rather than run.
        OperationCancelled: If *on_progress* asked it to stop. Everything
            written up to that point is kept and recorded.
        PathTooLong: If Windows cannot reach a prediction here. Checked before
            anything is written.
    """
    if plan.is_refused:
        raise RuntimeError(
            f"Refusing to predict: the selection holds "
            f"{' and '.join(plan.modalities)} images, and one model is for one "
            f"of them"
        )

    refuse_unreachable(planned.destination for planned in plan.predictions)
    provenance: dict[Path, dict[str, Any]] = {}
    written = 0
    for planned in plan.predictions:
        image = to_2d(read_array(planned.record.image_path))
        mask = predict(image)

        def write(temporary: Path, mask: np.ndarray = mask) -> None:
            save_label_image(mask, temporary, mask.dtype)

        write_atomically(planned.destination, write)

        directory = planned.destination.parent
        if directory not in provenance:
            provenance[directory] = read_provenance(directory)
        provenance[directory][planned.destination.name] = {
            "model": source.name,
            "run": source.run_name,
            "generation": source.generation,
            "predicted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image": planned.record.image_path.name,
            "condition": planned.record.image_path.parent.name,
        }
        write_provenance(directory, provenance[directory])

        written += 1
        if on_progress is not None:
            on_progress(written, planned)
    return written
