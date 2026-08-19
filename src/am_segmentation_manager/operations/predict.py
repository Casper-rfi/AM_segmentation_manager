"""Producing predictions for selected images with a trained model.

Nothing here imports torch. The model arrives as a callable and a
:class:`PredictionSource` describing it, so planning, resumption, provenance and
the writes are all testable without the inference stack; :mod:`..ml.model` is
what supplies them, and the CLI wires the two together.

One prediction is made per label group, from the group's primary image, because
a prediction exists to become a label and a group has one label. Predictions are
written only to the directory this package owns, so a run can replace what it
produced and can never touch the segmentation pipeline's output.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from ..io.image import read_array, to_2d
from ..io.yaml import read_yaml
from ..layout import (
    MODALITIES,
    OWN_PREDICTION_KIND,
    PREDICTION_PROVENANCE_FILENAME,
    resolve_predictions_directory,
)
from ..models import ImageRecord
from .label_groups import build_groups, primary_record
from .label_io import save_label_image, write_atomically

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
    """

    name: str
    run_name: str | None = None

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
        shared_with: How many other images share this prediction's label, and
            therefore benefit from it without being predicted themselves.
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
        represented: Images not predicted because another image of their label
            group is. Counted rather than listed, being the normal case.
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
    def is_refused(self) -> bool:
        """Return whether the run must not proceed: it spans both modalities."""
        return len(self.modalities) > 1


def prediction_destination(record: ImageRecord) -> Path:
    """Return where *record*'s prediction belongs.

    The predictions directory its image set resolves to, under the subdirectory
    this package owns. Creating no directory and touching no file.
    """
    image_directory = record.image_path.parent
    if image_directory.name.lower() in {"deselected", "corrected"}:
        # The state folder is not the image set; its parent is.
        image_directory = image_directory.parent
    stage = image_directory.parent
    siblings = {child.name.lower(): child for child in stage.iterdir() if child.is_dir()}
    directory, _, _ = resolve_predictions_directory(
        image_directory, record.modality or MODALITIES[0], siblings
    )
    return directory / OWN_PREDICTION_KIND / record.image_path.name


def provenance_path(prediction: Path) -> Path:
    """Return the provenance file recording how *prediction* was made."""
    return prediction.parent / PREDICTION_PROVENANCE_FILENAME


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
    """Record how each prediction in *directory* was made.

    Written whole on every update, and by rename, so an interrupted run leaves
    the previous record intact rather than a truncated one.
    """
    import yaml

    write_atomically(
        directory / PREDICTION_PROVENANCE_FILENAME,
        lambda temporary: temporary.write_text(
            yaml.safe_dump({"predictions": dict(entries)}, sort_keys=True),
            encoding="utf-8",
        ),
    )


def plan_predictions(
    records: Iterable[ImageRecord],
    source: PredictionSource,
    *,
    overwrite: bool = False,
) -> PredictionPlan:
    """Return what predicting *records* with *source* would do.

    Expands to label groups and keeps each group's primary image, so a scene
    captured under several conditions is predicted once, from the image the
    curation window shows. The whole set is resolved before any of it runs.

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

    for group in build_groups(candidates):
        chosen = primary_record(group.records)
        represented += len(group.records) - 1

        destination = prediction_destination(chosen)
        if destination.parent not in provenance:
            provenance[destination.parent] = read_provenance(destination.parent)
        recorded = provenance[destination.parent].get(destination.name)

        if not destination.exists():
            outcome: PredictionOutcome = "write"
        elif isinstance(recorded, dict) and source.matches(recorded) and not overwrite:
            outcome = "up-to-date"
        else:
            outcome = "replace"

        planned = PlannedPrediction(
            chosen, destination, outcome, shared_with=len(group.records) - 1
        )
        (up_to_date if outcome == "up-to-date" else predictions).append(planned)

    return PredictionPlan(
        tuple(predictions), tuple(up_to_date), modalities, represented
    )


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
    """
    if plan.is_refused:
        raise RuntimeError(
            f"Refusing to predict: the selection holds "
            f"{' and '.join(plan.modalities)} images, and one model is for one "
            f"of them"
        )

    provenance: dict[Path, dict[str, Any]] = {}
    written = 0
    for planned in plan.predictions:
        image = to_2d(read_array(planned.record.image_path))
        mask = predict(image)

        write_atomically(
            planned.destination,
            lambda temporary, mask=mask: save_label_image(mask, temporary, mask.dtype),
        )

        directory = planned.destination.parent
        if directory not in provenance:
            provenance[directory] = read_provenance(directory)
        provenance[directory][planned.destination.name] = {
            "model": source.name,
            "run": source.run_name,
            "predicted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image": planned.record.image_path.name,
        }
        write_provenance(directory, provenance[directory])

        written += 1
        if on_progress is not None:
            on_progress(written, planned)
    return written
