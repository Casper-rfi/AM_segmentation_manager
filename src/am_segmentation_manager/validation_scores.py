"""What a checkpoint scored on each image of a list, and the metrics of any part of it.

One file per checkpoint holds a confusion matrix per image, so the images of one
instrument, voltage or kind of lamella are scored by adding up theirs, with no second
inference. **The metrics are AM-model-training's own, formula for formula**
(``test_per_image_scores_give_the_metrics_am_model_training_reports``): IoU and Dice
are each image's, averaged over the images a class is in or predicted in; F1,
accuracy, precision and recall are of the matrices summed. Needs no numpy and no torch.
"""
from __future__ import annotations

import json
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .io.atomic import write_atomically
from .training_runs import EpochMetrics

#: Written into every scores file, so a reader can tell a layout it does not know.
FILE_VERSION: Final[int] = 1

#: What each class counts for in a weighted average: AM-model-training's loss
#: weights for generation 1, which it also averages its metrics by.
CLASS_WEIGHTS: Final[tuple[float, ...]] = (1.0, 4.0, 3.0, 6.0, 2.0)

Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ImageScores:
    """What one checkpoint scored on each image of a list.

    Attributes:
        epoch: The epoch the checkpoint was saved after.
        images: Each image as the training export names it, relative to the collection.
        confusion: A matrix per image: ``[label][prediction]`` in pixels.
        losses: The loss on each image.
        weights: What each class counts for in a weighted average.
    """

    epoch: int
    images: tuple[str, ...]
    confusion: tuple[Matrix, ...]
    losses: tuple[float, ...]
    weights: tuple[float, ...] = CLASS_WEIGHTS


def scores_filename(epoch: int) -> str:
    """Return what the scores of the checkpoint saved after *epoch* are called."""
    return f"scores_epoch{epoch:03}.json"


def write_scores(path: Path, scores: ImageScores) -> None:
    """Write *scores* to *path* whole, never over a file.

    Raises:
        FileExistsError: If *path* exists, or appears meanwhile.
    """
    body = json.dumps({
        "version": FILE_VERSION,
        "epoch": scores.epoch,
        "weights": list(scores.weights),
        "images": list(scores.images),
        "losses": list(scores.losses),
        "confusion": [[list(row) for row in matrix] for matrix in scores.confusion],
    })
    if path.exists() or not write_atomically(
        path, lambda temporary: temporary.write_text(body, encoding="utf-8"),
        commit_if=lambda _temporary: not path.exists(),
    ):
        raise FileExistsError(f"{path} already exists; nothing was written")


def read_scores(path: Path) -> ImageScores:
    """Return the scores at *path*.

    Raises:
        ValueError: If the file is not a scores file, was written by a later
            am-seg, or does not hold a matrix and a loss for each image.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["version"] > FILE_VERSION:
            raise ValueError(f"{path} was written by a later am-seg; update to read it")
        scores = ImageScores(
            epoch=int(document["epoch"]),
            images=tuple(str(image) for image in document["images"]),
            confusion=tuple(
                tuple(tuple(int(cell) for cell in row) for row in matrix)
                for matrix in document["confusion"]
            ),
            losses=tuple(float(loss) for loss in document["losses"]),
            weights=tuple(float(weight) for weight in document["weights"]),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a file of image scores: {error!r}") from error
    if not len(scores.images) == len(scores.confusion) == len(scores.losses):
        raise ValueError(f"{path} does not hold a matrix and a loss for each image")
    return scores


def metrics_for(scores: ImageScores, images: Collection[str] | None = None) -> EpochMetrics | None:
    """Return the metrics of *scores* over *images*, as AM-model-training would report them.

    Args:
        images: The images to score, as the scores name them; every one where omitted.

    Returns:
        None where none of *images* was scored: there is nothing to report, which is not a zero.
    """
    chosen = [
        index for index, image in enumerate(scores.images) if images is None or image in images
    ]
    if not chosen:
        return None
    matrices = [scores.confusion[index] for index in chosen]
    classes = len(matrices[0])

    def counts(matrix: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
        """Return ``(hit, false alarm, miss)`` per class."""
        return [
            (
                matrix[c][c],
                sum(matrix[other][c] for other in range(classes)) - matrix[c][c],
                sum(matrix[c]) - matrix[c][c],
            )
            for c in range(classes)
        ]

    per_image = [counts(matrix) for matrix in matrices]
    summed = counts([
        [sum(matrix[row][column] for matrix in matrices) for column in range(classes)]
        for row in range(classes)
    ])

    def over_images(ratio) -> tuple[float, ...]:
        """Average *ratio* over the images a class is in or predicted in; NaN where it is in none."""
        values = []
        for c in range(classes):
            present = [ratio(*image[c]) for image in per_image if sum(image[c])]
            values.append(sum(present) / len(present) if present else math.nan)
        return tuple(values)

    def of_sum(ratio) -> tuple[float, ...]:
        return tuple(ratio(*summed[c]) for c in range(classes))

    precision = of_sum(lambda hit, alarm, miss: _ratio(hit, hit + alarm))
    recall = of_sum(lambda hit, alarm, miss: _ratio(hit, hit + miss))
    per_class = {
        "iou": over_images(lambda hit, alarm, miss: hit / (hit + alarm + miss)),
        "dice": over_images(lambda hit, alarm, miss: 2 * hit / (2 * hit + alarm + miss)),
        "f1": of_sum(lambda hit, alarm, miss: _ratio(2 * hit, 2 * hit + alarm + miss)),
        "accuracy": recall,
        "precision": precision,
        "recall": recall,
    }

    scalars = {}
    for name, values in per_class.items():
        known = [value for value in values if not math.isnan(value)]
        scalars[f"mean_{name}"] = sum(known) / len(known) if known else math.nan
        scalars[f"weighted_average_{name}"] = sum(
            value * weight for value, weight in zip(values, scores.weights) if not math.isnan(value)
        ) / sum(scores.weights)
    losses = [scores.losses[index] for index in chosen if not math.isnan(scores.losses[index])]
    return EpochMetrics(
        scores.epoch, sum(losses) / len(losses) if losses else math.nan, scalars, per_class
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Return the ratio, 0 where there is nothing to divide by, as AM-model-training has it."""
    return numerator / denominator if denominator else 0.0
