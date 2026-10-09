"""Training runs side by side: one value per run and class, from each run's validation.

A table of what :mod:`.training_runs` read, and nothing it did not: a row per run, a
group of columns per class, a column per validation set in each group. A value is the
metric at one epoch -- the run's best, or the one marked as best -- or its mean or
median over a range of epochs. **A cell nothing was validated for is None, never 0.**
No file is read and nothing is drawn here; :func:`~.plots.comparison_figure` draws it.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from .training_runs import PER_CLASS_METRICS, EpochMetrics, TrainingRun
from .validation_scores import ImageScores, metrics_for

#: What a cell can hold, and what each is called.
VALUE_CAPTIONS: Final[dict[str, str]] = {
    "best": "Run's best epoch",
    "marked": "Marked best epoch",
    "mean": "Mean over epochs",
    "median": "Median over epochs",
}

#: The values taken over a range of epochs rather than at one.
RANGED_VALUES: Final[frozenset[str]] = frozenset({"mean", "median"})

#: The epochs a mean or median is taken over where nothing says: past the
#: encoder's unfreezing and its settling, in a new model's 100.
DEFAULT_EPOCH_RANGE: Final[tuple[int, int]] = (60, 90)

#: The values the table's colours run between where nothing says: all a metric can take.
DEFAULT_COLOUR_RANGE: Final[tuple[float, float]] = (0.0, 1.0)

#: What a colour range is rounded to, and the least it may span.
COLOUR_STEP: Final[float] = 0.01

#: What a run's own validation is called as a validation set.
OWN_VALIDATION: Final[str] = "validation"

#: Asks for every validation of a run beside its own, each a set.
EVERY_SET: Final[str] = "*"

#: The most values a column of the images file may hold to divide a set by: more
#: is a name or a measurement, not a kind.
MOST_PARTS: Final[int] = 12

#: Columns of the images file that say where an image is, not what it is of.
NOT_KINDS: Final[frozenset[str]] = frozenset({"image", "label"})


@dataclass(frozen=True)
class ScoredValidation:
    """A validation of a run as it has been scored so far.

    Attributes:
        name: What it is called.
        scores: What each epoch scored on each image, by epoch.
        images: Each image scored, as the export described it: ``image`` and its metadata.
    """

    name: str
    scores: Sequence[ImageScores] = ()
    images: Sequence[Mapping[str, str]] = ()


def dividing_columns(validations: Sequence[ScoredValidation]) -> tuple[str, ...]:
    """Return the metadata columns that divide the images of *validations* into a few kinds."""
    values: dict[str, set[str]] = {}
    for validation in validations:
        for row in validation.images:
            for column, value in row.items():
                if column not in NOT_KINDS and value:
                    values.setdefault(column, set()).add(value)
    return tuple(column for column, found in values.items() if 1 < len(found) <= MOST_PARTS)


def validation_sets(
    validations: Sequence[ScoredValidation], validated_on: str = "", split_by: str = ""
) -> dict[str, tuple[EpochMetrics, ...]]:
    """Return a run's validation curves beyond its own, by what each set is called.

    Args:
        validated_on: A validation's name, :data:`EVERY_SET` for each of them
            whole, or "" for none.
        split_by: A metadata column to divide the one validation named by; each
            value it holds is then a set, called by that value.
    """
    if not validated_on:
        return {}
    if validated_on == EVERY_SET:
        return {validation.name: _curve(validation, None) for validation in validations}
    named = next((v for v in validations if v.name == validated_on), None)
    if named is None:
        return {}
    if not split_by:
        return {named.name: _curve(named, None)}
    parts: dict[str, set[str]] = {}
    for row in named.images:
        if row.get(split_by):
            parts.setdefault(row[split_by], set()).add(row["image"])
    return {value: _curve(named, images) for value, images in sorted(parts.items())}


def _curve(validation: ScoredValidation, images: set[str] | None) -> tuple[EpochMetrics, ...]:
    """Return the metrics of each epoch scored, over *images* or over every image."""
    return tuple(
        metrics
        for scores in validation.scores
        if (metrics := metrics_for(scores, images)) is not None
    )


@dataclass(frozen=True)
class ComparedRun:
    """A run to compare.

    Attributes:
        name: What to call it.
        run: What it wrote.
        marked: The epoch marked as its best, or None.
        sets: Its validation curves beyond its own, by set, as
            :func:`validation_sets` returns them.
    """

    name: str
    run: TrainingRun
    marked: int | None = None
    sets: Mapping[str, Sequence[EpochMetrics]] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonRow:
    """One run's row.

    Attributes:
        name: What the run is called.
        epochs: Which epochs its values are from, in words: ``"74"``,
            ``"74 (marked)"``, ``"60-90 (31)"`` with how many of those its own
            validation holds, or ``""`` where it has validated none.
        values: A value per column of the table, None where nothing was validated.
    """

    name: str
    epochs: str
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class Comparison:
    """The table: its columns, and a row per run.

    Attributes:
        metric: The per-class metric every cell holds.
        value: Which of :data:`VALUE_CAPTIONS` a cell is.
        columns: Each column as ``(class, validation set)``, a class's together.
        rows: A row per run, in the order given.
    """

    metric: str
    value: str
    columns: tuple[tuple[str, str], ...]
    rows: tuple[ComparisonRow, ...]

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the classes, in column order."""
        return tuple(dict.fromkeys(name for name, _ in self.columns))

    @property
    def sets(self) -> tuple[str, ...]:
        """Return the validation sets, in column order."""
        return tuple(dict.fromkeys(name for _, name in self.columns))

    def header(self) -> tuple[str, ...]:
        """Return the table's header as a CSV has it."""
        several = len(self.sets) > 1
        return (
            "run", "epochs",
            *(f"{name} {where} {self.metric}" if several else f"{name} {self.metric}"
              for name, where in self.columns),
        )

    def cells(self) -> list[tuple[object, ...]]:
        """Return each row's cells beneath :meth:`header`, a missing value blank."""
        return [
            (row.name, row.epochs, *("" if value is None else f"{value:.4f}" for value in row.values))
            for row in self.rows
        ]


def colour_range(low: float, high: float) -> tuple[float, float]:
    """Return *low* and *high* as a range colours can run over: in order, within 0 to 1, and not empty."""
    low, high = sorted((min(max(low, 0.0), 1.0), min(max(high, 0.0), 1.0)))
    if high - low < COLOUR_STEP:
        high = min(low + COLOUR_STEP, 1.0)
        low = high - COLOUR_STEP
    return round(low, 6), round(high, 6)


def fitted_colour_range(table: Comparison) -> tuple[float, float]:
    """Return the range the values of *table* span, rounded outward to :data:`COLOUR_STEP`.

    :data:`DEFAULT_COLOUR_RANGE` where it holds no value.
    """
    values = [value for row in table.rows for value in row.values if value is not None]
    if not values:
        return DEFAULT_COLOUR_RANGE
    # Rounded first, or 0.29 * 100 floors to 28.
    low = math.floor(round(min(values) / COLOUR_STEP, 6)) * COLOUR_STEP
    high = math.ceil(round(max(values) / COLOUR_STEP, 6)) * COLOUR_STEP
    return colour_range(low, high)


def compare_runs(
    runs: Sequence[ComparedRun],
    *,
    metric: str = "iou",
    classes: Sequence[str] | None = None,
    sets: Sequence[str] = (OWN_VALIDATION,),
    value: str = "best",
    epoch_range: tuple[int, int] = DEFAULT_EPOCH_RANGE,
) -> Comparison:
    """Return *runs* side by side, *metric* per class on each validation set.

    **The best and the marked epoch are the run's**, chosen on its own
    validation, whichever set a column is of: the table asks how the model
    chosen does elsewhere, not which model each set would have chosen.

    Args:
        classes: The classes to give columns to, in this order; every class of
            the runs where omitted, in the order they first appear.
        sets: The validation sets to give a column to under each class:
            :data:`OWN_VALIDATION` and names from the runs' ``sets``.
        value: One of :data:`VALUE_CAPTIONS`. ``"marked"`` takes the run's own
            best where no epoch is marked, and the row says which it took.
        epoch_range: The first and last epoch a mean or median is taken over.

    Raises:
        ValueError: If *metric* or *value* is unknown, or the range is backwards.
    """
    if metric not in PER_CLASS_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {', '.join(PER_CLASS_METRICS)}")
    if value not in VALUE_CAPTIONS:
        raise ValueError(f"Unknown value {value!r}; choose from {', '.join(VALUE_CAPTIONS)}")
    first, last = epoch_range
    if value in RANGED_VALUES and not 1 <= first <= last:
        raise ValueError(f"Epochs {first} to {last} is not a range of epochs")

    if classes is None:
        classes = tuple(dict.fromkeys(name for compared in runs for name in compared.run.label_names))
    columns = tuple((name, where) for name in classes for where in sets)

    rows = []
    for compared in runs:
        wanted, said = _epochs_for(compared, value, first, last)
        names = compared.run.label_names
        values = []
        for name, where in columns:
            curve = compared.run.val if where == OWN_VALIDATION else compared.sets.get(where, ())
            scores = [
                epoch.per_class[metric][names.index(name)]
                for epoch in curve
                if epoch.epoch in wanted
                and name in names and len(epoch.per_class.get(metric, ())) > names.index(name)
            ]
            values.append(_summed(scores, value))
        rows.append(ComparisonRow(compared.name, said, tuple(values)))
    return Comparison(metric, value, columns, tuple(rows))


def _epochs_for(
    compared: ComparedRun, value: str, first: int, last: int
) -> tuple[frozenset[int], str]:
    """Return the epochs a run's values are from, and which those are in words."""
    run = compared.run
    if value in RANGED_VALUES:
        inside = frozenset(range(first, last + 1))
        scored = len([epoch for epoch in run.val if epoch.epoch in inside])
        return inside, f"{first}-{last} ({scored})"
    best = run.best()
    if value == "marked" and compared.marked is not None:
        return frozenset({compared.marked}), f"{compared.marked} (marked)"
    return (frozenset({best.epoch}), str(best.epoch)) if best is not None else (frozenset(), "")


def _summed(scores: Sequence[float], value: str) -> float | None:
    """Return the one number a cell holds for *scores*, or None where there are none.

    A class no image of a set holds scores NaN there, which is no score.
    """
    scores = [score for score in scores if not math.isnan(score)]
    if not scores:
        return None
    if value == "median":
        return statistics.median(scores)
    return statistics.fmean(scores)
