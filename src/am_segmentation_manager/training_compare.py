"""Training runs side by side: one value per run and class, from each run's validation.

A table of what :mod:`.training_runs` read, and nothing it did not: a row per run, a
group of columns per class, a column per validation set in each group. A value is the
metric at one epoch -- the run's best, or the one marked as best -- or its mean or
median over a range of epochs. **A cell nothing was validated for is None, never 0.**
No file is read and nothing is drawn here; :func:`~.plots.comparison_figure` draws it.
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .training_runs import PER_CLASS_METRICS, EpochMetrics, TrainingRun

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

#: What a run's own validation is called as a validation set.
OWN_VALIDATION: Final[str] = "validation"


@dataclass(frozen=True)
class ComparedRun:
    """A run to compare: what to call it, what it wrote, and the epoch marked as its best."""

    name: str
    run: TrainingRun
    marked: int | None = None


@dataclass(frozen=True)
class ComparisonRow:
    """One run's row.

    Attributes:
        name: What the run is called.
        epochs: Which epochs its values are from, in words: ``"74"``,
            ``"74 (marked)"``, ``"60-90 (31)"``, or ``""`` where there are none.
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


def compare_runs(
    runs: Sequence[ComparedRun],
    *,
    metric: str = "iou",
    classes: Sequence[str] | None = None,
    value: str = "best",
    epoch_range: tuple[int, int] = DEFAULT_EPOCH_RANGE,
) -> Comparison:
    """Return *runs* side by side, *metric* per class from each run's validation.

    Args:
        classes: The classes to give columns to, in this order; every class of
            the runs where omitted, in the order they first appear.
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
    columns = tuple((name, OWN_VALIDATION) for name in classes)

    rows = []
    for compared in runs:
        epochs, said = _epochs_for(compared, value, first, last)
        names = compared.run.label_names
        values = []
        for name, _ in columns:
            scores = [
                epoch.per_class[metric][names.index(name)]
                for epoch in epochs
                if name in names and len(epoch.per_class.get(metric, ())) > names.index(name)
            ]
            values.append(_summed(scores, value))
        rows.append(ComparisonRow(compared.name, said, tuple(values)))
    return Comparison(metric, value, columns, tuple(rows))


def _epochs_for(
    compared: ComparedRun, value: str, first: int, last: int
) -> tuple[list[EpochMetrics], str]:
    """Return the validated epochs a run's values are from, and which those are in words."""
    run = compared.run
    if value in RANGED_VALUES:
        inside = [epoch for epoch in run.val if first <= epoch.epoch <= last]
        return inside, f"{first}-{last} ({len(inside)})" if inside else ""
    best = run.best()
    if value == "marked" and compared.marked is not None:
        marked = [epoch for epoch in run.val if epoch.epoch == compared.marked]
        return marked, f"{compared.marked} (marked)" if marked else ""
    return ([best], str(best.epoch)) if best is not None else ([], "")


def _summed(scores: Sequence[float], value: str) -> float | None:
    """Return the one number a cell holds for *scores*, or None where there are none."""
    if not scores:
        return None
    if value == "median":
        return statistics.median(scores)
    return statistics.fmean(scores)
