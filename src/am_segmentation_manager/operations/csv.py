"""Writing the image and label lists a training run reads.

One row per acquisition rather than per file: both beams of one scene go side by
side (``sem_image, sem_label, fib_image, fib_label``), a single-beam selection
writes ``image, label``, and a cell is blank where that beam has no image or no
label worth pairing with one.

Two rules nothing may get past: an empty label is never written as a training
pair, and a split never cuts a group -- the label in a single-beam export, the
acquisition in a paired one. ``docs/decisions.md`` § Counting and export says why.
"""
from __future__ import annotations

import csv
import logging
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..frames import group_by_acquisition
from ..io.path import make_path_relative
from ..models import ImageRecord
from .label_groups import condition_sort_key
from .label_io import is_label_file_empty

logger = logging.getLogger(__name__)

#: Header of a training CSV covering one beam.
SINGLE_COLUMNS = ("image", "label")

#: Header of a training CSV covering both. One row is one acquisition under one
#: pair of imaging conditions, so a model can be trained on either beam from
#: one file, or on the two together. A cell is blank where that beam has no
#: image, or no label worth pairing with one.
PAIRED_COLUMNS = ("sem_image", "sem_label", "fib_image", "fib_label")


@dataclass(frozen=True)
class TrainingRow:
    """One row of a training CSV, and what it must not be split away from.

    Attributes:
        values: The cells, in the order of the header they are written under.
        group: What keeps related rows on one side of the train/validation
            split. Splitting near-identical images across the two would have
            the validation score measure how well the model recognises images
            it has effectively already seen.
    """

    values: tuple[str, ...]
    group: str


def find_matching_label(record: ImageRecord) -> Path | None:
    """Return the record's label, or None if it has none on disk.

    A group has one label in one place, so there is nothing to choose between.
    """
    for label in record.labels:
        if label.exists():
            return label
    return None


def keep_one_condition_per_scene(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    """Return one record per scene, dropping the other imaging conditions.

    Records sharing a label are the same scene captured under different imaging
    conditions, so exporting all of them multiplies a dataset by near-identical
    copies. This keeps each scene's primary condition, chosen by
    :func:`~am_segmentation_manager.operations.label_groups.primary_record`.

    Records with no label are kept as they are, since nothing identifies them
    as sharing a scene with anything else.
    """
    candidates = list(records)
    primary: dict[str, ImageRecord] = {}
    for record in candidates:
        if not record.labels:
            continue
        key = str(record.labels[0])
        current = primary.get(key)
        if current is None or condition_sort_key(record) < condition_sort_key(current):
            primary[key] = record

    chosen = {id(record) for record in primary.values()}
    kept = [
        record for record in candidates
        if not record.labels or id(record) in chosen
    ]
    if len(kept) < len(candidates):
        logger.info(
            "Exporting one condition per scene: %d of %d image(s); pass "
            "--conditions all to include every condition",
            len(kept), len(candidates),
        )
    return kept


def _image_cell(record: ImageRecord | None, start_path: Path) -> str:
    """Return an image's recorded path, or blank where that beam has none."""
    return "" if record is None else make_path_relative(record.image_path, start_path)


def _label_cell(
    record: ImageRecord | None,
    start_path: Path,
    include_empty_labels: bool,
    is_empty: Callable[[Path], bool | None],
) -> str:
    """Return the path of a label worth pairing with an image, or blank.

    Blank covers both "no label" and "a label holding nothing but background",
    which are different facts about the archive but the same instruction to a
    training run: there is nothing here to learn from. Which of the two it was
    is the preview's to report, not the file's.
    """
    if record is None:
        return ""
    label = find_matching_label(record)
    if label is None:
        return ""
    if not include_empty_labels and is_empty(label) is True:
        return ""
    return make_path_relative(label, start_path)


def build_training_rows(
    records: Iterable[ImageRecord],
    start_path: Path,
    *,
    paired: bool,
    include_empty_labels: bool = False,
    is_empty: Callable[[Path], bool | None] = is_label_file_empty,
) -> list[TrainingRow]:
    """Return the rows a training export would write.

    One function for the preview and for the file, because a count on screen
    that disagreed with what was written would make the preview worse than
    useless.

    Paired, a row is one acquisition under one pair of imaging conditions --
    the cross product, so five SEM conditions against one FIB image are five
    rows repeating the FIB path. That is what "one image per label" is for
    where it is not wanted. A row survives only if at least one beam has a
    label worth pairing with: a row of two images and no annotation teaches
    nothing.

    Unpaired, a row is one image and its label, exactly as before.

    Args:
        records: The images to write, already filtered.
        start_path: Base directory paths are written relative to.
        paired: Write both beams per row, under :data:`PAIRED_COLUMNS`.
        include_empty_labels: Pair an image with a label holding only
            background. Off by default; see :func:`export_training_csv`.
        is_empty: Answers whether a label holds nothing but background. Passed
            in so a caller holding the answers already -- the export view, which
            reads every label once -- does not open them all again.
    """
    records = list(records)
    if not paired:
        rows = []
        for record in records:
            label = _label_cell(record, start_path, include_empty_labels, is_empty)
            if not label:
                continue
            rows.append(
                TrainingRow((_image_cell(record, start_path), label), group=label)
            )
        return rows

    rows = []
    for acquisition in group_by_acquisition(records):
        # A beam with no image still yields one blank column rather than no
        # row, so an acquisition captured on one beam is exported rather than
        # dropped for what it does not have.
        for sem in acquisition.sem or (None,):
            for fib in acquisition.fib or (None,):
                values = (
                    _image_cell(sem, start_path),
                    _label_cell(sem, start_path, include_empty_labels, is_empty),
                    _image_cell(fib, start_path),
                    _label_cell(fib, start_path, include_empty_labels, is_empty),
                )
                if values[1] or values[3]:
                    rows.append(TrainingRow(values, group=acquisition.identity))
    return rows


def split_grouped(
    items: Sequence,
    key: Callable[[object], str],
    train_ratio: float = 0.8,
    seed: int | None = 42,
) -> tuple[list, list]:
    """Split *items* into training and validation, keeping a group together.

    The general form of :func:`split_pairs`: what a group *is* differs between
    a single-beam export, where it is the label, and a paired one, where it is
    the acquisition -- but the arithmetic of filling to the nearest side of the
    target without ever cutting a group is the same.

    Args:
        items: What to split.
        key: Returns the group an item belongs to.
        train_ratio: Share of items assigned to training.
        seed: Seed for a reproducible shuffle, or None to leave the order.

    Returns:
        ``(training, validation)``.
    """
    if train_ratio >= 1.0:
        return list(items), []
    if train_ratio <= 0.0:
        return [], list(items)

    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)

    groups = list(grouped.values())
    if seed is not None:
        random.Random(seed).shuffle(groups)

    target = len(items) * train_ratio
    training: list = []
    validation: list = []
    for group in groups:
        # Whole groups only. Filling to the nearest side of the target keeps
        # the ratio as close as the grouping allows without ever splitting one.
        if len(training) + len(group) / 2 <= target:
            training.extend(group)
        else:
            validation.extend(group)
    return training, validation


def split_rows(
    rows: Sequence[TrainingRow],
    train_ratio: float = 0.8,
    seed: int | None = 42,
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Split built rows, keeping each row's group whole."""
    return split_grouped(rows, lambda row: row.group, train_ratio, seed)


def split_pairs(
    pairs: list[tuple[str, str]],
    train_ratio: float = 0.8,
    seed: int | None = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split image/label pairs into training and validation sets.

    Pairs are grouped by **label** before splitting, so every image sharing a
    label lands on the same side. Several imaging conditions of one scene share
    one label; splitting them across the two sets would put near-identical
    images in both, and the validation score would then measure how well the
    model recognises images it has effectively already seen.

    With one image per label -- every project that has no imaging-condition
    variants -- each group holds a single pair and this behaves exactly as a
    plain shuffle-and-split.

    Reports nothing: the export view runs this on every filter change, and a log
    line per keystroke saying how a split came out reads as an export happening.
    :func:`export_training_csv` reports it, because it is the one that writes.

    Args:
        pairs: ``(image, label)`` relative paths.
        train_ratio: Share of pairs assigned to training.
        seed: Seed for a reproducible shuffle, or None to leave the order.

    Returns:
        ``(train_pairs, val_pairs)``.
    """
    return split_grouped(pairs, lambda pair: pair[1], train_ratio, seed)


def export_training_csv(
    records: Iterable[ImageRecord],
    start_path: Path,
    output_dir: Path,
    prefix: str = "dataset",
    train_ratio: float = 0.8,
    seed: int | None = 42,
    include_empty_labels: bool = False,
    conditions: str = "primary",
    paired: bool = False,
    is_empty: Callable[[Path], bool | None] = is_label_file_empty,
) -> tuple[Path, Path | None]:
    """Export image and label paths to CSV files for ML model training.

    Writes ``<output_dir>/<prefix>_train.csv`` and, unless *train_ratio* is
    1.0, ``<output_dir>/<prefix>_val.csv``.

    Two shapes, decided by *paired*::

        image,label
        rel/path/to/image.tif,rel/path/to/label.tif

        sem_image,sem_label,fib_image,fib_label
        rel/sem.tif,rel/sem_label.tif,rel/fib.tif,rel/fib_label.tif

    Args:
        records: ImageRecords to process, already filtered.
        start_path: Base directory used to resolve relative paths.
        output_dir: Target directory where the CSV files are written.
        prefix: Filename prefix for the output files.
        train_ratio: Ratio of rows assigned to training (0.0 to 1.0).
        seed: Optional seed for a reproducible split.
        include_empty_labels: Pair an image with a label holding nothing but
            background. Off by default: the segmentation software writes an
            empty label for every image it is pointed at, so an existing label
            file is not evidence of annotation, and exporting one actively
            teaches a model that the structure is background. Turn it on only
            to build negative examples deliberately.
        conditions: ``"primary"`` (default) exports one image per scene, or
            ``"all"`` every imaging condition of it. See
            :func:`keep_one_condition_per_scene`.
        paired: Write both beams on one row. What a selection holding both
            modalities should produce -- a file mixing SEM and FIB rows under
            one ``image`` column is not something anyone trains on.
        is_empty: Answers whether a label holds only background; see
            :func:`build_training_rows`.

    Returns:
        ``(train_csv_path, val_csv_path)``. The second is None when
        *train_ratio* is 1.0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(records)
    if conditions == "primary":
        selected = keep_one_condition_per_scene(selected)
    elif conditions != "all":
        raise ValueError(f"Unknown conditions policy: {conditions}")

    rows = build_training_rows(
        selected,
        start_path,
        paired=paired,
        include_empty_labels=include_empty_labels,
        is_empty=is_empty,
    )
    if not rows:
        logger.warning("No image/label pairs found to export")
    if not include_empty_labels:
        # Reported either way, because it is the same rule with two shapes: a
        # blank mask is never paired with an image, which drops the row where
        # there is one beam and empties a cell where there are two. Silence
        # would be a training set quietly short of what somebody labelled.
        if paired:
            blank = sum(
                1 for row in rows for cell in (row.values[1], row.values[3]) if not cell
            )
            if blank:
                logger.info(
                    "Left %d label cell(s) blank: no label, or one holding only "
                    "background; pass --include-empty-labels to pair the latter "
                    "anyway", blank,
                )
        elif (dropped := len(selected) - len(rows)) > 0:
            logger.info(
                "Excluded %d image(s) with no label, or whose label holds only "
                "background; pass --include-empty-labels to export the latter",
                dropped,
            )

    train_rows, val_rows = split_rows(rows, train_ratio=train_ratio, seed=seed)
    groups = {row.group for row in rows}
    if len(groups) < len(rows):
        logger.info(
            "Split %d row(s) as %d group(s) so images of one scene stay on the "
            "same side: %d training, %d validation",
            len(rows), len(groups), len(train_rows), len(val_rows),
        )

    columns = PAIRED_COLUMNS if paired else SINGLE_COLUMNS

    def _write_csv(path: Path, written: list[TrainingRow]) -> Path:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(row.values for row in written)
        logger.info("Wrote %d row(s) to %s", len(written), path)
        return path

    train_path = _write_csv(output_dir / f"{prefix}_train.csv", train_rows)
    val_path = None
    if val_rows or train_ratio < 1.0:
        val_path = _write_csv(output_dir / f"{prefix}_val.csv", val_rows)

    return train_path, val_path
