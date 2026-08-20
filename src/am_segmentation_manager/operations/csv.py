"""CSV helpers for exporting training and metadata files.

This module provides functions to export paired image and label file paths into
CSV format for machine learning model training and validation.
"""
from __future__ import annotations

import csv
import logging
import random
from collections.abc import Iterable
from pathlib import Path

from ..io.path import make_path_relative
from ..models import ImageRecord
from .label_groups import condition_sort_key
from .label_io import is_label_file_empty

logger = logging.getLogger(__name__)


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
    if train_ratio >= 1.0:
        return list(pairs), []
    if train_ratio <= 0.0:
        return [], list(pairs)

    grouped: dict[str, list[tuple[str, str]]] = {}
    for image, label in pairs:
        grouped.setdefault(label, []).append((image, label))

    groups = list(grouped.values())
    if seed is not None:
        random.Random(seed).shuffle(groups)

    target = len(pairs) * train_ratio
    train_pairs: list[tuple[str, str]] = []
    val_pairs: list[tuple[str, str]] = []
    for group in groups:
        # Whole groups only. Filling to the nearest side of the target keeps
        # the ratio as close as the grouping allows without ever splitting one.
        if len(train_pairs) + len(group) / 2 <= target:
            train_pairs.extend(group)
        else:
            val_pairs.extend(group)

    return train_pairs, val_pairs


def export_training_csv(
    records: Iterable[ImageRecord],
    start_path: Path,
    output_dir: Path,
    prefix: str = "dataset",
    train_ratio: float = 0.8,
    seed: int | None = 42,
    include_empty_labels: bool = False,
    conditions: str = "primary",
) -> tuple[Path, Path | None]:
    """Export paired image and label paths to CSV files for ML model training.

    Filters records for those with valid matching label files, calculates
    relative paths from *start_path*, and splits data into training and
    validation sets according to *train_ratio* and *seed*.

    Writes `<output_dir>/<prefix>_train.csv` and (if train_ratio < 1.0)
    `<output_dir>/<prefix>_val.csv`.

    CSV format:
        image,label
        rel/path/to/image.tif,rel/path/to/label.tif

    Args:
        records: Iterable of ImageRecord objects to process.
        start_path: Base directory used to resolve relative paths.
        output_dir: Target directory where CSV files will be written.
        prefix: Filename prefix for output CSV files (default: "dataset").
        train_ratio: Ratio of data assigned to training set (0.0 to 1.0).
        seed: Optional random seed for reproducible train/val splitting.
        include_empty_labels: Export images whose label holds nothing but
            background. Off by default: the segmentation software writes an
            empty label for every image it is pointed at, so an existing label
            file is not evidence of annotation, and exporting one pairs an
            image with a blank mask -- which actively teaches a model that the
            structure is background. Turn it on only to build negative examples
            deliberately.
        conditions: ``"primary"`` (default) exports one image per scene, or
            ``"all"`` every imaging condition of it. See
            :func:`keep_one_condition_per_scene`.

    Returns:
        A tuple of (train_csv_path, val_csv_path). val_csv_path is None if
        train_ratio >= 1.0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(records)
    if conditions == "primary":
        selected = keep_one_condition_per_scene(selected)
    elif conditions != "all":
        raise ValueError(f"Unknown conditions policy: {conditions}")

    pairs: list[tuple[str, str]] = []
    empty_count = 0
    for record in selected:
        label_path = find_matching_label(record)
        if label_path is None:
            logger.warning("Skipping image without matching label: %s", record.image_path)
            continue
        if not include_empty_labels and is_label_file_empty(label_path) is True:
            logger.warning(
                "Skipping %s: its label contains only background and is not an "
                "annotation", record.image_path,
            )
            empty_count += 1
            continue
        img_rel = make_path_relative(record.image_path, start_path)
        lbl_rel = make_path_relative(label_path, start_path)
        pairs.append((img_rel, lbl_rel))

    if empty_count:
        logger.info(
            "Excluded %d image(s) whose label holds only background; pass "
            "--include-empty-labels to export them anyway", empty_count,
        )
    if not pairs:
        logger.warning("No image/label pairs found to export")

    train_pairs, val_pairs = split_pairs(pairs, train_ratio=train_ratio, seed=seed)
    groups = {label for _, label in pairs}
    if len(groups) < len(pairs):
        logger.info(
            "Split %d pair(s) as %d label group(s) so images sharing a label stay "
            "on the same side: %d training, %d validation",
            len(pairs), len(groups), len(train_pairs), len(val_pairs),
        )

    def _write_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "label"])
            writer.writerows(rows)
        logger.info("Wrote %d pair(s) to %s", len(rows), path)
        return path

    train_path = _write_csv(output_dir / f"{prefix}_train.csv", train_pairs)
    val_path = None
    if val_pairs or train_ratio < 1.0:
        val_path = _write_csv(output_dir / f"{prefix}_val.csv", val_pairs)

    return train_path, val_path
