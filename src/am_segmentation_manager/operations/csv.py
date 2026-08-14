"""CSV helpers for exporting training and metadata files.

This module provides functions to export paired image and label file paths into
CSV format for machine learning model training and validation. It also includes
a helper to convert ImageRecord objects into a pandas DataFrame.
"""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Iterable

from ..io.path import make_path_relative
from ..layout import state_of_path
from ..models import ImageRecord

logger = logging.getLogger(__name__)


def records_to_dataframe(records: Iterable[ImageRecord]):
    """Convert ImageRecord objects into a pandas.DataFrame.

    Raises RuntimeError if pandas is not available.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("pandas is required to create DataFrames: install pandas") from exc

    rows = []
    for r in records:
        rows.append({
            "image_path": str(r.image_path),
            "project": r.project,
            "lamella": r.lamella,
            "stage": r.stage,
            "modality": r.modality,
            "selected": bool(r.selected),
            "corrected": bool(r.corrected),
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        })

    df = pd.DataFrame(rows)
    logger.debug("Created DataFrame with %d rows", len(df))
    return df


def find_matching_label(record: ImageRecord) -> Path | None:
    """Find the matching label file path for an ImageRecord.

    Prefers the label whose workflow state matches the record's own, so an
    export reflects the state the record is actually in. Falls back to any
    label that exists. Returns None if none do.
    """
    if not record.labels:
        return None

    wanted = "corrected" if record.corrected else "selected" if record.selected else "deselected"
    for label in record.labels:
        if state_of_path(label) == wanted and label.exists():
            return label

    for label in record.labels:
        if label.exists():
            return label

    return None


def export_training_csv(
    records: Iterable[ImageRecord],
    start_path: Path,
    output_dir: Path,
    prefix: str = "dataset",
    train_ratio: float = 0.8,
    seed: int | None = 42,
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

    Returns:
        A tuple of (train_csv_path, val_csv_path). val_csv_path is None if
        train_ratio >= 1.0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    for record in records:
        label_path = find_matching_label(record)
        if label_path is None:
            logger.warning("Skipping image without matching label: %s", record.image_path)
            continue
        img_rel = make_path_relative(record.image_path, start_path)
        lbl_rel = make_path_relative(label_path, start_path)
        pairs.append((img_rel, lbl_rel))

    if not pairs:
        logger.warning("No image/label pairs found to export")

    # Reproducibly shuffle pairs
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    # Calculate split point
    if train_ratio >= 1.0:
        train_pairs = pairs
        val_pairs = []
    elif train_ratio <= 0.0:
        train_pairs = []
        val_pairs = pairs
    else:
        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

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
