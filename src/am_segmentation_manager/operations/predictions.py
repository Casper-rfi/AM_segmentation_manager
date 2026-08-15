"""Prediction handling utilities.

Provides a function to convert a prediction image into a label image. If
image-processing dependencies (tifffile, numpy, cv2) are available the module
will perform nearest-neighbour resizing when prediction and target image
shapes differ. Otherwise it falls back to copying the file and logs a
warning.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

from ..io.image import to_2d
from ..layout import (
    MODALITIES,
    base_path,
    resolve_labels_directory,
    state_of_path,
    state_path,
    subdirectories_by_name,
)
from ..models import ImageRecord
from .label_io import has_image_stack_support, save_label_image

logger = logging.getLogger(__name__)


def prediction_to_label(prediction_path: Path, label_path: Path, target_image_path: Path | None = None) -> None:
    """Convert a prediction file into a label file.

    Where *target_image_path* is given and its dimensions differ from the
    prediction's, the prediction is resized to match by nearest neighbour,
    which preserves the discrete label values. Without the optional image
    stack the prediction is copied unchanged and a warning is logged.

    Args:
        prediction_path: Path to the prediction image file.
        label_path: Destination path for the label image.
        target_image_path: Optional path to the SEM/FIB image whose shape should
            be matched.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

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


def convert_record_predictions(records: Iterable[ImageRecord], prediction_kind: str = "clean") -> int:
    """Convert predictions for explicitly selected image records.

    Existing labels, including corrected labels, are never overwritten.  The
    generated label follows the image's selected/deselected/corrected folder.
    """
    converted = 0
    for record in records:
        if record.modality not in MODALITIES:
            continue
        matching_predictions = [
            prediction
            for prediction in record.predictions
            if not prediction_kind
            or any(parent.name.lower() == prediction_kind.lower() for parent in prediction.parents)
        ]
        if not matching_predictions:
            continue
        if record.labels:
            logger.warning("Skipping %s: a label already exists", record.image_path)
            continue

        image_directory = base_path(record.image_path).parent
        labels_directory, _, _ = resolve_labels_directory(
            image_directory, record.modality, subdirectories_by_name(image_directory.parent)
        )
        # The generated label follows the image's own workflow state.
        destination = state_path(
            labels_directory / record.image_path.name, state_of_path(record.image_path)
        )
        if destination.exists():
            logger.warning("Skipping %s: refusing to overwrite %s", record.image_path, destination)
            continue
        prediction_to_label(matching_predictions[0], destination, target_image_path=record.image_path)
        converted += 1
    return converted
