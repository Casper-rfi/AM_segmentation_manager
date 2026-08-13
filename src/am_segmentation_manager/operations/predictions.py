"""Prediction handling utilities.

Provides a function to convert a prediction image into a label image. If
image-processing dependencies (tifffile, numpy, cv2) are available the module
will perform nearest-neighbour resizing when prediction and target image
shapes differ. Otherwise it falls back to copying the file and logs a
warning.
"""
from __future__ import annotations

from pathlib import Path
import logging
import shutil
from typing import Iterable

from .label_io import has_image_stack_support, save_label_image
from ..io.image import to_2d
from ..models import ImageRecord

logger = logging.getLogger(__name__)


def prediction_to_label(prediction_path: Path, label_path: Path, target_image_path: Path | None = None) -> None:
    """Convert a prediction file into a label file.

    If `target_image_path` is provided and the prediction dimensions differ
    from the target, attempt to resize the prediction to match the target
    using nearest-neighbour interpolation (preserving label values).

    Behaviour:
    - If tifffile, numpy and cv2 are available: read prediction and target
      images and resize if necessary, then write label using tifffile.
    - Otherwise: fall back to copying the prediction file to the label path
      and log a warning.

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
    import tifffile  # type: ignore
    import numpy as np  # type: ignore
    import cv2  # type: ignore

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


def convert_stage_predictions(stage_path: Path, modality: str = "SEM", prediction_kind: str = "clean") -> int:
    """Convert all predictions for a given stage into labels.

    This convenience function locates prediction files for the requested
    modality and kind (clean/raw) and writes corresponding label files under
    the labels/ or labels_fib/ folder. Matching is performed by filename:
    for each prediction file named X the function looks for an image named X
    inside the stage's sem/ (or fib/) or sem/deselected/ (or fib/deselected/)
    folders and uses that image as the target for resizing. If no matching
    image is found the prediction is still converted but without a target
    (no resizing).

    Args:
        stage_path: Path to the stage folder (the folder that contains sem/ and fib/)
        modality: "SEM" or "FIB"
        prediction_kind: typically "clean" or "raw"; the function will look
            under predictions/ (or predictions_fib/) and its subfolders.

    Returns:
        The number of prediction files converted into labels.
    """
    # determine prediction and label base folders
    if modality.upper() == "SEM":
        preds_base = stage_path / "predictions"
        labels_base = stage_path / "labels"
        img_dirs = [stage_path / "sem", stage_path / "sem" / "deselected"]
    else:
        preds_base = stage_path / "predictions_fib"
        labels_base = stage_path / "labels_fib"
        img_dirs = [stage_path / "fib", stage_path / "fib" / "deselected"]

    # consider variants: predictions/<kind>/, predictions/<kind> may be nested
    search_dirs = []
    if preds_base.exists():
        search_dirs.append(preds_base)
    kind_dir = preds_base / prediction_kind
    if kind_dir.exists():
        search_dirs.append(kind_dir)

    # also include nested raw/clean under preds_base
    for sub in preds_base.glob("**/*"):
        if sub.is_dir() and sub.name.lower() == prediction_kind.lower():
            search_dirs.append(sub)

    # collect prediction files (deduplicate directories and files)
    image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    # deduplicate search dirs
    unique_dirs = []
    seen = set()
    for d in search_dirs:
        try:
            key = str(d.resolve())
        except Exception:
            key = str(d)
        if key not in seen:
            seen.add(key)
            unique_dirs.append(d)

    preds_set = []
    preds_seen = set()
    for d in unique_dirs:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in image_exts:
                k = str(p.resolve()) if p.exists() else str(p)
                if k not in preds_seen:
                    preds_seen.add(k)
                    preds_set.append(p)

    converted = 0
    for pred in preds_set:
        name = pred.name
        # try find matching image
        target = None
        for img_dir in img_dirs:
            candidate = img_dir / name
            if candidate.exists():
                target = candidate
                break
        # fallback: look for matching file anywhere under stage_path/sem or fib
        if target is None:
            for img_dir in img_dirs:
                for p in img_dir.rglob("*") if img_dir.exists() else []:
                    if p.is_file() and p.name == name:
                        target = p
                        break
                if target is not None:
                    break

        # destination label path
        dest = labels_base / name
        try:
            prediction_to_label(pred, dest, target_image_path=target)
            converted += 1
        except Exception as exc:
            logger.exception("Failed to convert prediction %s: %s", pred, exc)

    logger.info("Converted %d prediction(s) in %s for modality %s", converted, stage_path, modality)
    return converted


def convert_record_predictions(records: Iterable[ImageRecord], prediction_kind: str = "clean") -> int:
    """Convert predictions for explicitly selected image records.

    Existing labels, including corrected labels, are never overwritten.  The
    generated label follows the image's selected/deselected/corrected folder.
    """
    converted = 0
    for record in records:
        if record.modality not in {"SEM", "FIB"}:
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

        image_directory = record.image_path.parent
        state_directory = image_directory.name if image_directory.name in {"deselected", "corrected"} else None
        modality_directory = image_directory.parent if state_directory else image_directory
        labels_directory = modality_directory.parent / ("labels" if record.modality == "SEM" else "labels_fib")
        destination_directory = labels_directory / state_directory if state_directory else labels_directory
        destination = destination_directory / record.image_path.name
        if destination.exists():
            logger.warning("Skipping %s: refusing to overwrite %s", record.image_path, destination)
            continue
        prediction_to_label(matching_predictions[0], destination, target_image_path=record.image_path)
        converted += 1
    return converted
