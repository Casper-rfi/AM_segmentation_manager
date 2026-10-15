"""Scoring a checkpoint on each image of a list, through AM-model-training.

The model, the images' preparation and the loss are AM-model-training's validation,
built by its own ``setup_evaluation``, so a score here is the score a run's own
validation would give; what differs is that each image's confusion matrix is kept
rather than added to a total. AM-model-training is imported only once a checkpoint
is scored, as :mod:`.trainer` imports the same stack.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from ..validation_scores import CLASS_WEIGHTS
from .environment import numbered_device, require_training_stack, resolve_device
from .trainer import trainable_device

logger = logging.getLogger(__name__)

#: What ``am_model_training.main.evaluate`` builds generation 1 with. Its loss
#: weights are :data:`~..validation_scores.CLASS_WEIGHTS`, which a scores file records.
MODEL_NAME = "smp_fpn"
IMAGE_SIZE = 1536
LOSS_NAME = "compound_loss"
LOSS_ALPHA = 0.5


Scored = tuple[list[list[list[int]]], list[float]]


def score_checkpoints(
    checkpoints: Iterable[Path],
    list_file: Path,
    *,
    device: str = "auto",
    workers: int | None = None,
    on_device: Callable[[str], None] | None = None,
) -> Iterator[Scored]:
    """Score each of *checkpoints* on *list_file* in turn, on one device, yielding as each is done.

    **Start it before anything in the process uses CUDA**, so the GPU it
    resolves, once for them all, can be pinned.

    Args:
        list_file: An ``image,label`` list with absolute paths.
        device: ``"auto"``, or a torch device, as :func:`~.environment.resolve_device` takes.
        workers: Processes loading images; AM-model-training's own number where omitted.
        on_device: Told the device, as ``nvidia-smi`` numbers it, before the first is scored.

    Yields:
        ``(matrices, losses)`` per checkpoint: for each row of the list in its
        order, ``[label][prediction]`` in pixels, and its loss.

    Raises:
        RuntimeError: If the training stack is not installed or does not load, or
            the images loaded are not the rows listed.
    """
    require_training_stack("Scoring a checkpoint")
    resolved = resolve_device(device)
    cpu_only, gpu_number = trainable_device(resolved)
    if on_device is not None:
        on_device("cpu" if cpu_only else numbered_device(resolved))
    for weights in checkpoints:
        yield _score(weights, list_file, cpu_only, gpu_number, workers)


def evaluation_arguments(
    weights: Path, list_file: Path, cpu_only: bool, gpu_number: int, workers: int | None
) -> dict[str, Any]:
    """Return what ``am_model_training.run.setup_evaluation`` is called with."""
    return {
        "output_path": list_file.parent,
        "weights_file": weights,
        "run_id": weights.stem,
        "model_name": MODEL_NAME,
        "csv_path": list_file,
        "image_size": IMAGE_SIZE,
        "pad": False,
        "rgb": True,
        "loss_name": LOSS_NAME,
        "cpu_only": cpu_only,
        "model_kwargs": {"pretrained": False, "weights_file": weights},
        "loss_kwargs": {"weights": CLASS_WEIGHTS, "alpha": LOSS_ALPHA},
        "include_background": True,
        "gpu_number": gpu_number,
        # One image a step: a step's confusion matrix is then an image's.
        "batch_size": 1,
        "num_workers": workers,
    }


def _score(
    weights: Path, list_file: Path, cpu_only: bool, gpu_number: int, workers: int | None
) -> Scored:
    """Score one checkpoint on a device already resolved."""

    import torch
    from am_model_training import run
    from am_model_training.utils import MONAI_KEYS
    from monai.data import decollate_batch
    from torch.amp import autocast

    objects, parameters = run.setup_evaluation(
        **evaluation_arguments(weights, list_file, cpu_only, gpu_number, workers)
    )
    classes = parameters.num_classes
    stage = objects.validation
    objects.model.eval()
    logger.info("Scoring %s on %d image(s), on %s", weights.name, parameters.total_data,
                "the CPU" if cpu_only else f"GPU {gpu_number}")

    matrices: list[list[list[int]]] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in stage.dataloader:
            with torch.device(objects.device):
                images = batch[MONAI_KEYS.IMAGE].to(objects.device)
                labels = batch[MONAI_KEYS.LABEL].to(objects.device)
                with autocast(objects.device.type):
                    outputs = stage.inferer(images, objects.model)
                    loss = objects.loss_function(outputs, labels)
                predicted = torch.stack(
                    [stage.post_transform(output) for output in decollate_batch(outputs)]
                )
                counted = torch.bincount(
                    labels.long().flatten() * classes + predicted.long().flatten(),
                    minlength=classes * classes,
                )
            matrices.append(counted.reshape(classes, classes).cpu().tolist())
            losses.append(float(loss.item()))
    if len(matrices) != parameters.total_data:
        raise RuntimeError(
            f"{len(matrices)} image(s) were scored where {list_file} lists {parameters.total_data}"
        )
    return matrices, losses
