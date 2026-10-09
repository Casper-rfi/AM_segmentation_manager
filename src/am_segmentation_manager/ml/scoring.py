"""Scoring a checkpoint on each image of a list, through AM-model-training.

The model, the images' preparation and the loss are AM-model-training's validation,
built by its own ``setup_evaluation``, so a score here is the score a run's own
validation would give; what differs is that each image's confusion matrix is kept
rather than added to a total. Imported only inside :func:`score_checkpoint`, as
:mod:`.trainer` imports the same stack.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .environment import require_training_stack, resolve_device
from .trainer import trainable_device

logger = logging.getLogger(__name__)

#: What ``am_model_training.main.evaluate`` builds generation 1 with.
MODEL_NAME = "smp_fpn"
IMAGE_SIZE = 1536
LOSS_NAME = "compound_loss"
LOSS_WEIGHTS = (1.0, 4.0, 3.0, 6.0, 2.0)
LOSS_ALPHA = 0.5


def score_checkpoint(
    weights: Path, list_file: Path, *, device: str = "auto", workers: int | None = None
) -> tuple[list[list[list[int]]], list[float]]:
    """Return a confusion matrix and a loss for each row of *list_file*, in its order.

    Blocks until every image is scored. **Call it before anything in the
    process uses CUDA**, so the GPU it resolves can be pinned.

    Args:
        weights: The checkpoint to score.
        list_file: An ``image,label`` list with absolute paths.
        device: ``"auto"``, or a torch device, as :func:`~.environment.resolve_device` takes.
        workers: Processes loading images; AM-model-training's own number where omitted.

    Returns:
        ``(matrices, losses)``: per image, ``[label][prediction]`` in pixels, and its loss.

    Raises:
        RuntimeError: If the training stack is not installed or does not load, or
            the images loaded are not the rows listed.
    """
    require_training_stack("Scoring a checkpoint")
    resolved = resolve_device(device)
    cpu_only, gpu_number = trainable_device(resolved)

    import torch
    from am_model_training import run
    from am_model_training.utils import MONAI_KEYS
    from monai.data import decollate_batch
    from torch.amp import autocast

    objects, parameters = run.setup_evaluation(
        output_path=list_file.parent,
        weights_file=weights,
        run_id=weights.stem,
        model_name=MODEL_NAME,
        csv_path=list_file,
        image_size=IMAGE_SIZE,
        pad=False,
        rgb=True,
        loss_name=LOSS_NAME,
        cpu_only=cpu_only,
        model_kwargs={"pretrained": False, "weights_file": weights},
        loss_kwargs={"weights": LOSS_WEIGHTS, "alpha": LOSS_ALPHA},
        include_background=True,
        gpu_number=gpu_number,
        # One image a step: a step's confusion matrix is then an image's.
        batch_size=1,
        num_workers=workers,
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
