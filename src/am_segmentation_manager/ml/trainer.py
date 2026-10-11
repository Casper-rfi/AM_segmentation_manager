"""Training a model through AM-model-training, in the process that calls it.

The one place AM-model-training is imported, inside :func:`train`, so that only a
process started to train pays for MONAI and mlflow and takes the process-wide
settings they make. It trains generation 1 and nothing else, on a CUDA GPU or the
CPU, and is always given a training and a validation list: left to split one
itself it shuffles at random, through lamellae.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .environment import numbered_device, require_training_stack, resolve_device

logger = logging.getLogger(__name__)


def trainable_device(device: str) -> tuple[bool, int]:
    """Return ``(cpu_only, gpu_number)`` for AM-model-training from a torch device.

    Anything but CUDA trains on the CPU: AM-model-training knows no other
    accelerator, Apple silicon's included.
    """
    kind, _, index = device.partition(":")
    if kind != "cuda":
        return True, 0
    return False, int(index) if index.isdigit() else 0


def training_arguments(
    train_list: Path,
    validation_list: Path,
    output: Path,
    *,
    device: str,
    base_weights: Path | None,
    max_epochs: int,
    frozen_epochs: int,
    initial_learning_rate: float,
    max_learning_rate: float,
    training_batch_size: int,
    validation_batch_size: int,
    save_all: bool,
    seed: int,
) -> dict[str, Any]:
    """Return what ``am_model_training.main.train`` is called with.

    Args:
        device: The torch device already resolved for this process.
    """
    cpu_only, gpu_number = trainable_device(device)
    return {
        "output_path": output,
        "csv": (train_list, validation_list),
        "max_epochs": max_epochs,
        "frozen_epochs": frozen_epochs,
        "weights_path": base_weights,
        "cpu_only": cpu_only,
        "gpu_number": gpu_number,
        "training_batch_size": training_batch_size,
        "validation_batch_size": validation_batch_size,
        "initial_learning_rate": initial_learning_rate,
        "max_learning_rate": max_learning_rate,
        "save_all_models": save_all,
        "seed": seed,
        "log_mlflow": False,
    }


def train(
    train_list: Path,
    validation_list: Path,
    output: Path,
    *,
    device: str = "auto",
    base_weights: Path | None = None,
    on_device: Callable[[str], None] | None = None,
    **settings: Any,
) -> str:
    """Train a model on the two lists, writing into *output*, and return the device used.

    The device is returned as ``nvidia-smi`` numbers it, not as this process does.

    Blocks until training ends. **Call it before anything in the process uses
    CUDA**, so the GPU it resolves can be pinned.

    Args:
        device: ``"auto"``, or a torch device, as :func:`~.environment.resolve_device` takes.
        base_weights: The checkpoint to refine, or None to start from ImageNet weights.
        on_device: Told the device, as it is returned, before training begins.
        settings: The remaining arguments of :func:`training_arguments`.

    Raises:
        RuntimeError: If the training stack is not installed or does not load, or
            if torch cannot use this machine's NVIDIA GPUs and no CPU was named.
    """
    require_training_stack("Training")
    resolved = resolve_device(device, fall_back=False)
    if resolved != "cpu" and trainable_device(resolved)[0]:
        logger.warning(
            "AM-model-training trains on a CUDA GPU or the CPU, not on %s; training on the CPU",
            resolved,
        )
        resolved = "cpu"
    arguments = training_arguments(
        train_list, validation_list, output,
        device=resolved, base_weights=base_weights, **settings,
    )
    used = numbered_device(resolved)
    logger.info("Training on %s into %s", used, output)
    if on_device is not None:
        on_device(used)

    from am_model_training.main import train as am_train

    am_train(**arguments)
    return used
