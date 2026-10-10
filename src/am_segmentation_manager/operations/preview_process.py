"""Predicting a few images with several models to look at, in a process of its own.

**Nothing is written**: no prediction file, no provenance, no history. The masks come
back in memory, each as its model made it, and are the caller's to resize and draw.
Inference stays out of the window's process for the reason
:mod:`.predict_process` gives, whose child's plumbing this shares. One model is on
the device at a time. A model that will not load, or an image that will not open, is
said and passed over, the rest still predicted.
"""
from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..progress import Cancellation
from . import predict_process

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: Turns an image array into a 2D array of class indices.
Predict = Callable[[Any], Any]


@dataclass(frozen=True)
class PreviewOutcome:
    """What a preview's process made, brought back to whoever asked.

    Attributes:
        masks: The class of every pixel by ``(weights, image)``, at the
            model's resolution.
        failures: Why a model was not loaded, by its weights, or an image not
            predicted, by ``(weights, image)``.
        stopped: Whether it was asked to stop before it had finished.
        error: Why nothing ran, or "" when it did.
    """

    masks: Mapping[tuple[Path, Path], np.ndarray] = field(default_factory=dict)
    failures: Mapping[Path | tuple[Path, Path], str] = field(default_factory=dict)
    stopped: bool = False
    error: str = ""


def predict_previews(
    images: Sequence[Path],
    weights: Sequence[Path],
    load: Callable[[Path], Predict],
    read: Callable[[Path], Any],
    on_progress: Callable[[int, int], None] | None = None,
) -> PreviewOutcome:
    """Predict each of *images* with each of *weights* in turn, and return the masks.

    Args:
        load: Loads a model, returning what predicts with it.
        read: Opens an image as a 2D array.
        on_progress: Called as ``on_progress(done, total)`` after each prediction.

    A ``KeyboardInterrupt`` ends it between predictions, what was made kept.
    """
    masks: dict[tuple[Path, Path], Any] = {}
    failures: dict[Path | tuple[Path, Path], str] = {}
    opened: dict[Path, Any] = {}
    total, done = len(images) * len(weights), 0
    try:
        for model in weights:
            try:
                predict = load(model)
            except Exception as error:  # noqa: BLE001 - one model failing leaves the others
                logger.warning("Could not load %s: %s", model, error)
                failures[model] = f"{type(error).__name__}: {error}"
                done += len(images)
            else:
                for image in images:
                    try:
                        if image not in opened:
                            opened[image] = read(image)
                        masks[model, image] = predict(opened[image])
                    except Exception as error:  # noqa: BLE001 - one image failing leaves the others
                        logger.warning("Could not predict %s with %s: %s", image, model.name, error)
                        failures[model, image] = f"{type(error).__name__}: {error}"
                    done += 1
                    if on_progress is not None:
                        on_progress(done, total)
                del predict
            if on_progress is not None:
                on_progress(done, total)
    except KeyboardInterrupt:
        return PreviewOutcome(masks, failures, stopped=True)
    return PreviewOutcome(masks, failures)


def _child(images: Sequence[str], weights: Sequence[str], device: str, queue) -> None:
    """Load each model and predict, reporting down *queue*. Runs in the child."""
    predict_process._forward_logs(queue)
    try:
        # In the child, deliberately: torch is never imported into the parent.
        from ..io.image import read_array, to_2d
        from ..ml.environment import limit_cpu_threads, resolve_device
        from ..ml.model import load_model

        # Before anything asks torch about CUDA, so the GPU can be pinned.
        resolved = resolve_device(device)
        limit_cpu_threads()
        outcome = predict_previews(
            [Path(image) for image in images],
            [Path(model) for model in weights],
            load=lambda model: load_model(model, device=resolved).predict,
            read=lambda image: to_2d(read_array(image)),
            on_progress=lambda done, total: queue.put(("progress", done, total)),
        )
        queue.put(("done", outcome))
    except BaseException as error:  # noqa: BLE001 - the only way back is the queue
        queue.put(("done", PreviewOutcome(error=f"{type(error).__name__}: {error}")))


def preview_out_of_process(
    images: Sequence[Path],
    weights: Sequence[Path],
    device: str = "auto",
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancellation: Cancellation | None = None,
    context: str | None = None,
) -> PreviewOutcome:
    """Predict *images* with each of *weights* in a child process, and return the masks.

    Blocking, as :func:`~.predict_process.predict_out_of_process` is.

    Args:
        device: A torch device, or "auto" for the most capable the child can see.
        on_progress: Called as ``on_progress(done, total)`` as the child works.
        cancellation: Polled while waiting; stopping keeps what was predicted.
        context: Multiprocessing start method; "spawn" unless a test says otherwise.
    """
    if not images or not weights:
        return PreviewOutcome()
    ctx: Any = multiprocessing.get_context(context or "spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_child,
        args=([str(image) for image in images], [str(model) for model in weights], device, queue),
        daemon=True,
    )
    logger.info("Predicting %d image(s) with %d model(s) to look at", len(images), len(weights))
    process.start()
    try:
        outcome = predict_process._await(process, queue, on_progress, cancellation)
    finally:
        predict_process._reap(process)
    if isinstance(outcome, PreviewOutcome):
        return outcome
    # The child died without an outcome, which the shared loop reports as its own kind.
    return PreviewOutcome(error=outcome.error)
