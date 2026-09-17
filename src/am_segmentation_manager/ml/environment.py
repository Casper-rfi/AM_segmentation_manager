"""Reporting whether the inference stack is usable, and on which devices.

The counterpart of :mod:`..io.image` for the machine-learning dependencies:
torch is imported inside functions, never at module level, so this module can
be imported and asked its questions in an installation that has no torch at
all. ``scripts/check_environment.py`` and every inference command ask through
here rather than catching ``ImportError`` themselves.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping

from .gpus import (
    LOW_MEMORY_MIB,
    Gpu,
    GpuInventory,
    gpu_inventory,
    least_used,
    number_as_nvidia_smi,
    pin_to,
)

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "Install the ml extra: uv sync --extra gui --extra ml, or pip install "
    "'am_segmentation_manager[ml]' -- but never beside conda's opencv or "
    "opencv-python, since the extra brings opencv-python-headless and two "
    "providers of cv2 break the import."
)

#: The modules that must import before any prediction can be made, and the
#: distribution each comes from. Named by import name because that is what
#: fails, and by distribution because that is what you install.
ML_MODULES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
)


def ml_stack_problems() -> list[tuple[str, str]]:
    """Return ``(distribution, reason)`` for each dependency that will not import.

    Distinguishes "not installed" from "installed but broken" for the same
    reason :func:`..io.image.image_stack_problems` does: a torch that fails to
    load its native libraries reports as neither, and reading "not installed"
    sends you to install it again.
    """
    problems: list[tuple[str, str]] = []
    for module, distribution in ML_MODULES:
        try:
            __import__(module)
        except ImportError as error:
            problems.append((distribution, f"not installed ({error})"))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            problems.append(
                (distribution, f"installed but failed to import: {error!r}")
            )
    return problems


def ml_stack_present() -> bool:
    """Return whether the dependencies are installed, without importing them.

    Locates the modules rather than loading them, because importing torch costs
    seconds and the window asks this while building its buttons. It cannot tell
    an installed-but-broken torch from a working one, which is what
    :func:`has_ml_support` is for: this decides whether to offer the button,
    that decides whether the run can go ahead.
    """
    import importlib.util

    return all(
        importlib.util.find_spec(module) is not None for module, _ in ML_MODULES
    )


def has_ml_support() -> bool:
    """Return whether the optional inference stack is usable."""
    return not ml_stack_problems()


def require_ml_stack(operation: str) -> None:
    """Raise a helpful error if the inference stack is unavailable.

    Args:
        operation: What the caller was trying to do, used in the message.

    Raises:
        RuntimeError: If any dependency cannot be imported, naming which ones
            and why.
    """
    problems = ml_stack_problems()
    if not problems:
        return
    detail = "; ".join(f"{name}: {reason}" for name, reason in problems)
    logger.warning("%s unavailable -- %s", operation, detail)
    raise RuntimeError(f"{operation} needs {detail}. {INSTALL_HINT}")


def available_devices() -> tuple[str, ...]:
    """Return the torch devices this machine offers, most capable first.

    ``cpu`` is always present and always last: it is the fallback whenever a
    GPU is absent, busy or refuses the work, and inference that only runs on a
    GPU is inference that does not run on a laptop.

    Raises:
        RuntimeError: If the inference stack is not installed.
    """
    require_ml_stack("Listing devices")
    number_as_nvidia_smi()
    import torch

    devices: list[str] = []
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return tuple(devices)


def discovered_devices() -> tuple[str, ...]:
    """Return the devices torch reports, or ``()`` if asking would cost seconds.

    The window offers a device to predict on and must not import torch to build
    a dropdown -- the same reason :func:`ml_stack_present` locates the modules
    rather than loading them. Once anything in the process has imported torch,
    which pressing Predict does, this answers instantly and the dropdown fills
    in with whatever the machine actually has: several CUDA devices, an MPS
    one, or neither.

    Returns:
        The devices, most capable first, or ``()`` when torch is not loaded or
        cannot answer.
    """
    if "torch" not in sys.modules:
        return ()
    try:
        return available_devices()
    except RuntimeError:  # pragma: no cover - torch present but unusable
        return ()


def resolve_device(
    requested: str | None = None,
    inventory: GpuInventory | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Return the torch device to run on in this process, pinning the GPU it names.

    ``None`` or ``"auto"`` takes the NVIDIA GPU with the most free memory where
    ``nvidia-smi`` lists some and torch can use them, and otherwise the most
    capable device available: a GPU is several times faster, and inference over
    a whole project is the difference between an hour and an afternoon.

    A named device is not checked against what is installed -- the load that
    follows reports one that does not exist more precisely -- but a CUDA GPU
    ``nvidia-smi`` lists is pinned as the only one visible, so it is ``cuda:0``
    in what this returns. **Call it before anything in the process uses CUDA**,
    or the pinning cannot take effect and is skipped.

    Raises:
        RuntimeError: If the inference stack is not installed and a choice has
            to be made.
    """
    environ = os.environ if environ is None else environ
    number_as_nvidia_smi(environ)
    requested = (requested or "auto").strip()
    if inventory is None:
        wants_gpu = requested == "auto" or requested.startswith("cuda")
        inventory = gpu_inventory(environ=environ) if wants_gpu else GpuInventory()

    if requested != "auto":
        named = inventory.by_device(requested)
        return requested if named is None else _pin(named, environ)

    gpu = least_used(inventory)
    if gpu is None:
        return available_devices()[0]
    device = _pin(gpu, environ, chosen_by="auto")
    problem = cuda_problem(inventory)
    if problem:
        logger.warning("%s", problem)
        return available_devices()[0]
    return device


def cuda_problem(inventory: GpuInventory) -> str:
    """Return why torch cannot use the NVIDIA GPUs in *inventory*, or "" if it can.

    Also "" where there are none. Asking torch starts CUDA, so a GPU is pinned
    before this is called, never after.

    Raises:
        RuntimeError: If there are GPUs to ask about and no inference stack.
    """
    if not inventory.gpus:
        return ""
    require_ml_stack("Checking CUDA")
    import torch

    if torch.cuda.is_available():
        return ""
    found = f"{len(inventory.gpus)} NVIDIA GPU(s) found"
    if torch.version.cuda is None:
        return (
            f"{found}, but this torch ({torch.__version__}) was built without "
            f"CUDA, so predictions run on the CPU. Install the CUDA build: "
            f"uv sync --extra gui --extra ml"
        )
    return (
        f"{found}, but torch cannot use them: it was built for CUDA "
        f"{torch.version.cuda} and the driver is {inventory.driver_version or 'unknown'}, "
        f"and a driver too old for that CUDA version is the usual cause. "
        f"Predictions run on the CPU."
    )


def limit_cpu_threads() -> int:
    """Hold torch to as many CPU threads as am-seg may use, and return how many.

    Preprocessing runs on the CPU whatever the device, and torch otherwise takes
    every core of the machine. The count is :func:`~..parallel.worker_count`'s,
    so ``AM_SEG_WORKERS`` sets both.

    Raises:
        RuntimeError: If the inference stack is not installed.
    """
    require_ml_stack("Setting CPU threads")
    import torch

    from ..parallel import worker_count

    threads = worker_count()
    torch.set_num_threads(threads)
    return threads


def _pin(gpu: Gpu, environ: MutableMapping[str, str], chosen_by: str = "") -> str:
    """Pin *gpu* where CUDA has not started, returning the device to load onto."""
    how = f"{chosen_by}: " if chosen_by else ""
    if gpu.free_mib < LOW_MEMORY_MIB:
        logger.warning(
            "%s has %.1f GB free; a model may not fit beside what is already on it",
            gpu.device, gpu.free_mib / 1024,
        )
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        logger.info(
            "%s%s -- CUDA is already in use in this process, so other GPUs "
            "stay visible", how, gpu.describe(),
        )
        return gpu.device
    device = pin_to(gpu, environ)
    logger.info("%s%s, the only GPU visible to this run", how, gpu.describe())
    return device


def verify_preprocessing(device: str = "cpu", image_size: int = 512) -> tuple[int, ...]:
    """Run the vendored preprocessing over a synthetic image, returning its shape.

    An end-to-end exercise of the parts of torch and torchvision inference
    depends on -- normalisation, the bicubic resize and the greyscale-to-RGB
    expansion -- which is what a fresh installation on an unfamiliar platform
    gets wrong. Deliberately smaller than the 1536 a model expects: this checks
    that the operations run, not how fast they are.

    Args:
        device: The torch device to run on.
        image_size: Longest output edge, in pixels.

    Returns:
        The shape of the preprocessed tensor, ``(1, 3, height, width)``.

    Raises:
        RuntimeError: If the inference stack is not installed.
    """
    require_ml_stack("Checking preprocessing")
    import numpy as np

    # torch, imported when a check asks for it rather than when the module
    # naming the devices is read.
    from .preprocessing import image_preprocess

    image = np.random.default_rng(0).integers(
        0, 256, size=(256, 384), dtype=np.uint16
    ).astype(np.float32)
    tensor = image_preprocess(
        image,
        image_size=image_size,
        normalise_first=True,
        rgb=True,
        pad=False,
        device=device,
    )
    return tuple(int(size) for size in tensor.shape)
