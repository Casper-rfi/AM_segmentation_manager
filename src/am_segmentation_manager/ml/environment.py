"""Reporting whether the inference and training stacks are usable, and on which devices.

The counterpart of :mod:`..io.image` for the machine-learning dependencies:
torch and AM-model-training are imported inside functions, never at module level,
so this module can be imported and asked its questions in an installation that
has neither. ``am-seg check-environment`` and every inference or training
command ask through here rather than catching ``ImportError`` themselves.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

from ..installation import install_kind, install_record
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
    "'adaptive-milling-seg[ml]' -- but never beside conda's opencv or "
    "opencv-python, since the extra brings opencv-python-headless and two "
    "providers of cv2 break the import."
)

TRAINING_INSTALL_HINT = (
    "Install the train extra beside an inference one: uv sync --extra gui --extra ml "
    "--extra train (ml-cuda13 in place of ml where the card needs it), or pip install "
    "'adaptive-milling-seg[ml,train]'. AM-model-training supports Python up to 3.13."
)

#: The module whose import starts AM-model-training, and the distribution it
#: comes from. Importing it loads MONAI, mlflow, pandas and cv2 as well, so it
#: is the one import that tells whether training can start.
TRAINING_MODULE: tuple[str, str] = ("am_model_training.main", "am-model-training")

#: Set before AM-model-training is imported: mlflow otherwise logs advice meant
#: for someone writing tracing code into every process that loads it.
QUIET_MLFLOW: tuple[str, str] = ("MLFLOW_DISABLE_AGENT_HINT", "1")

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
    return _import_problems(ML_MODULES)


def _import_problems(modules: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    """Return ``(distribution, reason)`` for each of *modules* that will not import."""
    problems: list[tuple[str, str]] = []
    for module, distribution in modules:
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


def training_stack_present() -> bool:
    """Return whether the inference stack and AM-model-training are installed, without importing them.

    For the same reason as :func:`ml_stack_present`: this decides whether to
    offer training, and :func:`training_stack_problems` whether it can start.
    """
    import importlib.util

    package = TRAINING_MODULE[0].split(".")[0]
    return ml_stack_present() and importlib.util.find_spec(package) is not None


def training_stack_problems() -> list[tuple[str, str]]:
    """Return ``(distribution, reason)`` for each part of the training stack that will not import.

    Importing AM-model-training costs seconds and loads MONAI and mlflow, so
    this is asked where training is about to start, not to draw a button.
    """
    problems = ml_stack_problems()
    if problems:
        return problems
    os.environ.setdefault(*QUIET_MLFLOW)
    return _import_problems((TRAINING_MODULE,))


def require_training_stack(operation: str) -> None:
    """Raise a helpful error if the training stack is unavailable.

    Raises:
        RuntimeError: If any part cannot be imported, naming which and why.
    """
    problems = training_stack_problems()
    if not problems:
        return
    detail = "; ".join(f"{name}: {reason}" for name, reason in problems)
    logger.warning("%s unavailable -- %s", operation, detail)
    raise RuntimeError(f"{operation} needs {detail}. {TRAINING_INSTALL_HINT}")


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


def resolve_device(
    requested: str | None = None,
    inventory: GpuInventory | None = None,
    environ: MutableMapping[str, str] | None = None,
    *,
    fall_back: bool = True,
) -> str:
    """Return the torch device to run on in this process, pinning the GPU it names.

    ``None`` or ``"auto"`` takes the NVIDIA GPU with the most free memory where
    ``nvidia-smi`` lists some and torch can use them, and otherwise the most
    capable device available: a GPU is several times faster, and inference over
    a whole project is the difference between an hour and an afternoon.

    A named device ``nvidia-smi`` does not list is not checked against what is
    installed -- the load that follows reports one that does not exist more
    precisely. A CUDA GPU it does list is pinned as the only one visible, so it
    is ``cuda:0`` in what this returns. **Call it before anything in the process
    uses CUDA**, or the pinning cannot take effect and is skipped.

    Args:
        fall_back: Whether ``"auto"`` may take another device where torch cannot
            use the NVIDIA GPUs here. False is for work too long to start on a
            CPU nobody named.

    Raises:
        RuntimeError: If the inference stack is not installed and a choice has
            to be made, if torch cannot use the NVIDIA GPU named, or if it can
            use none of them and *fall_back* is False.
    """
    environ = os.environ if environ is None else environ
    number_as_nvidia_smi(environ)
    requested = (requested or "auto").strip()
    if inventory is None:
        wants_gpu = requested == "auto" or requested.startswith("cuda")
        inventory = gpu_inventory(environ=environ) if wants_gpu else GpuInventory()

    if requested != "auto":
        named = inventory.by_device(requested)
        if named is None:
            return requested
        device = _pin(named, environ)
        problem = cuda_problem(inventory)
        if problem:
            raise RuntimeError(f"{requested} cannot be used. {problem}")
        return device

    gpu = least_used(inventory)
    if gpu is None:
        return available_devices()[0]
    device = _pin(gpu, environ, chosen_by="auto")
    problem = cuda_problem(inventory)
    if not problem:
        return device
    if not fall_back:
        raise RuntimeError(
            f"Not started on the CPU, which nobody named. {problem} "
            f"To run on the CPU regardless, name cpu as the device."
        )
    # CUDA can be available and still unusable, a card newer than the build's
    # kernels, so the fallback is never whichever device is first.
    fallback = next(
        other for other in available_devices() if not other.startswith("cuda")
    )
    logger.warning("Running on %s, not a GPU. %s", fallback, problem)
    return fallback


def numbered_device(device: str, inventory: GpuInventory | None = None) -> str:
    """Return *device*, a resolved one, under the number ``nvidia-smi`` gives its GPU.

    A pinned GPU is ``cuda:0`` in its process whichever card it is, which tells
    a reader of the run nothing. Anything that is not a GPU ``nvidia-smi`` lists
    is returned as it came.

    Args:
        inventory: The GPUs this process may use now; asked of ``nvidia-smi`` where omitted.
    """
    if not device.startswith("cuda"):
        return device
    gpu = (gpu_inventory() if inventory is None else inventory).by_device(device)
    return device if gpu is None else f"cuda:{gpu.physical_index}"


def how_to_install(extra: str, record: Mapping[str, Any] | None = None) -> str:
    """Return the command that installs the inference *extra* into this copy.

    Args:
        record: The install record, read from the installed package where omitted.
    """
    sync = f"uv sync --extra gui --extra {extra}"
    tool = f"the uv tool install command again with [gui,{extra}] and --upgrade"
    kind = install_kind(install_record() if record is None else record)
    if kind == "checkout":
        return sync
    if kind == "repository":
        return tool
    return f"{sync} in a checkout, or {tool}"


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
        return _missing_kernels(torch)
    found = f"{len(inventory.gpus)} NVIDIA GPU(s) found"
    if torch.version.cuda is None:
        return (
            f"{found}, but this torch ({torch.__version__}) was built without "
            f"CUDA. Install the CUDA build: {how_to_install('ml')}"
        )
    return (
        f"{found}, but torch cannot use them: it was built for CUDA "
        f"{torch.version.cuda} and the driver is {inventory.driver_version or 'unknown'}, "
        f"and a driver too old for that CUDA version is the usual cause."
    )


def _missing_kernels(torch) -> str:
    """Return why torch has no kernels for the GPUs it can see, or "" if it has.

    A driver new enough to run CUDA is not a build that can run on the card:
    each torch wheel carries kernels for the architectures it was compiled for,
    and a card newer than all of them fails at the first launch rather than at
    load. Asked only where CUDA is available, every visible GPU having been
    checked before any of them is used.
    """
    architectures = [
        _compute_capability(name) for name in torch.cuda.get_arch_list()
    ]
    supported = {capability for capability in architectures if capability is not None}
    if not supported:
        return ""
    unusable = []
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        if any(built == major and level <= minor for built, level in supported):
            continue
        unusable.append(f"{torch.cuda.get_device_name(index)} (sm_{major}{minor})")
    if not unusable:
        return ""
    built_for = ", ".join(f"sm_{major}{minor}" for major, minor in sorted(supported))
    return (
        f"{len(unusable)} NVIDIA GPU(s) too new for this torch "
        f"({torch.__version__}): {', '.join(unusable)}, against kernels for "
        f"{built_for}. Install the build that has them: "
        f"{how_to_install('ml-cuda13')}"
    )


def _compute_capability(architecture: str) -> tuple[int, int] | None:
    """Return the (major, minor) a ``sm_90``-style name states, or None.

    ``compute_90`` entries name PTX rather than kernels, and are read the same
    way: a card can run PTX built for its own architecture.
    """
    _, _, digits = architecture.partition("_")
    if not digits.isdigit() or len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def limit_cpu_threads() -> int:
    """Hold torch to as many CPU threads as AM-seg may use, and return how many.

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
    """Run generation 1's preprocessing over a synthetic image, returning its shape.

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
    from .preprocessing import preprocess

    image = np.random.default_rng(0).integers(0, 256, size=(256, 384), dtype=np.uint16)
    tensor = preprocess(image, image_size=image_size, rgb=True, device=device)
    return tuple(int(size) for size in tensor.shape)
