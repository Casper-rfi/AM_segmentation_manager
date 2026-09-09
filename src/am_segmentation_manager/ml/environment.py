"""Reporting whether the inference stack is usable, and on which devices.

The counterpart of :mod:`..io.image` for the machine-learning dependencies:
torch is imported inside functions, never at module level, so this module can
be imported and asked its questions in an installation that has no torch at
all. The install script and every future inference command ask through here
rather than catching ``ImportError`` themselves.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "Install them into a separate environment with ./scripts/setup_ml.sh, "
    "which is kept apart from the main environment because the inference "
    "stack is several gigabytes and brings its own OpenCV. Outside conda: "
    "pip install 'am_segmentation_manager[ml]' -- but never into an "
    "environment that already has the [all] extra or conda's opencv, since "
    "opencv-python-headless and opencv-python both provide cv2 and having two "
    "breaks the import."
)

#: The modules that must import before any prediction can be made, and the
#: distribution each comes from. Named by import name because that is what
#: fails, and by distribution because that is what you install.
ML_MODULES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("albumentations", "albumentations"),
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


def resolve_device(requested: str | None = None) -> str:
    """Return the torch device to run on.

    ``None`` or ``"auto"`` takes the most capable device available, which on
    this collection's images is worth having: a GPU is several times faster,
    and inference over a whole project is the difference between an hour and an
    afternoon.

    An explicitly named device is returned unexamined -- not even checked
    against :func:`available_devices`, which would import torch to answer.
    Asking what is installed is only needed when something has to choose, and a
    device that does not exist is reported by the load that follows far more
    precisely than a warning here could.

    Raises:
        RuntimeError: If the inference stack is not installed and a choice has
            to be made.
    """
    if requested is None or requested == "auto":
        return available_devices()[0]
    return requested


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
