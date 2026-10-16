"""``am-seg check-environment``: whether this environment can run each part of AM-seg.

A surface onto the installed environment, reading no collection. It checks the image
stack, the window, charts, worker processes and, on Windows, long paths; then, where
the ``ml`` and ``train`` extras are installed, inference on each device torch sees and
AM-model-training. **Something installed that does not work fails the check; an extra
that was never installed does not.** torch is imported only inside the checks.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from .. import __version__
from ..installation import install_kind, install_record, version_date

#: How long a child process gets to open Qt, loading its plugins from what may
#: be a slow network drive.
WINDOW_TIMEOUT_S = 60

#: What the child runs: create the application, which is where Qt loads the
#: plugin that draws windows, and leave.
_OPEN_QT = "from qtpy.QtWidgets import QApplication; QApplication([])"

#: How each kind of install is named in the report's first line.
_INSTALLED_AS = {"checkout": "source checkout", "repository": "installed from git", "other": ""}


def check_image_stack() -> bool:
    """Report the image stack, returning whether it imports."""
    from am_segmentation_manager.io.image import image_stack_problems

    problems = image_stack_problems()
    for name, reason in problems:
        print(f"  MISSING  {name}: {reason}")
    if problems:
        print("  Images cannot be read until this is fixed.")
        return False
    print("  image stack   OK (numpy, tifffile, cv2)")
    return True


def check_window() -> None:
    """Report the Qt binding and matplotlib; neither is a failure when absent."""
    import importlib.util

    if importlib.util.find_spec("qtpy") is None:
        print("  Qt binding    not installed: 'am-seg gui' needs the gui extra")
    else:
        try:
            from qtpy import QtWidgets  # noqa: F401
            print("  Qt binding    OK")
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            print(f"  Qt binding    NOT USABLE: {error!r}")
        else:
            check_qt_window()

    from am_segmentation_manager.plots import has_plot_support

    if has_plot_support():
        print("  charts        OK (matplotlib)")
    else:
        print("  charts        not installed: the Chart button and 'stats --plot' decline to draw")


def check_qt_window(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    platform: str = sys.platform,
    environ: Mapping[str, str] = os.environ,
) -> None:
    """Report whether Qt can open a window here; not a failure when it cannot.

    In a child process, because Qt ends the process it runs in when no plugin
    for drawing windows will load, as when a system library is missing.
    """
    if platform.startswith("linux") and not (
        environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY")
    ):
        print("  Qt window     not checked: no display (DISPLAY and WAYLAND_DISPLAY are unset)")
        return
    try:
        result = run(
            [sys.executable, "-c", _OPEN_QT],
            capture_output=True, text=True, timeout=WINDOW_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"  Qt window     NOT CHECKED: Qt did not answer within {WINDOW_TIMEOUT_S} s")
        return
    except OSError as error:
        print(f"  Qt window     NOT CHECKED: {error}")
        return
    if result.returncode == 0:
        print("  Qt window     OK")
        return
    print(f"  Qt window     CANNOT OPEN: {_qt_reason(result.stderr)}")
    if "xcb-cursor" in result.stderr:
        print(
            "                On Ubuntu or Debian an administrator installs it with: "
            "sudo apt install libxcb-cursor0"
        )
    else:
        print(
            "                Set QT_DEBUG_PLUGINS=1 and run 'am-seg gui' to see which "
            "library Qt could not load"
        )


def _qt_reason(stderr: str) -> str:
    """Return the line of Qt's output that says why, or its last line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if "is needed" in line:
            return line.removeprefix("qt.qpa.plugin: ")
    return lines[-1] if lines else "Qt ended without saying why"


def check_workers() -> None:
    """Report how many worker processes a scan starts, and what changes it."""
    from am_segmentation_manager.parallel import WORKERS_ENV_VAR, available_cpus, worker_count

    print(
        f"  workers       {worker_count()} of {available_cpus()} CPU(s) "
        f"({WORKERS_ENV_VAR} changes it)"
    )


def check_long_paths() -> None:
    """Report, on Windows, whether paths past 260 characters can be reached here.

    Not a failure: a collection with short enough paths never needs them, and
    the Add data report says when one does not.
    """
    from am_segmentation_manager.io.long_paths import explanation, long_paths_work

    if sys.platform != "win32":
        return
    if long_paths_work():
        print("  long paths    OK")
    else:
        print(f"  long paths    off. {explanation()}")


def check_inference() -> bool:
    """Report the inference stack and its devices, returning whether it works."""
    from am_segmentation_manager.ml.environment import (
        available_devices,
        cuda_problem,
        ml_stack_present,
        ml_stack_problems,
        verify_preprocessing,
    )
    from am_segmentation_manager.ml.gpus import gpu_inventory

    if not ml_stack_present():
        print("  inference     not installed: predicting needs the ml extra")
        return True
    problems = ml_stack_problems()
    for name, reason in problems:
        print(f"  MISSING  {name}: {reason}")
    if problems:
        print("  torch is installed but does not load; its build does not match this machine.")
        return False

    import torch

    print(f"  inference     OK (torch {torch.__version__})")
    devices = available_devices()
    print(f"  devices       {', '.join(devices)}")
    if devices == ("cpu",):
        print("                CPU only. Inference will run; expect ~1 s per image.")

    working = True
    inventory = gpu_inventory()
    for gpu in inventory.gpus:
        print(f"  gpu           {gpu.describe()}")
    problem = cuda_problem(inventory)
    if problem:
        print(f"  GPUs          NOT USABLE: {problem}")
        working = False

    for device in dict.fromkeys([devices[0], "cpu"]):
        try:
            started = time.perf_counter()
            shape = verify_preprocessing(device=device)
            elapsed = time.perf_counter() - started
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            print(f"  preprocess    FAILED on {device}: {error!r}")
            if device == "cpu":
                working = False
            else:
                print("                Use --device cpu until this is resolved.")
            continue
        print(f"  preprocess    OK on {device} -> {shape} in {elapsed:.2f}s")
    return working


def check_training() -> bool:
    """Report the training stack, returning whether it works where installed."""
    from importlib.metadata import version

    from am_segmentation_manager.ml.environment import (
        training_stack_present,
        training_stack_problems,
    )

    if not training_stack_present():
        print("  training      not installed: training a model needs the train extra")
        return True
    problems = training_stack_problems()
    for name, reason in problems:
        print(f"  MISSING  {name}: {reason}")
    if problems:
        print("  AM-model-training is installed but does not load.")
        return False

    import torch

    print(
        f"  training      OK (am-model-training {version('am-model-training')}, "
        f"monai {version('monai')})"
    )
    if not torch.cuda.is_available():
        print(
            "                No CUDA GPU: AM-model-training trains on the CPU here, "
            "which takes minutes per image."
        )
    return True


def installation_line(record: Mapping[str, Any] | None = None) -> str:
    """Return the report's first line: the version, how it was installed, and how recent it is.

    Args:
        record: The install record, read from the installed package where omitted.
    """
    record = install_record() if record is None else record
    parts = [f"am-seg {__version__}", _INSTALLED_AS[install_kind(record)], version_date(record)]
    return " · ".join(part for part in parts if part)


def run_checks() -> int:
    """Run every check, returning the exit status."""
    print(installation_line())
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    images = check_image_stack()
    check_window()
    check_workers()
    check_long_paths()
    inference = check_inference()
    training = check_training()
    return 0 if images and inference and training else 1
