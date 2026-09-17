#!/usr/bin/env python3
"""Report whether this environment can run each part of the package.

Run after installing, on any platform:

    uv run python scripts/check_environment.py

Checks the image stack, the Qt binding and charts, how many worker processes a
scan starts, and on Windows whether long paths can be reached; then, where the ``ml`` extra is installed, the inference stack, the
devices torch can see, the NVIDIA GPUs and how full they are, and the vendored
preprocessing on each device. Exits non-zero when something installed does not
work, NVIDIA GPUs torch cannot use among it; an extra that was simply not
installed is reported and is not a failure.
"""
from __future__ import annotations

import sys
import time


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

    from am_segmentation_manager.plots import has_plot_support

    if has_plot_support():
        print("  charts        OK (matplotlib)")
    else:
        print("  charts        not installed: the Chart button and 'stats --plot' decline to draw")


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
    import sys

    from am_segmentation_manager.io.long_paths import explanation, long_paths_work

    if sys.platform != "win32":
        return
    if long_paths_work():
        print("  long paths    OK")
    else:
        print(f"  long paths    NOT AVAILABLE: {explanation()}")


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


def main() -> int:
    """Run every check, returning the exit status."""
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    images = check_image_stack()
    check_window()
    check_workers()
    check_long_paths()
    inference = check_inference()
    return 0 if images and inference else 1


if __name__ == "__main__":
    sys.exit(main())
