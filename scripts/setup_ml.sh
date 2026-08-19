#!/usr/bin/env bash

# setup_ml.sh
# Create the optional inference environment and install the package into it
# (macOS / Linux).
#
# This is a SECOND environment, beside the one setup_dev.sh builds. They are
# kept apart on purpose: the inference stack is several gigabytes, and
# albumentations brings opencv-python-headless, which cannot share an
# environment with conda's `opencv` -- both provide cv2 and having two breaks
# the import. So this environment takes its whole image stack from pip, and the
# curation environment takes its whole image stack from conda.
#
# The window is installed here too, so this one environment does the whole
# workflow: curate, predict, convert, proofread, export. Note that predicting is
# a command (`am-seg predict`) and not yet an action in the window.
#
# Qt comes from the `qt` extra rather than `gui`, which names opencv-python:
# this environment already has opencv-python-headless, which albumentations
# requires, and two providers of cv2 is the hazard described above. Headless is
# the better pairing with PySide6 anyway -- opencv-python carries Qt plugins of
# its own and headless is the build without them.
#
# GPU builds: pip's default torch wheel is CUDA-enabled on Linux and Windows
# (large) and CPU/MPS on macOS. To choose another, set an index URL first:
#
#   AM_SEG_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu  ./scripts/setup_ml.sh
#   AM_SEG_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./scripts/setup_ml.sh

set -euo pipefail

ENV_NAME="am_segmentation_manager_ml"
YML="environment-ml.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Please install Miniconda/Anaconda and try again." >&2
  exit 1
fi

if [ ! -f "$YML" ]; then
  echo "ERROR: $YML not found. Run this script from the project root." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing environment '$ENV_NAME' from $YML..."
  # No `|| true`, for the reason setup_dev.sh gives: a swallowed failure here
  # leaves a half-built environment that only shows itself at runtime.
  conda env update -n "$ENV_NAME" -f "$YML" --prune
else
  echo "Creating environment '$ENV_NAME' from $YML..."
  conda env create -f "$YML"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if [ -n "${AM_SEG_TORCH_INDEX_URL:-}" ]; then
  echo
  echo "Installing torch from $AM_SEG_TORCH_INDEX_URL ..."
  # Before the extra, so the requested build is what satisfies the dependency
  # and the install below leaves it alone.
  pip install torch torchvision --index-url "$AM_SEG_TORCH_INDEX_URL"
fi

echo
echo "Installing the package, the window and the inference dependencies (this downloads GBs)..."
# Not --no-deps, unlike setup_dev.sh: here pip IS the resolver. conda provides
# only python, so there is nothing for pip to install a second copy of.
pip install -e '.[ml,qt]'

echo
echo "Checking the installation..."
python - <<'PYCHECK'
import sys
import time

from am_segmentation_manager.io.image import image_stack_problems
from am_segmentation_manager.ml.environment import ml_stack_problems

failed = False

problems = image_stack_problems()
for name, reason in problems:
    print(f"  MISSING  {name}: {reason}")
if problems:
    print("\nThe image stack is incomplete, so predictions could not be written.")
    failed = True
else:
    print("  image stack   OK (numpy, tifffile, cv2)")

problems = ml_stack_problems()
for name, reason in problems:
    print(f"  MISSING  {name}: {reason}")
if problems:
    print("\nThe inference stack is incomplete. Re-run this script; if torch")
    print("itself failed to import, its wheel does not match this machine --")
    print("set AM_SEG_TORCH_INDEX_URL and try again.")
    sys.exit(1)

import torch

print(f"  inference     OK (torch {torch.__version__})")

try:
    import qtpy  # noqa: F401
    from qtpy import QtWidgets  # noqa: F401
    print("  Qt binding    OK")
except Exception as error:  # noqa: BLE001
    print(f"  Qt binding    NOT USABLE: {error!r}")
    print("  The 'gui' command will not work until this is fixed.")

# Which devices exist is the first thing to know and the last thing to assume:
# a CUDA build on a machine with no visible GPU reports cpu only, and that is
# worth seeing now rather than after an hour of inference at CPU speed.
from am_segmentation_manager.ml.environment import available_devices, verify_preprocessing

devices = available_devices()
print(f"  devices       {', '.join(devices)}")
if devices == ("cpu",):
    print("                CPU only. Inference will run; expect ~1 s per image.")

# Runs the vendored preprocessing end to end. This is the part most likely to
# behave differently on a new platform, and it costs a fraction of a second.
for device in dict.fromkeys([devices[0], "cpu"]):
    try:
        started = time.perf_counter()
        shape = verify_preprocessing(device=device)
        elapsed = time.perf_counter() - started
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        print(f"  preprocess    FAILED on {device}: {error!r}")
        if device == "cpu":
            failed = True
        else:
            print("                Use --device cpu until this is resolved.")
        continue
    print(f"  preprocess    OK on {device} -> {shape} in {elapsed:.2f}s")

if failed:
    sys.exit(1)
PYCHECK

echo
echo "Setup complete. Activate it with:"
echo "  conda activate $ENV_NAME"
echo "This environment does the whole workflow -- 'am-seg gui' as well as"
echo "'am-seg predict'. Run the tests with: pytest -q"
