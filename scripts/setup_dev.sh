#!/usr/bin/env bash

# setup_dev.sh
# Create the conda environment and install the package into it (macOS / Linux).
#
# The environment file provides everything, including the image-processing and
# GUI dependencies. Do NOT afterwards run `pip install -e '.[all]'` in this
# environment: conda's `opencv` and pip's `opencv-python` both provide `cv2`,
# and installing both leaves an import that fails. The pip extras are for a
# pip-only installation.

set -euo pipefail

ENV_NAME="am_segmentation_manager"
YML="environment.yml"

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
  # No `|| true` here. An earlier version swallowed the failure and printed
  # "Setup complete", leaving a half-built environment that only showed itself
  # later as a missing dependency at runtime.
  conda env update -n "$ENV_NAME" -f "$YML" --prune
else
  echo "Creating environment '$ENV_NAME' from $YML..."
  conda env create -f "$YML"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing the package in editable mode..."
# --no-deps deliberately: every dependency comes from environment.yml above --
# conda's, plus the PySide6 it takes from PyPI -- and letting pip resolve them
# here is what installs a second, conflicting opencv.
pip install -e . --no-deps

echo
echo "Checking the installation..."
python - <<'PYCHECK'
import sys

from am_segmentation_manager.io.image import image_stack_problems

problems = image_stack_problems()
for name, reason in problems:
    print(f"  MISSING  {name}: {reason}")
if problems:
    print("\nThe environment is incomplete. Re-run this script, and do not")
    print("pip install the [all] extra into a conda environment.")
    sys.exit(1)

print("  image stack   OK (numpy, tifffile, cv2)")
try:
    import qtpy  # noqa: F401
    from qtpy import QtWidgets  # noqa: F401
    print("  Qt binding    OK")
except Exception as error:  # noqa: BLE001
    print(f"  Qt binding    NOT USABLE: {error!r}")
    print("  The 'gui' command will not work until this is fixed.")

# Reported rather than fatal: everything works without it except drawing, and
# the window says so when asked. Checked here because the alternative is
# finding out from a dialog the first time the Chart button is pressed.
from am_segmentation_manager.plots import has_plot_support

if has_plot_support():
    print("  charts        OK (matplotlib)")
else:
    print("  charts        NOT AVAILABLE: matplotlib is missing")
    print("  The Chart button and 'stats --plot' will decline to draw.")
PYCHECK

echo
echo "Setup complete. Activate it with:"
echo "  conda activate $ENV_NAME"
echo "Run the tests with: pytest -q"
