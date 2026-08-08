#!/usr/bin/env bash

# setup_dev.sh
# Convenience script to create the conda environment and install the package
# in editable mode (macOS / Linux).

set -euo pipefail

ENV_NAME="am_segmentation_manager"
YML="environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Please install Miniconda/Anaconda and try again." >&2
  exit 1
fi

if [ ! -f "$YML" ]; then
  echo "ERROR: $YML not found in the current directory. Run this script from the project root." >&2
  exit 1
fi

echo "Creating or updating conda environment '$ENV_NAME' from $YML..."
# Create the env if it doesn't exist, otherwise try to update it.
set +e
conda env create -f "$YML"
CREATE_RC=$?
set -e
if [ $CREATE_RC -ne 0 ]; then
  echo "Environment may already exist; attempting to update it instead..."
  conda env update -f "$YML" || true
fi

# Activate the environment in this script so we can run pip install
# Use the conda shell hook for reliable activation
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing package in editable mode (pip install -e .)"
pip install -e .

echo
echo "Setup complete."
echo "To start using the environment in your shell run:"
echo "  conda activate $ENV_NAME"
echo "Optional: install extras for image processing:"
echo "  pip install -e '.[all]'"

echo "Run tests with: pytest -q"

exit 0
