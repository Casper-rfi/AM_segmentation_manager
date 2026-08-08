am_segmentation_manager
=======================

Package for managing Autolamella FIB/SEM segmentation projects. It provides
filesystem scanning, metadata merging, label management and prediction-to-label
conversion helpers.

CLI
---
Use the built-in CLI help to discover available commands and options.

Top-level help:

  python -m am_segmentation_manager.cli --help

Command-specific help:

  python -m am_segmentation_manager.cli <command> --help

Developer setup
---------------
The repository includes an environment file and a small setup script for
development.

- Create a conda environment from `environment.yml`.
- Install the package in editable mode with `pip install -e .`.
- The helper script `./scripts/setup_dev.sh` can automate these steps on
  macOS/Linux.

Running tests
-------------
Run the test suite from the project root:

  pytest -q

Project layout
--------------
- src/am_segmentation_manager/
  - models.py        : dataclasses for Project, Lamella, AdaptiveMillingStage, ImageRecord
  - scanner.py       : filesystem scanning and ProjectIndex construction
  - metadata.py      : YAML metadata merging helpers
  - io/              : I/O helpers (yaml parsing)
  - operations/      : label movement, CSV helpers, prediction handling
  - cli.py           : CLI entrypoint
- tests/             : pytest tests using tmp_path fixtures

Notes for developers
--------------------
The scanner is the only component that should interrogate the filesystem
layout directly. Other code should operate on the ProjectIndex and the
dataclasses produced by the scanner.
