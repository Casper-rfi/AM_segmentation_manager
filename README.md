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

Curation window
---------------
A desktop window for reviewing SEM and FIB images against their labels and
setting each image's workflow state:

  python -m am_segmentation_manager.cli gui [START_PATH]

`START_PATH` is optional; the folder can also be chosen in the window, and the
last one used is remembered. Reviewing is safe by default: the state controls
stay locked until *Enable editing* is ticked.

The window shows one lamella at a time, as a single chronological stack of
frames across all of its adaptive milling stages. The timeline at the bottom
shows curation progress for the whole lamella, SEM on the top row and FIB
below.

**Requires a Qt binding.** `qtpy` alone is not enough -- it is an adapter and
needs PySide6 (or PyQt5) underneath:

  conda install -c conda-forge pyside6 qtpy

Do not install a second Qt binding alongside the first: loading two into one
process crashes at import. If the environment already has PyQt5 (via napari,
for example), install `qtpy` only.

| Key | Action |
| --- | --- |
| Left / Right | Previous / next frame |
| Up / Down | Previous / next lamella |
| 1 / 2 / 3 | SEM / FIB / plot view |
| Space | Show or hide the label overlay |
| D / S / C | Set state: deselected / selected / corrected |
| F | Fit the images to the panes |
| F1 | List the shortcuts |

Class names and colours come from `segmentation_config_AM.yaml`, searched for
in the scan folder and falling back to the copy bundled with the package. Pass
`--config` to use a specific file.

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
  - frames.py        : per-lamella chronological view pairing SEM with FIB
  - config.py        : segmentation class names and colours
  - colors.py        : CSS colour name to RGB
  - display.py       : downscaling, bit depth and label overlay composition
  - io/              : I/O helpers (yaml parsing, image reading)
  - operations/      : label movement, CSV helpers, prediction handling
  - gui/             : curation window (only gui/widgets/ imports Qt)
  - cli.py           : CLI entrypoint
- tests/             : pytest tests using tmp_path fixtures
- docs/gui_plan.md   : design notes for the curation window

Notes for developers
--------------------
The scanner is the only component that should interrogate the filesystem
layout directly. Other code should operate on the ProjectIndex and the
dataclasses produced by the scanner.
