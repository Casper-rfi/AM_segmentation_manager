am_segmentation_manager
=======================

Package for managing Autolamella FIB/SEM segmentation projects. It provides
filesystem scanning, metadata merging, label management and prediction-to-label
conversion helpers.

Dataset layouts
---------------
A **project** is any directory holding `experiment.yaml` (written by
Autolamella) or `AM_metadata.yaml` (written by this package). Projects are
identified by their path relative to the scan root, not by name: names are not
unique, and neither is anything inside `experiment.yaml`.

Layouts back to 2023 are supported, including ones that collapse the hierarchy.
The examples below are ones that have been observed, not an exhaustive list:
the properties vary independently and can combine in ways none of them shows.
Where a dataset has no lamella or stage directory, the parent stands in and the
level is reported as synthetic rather than invented:

    project/lamella/Milling/Polishing/adaptive_polish_<timestamp>/sem/   2026
    project/lamella/adaptive_polish_<timestamp>/sem/                     2025
    project/lamella/adaptive_polish/sem/                                 2024
    project/lamella/SEM_8/                     no stage directory
    project/SEM/                               no lamella or stage directory

A label directory is named after the image directory it belongs to:
`sem/` -> `sem_labels/`, `sem_5kv/` -> `sem_5kv_labels/`. `<modality>_labels`
therefore acts as the **shared** label directory for that modality, and
`<modality>_<variant>_labels` is private to one variant. The pre-convention
names `labels`, `labels_fib` and `labels_FIB` are read indefinitely;
`migrate-layout` renames them.

**Resolution is most-specific-first.** An image directory takes its own
`<image-directory>_labels` when that directory exists, and otherwise falls back
to the modality's shared `<modality>_labels` (then to the pre-convention names).
So several conditions share a label *by default*, and giving one its own label
is done by creating the directory -- `sem_fi1_labels/` -- not by configuring
anything. `sem/` is not a special case: its private and shared names coincide.

An image directory named `<modality>_<variant>` holds one **imaging condition**
of an acquisition -- `sem_5kv`, `sem_lowmag`, `SEM_8_fi_16`. The variant string
is opaque and never parsed, so any naming works. Conditions sharing a label are
one scene, so a workflow state change follows the label and moves all of them
together.

A label file existing is **not** evidence of annotation: the segmentation
software writes an empty one for every image in its input folder. Commands that
treat a label as finished work look inside it.

CLI
---
Installing the package provides an `am-seg` command. Everything below can also
be run as `python -m am_segmentation_manager.cli`, which needs no installation
and is the form to use when working from a source checkout.

Use the built-in help to discover available commands and options.

Top-level help:

  am-seg --help

Command-specific help:

  am-seg <command> --help

Commands by purpose:

| Purpose | Command |
| --- | --- |
| See how a directory tree is interpreted, and what does not fit | `inspect-layout` |
| Make a pre-Autolamella dataset indexable, and record metadata | `init-project` |
| See each project's metadata and where each value came from | `show-metadata` |
| Set or clear metadata fields | `edit-metadata` |
| Rename pre-convention label directories | `migrate-layout` |
| List discovered projects and their counts | `list-projects` |
| Turn predictions into labels for proofreading | `convert-predictions` |
| Choose what gets proofread | `select` / `deselect` / `correct` |
| Fix images that disagree with their shared label | `repair-state` |
| Count images by one or two properties | `stats` |
| Export per-image metadata | `export-metadata-csv` |
| Export per-project metadata | `export-projects-csv` |
| Export training lists | `create-training-csv` / `check-training-csv` |
| Shrink label files on disk | `compress-labels` |

`inspect-layout` is read-only and the place to start on unfamiliar data.
Anything that moves or renames files takes a preview flag -- `--dry-run`, or
`--apply` for `migrate-layout`, which previews unless told otherwise.

Curation window
---------------
A desktop window for reviewing SEM and FIB images against their labels and
setting each image's workflow state:

  python -m am_segmentation_manager.cli gui [START_PATH]

`START_PATH` is optional; the folder can also be chosen in the window, and the
last one used is remembered. Reviewing is safe by default: the state controls
stay locked until *Enable editing* is ticked.

The window shows one lamella at a time, as a single chronological stack of
frames across all of its adaptive milling stages.

The timeline at the bottom shows curation progress for the whole lamella, SEM
on the top row and FIB below. A thin line outside each row shows that
modality's labels: violet where a label holds an annotation, grey where it is
an empty placeholder, and nothing where there is no label file. An amber wash
over a frame means its images disagree with the label they share -- the state
drawn for it is not the whole truth until `repair-state` is run.

Where an acquisition was captured under several imaging conditions, a
**Variant** dropdown appears beside the view buttons. It is a view control
only: a state change follows the label, so images sharing one always move
together whichever is on screen. The line above the state buttons says how far
the next change reaches, and names how many images a shared label pulls in.

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
  - layout.py        : **the filesystem conventions** -- what directories are called
  - discovery.py     : locating projects, lamellae and stages using those conventions
  - scanner.py       : building the Project/Lamella/Stage/ImageRecord model
  - survey.py        : read-only inspection and findings (`inspect-layout`)
  - models.py        : dataclasses for Project, Lamella, AdaptiveMillingStage, ImageRecord
  - metadata.py      : typed project metadata, resolved from the two YAML files
  - image_metadata.py: acquisition settings read from image headers
  - frames.py        : per-lamella chronological view pairing SEM with FIB
  - filtering.py     : matching records against CLI selectors and metadata filters
  - aggregate.py     : counting records by one or two properties
  - reporting.py     : human-readable summaries of projects and surveys
  - config.py        : segmentation class names and colours
  - colors.py        : CSS colour name to RGB
  - display.py       : downscaling, bit depth and label overlay composition
  - io/              : I/O helpers (yaml parsing, image reading, path relativising)
  - operations/      : everything that changes or exports data
    - selection.py       : workflow state transitions, validated as a batch
    - label_groups.py    : images sharing a label, and the state they share
    - label_io.py        : reading and writing label files
    - predictions.py     : prediction-to-label conversion
    - csv.py, csv_check.py : training list export and re-checking
    - label_compression.py : shrinking labels on disk
    - init_project.py    : AM_metadata.yaml templates
    - metadata_csv.py    : per-image and per-project metadata export
    - metadata_edit.py   : editing AM_metadata.yaml in place
    - migrate.py         : renaming pre-convention label directories
  - gui/             : curation window (only gui/widgets/ imports Qt)
  - cli.py           : CLI entrypoint
- tests/             : pytest tests using tmp_path fixtures
- docs/
  - 00_roadmap.md            : planning index and phase order
  - 01_refactor_plan.md      : cleanup checklist (consumable)
  - 02_gui_improvements_plan.md : planned curation window changes
  - 03_metadata_plan.md      : metadata reading, editing, export and statistics
  - 04_future_work.md        : sketches, not designs
  - reference_layouts.md     : real path examples, and the axes they vary along
  - design_record_gui.md     : why the curation window is built as it is
  - design_record_layouts.md : why legacy layouts are supported as they are

Notes for developers
--------------------
**`layout.py` is the single source of truth for filesystem conventions.**
Nothing else should embed a directory name. Supporting a new layout should be a
change there, not a change spread across the modules that happen to touch
files. See the Package Philosophy section of CLAUDE.md for why.

The layers are:

    layout.py      what things are called
    discovery.py   where they are
    scanner.py     what they mean (the model)
    operations/    what you can do to them

Higher layers depend on lower ones, never the reverse. Code outside
`scanner.py` and `survey.py` should operate on the ProjectIndex and the
dataclasses rather than re-traversing the filesystem.

Two invariants worth knowing before changing anything in `operations/`:

- **The label is the unit of workflow state.** Images sharing a label are one
  scene, so a state change expands across the whole group before it moves
  anything. `transition_records` validates the entire batch first and never
  overwrites; `plan_transition` returns that validation without acting.
- **Corrected labels are user-generated work.** Nothing moves them without an
  explicit confirmation flag, in a dry run as much as a real one.
