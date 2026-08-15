# Package structure

The **core** is four layers, each depending only on the ones below it:

    layout.py      what things are called (the only module holding directory names)
    discovery.py   where they are (locating projects, lamellae, stages)
    scanner.py     what they mean (building the model)
    operations/    what you can do to them

The rest of the package sits around that core and obeys the same direction of
dependency — nothing below reaches upward:

    io/            reading and writing files (yaml, images, path relativising)
                   sits *below* layout.py; knows nothing of the model
    models.py      the dataclasses the scanner builds
    metadata.py    typed project metadata, resolved from the two YAML files
    config.py      segmentation class names and colours
    colors.py      CSS colour name to RGB

    filtering.py   selecting records from the index by CLI selectors
    frames.py      per-lamella chronological view, pairing SEM with FIB
    survey.py      read-only inspection and findings (`inspect-layout`)
    reporting.py   human-readable summaries
    display.py     downscaling, bit depth and label overlay composition

    gui/           the curation window. gui/controller.py holds the view model
                   and must stay free of Qt; only gui/widgets/ imports it
    cli.py         the entrypoint

Code outside `scanner.py` and `survey.py` should operate on the ProjectIndex and
the dataclasses rather than re-traversing the filesystem.

Update the description of the package structure here after large updates, when appropriate.

---

# Data Model

Prefer Python dataclasses for domain models where appropriate. Do not use dataclasses merely for the sake of using them; choose the simplest suitable representation for each type of data.

The objects, as built:

Project

* path, and **relative_path** — the path relative to the scan root, which is
  the project's identity. Names are not unique: the reference collection holds
  two different projects whose name, created_at and recorded path in
  experiment.yaml are all identical.
* metadata (the raw merged mapping) and experiment_metadata (the typed view)
* lamellae

Lamella

* stages
* synthetic — True when the dataset has no lamella directory and the project
  directory stands in for one

AdaptiveMillingStage

* images
* image_sets — the image directories in the stage, each with the label
  directory it resolves to and whether that label directory is shared
* synthetic — True when the dataset has no stage directory

ImageRecord

Each ImageRecord should contain paths to:

* image
* prediction(s)
* label(s)

plus metadata such as:

* project
* lamella
* stage
* timestamp
* modality (SEM/FIB)
* variant — the imaging-condition suffix of the image directory, or None for
  the unsuffixed one. An opaque string; never parse it.
* label_shared — whether other images reference the same label
* selected status

---

# Filesystem

The package must support Autolamella projects containing multiple lamellae.

Each lamella may contain multiple Adaptive Milling stages.

A **project** is a directory holding experiment.yaml or AM_metadata.yaml.
Datasets predating Autolamella have neither until AM_metadata.yaml is written
for them, which is what `init-project` does.

The lamella and stage levels are **not always directories**. Older datasets
collapse them (`project/lamella/SEM_8/`, or `project/SEM/`). The missing level
is synthesised in memory, with the parent standing in and the model recording
that it is synthetic. Data is never reorganised to fit the newest layout.

Each stage may contain:

* one or more SEM image directories, and one or more FIB image directories.
  A directory is named `<modality>` or `<modality>_<variant>`, where the
  variant is one imaging condition of the same acquisition (`sem_5kv`,
  `SEM_8_fi_16`).
* a label directory per image directory, named `<image-directory>_labels`.
  `<modality>_labels` is therefore the shared label directory for a modality,
  and several image directories resolving to it means several conditions of
  one scene share one label. The pre-convention names `labels`, `labels_fib`
  and `labels_FIB` are still read.
  Resolution is **most-specific-first**: an image directory takes its own
  `<image-directory>_labels` when that exists, and otherwise falls back to the
  shared `<modality>_labels`, then to the pre-convention names. Conditions
  therefore share a label by default, and giving one its own is done by
  creating `sem_fi1_labels/` rather than by configuring anything.
* predictions, in `predictions/` with optional `raw/` and `clean/`
* `deselected/` and `corrected/` subdirectories of any image or label
  directory, which is where workflow state lives

The scanner should discover these automatically.

Avoid hardcoding unnecessary assumptions. **Directory names belong in
`layout.py` and nowhere else** — see Package Philosophy. Supporting a new
layout should be a change there, not one spread across every module that
touches files.
This document contains text moved from CLAUDE.MD. The text text needs to be reviewed to decide whether the information is relevant to preserve in another document.

---

# Metadata

Project-level Metadata should come from two sources.

experiment.yaml

and

AM_metadata.yaml

AM_metadata.yaml should override values from experiment.yaml where both contain the same information. Resolution is **per key**, so a project can override one field without restating the rest, and each resolved value records which file supplied it.

Nothing is inherited from parent directories. Collection directories group projects by session or storage convenience, not by sample, and lamellae in one directory may be prepared on different samples — so resolving a field from a parent risks attributing the wrong sample to a project, an error that would be invisible in the output. Values shared between projects are copied into each file when `init-project` generates it, not resolved at scan time.

**The project is the right level for this metadata, because all lamellae in one Autolamella project come from one sample** (confirmed 2026-08-14). Per-lamella metadata was considered and rejected on that basis. Note the distinction from the paragraph above: lamellae in one *directory* may be different samples, because a directory can hold several projects; lamellae in one *project* may not.

Fields that vary between projects are never guessed. `init-project` writes them as `TODO` placeholders, and an unfilled placeholder is reported by `inspect-layout` rather than silently treated as a value.

Per-image metadata is extracted directly from the image headers by
`image_metadata.py`. Two dialects are read: the fibsem JSON in TIFF tag 270,
and the older FEI tag 34682. Where each value lives is declared once, in a
field table -- that module is to header keys what `layout.py` is to directory
names.

**Every image header carries settings for both beams.** Only the beam the
instrument reports as having taken the image is ever read; `system.*` holds
both beams' configured defaults and is not a source of measurements.

Headers are never read during scanning. Metadata is a mapping keyed by image
path, built by the commands that need it, so a scan stays a filesystem walk.

Metadata should be represented internally using dataclasses instead of raw dictionaries whenever practical.