# Changelog

What changed for a user in each release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). For this package a breaking change
is one to the files it writes: the state, metadata and history files beside
your data, and the columns of the CSVs it exports.

## [Unreleased]

The first public release, to be published as 0.1.0.

### Added

**Adding data to the collection**

- Indexes Autolamella projects by their `experiment.yaml`, across every
  directory layout back to 2023, without reorganising anything on disk.
- Datasets from before Autolamella are made indexable with an
  `AM_metadata.yaml`, proposed in the window by *Add projects* or written by
  `init-project`.
- Every image Autolamella writes is indexed under one of four series:
  Adaptive Polishing, reference, alignment and other.
- `inspect-layout` reports, without changing anything, how a folder is read
  and which images fit no naming rule.
- Project metadata is shown with where each value came from, and corrected
  or extended in the window or with `edit-metadata`.

**Curating and predicting**

- A window, `am-seg gui`, with four tabs in workflow order: Add data, Curate,
  Label and Export.
- Each label group is in exactly one state, selected, deselected or
  corrected, recorded in an `AM_state.yaml` beside the images. Several
  imaging conditions sharing one label change state together.
- Nothing that writes is unlocked until *Enable editing* is ticked.
- Predicts labels with a trained model on a GPU or the CPU, and converts the
  predictions into labels ready to proofread. The models published with
  Adaptive Milling ([doi:10.5281/zenodo.21804785](https://doi.org/10.5281/zenodo.21804785))
  run as generation 1.
- The Curate tab draws each alignment run's summary figure, and each grid
  overview with its lamellae marked on it.

**Proofreading**

- The Label tab proofreads one lamella's selected images at full resolution,
  holds the edits until they are saved, and offers to mark the reviewed set
  corrected.
- A speck finder rings components too small to be likely real, in the window
  and with `find-specks`.
- A proofreading checklist on the Label tab walks each label through the
  steps of proofreading it, with a short explanation of each. C ticks the
  current step, and a label ticked through can be marked corrected without
  an edit. Progress is kept only while the window is open.
- `compress-labels` shrinks label files on disk without changing a pixel.

**Quantifying and exporting**

- `stats` counts images by one or two properties and charts them.
- `quantify-labels` measures what the labels contain, in µm².
- Exports per-image and per-project metadata as CSV for deposition, and
  image and label paths as CSV for training a model, as one train/validation
  split or as K folds for cross-validation. An empty label is never exported
  as a training pair, and no split puts one lamella on both sides, or one
  acquisition or one project where asked.
- `export-files` copies a selection of images and labels into a folder of
  its own, and `merge` brings the worked-on copy back.

**History**

- Records what changed, who changed it and when, and keeps every replaced
  label under `AM_archive/`.
- Savepoints name the state of the data. `revert` puts labels back to an
  earlier version.
- `merge` combines a worked-on copy three-way, with its conflicts decided
  one at a time or a folder, project or lamella at once.

**Safety**

- No automated operation overwrites or deletes a corrected label.
- A batch operation checks every change before it makes the first one.
  Commands that write can show what they would do first: most take
  `--dry-run`, and `merge`, `revert`, `savepoint` and `prune-archive` only
  report until given `--apply`.

**Platforms**

- Windows, macOS and Linux, on Python 3.11 to 3.13.
