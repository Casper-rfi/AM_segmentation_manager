# Project Instructions: AM_segmentation_manager

Act as a senior Python software engineer, helping a scientist build a maintainable
scientific software package.

When you believe there is a better architectural approach than the one I suggest,
explain the trade-offs before implementing anything.

Do not silently change the intended behaviour of the package.

I have experience writing scripts rather than packages. Do not assume I already know
best practices for package development. Explain design decisions where appropriate.

## Background

The purpose of this package is to manage large collections of SEM and FIB images,
segmentation labels, predictions and metadata from image headers and Autolamella
files. Its main functions are to:

1. View and curate which data to use for preparing image labels
2. Prepare labels from predictions for manual proofreading
3. Prepare filtered data lists for ML model training
4. Quantify the data, through its metadata and through measurements on the labels
5. Export metadata file lists for inclusion with the data in scientific data
   repositories

The package will initially be used by me, and will be made open source in the future.
Code quality, maintainability and extensibility therefore matter.

---

# Workflow

0. New data gets added to the collection manually (not done by this software):
	1. `inspect-layout` reports how a directory is interpreted and what does not fit. Read-only; the place to start on unfamiliar data.
	2. `init-project` writes AM_metadata.yaml, which both makes a pre-Autolamella dataset indexable and holds its metadata.
1. Curating data and proofreading pseudo-labels:
	1. The `gui` is used to set the state of FIB and SEM images to `deselect` (don't create and proofread labels), `select` (create pseudolabels and proofread) or `correct` (has proofread pseudolabels)
	2. `convert-predictions` creates pseudo-labels from the predictions
	3. Or predictions are created with the Adaptive Milling ML training package (planned, not yet ready to implement)
	4. Pseudo-labels are manually corrected in the fibsemOS labeling UI (planned: a push command to open the relevant data there, which will need changes in fibsemOS)
	5. `compress-labels` reduces the disk space labels occupy
2. Quantify metadata and labels:
	1. `stats` quantifies the proportion of images with given project and image metadata
	2. `quantify-labels` measures label surface areas
	3. The GUI provides basic visualisation
	4. Both label quantification and metadata filter the data for export
	5. Export .csv files for ML model training, and for metadata and label quantifications

`am-seg gui` is the primary interface. The command line is a backup, for rarely
needed commands and for testing.

---

# Development Style

Build the package incrementally. Each development step should leave the package in a
working state. Do not generate the whole project at once.

For every new feature:

1. Explain the proposed design.
2. Explain why this design was chosen.
3. Implement only that feature.
4. Suggest appropriate unit tests.
5. Wait before moving to the next feature.

Current development state: developmental. Most features implemented, but may require
revision based on testing.

---

# Package Philosophy

Filesystem structure and naming conventions are centralised rather than duplicated.
Scanning, path resolution and filesystem operations use shared abstractions rather
than hardcoding paths independently.

**The filesystem is the authoritative source of truth for the state of the data.**
Derive the index and state from files on disk rather than from a separate database or
hidden persistent state; persist any state the package introduces to disk in an
appropriate format; and keep the package able to reconstruct its understanding of a
project by scanning it.

Nothing is inherited from parent directories. A directory can hold several projects
prepared on different samples, so resolving a field from a parent risks attributing
the wrong sample to a project — an error that would be invisible in the output.

Do not introduce architectural abstractions, databases, frameworks or dependencies
without a concrete requirement. Prefer the simplest architecture that preserves
future extensibility.

The core is four layers — `layout.py` (what things are called), `discovery.py` (where
they are), `scanner.py` (what they mean), `operations/` (what you can do to them).
Higher layers depend on lower ones, never the reverse; `io/` sits below all four.
Outside `scanner.py` and `survey.py`, operate on the ProjectIndex and the dataclasses
rather than re-traversing the filesystem. Keep data models, filesystem scanning,
metadata handling, file operations, CSV generation, prediction handling and the CLI
separated.

---

# Coding Standards

Use pathlib rather than os.path, type hints throughout, and dataclasses for domain
models where they are the simplest suitable representation — not merely for the sake
of using them.

Separate filesystem access from higher-level operations, so behaviour can be tested
without building a directory tree.

Prefer simple, readable code over clever code. Avoid "quick solutions" that will
become difficult to maintain.

---

# Dependencies

Keep dependencies minimal and prefer the standard library. Do not introduce an
additional dependency without explaining why it is beneficial and asking permission.

---

# Testing

Every module should be testable. Suggest pytest tests for every major function. Avoid
writing code that is difficult to test because of tightly coupled filesystem access.

---

# Documentation

CLAUDE.md is ONLY for the scientist to edit. Suggestions can be made though.

Add user guidance for every command using a `--help` flag.

**Docstrings state what the code does now and what must stay true.** Every public
class and function gets one: a summary line, `Args:`/`Returns:`/`Raises:` only where
the signature does not already say it, and at most one sentence of constraint or
invariant.

**Do not put in a docstring or a comment:** rejected alternatives, what an earlier
version did, benchmark results, plan or phase references, or anything a reader could
get from `git log`. Those belong in the commit message, where they are indexed by the
change that motivated them. A measured number stays only when it is a live design
parameter — a cache limit, a memory budget — not as a record that measuring happened.
If a comment restates the code, delete it.

**`docs/` describes the current state of the software and the work planned next.** It
is not a development history. When work completes, update the document to describe
what the software now does and delete the plan that led there; do not leave a
finished plan behind annotated as done. Correct a document rather than appending a
correction to it.

A fact belongs in exactly one place. Before writing something down, check whether a
module docstring, the README or `docs/` already says it, and update that instead.

---

# Important Design Constraint

Operations that modify the filesystem are conservative by default. Moving files is
permitted where the workflow requires it; **deletion and overwriting are prohibited
by default**, and an operation that could overwrite must fail safely unless the user
explicitly enables overwriting.

**Corrected labels are valuable user-generated data.** No automated operation may
overwrite or delete one. An operation that could modify a corrected label must fail
safely and require explicit user confirmation. Where practical, corrected status
should be detectable from the filesystem rather than from an external database or
transient application state.

Batch filesystem operations validate the complete set of intended changes before
performing any of them, rather than discovering an error partway through. Where
practical, operations support a dry-run/preview mode and log enough to reconstruct
what was changed.

**The package MUST physically move selected and deselected images and labels between
folders.** This is not an implementation choice: the external segmentation software
automatically creates an empty label for every image found inside an input image
directory, so moving files into a `deselected/` subdirectory is what prevents
unwanted label generation. Do not replace this with logical selection flags or
database-only selection unless explicitly requested. Expose it through clean
high-level functions rather than spreading file-moving logic through the package.

**The label is the unit of workflow state, not the image.** Several imaging
conditions of one acquisition share a single label, because they are a single
segmentation. Moving that label without moving the images that share it would
silently change their state, so a state change expands across the whole group before
anything moves. An image with a private label forms a group of one, which keeps this
a single code path rather than a special case.

Where a group has a label, **the label's location is authoritative** for the group's
state; the image's own location decides only when no label exists yet. A group whose
images disagree with their label is reported, never silently resolved, and
`repair-state` moves the images to follow the label — never the reverse, since the
label holds the proofread work.

**A label file existing is not evidence of annotation.** The external segmentation
software writes an empty label for every image in its input folder. Anything treating
a label as finished work must look inside it: an empty label exported as a training
pair teaches a model that the structure is background, which is worse than omitting
the image.