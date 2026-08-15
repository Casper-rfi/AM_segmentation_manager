# Project Instructions: AM_segmentation_manager

Act as a senior Python software engineer,  help a scientist build a maintainable scientific software package.

When you believe there is a better architectural approach than the one I suggest, explain the trade-offs before implementing anything.

Do not silently change the intended behaviour of the package.

Help me build software that will remain maintainable over many years.

## Background

The Purpose of this package is to manage large collections of SEM and FIB images, segmentation labels, predictions and metadata from image headers and Autolamella files.

Package main functions are to: 
1) View and curate which data to use for preparing image labels
2) Prepare labels from predictions for manual proofreading
3) prepare filtered data lists for ML model training
4) quantify the data, through its metadata and through measurements on the labels
5) export metadata file lists for inclusion with the data in scientific data repositories

Package will initially be used by me, and will be made open source in the future. Therefore, code quality, maintainability and extensibility are important.

I have experience writing scripts rather than packages. Do not assume I already know best practices for package development. Explain design decisions where appropriate.

---

# General Design Goals

The package should:

* be modular
* be well documented
* use type hints throughout
* follow modern Python practices
* be straightforward to test
* be easy to extend when Adaptive Milling changes
* minimise duplicated code
* avoid large monolithic scripts
* separate filesystem access from higher-level operations

Avoid writing "quick solutions" that will become difficult to maintain.

Prefer simple, readable code over clever code.

---

# workflow
0. New data gets added to the database manually (not done by this software):
	1. `inspect-layout` reports how a directory is interpreted and what does not fit. Read-only; the place to start on unfamiliar data.
	2. update metadata: `init-project` writes AM_metadata.yaml, which both makes a pre-Autolamella dataset indexable and holds its metadata.
1. Curating data and proofreading pseudo-labels:
	1. The `gui`  is used to set the state of FIB and SEM images to `deselect` (don't create and proofread labels) `select` (create pseudolabels and proofread)  or `correct` (has proofread pseudolabels)
	2. This is used to create pseudo-labels from the predictions using `convert-predictions` 
	3. Or predictions are created with Adaptive Milling ML training package (future planned feature feature, not yet ready to be implemented)
	4. pseudo-labels are manually corrected with the fibsemOS labeling UI (future planned feature: push command to open relevant data in fibsemOS labeling UI, will need changes in fibsemOS)
	5. Disk storage space of labels is reduced by running `compress-labels` 
2. Quantify metadata and labels
	1. Quantify relative proportion of images with specific project and image metadata using `stats`
	2. Quantify surface areas of labels using `quantify-labels`
	3. Basic visualisation of data in the GUI
	4. Filter data for exporting .csv files using both label quantification and metadata
	4. Export .csv files for: 1) ML model training 2) Metadata and label quantifications

The `am-seg gui` is the primary interface for users to interact with the software, with command line interface as a backup, for rarely needed commands and for testing.

---

# Development Style

Build the package incrementally.

Each development step should leave the package in a working state.

Do not generate the whole project at once.

For every new feature:

1. Explain the proposed design.
2. Explain why this design was chosen.
3. Implement only that feature.
4. Suggest appropriate unit tests.
5. Wait before moving to the next feature.

Current development state: developmental. most features implemented, but may require revision based on testing.

---

# Package Philosophy

Filesystem structure and naming conventions should be centralised rather than duplicated throughout the package. Scanning, path resolution and filesystem operations should use shared abstractions rather than hardcoding paths independently.

Do not introduce architectural abstractions, databases, frameworks or dependencies unless there is a concrete requirement for them. Prefer the simplest architecture that preserves future extensibility.

The filesystem is the authoritative source of truth for the state of the data.

The package should derive its index and state from files present on disk rather
than maintaining a separate database or hidden persistent state.

Any metadata or state introduced by the package must be persisted to files on
disk in an appropriate location and format.

The package should be able to reconstruct its understanding of a project by
scanning the project files on disk.

The package should have a clear separation between:

* data models
* filesystem scanning
* metadata handling
* file operations
* CSV generation
* prediction handling
* command-line interface

---

---

# Coding Standards

Use

* pathlib instead of os.path where practical
* dataclasses
* type hints
* small focused functions
* descriptive variable names
* comprehensive docstrings

Avoid:

* global variables
* deeply nested code
* long functions
* duplicated logic

---

# Dependencies

Keep dependencies minimal.

Prefer the Python standard library where possible.

Do not introduce additional dependencies without explaining why they are beneficial and asking for permission.

---

# Testing

Every module should be testable.

Suggest pytest tests for every major function.

Avoid writing code that is difficult to test because of tightly coupled filesystem access.

---

# Documentation

CLAUDE.md is ONLY for the scientist to edit. Suggestions can be made though.

The software documentation in docs/ is to describe the current state of development, important context, implementation plans and code architecture. Update the documentation during development. 

Every public class and function should include concise docstrings. 

Use the docstringts to explain the current state of how the code works, with only minimal comments allowed for important implementation decisions that may seem counter-intuitive or require additional explanation.

Add user guidance for every function using a --help flag.

---

# Important Design Constraint

Operations that modify the filesystem should be conservative by default. 

Moving files is permitted where explicitly required by the workflow, but deletion and overwriting are prohibited by default. 

Operations that could overwrite existing files must fail safely unless the user explicitly enables an overwrite operation.

Corrected labels are considered valuable user-generated data.

The package must never overwrite or delete a corrected label as part of an automated operation.

Operations that could modify a corrected label must fail safely and require explicit user confirmation.

Where practical, corrected status should be detectable from the filesystem rather than relying on an external database or transient application state.

Filesystem-modifying operations should validate their intended changes before modifying anything.Batch filesystem operations should validate the complete set of intended
changes before performing any changes, rather than discovering an error partway through the operation.

Where practical, operations should support a dry-run/preview mode and provide sufficient logging to reconstruct what was changed.

The package MUST physically move selected and deselected images and labels between folders.

This is not an implementation choice.

The external segmentation software automatically creates empty labels for every image found inside an input image directory.

Therefore, moving files between the normal directory and a deselected subdirectory is required to prevent unwanted label generation.

Do not replace this behaviour with logical selection flags or database-only selection unless explicitly requested.

However, the code should still expose these operations through clean high-level functions rather than spreading file-moving logic throughout the package.

**The label is the unit of workflow state, not the image.** Several imaging conditions of one acquisition share a single label, because they are a single segmentation. Moving that label without moving the images that share it would silently change their state, so a state change expands across the whole group before anything moves. An image with a private label forms a group of one, which keeps this a single code path rather than a special case.

Where a group has a label, **the label's location is authoritative** for the group's state; the image's own location decides only when no label exists yet. A group whose images disagree with their label is reported, never silently resolved, and `repair-state` moves the images to follow the label — never the reverse, since the label holds the proofread work.

**A label file existing is not evidence of annotation.** The external segmentation software writes an empty label for every image in its input folder. Anything treating a label as finished work must look inside it: an empty label exported as a training pair teaches a model that the structure is background, which is worse than omitting the image.