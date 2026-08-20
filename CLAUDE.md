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

1. Correct and extend project-based metadata
2. View and curate which data to use for preparing image labels
3. Prepare labels from predictions for manual proofreading
4. Manually Proofread and correct labels and create new labels
5. Quantify the data, through its project and image metadata and through measurements on the labels
5. Export filtered CSV files containing per-project and per-image metadata for deposition of the data in scientific repositories
6. Export filtered CSV files with image and label file paths for ML model training

The package will initially be used by me, and will be made open source in the future.
Code quality, maintainability and extensibility therefore matter.

---

# Workflow

0. New data gets added to the collection manually (not done by this software):
	1. `inspect-layout` reports how a directory is interpreted and what does not fit. Read-only; the place to start on unfamiliar data.
	2. `init-project` writes AM_metadata.yaml, which both makes a pre-Autolamella dataset indexable and holds its metadata.
	3. `edit-metadata` is used to add any missing metadata or make corrections
	4. import legacy projects (not yet implemented)
	5. `compress-labels` reduces the disk space labels occupy of legacy projects (still in GUI under curate)
1. View & curate data and create predictions:
	1. The `gui` is used to set the state of FIB and SEM images to `deselect` (don't create and proofread labels), `select` (create pseudolabels and proofread) or `correct` (has proofread pseudolabels)
	2. Create predictions (planned, not yet implemented)
	3. `convert-predictions` creates pseudo-labels from the predictions
2. create labels & correct predictions manually in the GUI
3. Quantify metadata and labels & export:
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

A module docstring orients a reader in a few sentences: what the module is for, and any invariant holding across it. Keep it under ten lines. 
Detail about one function belongs on that function; reasoning about why the design is this way belongs in docs/decisions.md.

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

**`docs/` holds four files. Adding a fifth is a decision, not a default.**

| File | Holds |
|---|---|
| `00_roadmap.md` | What the software does today, and what to pick up next |
| `decisions.md` | Why the built software is the way it is, and environment traps |
| `04_future_work.md` | What is not built, and the questions to answer before designing it |
| `reference_layouts.md` | Real dataset layouts, and the axes they vary along |

Place a fact before writing it: how one module behaves goes in that module's
docstring; why the design is this way goes in `decisions.md`; work not yet built
goes in `04_future_work.md`; how to install or use the package goes in
`README.md`. If nothing fits, say so and ask, rather than starting a new document.

**Checking the documentation is part of finishing a feature, not a step after
it.** Before reporting work done, re-read whatever documents mention the thing
you changed and correct what is no longer true — including status lines and
anything that describes the old behaviour. Documentation goes stale because
nobody re-reads it, not because anyone decides to leave it wrong.

**Keep `docs/` short enough to read in one sitting — around 800 lines in total.**
Approaching that ceiling means something in there has finished and should be
deleted, not that the limit needs raising.

---

# Important Design Constraint

Operations that modify the filesystem are conservative by default. Moving files is
permitted where the workflow requires it; **deletion and overwriting are prohibited
by default**, and an operation that could overwrite must fail safely unless the user
explicitly enables overwriting.

**Corrected labels are valuable user-generated data.** No automated operation may
overwrite or delete one. An operation that could modify a corrected label must fail
safely and require explicit user confirmation. Where practical, corrected status
should be detectable from a file within the dataset rather than from an external 
database or transient application state.

Batch filesystem operations validate the complete set of intended changes before
performing any of them, rather than discovering an error partway through. Where
practical, operations support a dry-run/preview mode and log enough to reconstruct
what was changed.

**A group is in exactly one state.** selected, deselected and corrected are mutually 
exclusive. The state file maps each group to exactly one of them, and any representation 
that could express two at once is wrong. Deselecting a group that was corrected 
discards the corrected claim; the proofreading itself survives in the label's contents, 
which is what an export checks anyway. Expose it through clean high-level functions 
rather than spreading knowledge of how state is stored through the package. One 
function answers what state a record is in and one operation changes it; nothing 
else reads or writes the state file, and nothing re-derives a record's state for itself.

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