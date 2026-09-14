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
6. Export filtered CSV files containing per-project and per-image metadata for deposition of the data in scientific repositories
7. Export filtered CSV files with image and label file paths for ML model training

The package will initially be used by me, and will be made open source in the future.
Code quality, maintainability and extensibility therefore matter.

The work runs in four phases: add data to the collection, curate it and predict,
proofread in the window, then quantify and export. `README.md` catalogues the commands.
`am-seg gui` is the primary interface; the command line is a backup, for rarely needed
commands and for testing. `scripts/` holds standalone tools — environment setup, and
one-off jobs on legacy data — each explaining itself in its own docstring; none of them
is part of the workflow.

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

Current development state: `docs/00_roadmap.md` says what is built and what to pick
up next. It is the one place that answers this.

# Package Philosophy

**The filesystem is the authoritative source of truth for the state of the data.**
Derive the index and state from files on disk rather than from a separate database or
hidden persistent state; persist any state the package introduces to disk in an
appropriate format; and keep the package able to reconstruct its understanding of a
project by scanning it.

Do not introduce architectural abstractions, databases, frameworks or dependencies
without a concrete requirement. Prefer the simplest architecture that preserves
future extensibility.

The core is four layers — `layout.py` (what things are called), `discovery.py` (where
they are), `scanner.py` (what they mean), `operations/` (what you can do to them).
Higher layers depend on lower ones, never the reverse; `io/` sits below all four.
**Nothing imports `gui/`**: every decision lives below the widgets, so whatever the
window can do the command line can do, and the core installs without Qt. **Nothing
outside `ml/` imports it at module level**: the main environment has no torch, and the
window, the command line and the prediction child each import it inside the function
that needs it. Everything else supports that spine, and **its module docstring says
how** — whether it carries data between the layers, reads an index without writing to
it, or is a surface onto one. Resolve names through `layout.py` rather than embedding
a path of your own, and outside `scanner.py` and `survey.py` work on the ProjectIndex
and the dataclasses rather than re-traversing the filesystem.

---

# Coding Standards

Use pathlib rather than os.path, type hints throughout, and dataclasses for domain
models where they are the simplest suitable representation — not merely for the sake
of using them.

Separate filesystem access from higher-level operations, so behaviour can be tested
without building a directory tree.

Prefer simple, readable code over clever code. Avoid "quick solutions" that will
become difficult to maintain.

# Dependencies

Keep dependencies minimal and prefer the standard library. Do not introduce an
additional dependency without explaining why it is beneficial and asking permission.

# Testing

Every module should be testable. Suggest pytest tests for every major function. Avoid
writing code that is difficult to test because of tightly coupled filesystem access.

Use as testing environments:
am_segmentation_manager (without ml)
am_segmentation_manager_ml (with ml)

Run the suite from the repository root in one of those two. `docs/decisions.md`
§ Environment says what running it anywhere else looks like, which is worth reading
before believing a failure.

# Documentation

CLAUDE.md is ONLY for the scientist to edit. Suggestions can be made though.

Add user guidance for every command using a `--help` flag.

**Docstrings state what the code does now and what must stay true.** Every public class
and function gets one: a summary line, `Args:`/`Returns:`/`Raises:` only where the
signature does not already say it, and at most one sentence of constraint or invariant.
A module docstring orients a reader in a few sentences — what the module is for, and any
invariant holding across it — and stays under ten lines.

**Do not put in a docstring or a comment:** rejected alternatives, what an earlier
version did, benchmark results, plan or phase references, or anything a reader could get
from `git log`. Those belong in the commit message, where they are indexed by the change
that motivated them. A measured number stays only when it is a live design parameter — a
cache limit, a memory budget. If a comment restates the code, delete it.

**`docs/` describes the software now, not how it got here.** Correct a document rather
than appending a correction to it, and delete a plan when the work it led to lands.

**Do not use Claude Code memory** (it is switched off in `.claude/settings.json`). A
decision or plan that must outlast a session is written where it can be read: a plan in
`plans/`, which git ignores, until its work lands; a decision in `docs/`.

**A fact belongs in exactly one place**, and `docs/` holds four files. Adding a fifth is
a decision, not a default.

| File | Holds |
|---|---|
| `00_roadmap.md` | What the software does today, and what to pick up next |
| `decisions.md` | Why the built software is the way it is, and environment traps |
| `display.md` | Why the window is the way it is, and the Qt traps behind it |
| `04_future_work.md` | What is not built, and the questions to answer before designing it |

Place a fact before writing it: how one module behaves goes in that module's docstring;
why the design is this way goes in `decisions.md`, or `display.md` for anything under
`gui/`; work not yet built goes in `04_future_work.md`; how to install or use the package
goes in `README.md`. If nothing fits, say so and ask rather than starting a document.

**Read a document by section, not whole.** `grep -n "^## " docs/decisions.md` lists its
sections with line numbers; `sed -n` then reads the one you need.

**What `docs/` may say, and what may be taken out of it, is held by
`tests/test_docs.py`** — every entry names a test or a symbol, an entry is removed only
for one of the reasons on `MAX_DOC_LINES`, and never to make room for something new. A
document over that ceiling with none of those reasons available is a question for me.

**Checking the documentation is part of finishing a feature, not a step after it.**
Before reporting work done, re-read whatever documents mention what you changed and
correct what is no longer true, status lines included.

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

**A label file existing is not evidence of annotation.** The external segmentation
software writes an empty label for every image in its input folder. Anything treating
a label as finished work must look inside it: an empty label exported as a training
pair teaches a model that the structure is background, which is worse than omitting
the image.