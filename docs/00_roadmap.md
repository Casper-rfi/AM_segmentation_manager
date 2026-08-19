# Roadmap

What the software does today, and what to pick up next. This is the entry point for
picking the work up fresh. `CLAUDE.md` lists the other documents and holds the working
agreement; `README.md` is the user and developer entry point, and its command tables
are the catalogue of what each command does.

---

## What the software does today

Four of the five main functions in `CLAUDE.md` are built: curating which data to use,
preparing labels from predictions, exporting filtered training lists, and quantifying
the collection through its metadata and its labels. `README.md` says what each command
and tab does. What is worth knowing beyond that:

* **Every state change expands across the label group** and validates the whole batch
  before moving anything. Marking a set corrected, and exporting a training CSV, both
  stop to name any image whose label holds no annotation.
* **Each tab has a controller holding no Qt**, which is what makes navigation, cache
  eviction and state expansion testable without a display.
* **An operation is in the window when its input is the selection on screen.** Curate
  runs `convert-predictions`, `compress-labels` and `repair-state` against the ticked
  set; Add data runs `inspect-layout` and `init-project` over the scanned folder. Each
  previews first and applies only what the preview showed, on a worker thread.
* **Labels are proofread in the window**, not in external software. The Label tab
  reads one lamella's selected and corrected frames at full resolution, paints with a
  brush and a flood fill, and saves the lamella's edits in one action that offers to
  mark the reviewed set corrected. A save refuses the whole batch rather than
  overwrite a label something else has rewritten since it was opened. On request
  it rings the annotated components of a few pixels -- the stray spots that
  otherwise reach a training set as annotation -- which `find-specks` reports
  across the whole collection.
* **Export filters, counts and writes, and every filter narrows all three.** Its filter
  lists are derived from the breakdowns, so a property can be counted by and filtered
  on without being declared twice; the breakdown reads as a table or a chart in the
  same pane, and the crack threshold and area bin resolution sit beside the button
  that measures.

Why each of these is so is in [`decisions.md`](decisions.md).

---

## What to pick up next

In no particular order.

**Importing existing labels** into the project hierarchy, once the source structures
have been looked at. [`04_future_work.md`](04_future_work.md) §4.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py`
builders when editing them for other reasons, never as a sweep. Three files still
carry hand-built trees: `test_selection.py`, `test_selection_state.py` and
`test_cli_selection_integration.py`.

**Before publishing**, several packaging gaps are open by decision rather than
oversight, and all of them become necessary at the same moment:

* No `LICENSE`. `pyproject.toml` deliberately declares none and carries a
  `"Private :: Do Not Upload"` classifier.
* No CI, no `CHANGELOG.md`, and no `[project.urls]`.
* The version is pinned at `0.0.0`.
* `mypy` is not configured. `ruff` does not check types, so the annotations are
  currently unverified.
* The suite has not been run under a non-English locale. Do it before any release:
  [`decisions.md`](decisions.md) § Environment has what it catches, and why `LC_ALL=`
  is not sufficient. It does now pass under PySide6 6.11.1, which is what the
  development environment resolves to.
