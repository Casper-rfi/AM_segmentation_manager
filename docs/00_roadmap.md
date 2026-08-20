# Roadmap

What the software does today, and what to pick up next. This is the entry point for
picking the work up fresh. `CLAUDE.md` lists the other documents and holds the working
agreement; `README.md` is the user and developer entry point, and its command tables
are the catalogue of what each command does.

---

## What the software does today

All five main functions in `CLAUDE.md` are built: curating which data to use,
predicting and preparing labels from predictions, exporting filtered training lists,
and quantifying the collection through its metadata and its labels. `README.md` says
what each command and tab does. What is worth knowing beyond that:

* **Workflow state is declared in each stage's `AM_state.yaml`**, not implied by which
  folder a file sits in. A change expands across the label group, validates the whole
  batch, and writes one entry. Marking a set corrected, and exporting a training CSV,
  both stop to name any image whose label holds no annotation.
* **Each tab has a controller holding no Qt**, which is what makes navigation, cache
  eviction and state expansion testable without a display.
* **An operation is in the window when its input is the selection on screen.** Curate
  runs `convert-predictions` and `compress-labels` against the ticked set; Add data
  runs `inspect-layout` and `init-project` over the scanned folder. Each
  previews first and applies only what the preview showed, on a worker thread.
* **Labels are proofread in the window**, not in external software. A save writes one
  lamella's edits in a single action, offers to mark the reviewed set corrected, and
  refuses the whole batch rather than overwrite a label something else has rewritten
  since it was opened. On request it rings the components of a few pixels -- the stray
  spots that otherwise reach a training set as annotation, and the holes that size
  inside one -- which `find-specks` reports across the collection.
* **Predictions need an environment installed separately**: the inference stack is
  several gigabytes, so `scripts/setup_ml.sh` builds its own, which runs the window too.
  Writes go to `<predictions>/am_seg/`, the only prediction directory this package
  writes. One prediction per label group, one modality per run — a selection holding
  both is refused, since a model is for one and nothing can tell which — and resumable,
  which is what lets the window offer a Stop button no other operation here can.
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
builders when editing them for other reasons, never as a sweep. `test_selection.py`
still carries a hand-built tree.

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
