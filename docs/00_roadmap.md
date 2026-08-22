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
* **What a change applies to is one level at a time.** Ticking a project clears the
  ticked lamellae, picking frames out clears both, and ticked lamellae belong to the
  project on screen and go when it changes — so what is ticked on screen is the whole
  of what a state change or an operation will touch.
* **Where an operation sits says what it acts on.** Curate runs `predict` and
  `convert-predictions` against the ticked set, because choosing which images to
  predict is the decision; Add data runs `inspect-layout`, `init-project` and
  `compress-labels` over the whole scanned folder, because none of the three has a
  reason to be narrower. Each previews first and applies only what the preview
  showed, on a worker thread, and what was ticked is still ticked when it
  finishes -- so predicting and then converting run against one selection.
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
  which is what earns it a Stop button. **The window runs it in a child process** and
  imports no torch itself; which device to use is chosen beside the model.
* **Export filters, counts and writes, and every filter narrows all three.** A filter is
  added when it is wanted: each carries a dropdown of the properties still free, the
  values of the one it is on, and a button taking it away along with what it kept. The
  choice of property is derived from the breakdowns, so a property can be counted by and
  filtered on without being declared twice — bar a few that can be counted by but not
  usefully chosen from. Values are ticked rather than highlighted, with the count beside
  each list and one button that unticks them all. The breakdown reads as a table or a
  chart in the same pane, and the crack threshold and area bin resolution sit beside the
  button that measures. **Which modality is chosen decides the shape of what is written**:
  one beam gives one row per image, both give one row per acquisition with the beams side
  by side, and the training and per-image-metadata exports agree on it -- as does the
  command line, so one selection exported either way gives the same file.
* **A project is a directory; `project` is what the work was part of.** The directory
  the scanner indexes and Curate lists is `autolamella_project` in every export; the
  metadata field called `project` names the wider project that several of those
  directories usually share. Both can be counted and filtered on, and nothing derives
  one from the other — the folder name is a fact about storage, and which project a
  dataset was collected for is a question only a person can answer.

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
