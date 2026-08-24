# Roadmap

What the software does today, and what to pick up next — the entry point for picking the
work up fresh. `CLAUDE.md` holds the working agreement; `README.md` is the user and
developer entry point, and its command tables are the catalogue of what each command does.

---

## What the software does today

All five main functions in `CLAUDE.md` are built. `README.md` says what each command and
tab does, and [`decisions.md`](decisions.md) why each of the following is so:

* **Three series of images are indexed** — Adaptive Polishing, the reference images
  around each milling task, and the alignment runs — an axis apart from modality and
  imaging condition. The last two sit *loose* in a directory, so an image set is a group
  of filenames rather than a directory, ordered by the acquisition time in each header.
  `--series` chooses what a command covers: the reporting commands cover all three,
  everything that writes stays on Adaptive Polishing.
* **Workflow state is declared in each stage's `AM_state.yaml`**, not implied by which
  folder a file sits in. A change expands across the label group, validates the whole
  batch, and writes one entry. Marking a set corrected, and exporting a training CSV, both
  stop to name any image whose label holds no annotation.
* **The window is four tabs, each with a controller holding no Qt.** What a change applies
  to is one level at a time, so what is ticked on screen is the whole of what an operation
  will touch. Curate and Label each show one series at a time, and Curate runs `predict` and
  `convert-predictions` against the ticked set; Add data runs `inspect-layout`,
  `init-project` and `compress-labels` over the whole scanned folder. Each previews first,
  applies on a worker thread, and leaves the selection intact.
* **Labels are proofread in the window**, not in external software. One lamella and one
  series at a time. A save writes that lamella's edits in a single action, offers to mark
  the reviewed set corrected, and refuses the whole batch rather than overwrite a label
  something else has rewritten since it was opened. It will also ring the components of a
  few pixels, which `find-specks` reports across the collection.
* **Predictions need an environment installed separately**, built by
  `scripts/setup_ml.sh`, which runs the window too. One prediction per label group, one
  modality per run, resumable, in a child process the window shares no torch with.
* **Export filters, counts and writes, and every filter narrows all three.** Filters are
  added as they are wanted, derived from the breakdowns so a property is not declared
  twice. Opening the tab reads the image headers; opening the labels stays behind *Measure
  labels*. **Which modality is chosen decides the shape of what is written**, and the
  command line agrees, so one selection exported either way gives the same file.
* **A project is a directory; `project` is what the work was part of.** The directory is
  `autolamella_project` in every export; the metadata field names the wider project several
  directories usually share. Both can be counted and filtered on.

---

## What to pick up next

In no particular order.

**Importing existing labels** into the project hierarchy, once the source structures
have been looked at. [`04_future_work.md`](04_future_work.md) §4.

**Labelling a reference image** works as far as the layout goes — a label lands in
`<series>_<modality>_labels/` and its state is declared like any other — but nothing
predicts one: `_collect_predictions` gives a loose set none, because resolving would claim
the Adaptive Polishing pipeline's output in a dataset that collapses the stage level.
Predicting them needs the exact-name rule their labels have, and `layout.LABELS_EXPECTED`
widening so `inspect-layout` reports them as unlabelled.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py`
builders when editing them for other reasons, never as a sweep. `test_selection.py`
still carries a hand-built tree.

**Before publishing**, several packaging gaps are open by decision rather than
oversight, and all become necessary at the same moment: there is no `LICENSE`
(`pyproject.toml` deliberately declares none and carries a `"Private :: Do Not Upload"`
classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the version is pinned at
`0.0.0`. `mypy` is not configured, and `ruff` does not check types, so the annotations are
unverified. The suite has also not been run under a non-English locale — do that before any
release, for the reasons in [`decisions.md`](decisions.md) § Environment. It does now pass
under PySide6 6.11.1, which the development environment resolves to.
