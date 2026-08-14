# Roadmap and planning index

Planning written 2026-08-14 against `main` at c9c05e2. **Phases 0–7 are built.**
This file is the entry point for picking the work up fresh.

| Doc | Covers | State |
|---|---|---|
| [`01_refactor_plan.md`](01_refactor_plan.md) | Cleanup checklist | §1–§4 done; §4.1 and §5 open |
| [`02_gui_improvements_plan.md`](02_gui_improvements_plan.md) | The four GUI changes | Done |
| [`03_metadata_plan.md`](03_metadata_plan.md) | Metadata: read, edit, aggregate, export, present | §3–§7 done; §8 specified, not built |
| [`04_future_work.md`](04_future_work.md) | Segmentation launch, ML inference, deposition | Sketches only |
| [`reference_layouts.md`](reference_layouts.md) | Real dataset layouts, and the axes they vary along | Reference |
| [`design_record_gui.md`](design_record_gui.md), [`design_record_layouts.md`](design_record_layouts.md) | Why the window and legacy-layout support are as they are | As built |

---

## Where the work stands

**Suite: 859 passed, 8 skipped, ~40 s.** `ruff check` clean. Confirmed on both
PyQt5 (development machine) and PySide6 (the user's), and under `de_DE.UTF-8`.

Built across phases 0–7:

* **Cleanup** — dead code removed, `pandas` dropped as a requirement, stale
  documentation corrected, `ruff` and `pytest` configured, version
  single-sourced.
* **GUI** — timeline margin, right-click copy path and reveal, scan-root row
  collapsed, per-lamella and per-project curation status chips.
* **Test infrastructure** — shared dataset builders in `conftest.py`, tested in
  their own right, plus an opt-in smoke test against the real collection.
* **`image_metadata.py`** — acquisition settings from both header dialects,
  validated 1080/1080 field values against the standalone script it replaced.
* **`operations/metadata_edit.py`** — editing `AM_metadata.yaml` in place,
  behind `show-metadata` and `edit-metadata`.
* **`aggregate.py`, `plots.py`, `operations/metadata_csv.py`** — counting,
  charts and exports, behind `stats`, `export-metadata-csv` and
  `export-projects-csv`, plus metadata filters on `create-training-csv`.
* **GUI tabs** — Curate, Metadata and Export, each with a Qt-free controller.

## What to pick up next

**`03_metadata_plan.md` §8, label quantification.** Specified and unblocked:
the classes, units and crack threshold were settled on 2026-08-14 and are
written into §8.1. Two smaller decisions remain in §8.2, each with a
recommendation. This is the last substantial feature in the plan.

Then, in no particular order:

* `01_refactor_plan.md` §5 — test fixture consolidation, deliberately
  opportunistic: convert a file when editing it anyway, never in a sweep.
* `01_refactor_plan.md` §4.1 — run the suite under the other Qt binding and
  another locale. This has caught three real bugs and is the highest-value
  habit here.
* `03_metadata_plan.md` §9 — four open questions, the first being whether
  data-repository deposition is still a goal. `CLAUDE.md` lists it as a main
  function of the package and nothing implements it.

---

## Things a fresh context should know

Facts that cost time to rediscover, each learned the hard way.

**The reference collection is at
`<collection>/`** — 21 projects,
2670 indexed images, 40 GB, three microscopes. Treat it as read-only. Point the
smoke test at it with `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.

**The sample layouts are examples, not a taxonomy.** `conftest.build_dataset`
takes layout *axes* — lamella level, stage nesting, naming, modalities,
variants — and the named samples are presets over them. A suite passing across
all five does not show that every layout works. Vary the axis a behaviour
actually depends on.

**Modality presence is a property of a dataset, not of its layout.** Two
documented datasets have SEM only; the oldest dataset in the collection has FIB
throughout.

**Every image header carries settings for both beams.** `microscope_state.*`
describes the beam that took the image; `system.*` holds both beams' configured
defaults. Reading `system.ion.voltage` from an SEM image yields the FIB's
30 kV — plausible and wrong. `image_metadata.FIELDS` confines beam-specific
fields, including the fallback search, to the beam that fired.

**A missing measurement is never a default.** The single exception is the
integration counts, where absence means "not enabled", which is one pass — that
is decoding the field rather than defaulting it, and it is the only field with
a floor.

**`downscale_to_fit` uses `min(width_ratio, height_ratio)`.** The images are
landscape in a portrait pane, so **width binds**, and window chrome does not
change how they render. An earlier claim to the contrary was wrong; do not
repeat it.

**Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a
process-global side effect, and `test_gui_*` sorts before most other tests.
That is how a locale-dependent `strptime('%p')` bug appeared only in full runs,
only on a machine with a non-English locale. `LC_ALL=… pytest` does **not**
test this; `AM_SEG_TEST_LOCALE=… pytest` does.

**The development and user environments differ** — PyQt5 here, PySide6 there.
Three bugs reached the user because of it. See `01_refactor_plan.md` §4.1.

**Never mix conda-forge with Anaconda `defaults`.** `environment.yml` pins
`nodefaults` for this reason: a mixed environment paired Anaconda's OpenCV with
conda-forge's protobuf and `cv2` would not load at all.

---

## Why these documents are split

They have different lifetimes. The refactor plan is a checklist that gets
consumed and deleted. The GUI plan was a small change set. The metadata plan is
a long-lived design that records decisions and their reasons. Future work is a
holding area for things deliberately *not* being designed yet, kept separate so
speculative detail does not leak into the metadata plan and inflate its scope.

The failure mode of one big document is that the consumable checklist and the
long-lived design get edited together and neither reads cleanly. The failure
mode of too many is that nobody knows which is current — hence this index, and
the status line at the top of each.

---

## How often to refactor a package this size

There is no schedule, and treating refactoring as scheduled is usually a
mistake. Two habits work.

**Continuous, inside feature work.** Every feature leaves the code slightly
better arranged than it found it. This is the bulk of it and never appears as a
task in its own right.

**Deliberate consolidation at boundaries**, when something learned about the
problem invalidates the current arrangement and no amount of tidying will fix
it. The `layout.py` centralisation was one. Expect one or two a year, triggered
by a discovery rather than a date.

**The signal that one is overdue**, and the only heuristic worth memorising:
you find yourself changing the same fact in more than two places. One is right,
two is tolerable, three means the fact has no home and needs one.

That signal fired once here — `filtering`, `survey` and `reporting` each walked
the project tree, and statistics would have been a fourth. `aggregate.py` is
the answer, and it counts records the filter already yields rather than walking
anything itself.
