# Roadmap

What the software does today, and what to pick up next. This is the entry point for
picking the work up fresh. `CLAUDE.md` lists the other documents and holds the working
agreement; `README.md` is the user and developer entry point, and its command tables
are the catalogue of what each command does.

---

## What the software does today

Four of the five main functions in `CLAUDE.md` are implemented: curating which data to
use, preparing labels from predictions, exporting filtered training lists, and
quantifying the collection through its metadata and its labels. Every state change
expands across the label group and validates the whole batch before moving anything.

The window has three tabs in workflow order — Add data, Curate and Export — each with
a controller that holds no Qt. Curate is the tab that opens. Keys belong to a tab and
are live only while it shows, so `F1` lists what can actually be pressed. What a state
change applies to is ticked in the project and lamella lists, or picked out of the
timeline frame by frame.

Every command whose input is that selection is also in the window: Curate runs
`convert-predictions`, `compress-labels` and `repair-state` against the ticked set, and
Add data runs `inspect-layout` and `init-project` over the scanned folder. Each previews
first and applies only what the preview showed, on a worker thread.

**Export is three boxes with three jobs.** *Filters* decides which images are in play —
modality, workflow state, label state, acquisition metadata, and whether the repeated
imaging conditions of one scene collapse to one image. *Breakdown* counts what they
describe, by one property or two crossed, and charts it: bars for one, a grid for two
where the empty cells are the combinations nothing has been acquired for. *Export*
writes the three CSVs, taking the same split, seed and filenames as
`create-training-csv` and previewing the split before either file exists. Two buttons
buy what the filesystem cannot answer alone — reading the headers, and measuring the
labels, the second never implicit because it opens every one and adds an area breakdown
per class the segmentation config names. `stats --plot` writes the same charts to a
file. Scoping to particular projects or lamellae is still command-line only.

**Deposition is the one main function not implemented.** See
[`04_future_work.md`](04_future_work.md) §3.

---

## What to pick up next

In no particular order.

**Deposition** is the package's end state and the largest remaining piece — a
packaging and verification step rather than a restructuring one, since the three CSVs
already exist. [`04_future_work.md`](04_future_work.md) §3 lists what is left.

**Importing existing labels** into the project hierarchy, once the source structures
have been looked at. [`04_future_work.md`](04_future_work.md) §4.

**Project and lamella filters in the Export tab** — the last thing the command line can
narrow that the window cannot. Both are already offered as *breakdowns*, so you can see
that one lamella is a seventh of the annotated images and not then exclude it. The
matching exists (`filtering.matches_project`, `matches_lamella`) and the widget is the
same multi-select list the acquisition filters use; the open question is that lamella
names repeat between projects, so the filter has to be scoped by project or keyed on a
pair, as `aggregate.summarise_lamellae` already is.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py`
builders when editing them for other reasons, never as a sweep. Three files still
carry hand-built trees: `test_selection.py`, `test_selection_state.py` and
`test_cli_selection_integration.py`.

**Before publishing**, several packaging gaps are open by decision rather than
oversight, and all of them become necessary at the same moment:

* No git remote. The repository is backed up via OneDrive only. `.git` is thousands
  of small files and sync services occasionally write "conflicted copy" duplicates,
  which git cannot read. `git bundle create ~/am_seg_manager_$(date +%Y%m%d).bundle --all`
  is the cheap mitigation until then.
* No `LICENSE`. `pyproject.toml` deliberately declares none and carries a
  `"Private :: Do Not Upload"` classifier.
* No CI, no `CHANGELOG.md`, and no `[project.urls]`.
* The version is pinned at `0.0.0`.
* `mypy` is not configured. `ruff` does not check types, so the annotations are
  currently unverified.
* The suite has not been run under PySide6 or a non-English locale. Do both before any
  release: [`decisions.md`](decisions.md) § Environment has what it catches, and why
  `LC_ALL=` is not sufficient.
