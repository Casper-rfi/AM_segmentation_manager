# Roadmap

What the software does today, what is planned, and where the rest of the
documentation is. This is the entry point for picking the work up fresh.

| Doc | Covers |
|---|---|
| [`decisions.md`](decisions.md) | Why the built software is the way it is, and the environment traps |
| [`04_future_work.md`](04_future_work.md) | What is not built, and the questions to answer before designing it |
| [`reference_layouts.md`](reference_layouts.md) | Real dataset layouts, and the axes they vary along |

`README.md` is the user and developer entry point; `CLAUDE.md` holds the working
agreement and the safety constraints.

---

## What the software does today

Four of the five main functions in `CLAUDE.md` are implemented.

**Curate.** `inspect-layout` reports how a directory is interpreted and what does not
fit; `init-project` writes `AM_metadata.yaml` so a pre-Autolamella dataset becomes
indexable. The curation window is the primary interface, with `select` /
`deselect` / `correct` and `repair-state` on the command line. All of them expand
a change across the label group and validate the whole batch before moving
anything.

**Prepare labels.** `convert-predictions` turns predictions into labels, resizing
where prediction and image dimensions differ. `compress-labels` shrinks them on disk.

**Export for training.** `create-training-csv` filters by workflow state and by
metadata; `check-training-csv` re-checks an exported list against the current tree.

**Quantify.** `stats` counts records by one or two properties, with `--plot` for bar
charts. `quantify-labels` measures per-class areas and cracks in µm², rolled up per
lamella. `export-metadata-csv` and `export-projects-csv` write per-image and
per-project metadata. `show-metadata` and `edit-metadata` read and edit project
metadata in place.

The window has three tabs in workflow order — Add data, Curate and Export — each with
a controller that holds no Qt. Curate is the tab that opens. Keys belong to a tab and
are live only while it shows, so `F1` lists what can actually be pressed. What a state
change applies to is ticked in the project and lamella lists, or picked out of the
timeline frame by frame.

Every command whose input is that selection is also in the window: Curate runs
`convert-predictions`, `compress-labels` and `repair-state` against the ticked set,
Add data runs `inspect-layout` and `init-project` over the scanned folder, and Export
checks a CSV written earlier. Each previews first and applies only what the preview
showed, on a worker thread so a long one cannot freeze the window.

**Deposition is the one main function not implemented.** See
[`04_future_work.md`](04_future_work.md) §3.

---

## What to pick up next

In no particular order.

**Deposition** is the package's end state and the largest remaining piece. It is a
packaging and verification step, not a restructuring one: the three CSVs already
exist, and the substantive gap is that `segmentation_config_AM.yaml` must travel with
the data or the label indices mean nothing. Also wanted: a single "is this ready to
deposit" verdict over the checks `inspect-layout` already performs, and a manifest
with checksums so a 40 GB transfer can be verified.

**Importing existing labels** into the project hierarchy, once the source structures
have been looked at. [`04_future_work.md`](04_future_work.md) §4.

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

**Run the suite under both Qt bindings and a non-English locale before any release.**
This has caught three real bugs. See [`decisions.md`](decisions.md) § Environment for
why `LC_ALL=` is not sufficient.
