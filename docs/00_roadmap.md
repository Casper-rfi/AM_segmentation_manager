# Roadmap

Where the work stands and what to pick up next — the entry point for starting fresh. `README.md`
catalogues the commands; [`decisions.md`](decisions.md) says why the software is the way it is and
[`display.md`](display.md) why the window is; `CLAUDE.md` holds the working agreement. Nothing
those say is restated here.

---

## What the software does today

All five main functions in `CLAUDE.md` are built, across the three series of images Autolamella
writes and every layout era the collection holds. Curating, proofreading, quantifying and exporting
are four tabs, each with a controller holding no Qt, and the command line does the same work for
scripting and testing. Predictions need the second environment `scripts/setup_ml.sh` builds.

## What to pick up next

**Label history and merge**, on branch `label-history` and deliberately unmerged. An append-only
`AM_history.jsonl` per project records what changed, who changed it and when, and `am-seg history`
reads one back. **Recording is always on**: the first change to a project starts its log and records
what the whole project held, while looking writes nothing; `am-seg init-history` baselines a whole
collection at once. **No name is recorded unless somebody states one**; `--as` credits a dataset that
arrived annotated. A replaced version is kept under `AM_archive/`, which `am-seg archive` reports and
`am-seg prune-archive` thins, collapsing what lies between savepoints and never what one names. An
export carries the history and writes an `AM_export.yaml` naming the fork point; `am-seg merge`
brings a copy back three-way against the history the two logs share, refusing it whole while a
conflict stands. `am-seg revert` undoes a conversion or a merge, or a whole round by name with
`--to`. `am-seg savepoint` names a state — one `checkpoint` entry per project sharing an id, marking
a round of model improvement — and `am-seg list-savepoints` says what each still reaches.

In the proofreading view, *Compare versions* lists the versions of the segmentation on screen — those
kept when the label was replaced and the predictions beside it — and puts a kept one back.
**Taking a savepoint, putting one back over the whole collection, and reclaiming archive space are in
Add data's *Whole collection* group** with the other acts covering a whole collection, and what the
archive costs sits on a line beside them, measured when that tab is opened rather than during a scan.
**Revert is in Curate's operations** for a selection, going back to a conversion, a merge or a
savepoint, and **merge is in Export** beside the copy it brings home, a conflict being decided one
row per group with nothing chosen in advance and both versions drawn over the acquisition, swapped
with **B**. Who changes are recorded under is set on **F2** and shown beside the mark. Everything
here is now reachable from the window.

**A macOS `.icns` icon**, the current one being a full-bleed PNG where macOS wants a squircle inset
in its canvas. Artwork rather than code, and Casper's to draw.

**An "other" series** making unindexed images visible instead of merely counted.
[`04_future_work.md`](04_future_work.md) §6.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
tree.

**Before publishing**, several packaging gaps are open by decision rather than oversight and all
become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
`"Private :: Do Not Upload"` classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the
version pinned at `0.0.0`. `mypy` is not configured and `ruff` does not check types, so the
annotations are unverified. The suite has not been run under a non-English locale — do that before
any release, for the reasons in [`display.md`](display.md) § Testing the window.
