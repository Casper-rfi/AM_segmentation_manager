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
`AM_history.jsonl` per project records what changed, who changed it and when; `am-seg init-history`
writes a collection's first logs and baselines what is on disk; every operation that writes a label,
a state or a metadata field records into it, and `am-seg history` reads one back. A replaced version
is kept under `AM_archive/`, which `am-seg archive` reports and `am-seg prune-archive` thins. `export-files` carries the history and writes an `AM_export.yaml`; `am-seg merge` brings a
copy back three-way against the history the two logs share, re-hashing both trees rather than
believing either log, refusing the whole merge while any conflict stands, and going through the
ordinary save so it cannot replace proofread work unasked. `am-seg revert` undoes a conversion or a
merge across a selection. `am-seg savepoint` names a state — one `checkpoint` entry per project
sharing an id, marking a round of model improvement — and `am-seg savepoints` says what each covers
and how much of it is still reachable; `am-seg revert --to` puts a whole round back by name, and
`am-seg prune-archive` collapses what lies between savepoints, never what one names.
**Recording begins only where a log exists**, so a collection
with no history behaves exactly as it does today.

In the proofreading view, *Compare versions* lists the versions of the segmentation on screen — those
kept when the label was replaced, and the predictions beside the image — shows one on the canvas while
the compare key is held, marks where it and the label disagree, and puts a kept one back through the
ordinary save. Off by default, the count on the switch saying whether opening it would show anything.

The two-machine round trip has been rehearsed on real copied projects: two rounds of proofreading
merged back, a refusal over proofread work, a genuine conflict, and a revert. What is left:
`init-history --as`; then whether any of this earns a surface in the window.

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
