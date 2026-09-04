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
`AM_history.jsonl` per project records what changed, who changed it and when; every operation that
writes a label, a state or a metadata field records into it, `am-seg init-history` baselines what is
already on disk, and `am-seg history` reads one back. **No name is recorded unless somebody states
one**; `--as` credits a dataset that arrived annotated. A collection opts in once, and a project
added later joins without anybody asking. A replaced version is kept under `AM_archive/`, which
`am-seg archive` reports and `am-seg prune-archive` thins, collapsing what lies between savepoints
and never what one names. An export carries the history and writes an `AM_export.yaml` naming the
fork point; `am-seg merge` brings a copy back three-way against the history the two logs share,
refusing the whole merge while any conflict stands. `am-seg revert` undoes a conversion or a merge
across a selection, or a whole round by name with `--to`.
`am-seg savepoint` names a state — one `checkpoint` entry per project sharing an id, marking a round
of model improvement — and `am-seg list-savepoints` says what each covers, how much is still
reachable, and which projects record nothing at all.

In the proofreading view, *Compare versions* lists the versions of the segmentation on screen — those
kept when the label was replaced, and the predictions beside the image — and puts a kept one back.

The two-machine round trip has been rehearsed on real copied projects: two rounds of proofreading
merged back, a refusal over proofread work, a genuine conflict, and a revert. What is left is the
window — merge in Export beside the copy it undoes, revert in Curate's operations, savepoints and
pruning in Add data's *Whole collection* group where a collection-wide operation already lives.
`archive` and `history` get no surface, the version strip answering both. Two repairs come first, and
**the copy the window writes now carries its fork point**, so it can be merged back at all. The
second: recording still begins only where a log exists, where the first write into a project should
start one and baseline it, leaving `init-history` the catch-up rather than the switch.

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
