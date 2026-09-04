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
a state or a metadata field records into it, and `am-seg history` reads one back. Replacing a label
keeps the version under `AM_archive/` unless the pixels are provably untouched machine output, which
`am-seg archive` reports and `am-seg prune-archive` thins. `export-files` carries the history and
writes an `AM_export.yaml` naming the ancestor `am-seg merge` compares against — three-way, re-hashing
both trees rather than believing either log, refusing the whole merge while any conflict stands.
`am-seg revert` undoes a conversion or a merge across a selection. **Recording begins only where a
log exists**, so a collection with no history behaves exactly as it does today.

In the proofreading view, *Compare versions* lists the versions of the segmentation on screen — those
kept when the label was replaced, and the predictions beside the image — shows one on the canvas while
the compare key is held, marks where it and the label disagree, and puts a kept one back through the
ordinary save. Off by default, the count on the switch saying whether opening it would show anything.

What is left, in order: the defects on the merge path, chief among them that a merge replaces a
corrected label without asking; rehearsing the round trip on two machines; savepoints marking rounds
of model improvement, with `revert --to` and pruning defined by them; `init-history --as`, so a
dataset annotated elsewhere is not attributed to whoever ran the command; and then deciding whether
any of this earns a surface in the window. None of the six commands is reachable from it today.

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
