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

**Label history**, on branch `label-history` and not merged. An append-only `AM_history.jsonl` per
project records what changed, who changed it and when. **Recording is always on**: the first change
to a project starts its log and records what the whole project held, while looking writes nothing.
**No name is recorded unless somebody states one**; who changes are recorded under is set on F2 and
shown beside the mark. A replaced label is kept under `AM_archive/`, which `prune-archive` thins by
collapsing what lies between savepoints. `savepoint` names the state of every label as a round of
model improvement, and `revert` puts labels back to a savepoint, a conversion or a merge. An export
carries the history and writes an `AM_export.yaml` naming the fork point; `merge` brings a copy back
three-way against the history the two logs share, refusing the whole merge while a conflict stands.

All of it is reachable from the window. Taking a savepoint, restoring one over the whole collection,
and reclaiming archive space are in Add data's *Whole collection* group, with what the archive costs
on a line beside them; reverting a selection is in Curate's operations; merging is in Export beside
the copy it brings home, with a conflict decided one row per group and both versions drawn over the
acquisition. In the proofreading view, *Compare versions* lists the versions of the label on screen
and puts a kept one back.

## What to pick up next

**Finish the history work.** Three things, in order. Run `init-history` once on the real collection,
so a savepoint taken there covers everything rather than only what has been touched since — around
950 KB across 3,236 labels, two minutes, adding two files per project and modifying nothing, and the
one act never yet performed on it. Then close **one whole round** with a real training run in the
middle: the parts have each been rehearsed, the loop never once end to end. Then merge the branch.

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
