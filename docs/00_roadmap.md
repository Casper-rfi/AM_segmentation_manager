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

**Label history and merge**, five stages of seven built. An append-only `AM_history.jsonl` per
project records what changed, who changed it and when; `am-seg init-history` writes a collection's
first logs and baselines what is already on disk, at about 300 bytes an entry; every operation that
writes a label, a state or a metadata field records into it, and `am-seg history` reads one back.
Replacing a label keeps the version under `AM_archive/` unless the pixels are provably untouched
machine output, which `am-seg archive` reports and `am-seg prune-archive` thins. `export-files`
carries the history and writes an `AM_export.yaml` naming the ancestor a merge back compares
against. **Recording begins only where a log exists**, so nothing happens to a collection until
`init-history` runs on it — and a collection with no history behaves exactly as it does today.

`am-seg merge` brings a copy back: three-way per label, state entry and metadata field against the
ancestor the fork point names, re-hashing both trees rather than believing either log, taking what
only one side changed and refusing the whole merge while any conflict stands.

Next, and last of this work: the window. A version list per label, an A/B comparison against a kept
version, restoring, and deciding a merge conflict without the command line.

**A macOS `.icns` icon**, the current one being a full-bleed PNG where macOS wants a squircle inset
in its canvas. Artwork rather than code, and Casper's to draw.

**An "other" series** making unindexed images visible instead of merely counted.
[`04_future_work.md`](04_future_work.md) §6.

**Summary plots for the older eras**, whose filenames carry no acquisition number; the current
era's already pair. [`04_future_work.md`](04_future_work.md) §5.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
tree.

**Before publishing**, several packaging gaps are open by decision rather than oversight and all
become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
`"Private :: Do Not Upload"` classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the
version pinned at `0.0.0`. `mypy` is not configured and `ruff` does not check types, so the
annotations are unverified. The suite has not been run under a non-English locale — do that before
any release, for the reasons in [`display.md`](display.md) § Testing the window.
