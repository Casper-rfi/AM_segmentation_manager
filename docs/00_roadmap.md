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

**Label history and merge**, a staged piece of work of which the first stage is built: `history.py`
records what changed, who changed it and when, into an append-only `AM_history.jsonl` per project,
and `am-seg history` reads one back. Nothing writes entries yet — the operations are instrumented
next, then a baseline command for the collection as it stands, then an archive of replaced labels,
then merging a worked-on subset back. The invariant across all of it: a collection with no history
behaves exactly as it does today.

**A macOS `.icns` icon**, the current one being a full-bleed PNG where macOS wants a squircle inset
in its canvas. Artwork rather than code, and Casper's to draw.

**An "other" series** making unindexed images visible instead of merely counted.
[`04_future_work.md`](04_future_work.md) §6.

**Matching summary plots** to the frames they describe — no plot reaches a frame in any project
today. [`04_future_work.md`](04_future_work.md) §5.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
tree.

**Before publishing**, several packaging gaps are open by decision rather than oversight and all
become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
`"Private :: Do Not Upload"` classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the
version pinned at `0.0.0`. `mypy` is not configured and `ruff` does not check types, so the
annotations are unverified. The suite has not been run under a non-English locale — do that before
any release, for the reasons in [`display.md`](display.md) § Testing the window.
