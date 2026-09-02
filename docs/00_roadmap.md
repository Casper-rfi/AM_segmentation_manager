# Roadmap

Where the work stands and what to pick up next — the entry point for starting fresh. `README.md`
is the user and developer entry point and its command tables are the catalogue of what each
command does; [`decisions.md`](decisions.md) says why the software is the way it is and
[`display.md`](display.md) why the window is; `CLAUDE.md` holds the working agreement. Nothing
those say is restated here.

---

## What the software does today

All five main functions in `CLAUDE.md` are built, across the three series of images Autolamella
writes and every layout era from pre-Autolamella to 2026-05. Curating, proofreading, quantifying
and exporting are four tabs, each with a controller holding no Qt, and the command line does the
same work for scripting and testing. Predictions need the second environment
`scripts/setup_ml.sh` builds. Curate lists its projects as a tree under the session folder each
one sits in, and F1 lists the keys of the tab in front along with what every mark on it means;
[`display.md`](display.md) says how those behave and why.

## What to pick up next

**The window's remaining rework** is the work in hand, and the only entry here with an order to it.
First the one shape that needs the controller to change: a folder whose *lamellae* can all be
listed at once across the projects in it, an Autolamella restart being an artifact rather than a
boundary anyone works to. Then the tree in Add data and Export, Add data wanting it most —
editing one folder's projects together is the point of grouping them. Then a contrast control
applied after the cache, so moving it costs a lookup table rather than a re-read; a curation
timeline carrying one colour vocabulary fewer; a `QPainter` keyboard diagram for F1; and a macOS
`.icns` icon, the current one being a full-bleed PNG where macOS wants a squircle inset in its
canvas.

The rest, in no particular order.

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
