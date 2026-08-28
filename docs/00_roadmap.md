# Roadmap

Where the work stands and what to pick up next — the entry point for starting fresh. `README.md`
is the user and developer entry point and its command tables are the catalogue of what each
command does; [`decisions.md`](decisions.md) says why the software is the way it is; `CLAUDE.md`
holds the working agreement. Nothing those say is restated here.

---

## What the software does today

All five main functions in `CLAUDE.md` are built, across the three series of images Autolamella
writes and every layout era from pre-Autolamella to 2026-08. Curating, proofreading, quantifying
and exporting are four tabs, each with a controller holding no Qt, and the command line does the
same work for scripting and testing. Predictions need the second environment
`scripts/setup_ml.sh` builds.

## What to pick up next

In no particular order.

**Matching summary plots** to the frames they describe — no plot reaches a frame in any project
today. [`04_future_work.md`](04_future_work.md) §5.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
tree.

**Before publishing**, several packaging gaps are open by decision rather than oversight, and all
become necessary at the same moment: there is no `LICENSE` (`pyproject.toml` deliberately declares
none and carries a `"Private :: Do Not Upload"` classifier), no CI, no `CHANGELOG.md`, no
`[project.urls]`, and the version is pinned at `0.0.0`. `mypy` is not configured, and `ruff` does
not check types, so the annotations are unverified. The suite has also not been run under a
non-English locale — do that before any release, for the reasons in
[`decisions.md`](decisions.md) § Environment.
