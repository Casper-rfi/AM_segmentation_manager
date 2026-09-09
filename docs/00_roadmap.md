# Roadmap

Where the work stands and what to pick up next — the entry point for starting fresh. `README.md`
catalogues the commands; [`decisions.md`](decisions.md) says why the software is the way it is and
[`display.md`](display.md) why the window is; `CLAUDE.md` holds the working agreement. Nothing
those say is restated here.

**One line per thing.** A capability that lands adds a line and a branch that merges collapses to
one; prose has to be rewritten whole every time either happens.

---

## What the software does today

Every main function `CLAUDE.md` names is built, across the three series of images Autolamella writes
and every layout era the collection holds. Add data, Curate, Label and Export are four tabs, each
with a controller holding no Qt, and the command line does the same work for scripting and testing.
Predictions need the second environment `scripts/setup_ml.sh` builds.

**Label history** records what changed, who changed it and when, keeps every replaced label under
`AM_archive/`, names a round with a savepoint, and merges a worked-on copy home three-way.
[`decisions.md`](decisions.md) § History says how each of those decides.

**Settings** (F2) hold the name changes are recorded under and what each control opens at, the
latter declared in `gui/preferences.py` and reaching no write path.

## What to pick up next

- **Close one round on the real collection.** Run `init-history` there, so a savepoint covers
  everything rather than only what has been touched since — around 950 KB across 3,236 labels, two
  minutes, adding two files per project and modifying nothing. Then close **one whole round** with a
  real training run in the middle: the parts have each been rehearsed, the loop never once end to
  end.
- **A macOS `.icns` icon**, the current one being a full-bleed PNG where macOS wants a squircle
  inset in its canvas. Artwork rather than code, and Casper's to draw.
- **An "other" series** making unindexed images visible instead of merely counted.
  [`04_future_work.md`](04_future_work.md) §6.
- **Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
  editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
  tree.
- **Before publishing**, several packaging gaps are open by decision rather than oversight and all
  become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
  `"Private :: Do Not Upload"` classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the
  version pinned at `0.0.0`. `mypy` is not configured and `ruff` does not check types, so the
  annotations are unverified. The suite has not been run under a non-English locale — do that
  before any release, for the reasons in [`display.md`](display.md) § Testing the window. And
  [`decisions.md`](decisions.md) § Environment names the collection twice by absolute path, which
  is what makes never writing to it a rule about a real place rather than a principle: decide then
  whether a reader outside this machine needs the paths or only the rule.
