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
— and a fourth, `other`, holding every image no naming rule reaches and listed once switched on —
and every layout era the collection holds. Add data, Curate, Label and Export are four tabs, each
with a controller holding no Qt, and the command line does the same work for scripting and testing;
Curate also draws a project's grid overviews, whatever wrote them, with its lamellae on it. Predictions
need the `ml` extra.

**Label history** records what changed, who changed it and when, keeps every replaced label under
`AM_archive/`, names a round with a savepoint, and merges a worked-on copy home three-way, its
conflicts decided one at a time or a folder, project or lamella at once.
[`decisions.md`](decisions.md) § History says how each of those decides.

**Settings** (F2) hold the name changes are recorded under, the window's accent and what each
control opens at, the last two declared in `gui/preferences.py` and reaching no write path.

## What to pick up next

- **Cross-platform readiness**, in order: a CI matrix on Linux, Windows and macOS for Python
  3.11–3.13, trimmed while the repository is private; what it finds on Windows; a cap on
  `POOL_WORKERS`; choosing a GPU by name and free memory. Without extras the suite already fails:
  four test files and eight tests need numpy or Qt rather than skipping.
- **Try the conflict resolver on a real merge.** Its keys have been exercised only through their
  handlers, an offscreen window never being activated.
- **Close one round on the real collection.** Its record starts at savepoint `proofreading001`,
  taken 2026-09-11 over all 123 projects with 3,236/3,236 labels reachable. Close **one whole
  round** from there with a real training run in the middle: the parts have each been rehearsed,
  the loop never once end to end.
- **Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
  editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
  tree.
- **Before publishing**, several packaging gaps are open by decision rather than oversight and all
  become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
  `"Private :: Do Not Upload"` classifier, and `THIRD_PARTY_NOTICES.md` now records an MIT
  obligation this package carries whatever licence it takes), no CI, no `CHANGELOG.md`, no
  `[project.urls]`, and the version pinned at `0.0.0`. `mypy` checks every module's annotations
  and reports nothing. The suite passed under German and
  Japanese locales on 2026-09-11; run it so again before any release, for the reasons in
  [`display.md`](display.md) § Testing the window. And
  [`decisions.md`](decisions.md) § Environment names the collection by absolute path, which
  is what makes never writing to it a rule about a real place rather than a principle: decide then
  whether a reader outside this machine needs the paths or only the rule.
