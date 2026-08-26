# Roadmap

Where the work stands and what to pick up next — the entry point for starting fresh.
`README.md` is the user and developer entry point and its command tables are the catalogue
of what each command does; [`decisions.md`](decisions.md) says why the software is the way
it is; `CLAUDE.md` holds the working agreement. Nothing those say is restated here.

---

## What the software does today

All five main functions in `CLAUDE.md` are built, across the three series of images
Autolamella writes and every layout era from pre-Autolamella to 2026-08. In outline:

* **The filesystem is the index.** `layout.py` holds every naming convention,
  `discovery.py` finds projects, lamellae and stages, `scanner.py` builds a `ProjectIndex`,
  and everything else works on that rather than walking the tree again. Layouts that
  collapse the lamella or stage level are indexed where they lie, never reorganised.
* **Workflow state is declared**, in each stage's `AM_state.yaml`, keyed by label group.
* **Curating, proofreading, quantifying and exporting are four tabs**, each with a
  controller holding no Qt; the command line does the same work, for scripting and testing.
* **Predictions need a second environment**, built by `scripts/setup_ml.sh`.
* **A record carries what it is** — series, modality, imaging condition, milling task,
  workflow state, and where its label and predictions live — and one `BREAKDOWNS` entry in
  `aggregate.py` turns any of those into a count, a CLI selector and an Export filter row. A
  row in `metadata.FIELDS` does the same for a project's descriptive metadata, and decides the
  editor it gets and whether it can be defaulted.

---

## What to pick up next

In no particular order.

**Importing existing labels** into the project hierarchy, once the source structures
have been looked at. [`04_future_work.md`](04_future_work.md) §4.

**Test fixture consolidation** — migrate test files onto the shared `conftest.py`
builders when editing them for other reasons, never as a sweep. `test_selection.py`
still carries a hand-built tree.

**Before publishing**, several packaging gaps are open by decision rather than
oversight, and all become necessary at the same moment: there is no `LICENSE`
(`pyproject.toml` deliberately declares none and carries a `"Private :: Do Not Upload"`
classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the version is pinned at
`0.0.0`. `mypy` is not configured, and `ruff` does not check types, so the annotations are
unverified. The suite has also not been run under a non-English locale — do that before any
release, for the reasons in [`decisions.md`](decisions.md) § Environment. It does now pass
under PySide6 6.11.1, which the development environment resolves to.
