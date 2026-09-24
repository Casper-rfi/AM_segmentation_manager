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
need the `ml` extra, or `ml-cuda13` where the card is newer than its torch.

**Label history** records what changed, who changed it and when, keeps every replaced label under
`AM_archive/`, names a round with a savepoint, and merges a worked-on copy home three-way, its
conflicts decided one at a time or a folder, project or lamella at once.
[`decisions.md`](decisions.md) § History says how each of those decides.

**Settings** (F2) hold the name changes are recorded under, the window's accent and what each
control opens at, the last two declared in `gui/preferences.py` and reaching no write path.

## What to pick up next

- **Cross-platform readiness**: CI passes on Linux, Windows and macOS, and Windows 11 has been worked through
  by hand — all four tabs, predicting on a 50-series GPU with `ml-cuda13`. Left: the window at 125% and 150%
  scaling on Windows and on an Ubuntu desktop, and an A100 workstation taking the freest GPU with `nvidia-smi`
  showing the run on it alone. **No CI job installs `ml` on Windows, or `ml-cuda13` anywhere.** **Long
  paths off stays unchecked on a real machine**, `test_long_paths.py` forcing the probe's answer. **The
  Windows suite crashed once in five runs** of 2026-09-17, near `test_gui_thread_shutdown.py`, nothing logged;
  CI now names each test.
- **Finish the code review of 2026-09-21**: two findings to verify, two minor; `plans/review_fixes.md`.
- **Close one round on the real collection.** Its record starts at savepoint `proofreading001`,
  taken 2026-09-11 over all 123 projects with 3,236/3,236 labels reachable. Close **one whole
  round** from there with a real training run in the middle: the parts have each been rehearsed,
  the loop never once end to end.
- **Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
  editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
  tree.
- **Before publishing** (`plans/release_preparation.md`; `create-shortcut` untried on Linux, and on Windows tried only as far as writing both shortcuts and starting the window from them), several packaging gaps are open by decision rather than oversight and all
  become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
  `"Private :: Do Not Upload"` classifier, and `THIRD_PARTY_NOTICES.md` now records an MIT
  obligation this package carries whatever licence it takes), and no `[project.urls]`; the
  version is `0.1.0.dev0`, and `CHANGELOG.md` holds the release's features under *Unreleased*. Run the workflow's `locale` job, German and Japanese, before any release,
  for the reasons in [`display.md`](display.md) § Testing the window; the suite last passed
  under both on 2026-09-17.
