# Refactor and cleanup plan

Status: **§1–§3 done (phases 0–1). §4 and §5 outstanding.**
Written 2026-08-14 against `main` at c9c05e2.

A consumable checklist. When every box is ticked this document should be
deleted — it has no value as a record. The two remaining sections are §4
(packaging and tooling gaps: licence, author metadata, ruff, CI) and §5 (test
fixture consolidation, which the roadmap deliberately schedules as
opportunistic rather than as a single pass).

Baseline: `pytest -q` gives **667 passed, 1 skipped in 39 s**. Nothing below is
a bug fix; the package works. This is removal of things that stopped being true
and of code nothing calls. The suite has stayed at 667 throughout.

---

## 1. Branches and repo hygiene

### 1.1 Branches

At the time of the audit there were four local branches and **no remote**:

```
  empty-labels-and-migration   917350f
  feature/curation-gui         c494f49
* main                         c9c05e2
  retrograde-layout-support    059ddf3
```

`git branch --no-merged main` returns nothing: **all three feature branches are
fully contained in `main`.** The history is linear — each branch tip is simply
an older commit on `main`'s first-parent line. They are labels on the past, not
divergent work.

- [x] Delete all three. **Done 2026-08-14.**

```bash
git branch -d empty-labels-and-migration feature/curation-gui retrograde-layout-support
```

`-d` (not `-D`) is deliberate: it refuses to delete anything not merged, so if
one of them unexpectedly holds unique work the command fails safely rather than
losing it. If it fails, stop and investigate rather than reaching for `-D`.

The commits stay in history and the messages remain findable with `git log`.
Nothing is lost. Branch names are for work in progress; keeping them after a
merge means `git branch` stops telling you what is actually in flight.

**There is also no remote.** For a package intended to become open source, and
for data-management code whose loss would be expensive, a single copy inside a
OneDrive folder carries a specific risk: `.git` is thousands of small files, and
sync services occasionally resolve concurrent writes by writing "conflicted
copy" duplicates. Git cannot read those, so a conflict inside `.git/objects` or
`.git/refs` can leave a repository that will not open — and per-file version
history is an awkward thing to rebuild a repository from.

**Deferred, 2026-08-14.** The repository is backed up via OneDrive and a remote
is not wanted yet. Recorded here so this reads as a decision rather than an
outstanding task; revisit when the package is published, since a remote is
needed then anyway.

The cheap mitigation in the meantime, which covers the case OneDrive is worst
at and involves no hosting:

```bash
git bundle create ~/am_seg_manager_$(date +%Y%m%d).bundle --all
```

One file, entire history, all branches, restorable with `git clone`. Keep it
outside the synced folder.

- [ ] *(deferred)* Create a private remote and push `main`.

### 1.2 Duplicated configuration file

`segmentation_config_AM.yaml` exists twice, byte-for-byte identical:

- `./segmentation_config_AM.yaml` (repo root)
- `./src/am_segmentation_manager/data/segmentation_config_AM.yaml` (packaged)

Only the packaged copy is used as the fallback. The root copy is a leftover from
commit c494f49 ("Track the segmentation config at the repository root") which
the packaging change superseded. Two copies of the same fact, and nothing keeps
them in step.

- [x] Delete the repo-root copy. **Done 2026-08-14.** `config.py`'s
      `default_config_path()` resolves to the packaged file, the five classes
      load from it, and the suite is unchanged at 667 passed / 1 skipped.

### 1.3 Files that are gitignored yet tracked

**No action needed — this section was written on a misreading.**

The claim was that `.gitignore` lists `implementation_suggestion.txt` and
`Makefile` while `implementation_suggestion.txt` is tracked. It is not:

```
implementation_suggestion.txt    tracked=no exists=yes ignored=yes
Makefile                         tracked=no exists=yes ignored=yes
```

Both are ignored and untracked, which is consistent. There is no contradiction
to resolve and nothing to change.

Retained rather than deleted so the claim is not re-derived from the original
notes later. §3.2 still applies to the *content* of
`implementation_suggestion.txt` — it is superseded and now misleading — but
since it is untracked, "deleting" it is a local file operation with no bearing
on the repository.

---

## 2. Dead code

Each item below was verified with a grep across `src/` and `tests/`.

### 2.1 `index.py` — a shim nothing imports

```python
"""Compatibility shim providing ProjectIndex at package level."""
from .scanner import ProjectIndex
```

Zero importers. `ProjectIndex` is already re-exported from
`am_segmentation_manager/__init__.py`, so the shim is not even the convenient
path. Compatibility with what, is no longer answerable.

- [x] Delete `src/am_segmentation_manager/index.py`. **Done** (e598b28).

### 2.2 `records_to_dataframe` — dead, and the sole reason for a hard dependency

`operations/csv.py:23`. Nothing calls it. It is the **only** use of pandas in the
package, and `pandas>=1.0` sits in `pyproject.toml`'s required dependencies
because of it.

This matters more than a normal dead function: a required dependency is
something every user must install, and pandas is not small. `CLAUDE.md` asks for
minimal dependencies, and this one currently buys nothing.

- [x] Delete `records_to_dataframe`. **Done** (e598b28).
- [x] Remove `pandas>=1.0` from `[project].dependencies`, leaving `PyYAML` as
      the only required runtime dependency. **Done** (e598b28).
- [x] Remove `pandas` from `environment.yml`'s core list. **Done** (e598b28).

**Caveat, and the reason to do this now rather than "leave it, we'll need it":**
`03_metadata_plan.md` may reintroduce pandas for the statistics work. Adding it
back later, with a stated reason, is the correct outcome. Leaving it in now on
the assumption that it will be needed means it is never re-examined and you
never find out whether the standard library's `csv` and `collections.Counter`
would have done. Depend on things you use, when you use them.

### 2.3 `operations/labels.py` — thin wrappers used only by one test

`move_to_deselected`, `move_to_selected` and `move_to_corrected` are one-line
wrappers around `move_to_state`. `move_to_corrected` has no caller at all; the
other two are called only from `tests/test_stage_discovery.py`. The real
production path for state changes is `operations/selection.py`, which calls
`move_to_state` directly.

This is a small thing, but it is a test asserting the behaviour of an API that
nothing else uses — which means the test passes forever and protects nothing.

- [x] **Done** (e598b28) — the wrappers are deleted and the test now calls
      `move_to_state`. The alternative was to keep them and document them as the
      intended low-level public API; deleting won because
      `move_to_state(path, "deselected")` is no harder to read than
      `move_to_deselected(path)`, and one function with a validated state
      argument is less surface than four.

### 2.4 Sweep for others

The three above were found by hand. Before declaring this section done:

- [x] Run a dead-code pass. **Done** (441df03) — neither ruff nor vulture is
      installed, so pyflakes was used: 18 findings, 13 real, 5 false positives
      left alone (noqa-marked availability probes and a TYPE_CHECKING import).
      It also turned up a real gap: the CLI compress-labels test never asserted
      the file shrank.

---

## 3. Documentation that is no longer true

The package changed faster than the documents describing it. Each item is a
specific false statement, not a style preference.

### 3.1 `CLAUDE.md`

Recently refreshed (c9c05e2) and largely accurate. Four remaining problems:

**(a) The layer diagram is presented as complete and is not.** It lists four
layers:

```
layout.py      discovery.py      scanner.py      operations/
```

Absent: `gui/`, `frames.py`, `survey.py`, `reporting.py`, `filtering.py`,
`display.py`, `config.py`, `colors.py`, `io/`. That is more than half the
package. The diagram is a genuinely good statement of the *core dependency
rule* and should stay, but it should say it is describing the core, with the
peripheral modules named and placed relative to it.

- [x] Extend the diagram, or label it as "the core rule" and add the rest.
      **Done** — the four layers are now labelled the core, with the remaining
      modules listed around them.

**(b) The GUI is absent from the workflow section.** The "expected workflow"
walks through steps 0–3 entirely in CLI terms, and never mentions that curation
step 2 is now normally done in the curation window. Someone reading `CLAUDE.md`
to learn what the package does would not learn that it has a GUI.

- [x] Add the GUI to the workflow at step 2. **Done.**

**(c) The per-image metadata line will go stale imminently.**

> Per-image metadata should be directly extracted from the image headers.
> *(Not yet implemented.)*

- [ ] Leave for now; remove the parenthetical as part of
      `03_metadata_plan.md` phase 4. Noted here so it is not forgotten.

**(d) One stated main function has no implementation and no plan.** From the
Background section:

> export metadata file lists and **prepare data for deposition in scientific
> data repositories**

Nothing in the package does this, and it does not appear in your notes.
Metadata CSV export (metadata plan §6) is adjacent but is not the same thing —
deposition to something like EMPIAR needs a specified directory structure and a
specified metadata schema, which nobody has written down.

- [ ] Decide: is deposition still a goal? If yes, it needs its own plan and
      `CLAUDE.md` should say it is unimplemented. If no, remove the claim.
      Either is fine; leaving an unimplemented headline function stated as a
      main function is what should not persist.

### 3.2 `implementation_suggestion.txt` — superseded and now actively wrong

The original specification. Parts of it are now false, not merely outdated:

* It gives `labels/` and `labels_fib/` as the label directory names. Those are
  the **pre-convention** names now; the convention is `<image-dir>_labels`.
* It gives `predictions_fib/` for FIB predictions, which is not what the
  package implements.
* Its "index all Autolamella projects by finding experiment.yaml" is superseded
  by `AM_metadata.yaml` also marking a project.

Anyone reading it as guidance would build the wrong thing. It has served its
purpose: everything in it that was right is now either implemented or recorded
in `CLAUDE.md`.

- [ ] Delete the local file. **Note it is untracked (§1.3), so git holds no
      copy** — unlike every other deletion in this document, this one is not
      recoverable from history. Read it once more before removing it, or move
      it outside the repository instead. There is no `.gitignore` change to
      make.

### 3.3 `data_structures.txt` — keep, but move and re-title

Unlike the above this is still valuable: it is the ground-truth catalogue of
real path examples across five layout generations, and it is what future layout
work will be checked against. Two problems: it sits in the repository root as a
loose `.txt`, and it has answers annotated into it (`--> RESOLVED: ...`), so it
reads as a half-finished question list rather than a reference.

- [x] Move to `docs/reference_layouts.md`, convert to Markdown, and separate
      the observed paths from the resolutions. **Done** — observations are
      §1–§6, resolutions are §7.

### 3.4 `docs/gui_plan.md` (1209 lines) and `docs/retrograde_support_plan.md` (956 lines)

Both describe work that is **done**. Together that is 2165 lines of
forward-looking plan for a finished past. They are not worthless — the reasoning
in them is genuinely good, and `gui_plan.md` §18 in particular records why the
timeline colours are what they are, which would otherwise be lost and
re-litigated.

The problem is only that they are titled and framed as plans. Two specific false
statements in `gui_plan.md` §1:

* "Prediction→label conversion, CSV export, compression … stay CLI-only for
  now; the GUI is a curation tool, not a second front-end for everything."
  **`03_metadata_plan.md` §7 reverses this deliberately.**
* "The pre-Autolamella legacy layouts … The scanner does not discover them
  today, so the GUI will not show them either." **Now implemented.**

- [x] Rename both to `docs/design_record_gui.md` and
      `docs/design_record_layouts.md`, add a status header saying "as built,
      retained for the reasoning", and strike or annotate the two statements
      above. **Done** — both statements are struck through in place.
- [x] In `design_record_gui.md`, record the scope reversal explicitly (now
      its §19, with the replacement boundary "the GUI hosts operations that are
      iterative or exploratory; batch operations stay in the CLI"): the GUI
      *is* now growing a second front-end for CSV export and metadata, this was
      a considered change of mind, and the reason is that the metadata filters
      that drive CSV generation are only usable interactively. Silently
      contradicting a recorded decision is the thing to avoid; changing your
      mind in writing is fine.

### 3.5 `README.md`

Accurate. Two small things:

- [x] The "previously hardcoded in some 200 places across 18 modules" paragraph
      is duplicated verbatim from `CLAUDE.md`. Keep it in one place — `CLAUDE.md`
      is the instruction set, so it belongs there; the README can point at it.
      **Done.**
- [x] Add the `am-seg` console script to the CLI section. `pyproject.toml`
      defines it but the README only documents
      `python -m am_segmentation_manager.cli`. **Done.**

---

## 4. Packaging and tooling gaps

These are not code quality problems today, but every one of them is a blocker
for the stated goal of open-sourcing, and all are cheap now and annoying later.

- [x] **`pyproject.toml` metadata is placeholder.** **Done** — now
      `Casper Berger <casper.berger@rfi.ac.uk>`, the work address, matching the
      identity git already uses for every commit.
- [x] **`license = { text = "MIT" }` but there is no `LICENSE` file.**
      **Resolved by removing the declaration, 2026-08-14.** The MIT claim was
      never a decision that had been made, and a licence grant nobody chose is
      worse than none: undeclared defaults to all rights reserved, which is the
      accurate state. A `Private :: Do Not Upload` classifier now also blocks an
      accidental PyPI upload.

- [ ] *(deferred)* Choose a licence and add a `LICENSE` file. Open until the
      package is published. The question is not only which licence but **who
      the copyright holder is** — the author or the Rosalind Franklin Institute
      — which is likely an institutional decision rather than a personal one,
      and worth asking before the code goes anywhere.
- [ ] **Version is stated twice** — `pyproject.toml` `version = "0.0.0"` and
      `__init__.py` `__version__ = "0.0.0"`. They will drift. Pick one source:
      simplest is to keep it in `pyproject.toml` and read it in `__init__.py`
      with `importlib.metadata.version()`.
- [ ] **No linter or formatter.** `CLAUDE.md` requires type hints throughout and
      nothing checks that. Add `ruff` config to `pyproject.toml` (lint +
      format, one tool, no new concepts) and optionally `mypy`. Do not turn on
      every rule — start with the defaults plus `I` (import sorting) and fix
      what it finds.
- [ ] **No `[tool.pytest.ini_options]`.** `tests/conftest.py` manipulates
      `sys.path` to run against the source tree. That works, but declaring
      `pythonpath = ["src"]` in `pyproject.toml` is the supported way and lets
      the conftest be emptied of everything except real fixtures.
- [ ] **No CI.** A single GitHub Actions workflow running `pytest` on push
      would catch the class of breakage that a 667-test suite exists to catch
      but only when someone remembers to run it. Do this once the remote exists.
- [ ] **No `CHANGELOG.md`.** Optional at this stage; becomes necessary the
      moment a second person uses the package.

---

## 5. The test suite

667 tests, ~8.8k lines against ~10k lines of source. That ratio is healthy and
the suite is a real asset. Two observations, one of them significant.

### 5.1 The significant one: no shared fixtures

`tests/conftest.py` is 9 lines and contains **no fixtures at all** — only
`sys.path` manipulation. Meanwhile **32 of 45 test files build directory trees
by hand** with `mkdir(parents=True)`, and at least 25 define their own local
builder function.

The consequence is concrete: adding a sixth layout generation, or renaming a
directory convention, means editing tree-construction code in dozens of files.
That is precisely the duplication `layout.py` was created to eliminate in the
source, reproduced in the tests.

**Recommendation: add the builders, migrate opportunistically.**

- [ ] Add to `conftest.py` a small set of fixtures that build the canonical
      layouts — one per era in `data_structures.txt` — parameterised by what
      varies (modalities present, variants, which state folders exist, whether
      labels are empty).
- [ ] Migrate test files to them **as you touch each file for other reasons**,
      not in one sweep.

The reason not to do it as a big bang: a mass rewrite of 32 test files is a
large diff with no behaviour change, in exactly the code that would tell you if
you broke something. The risk-to-benefit is poor, and there is no deadline —
the metadata work does not change directory layouts, so the duplication will not
bite in the next few months. Add the shared builders now so new tests use them
from the start; let the old ones converge.

### 5.2 The minor one: names do not say which layer is under test

Several clusters cover the same *concept* at different layers, and the names
don't distinguish them:

| Concept | Files |
|---|---|
| Scanning | `test_scanner.py`, `test_scanner_collapsed.py`, `test_scanner_layouts.py`, `test_stage_discovery.py` |
| Selection | `test_selection.py`, `test_selection_state.py`, `test_shared_label_operations.py`, `test_corrected.py` |
| Variants | `test_variants.py`, `test_csv_variants.py`, `test_gui_variants.py`, `test_gui_variant_widgets.py` |

I checked a sample and these are **not redundant** — variants genuinely cut
across the model, CSV export, controller and widgets, and each file tests a
different layer. So: do not delete them. But `test_selection.py` versus
`test_selection_state.py` cannot be told apart from the name, and the next
person to add a selection test will not know where to put it.

- [ ] Low priority. Rename for layer rather than merge, e.g.
      `test_selection_operations.py` / `test_selection_scanner_state.py`.
      Cosmetic; do it when convenient.

### 5.3 Tests to actually remove

Only one candidate found:

- [ ] The `move_to_deselected` / `move_to_selected` tests in
      `test_stage_discovery.py`, per §2.3 — they are the sole users of an
      otherwise-unused API.

### 5.4 A gap worth filling

Every test uses `tmp_path`. Nothing exercises the real reference collection at
`<collection>/`, which is where all
the awkward real-world layouts actually live.

- [ ] Add an opt-in smoke test (skipped unless an env var points at the
      collection) that scans it and asserts the project count, that no findings
      of severity "error" appear, and that scanning completes. It should never
      write. This is the test that would catch "the scanner silently stopped
      seeing a whole era of data", which no `tmp_path` test can.

---

## 6. Is the package still reusing code where it should?

Broadly yes — this is the healthiest part of the audit. Spot checks:

* `layout.py` discipline is **holding**. The two most recent features (legacy
  layouts, empty labels/migration) were absorbed without spreading directory
  names back through the modules.
* `gui/widgets/state_panel.py` imports `STATE_COLOURS` from `frame_strip.py`,
  so the swatches really do follow the timeline colours as their docstring
  claims. Verified.
* `operations/selection.py` and the GUI controller share
  `label_groups.expand_to_groups` rather than each expanding shared labels.

**The one place it is about to break**, flagged here and addressed in
`03_metadata_plan.md` §5: `filtering.iter_matching_records`,
`survey.survey_collection` and `reporting.summarize_lamella` each walk the
project → lamella → stage → image tree independently. Three is already the
threshold where a shared traversal is worth extracting; adding a fourth for
statistics without extracting it would be the mistake.

---

## 7. Suggested commit sequence

Small commits, each leaving the suite green, per `CLAUDE.md`'s incremental rule.

1. `Delete merged feature branches` *(no commit — a branch operation)*
2. `Remove the duplicated segmentation config from the repository root`
3. `Delete the unused ProjectIndex shim`
4. `Drop the unused DataFrame helper and the pandas dependency`
5. `Collapse the state-move wrappers onto move_to_state`
6. `Retire the superseded implementation notes` (also moves `data_structures.txt`)
7. `Reframe the finished design plans as design records`
8. `Correct the architecture and workflow descriptions in CLAUDE.md`
9. `Fill in the package metadata and add a licence`
10. `Add ruff configuration and fix what it reports`
11. `Add shared layout fixtures for the test suite`

Run `pytest -q` after each. Expect 667 to become slightly fewer at step 5.
