# Roadmap and planning index

Written 2026-08-14, against `main` at c9c05e2 (667 tests passing, 1 skipped).

This is the entry point to the planning documents. It answers *what order* and
*why that order*, and records the state of the repository at the point the plan
was made.

| Doc | Covers | Size |
|---|---|---|
| [`01_refactor_plan.md`](01_refactor_plan.md) | Alignment audit, dead code, doc truth, test suite, tooling gaps, branches | Small, mostly deletions |
| [`02_gui_improvements_plan.md`](02_gui_improvements_plan.md) | The four requested GUI changes | Small |
| [`03_metadata_plan.md`](03_metadata_plan.md) | Reading, editing, exporting and summarising metadata | Large — the main body of new work |
| [`04_future_work.md`](04_future_work.md) | Launching the segmentation software, ML inference | Sketch only |

---

## Why four documents and not one

Splitting is worth it here for one reason: **these have different lifetimes.**

* The refactor plan is a checklist that gets consumed and deleted. When every
  box is ticked it has no further value.
* The GUI plan is a small, self-contained change set — it will be done in one
  sitting and is easier to hold in one file than scattered through a larger doc.
* The metadata plan is a multi-month design that will be revised as it is
  built. It needs room to record decisions and their reasons, the way
  `gui_plan.md` and `retrograde_support_plan.md` did.
* Future work is a holding area for things deliberately *not* being designed
  yet. Keeping it separate stops speculative detail leaking into the metadata
  plan and inflating its scope.

The failure mode of one big doc is that the consumable checklist and the
long-lived design get edited at the same time, and neither reads cleanly. The
failure mode of too many docs is that nobody knows which is current — which is
why this index exists and why each doc carries a status line at the top.

---

## Recommended order

The ordering principle: **do the things that make later work cheaper first, and
do them only if they actually make later work cheaper.** Cleanup for its own
sake can wait indefinitely; cleanup that removes an obstacle cannot.

### Phase 0 — Branches and repo hygiene (minutes)

`01_refactor_plan.md` §1. Delete merged branches, remove the duplicated config
file. Zero risk, and it clears noise before anything else.

### Phase 1 — Dead code and documentation truth (half a day)

`01_refactor_plan.md` §2–§4. Removes ~unused code, drops `pandas` from the
required dependencies, and corrects the statements in `CLAUDE.md`, `README.md`
and the two design docs that are no longer true.

**Do this before the metadata work, not after.** `CLAUDE.md` is the instruction
set the metadata work will be built against; building against stale
instructions produces code that then has to be re-argued. In particular
`gui_plan.md` currently says CSV export stays out of the GUI, and the metadata
plan reverses that — the reversal should be recorded deliberately rather than
discovered as a contradiction halfway through.

### Phase 2 — GUI improvements (one to two days)

`02_gui_improvements_plan.md`. Three of the four are cosmetic and independent of
everything else. The fourth (lamella/project status indicators) needs only data
the controller already has. None of it depends on the metadata work, and doing
it now means the tool is more pleasant to use during the months the metadata
work takes.

### Phase 3 — Test fixture consolidation (opportunistic, not a big bang)

`01_refactor_plan.md` §5. 32 of 45 test files build directory trees by hand and
`conftest.py` holds no fixtures. This is the largest maintenance liability in
the repository, but it is **not urgent**, because the metadata work does not
change directory layouts. Add the shared builders now; migrate tests to them as
you touch each file, not in one sweep.

### Phase 4 — Reading per-image metadata

`03_metadata_plan.md` §3. The foundation for everything else in that document.
Absorbs and then retires `read_sem_tiff_header_csv.py`.

### Phase 5 — Editing project metadata (API → CLI → GUI)

`03_metadata_plan.md` §4. Independent of phase 4 and could be swapped with it.
Phase 4 first is recommended only because it is the harder unknown, and finding
out early that FIB headers differ from SEM headers is better than finding out
late.

### Phase 6 — Aggregation, statistics and filtered CSV export

`03_metadata_plan.md` §5–§6. Depends on phase 4.

### Phase 7 — GUI tabs for CSV export and metadata

`03_metadata_plan.md` §7. Depends on phases 5 and 6, and on the deliberate
scope reversal recorded in phase 1.

### Phase 8 — Label quantification (crack/hole proportions, size distributions)

`03_metadata_plan.md` §8. Deliberately last, and deliberately fenced. This is
where feature creep is most likely, and the plan says where to stop.

### Later — Future work

`04_future_work.md`. Not scheduled.

---

## How often should a package this size be refactored?

Since you asked directly: there is no schedule, and treating refactoring as a
scheduled activity is generally a mistake. The practice that works is two
layered habits.

**Continuous, inside feature work.** Every feature leaves the code slightly
better arranged than it found it. This is the bulk of it and it never appears
as a task in its own right. Your commit history already shows this happening.

**Deliberate consolidation at boundaries.** Occasionally you learn something
about the problem that invalidates the existing arrangement, and no amount of
small tidying will fix it. That is what the `layout.py` centralisation was: the
directory conventions were duplicated in ~200 places, and the fix was not
incremental. These are rare — for a package this size, expect one or two a
year, triggered by a discovery rather than a date.

**The signal that one is overdue**, and the only heuristic worth memorising: you
find yourself changing the same fact in more than two places. One place is
right. Two is tolerable. Three means the fact has no home and needs one.

By that measure the current package is in good shape. The `layout.py` discipline
is holding — the last two features (legacy layouts, empty labels) were absorbed
without spreading. What is proposed in `01_refactor_plan.md` is not a structural
refactor at all; it is deletion of things that stopped being true, which is a
much smaller and safer category of work. Roughly a day, not a week.

The one genuine structural risk ahead is described in
`03_metadata_plan.md` §5: three modules currently walk the index independently,
and the statistics work would make it four. That is the "same fact in more than
two places" signal firing, and the plan addresses it rather than adding to it.
