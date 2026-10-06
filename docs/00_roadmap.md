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
need the `ml` extra, or `ml-cuda13` where the card is newer than its torch. Imaging conditions form
**families**, each sharing one label: a folder's name gives the family and `AM_families.yaml` corrects it,
for the project or for one lamella over it, from Curate's Families tab or `edit-families`.

**Label history** records what changed, who changed it and when, keeps every replaced label under
`AM_archive/`, names a round with a savepoint, and merges a worked-on copy home three-way, its
conflicts decided one at a time or a folder, project or lamella at once.
[`decisions.md`](decisions.md) § History says how each of those decides.

**Settings** (F2) hold the name changes are recorded under, the window's accent and what each
control opens at, the last two declared in `gui/preferences.py` and reaching no write path.

**The viewer**, `am-seg gui --viewer`, is Curate alone for looking at a collection on the microscope,
its controller built read-only; `create-shortcut --viewer` gives it a launcher of its own.

## What to pick up next

- **Cross-platform readiness**: CI passes on Linux and Windows at every push, and passed on macOS
  when last run by hand on 2026-09-17; Windows 11 has been worked through by hand — all four tabs, predicting on a 50-series GPU with `ml-cuda13`; on a four-V100 Linux
  machine `auto` took a free GPU over a full one, and `nvidia-smi` showed the run on it alone. **Every
  torch build is installed by some CI job on each platform it is chosen for**
  (`test_each_torch_build_is_installed_on_every_platform_it_is_chosen_for`); the three beyond `ml` on
  Linux run on request and have not run yet. **Long paths off stays unchecked on a real machine**,
  `test_long_paths.py` forcing the probe's answer. **The Windows suite crashed once in five runs** of
  2026-09-17, near `test_gui_thread_shutdown.py`, nothing logged; CI now names each test.
- **Close one round on the real collection.** Its record starts at savepoint `proofreading001`,
  taken 2026-09-11 over all 123 projects with 3,236/3,236 labels reachable. Close **one whole
  round** from there with a real training run in the middle: the parts have each been rehearsed,
  the loop never once end to end.
- **Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
  editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
  tree.
- **Before publishing** (`plans/release_preparation.md`; `create-shortcut` untried under Wayland and with a Desktop moved into OneDrive; Xfce 4.18's panel menu cannot start the entry `gtk-launch` starts), several packaging gaps are open by decision rather than oversight and all
  become necessary at once: `pyproject.toml` carries a `"Private :: Do Not Upload"` classifier,
  and there are no `[project.urls]`; the
  version is `0.1.0.dev0`, and `CHANGELOG.md` holds the release's features under *Unreleased*. Run the workflow's `locale` job, German and Japanese, before any release,
  for the reasons in [`display.md`](display.md) § Testing the window; the suite last passed
  under both on 2026-09-17.
