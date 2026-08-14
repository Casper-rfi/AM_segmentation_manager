# Plan: retrograde support for older datasets

Design proposal for indexing pre-2026 Autolamella projects and pre-Autolamella
datasets, and for supporting multiple imaging conditions per modality with
shared labels.

Revision 4 — survey of the expanded test collection, plus corrections on
metadata scope, label import and indexing scope. Conventions in §4 are settled
(D1–D5 answered) and nothing in §10 is blocking, so this is ready to implement
from Phase 0.

---

## 1. What this is based on

- `data_structures.txt` (path examples across five layout generations)
- `extended_metadata_example.yaml`
- The expanded test collection in
  `<collection>/`
- The current package source

The single most important structural fact about the current code: the
filesystem convention is **not** centralised, despite `CLAUDE.md` requiring it.
Hardcoded occurrences of `sem`/`fib`/`labels`/`labels_fib`/`deselected`/
`corrected`/`predictions`:

| file | occurrences |
|---|---|
| `scanner.py` | 35 |
| `operations/selection.py` | 26 |
| `cli.py` | 26 |
| `operations/predictions.py` | 22 |
| `gui/controller.py` | 17 |
| `operations/labels.py` | 16 |
| `gui/widgets/main_window.py` | 16 |
| 11 further modules | 54 |

Adding three more layouts on top of that would multiply the problem. So the
first substantive step of this plan is a single `layout` module that every other
module resolves paths through. Everything else becomes small once that exists.

---

## 2. Survey of the test collection

**20 Autolamella projects**, in 4 collection directories plus 2 at the top
level, spanning four layout eras. Plus **1 pre-Autolamella dataset**.

| era | example | lamella | stage | image dirs |
|---|---|---|---|---|
| pre-Autolamella | `20240228_adaptive_milling_ai/lam_010/SEM_8/` | `lam_010` | *absent* | `SEM_8`, `SEM_8_fi_1`, `SEM_8_fi_4`, `SEM_8_fi_16`, `FIB` |
| 2024-08 | `20240806_auNP/grid12-.../01-real-marmot/adaptive_polish/sem/` | ✅ | untimestamped | `sem`, `fib` |
| 2025-04 / 2025-07 | `.../01-full-gannet/adaptive_polish_2025-04-15-02-48-15AM/sem/` | ✅ | timestamped | `sem`, `fib` |
| 2026-07 | `.../01-neat-ape/Milling/Polishing/adaptive_polish_.../sem/` | ✅ | nested + timestamped | `sem`, `fib` |

Counts: 108 `sem` dirs, 108 `fib` dirs, 14 label dirs.

### 2.1 Scope

**The indexable unit is Adaptive Polishing SEM and FIB images and their
labels.** Other Autolamella images — alignment references, fiducial and
rough-milling images, overviews — are read for *context* only and are never
indexed as segmentation targets. Autolamella images are not segmented.

This is what justifies the "a stage is a directory containing image
directories" heuristic: today it selects exactly the adaptive-polish stages and
nothing else. It is an inference rather than a guarantee, so `inspect-layout`
should report any indexed stage whose folder name is not `adaptive_polish*`,
making a future layout that breaks the coincidence visible immediately rather
than silently widening scope.

### 2.2 Findings that change the plan

**(a) All 20 Autolamella projects have `experiment.yaml`.** The Phase 0
assumption is confirmed, so only the one pre-Autolamella dataset needs a
hand-written marker. Note that `protocol.yaml` is **not** a reliable marker —
`AutoLamella-2025-04-15-08-37_v3` and `FIB/AutoLamella-2025-07-21-13-46_AMtest2`
have `experiment.yaml` but no `protocol.yaml`. (`data_structures.txt:20`
identifies old data by the *absence* of `protocol.yaml`; that test would
misclassify two Autolamella projects here.)

**(b) Projects are nested inside collection directories.** `20240806_auNP/`,
`20250414_adaptive_milling/`, `20260710_CB_AM_rosalind_LAM/` and `FIB/` each
hold several projects. `FIB/` also holds `overviews/` and `protocols_paper/`,
which are not projects. Discovery by rglob already handles this; no
`experiment.yaml` appears below a project root, so phantom nested projects are
not a live risk. This does have a metadata consequence — see (e).

**(c) Two different projects share one identity — a live bug.**
`AutoLamella-2025-07-21-15-21_AM01` exists both at the top level (6.3 GB) and
under `FIB/` (6.4 GB). Their `experiment.yaml` files carry *identical*
`name`, `created_at` and `path`:

```
created_at: 1753107713.652424
name: AutoLamella-2025-07-21-15-21_AM01
path: Z:/group/user/20250721_AM_vs_nonAM\AutoLamella-2025-07-21-15-21_AM01
```

`ProjectIndex.projects` is keyed by name (`scanner.py:32`, `add()` at
`scanner.py:34`), so scanning this collection today **silently discards one of
the two 6 GB projects**. Nothing in the metadata can disambiguate them; only the
filesystem path can. §5.1 addresses this.

**(d) No Autolamella project contains any labels.** All 14 label directories are
in `20240228_adaptive_milling_ai`. The labels for the 20 Autolamella projects
live outside this collection, in hand-made folder structures, and must be
brought in — the manual work you described. §7 explains why this plan
deliberately does *not* design that step yet, and what it commits to now to make
it cheap later.

**(e) Metadata is per-project.** Collection directories group projects by
session or by storage convenience, *not* by sample: lamellae in one directory
may have been prepared on different samples. So metadata must be recorded per
project, and a scheme that resolves it from a parent directory at scan time
would quietly attribute the wrong sample to a project. §8.2 handles the
repetition at file-generation time instead, which keeps each project's metadata
local and explicit.

**(f) `lamella_type` is derivable for most, but not all, projects.**
`experiment.yaml` carries `method:` — `AutoLamella-OnGrid` in most projects,
`AutoLiftout` in the 2024-08 grid projects — which maps directly onto
`lamella_type` (`cells-on-grid` / `lift-out`).

**The 2026 schema drops `method:` entirely.** Confirmed by running the survey:
all four 2026 projects report no method, and their top-level keys are `_id`,
`created_at`, `landing_positions`, `metadata`, `name`, `path`, `positions`,
where the 2025 file also has `method`. So derivation covers 16 of 20 projects
and the remaining four need `lamella_type` entered by hand.

The same comparison turns up something useful: **the 2026 `experiment.yaml`
carries an empty top-level `metadata: {}` block.** Autolamella has reserved the
key we intend to use. If it starts populating it, our `AM_metadata.yaml`
`metadata:` block layers onto it directly with no schema conflict — provided
the merge is per key rather than whole-block, which §8.1 already requires.
Worth re-checking whenever Autolamella is upgraded.

**(g) `Alignment/` subtrees hold images but no `sem`/`fib` subdirectories.**
In the 2026 layout, `01-neat-ape/Alignment/…` and
`Milling/Polishing/Alignment/…` contain alignment reference images directly.
The current stage heuristic (a directory containing `sem` or `fib`) already
ignores them, and per §2.1 that is correct and stays.

---

## 3. What breaks today, concretely

I traced each layout through `Scanner.build_index()`.

### 3.1 Project discovery is `experiment.yaml`-only

`Scanner.find_experiment_files()` (`scanner.py:67`) rglobs for `experiment.yaml`.
`20240228_adaptive_milling_ai` has no such file, so it is invisible.

### 3.2 Collapsed hierarchies are not handled at all

`build_index` requires project → lamella → stage, and `_discover_stages`
(`scanner.py:132`) only considers directories *below* the lamella
(`lamella_path.rglob("*")`), never the lamella directory itself. In
`20240228_adaptive_milling_ai`, `lam_010` **is** the stage, so even with a
marker file it would index nothing.

### 3.3 Image directory names are assumed lowercase and exact

`_discover_images_in_stage` (`scanner.py:240`) looks only for literal `sem` and
`fib`. The legacy dataset uses `SEM_8`, `SEM_8_fi_1`, `SEM_8_fi_4`,
`SEM_8_fi_16`, `FIB`, and label directories `labels` / `labels_FIB`.

### 3.4 Label lookup is keyed by filename per *stage*, not per image directory

`collect_by_name` builds `{filename: [paths]}` for the whole stage. In
`lam_010`, five SEM directories all contain `img_0004.tif`. Any lookup keyed by
filename alone is ambiguous the moment a second image directory exists. This
must become **(image directory, filename)**.

### 3.5 Extra images per acquisition are silently discarded from the UI

`frames._group_records_by_acquisition` (`frames.py:148-157`) detects this case
and drops all but the first, with a warning. That was the right call at the
time, but four of five imaging conditions never reach the GUI. Since you intend
to re-add multi-condition acquisition to Adaptive Polishing, this stops being a
legacy concession and becomes a first-class feature.

### 3.6 Shared labels break the state model

State is physical location (`operations/selection.py:42`, `labels.py:10`).
`transition_records` moves an image and its labels together. If
`SEM_8_fi_1/img_0004.tif` is deselected and its label lives in a `labels/`
directory shared with four other image directories, moving that label silently
changes the effective state of four images nobody asked about. §6 addresses it.

### 3.7 Project identity collision

See §2.2(c). Path-keying is required, not optional.

### 3.8 What already works and should not be disturbed

- Stage discovery is depth-agnostic, so `Milling/Polishing/adaptive_polish_*`
  needs no special case — the 2026 layout already works.
- Untimestamped (`adaptive_polish`) and timestamped stage names both sort
  correctly; `_stage_sort_key` puts undated stages last.
- FIB files misnamed with an `_SEM` suffix (the 2024-10-07 example in
  `data_structures.txt`) are already handled: `frames.py` groups on the parsed
  image *number*, not the stem.
- Collection directories and non-project siblings (`FIB/overviews/`) are
  handled by rglob discovery.

---

## 4. Conventions (settled)

### 4.1 Label directory naming — **D2 answered**

**A label directory is named `<image-directory-name>_labels`, as a sibling of
the image directory. `<modality>_labels` is the shared directory for that
modality; `<modality>_<variant>_labels` is private to that variant.**

```
lam_010/
  sem/                 sem_labels/          <- shared: used by sem/ and by any
  sem_kv5/                                     sem_* directory with no private
  sem_lowmag/          sem_lowmag_labels/   <- private to sem_lowmag/
  fib/                 fib_labels/
```

Resolution order for image directory `D` with modality `M`:

1. `D_labels` if it exists → **private**
2. `M_labels` if it exists → **shared**
3. legacy aliases (`labels` → SEM, `labels_fib` / `labels_FIB` → FIB) → shared
4. `D_labels`, created on first write

Why this shape:

- The image-dir → label-dir mapping is a pure string function in both
  directions, with no lookup table. That is what lets one `layout` module
  replace 200-odd hardcoded strings.
- It removes the `labels` / `labels_fib` asymmetry, which forces a branch at
  every call site.
- Sibling directories keep the existing state mechanism untouched:
  `sem_lowmag_labels/deselected/` works exactly like `sem/deselected/` does
  today. A `labels/<imagedir>/` subtree would push state folders three deep and
  gains nothing.
- `sem` and `sem_labels` sort adjacently in a file browser.

**Image directories keep their Autolamella names** (`sem`, `fib`). We do not own
those, and renaming them would break reopening the project in Autolamella. Label
directories are ours, so those are the only ones that migrate.

### 4.2 Imaging-condition variants

**An image directory is named `<modality>` or `<modality>_<variant>`,**
modality ∈ {`sem`, `fib`}, case-insensitive, with legacy aliases.

| directory | modality | variant |
|---|---|---|
| `sem`, `SEM` | SEM | *(base)* |
| `fib`, `FIB` | FIB | *(base)* |
| `SEM_8` | SEM | `8` |
| `SEM_8_fi_1` | SEM | `8_fi_1` |
| `sem_kv5`, `sem_defocus`, `sem_lowmag` | SEM | `kv5`, `defocus`, `lowmag` |

**The variant is an opaque string.** The package never parses it, so
`sem_5kv`, `sem_defocus`, `sem_lowmag` and anything else you choose all work
without code changes. Authoritative imaging parameters come from the image
headers, which is where they belong; the directory name only has to be unique
and stable.

Two constraints follow:

- **Reserved suffixes.** A variant may not be named `labels`, `deselected`,
  `corrected` or `predictions`, since `sem_labels` etc. would collide with the
  §4.1 rule. `layout.py` validates this and `inspect-layout` reports a violation
  rather than mis-indexing.
- **Optional human-readable descriptions.** Because the variant is opaque, the
  GUI would otherwise show `kv5`. `AM_metadata.yaml` may name them:

  ```yaml
  variants:
    kv5:     "5 kV landing energy"
    lowmag:  "low magnification overview"
  ```

  Purely for display; absent descriptions fall back to the raw variant string.

### 4.3 Project marker file — **D1 answered**

**A directory is a project if it contains `experiment.yaml` or
`AM_metadata.yaml`.** Recursion stops at a project root.

`AM_metadata.yaml` rather than a fabricated `experiment.yaml`, because
`experiment.yaml` is Autolamella's file — writing one misrepresents provenance
and could collide if Autolamella ever opens that directory — and because
`AM_metadata.yaml` is already the override layer in `CLAUDE.md`, so one
hand-written file does both jobs.

Given §2.2(a), this affects exactly one dataset today:
`20240228_adaptive_milling_ai`.

---

## 5. Model changes

### 5.1 Project identity — path, not name

`ProjectIndex.projects: Dict[str, Project]` keyed by name becomes keyed by
**path relative to the scan root**, which is the only thing that distinguishes
the two `AutoLamella-2025-07-21-15-21_AM01` projects (§2.2c).

Consequences:

- `Project` gains `relative_path` and keeps `name` for display.
- `display_name` shows the disambiguating parent when two projects share a name
  — `FIB/AutoLamella-2025-07-21-15-21_AM01` rather than a bare, ambiguous name.
- `filtering.matches_project` (`filtering.py:74`) matches on relative path,
  name, or scan-order number. Selecting by name stays available but must warn
  when it is ambiguous instead of quietly picking one.
- Scan-order project numbers (`001`, `002`, …) become the reliable CLI selector,
  which is what they were introduced for.

This is a small change with an outsized payoff: without it, scanning the test
collection loses 6 GB of data with no error message.

### 5.2 New module: `layout.py`

Pure functions and small frozen dataclasses. No scanning, no I/O beyond
`Path.exists()`. The only module that knows directory names.

```python
MODALITIES = ("SEM", "FIB")
STATE_FOLDERS = {"selected": None, "deselected": "deselected", "corrected": "corrected"}
LEGACY_LABEL_ALIASES = {"labels": "SEM", "labels_fib": "FIB", "labels_FIB": "FIB"}
RESERVED_VARIANTS = {"labels", "deselected", "corrected", "predictions"}

@dataclass(frozen=True)
class ImageSet:
    """One directory of images acquired under a single imaging condition."""
    directory: Path
    modality: str            # "SEM" | "FIB"
    variant: str | None      # None for the base directory
    labels_directory: Path
    labels_are_shared: bool  # more than one ImageSet resolves to labels_directory

def classify_image_directory(path) -> tuple[str, str | None] | None: ...
def labels_directory_for(image_dir, modality) -> Path: ...
def predictions_directory_for(image_set) -> Path: ...
def state_directory(path, state) -> Path: ...
```

Everything currently hardcoding a directory name calls into this instead. The
refactor is mechanical and testable in isolation — the highest-value step in the
plan, and what makes each newly discovered layout cheap.

### 5.3 `models.py`

```python
@dataclass
class AdaptiveMillingStage:
    ...
    image_sets: list[ImageSet] = field(default_factory=list)
    synthetic: bool = False   # stage level absent on disk; stands in for the lamella dir

@dataclass
class Lamella:
    ...
    synthetic: bool = False   # lamella level absent on disk; stands in for the project dir

@dataclass
class ImageRecord:
    ...
    variant: str | None = None
    label_shared: bool = False
    group_key: str | None = None   # see §6
```

Collapsed hierarchies are handled by **synthesising** the missing levels in
memory rather than asking you to move data. For `20240228_adaptive_milling_ai`,
`lam_010` becomes both the lamella and a synthetic stage holding five SEM image
sets and one FIB image set. Displays mark synthetic levels, so nothing pretends
a stage folder exists on disk.

### 5.4 `frames.py`

`LamellaFrame` currently holds `sem: ImageRecord | None`. It becomes an ordered
list per modality, base variant first:

```python
@dataclass(frozen=True)
class LamellaFrame:
    sem: list[ImageRecord]      # was: ImageRecord | None
    fib: list[ImageRecord]
    ...
    def record(self, modality, variant=None) -> ImageRecord | None: ...
```

The warn-and-drop branch at `frames.py:148` is deleted. Every current project is
single-variant and produces one-element lists, so this is a shape change rather
than a behaviour change for existing data.

---

## 6. Shared labels and linked state — **D4 answered**

**The unit of proofreading is the label, not the image.** Five SEM images of the
same scene under different imaging conditions share one segmentation because
they *are* one segmentation. "Do not proofread this scene" is a statement about
the scene, so it necessarily applies to all five. Linked state is the correct
semantics, not a compromise.

### 6.1 The label group

```python
@dataclass(frozen=True)
class LabelGroup:
    label_path: Path          # the shared (or private) label
    records: tuple[ImageRecord, ...]
```

Group key = `(labels_directory, filename)`. A private label yields a group of
one, so there is a single code path rather than a special case.

### 6.2 State authority

**The label's location determines the group's state.** Where a group has no
label yet (before prediction conversion), the image's own location determines
it:

| group has label | state read from |
|---|---|
| yes | label directory (`sem_labels/deselected/img.tif` → deselected) |
| no | image directory |

A scan can find a group whose images disagree with the label — files moved by
hand, or an interrupted operation. That is reported as **inconsistent**, never
silently resolved: `inspect-layout` lists them, and a `repair-state` command
moves images to follow the label, with `--dry-run` and no overwriting.

### 6.3 Transitions

`transition_records` gains a group-expansion step in front of its existing
logic:

1. Expand the requested records to the closure over their label groups.
2. Plan every move (unchanged).
3. Validate the whole batch before moving anything (unchanged).
4. Refuse to move corrected data without confirmation (unchanged).
5. Move.

Every existing safety property is preserved exactly; only the *set* they apply
to changes. Callers get the expanded set back so they can report "deselecting
this image also deselected 4 others" — the GUI must show that count before
acting, and the CLI must print it.

`--dry-run` should extend from `compress-labels` to `select`/`deselect`/
`correct`, because with group expansion the blast radius of a state change is no
longer obvious from the command line.

---

## 7. Importing labels — deferred, deliberately

§2.2(d): none of the 20 Autolamella projects contains labels. They live in
folder structures you created by hand, and bringing them back into the project
hierarchy is manual work you expect to do with helper scripts.

An earlier draft proposed an `import-labels` command as a full phase of this
plan. **It is deferred instead**, because designing a matching strategy without
having seen the source structures would be speculation, and a bulk
filesystem-writing command built on a guessed input shape is the worst thing to
have in a plan whose whole point is not losing data.

What *is* worth committing to now, because it costs nothing and makes the
eventual import cheap:

- **The destination is unambiguous.** §4.1 fixes where a label for a given image
  must go: `<image-dir>_labels/`, resolved by `layout.py`. Any helper script you
  write, in this package or outside it, can call
  `layout.labels_directory_for(...)` and get the right target without
  reimplementing the convention.
- **Verification is available from Phase 0.** `inspect-layout` reports images
  with no label and labels matching no image, so after any import — by script,
  by hand, or by a future command — you can check the result rather than trust
  it.

When the source structures are known, the natural shape is a dry-run-first
command matching on filename within a selected scope, refusing to overwrite and
never deleting from the source, and reporting unmatched and ambiguous cases
before writing anything. That is a small amount of work *given* `layout.py`, and
it is much better specified once there is a real input to look at. Promote it
to a phase then.

---

## 8. Metadata

### 8.1 Typed metadata

`metadata.py` gains a dataclass matching `extended_metadata_example.yaml`:

```python
@dataclass(frozen=True)
class ExperimentMetadata:
    description: str | None = None
    organisation: str | None = None
    project: str | None = None
    user: str | None = None
    sample: str | None = None
    lamella_type: str | None = None   # cells-on-grid | waffle | lift-out | other
    source: dict[str, Path] = field(default_factory=dict)   # field -> file it came from
```

`source` matters: with several files layered, you need to know which one supplied
a value when something looks wrong. The raw merged dict stays available for
Autolamella keys we do not model.

`lamella_type` is derived from `experiment.yaml`'s `method:` when present
(§2.2f): `AutoLamella-OnGrid` → `cells-on-grid`, `AutoLiftout` → `lift-out`. An
explicit `AM_metadata.yaml` value always wins. The 2026 schema has no `method:`,
so four projects need it entered by hand; `init-project` leaves it as a `TODO`
rather than guessing.

**Metadata is project-scoped, and stays that way.** All lamellae in one
Autolamella project come from one sample, so every field above belongs on
`Project`. No per-lamella metadata is modelled.

The one plausible future extension is per-lamella milling parameters, which is
explicitly out of scope for now. Worth noting what it would take, so the current
design does not foreclose it: Autolamella already records milling parameters in
`experiment.yaml` and `protocol.yaml`, so they would be *derived* rather than
hand-entered. That means a new dataclass attached to `Lamella` and a reader for
those files — no change to the `AM_metadata.yaml` schema, no new hand-editing,
and no revision of anything written before then. Nothing in this plan needs to
anticipate it.

### 8.2 One file per project, generated in bulk

An earlier draft proposed resolving metadata from parent directories at scan
time, so one file per collection would cover the projects beneath it. **That was
wrong**: collection directories group projects by session or storage
convenience, not by sample, and lamellae in one directory may be prepared on
different samples (§2.2e). Inheritance would silently attribute one project's
sample to another, and the error would be invisible — precisely the kind of
quiet wrong answer that is worst in a metadata system feeding a data
repository deposition.

So metadata resolution stays exactly as `CLAUDE.md` already specifies, with no
new concept:

1. `experiment.yaml` (Autolamella's own values)
2. `AM_metadata.yaml` **in the project root** — overrides the above

Every project has its own file, and a project's metadata is always readable from
one place.

The repetition is handled at *generation* time instead, where a wrong value is
visible in the file and can be corrected:

```
am-seg init-project <collection> --recursive [--defaults shared.yaml]
```

writes one `AM_metadata.yaml` per project, pre-filling everything derivable
(project name, dates, `lamella_type` from `method:`, detected image sets) and
leaving the rest as explicit `TODO` markers. `--defaults` copies genuinely
shared values — typically `organisation`, sometimes `user` — **into each file**
rather than resolving them at scan time. Fields that vary, above all `sample`
and `description`, are never pre-filled from a defaults file; they stay `TODO`
so an unfilled one is obvious.

`inspect-layout` reports any project whose metadata still contains `TODO`
markers, so the remaining work is a checklist rather than something you have to
remember.

This is more files than the inheritance scheme (21 rather than 6) but almost the
same amount of *typing*, since a tool writes them — and it cannot get a sample
wrong by inference.

### 8.3 Layout overrides

`AM_metadata.yaml` gets an optional `layout:` block as the escape hatch for
projects where inference is wrong. Used sparingly — every override is a
convention we failed to capture:

```yaml
metadata:
  user: Casper Berger
  sample: HeLa
  lamella_type: cells-on-grid
variants:
  fi_1: "frame integration 1"
layout:
  image_directories:
    SEM_8_fi_1: {modality: SEM, variant: fi_1}
  labels:
    SEM_8_fi_1: labels        # explicit share, overriding the derived rule
```

### 8.4 Performance note

`experiment.yaml` in the 2025-07-21 project is **681 KB**, and `build_index`
parses it in full for every project on every scan. With 20 projects on OneDrive
this will dominate scan time. Extracting only the keys we use, or caching the
parse, is a small change with a large effect — and it becomes noticeable exactly
when this plan lands 20 projects at once.

---

## 9. Phased implementation

Each phase leaves the package working and independently testable, per
`CLAUDE.md`.

**Phase 0 — survey tooling. ✅ Done.** `am-seg inspect-layout <dir>` reports,
per project, the detected hierarchy, image sets, modality/variant, resolved
label directory and whether it is shared, plus unmatched files, unrecognised
directories, unexpected stage names, duplicate project identities and
images/labels without a counterpart. Inconsistent state groups are left to
phase 4, where the state model they depend on is built.

`layout.py` was written here rather than in phase 1: inspection *is* layout
logic, so building it on throwaway classification would have duplicated the
conventions this plan exists to centralise.

**Phase 1 — refactor onto `layout.py` + path-keyed identity. ✅ Done.**
`scanner.py`, `operations/{labels,selection,csv,predictions,label_compression}`,
`filtering.py` and `reporting.py` now resolve every directory name through
`layout`. `ProjectIndex` is keyed by relative path (§5.1), `find_by_name` raises
on an ambiguous name instead of guessing, and `Project.display_name` shows the
relative path.

Verified on the real collection: 20 projects, 2444 images, 4.5 s. Both
`AutoLamella-2025-07-21-15-21_AM01` projects are indexed with their own 308
images each — name-keying was dropping 308 images silently.

*One deliberate behaviour change*: when a project has **no** label directory
yet, a newly written label now creates `<image-dir>_labels/` rather than
`labels/` or `labels_fib/`. An existing legacy directory is still used as-is, so
no project ends up with its labels split across two directories. This follows
D2/D3 — write the current convention, read the legacy names indefinitely — and
is pinned by a test.

**Phase 2 — project discovery and metadata. ✅ Done.** `AM_metadata.yaml` is a
second project marker; `ExperimentMetadata` is the typed view, resolved per key
with `method:` → `lamella_type` derivation; absent lamella and stage levels are
synthesised; `am-seg init-project [--recursive] [--defaults] [--dry-run]`
writes one pre-filled file per project. Traversal moved into a new
`discovery.py` shared by the scanner and the survey, so both agree on what a
project is.

Verified: `20240228_adaptive_milling_ai` indexes once given a marker — 7
lamellae, 226 images, 226 labels, with synthetic stages and the shared `labels/`
directory resolved. `lamella_type` derives for **16 of 20** projects; the four
2026 ones need it by hand, as §2.2(f) predicted.

*Scan cost solved differently than planned.* §8.4 proposed extracting only the
keys we use. Measurement showed 4.1 s of the 4.5 s scan was YAML parsing, and
that PyYAML's libyaml loader is ~7x faster with identical results — so
`io/yaml.py` now prefers `CSafeLoader`, and full-fidelity documents are kept.
**Whole-collection scan: 4.5 s → 1.2 s.** No partial parsing, no heuristics.

*A discovery flaw found by a test and fixed.* Stopping project discovery at the
outermost marker meant a marker file in a **collection** directory silently
swallowed every project beneath it: their stages were absorbed as that
directory's lamellae and the real projects vanished. Given the natural instinct
is to write one metadata file per collection, this was likely to be hit. The
rule is now: discovery continues past a project root, stage discovery never
crosses a project boundary, and nesting is reported as an error
(`nested-project`). Splitting is visible and recoverable; swallowing was
neither.

Two related subtleties, both pinned by tests: directory *roles* are only
meaningful inside a stage — the collection directory named `FIB/` holds five
real projects and a name-based filter skipped it entirely — and outside a
project a directory is only treated as a stage if it actually **contains
images**, not merely because it holds a directory named like one.

**Phase 3 — image sets and variants. ✅ Done.** The scanner indexes every image
set, attaches `ImageSet`s to each stage, and records each image's `variant` and
whether its label is `label_shared`. `LamellaFrame.sem`/`fib` are now tuples
ordered base-first, with `record(modality, variant=None)`, `base_records` and
`variants()`; the warn-and-drop branch is gone. `filtering` gains
`matches_variant` and the CLI a `--variant` selector (`base` selects the
unsuffixed directory, `all` everything, default everything).

Verified on the legacy dataset's real structure: **226 images indexed, up from
45** — it was FIB-only before, because the four SEM conditions were skipped.
One acquisition now forms one frame carrying all four SEM variants
(`8`, `8_fi_1`, `8_fi_16`, `8_fi_4`) plus its FIB image, with all four SEM
records referencing the same shared label in `labels/`. The Autolamella
collection is unchanged: 20 projects, 2444 images, 1.2 s, no variants present.

*Frame shape is a breaking model change.* `frame.sem` was a record and is now a
tuple. `frame.record("SEM")` has exactly the old meaning and is what call sites
were migrated to. The GUI continues to display the base image of each modality
(`base_records`), so there is no visible change until phase 5 adds a variant
selector.

*Not yet variant-aware, by design*: a state change from the GUI still targets
the base image only, and CSV export emits one row per variant. Both are
addressed by the phases that follow — label groups (4) make a state change
follow the shared label, and the CSV variant policy (6) makes "five conditions
of one scene" an explicit choice rather than an accident.

**Phase 4 — label groups and linked state. ✅ Done.** `operations/label_groups.py`
holds `LabelGroup`, built from the filesystem so a command given a single file
path still expands correctly. `plan_transition` validates a whole change up
front and reports how far it reaches; `transition_records` expands across shared
labels before moving anything. `--dry-run` is on `select`/`deselect`/`correct`,
`repair-state` is new, and the survey reports `state-mismatch`.

Verified on the legacy structure — deselecting *one* imaging condition:

```
3 further image(s) share a label with your selection and will also be moved:
  lam_010/SEM_8_fi_1/img_0004.tif
  lam_010/SEM_8_fi_16/img_0004.tif
  lam_010/SEM_8_fi_4/img_0004.tif
Dry run: 5 file(s) would move, affecting 4 image(s)
```

All four conditions and the single shared label move together; the FIB image,
whose label directory is its own, is untouched.

*The shared label is moved once, not once per image.* Each of the four records
references the same label, so the naive plan contained four identical moves and
the existing collision check rejected them. Identical moves now collapse; two
*different* files heading for one destination remains an error.

*Validation runs in dry-run too.* A rehearsal that skipped the corrected-data
guard or the collision checks would be worse than useless, so `--dry-run`
executes the same `plan_transition` and raises the same errors — it only stops
short of moving.

*State authority is implemented as specified in §6.2*: the label's location
decides, the image's own location only when no label exists. A group whose
images disagree is reported rather than silently resolved, and `repair-state`
moves the *images* to follow the label — never the reverse, since the label
holds the proofreading work.

**Phase 5 — GUI. ✅ Done.** Built as designed. Full design in
[`gui_plan.md` §17](gui_plan.md). The short version, because the design turned
out smaller than this plan assumed:

*The variant is not a fourth edit axis.* Since §6 made state changes follow the
label, choosing which variant to edit is either meaningless (they share a label
and all move) or already implied (they have private labels, and the one on
screen is the one meant). So the variant becomes a **view** control in the image
panel — a dropdown, hidden when there is only one variant, which is every
Autolamella project in the collection — and the state panel gains **no
selector**. It gains a reach line: "Moves 4 images + 1 label / 3 share a label
with this one", always shown, above the state buttons, counted from
`plan_transition` so it cannot drift from what the operation actually does.

*No third dialog.* Shared labels are confirmed by the reach line being on screen
before the click, not by a new prompt. One fix is needed instead: `_confirm_bulk`
counts `scoped_records()`, which excludes images pulled in by a shared label, so
it currently under-reports. It must count the expanded set.

*Mismatches are reported, not fixed, in the GUI.* The frame strip marks a frame
whose images disagree with their label; `repair-state` stays CLI-only, where it
has `--dry-run`.

Decisions taken: reach line always visible; dropdown rather than segmented
buttons; repair CLI-only.

*The `_confirm_bulk` under-report was fixed first, on its own.* It counted
`scoped_records()`, so at lamella or project scope on data with shared labels it
promised to move fewer files than it would. It now counts
`controller.expanded_records()` and says how many were pulled in by a shared
label.

*Cache entries are keyed by variant*, so switching between two conditions to
compare them evicts neither. A frame that lacks the viewed condition falls back
to its base image rather than blanking the pane, which matters when only some
acquisitions in a lamella were captured under a given condition.

*A test that passed for the wrong reason.* The first widget tests asserted
`isVisible() is False` for the hidden variant control — but a child of a window
that was never shown is never visible, so that assertion would have passed
whether or not the control was hidden. They now use `isHidden()`, which reflects
the explicit state.

**Phase 6 — migration and downstream operations.** `migrate-layout --dry-run`
renames the 14 legacy label directories. CSV export gains `--variants
base|all|<name>` (`all` is genuinely useful: five imaging conditions of one scene
sharing a label are five valid training pairs). Label compression deduplicates by
label path so a shared label is not processed five times.
`check-training-csv` learns the new state locations.

**Deferred — label import (§7).** Promoted to a phase once the source label
structures are known.

Phases 1–3 unblock indexing; the GUI work deliberately comes after. Note that
Phases 4 and 5 only pay off once labels exist in the projects, so if the label
reorganisation slips, Phase 6's CSV and migration work is the more useful thing
to pull forward.

---

## 10. Answers to your questions, and what is still open

### Should you add `AM_metadata.yaml` to the pre-Autolamella project now?

**Not by hand — wait for `init-project` in Phase 2.** It is one file, and the
tool will pre-fill everything derivable (project name, dates, detected image
sets) so you only type what genuinely cannot be inferred: `description`,
`organisation`, `user`, `sample`, `lamella_type`. Hand-writing it now risks
mismatching the final schema and needing a rewrite.

If the metadata is fresh in your memory and you would rather not lose it, jot it
in a plain text note and I will fold it into the generated file.

### Should you add `AM_metadata.yaml` to the Autolamella projects?

**Not as marker files — they are unnecessary.** All 20 have `experiment.yaml`
(§2.2a), so they are already discoverable.

**Yes, eventually, as metadata enrichment** — one file per project, after
Phase 2, generated in bulk by `init-project --recursive` rather than written by
hand (§8.2). 21 files, but the tool writes them pre-filled and
`inspect-layout` tells you which still have `TODO` fields outstanding.

`lamella_type` comes from `method:` automatically, so what you actually fill in
per project is `description`, `sample`, and — if `--defaults` has not already
copied them in — `organisation`, `project`, `user`.

### D3, restated — this was my unclear question

I was asking permission to **rename the label directories in your data**:
`labels` → `sem_labels`, `labels_FIB` → `fib_labels`, via a `migrate-layout`
command with `--dry-run` and no overwriting.

The survey mostly dissolves the question. The only label directories on disk are
the 14 in `20240228_adaptive_milling_ai` (7 lamellae × 2). Everything else is
labels you have yet to bring in, which will be written into the new naming from
the start. So it is 14 renames, in Phase 6, previewed first — and the legacy
aliases stay readable indefinitely in case older copies resurface. Nothing is
lost if you would rather leave them; the aliases mean it works either way.

### Answered — no longer open

- **`Alignment/` images are not indexed.** Recorded as a scope statement in
  §2.1: the indexable unit is Adaptive Polishing SEM/FIB images and their
  labels; other Autolamella images are context only and are never segmented.
- **Label sources.** Deferred rather than designed speculatively — see §7.
- **Metadata is project-scoped.** All lamellae in one Autolamella project come
  from one sample, so no per-lamella metadata is modelled (§8.1). Per-lamella
  milling parameters are a possible future addition, out of scope for now, and
  the design does not need to anticipate them.

### Still open — neither is blocking

1. **The two `AutoLamella-2025-07-21-15-21_AM01` copies** (§2.2c) — is one a
   working copy that should be removed, or are both intentional? Path-keying
   makes the scanner handle both correctly either way.
2. **`20231102` and `20240223` layouts** from `data_structures.txt` are not in
   the collection. If those datasets still exist, adding one of each would let me
   cover the fully-collapsed (`20231102/SEM/`) case with a real fixture rather
   than a synthetic one. Otherwise I will build synthetic fixtures from the path
   examples.

---

## 11. Testing

The layout variety *is* the risk, so the test structure should mirror it: a
fixture builder parameterised by layout era, with one shared behavioural suite
run across all of them.

```python
@pytest.fixture(params=["v2026_nested", "v2025_timestamped", "v2024_untimed",
                        "legacy_variants", "legacy_collapsed"])
def project(request, tmp_path): ...
```

Discovery, image/label pairing, state transitions and round-trips, and CSV
export then run identically against every era. New layouts found by
`inspect-layout` get added as a param, and a bug found in one era is
automatically checked against all.

Targeted tests beyond that:

- `layout.py` name classification and label resolution (pure, exhaustive),
  including reserved-variant rejection
- two projects with identical names and metadata both survive indexing
- metadata precedence: project `AM_metadata.yaml` overrides `experiment.yaml`,
  and no value is ever resolved from a parent directory at scan time
- `init-project --defaults` copies shared fields into each file but never
  pre-fills `sample` or `description`
- group expansion: deselecting one variant moves all five images and one label
- inconsistent groups are detected and not silently repaired
- corrected-label protection still holds through group expansion
- synthetic lamella/stage levels produce stable numbering and display names
- `Alignment/` and other non-adaptive-polish directories are never indexed, and
  an indexed stage not named `adaptive_polish*` is reported

---

## 12. Risks

- **Scan cost.** `_discover_stages` rglobs the whole lamella subtree and stats
  every directory. With 20 projects on OneDrive — where reads intermittently
  time out — this will be slow. Phase 1 should add a depth bound and stop
  descending into directories already identified as image or label directories.
- **Metadata is the easiest thing to get quietly wrong**, because a wrong
  `sample` looks exactly like a right one. §8.2 is deliberately conservative for
  that reason: one file per project, no inference across directories, and
  unfilled fields left as visible `TODO`s rather than defaulted. The same
  caution should apply to anything later added to the metadata model.
- **The GUI change in Phase 5 is the largest UI change so far** — a second
  selector axis (variant) on top of mode/target/scope. Worth a short design pass
  on the state panel first, so it does not accumulate four independent dropdowns.
- **Label import, whenever it happens, is the highest-consequence filesystem
  operation** in the workflow, since it writes into project directories in bulk.
  Whether it ends up as a command or your own helper scripts, it should follow
  the existing discipline: validate the whole batch, never overwrite, dry-run
  first, and verify afterwards with `inspect-layout`.
