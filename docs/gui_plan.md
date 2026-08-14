# Design plan: data curation GUI

Status: **phases 1–7 built and working.** Phase 7 was rewritten after a first
round of testing and then implemented — see [§13](#13-phase-7-revised-after-testing).

Target: a desktop viewer for reviewing Autolamella adaptive-milling images against
their labels, and setting each image's workflow state (selected / deselected /
corrected) without leaving the viewer.

This document explains *what* was built and *why* each choice was made. Sections
1–12 are the original design, with measured numbers substituted for the initial
estimates where testing produced them. Section 13 supersedes the phase 7 row in
[Phased implementation](#8-phased-implementation).

Two decisions recorded in §1 have since been overtaken by §13: the lamella
slider is to be removed, and the frame slider merged into the state strip.

---

## 1. Scope

In scope:

* Choose a scan root, scan it with the existing `Scanner`, and pick a project
  from the discovered list.
* Per lamella, present **one chronological stack** of frames spanning all of
  that lamella's adaptive-milling stages.
* For each frame, show the SEM image, the FIB image, each optionally overlaid
  with its label, or the pre-rendered `plots/*.png` for that frame.
* Navigate with two sliders: frame-within-lamella, and lamella-within-project.
* Change the workflow state of the current frame (SEM, FIB, or both) using the
  existing `operations.selection` API.

Explicitly out of scope (call these out now so the design does not quietly grow):

* Any label *editing*. Proofreading stays in the external segmentation software.
  This tool decides *which* images get proofread and records when they are done.
* Prediction→label conversion, CSV export, compression. Those stay CLI-only for
  now; the GUI is a curation tool, not a second front-end for everything.
* The pre-Autolamella legacy layouts in `data_structures.txt` (`SEM_8_fi_1/`,
  `lamella_01/SEM/`). The scanner does not discover them today, so the GUI will
  not show them either. Handling them is a scanner change, tracked separately.

---

## 2. Key decisions and trade-offs

### 2.1 GUI toolkit: plain Qt via `qtpy`

**Decision: a plain Qt application, written against `qtpy` so it runs on
whichever binding is present.**

Considered:

| Option | Verdict |
| --- | --- |
| **napari** | Rejected for the main viewer. |
| **Qt (PyQt5/PySide6) directly** | Rejected — hard-codes one binding. |
| **Qt via `qtpy`** | **Chosen.** |
| Web UI (Flask + browser) | Rejected — adds a server, a second language, and slow transfer of large images for no benefit in a single-user desktop tool. |

Why not napari, despite it already being installed in your environment and being
the obvious choice in this domain:

1. **Variable image shapes.** A napari image layer wants a uniform `(N, Y, X)`
   stack. Your frames span several stages, and stage image dimensions are not
   guaranteed equal (the prediction pipeline already has to resize). One stack
   per lamella would break the moment two stages differ.
2. **The plots mode does not fit.** `plots/*_plot.png` are RGB figures with a
   different aspect ratio from the images. They cannot share a layer or a dims
   slider with the image stack, so you end up hand-rolling that part anyway.
3. **Testability.** napari's viewer state is hard to assert on in pytest; a
   plain controller object is not (see §2.2).
4. **Dependency weight and churn.** napari is a large, fast-moving dependency,
   and `CLAUDE.md` asks for minimal dependencies. Making the core curation tool
   depend on it is a long-term maintenance cost.

napari is still worth using *later*, in one specific place: an **"open in napari"
button** that launches the current frame's image + label in napari for a quick
look, or hands off to the proofreading tool. That is an optional integration,
not the foundation.

Why `qtpy` rather than picking PySide6 outright: your environment already has
PyQt5 (pulled in by napari), and `qtpy` 2.4.1 and `superqt` are already
installed. Loading two different Qt bindings into one process is a well-known
source of hard-to-debug crashes. Writing against `qtpy` means the app uses
whatever binding is already there, and you can migrate to PySide6 (LGPL, better
for open-sourcing) later by changing an install, not the code.

Dependencies added: `qtpy` only, as a new optional extra. Nothing is added to
the core `dependencies` list, so `am-seg` CLI users and CI never need Qt.

```toml
[project.optional-dependencies]
gui = ["qtpy", "numpy", "tifffile", "opencv-python"]
```

Documented note: if the environment has no Qt binding at all, install PySide6
alongside; if it already has PyQt5 (e.g. via napari), do not add a second one.

### 2.2 The single most important structural decision: no Qt below the widgets

Everything the GUI *knows* lives in ordinary Python objects with no Qt imports.
Qt widgets are a thin skin that renders what the controller says and forwards
user events back to it.

```
 user event  ->  CurationController (plain Python)  ->  existing package API
                        |                                 (scanner, selection,
                        v                                  filtering, labels)
                 state snapshot (dataclass)
                        |
                        v
                 Qt widgets (render only)
```

Why this matters here specifically:

* It is the difference between a package you can test and one you cannot.
  Navigation, mode switching, cache eviction, and "which records does 'both'
  apply to" are all logic worth testing, and none of it needs a display server.
* It keeps a future alternative front-end (a napari plugin, a Jupyter widget)
  cheap, because the front-end would only need to re-implement rendering.
* It stops filesystem calls from creeping into widget callbacks, which is the
  usual way GUIs become unmaintainable.

The rule to hold to: **`gui/widgets/*.py` may import Qt; nothing else in the
package may.** A test asserting this is worth writing (`test_no_qt_imports`).

### 2.3 Reuse, not reimplementation

The GUI calls the existing API and adds nothing parallel to it:

| Need | Existing API used |
| --- | --- |
| Discover projects | `scanner.Scanner.build_index()` |
| Filter to a lamella/stage | `filtering.iter_matching_records()` |
| Image number from filename | `filtering.extract_image_number()` |
| Change state | `operations.selection.transition_records()` |
| Corrected-data safety | `transition_records(..., allow_move_from_corrected=)` |
| Read YAML config | `io.yaml.read_yaml()` |
| Relative display paths | `io.path.make_path_relative()` |

Two small changes to existing modules are needed (§5). Both are improvements the
CLI benefits from too, and neither changes existing behaviour.

---

## 3. Module map

New code lives in two places: a headless layer usable by anything, and a `gui/`
subpackage.

```
src/am_segmentation_manager/
├── config.py            NEW  load segmentation_config_AM.yaml -> SegmentationConfig
├── colors.py            NEW  CSS colour name -> RGB tuple (no matplotlib dep)
├── frames.py            NEW  group ImageRecords into per-lamella chronological frames
├── display.py           NEW  headless image loading, downscaling, overlay compositing
├── scanner.py           MOD  discover plots/*.png per stage
├── operations/
│   └── selection.py     MOD  update record paths after a move; add current_state()
├── cli.py               MOD  add `am-seg gui` subcommand
└── gui/                 NEW
    ├── __init__.py
    ├── app.py                Qt application entry point
    ├── controller.py         headless view-model — the brain, no Qt
    ├── cache.py              per-lamella display-frame cache (no Qt)
    ├── loader.py             background loading worker (thin QThread wrapper)
    └── widgets/
        ├── main_window.py    three-panel layout, sliders, wiring
        ├── project_panel.py  left: scan root + project/lamella list
        ├── image_panel.py    middle: image display, mode toggle, legend
        └── state_panel.py    right: edit guard, target, state radio group
```

Only `gui/app.py`, `gui/loader.py` and `gui/widgets/*` import Qt.

---

## 4. New data model: `LamellaFrame`

### The problem

`ImageRecord` is one record per *file*: one for the SEM image, a separate one for
the FIB image, scattered across stages. The GUI needs the opposite view — one
*frame* per acquisition moment, holding both modalities and the plot.

### The model

```python
@dataclass(frozen=True)
class LamellaFrame:
    """One acquisition point in a lamella's chronological milling history."""
    lamella: str
    stage: str                     # stage folder name
    stage_number: str              # "01", "02" … from AdaptiveMillingStage.number
    image_number: int | None       # from filtering.extract_image_number()
    sem: ImageRecord | None
    fib: ImageRecord | None
    plot_path: Path | None

    @property
    def label(self) -> str:        # e.g. "02 · img_013"
        ...
```

`build_lamella_frames(lamella: Lamella) -> list[LamellaFrame]` lives in
`frames.py` and is a pure function over the already-scanned model — no
filesystem access, so it is trivially testable.

### Ordering

1. Stages are already sorted chronologically by `Scanner._stage_sort_key`, which
   parses the timestamp out of the folder name. **However** — the current regex
   ignores the `AM`/`PM` suffix in names like
   `adaptive_polish_2026-07-28-02-43-32PM`. It reads `02-43-32` as 02:43 and
   drops the `PM`. So `…-02-43-32PM` (14:43) sorts *before* `…-11-15-36AM`
   (11:15) — wrong. **This is a bug to fix in phase 2**, not something the GUI
   should work around. Sorting within a lamella is currently correct only by
   luck in some projects.
2. Within a stage, order by `image_number`; frames with no parseable number go
   last, ordered by filename.

### Pairing SEM to FIB

Key on `(stage, image_number)`, **not** on filename. In new projects the SEM file
ends `_SEM.tif` and the FIB `_FIB.tif`, but `data_structures.txt` lines 12–18
show older projects where the file inside `fib/` is *also* named `..._SEM.tif`.
Keying on the extracted image number is robust to both.

Edge cases the builder must handle rather than crash on:

* SEM present, no FIB (or vice versa) → frame with one side `None`.
* More than one record for the same `(stage, number, modality)` — possible in the
  legacy multi-condition layouts. Take the first and log a warning; do not
  silently drop. (Full support is out of scope, §1.)
* No image number parseable → frame keyed by filename stem instead.

---

## 5. Changes to existing modules

### 5.1 `scanner.py` — discover plots

Plot PNGs are currently invisible to the package. The scanner is the single place
allowed to touch the filesystem for discovery, so plot discovery belongs there,
not in the GUI.

Add to `AdaptiveMillingStage`:

```python
plots: List[Path] = field(default_factory=list)
```

populated from `stage_path / "plots"` (`*.png`, sorted). `frames.py` then matches
plots to frames with the same `extract_image_number()` used for images.

This is a defaulted field, so existing tests and callers are unaffected.

Note from the sample data: plot counts are consistently *lower* than image counts
(28 images / 26 plots, 25 / 23, 13 / 12). **Missing plots are normal**, and the
plot mode must handle it gracefully rather than treat it as an error.

### 5.2 `operations/selection.py` — keep records accurate after a move

`transition_records()` currently updates `record.selected` and
`record.corrected`, but leaves `record.image_path` and `record.labels` pointing
at the **old** location. In the CLI this is harmless — the process exits. In a
GUI the record is reused for the next render and the next state change, so it
would try to display and then move a file that is no longer there.

Proposed change: after each successful move, write the new destination back onto
the record. `_plan_record_transition` already computes exactly these paths, so
this is a small, contained change.

This is a behaviour change to a shared module, so it is proposed explicitly
rather than slipped in: it makes `ImageRecord` accurate after the call, which is
what a caller would already assume. It needs its own test
(`test_transition_updates_record_paths`).

Also add a small read-only helper, used by the state panel and worth having in
the CLI too:

```python
def current_state(record: ImageRecord) -> RecordState:
    """Return the record's current on-disk workflow state."""
```

### 5.3 `cli.py` — launch the GUI

```
am-seg gui [START_PATH] [--config PATH]
```

with `--help` text, consistent with every other subcommand. The import of
`gui.app` happens inside the handler so that a missing Qt binding produces a
clear message ("install the 'gui' extra") rather than breaking `am-seg --help`.

---

## 6. Image pipeline

### 6.1 The numbers (measured on `20260728_CB_AM_bitmap_overnight`)

Measured during phase 4, not estimated:

* One acquisition: 2048×3072 **8-bit** greyscale, 6.29 MB on disk. Newer data
  is 8-bit; older projects may be 16-bit, so the pipeline handles both.
* One stage: up to 26 frames. One lamella: up to 3 stages → **23–26 frames** in
  this dataset. Budget for more; a longer polish run scales this linearly.
* A full lamella, both modalities, loaded raw: **~315 MB** and ~50 file reads at
  25 frames. Over OneDrive with cloud-on-demand files, some of those reads are
  slow or blocking: a cold read measured **206 ms**, a warm one 22 ms.
* Summary plots are **3000×4500 RGBA** — 5.6 MB as PNG but ~54 MB decoded, and
  **575 ms** to read. They are the most expensive thing the viewer touches.

### 6.2 Strategy: per-lamella cache of *display-sized* frames, filled progressively

Your instinct — load per lamella — is right, but load the *display form*, not the
raw data:

1. On load, immediately reduce each image to a display frame: downscale so the
   long edge is ≤ 1024 px. **No contrast adjustment** — acquisition already
   sets the intensity range, so 8-bit data is shown exactly as recorded and
   16-bit data is only linearly mapped onto 8 bits so it can be displayed at
   all. → 0.70 MB per image instead of 6.29 MB.
   A lamella of display frames (image + label, both modalities) measured
   **70 MB** at 25 frames. Comfortable.
2. Do not block on loading the whole lamella first. Load the *current* frame,
   render it, then let a background worker fill outwards from the current
   position (`i, i±1, i±2, …`) until the lamella is cached.
3. Evict the whole cache when the lamella changes.

Measured in phase 5 on the real dataset: the first frame of a lamella costs
287 ms, prefetching ±2 frames (both modalities) a further 170 ms, and stepping
to an already-prepared frame is instant. Warm reads are ~20 ms, so the
background worker keeps well ahead of a user reviewing frames by hand.

This gives instant first paint, smooth navigation once the worker has run
ahead, and bounded memory — without the multi-second stall a strict eager
per-lamella load would cause on a cloud-synced folder.

Labels are small and decode fast, but are cached alongside the image so overlay
toggling is instant.

**Plots need their own, smaller cache.** At 5.12 MB and 575 ms each after
reduction to a 1600 px long edge, caching a lamella's worth would cost 128 MB
and 14 s — twice the images for a mode that is only sometimes on. Load them
lazily, prefetch only the immediate neighbours, and bound the cache to a handful
of entries. The 575 ms is PNG decode of a 13.5-megapixel figure, so it cannot be
reduced by targeting a smaller size; it can only be moved off the main thread
and hidden behind prefetch.

Two rules that follow:

* **No filesystem or decode work on the Qt main thread, ever.** A blocked
  OneDrive read would freeze the whole window.
* The cache and its policy live in `gui/cache.py` with **no Qt** — the loader
  function is injected, so tests drive it with a synthetic loader and assert on
  eviction and prefetch order without touching disk or Qt.

An on-disk thumbnail cache (keyed by path + mtime + size, in a platform cache
dir) is a possible later optimisation. Not in v1 — measure first, and never write
cache files into the user's data folders.

### 6.3 Overlay rendering — `display.py`

A pure function over numpy arrays, no Qt, no filesystem:

```python
def compose_overlay(image_8bit, label, palette, *, alpha=0.4,
                    hide_background=True) -> np.ndarray:  # RGB uint8
```

* Class 0 (`background`) is never drawn, per the requirement.
* Colours and class names come from `segmentation_config_AM.yaml` (§7).
* If the label's shape differs from the image's, resize the *label* with
  nearest-neighbour (never interpolate class indices) and surface a warning in
  the UI rather than crashing — predictions are known to differ in size.
* Composition is **~17 ms** for a 683×1024 frame with 28% foreground coverage,
  fast enough to compose on demand. That is why the cache stores the image and
  label separately rather than a finished overlay: opacity and the overlay
  toggle then cost nothing extra, and the cache stays at 1.4 MB per image
  instead of 3.5 MB.

Converting the resulting RGB array to a `QImage` is a two-line adapter in the
widget layer.

The middle panel shows **image | image+overlay side by side**, as requested, with
a synchronised zoom/pan (`QGraphicsView` with a shared transform). Zoom/pan is
not in your list but is close to free here and is the difference between
"I can see there's a problem" and "I can see what the problem is."

---

## 7. Segmentation config — `config.py` and `colors.py`

`segmentation_config_AM.yaml` currently sits untracked at the repo root. The GUI
needs it, so it needs a defined home and a loader.

```python
@dataclass(frozen=True)
class SegmentationConfig:
    class_colors: dict[int, str]
    class_labels: dict[int, str]

    def rgb(self, class_index: int) -> tuple[int, int, int]: ...
    def name(self, class_index: int) -> str: ...
```

Resolution order, first hit wins: explicit `--config` path → `AM_SEG_CONFIG` env
var → `segmentation_config_AM.yaml` in the scan root → the project root → a copy
bundled with the package as the default. Loaded with the existing
`io.yaml.read_yaml`.

Colour names in the config (`orange`, `salmon`, `orchid`, `turquoise`…) are all
CSS/X11 names. Rather than depend on matplotlib for `to_rgb`, `colors.py` carries
the static CSS4 name→RGB table plus `#rrggbb` parsing. It is ~150 lines of data,
zero dependencies, and trivially testable. An unknown name raises a clear error
naming the offending class rather than silently rendering black.

The class names are shown as a **legend** beside the image, so you can read the
overlay without memorising the palette.

---

## 8. Layout and interaction

```
┌────────────────┬─────────────────────────────────────────┬──────────────────┐
│ Scan root      │  ○ SEM   ○ FIB   ○ Plots                │ ☐ Enable editing │
│ [ /path… ][…] │                                          │                  │
│ [ Rescan ]     │  ┌──────────────┐ ┌──────────────┐      │ Apply to:        │
│                │  │              │ │              │      │  ○ SEM           │
│ Projects       │  │    image     │ │ image+label  │      │  ○ FIB           │
│ 001 Auto…13-57 │  │              │ │              │      │  ● Both          │
│ 002 Auto…10-31 │  └──────────────┘ └──────────────┘      │                  │
│                │                                          │ State            │
│ Lamellae       │  Legend: ■ GIS ■ lamella ■ crack ■ vac   │  ● selected      │
│ 01 alive-muskox│                                          │  ○ deselected    │
│ 02 loving-mole │  Frame  ├──────●───────────────┤ 27/84   │  ○ corrected     │
│                │         ▇▇▇░░▇▇▇▇▓▓▓▇▇▇  ← state strip  │                  │
│ 84 frames      │  Lamella├──●─────────────────┤  2/12     │ SEM: selected    │
│ 40 sel/30 corr │                                          │ FIB: deselected  │
│                │  stage 02 · img_013 · SEM+FIB · label ✓  │                  │
└────────────────┴─────────────────────────────────────────┴──────────────────┘
```

Points worth noting:

* **State is a radio group, not three checkboxes.** On disk the three states are
  mutually exclusive folders (`STATE_DIRECTORIES`), so a record is always in
  exactly one. Three checkboxes would suggest combinations that cannot exist.
  The group also *displays* the current state, so it doubles as the status
  readout.
* **The edit guard** (`☐ Enable editing`) disables the entire right panel until
  ticked, as requested. It resets to off when the project changes.
* **The state strip** under the frame slider colour-codes every frame in the
  lamella (selected / deselected / corrected / no label). This is not in your
  list, but it turns "where have I got to?" from a scrubbing exercise into a
  glance. Cheap to draw from data already in memory.
* **Keyboard shortcuts** matter more than they sound for curation throughput:
  `←/→` frame, `↑/↓` lamella, `1/2/3` display mode, `S/D/C` set state,
  `Space` toggle overlay. Curating hundreds of frames by mouse is painful.
* When "Apply to: Both" is chosen and the two modalities are currently in
  *different* states, the readout shows both (`SEM: selected, FIB: deselected`)
  and the radio group shows no selection until you pick one.

### Safety rules in the GUI

* Any transition **out of** `corrected` opens a modal confirmation naming the
  files, and only then passes `allow_move_from_corrected=True`. This mirrors the
  CLI's `--confirm-corrected` exactly; the GUI never sets it implicitly.
* `transition_records` validates the whole batch before moving anything, so a
  "both modalities" change either fully succeeds or changes nothing.
* `FileExistsError` / `PermissionError` / `FileNotFoundError` become a clear
  dialog, not a traceback in a terminal the user may not be watching.
* Every move is logged through the existing `logging` calls; the GUI adds a
  small in-window log pane so there is a visible record of what was changed.

---

## 9. Handling missing or awkward data

The external segmentation software auto-creates an *empty* label for every image
it sees, so "a label file exists" does not mean "this image is labelled". The
display must distinguish these cases:

| Situation | Display |
| --- | --- |
| Image + non-empty label | Both panes; overlay active. |
| Image + label that is all zeros | Image in both panes; badge **"empty label"**. This is the auto-generated placeholder, not real work. |
| Image, no label file | Image in both panes; badge **"no label"**; overlay toggle disabled. |
| Label shape ≠ image shape | Overlay shown with nearest-neighbour resize; badge **"label resized"**. |
| Modality absent for this frame | Placeholder panel: "no FIB image for this frame"; mode toggle marks it unavailable. |
| No plot for this frame | Placeholder: "no plot for this frame" (normal — plots are sparser than images). |
| Image file unreadable / cloud-stub timeout | Error placeholder with the path and the exception; navigation keeps working. |

The empty-label check is `label.max() == 0`, computed once when the label is
loaded into the cache — negligible cost, and it is arguably the single most
useful signal in the whole tool for deciding what still needs proofreading.

---

## 10. Phased implementation

Each phase is independently reviewable, leaves the package working, and ships
with tests. Nothing in phases 1–4 requires Qt, so most of the logic is tested
headless.

| # | Phase | Deliverable | Tests |
| --- | --- | --- | --- |
| 1 | ~~**Config + colours**~~ **DONE** | `config.py`, `colors.py`; default config bundled at `data/segmentation_config_AM.yaml` | resolution order; CSS + hex parsing; unknown-colour error; missing-class fallback |
| 2 | ~~**Frames + scanner**~~ **DONE** | Fixed the AM/PM sort bug; `stage.plots`; `frames.py` | AM/PM ordering; SEM-only / FIB-only frames; `_FIB.tif` vs `_SEM.tif` naming; plot matching incl. missing plots; unparseable image numbers |
| 3 | ~~**Selection API**~~ **DONE** | Record paths updated after a move; `current_state()` | paths correct after each transition; round-trip select→deselect→select; corrected still guarded |
| 4 | ~~**Display pipeline**~~ **DONE** | `io/image.py` (shared reading); `display.py`: downscale, bit-depth conversion, `compose_overlay` | synthetic arrays: class 0 hidden, alpha blend, shape mismatch resize, empty-label detection, 8-bit passthrough, 16-bit → 8-bit |
| 5 | ~~**Controller + cache**~~ **DONE** | `gui/controller.py`, `gui/cache.py` (no Qt; injected loaders) | navigation clamping; lamella switch evicts; prefetch order; "both" target resolution; edit guard blocks transitions; **layering guard** (brought forward from phase 6) |
| 6 | ~~**Qt shell**~~ **DONE** | `app.py`, `loader.py`, `widgets/*`; `am-seg gui` | offscreen window tests driving real widgets against a synthetic project (skipped without Qt) |
| 7 | ~~**Revised after testing**~~ **DONE** | See [§13](#13-phase-7-revised-after-testing) — rewritten from real usage findings, then implemented | zoom linkage, timeline rows, list filtering, state ordering, shortcut coverage |

Phase 1 alone is worth doing regardless of the GUI: the config file is currently
untracked and unread by any code.

`pytest-qt` becomes a `dev` extra. CI must keep running the full suite with no Qt
installed — that is the check that keeps the layering honest.

---

## 11. Open questions

Answers change the work, so worth settling before phase 2:

1. **Which stages belong in a lamella's stack?** The scanner treats *any* folder
   containing `sem/` or `fib/` as a stage, so a lamella may include stages that
   are not `adaptive_polish_*`. Include everything the scanner finds (my
   default), or filter to `adaptive_polish*` with a toggle to show the rest?
2. **Should the GUI offer a state filter** (e.g. "show only frames not yet
   corrected")? `filtering.iter_matching_records` already supports it and it
   would make long lamellae much faster to work through — but it adds a control
   and makes the frame slider's index non-obvious.
3. **Downscale target of 1024 px long edge** — enough to judge a segmentation
   boundary, or do you want 1536 (≈2× the memory) for reviewing thin lamella
   edges?
4. **Does `AP_metadata.json` (3.7 MB per stage) hold per-image information**
   worth showing in the frame status line — milling current, depth, dose? If so
   it is a natural phase-8 addition, but it needs a parser and belongs in
   `metadata.py`.

---

## 12. Risks

* **Qt binding conflicts.** Installing PySide6 into an environment that already
  has PyQt5 (via napari) can crash at import. Mitigated by `qtpy` + documenting
  that you should not add a second binding.
* **OneDrive cloud-on-demand files.** A first read can block for seconds.
  Mitigated by never loading on the main thread, and by surfacing a per-frame
  loading state instead of freezing.
* **Stale records after external changes.** If files move on disk while the GUI
  is open (e.g. a CLI run in another terminal), the in-memory index goes stale.
  Mitigated by an explicit Rescan button and by re-validating a record's
  existence immediately before any transition — `existing_variant()` already
  does exactly this.
* **Preparation racing a lamella change.** A frame that finishes preparing after
  the user has moved to another lamella must not be filed against the new one.
  Handled in phase 6 by a generation counter on the cache: a load that began
  before `set_frames` is discarded rather than stored. The worker also carries a
  token, so a batch queued for a position the user has left stops at its next
  item instead of making the visible frame wait behind stale work.
* **Scope creep into label editing.** The moment the viewer is good, editing
  labels in it will look tempting. Resist: it duplicates the external tool and
  puts corrected data — the most valuable data in the project — at risk.

---

## 13. Phase 7, revised after testing

Phase 7 was originally a guess at what polish would be needed. This section
replaces it with what a real session on real data actually showed.

**Confirmed working, no action:** display scaling on both 4K and 1080p; SEM and
FIB navigation is fast and responsive.

Work is grouped by kind and ordered so that the two genuine defects land first.

### 7a. Defects

**A1 — Zoom sync stops working when an image fits the pane.**
Zooming out in one pane stops moving the other once the image is fully visible;
zooming back in eventually re-couples them.

*Cause, confirmed in the code:* `LinkedImageViews._link` mirrors the panes by
listening to **scrollbar** `valueChanged`, and copies the transform as a side
effect of that. When an image fits its pane there are no scrollbars to move, so
nothing fires and the transform never propagates. The zoom was never actually
synchronised — it was riding on scroll events that happen to accompany it.

*Fix:* give `ImageView` a `zoomed` signal emitted from `wheelEvent`, and have
`LinkedImageViews` copy the transform on that signal directly, independently of
scrolling. Then also clamp zoom-out at fit-to-pane, as suggested: below that the
image only shrinks into a corner, which is never useful.

**A2 — The state strip conflates "mixed" with the individual states.**
Reported as "deselected and corrected are both orange".

*Cause, confirmed against the data:* they are not. `deselected` is grey and
`corrected` is blue; the orange is `mixed`. In `01-ruling-ibex` one frame has
SEM deselected and another has SEM corrected, with **both FIBs still selected**,
so `combined_state` collapses both frames to `mixed` and paints them the same
colour. The strip cannot show what is actually going on, which is precisely the
gap behind the per-modality question below.

*Fix:* covered by C1 — showing each modality on its own row removes `mixed`
from the strip entirely.

### 7b. Plot responsiveness

**B1 — Plot preparation is 2.6x slower than it needs to be.**
Measured breakdown of the 470 ms per plot:

| Stage | Cost |
| --- | --- |
| PNG decode (`IMREAD_UNCHANGED`, 3000x4500 RGBA) | 170 ms |
| Compositing the alpha channel over white | **275 ms** |
| Downscale to 1600 px | 21 ms |

The alpha compositing dominates, and it is entirely wasted work: these plots are
opaque, so the channel carries no information. Decoding with
`cv2.IMREAD_REDUCED_COLOR_2` returns 3-channel data at 1500x2250 directly,
skipping both the composite and most of the downscale:

**470 ms -> 183 ms, at identical output resolution (1067x1600) and identical
memory (5.1 MB).**

Reduced-decode flags do *not* speed up the decode itself (libpng expands the
full image regardless: 189 ms vs 172 ms) — the whole gain is in avoiding the
RGBA path. Keep `to_rgb` for the general case; just stop routing these plots
through it.

**B2 — Prefetch plots as aggressively as images.**
Plots currently use `radius=3` and a cache of 8. With B1 making each plot a
third of the cost, raise the plot cache to ~16 entries (~82 MB) and prefetch
them at the same radius as images while the plot view is on.

**B3 — Only if B1+B2 prove insufficient:** decode at `IMREAD_REDUCED_COLOR_4`
(750x1125, 2.5 MB) for a first paint and refine to full size when idle. This
adds real complexity for a second-order gain, so it stays out unless measured to
be needed.

### 7c. The frame timeline

**C1 — Merge the slider and the state strip, and show both modalities.**
The strip is wider than the slider because the slider row carries "Frame" and a
"5 / 26" label, so the two never line up. Rather than padding them into
alignment, merge them: one `FrameTimeline` widget that *is* the slider.

Since arrow keys already handle stepping, the QSlider handle adds little. The
timeline paints:

* **two stacked rows** — SEM on top, FIB below — each showing that modality's own
  state. This answers the per-modality question and removes `mixed` (A2). Rows
  are preferable to a mode-dependent single row, because a state change can
  target one modality while you are looking at the other.
* **tick marks** between frames, drawn only when there is room (skip below ~4 px
  per frame, otherwise they become noise).
* a **position cursor**, with click and drag to seek.

Alignment stops being a problem because there is only one widget. The "5 / 26"
counter moves inline to its right, or into the status line.

**C2 — Remove the lamella slider.** Agreed: the Lamellae list plus Up/Down
covers it. Keep the lamella position visible in the status line so the count is
not lost.

**C3 — Reclaim margins.** Two specific sources:
`LinkedImageViews` nests a `QWidget`+`QVBoxLayout` inside another
`QVBoxLayout` for no reason, each contributing default margins; and the
`QSplitter` panels use default spacing. Flatten that nesting, set explicit
`setContentsMargins(0,0,0,0)` and a small `setSpacing` between the panes, and
narrow the splitter handles.

### 7d. Colour scheme — a decision for you

With per-modality rows (C1), `mixed` disappears and the palette needs only three
states plus "not present". Two candidates:

| | Option 1 — semantic (your proposal) | Option 2 — progress ramp (recommended) |
| --- | --- | --- |
| deselected | red | grey |
| selected | yellow / orange | blue |
| corrected | green | deep teal |

Your own concern about Option 1 is well founded, and stronger than it looks: the
class palette is **orange (GIS), green (lamella), red (crack/hole), yellow
(vacuum)** — Option 1 collides with all four, and the legend sits directly above
the timeline. Option 1 also puts red and green in opposition, the pairing that
red-green colour deficiency (~8% of men) cannot separate; for a tool you intend
to open-source that is worth avoiding.

Option 2 uses a hue family the labels never use, reads as progress (inert ->
active -> done), and stays distinguishable under the common colour
deficiencies.

Either way, **D1 below makes the mapping self-documenting**, which removes most
of the need for the colours to be guessable on their own.

### 7e. Controls and discoverability

**D1 — State selector: swatches and order.** Put the workflow colour beside each
option, and reorder to `deselected / selected / corrected` so the control reads
in workflow order and matches the timeline's row ordering.

**D2 — Tooltips throughout, including hotkeys.** Every control gets a tooltip
naming both what it does and its key: view radios (`1` / `2` / `3`), overlay
(`Space`), state buttons (`S` / `D` / `C`), fit (`F`), the lists, the opacity
slider, the timeline, and the editing guard.

**D3 — A keyboard shortcut reference**, on `F1` and from a Help menu. Tooltips
only help when you already know where to hover.

### 7f. Scale

**E1 — Long project and lamella lists.** Currently plain `QListWidget`s: they
scroll, but finding one project among many means hunting. Add a filter box above
each list, make the left panel a vertical `QSplitter` so the two lists can be
resized against each other, and show counts in the group titles
("Projects (37)"). Worth doing before the tool meets an archive rather than a
two-project test set.

### 7g. Carried over from the original phase 7

* Remember the scan root between sessions (`QSettings`, in the user config
  directory — never inside the data folders).
* README section covering installation, `am-seg gui`, and the shortcuts.

### Suggested order

1. **7a** — defects, smallest and highest value.
2. **7b** (B1, B2) — one-line decode change for the biggest single speedup.
3. **7c** + **7d** + **7e** (D1) — the timeline rewrite and the controls that
   explain it; these interlock and are best done together.
4. **7e** (D2, D3), **7f**, **7g**.

Each remains independently shippable, and 7a/7b are worth doing on their own
even if the timeline rewrite is deferred.


### Outcome

All of §13 is implemented. Measured results:

* Plot preparation **470 ms -> 198 ms**. Plot navigation now keeps up completely
  when stepping at 250 ms per frame (8/8 ready on arrival, against 5/8 at
  120 ms). At ~198 ms per plot on one loader thread, stepping faster than that
  cannot keep up by construction, so this is now bounded by decode rather than
  by avoidable work.
* The zoom linkage defect is fixed and covered by a regression test that
  reproduces the original condition (zooming out past the point where the image
  fits its pane).
* The timeline shows SEM and FIB on separate rows; `mixed` no longer appears in
  it. The lamella summary is likewise reported per modality, since a combined
  count would have had the same defect.

Colour scheme: **Option 2** (grey -> blue -> teal) is in place, as recommended.
It lives in `STATE_COLOURS` in `gui/widgets/frame_strip.py`; changing those three
values also updates the swatches in the state panel, which read from the same
mapping.

Not carried out, and deliberately: the black bands above and below the images
are the panes letterboxing landscape acquisitions, not margin. Stacking the two
panes vertically would use the space better, but that changes the side-by-side
arrangement the tool was specified around, so it is left as an open question
rather than assumed.

---

## 14. Second round of testing

### Defect: zooming stopped working in the image panes

Zoom was refused entirely in the SEM and FIB panes, while the plot view behaved
correctly.

*Cause:* the fit clamp added in §7a was applied to **both** directions. Zooming
in computes `target = current * 1.25`; when `current` sits below `fit_scale` by
more than one zoom step, that target is still below the minimum, and the guard
returned without doing anything — in either direction.

`current` is *always* slightly below `fit_scale` after a fit, because
`fitInView` leaves room for the frame. On the development machine the gap was
1%, far under the 20% needed to trigger it; on another display it exceeded a
zoom step and zoom died completely. The plot view escaped because its fit gap
was smaller relative to its scale.

*Fix:* the fit limit now constrains zooming **out** only. Zooming in is bounded
by `MAX_SCALE` alone. Tested from 20%, 50%, 75%, 85% and 99% of the fit scale.

The lesson worth keeping: a limit that exists to stop one direction must be
applied to that direction, not to the magnitude of the result.

### The edit target follows the view

Changing the state of the modality you are *not* looking at is far more likely
to be a slip than an intention. Switching to the SEM or FIB view now points the
edit target at that modality, and the target starts on SEM to match the opening
view. "Both" remains available by choosing it explicitly, and the plot view
leaves the target alone since it shows neither modality in particular.

`CurationController.follow_mode` turns this off for anyone who prefers the old
behaviour.

### Defect: the corrected-data warning named the wrong files

With a corrected SEM and a deselected FIB, confirming a change listed *both*
files as holding proofread work. `_confirm_corrected` listed everything
targeted rather than what was actually corrected. It now lists only records in
the corrected state, and caps the list at 12 names with a count for the rest —
which matters now that a change can cover a whole project.

### Bulk state changes

A new "Across" control applies a state change to the current frame (default),
the whole lamella, or the whole project, combined with the existing SEM / FIB /
Both target.

Safety comes from three existing properties rather than anything new:
`transition_records` validates the entire batch before moving a single file, so
a bulk change either fully succeeds or changes nothing; corrected data still
requires its own confirmation; and the editing guard still applies. On top of
those, any scope wider than one frame is confirmed with a count and a
breakdown by modality before proceeding.

Scopes read records straight from the scanned stages rather than building
frames, since the frame grouping only matters for display.

---

## 15. Plot navigation and plot sharpness

Compared against a separate single-purpose tkinter viewer
(`20260813_Adaptive_Polishing_viewer/adaptive_polishing_viewer.py`) that shows
the same plots without either problem.

### Stutter when stepping quickly

*Cause:* plots were prepared with the same narrow prefetch used for
acquisitions -- the current frame plus three either side. A plot takes ~200 ms
to decode, and a held-down arrow key repeats far faster than that, so the user
simply outran the seven-frame window and kept arriving at frames that had not
been prepared.

The tkinter viewer does not have this problem because it does something
simpler: on selecting a lamella it decodes **every** plot in that lamella into
memory, once, in a background thread. After that, navigation is a dictionary
lookup.

*Fix:* the same approach, ordered better. In the plot view the whole lamella is
now prepared, working outwards from the current frame, and the plot cache is
large enough to hold it (64 entries). Plots are still not prepared at all
unless the plot view is on, so this costs nothing the rest of the time.

No amount of prefetching *radius* could have fixed this. At 200 ms a plot, the
only way to be ready for a key held down at 30 ms intervals is to have the data
already in memory.

Measured after the change: all 25 plots of a lamella prepared in the
background, then 24 rapid steps with no waiting at all, in 79 ms total.

### Blurred text and plots

*Cause:* two lossy resamples, where the tkinter viewer does one.

1. Plots were prepared at a fixed 1600 px long edge, then the graphics view
   scaled that again to fit the pane -- about 0.67x, with the view's own
   bilinear filter. Bilinear downscaling samples 2x2 pixels and is a poor
   choice for small text.
2. Even the first step used plain area averaging, which is correct but soft.
   The tkinter viewer uses PIL's LANCZOS, which is properly antialiased *and*
   sharp.

*Fix, both parts:*

* Plots are prepared at the size they are actually displayed at. The cache is
  told the pane's dimensions, and re-prepares if they change by more than 20%
  -- enough to ignore a window being dragged. The view now scales by 0.996
  instead of 0.667, so the second resample is gone.
* `downscale_for_detail` decimates most of the way with area averaging (which
  is what prevents aliasing) and takes the final factor of two with a sharp
  filter, approximating what PIL's LANCZOS does in one step. It costs ~19 ms
  more per plot.

Acquisitions deliberately keep plain area averaging. A sharp filter can ring
around hard edges, and adding halos to scientific image data to make it look
crisper is not a trade worth making. The change is confined to rendered
figures, where the detail being preserved is text.

A side benefit: at pane size a prepared plot is 2.3 MB rather than 5.1 MB,
which is what makes holding a whole lamella of them reasonable.

---

## 16. Removing the last resampling of the plots

Plots were still softer than the same PNGs in the tkinter viewer after §15.
Comparing the two pipelines end to end found two remaining resamples, neither
of which was needed.

### The display is not one pixel per pixel

**This was the dominant cause.** Qt reports widget sizes in *logical* pixels.
On a high-resolution display one logical pixel is two physical ones, so the
plot pane measured 1068 logical pixels and occupied 2136 real ones. §15
prepared plots to the logical size, and the display then **magnified them 2x**
to fill the pane.

That is a 2x upscale of an already-downsampled image, and no choice of
downsampling filter can recover from it. It also explains why the problem
looked worse on the 4K monitor than on the 1080p one.

*Fix:* the pane size is now reported in physical pixels
(`viewport * devicePixelRatioF()`), plots are prepared at that size, and the
pixmap is tagged with the same device pixel ratio so Qt maps it one array pixel
to one physical pixel.

Measured: on a 2x display the plot is now prepared at 2136x1424 rather than
1068x712 -- four times the pixels, all of them real.

### A 0.996 scale is still a full resample

The view was fitting the plot to the pane, which produced a scale of 0.9963
rather than 1.0: `fitInView` leaves a small margin for the frame. Any transform
that is not the identity makes Qt filter every pixel, so being *almost* right
cost as much as being wrong.

*Fix:* plots are prepared to fit the pane in both dimensions
(:func:`fit_within`, rather than a single long edge, which could still overflow
a differently shaped pane), and the view is then set to the identity transform
instead of fitting. Qt blits rather than resamples. Zoom still works, and is
preserved between frames as before.

The net effect is that a plot is now resampled exactly **once**, from the
original PNG straight to the number of physical pixels it will occupy -- which
is what the tkinter viewer does in its single `thumbnail()` call.

### Memory

Preparing at physical resolution costs four times as much on a 2x display:
9.1 MB a plot rather than 2.3 MB. The plot cache limit is therefore derived
from a memory budget instead of being a fixed count, so a normal lamella stays
entirely cached either way -- 64 entries at 1x, 31 at 2x, ~230 MB in both
cases.

### Remaining

The sharp final filter from §15 is still in use, but it now has much less to
do: at native resolution the reduction from a 4500 px source to a 2136 px pane
is a factor of two rather than four. If the two-stage filter is ever
unwanted, plots would still be markedly better than before this section
purely from the two fixes above.

Acquisitions are still prepared to a fixed 1024 px and fitted to their panes,
so they are resampled twice as plots used to be. That is far less noticeable on
photographic data than on rendered text, and it is what allows the two panes to
share a zoom. Worth revisiting only if image sharpness is ever raised as a
problem in its own right.

---

## 17. Imaging-condition variants and shared labels

Status: **built.** This is phase 5 of
[the retrograde support plan](retrograde_support_plan.md). Sections 1–16
describe a GUI where a frame has at most one SEM and one FIB image; that is no
longer true.

Two things changed underneath the GUI:

* A frame's `sem` and `fib` are now **tuples**, because one acquisition can be
  captured under several imaging conditions — five SEM images of one scene at
  different frame-integration settings. `frame.record(modality)` returns the
  base image and has the old meaning; `base_records` is the pair a
  single-image view shows.
* A state change **follows the label**, not the image. Images sharing a label
  move together, because the label is the unit of proofreading.

### 17.1 Variant is not a fourth edit axis

The obvious design — a "Variant" selector beside "Apply to" and "Across" —
is wrong, and the reason is worth recording because it is what keeps the panel
from growing a fourth control.

State changes follow the label. So:

* variants that **share** a label: choosing one to edit is meaningless, since
  all of them move regardless;
* variants with **private** labels: each is its own group, and the one on
  screen is the one meant.

Either way the edit target is implied by what is being viewed. There is no
state decision left for a variant selector to make, and offering one would
present a control that silently does something wider than it claims.

So the variant is a **view** control, and it lives with the other view controls
in the image panel. The state panel gains no selector — it gains an honest
readout of how far a change reaches.

This is the same rule as `follow_mode` (§8): the modality on screen is the one
a state change is meant for, extended one axis.

### 17.2 Image panel: the variant control

```
( ) SEM   (•) FIB   ( ) Plot      Variant [ base ▾ ]      [x] Overlay   Opacity [====----]
```

A dropdown rather than segmented buttons: it stays narrow with five conditions,
and variant names run long (`8_fi_16`). It is **hidden entirely** when the
current modality has one variant, which is every Autolamella project in the
reference collection — the common case gains nothing to look at.

The overlay needs no special handling: each record carries its own label, so
variants sharing a label show the same overlay and variants with private labels
show their own, automatically.

### 17.3 State panel: a reach line, not a selector

```
┌────────────────────────────────┐
│ [x] Enable editing             │
│ ┌ Apply to ──────────────────┐ │
│ │ (•) SEM   ( ) FIB  ( ) Both│ │
│ └────────────────────────────┘ │
│ ┌ Across ────────────────────┐ │
│ │ (•) This frame             │ │
│ │ ( ) This lamella           │ │
│ │ ( ) This project           │ │
│ └────────────────────────────┘ │
│                                │
│  Moves 4 images + 1 label      │  <- new
│  3 share a label with this one │  <- new, only when shared
│                                │
│ ┌ State ─────────────────────┐ │
│ │ ( ) # deselected           │ │
│ │ (•) # selected             │ │
│ │ ( ) # corrected            │ │
│ └────────────────────────────┘ │
│ SEM: selected                  │
│ FIB: selected                  │
│ ┌ Changes ───────────────────┐ │
└────────────────────────────────┘
```

The reach line sits **above** the State buttons, because it describes what is
about to happen; below them it would read as a report of what already did.

It is **always shown**, even when it says "Moves 1 image + 1 label". A line that
appears only sometimes is a line nobody has learned to read, and its absence
would have to be interpreted rather than simply seen. It also surfaces a missing
label without a separate indicator.

Counts come from `plan_transition`, so the number shown is the number of files
that will actually move — the same validated plan the operation performs, not a
second estimate that could drift from it.

### 17.4 Confirmation: confirm what is invisible

Two dialogs already exist (bulk scope, corrected data). A third would teach the
habit of dismissing dialogs unread, so shared labels get none:

| situation | how it is confirmed |
|---|---|
| frame scope, shared label | the reach line — on screen before the click |
| lamella / project scope | the existing bulk dialog |
| corrected data | the existing dialog, unchanged |

One fix is needed rather than one addition: `_confirm_bulk` counts
`scoped_records()`, which does not include images pulled in by a shared label.
It must count the expanded set from `plan_transition`, or it under-reports how
much it is about to move.

### 17.5 Timeline: report mismatches, do not fix them

A frame whose images disagree with their label gets a marker and a tooltip
naming the mismatch. The state drawn stays the base variant's, which equals the
group state whenever things are consistent — so the marker means precisely
"this frame is not telling you the truth".

Repair stays **CLI-only** (`am-seg repair-state`, with `--dry-run`). Moving
files from a tooltip is a long way from a previewed batch operation, and a
mismatch is rare enough that reaching for the command line is proportionate.

### 17.6 Rejected alternatives

**A "Variant" radio group in the state panel.** Implies per-variant editing
that the group semantics forbid, and is permanently visible while being empty
for every current project.

**Replacing "Apply to" with a checkbox list of the frame's images.** Honest
about what exists, but invites unchecking one member of a shared group — the
one operation that must not be possible — and grows unbounded with variants.

### 17.7 What this touches

* `image_panel.py`: variant dropdown, shown only when there is a choice
* `controller.py`: viewed variant as display state; `shared_label_records`
  (already written) feeds the reach line
* `cache.py`: prepared images keyed by variant as well as modality
* `state_panel.py`: the reach line
* `main_window.py`: `_confirm_bulk` counting the expanded set
* `frame_strip.py`: the mismatch marker

### 17.8 Built

All of the above, with three notes from the build:

* **`_confirm_bulk` was fixed first, separately.** It counted the scope rather
  than the expanded set, so at lamella or project scope on data with shared
  labels it promised to move fewer files than it would. It now counts
  `controller.expanded_records()` and reports how many were pulled in.
* **Cache entries are keyed by `(frame, modality, variant)`.** Switching
  between two conditions to compare them therefore evicts neither, which is
  the thing this data invites. A frame lacking the viewed condition falls back
  to its base image rather than blanking the pane.
* **The variant dropdown is populated with descriptions from
  `AM_metadata.yaml`** where the project supplies them (`8_fi_1` shows as
  "frame integration 1"), and with the raw suffix where it does not.

---

## 18. Empty labels, and label progress on the timeline

The external segmentation software writes a label for every image in its input
folder, so **a label file existing is not evidence that anything was
annotated**. This surfaced from a real report: labels appeared for SEM but not
FIB, and the FIB label turned out to be 2048x3072 pixels of class 0. The
overlay was correct -- background is never drawn -- but nothing else
distinguished that file from finished work.

`display.is_empty_label()` already existed and the image panel already badged
`"empty label - not yet annotated"`. The gap was that only the *display* layer
knew: the survey counted the file as a label, and a training export would have
paired the image with a blank mask, which actively teaches a model that the
structure is background.

### 18.1 Where the check now runs

`operations.label_io.is_label_file_empty()` returns True, False, or **None when
it cannot tell** -- no image stack, unreadable file. None is deliberately
distinct from True: reporting "not annotated" for a label nobody could read
would be a guess presented as a fact.

* `inspect-layout` reports an `empty-labels` warning and counts annotated
  labels separately. On by default, since a survey that hides this is the
  reason the problem went unnoticed; `--no-label-check` skips it. Measured cost
  is ~2.5 ms a label.
* `create-training-csv` excludes them by default and says so, with
  `--include-empty-labels` for deliberately building negative examples.
* The curation window badges them, as it already did.

### 18.2 Label lines on the timeline

The timeline now answers two questions at once, so they are drawn as two
different kinds of mark:

```
 ---------------  SEM labels     (thin line, above)
 ===============  SEM state      (thick row)
 ===============  FIB state      (thick row)
 ---------------  FIB labels     (thin line, below)
```

Each modality's label line sits on the **outside** of its state row, so the two
state rows stay adjacent in the middle where they are easiest to compare, and
neither label line can be misread as the other modality's state.

The line is 3 px against the rows' 11 px: it should read as a rule, not as a
third and fourth state bar.

Violet (`#a78bfa`) marks an annotation, because the workflow states use
grey/blue/teal and the class legend directly above uses orange, green, red and
yellow; a separate hue says "this line is about labels, not workflow state".

An empty placeholder is **grey (`#8a8a8a`), not a dim violet**. The first
attempt used a dark violet on the reasoning that brightness could carry the
meaning within one hue, and it failed on sight: against a near black background
a dark colour is nearly indistinguishable from nothing drawn at all, which
collapses "placeholder" into "no label" -- precisely the distinction this line
exists to make. Grey is unmistakably lighter than the background and
unmistakably not the annotation colour. It is lighter than the deselected state
grey too, and sits on the thin line rather than in a state row, so the two do
not read as the same mark.

A test asserts the lightness ordering (background < empty < annotated) so a
future palette change cannot quietly reintroduce the problem.

### 18.3 The mismatch marker moved

§17.5's mismatch stripe was drawn along the bottom edge, which the FIB label
line now occupies. Rather than squeezing it elsewhere, it became a translucent
amber wash over the frame's whole column. That reads better anyway: the stripes
are each "a fact about this row", whereas a mismatch means "do not trust this
frame".

### 18.4 Cost

The timeline asks for a label state per frame per modality, so results are
cached by path on the controller. Navigating back and forth across a lamella
re-reads nothing, and a typical 10-20 frame lamella costs 50-100 ms once.
