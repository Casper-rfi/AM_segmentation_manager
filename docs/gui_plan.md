# Design plan: data curation GUI

Status: **proposal — not yet implemented.**
Target: a desktop viewer for reviewing Autolamella adaptive-milling images against
their labels, and setting each image's workflow state (selected / deselected /
corrected) without leaving the viewer.

This document explains *what* will be built, *why* each choice was made, and the
order in which it will be built. Nothing here is implemented yet; each phase in
[Phased implementation](#8-phased-implementation) leaves the package working.

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
| 7 | **Polish** | Keyboard shortcuts, state strip, legend, log pane, remembered scan root, README section | shortcut→controller mapping (headless) |

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
