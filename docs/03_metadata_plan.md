# Metadata: reading, editing, exporting and summarising

Status: **proposed, not started.** Written 2026-08-14.

The main body of new work. Four capabilities, built in order, each usable on its
own:

1. **§3 Read** per-image metadata from image headers.
2. **§4 Edit** per-project extended metadata (API → CLI → GUI).
3. **§5–6 Aggregate**: statistics, metadata-filtered CSV export, metadata CSV export.
4. **§7 Present**: GUI tabs matching the workflow.
5. **§8 Label quantification** — fenced, deliberately last.

Sections 3 and 4 are independent of each other. Everything after depends on 3.

---

## 1. Findings from the reference collection

Before designing anything I measured the actual data at
`<collection>/` (40 GB, 8503 TIFFs,
of which ~2680 are images in `sem/`/`fib/` directories). Three findings change
the design, so they come first.

### 1.1 There are exactly two header dialects, not a continuum

A census of 400 random TIFFs:

| Count | Dialect | Where |
|---|---|---|
| 354 | **A** — JSON in TIFF tag 270, keys `experiment, image, microscope_state, pixel_size, shape, system, user, version` | All Autolamella-era data, written by `fibsem` |
| 18 | **B** — FEI/Thermo tag 34682, sections `User, System, Beam, EBeam, Scan, EScan, Image, …` | Pre-Autolamella datasets (`20240228_adaptive_milling_ai/…`) |
| 24 | JSON with only `shape` | Autofocus iteration images, not in `sem/`/`fib/` — out of scope |
| 3 | no metadata | Label files, as expected |

This is much better news than "different generations of SEM images" suggested.
It is two well-defined schemas, and the split is *by era*, not by modality.

### 1.2 SEM and FIB share one schema — your open question is answered

You wrote: *"I think these also all apply to FIB, although the unit for beam
current may need to be changed."*

**They do apply, and the unit does not change.** In dialect A, SEM and FIB
images have byte-identical schemas; the only difference is which sub-object is
populated — `microscope_state.electron_beam` versus `microscope_state.ion_beam`,
and `system.electron` versus `system.ion`. Both record `voltage` in volts and
`beam_current` in amps. A FIB image reads `voltage: 30000.0, beam_current: 2e-11`;
the same `× 1e12` gives 20 pA. The same holds for dialect B
(`EBeam.BeamCurrent` / `IBeam.BeamCurrent`, both amps).

So there is no FIB-specific code path to write. The modality selects which
sub-object to read from, and that is all.

**Bonus:** dialect A records `image.beam_type` as `"ION"` or `"ELECTRON"`. That
is an authoritative modality marker from the instrument itself, independent of
the directory name the package currently infers modality from. Worth reading as
a **cross-check**, not a replacement — the directory is still what organises the
data, but a header disagreeing with its directory is a finding
`inspect-layout` should report.

### 1.3 Integrations exist only in the legacy dialect — and the reference script gets this wrong

`read_sem_tiff_header_csv.py:120–130` defaults `image_integrations` and
`line_integrations` to `1` when the header does not mention them.

For dialect B that is harmless — the fields are always present
(`Scan.Integrate: 16`, `EScan.LineIntegration: 1`). For dialect A they are
**absent entirely**: the Autolamella schema has no integration fields at all. So
the script's default would report "1 integration" for every modern image, which
is a *claim about the acquisition* invented by the reader.

**Carry the extraction logic over; do not carry that default over.** Absent must
map to `None` / empty, never to a plausible-looking number. This is the same
principle already in `metadata.py`, where a `TODO` placeholder resolves to
`None` rather than to a value, and the same principle as "an empty label is not
evidence of annotation". A missing measurement is not a measurement.

Expect, and design for, `image_integrations` being empty for all Autolamella-era
data. If that turns out to matter scientifically, the fix is to record it in
`AM_metadata.yaml` by hand, not to guess it in the reader.

### 1.4 The archive spans three microscopes, and is heavily imbalanced

I ran the extraction over all 2680 images. It succeeded on **2668**:

| Images | Microscope | Serial |
|---|---|---|
| 2386 | Arctis | 9952468 |
| 182 | Helios 5 Hydra CX | 9952064 |
| 100 | Aquilos | 9951718 |

| Images | Beam type | | Images | Voltage |
|---|---|---|---|---|
| 1221 | `ELECTRON` (dialect A) | | 1351 | 2000 V |
| 1221 | `ION` (dialect A) | | 50 | 2500 V |
| 180 | `EBeam` (dialect B) | | 1267 | 30000 V |
| 46 | `IBeam` (dialect B) | | | |

Three things follow, and they justify most of what this document proposes:

* **The per-microscope breakdown is real and worth building.** Three
  instruments, and the archive is 89 % one of them. That is exactly the
  imbalance you want to detect before training, and it is invisible today.
  (This also answers what was open question 4 in an earlier draft.)
* **SEM and FIB are exactly balanced in the Autolamella era** (1221 / 1221) and
  strongly *un*balanced in the legacy data (180 / 46). Worth knowing before
  assuming a modality split is even.
* **12 of 2680 images failed to yield metadata.** A 0.4 % failure rate over
  real data is the concrete reason §3.6 requires that an unreadable header
  return `None` and log, rather than abort. A batch reader that raises would
  fail on this collection today.

### 1.5 Reading headers is cheap — no cache is needed

Measured on the reference collection: **0.5 ms per header**, 200-file sample.

    2680 images × 0.5 ms ≈ 1.4 seconds for the entire collection.

(Confirmed at scale: the full 2668-image extraction in §1.4 took under a minute
including JSON parsing.) This settles what would otherwise have been the largest
design question in this document. I had expected to have to argue about a derived metadata cache and how
to keep it from becoming the "separate database" `CLAUDE.md` forbids.

**No cache. Read on demand.** The measurement makes it unnecessary, and an
unnecessary cache is pure liability: a second source of truth, an invalidation
problem, and a file that can disagree with the data.

Two guards on that conclusion:

* The measurement is on a warm page cache and local disk. Cold, or over a
  network share, it could be 10–50× slower — 15 s to a minute. Still tolerable
  for a command that is explicitly asking for metadata; **not** tolerable inside
  `Scanner.scan()`, which runs on every GUI launch.
* Therefore: **never read headers during scanning.** Scanning stays a pure
  filesystem walk. Metadata is read by the commands that need it, over the
  records they have already selected. §3.4 makes this concrete.

Re-measure if a collection ever reaches ~50k images or moves to network storage.
Until then this is a solved problem, and the correct amount of machinery is
none.

---

## 2. What exists today

| Piece | Where | State |
|---|---|---|
| Project metadata, two-file per-key resolution | `metadata.py` | Done, good |
| `ExperimentMetadata` typed view | `metadata.py` | Done |
| `AM_metadata.yaml` **creation** | `operations/init_project.py` | Done |
| `AM_metadata.yaml` **editing** | — | **Missing** (§4) |
| Per-image metadata | — | **Missing** (§3) |
| Structural filtering | `filtering.py` | Done |
| Metadata filtering | — | **Missing** (§6) |
| Counting / summarising | `survey.py`, `reporting.py` | Partial — structural only |
| Training CSV export | `operations/csv.py` | Done |
| Per-image metadata CSV | — | **Missing** (§6) |
| Per-project metadata list | — | **Missing** (§6) |

---

## 3. Phase 4 — Reading per-image metadata

### 3.1 Design: a declarative field table, not nine extraction functions

`read_sem_tiff_header_csv.py` extracts nine fields with a hand-written block
each: find these candidate keys, convert this unit, fall back to that. It works,
but adding a field means writing another block, and adding a dialect means
editing all nine.

**Make the field map data.** One table declaring, per field, where to look in
each dialect and what unit conversion applies:

```python
@dataclass(frozen=True)
class HeaderField:
    """Where one metadata value lives, in each header dialect."""
    name: str
    autolamella: tuple[str, ...]   # dotted path within the tag-270 JSON
    legacy: tuple[str, ...]        # section/key within the tag-34682 mapping
    scale: float = 1.0             # e.g. 1e9 for metres -> nm
    unit: str | None = None
```

**This is `layout.py`'s discipline applied to header keys**, and the argument is
identical. Directory names live in exactly one module because they were once
spread across 200 sites; vendor header keys should live in exactly one table for
the same reason. Supporting a third dialect — and there will be one, the
instrument software is still changing — should be adding a column to a table,
not editing nine functions.

Keep the recursive key search from the reference script
(`_find_key_value`, matching case- and underscore-insensitively) as the
**fallback** when the declared path misses. The declared path is fast and
unambiguous; the search is what makes an unanticipated variant still produce a
value. Log at debug when the fallback fires — that log line is how you find out
a new dialect has appeared.

### 3.2 The field map

Derived from real headers of both dialects. This is the table to implement.

| Field | Unit | Dialect A (tag 270 JSON) | Dialect B (tag 34682) |
|---|---|---|---|
| `image_width` | px | `shape[1]` / `image.resolution[0]` | `Image.ResolutionX` |
| `image_height` | px | `shape[0]` / `image.resolution[1]` | `Image.ResolutionY` |
| `pixel_size_nm` | nm | `pixel_size.x`, `.y` (m) | `Scan.PixelWidth`, `.PixelHeight` (m) |
| `dwell_time_ns` | ns | `microscope_state.<beam>.dwell_time` (s) | `Scan.Dwelltime` / `EScan.Dwell` (s) |
| `image_integrations` | count | **absent** → `None` | `Scan.Integrate` |
| `line_integrations` | count | **absent** → `None` | `EScan.LineIntegration` |
| `microscope_model` | — | `system.info.model` (`"Arctis"`) | `System.SystemType` (`"Arctis"`) |
| `microscope_serial` | — | `system.info.serial_number` (`"9952468"`) | `System.Dnumber` (`9952468`) |
| `acceleration_voltage_v` | V | `microscope_state.<beam>.voltage` | `Beam.HV` / `EBeam.HV` |
| `beam_current_pa` | pA | `microscope_state.<beam>.beam_current` (A) | `EBeam.BeamCurrent` (A) |
| `beam_type` | — | `image.beam_type` (`ION`/`ELECTRON`) | `Beam.Beam` (`EBeam`/`IBeam`) |
| `acquired_at` | datetime | `microscope_state.timestamp` | `User.Date` + `User.Time` |

`<beam>` is `ion_beam` or `electron_beam`, chosen by modality.

**Note the serial numbers match across dialects** — `9952468` in both. That is
not a coincidence: it is the same physical instrument, imaged four years apart
under two software generations. It means "breakdown per microscope" works across
your whole archive, and it means **serial, not model, is the right key for
"fine-tune for one microscope"** — two Arctis systems share a model and are
different instruments.

### 3.3 Fields worth capturing beyond your list

Your list has nine fields. These four cost nothing extra (same header, same
parse) and directly serve goals you stated:

| Extra field | Why |
|---|---|
| `microscope_serial` | Distinguishes instruments sharing a model — needed for "fine-tune for one microscope" |
| `detector_type` / `detector_mode` (`ETD` / `SecondaryElectrons`) | Changes image appearance markedly; a plausible source of dataset imbalance |
| `plasma_gas` (`Argon`) | Arctis FIB; affects milled-surface appearance |
| `hfw_um` (horizontal field width) | With pixel size, pins down magnification — the obvious thing to check a training set is balanced over |

**Where to stop.** Dialect B alone has ~30 sections including hot-stage
temperature and fluorescence settings. Do not capture everything "since it is
there": every field is one more column to explain, one more thing to keep
working, and one more axis nobody looks at. Thirteen fields is a good working
set. The raw header is always still on disk, so a field can be added later on
demand — but a field once exported becomes something other people depend on.

### 3.4 Module placement and API

New module `src/am_segmentation_manager/image_metadata.py`, sibling to
`metadata.py`. One module, not two: at ~250 lines splitting tag extraction from
interpretation would be over-structuring.

`tifffile` is an optional extra, so guard the import exactly as `io/image.py`
already does with `has_image_stack_support()` / `require_image_stack()` — reuse
those, do not write a second guard.

```python
@dataclass(frozen=True)
class ImageMetadata:
    """Acquisition metadata read from one image header.

    Every field is optional. None means the header did not record it, which
    is never the same as a default value.
    """
    ...

def read_image_metadata(path: Path) -> ImageMetadata | None: ...
def read_records_metadata(records) -> dict[Path, ImageMetadata]: ...
```

**`ImageMetadata` must not become a field on `ImageRecord`.** Putting it there
creates pressure to populate it during scanning, which §1.5 says must not
happen. A separate mapping keyed by path keeps the cost where it is asked for
and keeps `Scanner` a pure filesystem walk.

### 3.5 Retire the reference script

Once `image_metadata.py` reproduces its nine fields on the same inputs:

- [ ] Delete `read_sem_tiff_header_csv.py` from the repository root.

Its CSV-joining behaviour is superseded by §6's export. Keep a test that pins
the extracted values for one dialect-A and one dialect-B file, so the retirement
is verifiable rather than assumed.

### 3.6 Tests

* One committed dialect-A and one dialect-B fixture. **Do not commit real
  TIFFs** — `.gitignore` excludes `*.tif` for good reason. Write tiny synthetic
  TIFFs in the test with `tifffile`, embedding the real tag structures with the
  pixel data reduced to a few pixels. The structure is what is under test.
* Every field, both dialects, including the unit conversions (m → nm, A → pA,
  s → ns) which are where the bugs will be.
* **Absent means None**: a dialect-A file yields `image_integrations is None`,
  not `1`. This is the §1.3 regression and deserves a named test.
* A header whose `beam_type` disagrees with its directory produces a finding,
  not an exception.
* A corrupt or truncated file returns `None` and logs, rather than aborting a
  batch of 2000.

---

## 4. Phase 5 — Editing project metadata

Creation exists (`init-project`). Editing does not. You asked for API first,
then CLI, then GUI — that is the right order and it is what `CLAUDE.md`'s
incremental rule implies.

### 4.1 API

In `operations/` (it writes to disk), new `operations/metadata_edit.py`:

```python
def update_project_metadata(
    project: Project,
    values: Mapping[str, str | None],
    *,
    dry_run: bool = False,
) -> MetadataChange: ...
```

Design constraints, all inherited from existing package rules:

* **Writes only to `AM_metadata.yaml`, never to `experiment.yaml`.** The latter
  is Autolamella's file; this package does not own it. The override mechanism
  in `metadata.py` exists precisely so nothing has to.
* **Rewrites the whole file, preserving unknown keys.** Read, update the
  `metadata:` block, write back. A user may have added keys the package does not
  model, and losing them silently would be the worst possible failure.
* **Returns the change before applying it.** `MetadataChange` lists per field
  the old value, the new value, and which file the old one came from. `dry_run`
  returns it without writing — the same shape as `plan_transition` in
  `operations/selection.py`. Reuse that pattern rather than inventing a second
  one.
* **Validates before writing.** `lamella_type` must be in `LAMELLA_TYPES`; an
  unknown field name is an error, not a silently added key. Batch validation
  first, per `CLAUDE.md`.
* **Setting a field to `None` removes it**, falling back to `experiment.yaml`'s
  value. Distinct from setting it to `"TODO"`, which marks it as outstanding
  work. Both are needed and they are not the same operation.

### 4.2 CLI

```
am-seg edit-metadata <project-selector> --set sample="S. cerevisiae" --set user="C Berger"
am-seg edit-metadata <project-selector> --clear lamella_type
am-seg edit-metadata --all --set organisation="Rosalind Franklin Institute"
am-seg show-metadata <project-selector>
```

* Project selectors go through `filtering.matches_project` — already supports
  name, relative path and scan-order number. Do not add a second selector
  syntax.
* `--dry-run` prints the change table and writes nothing.
* `--all` is how you fill in a field shared across a session, which
  `init-project` currently handles only at creation time. Note that this
  **copies** the value into each file rather than resolving it from a parent,
  which is `CLAUDE.md`'s explicit rule and the reason projects in one collection
  directory can hold different samples.
* `show-metadata` prints each resolved value with its source file — the data is
  already in `ExperimentMetadata.sources` and nothing surfaces it today.

### 4.3 GUI

A *Metadata* tab, per §7. Build it after the CLI has been used on real data:
the CLI is where you find out which fields you actually edit and which
validation is too strict, and it is far cheaper to change there.

### 4.4 Tests

* Round-trip: write, re-scan, read back the same value.
* Unknown top-level keys in an existing `AM_metadata.yaml` survive an edit.
  **The most important test in this section.**
* `experiment.yaml` is never modified.
* Clearing a field falls back to `experiment.yaml`'s value; setting `TODO`
  makes it appear in `todo_fields` and in `inspect-layout`'s report.
* Invalid `lamella_type` is rejected before anything is written.
* `--dry-run` leaves the file byte-identical.

---

## 5. Phase 6a — A shared aggregation layer

**This is the one genuine architectural decision in this document.**

### 5.1 The problem

Three modules already walk `project → lamella → stage → image` independently:

| Module | Walks the tree to |
|---|---|
| `filtering.iter_matching_records` | yield records matching selectors |
| `survey.survey_collection` | build per-set counts and findings |
| `reporting.summarize_lamella` | count states per modality |

Every statistic in your notes — images per microscope model, per lamella type,
per accelerating voltage, per biological sample, split by SEM/FIB and by
workflow state — is a fourth traversal. And the GUI status indicators
(`02_gui_improvements_plan.md` §4) are arguably a fifth.

Three is the point at which the duplication is worth naming. `01_refactor_plan.md`
§6 flags this; here is the fix.

### 5.2 Design: group and count, over records the existing filter already yields

Do **not** build a general query engine. Build one small module,
`aggregate.py`:

```python
def count_by(
    records: Iterable[ImageRecord],
    key: Callable[[ImageRecord], Hashable],
) -> dict[Hashable, int]: ...

def cross_tabulate(
    records: Iterable[ImageRecord],
    row: Callable[[ImageRecord], Hashable],
    column: Callable[[ImageRecord], Hashable],
) -> dict[tuple[Hashable, Hashable], int]: ...
```

plus a small library of named key functions (`by_modality`, `by_state`,
`by_microscope_model`, `by_sample`, `by_lamella_type`, `by_voltage`) so callers
name a breakdown rather than write a lambda.

The records come from `filtering.iter_matching_records`, unchanged. Metadata-derived
keys close over the `dict[Path, ImageMetadata]` from §3.4, so `aggregate.py`
needs no knowledge of TIFF headers and stays trivially testable with
hand-built records.

**Why this and not pandas.** `collections.Counter` does all of the above in a
few lines, with no dependency, and `01_refactor_plan.md` §2.2 has just removed
pandas as an unused requirement. Reintroduce it only if a real need appears —
and reintroduce it as an extra (`[stats]`), not a core dependency. Counting
2700 records is not a job that needs a DataFrame.

**Scope boundary.** `aggregate.py` counts. It does not filter (that is
`filtering.py`), does not read files, and does not format output (that is
`reporting.py`). If it starts growing a query language, that is the signal it
has outgrown its purpose.

### 5.3 Retrofit, don't just add

Once `aggregate.py` exists, move `reporting.summarize_lamella`'s counting onto
it, and the GUI status indicators too. Adding a fourth traversal alongside three
others would make the problem worse; the point is to end with fewer.

`survey.py` is the judgement call: it produces findings as well as counts, and
findings are not aggregation. Move its counting and leave its findings alone.

---

## 6. Phase 6b — Exports and statistics

Three outputs, all built on §5.

### 6.1 Per-image metadata CSV

```
am-seg export-metadata-csv [filters] -o images.csv
```

One row per image: identity (project relative path, lamella, stage, modality,
variant), workflow state, label state (present / empty / missing — reuse
`label_io.is_label_file_empty`), the resolved project metadata, and the §3
header fields.

This is the direct replacement for `read_sem_tiff_header_csv.py`, and it is
better in the way that matters: that script needed a CSV of paths and a
hand-maintained `sessions_metadata_manual.csv` to join session metadata onto.
Here both come from the index, so the manual join disappears.

Accept the same filters as `create-training-csv`, via the same
`iter_matching_records` call.

### 6.2 Per-project metadata list

```
am-seg export-projects-csv -o projects.csv
```

One row per project: relative path, name, `created_at`, every
`ExperimentMetadata` field, **which file supplied each value**, outstanding
`TODO` fields, and counts (lamellae, stages, SEM images, FIB images, labels,
corrected).

The source column is worth including even though it looks like bookkeeping: it
is how you answer "why does this project say the sample is X" without opening
two YAML files.

### 6.3 Statistics

```
am-seg stats [filters] --by microscope_model --by state
```

Prints a table; `--csv` writes it. Breakdowns from your notes, all reduced to
one key function each:

* images per microscope model (and per serial — see §3.2)
* images per lamella type
* images per accelerating voltage
* images per biological sample
* each of the above split by SEM/FIB, and by selected/deselected/corrected

`--by` twice gives a cross-tabulation, which is what "is the training set
balanced" actually asks: not "how many Arctis images" but "how many corrected
FIB images at 30 kV on each instrument".

### 6.4 Metadata filters for training CSV export

The point of all of it. `create-training-csv` gains filters mirroring the
breakdowns:

```
am-seg create-training-csv --modality SEM --state corrected \
    --microscope-model Arctis --lamella-type lift-out --voltage 2000
```

Implementation note: metadata filters must apply **after** structural filtering,
because that is what keeps header reading proportional to the query — filter
structurally first, read headers only for what survives, then filter on
metadata. With §1.5's numbers this is not a performance necessity, but it keeps
the cost bounded if a collection grows.

Extend `iter_matching_records` with an optional metadata-predicate parameter
rather than writing a parallel iterator. One traversal, one place.

### 6.5 Plots

Optional extra `[plots]`, matplotlib, and **a fixed small set**: a bar chart per
`--by` breakdown, a grouped bar chart for a cross-tabulation, PNG and SVG out.

```
am-seg stats --by microscope_model --plot breakdown.png
```

You flagged feature creep here yourself, and you were right. The boundary that
holds: **the package produces counts; it produces plots only for breakdowns it
already computes.** Anything bespoke — a figure for a paper, an unusual
comparison — is a short script reading the CSV, and the CSV export is what makes
that easy. If a plotting request cannot be expressed as "bar chart of a `--by`
breakdown", it belongs in a script, not here.

Never make matplotlib a required dependency. Every stats command must work
without it and simply decline to plot.

---

## 7. Phase 7 — GUI tabs

### 7.1 This reverses a recorded decision — record the reversal

`docs/design_record_gui.md` §1 states:

> Prediction→label conversion, CSV export, compression. Those stay CLI-only for
> now; the GUI is a curation tool, not a second front-end for everything.

That was a good decision at the time and it kept the curation window focused.
It is now being reversed, and `01_refactor_plan.md` §3.4 asks for the reversal
to be written into that document rather than left as a silent contradiction.

**The reason it is right to reverse:** the CSV filters in §6.4 are the case the
original decision did not anticipate. Choosing a training set is iterative —
apply a filter, look at the counts, adjust — and a command line that re-scans
and re-reads on every attempt makes that loop slow enough that you stop
exploring. That is a genuinely different activity from "run the export", which
is what the original decision had in mind.

**The boundary that replaces it**, so this does not become a licence for
everything: *the GUI hosts operations that are iterative or exploratory. Batch
operations stay CLI.* By that test, CSV filtering and metadata editing move in;
`convert-predictions`, `compress-labels` and `migrate-layout` stay out — you run
each once and read the log.

### 7.2 Structure

Tabs matching the workflow, as you suggested:

| Tab | Contents | Status |
|---|---|---|
| **Curate** | The existing window, unchanged | Built |
| **Metadata** | Per-project metadata viewer/editor (§4) | New |
| **Export** | Filters, live counts, statistics, CSV/plot output (§6) | New |

The existing curation window becomes the first tab and must not otherwise
change. Keep `MainWindow`'s current contents as one widget and add a
`QTabWidget` above it — a mechanical change that keeps the curation code out of
the diff.

**The Qt-free controller rule still holds.** Each tab gets its own controller in
`gui/`, testable without a display, exactly as `CurationController` is. This is
the structural rule of the UI and adding tabs is not a reason to bend it.

### 7.3 The Export tab is where the value is

Filters on the left, a live count in the middle, export buttons on the right.
The count updates as filters change — *"1,204 images · 402 corrected · 31 empty
labels excluded · 712 after one-per-scene"* — so the composition of the export
is visible before it is written.

That last figure matters: `keep_one_condition_per_scene` and the empty-label
exclusion both silently drop images today, reported only in the log. Showing
them is arguably the single most useful thing in the tab.

---

## 8. Phase 8 — Label quantification

Deliberately last, and fenced.

Your examples: proportion of lamellae with a crack or hole; size distributions.
Genuinely useful, and genuinely a different kind of work from everything above —
it reads **pixel data**, not headers, so it is orders of magnitude more
expensive and it needs decisions the rest of this document does not.

Before building it, three questions need answers that only you have:

1. **What counts as a crack?** A connected component of the crack class above
   some pixel area? Per image, or per lamella (a crack visible in one frame of
   twenty)? This is a scientific definition, not an implementation detail, and
   it must be written down before code, because everything downstream inherits
   it.
2. **Measured on which label?** Only `corrected` ones, presumably — a
   prediction-derived label would measure the model, not the sample. Say so
   explicitly.
3. **In pixels or in nanometres?** Nanometres needs the pixel size from §3, and
   is the only version comparable across images. Almost certainly nm, which
   makes this depend on phase 4.

Implementation sketch, once those are settled: extend
`operations/label_io.py`, which already opens label files for
`is_label_file_empty`, with per-class pixel counts and connected-component
sizes. Feed the result into `aggregate.py` as another key function. Cache
per-label results keyed by path and mtime — **here** a cache is justified,
because reading and labelling components in a 3072×2048 image is milliseconds
to seconds, not 0.5 ms.

**Where to stop.** Per-class area, component count, component size distribution.
That is enough to answer "what proportion have a crack" and "how big are they".
Shape descriptors, morphology, per-class co-occurrence matrices: those are
analysis, and analysis belongs in a script reading the CSV. If phase 8 starts
growing an image-analysis library, it has gone past its purpose.

---

## 9. Open questions

Ordered by how much they would change the plan.

1. **Is data-repository deposition still a goal?** `CLAUDE.md` lists it as a
   main function; nothing implements it and your notes do not mention it. If it
   is live, it needs its own plan (target repository, required schema,
   directory layout) and it would substantially reuse §6. If not, remove the
   claim. See `01_refactor_plan.md` §3.1(d).
2. **Should `AM_metadata.yaml` record per-lamella metadata?** Today metadata is
   per project. If lamellae in one project can be different samples — plausible
   given `CLAUDE.md`'s warning that lamellae in one *directory* may be — then
   `sample` is on the wrong object, and finding that out after building the
   editor is expensive.
3. **Does a header disagreeing with its directory ever happen in real data?**
   §3.2 reads `beam_type`; if disagreements exist, the question is whether they
   are miscopied files (a finding) or something legitimate. The §1.4 census read
   `beam_type` but did not cross-check it against the directory — a five-line
   script would settle it before §3 is built.
4. **What are the 12 unreadable images in §1.4?** Truncated files, a third
   dialect, or something benign. Worth identifying: if they are truncated, that
   is a data-integrity finding `inspect-layout` should probably report, which is
   a different outcome from "the reader needs to be more tolerant".

*(The earlier draft's fourth question — how many microscopes are in the archive
— is answered in §1.4: three.)*
