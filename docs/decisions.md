# Decisions

Why the software is built this way, and facts not safely derivable from the code: the
code shows what happens, not what breaks if it changes. Anything one module does belongs
in that module's docstring; work not built yet belongs in
[`04_future_work.md`](04_future_work.md). Correct an entry in place rather than
appending to it.

---

## Architecture

- **No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python
  controllers, which is what makes navigation, cache eviction and state expansion
  testable without a display server. Adding a tab is not a reason to bend it.
- **No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few
  lines with no dependency, over thousands of records rather than millions.

## Filesystem and identity

- **A label directory is `<image-directory>_labels`, a sibling of the image directory.** A pure
  string function both ways with no lookup table, which lets one module hold the convention;
  `sem/` is no special case, its private and shared names coinciding. Conditions share a label
  by default, and giving one its own means creating the directory rather than configuring
  anything. Label directories are the only ones anything here renames — we do not own `sem` and
  `fib`, and renaming those breaks reopening the project in Autolamella. **Legacy spellings stay
  readable indefinitely** (`labels`, `labels_fib`, `labels_FIB`), a new label landing in
  whichever directory an image set already has, so cosmetic tidiness is the only reason to
  normalise — hence `scripts/rename_legacy_label_dirs.py` rather than a command.
- **The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`,
  `sem_lowmag` all work with no code change, authoritative imaging parameters coming from the
  image headers. A variant may not be named `labels`, `deselected`, `corrected` or
  `predictions`: those collide with the label naming rule, so `layout.py` validates it and
  `inspect-layout` reports a violation rather than mis-indexing.
- **A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`.**
  `init-project` writes the latter rather than fabricating the former: `experiment.yaml` is
  Autolamella's file, and writing one misrepresents provenance and could collide if Autolamella
  ever opens that directory. `protocol.yaml` looks like a third marker and is not usable as one,
  some genuine Autolamella projects having `experiment.yaml` without it. `init-project`
  **refuses a directory that holds projects of its own**: that is a collection, and a marker
  there makes every unmarked dataset beneath it a *lamella* of one project — several samples
  under one metadata record, invisible in the output. `--recursive` does the right thing.
- **Identity is the path relative to the scan root, never the name** — for a project, and for a
  lamella within its project. The reference collection holds two different projects whose name,
  `created_at` and recorded path in `experiment.yaml` are all identical, so keying by name
  silently drops one; and two projects each hold an `05-pet-mutt`, which an unqualified row
  reported as one lamella of 140 images. A lamella is counted as `project / lamella`.
- **The lamella and stage levels are not always directories.** Older datasets collapse them;
  the missing level is synthesised in memory with the parent standing in, and the model records
  that it is synthetic. Data on disk is never reorganised to fit the newest layout.
- **The series is a property of an image set, not of a directory.** A lamella directory holds
  reference images of both beams and alignment images at once, so a set is a *group of
  filenames* within a directory rather than the directory itself; `image_files_of` is the one
  place either shape is read back.
- **A loose-image stage does not end the stage search; a directory-shaped one does.** The
  invariant the 2026 layout rests on: a lamella holds reference images loose and its milling
  tasks below, and each task holds reference images loose and its Adaptive Polishing stages
  below. Were a lamella of reference images to end the search the way `sem/` does, every stage
  beneath it would vanish from the index silently. Nothing sits below an image directory.
- **A loose set's label directory is resolved exactly, with no fallback chain**, unlike
  `resolve_labels_directory`, which falls back to the shared `<modality>_labels`. A dataset that
  collapses the stage level keeps its Adaptive Polishing images and its `labels/` in the lamella
  directory, so a reference set falling back would pair every reference image with a label of an
  entirely different scene: right dimensions, plausible shapes, wrong data.
- **A series nobody has started labelling is not reported as unlabelled.** `no-labels-yet` fires
  per image set, and the loose series add 460 sets nothing has annotated — enough to bury every
  finding worth acting on. `layout.LABELS_EXPECTED` names the series that get the report; a set
  whose label directory *does* exist is reported on normally whatever its series.
  `unexpected-stage-name` is narrowed the same way, to a stage found by holding image
  *directories* — a lamella was never going to be named `adaptive_polish*`.
- **The survey counts what it could not reach at all.** Every other finding describes
  something the conventions recognised; `unindexed-images` counts image files no image set
  claims, which is the failure that hides best — a project holding 430 images can survey as
  holding 100 with nothing saying so. It is also what makes the project-root overview images
  visible without indexing them as lamella data. Labels, predictions and deliberately ignored
  names are excluded, or annotating a project would report thousands.
- **A loose frame is ordered by its header's acquisition time, across its whole series.**
  Nothing in a loose filename orders it: 44% carry no timestamp, and the epochs that do appear
  are when the milling *stage* started, up to eleven minutes before the image. **mtime is worse
  than useless** — not one of 4,000 checked was within two minutes of the header, the collection
  having been copied. Nor can the stages be walked in order: a lamella directory holds reference
  images spanning the whole session while the milling tasks below hold theirs in the middle, so
  they interleave. `_in_acquisition_order` therefore sorts by time whenever every frame has one
  — exactly a loose series — and leaves an Adaptive Polishing timeline, ordered by the number in
  the filename, alone. Ties go to `alignment_run_order`, ranking the fixed names Autolamella
  writes: a run takes several images inside the whole second a header records, so time alone
  left 11 of 121 runs reading `final_image` before the last `new_image`.
- **Only the scanner opens an image, and only a loose one.** Its header gives the beam
  where the filename does not state it — 668 images here, every one ion-beam — and the
  acquisition time always; the two come from one read, memoised, costing the collection
  scan 1.3s to 2.8s. Discovery opens nothing: it asks `holds_loose_images`, which reads
  filenames. The survey opens only what it must, reporting what is on disk rather than
  ordering it.
- **One record per acquisition, where a lamella holds the same picture several times.**
  Autolamella duplicates the image an alignment aligns *against* in two ways and one rule
  cannot see both: it copies it into every run using it (121 `reference_image.tif` over 51
  acquisitions here), and saves it outside the runs under a name of its own
  (`Polishing_alignment_reference_20-12-30_ib`, `ref_alignment_ib`). So a run's
  `reference_image.tif` is a copy wherever the directory holding that run's `Alignment/` folder
  already holds an image of the same acquisition time, and **the original is kept** — the one
  file several runs copy, its name carrying the milling task. That matched 36 of 51 run
  references, all byte-identical to their match, none of the other 15 touched, the directory and
  the time agreeing independently on every one: **neither signal carries it alone**. Otherwise a
  copy is a filename and a time another directory has used, which collapses the run copies among
  themselves; there the filename is the guard, time and size alone having merged 38 groups of
  genuinely different pictures. **Nothing is hashed** — the time is already read, and with either
  rule's second condition it partitions these as contents do.
- **A fixed filename marker is matched to the shape Autolamella writes, not loosely.**
  Everything after `ref_` is a *task* name somebody chose, so a loose marker collides with one.
  `ref_alignment` is matched entire rather than as a prefix — as a prefix it reads a task called
  `Alignment Check` as an alignment image, moving that task's reference images into a series
  nothing predicts or labels. `_centring_` is anchored to the end, the plugin writing
  `<lamella>_centring_<modality>` and nothing else — as a bare substring it drops every image of
  a task called `Centring`, and a *deliberately ignored* image is not even reported as
  unrecognised, the worst way for a naming rule to be wrong. Tightening cost nothing: 1,704
  `beam_shift_alignment_*`, 76 `ref_alignment_ib` and 17 centring images classify unchanged.
- **Where a record's label and predictions live is carried, never re-derived.** Walking up
  from the image path only ever worked for an Adaptive Polishing image — one level below its
  stage, in a directory naming a modality. A reference image sits *directly* in a directory
  naming a lamella, so deriving put its files in a `<lamella>_labels` and
  `<lamella>_predictions` outside the lamella: predictions were written where the next scan
  never looked, and `convert-predictions` reported every one as missing. The scanner resolves
  both once per image set and the record carries them, so reading and writing cannot disagree;
  a record that does not say raises rather than guessing. Predictions follow the label rule
  above for a loose set too — exact name, no fallback — the shared `<modality>_predictions` in
  a collapsed layout being the Adaptive Polishing pipeline's own output.
- **A ticked lamella covers only the series on screen.** Curate shows one series at a time,
  and what is ticked has to be the whole of what an operation touches; filtering by modality
  alone reached every series, so predicting a lamella while looking at its Adaptive Polishing
  timeline predicted its reference and alignment images too. Each lamella is asked what it can
  show of the chosen series, as navigating to it would — `frames.resolve_series` — so one
  holding none of it contributes what it has rather than nothing.
- **`predictions/clean/` is indexed but can never become a label.** Adaptive Polishing's
  post-processing rewrites the intensity values, so they no longer index the segmentation
  classes: a label converted from one looks entirely valid — right dimensions, plausible shapes
  — and teaches a model the wrong classes. That will not change upstream, so it is a property of
  `layout.py` (`is_convertible_prediction`) rather than a choice offered to the caller. Still
  indexed deliberately: otherwise a project holding only post-processed predictions reports "no
  predictions", indistinguishable from one the pipeline has never reached, and those need
  opposite actions.

## Workflow state

- **State is declared in a file, not implied by location, and the key names the label.** Every
  directory holding image and label directories carries an `AM_state.yaml` naming the groups
  that are not selected, keyed by the group's label path (`<label-directory>/<filename>`,
  relative to the state file) whether or not that label exists. Keying on the label rather than
  the image makes an image disagreeing with its label *unrepresentable* rather than merely
  detectable: the conditions of one scene resolve to one key, so a change reaches all or none.
  The key is resolved rather than located, so it survives annotation beginning.
- **Anything not listed is selected.** An absent entry and an absent file give the same answer,
  so images dropped into a stage tomorrow arrive selected with no reconciliation. The cost is
  that "reviewed and kept" cannot be told from "not yet looked at".
- **The three states are mutually exclusive, and both representations enforce it.** A mapping of
  key to state cannot express two states for one group, where a file grouping keys under
  `corrected:` / `deselected:` headings could. In memory `ImageRecord` carries one `state` field
  rather than a `selected` and a `corrected` flag for the same reason: two booleans have a
  fourth combination that means nothing, and every reader must then spell out `selected and not
  corrected` and be trusted to get it right. Duplicate keys are rejected on read rather than
  resolved last-wins, a silent winner being the failure the mapping was chosen to prevent.
- **Whether a group is corrected is asked of the record, never of a path.** Every guard
  protecting proofread work — conversion, compression, label saving — reads the declared state
  the scanner put on the record; a guard that quietly stops firing is worse than none.
- **A stage is transcribed whole or not at all, and one that has not been is refused.**
  `migrate-state` leaves a stage alone if anything in it is ambiguous — images disagreeing with
  their label, a file in two state folders — because writing the groups that were fine leaves
  the rest unlisted, and unlisted reads as *selected*; a state change refuses such a stage for
  the same reason. It is also **the only thing that reads a folder name as a state**: the
  scanner does not look inside `deselected/` or `corrected/` at all, so a nested file is
  *missing and reported* rather than present and wrong, and those names stay reserved.
- **A stale entry is reported, never repaired.** Declared state can disagree with the filesystem
  in ways location could not, and each needs a different action, so `inspect-layout` separates
  them: unmigrated folder state; nested files after migration, wanting `--move-files`; an empty
  state folder; and an entry naming a group that is not there. That last is kept rather than
  dropped — a group that vanished because a drive was not mounted must not have its curation
  deleted for it.
- **A write is read-modify-write, and refuses a file that changed underneath it.** Two windows
  on one collection, or a window and a command line, would otherwise discard each other's
  curation with no error. Every file the batch will touch is compared against what the plan read
  *before the first is written*, so the answer is "nothing happened" rather than "some of it
  happened"; each write then checks again for itself. That is what lets `plan_transition` and
  `apply_plan` be separate — the decisions are the previewed ones however long ago, the safety
  check fresh. Not a transaction: each stage's records come into step as that stage lands.
- **Every label write goes through `io/atomic.py`, including compression.** It restores the
  destination's own permissions before the rename, because `mkstemp` creates owner-only and
  OpenCV truncates rather than recreates — a recompressed label kept `0o600` becomes unreadable
  to everyone but its author, which on a shared archive is the same as losing it. It also takes
  a fingerprint, so a label something else rewrote mid-run is left as it now is. Compression has
  to decide *after* encoding, the saving being unknowable any other way, so `write_atomically`
  takes a `commit_if` predicate and reports whether it committed — a run that skips a file says
  which of no saving, a dry run, or "somebody else rewrote it" applied, only the last of which
  is worth acting on.
- **Overwriting a label is tiered by what replacing it would cost, not by one flag.**
  Conversion sorts destinations three ways: an empty placeholder costs nothing to replace; a
  pseudo-label holding annotation is regenerable machine output, replaced once the user
  confirms; a corrected group is refused, and no argument to `plan_conversions` reaches it.
  Stricter than `compress-labels`, which *can* be confirmed for corrected labels, because
  compression preserves the pixels exactly where conversion replaces proofread work with a
  machine guess — re-converting a corrected label means setting that group back to selected
  first. That is also why **compressing is not scoped to a selection**: there is no label it
  should be withheld from, and a per-selection button leaves uncompressed exactly the labels
  nobody happened to tick.
- **Corrected is a claim, and it is checked where it is made.** A group marked corrected with no
  label or with a placeholder breaks the claim, and a training export is *right* to drop it —
  what is wrong is doing so in silence, leaving a training set short of exactly the images you
  believe you annotated. So the window asks on either side of the gap, both times through
  `classify_labels` so the two dialogs cannot disagree. The export side asks only about
  *corrected* groups: an unlabelled selected image is the ordinary condition of an archive being
  worked through, and a warning firing on every export is one nobody reads. The command line
  warns and proceeds where the window asks.
- **What a state change applies to is a set the user builds, at one level at a time.** Ticking
  a project clears the ticked lamellae and vice versa, and picking frames out clears both — so
  what is ticked on screen is the whole of what a change will touch. Not a precedence rule
  between levels: that leaves ticks visible which decide nothing and, once a tick outlives the
  project it was made in, ticks *invisible* which decide everything, which is how an operation
  reaches a project nobody was looking at. Everything reaches `transition_records` through
  `scoped_records()`, so group expansion and the corrected-data guard apply unchanged.
- **A scan the user did not ask for keeps the selection; one they did asks again.** Every
  operation that writes files is followed by a rescan, and preparing labels is two of them
  against the same set, so a rescan that emptied the selection would have *convert* ask for the
  choice *predict* was just given; pressing *Scan* starts clean. What comes back is filtered
  against what the scan found, and **frames come back by which acquisition they are, not by
  where they sat** — `LamellaFrame.key` rather than an index — an image added ahead of them
  shifting every index after it. The *view* is still restored by index: landing on a
  neighbouring frame is a nuisance, acting on one is a fault.

## Metadata

- **Nothing is inherited from parent directories.** A collection directory groups projects by
  session or storage convenience, so lamellae in one *directory* may be different samples;
  lamellae in one *project* do share a sample, which is why that is the right level.
- **The Add data tab suggests, and a suggestion is never a value.** It offers the values other
  projects already use for a field, computed in the controller and reaching a file only when the
  user saves, so `build_experiment_metadata` still resolves from the two files and nothing else.
  `project` is on that list because it names something several datasets share, which is how a
  collection ends up holding two spellings of one project.
- **`project` names the wider project; the directory is the `autolamella_project`.** Different
  facts, and holding both under one word showed as a `project_project` column in one export and
  a `metadata_project` in another. The directory is what the scanner indexes and Curate lists,
  so it is qualified wherever written; the metadata field keeps the plain name. Neither is
  derived from the other, which is why Add data offers no button to copy one into the other.
- **`grid_type` is free text where `lamella_type` is a vocabulary.** A preparation is one of
  four things and a typo is a real error; a grid is a material, a mesh and a film in
  combinations no list would survive, and rejecting a real grid is worse than tolerating two
  spellings of one. The Add data tab offers the values already in use, which is what keeps
  spellings consistent without a vocabulary to maintain. Like `sample`, it cannot be pre-filled
  from a shared defaults file: a grid belongs to the specimen, and one collection directory
  here holds a yeast project beside a virus one.
- **Every image header carries settings for both beams, and some values exist for only one.**
  `microscope_state.*` describes the beam that took the image; `system.*` holds both beams'
  configured defaults, so reading `system.ion.voltage` from an SEM image yields the FIB's 30 kV
  — plausible and wrong. `image_metadata.FIELDS` therefore confines beam-specific fields,
  including the fallback search, to the beam that fired, and declares which beams a value exists
  for at all. A plasma gas belongs to an ion column and an emitter type to an electron one, so
  they are two fields rather than one "beam source", which reported the FIB's Argon on SEM
  images and the SEM's `FEG` emitter as a plasma gas. A blank `plasma_gas` on a FIB image is
  therefore a real answer, a gallium column having no plasma.
- **A value is stored in the unit its field name states, converted where the header is read**,
  so `acceleration_voltage_kv` is kV even though both dialects record volts and nothing
  downstream converts anything. A breakdown grouping a *measured* value rounds it, a header
  recording what was measured rather than what was set: a voltage to 0.1 kV, a pixel size and
  beam current to four significant figures, those two being derived from the magnification and
  so differing in the last decimal between two images of one field of view. Significant figures
  rather than decimal places, pixel sizes spanning decades. The conversion itself rounds, to
  six decimals in the stored unit: a header holds 6e-05 m, which binary floating point cannot
  represent exactly, and multiplying by 1e6 turns that into 59.99999999999999 um. Six decimals
  is finer than anything these instruments report and ten orders of magnitude coarser than the
  noise, so only the noise goes — a beam measured at 29.996416 kV against a 30 kV set point is
  the header's own value and survives.
- **A declared path counts as matched only if its value converts.** A key can be present and
  still state nothing: a blank, a `"None"` marker, or uninitialised memory, which is what the
  FEI dialect leaves in the stage keys it never fills — `Stage.StageX` reads 6.95e-310 while
  the real position sits in `Stage.CompustageX`. So a subnormal or non-finite number reads as
  absent, and the reader moves on to the next candidate path and then to the search fallback
  rather than stopping at a key that merely exists.
- **A stage position is comparable only within its coordinate system, so the system is a
  column of its own.** The 2024–2025 projects state specimen positions, z around −59 µm, and
  the 2026 ones RAW positions, z around +32 487 µm; the same archive holds `Specimen`,
  `SPECIMEN` and `RAW`. The value is carried as the header spells it rather than normalised
  into a vocabulary this package would then have to maintain. A FEI cryo stage states its
  position under `Compustage*` and a tilt about x, and nothing that is a rotation about z, so
  `stage_rotation_deg` is empty there rather than filled from the tilt.
- **Three programs take part in an acquisition, and the header names their versions
  misleadingly.** openfibsem fills `system.info.hardware_version` from the *microscope
  server's* version and `system.info.software_version` from the *scripting client's*, so both
  keys say the opposite of what they hold; `experiment.application_version` is the application's
  own. They are on unrelated release series — xT at 28.x, AutoScript at 4.x, autolamella at
  0.x — so one "software version" column would answer a different question per image. The
  fields are named for what they are, and the microscope's own version is the only one the FEI
  dialect states, under `System.Software`, which makes it the only one covering the
  pre-Autolamella datasets. That the two are one quantity is not an assumption: one Arctis
  writes 28.4.1.25560 under both keys, in the two dialects.
- **A missing measurement is never a default.** The one exception is the integration counts,
  where absence means "not enabled" — that is decoding the field, not defaulting it.
- **Headers are never read during scanning.** Metadata is a mapping keyed by image path,
  built by the commands that need it, so a scan stays a filesystem walk. Reading one header
  is cheap; reading one per image on every GUI launch is not.
- **Where each header value lives is declared once, in `image_metadata.FIELDS`.** That module is
  to header keys what `layout.py` is to directory names: the alternative is a hand-written
  extraction block per field, so adding a field means another block and adding a dialect means
  editing every one. Supporting a third dialect should be a column in a table.
- **A label carries no header.** Its scale comes from the image it was drawn from, rescaled by
  the ratio of their widths, a label converted from a prediction not necessarily sharing its
  image's pixel dimensions. `label_measure` returns no areas rather than assuming 1:1.
- **Most TIFFs in the collection have no acquisition header, and should not.** Predictions,
  labels and autofocus frames are not acquisitions; only indexed *images* carry one. A count of
  "unreadable headers" taken over `rglob("*.tif")` measures the wrong population.

## Counting and export

- **A filter and a breakdown are one declaration.** The same key function used two ways — group
  by it, or keep the records whose group was chosen — so the filter lists come from `BREAKDOWNS`
  rather than a second table, and cannot disagree with it about what a value *is*. Two
  consequences: "(not recorded)" is selectable, the only way to find images whose header would
  not read; and a measured property is not offered, being unable to answer before Measure is
  pressed. Modality, workflow state and the label filter keep dedicated controls, being fixed
  questions rather than data.
- **A command that reports covers every series; a command that acts stays on Adaptive
  Polishing.** The loose series more than doubled what a scan holds, and none of that may
  change what an existing command prints or writes. So `inspect-layout` and `stats` cover
  everything and say which is which, while everything that writes, exports or measures defaults
  to `--series adaptive` (`cli.DEFAULT_SERIES` and `REPORTING_SERIES`). `by_series` returns the
  slug rather than the display label, a value counted having to be a value `--series` accepts;
  and unlike a variant a series is a closed vocabulary, so `--series refrence` raises — a typo,
  not a request for an empty set.
- **The milling task is resolved against each lamella's own vocabulary, never a pattern.**
  Task names come from whatever protocol somebody wrote — 26 distinct subjects here, `Rough 1`
  spelt both ways — so no fixed set of patterns keeps up, and none splits
  `Rough-Milling-Rough-2` into a task and a stage. A directory or filename resolves by offering
  every run of its words to the names that lamella's `experiment.yaml` records, longest first.
  **Stages roll up into their task** and there is no separate stage axis, `stage-name` already
  answering "which directory". The directory is tried first and is the only thing resolving a
  name Autolamella spells two ways (`Milling/Fiducial/` for `Mill Fiducial`); the lamella
  directory is never offered, or a petname sharing a word with a task claims every image below
  it, and a name two tasks claim is dropped rather than guessed. **None is a real answer** —
  beam-shift alignment, an alignment run's own images and every Adaptive Polishing stage
  predating `Milling/<Task>/` name none, 4,818 of 8,895 images here. **Timestamps were ruled
  out**: the windows in `history` are sparse rather than tiling the session, and
  `ref_MillRough_start_eb.tif` falls inside the `SetupLamella` one.
- **Counting is split between the tab and the commands, not moved.** The Export tab counts the
  set its filters describe, choosing a training set meaning looking at what survives a filter,
  and that loop cannot run through a command that prints and exits. `stats` keeps the
  whole-collection sweep and `quantify-labels` the per-lamella table: numbers per class per
  lamella is not something anyone acts on one row at a time.
- **Opening the Export tab reads the headers and nothing else.** Half a millisecond an image,
  two seconds over the reference collection, and it is what every acquisition filter and
  breakdown is made of. Offering them greyed out until a button was found and pressed was a
  control that did nothing with no way to tell why.
- **Opening a label is asked for, never assumed.** Ten times the cost of a header — 1,232 files
  and 14 seconds here — and it answers a different question: what is *in* the label. So the
  summary shows what a record knows about itself and says plainly that the rest needs Measure
  labels. **Measured over what the filters match**, not the collection, so widening a filter asks
  again for what it brought in. One pass answers the exportable count, the label filter and the
  label-state breakdown together, whether there is anything in a label falling out of the class
  counts; `classify_label` is the shared rule and `_label_empty` the shared cache, keeping two of
  either having meant reading every label twice.
- **What a measured number means is set beside the breakdown, not among the filters.** The crack
  threshold and the area bin resolution change how a measurement is *read*, not which images are
  in play. Changing the threshold re-measures at once rather than spending the Measure button
  again: `read_label_pixels` caches component pixel counts *without* the threshold, so a new one
  costs arithmetic rather than a second pass; `load_measurements` keeps its headers likewise.
- **A filter is added, not permanently shown.** The breakdowns cover far more properties than
  any one export is chosen by, so one list per property leaves too little of each to read and
  buries the three that matter among a dozen nobody touches. A row carries the property it is
  on, and a property already in one row is not offered in another, which stops two rows
  disagreeing. Removing the last row empties it instead, a box with no filter in it still
  saying what could be filtered on. **A few properties are counted by but never filtered on** —
  a lamella and a fully qualified milling stage run to hundreds of values, an electron source
  does not vary here at all — named by `UNFILTERABLE`; they stay breakdowns, and the command
  line still filters on `--electron-source`.
- **A stage is filtered on by its name, and counted on by its path.** `stage` names a folder
  (`project / lamella / adaptive_polish_<timestamp>`), which finds one run again but is useless
  as a list to choose from; `stage-name` is that with the timestamp removed by
  `layout.stage_base_name`. **What names no milling stage counts under `LAMELLA_ROOT`** — a
  stage synthesised from the lamella, and an alignment run directly under one. **A run counts
  under the task it aligns for**, the directory its `Alignment/` folder sits in, Autolamella
  naming a run after the lamella, the task and the time, which is otherwise a row per run
  beginning with a lamella's name. The timestamp pattern lives in `layout.py`: the scanner reads
  it to order stages and this reads it to remove one, and the two must agree on what one is.
- **Series is the one filter that does not open on everything.** Every other control starts wide
  and narrows towards an export; this one starts on Adaptive Polishing, a tab silently counting
  the rest disagreeing with `create-training-csv`. A dedicated control rather than a derived
  filter row, and checkboxes rather than a dropdown, an export covering two series being a real
  request. The metadata CSV names the series of every row — once more than one can be exported,
  a row not saying which it came from cannot be read — shared rather than per-beam in the paired
  shape, a series being decided by filename with the beam suffix playing no part. `task` rides
  beside it on both counts.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than *corrected*: the tab is narrowed towards an
  export rather than opening on one and hiding the archive. The cost is that an image somebody
  decided against can reach a training set — only if it was labelled and deselected afterwards,
  the empty-label rule catching every other case — so the summary line counts the deselected
  images that would actually be written. Visible rather than impossible, "show me everything I
  rejected" being a real question.
- **Both modalities means side by side, not mixed together.** A training CSV holding SEM and FIB
  rows under one `image` column is not something anyone trains on, the two beams needing
  different models. So a selection holding both writes `sem_image, sem_label, fib_image,
  fib_label`, one row per acquisition; one naming a single beam writes `image, label`. A cell is
  blank where that beam has no image or no label worth pairing with one, and a row survives only
  if at least one beam has a label — the empty-label rule drops a row where there is one beam
  and empties a cell where there are two, reported either way.
- **What makes an SEM and a FIB image the same acquisition is stated once**, in
  `frames.group_by_acquisition`. They do not reliably share a stem — older projects name the file
  inside `fib/` with the same `_SEM` suffix — and **what pairs them depends on the era of the
  filename**: Adaptive Polishing images carry an acquisition number, matched within one stage
  directory, while reference and alignment filenames carry none and differ only by an `_eb`/`_ib`
  suffix, so those match on the stem with the suffix removed (`layout.acquisition_stem` beside
  `extract_image_number`). Conditions multiply out — five SEM conditions against one FIB image
  are five rows repeating the FIB path, each a genuine pair nothing here can rank — and "one
  image per label" collapses it. The train/validation split groups by the *acquisition*, a paired
  row holding two images of one scene that splitting would score the model on what it has seen.
- **An export can also copy the files out, and it never overwrites one.** `export-files` and the
  Export tab's *Export images and labels* write the matched images, each label, every contributing
  stage's `AM_state.yaml` and every contributing project's metadata and protocol files under the
  paths they already have — a smaller collection of the same shape, which `am-seg` scans and which
  can be proofread further. Every destination is classified before a byte moves: one holding the
  *same* file is skipped, which is what makes an interrupted export resume, and one holding
  anything else refuses the whole run. Free space is checked first, and a destination inside the
  folder being copied from is refused — it would walk into its own output. Files land by rename
  with their modification time kept, which both leaves whole files behind and lets the next run
  recognise them.
- **`check-training-csv` only checks, and reads the header rather than assuming it.** Nothing
  moves files, so a path that does not resolve means the file is gone; what survives is the
  check and a `--prune` dropping the dead rows, a CSV holding a particular train/validation
  split that losing a few images is no reason to regenerate. Any column called `image` or
  `label`, or ending in `_image` or `_label`, holds a path, so one command checks either shape
  and files written before both. A blank cell is its own answer rather than an absent column: it
  records nothing, so it cannot have gone missing, and keeping it is what lets a pruned copy
  come out the same shape.

## Display

- **A window that has not been drawn cannot say it is working.** `show()` only asks for a
  paint; nothing reaches the screen until `exec()` starts delivering events, so a startup scan
  run between the two gives seconds of blank grey indistinguishable from a hang — it is queued
  at zero delay instead. Where the interface thread does have to block, the line saying so is
  `repaint`ed rather than merely set: an ordinary `setText` reaches the screen only after the
  work it announces has finished, and `repaint` draws one widget synchronously without the
  re-entrancy hazard `BusyDialog` exists to avoid. Anything longer goes to `gui/worker.py`
  behind that dialog, which opens indeterminate and becomes a count as soon as the operation
  reports — one `on_progress(done, total)` contract and one `progressed` signal connected once
  for the whole window, so giving an operation a counter is a callback in its loop rather than
  a mechanism. Only the applying run counts, never the preview, which would fill a bar
  instantly and describe the wrong run.
- **One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a palette and
  a short stylesheet; nothing else picks a chrome colour. Fusion rather than the platform's
  style because the two bindings draw the same widgets differently. Where a rule and the palette
  disagree the rule wins, and a widget palette edited in Python is ignored — yet reads back as
  the value it was set to, so a test asserting on it passes while the screen shows something
  else: **anything about how something *renders* is asserted on the pixels**. The stylesheet
  also suppresses Qt's emphasis on a dialog's default button and a checked tool button, so
  anything relying on either must re-state it; and its 12px padding exceeds a one-character
  label, clipping such a button blank while it reports its text correctly, which
  `COMPACT_BUTTON` exempts.
- **A chosen row is ticked, never highlighted.** Highlighting cannot say what it means: it is
  also how Qt marks the row you are looking at, it is lost whenever the list is rebuilt, and in
  four visible rows over dozens of values there is no seeing what is chosen without scrolling.
  `CheckList` is the one implementation. It reports nothing while its contents are replaced —
  a redraw rather than a gesture, and an owner rebuilding its list in response to the current
  row changing drives itself in a circle. Two further list traps: Qt draws a selected row's icon
  blended with the highlight colour, so a status chip whose colour *is* the information comes
  out wrong on exactly the row being looked at (`chip_icon()` supplies the pixmap for every mode
  and state, and any icon carrying meaning goes through it); and a `QListWidget` with a
  horizontal scrollbar scrolls instead of eliding, cutting rows carrying paths off where they
  differ, so `ElideMiddle` plus `ScrollBarAlwaysOff` keeps both ends.
- **`QFontDatabase.systemFont(FixedFont)` is not a fixed-width font.** On macOS under PyQt5 it
  returns the proportional `.AppleSystemUIFont` while reporting the family it was asked for, so
  a table padded into columns comes out unaligned with nothing saying why; a style hint does not
  help, Qt consulting it only for a family that does not exist. `common.fixed_width_font` names
  real families per platform, asserted on the metrics (`"i"` and `"W"` advance equally) and
  never on `QFont.fixedPitch()`, which reads back what was requested rather than what resolved.
- **The image panes have no scrollbars, their scene is bigger than the image, and zooming out
  lands through `fitInView`.** The scene carries a quarter of the image as slack on every side,
  so an edge can be dragged into the middle of the pane to be worked on — which is also why
  scrollbars go, a scene deliberately larger than its content leaving them permanently visible
  and permanently not full. `downscale_to_fit` uses `min(width_ratio, height_ratio)`, and these
  images being landscape in a portrait pane, **width binds**. Reviewing is done at fit, so the
  curation panes stop there; the proofreading canvas allows one step past it, checking that a
  label runs to the edge of the frame meaning seeing that edge clear of the border. The wheel's
  anchor holds the point under the pointer fixed, which makes zooming *in* follow the eye and
  would leave a frame that now fits pushed into a corner, so at the floor fitting wins;
  `fitInView` also keeps a two-pixel border that scaling to the computed floor would crop, so
  "zoomed all the way out" and "fitted" are the same picture.
- **A summary plot is held twice: pane-sized to step through, whole to look into.** A plot is
  ~180 ms to read however it is asked for (a PNG decompresses whole, so OpenCV's reduced-decode
  flags buy nothing), slower than a held-down arrow key, so stepping is smooth only with every
  plot already in memory — which bounds those copies to the size of the pane. But these figures
  are 4500x3000 and carry annotations existing only at that resolution. So a *fitted* plot is
  the pane-sized copy drawn at one array pixel per physical pixel, the only scale that resamples
  nothing; the whole frame is read last and takes over once magnified past what the pane copy
  holds, with a hair of margin because returning to fit lands either side of exactly 1.
  Read-ahead asks for no more than the cache can keep — unbounded, the read finishing last
  evicts the one that finished first — so `plot_radius` derives from the entry limit.
- **Where a copy is placed is read off the copy, and remembered across a swap.** The copy is
  reduced to a box in *physical* pixels, so drawing it at whatever device ratio the widget
  reports comes out half or twice the size wherever the two disagree; whichever of its edges
  touches that box gives the ratio back exactly. An identity transform says nothing about where
  the view is looking and the scene is deliberately larger than the picture, so drawing 1:1 has
  to centre as fitting would. Where the view *is* looking is held as a fraction of the picture
  rather than re-derived through the scroll position, which is a whole number, so the picture
  crept towards a corner while stepping; it is dropped the moment the user pans.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**,
  which defaults to nearest neighbour. A pane showing something several times its own size drops
  pixels rather than averaging them: against a figure of one-pixel rules, nearest loses whole
  gridlines, so the plot view asks for smooth scaling. Acquisitions do not — zoomed *in* on a
  label, nearest is what shows where one pixel ends and the next begins.
- **The busy dialog is not a `QProgressDialog`.** That class calls `processEvents()` from inside
  `setValue()` whenever it is modal — which this one must be, its whole job being to keep the
  window out of files being written. Processing events there re-enters the event loop from
  inside a queued-signal handler, so the worker's *finished* can arrive mid-update: the dialog
  is closed and its last reference dropped while one of its own methods is still on the stack,
  which shows as the window freezing on the final item of a run. `busy_dialog.py` is a plain
  `QDialog` that processes nothing, and the window detaches it before closing it. Escape is
  refused for the same reason the close button is absent: on a stoppable operation it asks it to
  stop, and on any other there is nothing it could safely do.
- **Whether it can be *stopped* is a different question, answered by what a half-done run leaves
  behind.** Each prediction is written atomically and recorded before the next begins, so a
  stopped run has produced finished files and an accurate account of them, and the next resumes
  from there; recompression is safer still, a re-encoded label holding the same pixels.
  Converting predictions must not stop — it validates the whole set of destinations before
  writing any, and stopping it leaves exactly the half-applied collection that rule exists to
  prevent. Cancellation is a `threading.Event` in `progress.py` polled by the work closure: a
  queued Qt signal reaches the worker when it next returns to its event loop, which inside a
  long call is never.
- **Curate and Label show one series at a time; Export ticks several.** A timeline mixing
  alignment runs with polishing frames is not something anyone proofreads, and a 2026 lamella
  runs to 21 stages across three series. Each tab has one choice, which rebuilds its frame stack
  — the position and the ticks go with it, a frame index meaning anything only within one
  timeline. **What was chosen and what a lamella can show are two things**: `resolve_series`
  falls back to a series the lamella holds without touching the choice, so the next lamella that
  has it opens on it; otherwise a lamella with no Adaptive Polishing images shows an empty
  timeline and, the control being hidden for having one entry, no way back. **The beam follows
  the same rule** in `_settle_mode`, asked of the whole stack rather than the frame on screen, an
  acquisition captured with one beam being ordinary; the plot view is exempt. Both settle in
  `_rebuild_frames`, the one path a change of lamella and a change of series share. The Label tab
  settles on the frames it has to *review* and holds its own choice of each.
- **A plot describes one acquisition or a whole stage, and the window says which.** An Adaptive
  Polishing stage writes one plot per milling cycle into `plots/`, matched to a frame by its
  acquisition number. An alignment run writes one `figure.png` at its root summarising the
  several images it took to converge, so there is nothing to match it to: every frame of the run
  shows it and carries a badge saying so, a figure describing five frames read as describing one
  being worse than no figure. `displayed_plot` prefers `frame.plot_path` over `stage_plot`.
  `AP_summary_plots.png` follows the same rule; `centring.png` is excluded, recording where the
  plugin placed its field of view rather than what the run achieved.

## Environment

- **A save refuses a label whose file changed since it was opened.** Proofreading holds a
  lamella's edits in memory until the reviewer saves, leaving minutes to hours in which
  something else can rewrite a label file — not hypothetical here, this collection living on
  OneDrive, which writes files back under a running program, and the annotation in a corrected
  label cannot be reconstructed. So the size and modification time are recorded on open and
  compared before writing, and a mismatch stops the entire batch rather than the one file:
  which version is wanted is not a question this package can answer.
- **Run the suite from the repository root, in `am_segmentation_manager` or
  `am_segmentation_manager_ml`.** Two ways to get this wrong, neither of which announces itself.
  Run `pytest` from *outside* the repository and there is no `pyproject.toml` above the working
  directory, so rootdir becomes that directory, `testpaths` never applies, and pytest recurses
  into everything below it — from `$HOME` that means importing every `.py` under a 52 GB
  `Data/`, which looks exactly like a hang; anywhere *inside* is safe. Run it in conda's `base`
  and the imports resolve to a different stack entirely, PyQt5 rather than PySide6, giving
  several hundred failures that are all one wrong interpreter.
- **The reference collection is at
  `<collection>/`** — 13 projects, 108 lamellae,
  8,895 indexed images, 54 GB, spanning every layout era from pre-Autolamella to 2026-08.
  Read-only. Point the smoke test at it with `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.
- **Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side effect,
  and `test_gui_*` sorts before most other tests. That is how a locale-dependent `strptime('%p')`
  bug appears only in full runs, and only on a non-English locale. `LC_ALL=… pytest` does **not**
  test this; `AM_SEG_TEST_LOCALE=… pytest` does.
- **Never disconnect a signal from inside the handler for that signal.** A connection list
  mutated mid-emit works under PyQt5 and hangs under PySide6 — the suite stalled indefinitely
  on the first test to run a real background operation, with no traceback, because the
  failure path opens a modal box nobody is there to dismiss. Connect once and track what is
  pending in an attribute. The same shape hides in any connect-per-call pattern.
- **`QProxyStyle.styleHint` segfaults PySide6.** Overriding it to answer `SH_DialogButtonLayout`
  — to fix the button order in every dialog at once, Fusion reporting the Windows order wherever
  it runs and ending a row with *Cancel* — crashes the process while a `QMessageBox` is being
  built. Returning the enum rather than an `int` is separately wrong and only warns. The save
  dialog lays its own buttons out instead; any dialog that cares about the order has to.
- **An offscreen window cannot answer three questions**, and asking anyway produces tests that
  pass on nothing. It is never activated, so `hasFocus()` is False for every widget and
  `focusWidget()` records who asked. Nothing is shown, so `isVisible()` is False whatever the
  code did and `isHidden()` is the state to assert. And a modal dialog has nobody to dismiss it,
  so its `exec_` must be intercepted — per dialog class, not just `QMessageBox` — or the suite
  hangs rather than fails.
- **Every file dialog is Qt's own, never the platform's.** The macOS panel under PyQt5 5.15.2
  offers no selectable files at all for a `*.csv` filter, only folders to navigate into, so
  "Check exported CSV" could not be given a CSV — and because directory dialogs are unaffected
  it reads as a missing feature rather than a broken one.
  `widgets/common.build_path_dialog` sets `DontUseNativeDialog` for all of them; revealing a
  folder in the file manager is not a dialog and stays native. Style hints bite the same way:
  whether Up and Down step through a combo box or open its popup is `SH_ComboBox_Popup`, which
  under PyQt5 with the `macintosh` style steps — so a dropdown you can arrow through is
  `StepCombo`, not Qt's own keys.
- **Figures are built as `Figure()`, never through `pyplot`.** pyplot keeps every figure it
  makes in a global list until something closes it, and choosing a backend for that list is what
  makes a chart drawn over ssh look for a display. A bare `Figure` has neither problem: `savefig`
  works on it, the window attaches its own canvas, and a session of charts is collected as the
  dialogs holding them close. `matplotlib` is in the `gui` and `all` extras rather than only
  `plots`: against the OpenCV and Qt those already install, its size is not what makes an
  environment large.
- **Every `cv2` hazard is a provider conflict.** `environment.yml` pins `nodefaults`, a mixed
  conda-forge/Anaconda environment having paired Anaconda's OpenCV with conda-forge's protobuf
  so that `cv2` would not load at all; and the main environment is conda-first because conda's
  OpenCV and pip's cannot coexist. **Inference therefore lives in a second environment, and that
  one is pip-first**: albumentations, which the vendored preprocessing imports, requires
  `opencv-python-headless` — a third provider of `cv2`, and one no ordering of install commands
  avoids. So `environment-ml.yml` takes only python and pip from conda and the whole stack from
  the `[ml]` extra, where there is exactly one OpenCV; it also lets the torch build be chosen
  per machine, which conda-forge cannot express (`AM_SEG_TORCH_INDEX_URL` selects a wheel
  index). **It is a superset, not a peer**: it gets the window too, via the `qt` extra, with no
  OpenCV. The curation environment is the *lighter* of the two, for someone who never predicts;
  splitting them the other way forces an environment switch mid-workflow.
- **Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
  `ml/environment.py` answers "is inference possible here" without importing it, as
  `io/image.py` does for the image stack. `ml/preprocessing.py` is a verbatim copy of
  AM_training's module, excluded from ruff: reformatting it destroys the only cheap answer to
  "is this still the preprocessing the model was trained with" — `diff`.

## Inference

- **A fact recoverable from the weights is never declared.** A checkpoint is a bare
  `state_dict` whose class count, input channels and tensor names are all readable from it;
  declaring those beside it adds a second statement that can disagree with the first invisibly —
  a description naming four classes over a five-class head loads whatever fits. So `ml/model.py`
  reads them, and `load_state_dict(strict=True)` proves the description's architecture and the
  weights are the same model. What the weights cannot state — preprocessing settings and the
  architecture name — goes in a `.yaml` beside them, each showing itself immediately if wrong,
  which is what makes a default safe. One such default is load-bearing:
  **`decoder_attention_type="scse"` is not part of these models** — the training call passes it,
  smp's `FPN` has no such parameter and swallows it into `**kwargs`, the checkpoints hold no
  attention tensors, and a plain FPN strict-loads all 1015.
- **Which modality a model is for is not recorded anywhere, deliberately.** Nothing in a
  checkpoint says, and `AM_SEM_All_V01.pth` is whatever whoever trained it chose to type;
  reading that as evidence makes the software's answer depend on someone else's file-naming
  habit, and declaring it in the sidecar only moves the guess. So the selection states it, and
  one spanning both modalities is refused — a guard on something the window can show you rather
  than a line in a file you would have to open, and every report leads with the modality.
  **Predicting with the wrong model is cheap to undo**, which is what makes that guard
  sufficient: resumption keys on the model and run name, so re-predicting with the right model
  replaces the wrong ones without `--overwrite`. **The window remembers a model per series and
  beam**, the box following the view — a convenience and not a claim, nothing being inferred
  from which slot a path sits in, but it stops the box offering the SEM model while a FIB frame
  is on screen.
- **One prediction per label group, not per image.** The label is the unit of workflow state
  everywhere else, and a prediction exists to become a label, so a group gets one — from
  `label_groups.primary_record`, the same image the curation window shows and a training export
  keeps. Predicting the other conditions would spend inference on worse pseudo-labels
  (`SEM_8_fi_1` is one frame integration, far noisier than the `SEM_8` it shares a label with)
  and removes a hazard rather than managing one: conditions carry identical filenames into a
  shared predictions directory, so predicting all of them has each write replace the last.
- **Inference runs in a process of its own, and the window imports no torch at all.** The window
  crashed repeatedly after a completed prediction — a different place each time, always in an
  allocator or destructor on the interface thread, never once in ten command-line runs doing the
  same work on the same machine and device. The two differ in the thread inference runs on and
  the Qt event loop around it, neither fixable from inside. Nothing is lost when that child dies,
  however it dies: each prediction is written by rename and its provenance flushed before the
  next. Stopping is `SIGINT`, caught between whole images, a signal it could not handle
  stranding a half-written temporary. The child's log is forwarded so one file holds the whole
  run. *spawn*, so it inherits nothing Qt has touched.
- **Whether to offer the predict button is decided by locating torch, not importing it.** An
  import costs seconds at every window start, including for someone who never predicts, so
  `ml_stack_present` uses `find_spec`; a missing or broken stack is reported by the child, which
  has to import it anyway. **The device box does not import torch either.** Which devices exist
  differs per machine, so it opens on `auto` and `cpu`, true everywhere, and gains the machine's
  own the moment a child process loads torch, asked once per session. Editable, since
  `resolve_device` passes a named device through unexamined.
- **Predictions are written at model resolution.** A 2048x3072 image yields a 1024x1536
  prediction, which is what the segmentation pipeline's own `raw/` already holds;
  `convert-predictions` resizes to the image when it makes the label, so upsampling earlier
  would only be undone.
