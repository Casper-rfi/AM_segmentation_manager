# Decisions

Why the software is built this way, and facts not safely derivable from the code: the
code shows what happens, not what breaks if it changes. Anything one module does belongs
in that module's docstring; work not built yet belongs in
[`04_future_work.md`](04_future_work.md). Correct an entry in place rather than
appending to it.

---

## Architecture

- **No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python controllers,
  which is what makes navigation, cache eviction and state expansion testable without a display
  server. Adding a tab is not a reason to bend it.
- **No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few lines with
  no dependency, over thousands of records rather than millions.

## Filesystem and identity

- **A label directory is `<image-directory>_labels`, a sibling of the image directory.** A pure
  string function both ways with no lookup table, so one module holds the convention; `sem/` is no
  special case, its private and shared names coinciding. Conditions share a label by default, and
  giving one its own means creating the directory rather than configuring anything. Label
  directories are the only ones anything here renames — we do not own `sem` and `fib`, and
  renaming those breaks reopening the project in Autolamella. **Legacy spellings stay readable
  indefinitely** (`labels`, `labels_fib`, `labels_FIB`), a new label landing in whichever
  directory an image set already has, so normalising is cosmetic — hence
  `scripts/rename_legacy_label_dirs.py` rather than a command.
- **A renamed copy is paired back to its original by the header GUID, not by name or order.** A
  subset of some early projects was copied out, renamed and segmented, and the renaming kept no
  record of what came from where; the FEI XML tag (34683) every acquisition carries does, uniquely
  across a project. `scripts/place_renamed_labels.py` pairs on it, then checks the two files still
  decode identically before trusting the pairing, and refuses the whole batch over one doubt. A
  script for the same reason as the rename above, and it reads that tag itself: `image_metadata.py`
  reads headers for acquisition settings and has no use for an identity.
- **The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`, `sem_lowmag`
  all work with no code change, authoritative imaging parameters coming from the image headers. A
  variant may not be named `labels`, `deselected`, `corrected` or `predictions`: those collide
  with the label naming rule, so `layout.py` validates it and `inspect-layout` reports a violation
  rather than mis-indexing.
- **A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`.** `init-project`
  writes the latter rather than fabricating the former: `experiment.yaml` is Autolamella's file,
  and writing one misrepresents provenance and could collide if Autolamella ever opens that
  directory. `protocol.yaml` looks like a third marker and is not usable as one, some genuine
  projects having `experiment.yaml` without it. `init-project` **refuses a directory that gathers
  datasets rather than being one**: a marker there makes every dataset beneath it a *lamella* of
  one project — several samples under one metadata record, invisible in the output.
  `project_placement_problem` is the single guard both interfaces ask, and it tests two shapes,
  because a gathering directory only sometimes holds *marked* projects to give it away. The other
  shape is structural: one project's Adaptive Polishing stages sit at a single depth whatever its
  layout era, so stages at several depths mean several datasets. Measured across the reference
  collection, every one of its 13 projects sits at one depth.
- **Which directory is "the project" is decided by a person, not inferred.** It is a fact about
  the acquisition, and a folder gathering unmarked datasets is shaped exactly like a project whose
  lamellae hold subdirectories — so no structural test separates them in general, and the depth
  signal above misses a folder gathering datasets of a single era. The search is reliable and the
  levelling is not, so they are separate calls: `find_dataset_candidates` proposes,
  `describe_dataset_candidate` says what each proposal would become, and
  `dataset_candidates_below` descends into one proposed too high. The window's *Add projects*
  dialog is where the confirming happens; `--recursive` writes the proposals it is sure of and
  reports the rest rather than guessing. What a proposal *would become* is the useful column:
  lamella names that read like dataset names are what tells a reader the level is wrong, and no
  rule states that.
- **Identity is the path relative to the scan root, never the name** — for a project, and for a
  lamella within its project. Two different projects here have identical names, `created_at` and
  recorded paths, so keying by name silently drops one; and two projects each hold an
  `05-pet-mutt`, which an unqualified row reported as one lamella of 140 images. A lamella is
  counted as `project / lamella`.
- **The lamella and stage levels are not always directories.** Older datasets collapse them; the
  missing level is synthesised in memory with the parent standing in, and the model records that
  it is synthetic. Data on disk is never reorganised to fit the newest layout.
- **The series is a property of an image set, not of a directory.** A lamella directory holds
  reference images of both beams and alignment images at once, so a set is a *group of filenames*
  within a directory rather than the directory itself; `image_files_of` is the one place either
  shape is read back.
- **A loose-image stage does not end the stage search; a directory-shaped one does.** The
  invariant the 2026 layout rests on: a lamella holds reference images loose and its milling tasks
  below, and each task holds reference images loose and its Adaptive Polishing stages below. Were
  a lamella of reference images to end the search the way `sem/` does, every stage beneath it
  would vanish from the index silently. Nothing sits below an image directory.
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
- **The survey counts what it could not reach at all.** Every other finding describes something
  the conventions recognised; `unindexed-images` counts image files no image set claims, and
  `unindexed-dataset` counts whole directories no project claims — the two failures that hide
  best. A project holding 430 images can survey as holding 100 with nothing saying so, and a
  folder holding three unindexed datasets surveyed as holding none at all until the count moved
  onto the headline. It is also what makes the project-root overview images visible without
  indexing them as lamella data. Labels, predictions and ignored names are excluded, or annotating
  a project would report thousands.
- **A loose frame is ordered by its header's acquisition time, across its whole series.** Nothing
  in a loose filename orders it: 44% carry no timestamp, and the epochs that do appear are when
  the milling *stage* started, up to eleven minutes before the image. **mtime is worse than
  useless**, the collection having been copied. Nor can the stages be walked in order: a lamella
  directory holds reference images spanning the whole session while the milling tasks below hold
  theirs in the middle, so they interleave. `_in_acquisition_order` therefore sorts by time
  whenever every frame has one — exactly a loose series — and leaves an Adaptive Polishing
  timeline, ordered by the number in the filename, alone. Ties go to `alignment_run_order`: a run
  takes several images inside the whole second a header records, so time alone left 11 of 121 runs
  reading `final_image` before the last `new_image`.
- **Only the scanner opens an image, and only a loose one**, for the beam its filename may not
  state and the time it never does — one memoised read, costing the collection scan 1.3s to 2.8s.
  Discovery opens nothing: it asks `holds_loose_images`, which reads filenames.
- **One record per acquisition, where a lamella holds the same picture several times.**
  Autolamella duplicates the image an alignment aligns *against* in two ways and one rule cannot
  see both: it copies it into every run using it, and saves it outside the runs under a name of
  its own (`ref_alignment_ib`). So a run's `reference_image.tif` is a copy wherever the directory
  holding that run's `Alignment/` folder already holds an image of the same acquisition time, and
  **the original is kept** — the one file several runs copy, its name carrying the milling task.
  Directory and time had to agree independently on every match: **neither signal carries it
  alone**. Otherwise a copy is a filename and a time another directory has used, the filename
  being the guard there — time and size alone merged 38 groups of genuinely different pictures.
  **Nothing is hashed**: the time is already read, and with either rule's second condition it
  partitions these as contents do.
- **A fixed filename marker is matched to the shape Autolamella writes, not loosely.** Everything
  after `ref_` is a *task* name somebody chose, so a loose marker collides with one.
  `ref_alignment` is matched entire rather than as a prefix — as a prefix it reads a task called
  `Alignment Check` as an alignment image, moving that task's reference images into a series
  nothing predicts or labels. `_centring_` is anchored to the end, the plugin writing
  `<lamella>_centring_<modality>` and nothing else — as a bare substring it drops every image of a
  task called `Centring`, and a *deliberately ignored* image is not even reported as unrecognised.
- **Where a record's label and predictions live is carried, never re-derived.** Walking up from
  the image path only ever worked for an Adaptive Polishing image — one level below its stage, in
  a directory naming a modality. A reference image sits *directly* in a directory naming a
  lamella, so deriving put its files outside the lamella, where the next scan never looked. The
  scanner resolves both once per image set and the record carries them, so reading and writing
  cannot disagree; a record that does not say raises rather than guessing. Predictions follow the
  label rule above for a loose set too — exact name, no fallback — the shared
  `<modality>_predictions` being the Adaptive Polishing pipeline's own output.
- **A ticked lamella covers only the series on screen.** Curate shows one series at a time, and
  what is ticked has to be the whole of what an operation touches; filtering by modality alone
  reached every series, so predicting a lamella while looking at its Adaptive Polishing timeline
  predicted its reference and alignment images too. Each lamella is asked what it can show of the
  chosen series, as navigating to it would — `frames.resolve_series` — so one holding none of it
  contributes what it has rather than nothing.
- **`predictions/clean/` is indexed but can never become a label.** Adaptive Polishing's
  post-processing rewrites the intensity values, so they no longer index the segmentation classes:
  a label converted from one looks entirely valid and teaches a model the wrong classes. That will
  not change upstream, so it is a property of `layout.py` (`is_convertible_prediction`) rather
  than a choice offered to the caller. Still indexed deliberately: otherwise a project holding
  only post-processed predictions reports "no predictions", indistinguishable from one the
  pipeline has never reached, and those need opposite actions.

## Workflow state

- **State is declared in a file, not implied by location, and the key names the label.**
  `state.py` states the format; why the *label* is the key is that it makes an image disagreeing
  with its label unrepresentable rather than merely detectable — the conditions of one scene
  resolve to one key, so a change reaches all or none — and the key is resolved rather than
  located, so it survives annotation beginning. The cost of "no entry means selected" is that
  "reviewed and kept" cannot be told from "not yet looked at".
- **The three states are mutually exclusive, and both representations enforce it.** A mapping of
  key to state cannot express two states for one group, where a file grouping keys under
  `corrected:` / `deselected:` headings could. In memory `ImageRecord` carries one `state` field
  rather than a `selected` and a `corrected` flag for the same reason: two booleans have a fourth
  combination that means nothing, and every reader must then spell out `selected and not
  corrected` and be trusted to get it right.
- **Whether a group is corrected is asked of the record, never of a path.** Every guard protecting
  proofread work — conversion, compression, label saving — reads the declared state the scanner
  put on the record; a guard that quietly stops firing is worse than none.
- **A stage is transcribed whole or not at all, and one that has not been is refused.**
  `migrate-state` leaves a stage alone if anything in it is ambiguous — images disagreeing with
  their label, a file in two state folders — because writing the groups that were fine leaves the
  rest unlisted, and unlisted reads as *selected*; a state change refuses such a stage for the
  same reason. It is also **the only thing that reads a folder name as a state**: the scanner does
  not look inside `deselected/` or `corrected/` at all, so a nested file is *missing and reported*
  rather than present and wrong, and those names stay reserved.
- **A stale entry is reported, never repaired.** Declared state can disagree with the filesystem
  in ways location could not, and each needs a different action, so `inspect-layout` separates
  them: unmigrated folder state; nested files after migration, wanting `--move-files`; an empty
  state folder; and an entry naming a group that is not there. That last is kept rather than
  dropped — a group that vanished because a drive was not mounted must not have its curation
  deleted for it.
- **A write is read-modify-write, and the whole batch is checked before any of it is written.**
  Two windows on one collection, or a window and a command line, would otherwise discard each
  other's curation with no error. Every file the batch will touch is compared against what the
  plan read *before the first is written*, so the answer is "nothing happened" rather than "some
  of it happened"; each write then checks again for itself. That is what lets `plan_transition`
  and `apply_plan` be separate — the decisions are the previewed ones however long ago, the safety
  check fresh. Not a transaction: each stage's records come into step as that stage lands.
- **Every label write goes through `io/atomic.py`, including compression.** It restores the
  destination's own permissions before the rename, because `mkstemp` creates owner-only and OpenCV
  truncates rather than recreates — a recompressed label kept `0o600` becomes unreadable to
  everyone but its author, which on a shared archive is the same as losing it. Compression has to
  decide *after* encoding, the saving being unknowable any other way, so `write_atomically` takes
  a `commit_if` predicate and reports whether it committed — a run that skips a file says which of
  no saving, a dry run, or "somebody else rewrote it" applied, only the last of which is worth
  acting on.
- **Overwriting a label is tiered by what replacing it would cost, not by one flag.** Conversion
  sorts destinations three ways: an empty placeholder costs nothing to replace; a pseudo-label
  holding annotation is regenerable machine output, replaced once the user confirms; a corrected
  group is refused, and no argument to `plan_conversions` reaches it. Stricter than
  `compress-labels`, which *can* be confirmed for corrected labels, because compression preserves
  the pixels exactly where conversion replaces proofread work with a machine guess — re-converting
  a corrected label means setting that group back to selected first. That is also why
  **compressing is not scoped to a selection**: a per-selection button leaves uncompressed exactly
  the labels nobody happened to tick.
- **Corrected is a claim, and it is checked where it is made.** A group marked corrected with no
  label or with a placeholder breaks the claim, and a training export is *right* to drop it — what
  is wrong is doing so in silence, leaving a training set short of exactly the images you believe
  you annotated. So the window asks on either side of the gap, both times through
  `classify_labels` so the two dialogs cannot disagree. The export side asks only about
  *corrected* groups: an unlabelled selected image is the ordinary condition of an archive being
  worked through. The command line warns and proceeds where the window asks.
- **What a state change applies to is a set the user builds, at one level at a time.** Ticking a
  project clears the ticked lamellae and vice versa, and picking frames out clears both — so what
  is ticked on screen is the whole of what a change will touch. Not a precedence rule between
  levels: that leaves ticks visible which decide nothing and, once a tick outlives the project it
  was made in, ticks *invisible* which decide everything — how an operation reaches a project
  nobody was looking at. Everything reaches `transition_records` through `scoped_records()`, so
  group expansion and the corrected-data guard apply unchanged.
- **A scan the user did not ask for keeps the selection; one they did asks again.** Every
  operation that writes files is followed by a rescan, and preparing labels is two of them against
  the same set, so a rescan that emptied the selection would have *convert* ask for the choice
  *predict* was just given; pressing *Scan* starts clean. What comes back is filtered against what
  the scan found, and **frames come back by which acquisition they are, not by where they sat** —
  `LamellaFrame.key` rather than an index — an image added ahead of them shifting every index
  after it. The *view* is still restored by index: landing on a neighbouring frame is a nuisance,
  acting on one is a fault.

## Metadata

- **The Add data tab suggests, and a suggestion is never a value.** The values other projects use
  are computed in the controller and reach a file only when the user saves, so
  `build_experiment_metadata` still resolves from the two files and nothing else.
- **`project` names the wider project; the directory is the `autolamella_project`.** Different
  facts, and holding both under one word showed as a `project_project` column in one export and a
  `metadata_project` in another. The directory is what the scanner indexes and Curate lists, so it
  is qualified wherever written; the metadata field keeps the plain name. Neither is derived from
  the other, which is why Add data offers no button to copy one into the other.
- **A descriptive field is a row in `metadata.FIELDS`**, as a header value is a row in
  `image_metadata.FIELDS`: adding a field is one row rather than an edit in six modules — and
  adding a header *dialect* should be a column in that table rather than an edit to a hand-written
  extraction block per field. The places that named `lamella_type` to find the one field with a
  vocabulary now ask the field instead. A field with a `breakdown_label` gets its count, its
  `--by` selector and its Export filter row without `aggregate.py` naming it, offered in the order
  the Add data tab asks for them. `defaults` are the usual answers, offered where `choices` would
  require: two freezing methods cover almost every sample, and jet freezing still has to be
  typable.
- **A value is the type its field says, from the moment it is typed.** One `format_field_value`
  renders every one of them back, because the editor decides whether you edited a field by
  comparing what it shows against the resolved value: a deposition time shown as `30.0` where `30`
  was typed makes every project merely looked at report an edit. So a number is refused unless it
  parses, written unquoted, and carried as a float — which makes zero a recorded value, no
  deposition, and a falsy one, so `is_recorded` asks rather than testing for truth. A number that
  is not one is dropped with a warning rather than kept as text, a column of sums being worth less
  than an admitted gap.
- **A field holding several values stores them as a YAML list, and a bare scalar is one value.**
  That is what lets a field become multi-valued with no migration: every file already written
  stays valid, and `experiment.yaml` can carry one without knowing this package's conventions.
  Written back on one line, because `AM_metadata.yaml` is edited line by line to keep its comments
  — but a file edited by hand can hold a block list, and replacing only its key line left the
  items behind as the next field's value, so a replacement takes whatever is indented under it.
  `VALUE_SEPARATOR` is presentation only, for an editor, a CSV cell and the name of a group;
  storage being a list is what lets a value hold a comma.
- **`grid_type` is free text where `lamella_type` is a vocabulary.** A preparation is one of four
  things and a typo is a real error; a grid is a material, a mesh and a film in combinations no
  list would survive, and rejecting a real grid is worse than tolerating two spellings of one.
  Like `sample` it cannot be defaulted from a shared file: a grid belongs to the specimen, and one
  collection directory here holds a yeast project beside a virus one.
- **Every image header carries settings for both beams, and some values exist for only one.**
  `microscope_state.*` describes the beam that took the image; `system.*` holds both beams'
  configured defaults, so reading `system.ion.voltage` from an SEM image yields the FIB's 30 kV —
  plausible and wrong. `image_metadata.FIELDS` therefore confines beam-specific fields, including
  the fallback search, to the beam that fired, and declares which beams a value exists for at all.
  A plasma gas belongs to an ion column and an emitter type to an electron one, so they are two
  fields rather than one "beam source", which reported the FIB's Argon on SEM images and the SEM's
  `FEG` emitter as a plasma gas — and a blank `plasma_gas` on a FIB image is a real answer, a
  gallium column having no plasma.
- **A value is stored in the unit its field name states, converted where the header is read**, so
  `acceleration_voltage_kv` is kV even though both dialects record volts. A breakdown grouping a
  *measured* value rounds it, a header recording what was measured rather than what was set: a
  voltage to 0.1 kV, a pixel size and beam current to four significant figures, those two being
  derived from the magnification and so differing in the last decimal between two images of one
  field of view. Significant figures rather than decimal places, pixel sizes spanning decades. The
  conversion itself rounds to six decimals in the stored unit: a header holds 6e-05 m, which
  binary floating point cannot represent exactly, so multiplying by 1e6 gives 59.99999999999999
  um. Six decimals is finer than anything these instruments report, so only the noise goes.
- **A declared path counts as matched only if its value converts.** A key can be present and still
  state nothing: a blank, a `"None"` marker, or uninitialised memory, which is what the FEI
  dialect leaves in the stage keys it never fills — `Stage.StageX` reads 6.95e-310 while the real
  position sits in `Stage.CompustageX`. So a subnormal or non-finite number reads as absent, and
  the reader moves on to the next candidate path and then to the search fallback rather than
  stopping at a key that merely exists.
- **A stage position is comparable only within its coordinate system, so the system is a column of
  its own.** The 2024–2025 projects state specimen positions, z around −59 µm, and the 2026 ones
  RAW positions, z around +32 487 µm; the same archive holds `Specimen`, `SPECIMEN` and `RAW`,
  carried as the header spells it rather than normalised into a vocabulary this package would then
  maintain. A FEI cryo stage states its position under `Compustage*` and a tilt about x, and
  nothing that is a rotation about z, so `stage_rotation_deg` is empty there rather than filled
  from the tilt.
- **Three programs take part in an acquisition, and the header names their versions
  misleadingly.** openfibsem fills `system.info.hardware_version` from the *microscope server's*
  version and `system.info.software_version` from the *scripting client's*, so both keys say the
  opposite of what they hold; `experiment.application_version` is the application's own. They are
  on unrelated release series — xT at 28.x, AutoScript at 4.x, autolamella at 0.x — so one
  "software version" column would answer a different question per image. The fields are named for
  what they are, and the microscope's own version is the only one the FEI dialect states, under
  `System.Software` — the only one covering the pre-Autolamella datasets. That the two are one
  quantity is not an assumption: one Arctis writes 28.4.1.25560 under both keys, in both dialects.
- **The integration counts are the one field decoded rather than defaulted**: absence means "not
  enabled", where every other missing measurement stays missing.
- **A label carries no header.** Its scale comes from the image it was drawn from, rescaled by the
  ratio of their widths, a label converted from a prediction not necessarily sharing its image's
  pixel dimensions. `label_measure` returns no areas rather than assuming 1:1.
- **Most TIFFs in the collection have no acquisition header, and should not.** Predictions, labels
  and autofocus frames are not acquisitions; only indexed *images* carry one. A count of
  "unreadable headers" taken over `rglob("*.tif")` measures the wrong population.

## Counting and export

- **A record can be in more than one group, and the table says so.** A property an image has
  several of — two organisations, a sample both infected and differentiated — counts the image
  under each. The alternative, a bucket per combination, answers "how many images are of infected
  samples" with nothing, which is the question holding them apart exists for. The rows then add to
  more than the images counted, so the percentages stay a share of the *images* and a line under
  the title says why. A filter matches any of a record's values, and offers each on its own. Which
  kind a key is comes from `Breakdown.multi`, off the field table, never from what it returned;
  and `groups_of` is the one place either kind is read, so counting, cross-tabulating and
  filtering cannot reach different conclusions.
- **A filter and a breakdown are one declaration.** The same key function used two ways — group by
  it, or keep the records whose group was chosen — so the filter lists come from `BREAKDOWNS`
  rather than a second table, and cannot disagree with it about what a value *is*. Two
  consequences: "(not recorded)" is selectable, the only way to find images whose header would not
  read; and a measured property is not offered, being unable to answer before Measure is pressed.
  Modality, workflow state and the label filter keep dedicated controls, being fixed questions
  rather than data.
- **A command that reports covers every series; a command that acts stays on Adaptive Polishing.**
  The loose series more than doubled what a scan holds, and none of that may change what an
  existing command prints or writes. So `inspect-layout` and `stats` cover everything and say
  which is which, while everything that writes, exports or measures defaults to `--series
  adaptive` (`cli.DEFAULT_SERIES` and `REPORTING_SERIES`). `by_series` returns the slug rather
  than the display label, a value counted having to be a value `--series` accepts; and unlike a
  variant a series is a closed vocabulary, so `--series refrence` raises — a typo, not a request
  for an empty set.
- **A milling task is matched by `tasks.py`, and there is no separate stage axis**, `stage-name`
  already answering "which directory". The directory is tried before the filename and is the only
  thing resolving a name Autolamella spells two ways (`Milling/Fiducial/` for `Mill Fiducial`);
  the *lamella* directory is never offered, or a petname sharing a word with a task claims every
  image below it, and a name two tasks claim is dropped rather than guessed. **None is a real
  answer** — beam-shift alignment, an alignment run's own images and every Adaptive Polishing
  stage predating `Milling/<Task>/` name none, over half the images here. **Timestamps were ruled
  out**: the windows in `history` are sparse rather than tiling the session, and
  `ref_MillRough_start_eb.tif` falls inside the `SetupLamella` one.
- **Counting is split between the tab and the commands, not moved.** The Export tab counts the set
  its filters describe, choosing a training set meaning looking at what survives a filter, and
  that loop cannot run through a command that prints and exits. `stats` keeps the whole-collection
  sweep and `quantify-labels` the per-lamella table: numbers per class per lamella is not
  something anyone acts on one row at a time.
- **Opening the Export tab reads the headers and nothing else.** Half a millisecond an image, two
  seconds over the reference collection, and it is what every acquisition filter and breakdown is
  made of. Offering them greyed out until a button was found and pressed was a control that did
  nothing with no way to tell why.
- **Opening a label is asked for, never assumed.** Ten times the cost of a header — 14 seconds
  here — and it answers a different question: what is *in* the label. So the summary shows what a
  record knows about itself and says plainly that the rest needs Measure labels. **Measured over
  what the filters match**, not the collection, so widening a filter asks again for what it
  brought in. One pass answers the exportable count, the label filter and the label-state
  breakdown together; `classify_label` is the shared rule and `_label_empty` the shared cache,
  keeping two of either having meant reading every label twice.
- **What a measured number means is set beside the breakdown, not among the filters.** The crack
  threshold and the area bin resolution change how a measurement is *read*, not which images are
  in play. Changing the threshold re-measures at once rather than spending the Measure button
  again: `read_label_pixels` caches component pixel counts *without* the threshold, so a new one
  costs arithmetic rather than a second pass; `load_measurements` keeps its headers likewise.
- **A filter is added, not permanently shown.** The breakdowns cover far more properties than any
  one export is chosen by, so one list per property leaves too little of each to read and buries
  the three that matter among a dozen nobody touches. A row carries the property it is on, and a
  property already in one row is not offered in another, which stops two rows disagreeing. **A few
  properties are counted by but never filtered on** — a lamella and a fully qualified milling
  stage run to hundreds of values, an electron source does not vary here at all — named by
  `UNFILTERABLE`; they stay breakdowns, and the command line still filters on `--electron-source`.
- **A stage is filtered on by its name, and counted on by its path.** `stage` names a folder
  (`project / lamella / adaptive_polish_<timestamp>`), which finds one run again but is useless as
  a list to choose from; `stage-name` is that with the timestamp removed by
  `layout.stage_base_name`. **What names no milling stage counts under `LAMELLA_ROOT`** — a stage
  synthesised from the lamella, and an alignment run directly under one. **A run counts under the
  task it aligns for**, the directory its `Alignment/` folder sits in — Autolamella names a run
  after the lamella, the task and the time, which is otherwise a row per run. The timestamp
  pattern lives in `layout.py`: the scanner reads it to order stages and this reads it to remove
  one, and the two must agree on what one is.
- **Series is the one filter that does not open on everything.** Every other control starts wide
  and narrows towards an export; this one starts on Adaptive Polishing, a tab silently counting
  the rest disagreeing with `create-training-csv`. A dedicated control rather than a derived
  filter row, and checkboxes rather than a dropdown, an export covering two series being a real
  request. The metadata CSV names the series of every row, and `task` beside it — once more than
  one can be exported, a row not saying which it came from cannot be read.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than *corrected*: the tab is narrowed towards an
  export rather than opening on one and hiding the archive. The cost is that an image somebody
  decided against can reach a training set — only if it was labelled and deselected afterwards,
  the empty-label rule catching every other case — so the summary line counts the deselected
  images that would actually be written. Visible rather than impossible, "show me everything I
  rejected" being a real question.
- **Both modalities means side by side, not mixed together.** A training CSV holding SEM and FIB
  rows under one `image` column is not something anyone trains on, the two beams needing different
  models. So a selection holding both writes `sem_image, sem_label, fib_image, fib_label`, one row
  per acquisition; one naming a single beam writes `image, label`. A cell is blank where that beam
  has no image or no label worth pairing with one, and a row survives only if at least one beam
  has a label — the empty-label rule drops a row where there is one beam and empties a cell where
  there are two, reported either way.
- **What makes an SEM and a FIB image the same acquisition is stated once**, in
  `frames.group_by_acquisition`. They do not reliably share a stem — older projects name the file
  inside `fib/` with the same `_SEM` suffix — and **what pairs them depends on the era of the
  filename**: Adaptive Polishing images carry an acquisition number, matched within one stage
  directory, while reference and alignment filenames carry none and differ only by an `_eb`/`_ib`
  suffix, so those match on the stem with the suffix removed (`layout.acquisition_stem` beside
  `extract_image_number`). Conditions multiply out — five SEM conditions against one FIB image are
  five genuine pairs nothing here can rank — and "one image per label" collapses it. The
  train/validation split groups by the *acquisition*: split a paired row and the model is scored
  on a scene it has seen.
- **An export can also copy the files out**, state files and project metadata included, which is
  what makes the copy a collection rather than a pile — `copy_export.py` states the shape and how
  it refuses to overwrite. Two checks are not its: free space is estimated first, and a
  destination inside the folder being copied from is refused, it being about to walk into its own
  output.
- **`check-training-csv` reads the header rather than assuming it.** Any column called `image` or
  `label`, or ending in `_image` or `_label`, holds a path, so one command checks either shape
  above and files written before both. A blank cell is its own answer rather than an absent
  column: it records nothing, so it cannot have gone missing, and keeping it is what lets a pruned
  copy come out the same shape.

## Display

- **A window that has not been drawn cannot say it is working.** A startup scan run before the
  event loop gives seconds of blank grey indistinguishable from a hang, which is why it is queued
  with the activation below. Where the interface thread does have to block, the line saying so is
  `repaint`ed rather than merely set: an ordinary `setText` reaches the screen only after the work
  it announces has finished. Anything longer goes to `gui/worker.py` behind the busy dialog, which
  opens indeterminate and becomes a count as soon as the operation reports — one
  `on_progress(done, total)` contract and one `progressed` signal connected once for the whole
  window, so giving an operation a counter is a callback in its loop rather than a mechanism. Only
  the applying run counts, never the preview, which would fill a bar instantly and describe the
  wrong run. That dialog is not a `QProgressDialog`, for the re-entrancy `busy_dialog.py` sets
  out; the window owns it and detaches it before closing it.
- **One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a palette and a
  short stylesheet; nothing else picks a chrome colour. Fusion rather than the platform's style,
  which differs by platform and by Qt version. Where a rule and the palette disagree the rule
  wins, and a widget palette edited in Python is ignored — yet reads back as the value it was set
  to, so a test asserting on it passes while the screen shows something else: **anything about how
  something *renders* is asserted on the pixels**. The stylesheet also suppresses Qt's emphasis on
  a dialog's default button and a checked tool button, so anything relying on either must re-state
  it; and its 12px padding clips a one-character button blank while it reports its text correctly,
  which `COMPACT_BUTTON` exempts.
- **A chosen row is ticked, never highlighted.** Highlighting cannot say what it means: it is also
  how Qt marks the row you are looking at, it is lost whenever the list is rebuilt, and in four
  visible rows over dozens of values there is no seeing what is chosen without scrolling.
  `CheckList` is the one implementation. It reports nothing while its contents are replaced — a
  redraw rather than a gesture, and an owner rebuilding its list in response to the current row
  changing drives itself in a circle. Two further list traps: Qt draws a selected row's icon
  blended with the highlight colour, so a status chip whose colour *is* the information comes out
  wrong on exactly the row being looked at (`chip_icon()` supplies the pixmap for every mode and
  state, and any icon carrying meaning goes through it); and a `QListWidget` with a horizontal
  scrollbar scrolls instead of eliding, cutting paths off where they differ, so `ElideMiddle` plus
  `ScrollBarAlwaysOff` keeps both ends.
- **The window asks to be the one in front, and asks after the event loop has started.** Launched
  from a terminal, macOS leaves the terminal in front, and a window that is not in front receives
  no key presses and no pointer motion at all — so F1 does nothing, no tooltip appears, and the
  brush outline sits where it was last drawn. Three faults that look unrelated, one cause, and it
  clears the instant anything is clicked, which is what makes it read as occasional.
  `raise_()`/`activateWindow()` go through the same zero-delay timer the startup scan does, ahead
  of it. `changeEvent` logs activation under `-v`: it is the first thing to establish about any
  report of this shape, and not recoverable after the fact.
- **A widget that wraps takes its height from the width it actually has.** A form asks a field how
  tall it would be at one width and then hands it another, so a flow layout answering only
  `heightForWidth` lays its later rows below the bottom edge — addressable, reported, and
  invisible: a field said it held two organisations while showing one. `_FlowArea` fixes its own
  height from its real width on every resize. Its layout's minimum width is zero and its hint is
  one unwrapped row, which must differ: made equal, the minimum is a floor and the pane could
  never be narrower than the longest value anybody typed.
- **An editor that collects values commits what is in its box before anyone reads it.** A chip
  editor turns typed text into a value on Return, on a pick, and on losing focus — and `_edits`
  asks it to anyway. Held until confirmed, a value typed by someone who then reaches straight for
  Save is discarded without a word, which is what every first attempt at the field looks like.
  **Adding a widget to a layout reparents it, and reparenting hides it**, so it must be shown
  again before anything measures it: a layout of hidden widgets measures zero, and the chips were
  given no height at all. Held, addressable and saved, they were simply not on screen — which
  reads as a field that refuses a value *and* loses the one already in it.
- **F1 is an application shortcut, and the list closes only when it is the window in front.** A
  window-scoped `QShortcut` is dead while a second top-level window holds the keyboard, and a
  parented `QDialog` is one — so the key that opens the shortcut list could not close it again.
  Closing on `isVisible()` was the other half: a list buried behind the window is still visible,
  so F1 dismissed unseen the one thing that answers "what can I press?". The key now arrives from
  any window, so it is guarded to do nothing while a modal dialog is up, a list opened over one
  being neither readable nor closable until the operation ends.
- **The brush outline is redrawn from state, never from mouse movement alone.** It is drawn in
  *scene* coordinates, so everything that moves the scene under a stationary pointer — a
  middle-button pan, a fit, centring on the next speck, a frame arriving on the loader thread —
  leaves a ring around a pixel the pointer is no longer over; and every state that hides it left
  it hidden until the pointer was nudged. `_sync_brush` is the single answer to "should there be a
  ring, and where"; `scrollContentsBy` and `zoomed` catch the view moving on its own. Wheel zoom
  needs nothing, `AnchorUnderMouse` holding the scene point under the pointer.
- **The image panes have no scrollbars, their scene is bigger than the image, and zooming out
  lands through `fitInView`.** The scene carries a quarter of the image as slack on every side, so
  an edge can be dragged into the middle of the pane to be worked on — which is also why
  scrollbars go, a scene deliberately larger than its content leaving them permanently visible and
  permanently not full. `downscale_to_fit` uses `min(width_ratio, height_ratio)`, and these images
  being landscape in a portrait pane, **width binds**. Reviewing is done at fit, so the curation
  panes stop there; the proofreading canvas allows one step past it, checking that a label runs to
  the edge of the frame meaning seeing that edge. At the floor fitting wins over the wheel's
  anchor, which would otherwise leave a frame that now fits pushed into a corner, and keeps a
  two-pixel border the computed floor would crop.
- **A summary plot is held twice: pane-sized to step through, whole to look into.** A plot is ~180
  ms to read however it is asked for (a PNG decompresses whole, so OpenCV's reduced-decode flags
  buy nothing), slower than a held-down arrow key, so stepping is smooth only with every plot
  already in memory — which bounds those copies to the size of the pane. But these figures are
  4500x3000 and carry annotations existing only at that resolution. So a *fitted* plot is the
  pane-sized copy drawn at one array pixel per physical pixel, the only scale that resamples
  nothing; the whole frame is read last and takes over once magnified past what the pane copy
  holds. Read-ahead asks for no more than the cache can keep — unbounded, the read finishing last
  evicts the one that finished first — so `plot_radius` derives from the entry limit.
- **Where a copy is placed is read off the copy, and remembered across a swap.** The copy is
  reduced to a box in *physical* pixels, so drawing it at whatever device ratio the widget reports
  comes out half or twice the size wherever the two disagree; whichever of its edges touches that
  box gives the ratio back exactly. An identity transform says nothing about where the view is
  looking and the scene is deliberately larger than the picture, so drawing 1:1 has to centre as
  fitting would. Where the view *is* looking is held as a fraction of the picture rather than
  re-derived through the scroll position, which is a whole number and crept the picture towards a
  corner while stepping; it is dropped the moment the user pans.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**,
  which defaults to nearest neighbour. A pane showing something several times its own size drops
  pixels rather than averaging them: against a figure of one-pixel rules, nearest loses whole
  gridlines, so the plot view asks for smooth scaling. Acquisitions do not — zoomed *in* on a
  label, nearest is what shows where one pixel ends and the next begins.
- **Whether a run can be *stopped* is answered by what a half-done run leaves behind.** Each
  prediction is written atomically and recorded before the next begins, so a stopped run has
  produced finished files and an accurate account of them, and the next resumes from there;
  recompression is safer still, a re-encoded label holding the same pixels. Converting predictions
  must not stop — it validates the whole set of destinations before writing any, and stopping it
  leaves exactly the half-applied collection that rule exists to prevent. Cancellation is a
  `threading.Event` in `progress.py` polled by the work closure: a queued Qt signal reaches the
  worker when it next returns to its event loop, which inside a long call is never.
- **Curate and Label show one series at a time; Export ticks several.** A timeline mixing
  alignment runs with polishing frames is not something anyone proofreads, and a 2026 lamella runs
  to 21 stages across three series. Each tab has one choice, which rebuilds its frame stack — the
  position and the ticks go with it, a frame index meaning anything only within one timeline.
  **What was chosen and what a lamella can show are two things**: `resolve_series` falls back to a
  series the lamella holds without touching the choice, so the next lamella that has it opens on
  it; otherwise a lamella with no Adaptive Polishing images shows an empty timeline and, the
  control being hidden for having one entry, no way back. **The beam follows the same rule** in
  `_settle_mode`, asked of the whole stack rather than the frame on screen; the plot view is
  exempt. Both settle in `_rebuild_frames`, the one path a change of lamella and a change of
  series share. The Label tab settles on the frames it has to *review*.
- **A plot describes one acquisition or a whole stage, and the window says which.** An Adaptive
  Polishing stage writes one plot per milling cycle into `plots/`, matched to a frame by its
  acquisition number. An alignment run writes one `figure.png` at its root summarising the several
  images it took to converge, so there is nothing to match it to: every frame of the run shows it
  and carries a badge saying so, a figure describing five frames read as describing one being
  worse than no figure. `displayed_plot` prefers `frame.plot_path` over `stage_plot`.
  `AP_summary_plots.png` follows the same rule; `centring.png` is excluded, recording where the
  plugin placed its field of view rather than what the run achieved.

## Environment

- **A save refuses a label whose file changed since it was opened.** Proofreading holds a
  lamella's edits in memory until the reviewer saves, leaving minutes to hours in which something
  else can rewrite a label file — not hypothetical here, this collection living on OneDrive, which
  writes files back under a running program, and the annotation in a corrected label cannot be
  reconstructed. So the size and modification time are recorded on open and compared before
  writing, and a mismatch stops the entire batch rather than the one file: which version is wanted
  is not a question this package can answer.
- **Run the suite from the repository root, in `am_segmentation_manager` or
  `am_segmentation_manager_ml`.** Two ways to get this wrong, neither of which announces itself.
  Run `pytest` from *outside* the repository and there is no `pyproject.toml` above the working
  directory, so rootdir becomes that directory, `testpaths` never applies, and pytest recurses
  into everything below it — from `$HOME` that means importing every `.py` under a 52 GB `Data/`,
  which looks exactly like a hang; anywhere *inside* is safe. Run it in conda's `base` and the
  imports resolve to a different stack entirely, giving several hundred failures that are all one
  wrong interpreter.
- **The reference collection is at `<collection>/`**
  — 13 projects, 108 lamellae, 8,895 indexed images, 54 GB, spanning every layout era from
  pre-Autolamella to 2026-08. Read-only. Point the smoke test at it with
  `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.
- **PySide6 is the binding, and PyQt5 is no longer one.** The sixty lines of `qtpy` compatibility
  stay, costing nothing, but nothing runs the suite under PyQt5 and it had already broken there
  unnoticed on one `setBrush(Qt.NoBrush)`. It may still open; nothing promises it.
- **The window is written for a binding and a platform other than the one it is developed on.**
  Where Qt's behaviour is a style's or a platform's to decide, this package decides instead: every
  file dialog is Qt's own rather than the platform's (`common.build_path_dialog` sets
  `DontUseNativeDialog`; revealing a folder in the file manager is not a dialog and stays native),
  Up and Down step through a combo box through `StepCombo` rather than through Qt's own keys and
  `SH_ComboBox_Popup`, and `common.fixed_width_font` names real monospace families per platform.
  That last is asserted on the metrics (`"i"` and `"W"` advance equally) and never on
  `QFont.fixedPitch()`, which reads back what was requested rather than what resolved — as
  `QFontDatabase.systemFont(FixedFont)` does, returning the proportional UI font on macOS while
  reporting the family it was asked for.
- **Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side effect,
  and `test_gui_*` sorts before most other tests. That is how a locale-dependent `strptime('%p')`
  bug appears only in full runs, and only on a non-English locale. `LC_ALL=… pytest` does **not**
  test this; `AM_SEG_TEST_LOCALE=… pytest` does.
- **Never disconnect a signal from inside the handler for that signal.** A connection list mutated
  mid-emit works under PyQt5 and hangs under PySide6 — the suite stalled indefinitely on the first
  test to run a real background operation, with no traceback, because the failure path opens a
  modal box nobody is there to dismiss. Connect once and track what is pending in an attribute.
  The same shape hides in any connect-per-call pattern.
- **`QProxyStyle.styleHint` segfaults PySide6.** Overriding it to answer `SH_DialogButtonLayout` —
  to fix the button order in every dialog at once, Fusion reporting the Windows order wherever it
  runs and ending a row with *Cancel* — crashes the process while a `QMessageBox` is being built.
  The save dialog lays its own buttons out instead; any dialog that cares about the order has to.
- **An offscreen window cannot answer three questions**, and asking anyway produces tests that
  pass on nothing. It is never activated, so `hasFocus()` is False for every widget and
  `focusWidget()` records who asked. Nothing is shown, so `isVisible()` is False whatever the code
  did and `isHidden()` is the state to assert. And a modal dialog has nobody to dismiss it, so its
  `exec_` must be intercepted — per dialog class, not just `QMessageBox` — or the suite hangs
  rather than fails.
- **`matplotlib` is in the `gui` and `all` extras rather than only `plots`**: against the OpenCV
  and Qt those already install, its size is not what makes an environment large. Why figures are
  never built through `pyplot` is `plots.py`'s own docstring.
- **Every `cv2` hazard is a provider conflict**, which is why `environment.yml` pins `nodefaults`
  — a mixed conda-forge/Anaconda environment paired Anaconda's OpenCV with conda-forge's protobuf
  so that `cv2` would not load at all. **The second environment is pip-first for the same
  reason**: albumentations, which the vendored preprocessing imports, requires
  `opencv-python-headless`, a third provider no ordering of install commands avoids, so
  `environment-ml.yml` takes only python and pip from conda and the whole stack from the `[ml]`
  extra. That also lets the torch build be chosen per machine, which conda-forge cannot express
  (`AM_SEG_TORCH_INDEX_URL` selects a wheel index). **It is a superset, not a peer**: it gets the
  window too, via the `qt` extra, with no OpenCV. The curation environment is the *lighter* of the
  two, for someone who never predicts; splitting them the other way forces an environment switch
  mid-workflow.
- **Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
  `ml/environment.py` answers "is inference possible here" without importing it, as `io/image.py`
  does for the image stack. `ml/preprocessing.py` is a verbatim copy of AM_training's module,
  excluded from ruff: reformatting it destroys the only cheap answer to "is this still the
  preprocessing the model was trained with" — `diff`.

## Inference

- **A fact recoverable from the weights is never declared**, the division `ml/model.py` sets out.
  One default in the description file is load-bearing: **`decoder_attention_type="scse"` is not
  part of these models** — the training call passes it, smp's `FPN` has no such parameter and
  swallows it into `**kwargs`, the checkpoints hold no attention tensors, and a plain FPN
  strict-loads all 1015.
- **Which modality a model is for is not recorded anywhere, deliberately.** Nothing in a
  checkpoint says, and `AM_SEM_All_V01.pth` is whatever whoever trained it chose to type; reading
  that as evidence makes the software's answer depend on someone else's file-naming habit, and
  declaring it in the sidecar only moves the guess. So the selection states it, and one spanning
  both modalities is refused — a guard on something the window can show you rather than a line in
  a file you would have to open, and every report leads with the modality. **Predicting with the
  wrong model is cheap to undo**, which is what makes that guard sufficient: resumption keys on
  the model and run name, so re-predicting with the right model replaces the wrong ones without
  `--overwrite`. **The window remembers a model per series and beam** — a convenience and not a
  claim, but it stops the box offering the SEM model while a FIB frame is on screen.
- **One prediction per label group, not per image.** The label is the unit of workflow state
  everywhere else, and a prediction exists to become a label, so a group gets one — from
  `label_groups.primary_record`, the same image the curation window shows and a training export
  keeps. Predicting the other conditions would spend inference on worse pseudo-labels
  (`SEM_8_fi_1` is one frame integration, far noisier than the `SEM_8` it shares a label with) and
  removes a hazard rather than managing one: conditions carry identical filenames into a shared
  predictions directory, so predicting all of them has each write replace the last.
- **Inference runs in a process of its own**, for the reasons `predict_process.py` gives, and
  **the window imports no torch at all**. Stopping is `SIGINT`, caught between whole images, a
  signal it could not handle stranding a half-written temporary. The child's log is forwarded so
  one file holds the whole run.
- **Whether to offer the predict button is decided by locating torch, not importing it.** An
  import costs seconds at every window start, including for someone who never predicts, so
  `ml_stack_present` uses `find_spec`; a missing or broken stack is reported by the child, which
  has to import it anyway. **The device box does not import torch either.** Which devices exist
  differs per machine, so it opens on `auto` and `cpu`, true everywhere, and gains the machine's
  own the moment a child process loads torch, asked once per session. Editable, since
  `resolve_device` passes a named device through unexamined.
- **Predictions are written at model resolution.** A 2048x3072 image yields a 1024x1536
  prediction, which is what the segmentation pipeline's own `raw/` already holds;
  `convert-predictions` resizes to the image when it makes the label, so upsampling earlier would
  only be undone.
