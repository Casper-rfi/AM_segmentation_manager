# Decisions

Why the software is built this way, and facts not safely derivable from the code: the code shows
what happens, not what breaks if it changes. The window, the Qt binding and every trap in either
are [`display.md`](display.md). `CLAUDE.md` says where a fact belongs; correct an entry in place
rather than appending to it.

---

## Architecture

- **No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python controllers, which
  is what makes navigation, cache eviction and state expansion testable without a display server.
  Adding a tab is not a reason to bend it.
- **No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few lines with
  no dependency, over thousands of records rather than millions.

## Filesystem and identity

- **A label directory is `<image-directory>_labels`, a sibling of the image directory.** A pure
  string function both ways with no lookup table, so one module holds the convention; `sem/` is no
  special case, its private and shared names coinciding. Conditions share a label by default, and
  giving one its own means creating the directory rather than configuring anything. Label
  directories are the only ones anything here renames — we do not own `sem` and `fib`, and renaming
  those breaks reopening the project in Autolamella. **Legacy spellings stay readable indefinitely**
  (`labels`, `labels_fib`, `labels_FIB`), a new label landing in whichever directory an image set
  already has, so normalising is cosmetic and lives in `scripts/` rather than in a command.
- **A copy is paired back to its original by what the acquisition software recorded, never by name
  or order**, both of which survive a copy unchanged. `scripts/place_renamed_labels.py` names the
  header facts that do and reads them itself, `image_metadata.py` having no use for an identity.
- **Adaptive Polishing images can sit loose in the lamella directory.** Usually they fill a
  `<modality>` directory that names them, but the script-based prototype wrote them beside
  Autolamella's own references as `adapt_mill_img_000_electron_000.tif` — spelling the beam out
  rather than suffixing it, which is why `beam_modality` cannot read it and the rule does. So all
  three series are in `LOOSE_SERIES`, and such a set's labels go to `adaptive_sem_labels/` by the
  same naming rule the other loose series use. That name has no shared form on purpose: a shared
  `sem_labels/` would let a reference set claim the same lamella's Adaptive Polishing labels.
- **A condition that repeats its variant in the filename is renamed, not parsed around.** Stripping
  the suffix in code would change the identity an image and its label share in four places, one of
  them the state key, and would parse a variant the model keeps opaque.
  `scripts/share_variant_labels.py` renames the data instead.
- **The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`, `sem_lowmag`
  all work with no code change, authoritative imaging parameters coming from the image headers. A
  variant may not be named `labels`, `deselected`, `corrected` or `predictions`: those collide with
  the label naming rule, so `layout.py` validates it and `inspect-layout` reports a violation.
- **A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`, and which
  directory that is gets decided by a person.** `init-project` writes the latter rather than
  fabricating the former: `experiment.yaml` is Autolamella's file, and writing one misrepresents
  provenance and could collide if Autolamella ever opens that directory. `protocol.yaml` looks like
  a third marker and is not usable as one. A folder gathering unmarked datasets is shaped exactly
  like a project whose lamellae hold subdirectories, so no structural test separates them in
  general, and a marker on the gathering one makes every dataset beneath it a *lamella*.
  `project_placement_problem` is the single guard both interfaces ask, testing two shapes — a
  gathering directory only sometimes holds *marked* projects to give it away, and one project's
  stages sit at a single depth whatever its era. The search is reliable and the levelling is not, so
  they are separate calls: `find_dataset_candidates` proposes, `describe_dataset_candidate` says
  what each proposal would become, and `dataset_candidates_below` descends into one proposed too
  high. The *Add projects* dialog confirms; `--recursive` writes what it is sure of and reports the
  rest. What a proposal *would become* is the useful column: lamella names reading like dataset
  names tell a reader the level is wrong.
- **Identity is the path relative to the scan root, never the name** — for a project, and for a
  lamella within its project. Two projects can have identical names, `created_at` and recorded
  paths, so keying by name silently drops one; and two projects each holding an `05-pet-mutt` were
  reported as one lamella of 140 images. A lamella is counted as `project / lamella`.
- **The lamella and stage levels are not always directories.** Older datasets collapse them; the
  missing level is synthesised in memory with the parent standing in, and the model records that it
  is synthetic. Data on disk is never reorganised to fit the newest layout.
- **The series is a property of an image set, not of a directory.** A lamella directory holds
  reference images of both beams and alignment images at once, so a set is a *group of filenames*
  within a directory rather than the directory itself; `image_files_of` reads back either shape.
- **A loose-image stage does not end the stage search; a directory-shaped one does.** The
  invariant the 2026 layout rests on: a lamella holds reference images loose and its milling tasks
  below, each task holding its own references loose and its Adaptive Polishing stages below. Were
  a lamella of reference images to end the search the way `sem/` does, every stage beneath it would
  vanish silently. Nothing sits below an image directory.
- **A loose set's label directory is resolved exactly, with no fallback chain**, unlike
  `resolve_labels_directory`, which falls back to the shared `<modality>_labels`. A dataset
  collapsing the stage level keeps its Adaptive Polishing images and its `labels/` in the lamella
  directory, so a reference set falling back would pair every reference image with a label of a
  different scene: right dimensions, plausible shapes, wrong data.
- **A series nobody has started labelling is not reported as unlabelled.** `no-labels-yet` fires
  per image set, and the loose series add hundreds of sets nothing has annotated — enough to bury
  every finding worth acting on. `layout.LABELS_EXPECTED` names the series that get the report; a
  set whose label directory *does* exist is reported on normally whatever its series.
  `unexpected-stage-name` is narrowed the same way, to a stage found by holding image *directories*.
- **The survey counts what it could not reach at all.** Every other finding describes something
  the conventions recognised; `unindexed-images` counts image files no image set claims, and
  `unindexed-dataset` counts whole directories no project claims — the two failures that hide best.
  A project holding 430 images can survey as holding 100 with nothing saying so. Labels,
  predictions and ignored names are excluded, or annotating a project would report thousands.
  Turning that count into something viewable is [`04_future_work.md`](04_future_work.md) §6.
- **A loose frame is ordered by its header's acquisition time, across its whole series.** Nothing in
  a loose filename orders it: many carry no timestamp, and the epochs that do appear are when the
  milling *stage* started, minutes before the image. **mtime is worse than useless**, the collection
  having been copied. Nor can the stages be walked in order: a lamella holds reference images
  spanning the whole session while the milling tasks below hold theirs in the middle.
  `_in_acquisition_order` sorts by time whenever every frame has one — exactly a loose series —
  leaving an Adaptive Polishing timeline, ordered by filename, alone. Ties go to
  `alignment_run_order`: a run takes several images inside the whole second a header records.
- **Only the scanner opens an image, and only a loose one**, for the beam its filename may not
  state and the time it never does — one memoised read, costing the collection scan a second or
  two. Discovery opens nothing: it asks `holds_loose_images`, which reads filenames.
- **One record per acquisition, where a lamella holds the same picture several times.** Autolamella
  duplicates the image an alignment aligns *against* in two ways and one rule cannot see both: it
  copies it into every run using it, and saves it outside the runs under a name of its own
  (`ref_alignment_ib`). So a run's `reference_image.tif` is a copy wherever the directory holding
  that run's `Alignment/` folder already holds an image of the same acquisition time, and **the
  original is kept**. Directory and time must agree independently on every match — **neither signal
  carries it alone**, time and size having merged genuinely different pictures — and **nothing is
  hashed**, the time being already read and partitioning these as contents do. A third pass runs
  over the whole *project*, the prototype having duplicated across lamellae; it keeps the
  filename-and-time key rather than loosening to time alone, which would also merge a misfiled
  reference whose indexing keeps the misfiling visible. **It never empties a lamella**, a duplicate
  that can be seen beating a directory dropped out of the index and the survey at once.
- **A fixed filename marker is matched to the shape Autolamella writes, not loosely.** Everything
  after `ref_` is a *task* name somebody chose, so a loose marker collides with one.
  `ref_alignment` is matched entire rather than as a prefix — as a prefix it reads a task called
  `Alignment Check` as an alignment image, moving that task's references into a series nothing
  predicts or labels. `_centring_` is anchored to the end for the same reason: as a bare substring
  it silently drops every image of a task called `Centring`, ignored images not being reported.
- **Where a record's label and predictions live is carried, never re-derived.** Walking up from the
  image path only works for an Adaptive Polishing image, one level below its stage in a directory
  naming a modality; a reference image sits *directly* in a directory naming a lamella, so deriving
  put its files outside the lamella where the next scan never looked. The scanner resolves both once
  per image set and the record carries them, so reading and writing cannot disagree; a record that
  does not say raises rather than guessing.
- **A ticked lamella covers only the series on screen.** What is ticked has to be the whole of
  what an operation touches; filtering by modality alone reached every series, so predicting a
  lamella while looking at its Adaptive Polishing timeline predicted its reference and alignment
  images too. Each lamella is asked what it can show of the chosen series, as navigating to it
  would — `frames.resolve_series` — so one holding none of it contributes what it has.
- **`predictions/clean/` is indexed but can never become a label.** Adaptive Polishing's
  post-processing rewrites the intensity values, so they no longer index the segmentation classes:
  a label converted from one looks entirely valid and teaches a model the wrong classes. That will
  not change upstream, so it is a property of `layout.py` (`is_convertible_prediction`) rather
  than a choice offered to the caller. Still indexed deliberately: otherwise a project holding
  only post-processed predictions reports "no predictions", indistinguishable from one the
  pipeline never reached, and those need opposite actions.

## Workflow state

- **State is declared in a file, not implied by location, and the key names the label.**
  `state.py` states the format; why the *label* is the key is that it makes an image disagreeing
  with its label unrepresentable rather than merely detectable — the conditions of one scene
  resolve to one key, so a change reaches all or none — and it is resolved rather than located, so
  it survives annotation beginning. The cost of "no entry means selected" is that "reviewed and
  kept" cannot be told from "not yet looked at".
- **The three states are mutually exclusive, and both representations enforce it.** A mapping of
  key to state cannot express two states for one group, where a file grouping keys under
  `corrected:` / `deselected:` headings could. `ImageRecord` carries one `state` field rather than
  a `selected` and a `corrected` flag for the same reason: two booleans have a fourth combination
  that means nothing, and every reader must then be trusted to spell out `selected and not
  corrected`.
- **Whether a group is corrected is asked of the record, never of a path.** Every guard protecting
  proofread work — conversion, compression, label saving — reads the declared state the scanner put
  on the record; a guard that quietly stops firing is worse than none.
- **A stage is transcribed whole or not at all.** `migrate-state` leaves a stage alone if anything
  in it is ambiguous — images disagreeing with their label, a file in two state folders — writing
  the groups that were fine leaving the rest unlisted, and unlisted reads as *selected*. It is also
  **the only thing that reads a folder name as a state**: the scanner does not look inside
  `deselected/` or `corrected/`, so a nested file is *missing and reported* rather than present and
  wrong, and those names stay reserved.
- **A stale entry is reported, never repaired.** Declared state can disagree with the filesystem
  in ways location could not, and each needs a different action, so `inspect-layout` separates
  them: unmigrated folder state; nested files after migration, wanting `--move-files`; an empty
  state folder; and an entry naming a group that is not there. That last is kept rather than
  dropped — a group that vanished because a drive was not mounted must not lose its curation.
- **A write is read-modify-write, and the whole batch is checked before any of it is written** —
  the rule every batch here follows, state changes, conversion and folder-wide metadata alike. Two
  windows on one collection, or a window and a command line, would otherwise discard each other's
  curation with no error. Every file the batch will touch is compared against what the plan read
  *before the first is written*, so the answer is "nothing happened" rather than "some of it
  happened"; each write then checks again for itself. That is what lets `plan_transition` and
  `apply_plan` be separate — previewed decisions, fresh safety check.
- **Every label write goes through `io/atomic.py`, including compression.** It restores the
  destination's own permissions before the rename, because `mkstemp` creates owner-only and OpenCV
  truncates rather than recreates — a recompressed label kept `0o600` becomes unreadable to everyone
  but its author, which on a shared archive is the same as losing it. Compression decides *after*
  encoding, so `write_atomically` takes a `commit_if` predicate and reports whether it committed: a
  skipped file says which of no saving, a dry run, or "somebody else rewrote it" applied.
- **Overwriting a label is tiered by what replacing it would cost, not by one flag.** Conversion
  sorts destinations three ways: an empty placeholder costs nothing to replace; a pseudo-label
  holding annotation is regenerable machine output, replaced once the user confirms; a corrected
  group is refused, and no argument to `plan_conversions` reaches it. Stricter than
  `compress-labels`, which *can* be confirmed for corrected labels, compression preserving the
  pixels where conversion replaces proofread work with a machine guess. That is also why
  **compressing is not scoped to a selection**: a per-selection button leaves uncompressed exactly
  the labels nobody happened to tick.
- **Corrected is a claim, and it is checked where it is made.** A group marked corrected with no
  label or with a placeholder breaks the claim, and a training export is *right* to drop it — what
  is wrong is doing so in silence, leaving a training set short of exactly the images you believe
  you annotated. So the window asks on either side of the gap, both times through
  `classify_labels` so the two dialogs cannot disagree, and only about *corrected* groups: an
  unlabelled selected image is the ordinary condition of an archive being worked through. The
  command line warns and proceeds where the window asks.
- **What a state change applies to is a set the user builds, at one level at a time.** Ticking a
  project clears the ticked lamellae and vice versa, and picking frames out clears both — so what
  is ticked on screen is the whole of what a change will touch. Not a precedence rule between
  levels: that leaves ticks visible which decide nothing and, once a tick outlives the project it
  was made in, ticks *invisible* which decide everything. Everything reaches `transition_records`
  through `scoped_records()`, so group expansion and the corrected-data guard apply unchanged.
- **A scan the user did not ask for keeps the selection; one they did asks again.** An operation
  that writes files is followed by a rescan, and preparing labels is two of them against the same
  set, so a rescan that emptied the selection would have *convert* ask for the choice *predict* was
  just given; pressing *Scan* starts clean. What comes back is filtered against what the scan
  found, and **frames come back by which acquisition they are, not by where they sat** —
  `LamellaFrame.key` rather than an index. The *view* is still restored by index: landing on a
  neighbouring frame is a nuisance, acting on one is a fault.
- **Saving labels is the write that does not rescan**, being the one whose whole effect is already
  in memory when it finishes: `_attach_created_labels` names a label that did not exist on the
  records proofread through it, and declaring a group corrected writes a state file and brings
  those records with it, moving nothing. It is also the action taken most often — one lamella at a
  time, all day — and reading a collection again to learn what is already known cost twenty seconds
  of each one and discarded the prepared frames of the lamella being worked on. So it recounts from
  the records instead, and `LabelController.save` settles the frames worth visiting itself.

## Metadata

- **The Add data tab suggests, and a suggestion is never a value.** The values other projects use
  are computed in the controller and reach a file only when the user saves, so
  `build_experiment_metadata` still resolves from the two files and nothing else. **A completer that
  matches anywhere must complete into a popup, and the popup must be taught Tab** — two halves of
  one fault, both putting a value nobody typed into metadata that is exported: inline, an editable
  `QComboBox` appends the tail of a match that was never a prefix (`a` against `UltrAuFoil Au R2/2
  200 mesh` wrote `altrAuFoil …`); untaught, Tab moves the focus and commits the half-typed letter.
  `common.match_anywhere` is the only place any of it is set, and where the key has to be caught is
  a Qt matter in [`display.md`](display.md) § The binding. Matching anywhere is worth both:
  `Quantifoil` and `UltrAuFoil` differ in their middles.
- **`project` names the wider project; the directory is the `autolamella_project`.** Different
  facts, and holding both under one word showed as a `project_project` column in one export and a
  `metadata_project` in another. The directory is what the scanner indexes and Curate lists, so it
  is qualified wherever written; the metadata field keeps the plain name. Neither is derived from
  the other, which is why Add data offers no button to copy one across.
- **A descriptive field is a row in `metadata.FIELDS`**, as a header value is a row in
  `image_metadata.FIELDS`: adding a field is one row rather than an edit in six modules, and
  adding a header *dialect* is a column rather than a hand-written extraction block per field.
  Nothing names a particular field to find out what it is; a field with a `breakdown_label` gets
  its count, its `--by` selector and its Export filter row without `aggregate.py` naming it.
  `defaults` are the usual answers, offered where `choices` would require: two freezing methods
  cover almost every sample, and jet freezing still has to be typable.
- **A value is the type its field says, from the moment it is typed.** One `format_field_value`
  renders every one of them back, because the editor decides whether you edited a field by comparing
  what it shows against the resolved value: a deposition time shown as `30.0` where `30` was typed
  makes every project merely looked at report an edit. So a number is refused unless it parses,
  written unquoted, and carried as a float — which makes zero a recorded value and a falsy one, so
  `is_recorded` asks rather than testing for truth. A number that is not one is dropped with a
  warning, a column of sums being worth less than an admitted gap.
- **A field holding several values stores them as a YAML list, and a bare scalar is one value.**
  That is what lets a field become multi-valued with no migration: every file already written stays
  valid, and `experiment.yaml` can carry one without knowing this package's conventions. Written
  back on one line, `AM_metadata.yaml` being edited line by line to keep its comments — but a
  hand-edited file can hold a block list, so a replacement takes whatever is indented under the key.
  `VALUE_SEPARATOR` is presentation only; storage being a list is what lets a value hold a comma.
- **`grid_type` is free text where `lamella_type` is a vocabulary.** A preparation is one of four
  things and a typo is a real error; a grid is a material, a mesh and a film in combinations no list
  would survive, and rejecting a real grid is worse than tolerating two spellings of one. Like
  `sample` it cannot be defaulted from a shared file: a grid belongs to the specimen, and one
  collection directory can hold a yeast project beside a virus one.
- **Every image header carries settings for both beams, and some values exist for only one.**
  `microscope_state.*` describes the beam that took the image; `system.*` holds both beams'
  configured defaults, so reading `system.ion.voltage` from an SEM image yields the FIB's 30 kV —
  plausible and wrong. `image_metadata.FIELDS` therefore confines beam-specific fields, including
  the fallback search, to the beam that fired, and declares which beams a value exists for at all.
  A plasma gas belongs to an ion column and an emitter type to an electron one, so they are two
  fields rather than one "beam source" — and a blank `plasma_gas` on a FIB image is a real answer,
  a gallium column having no plasma.
- **A value is stored in the unit its field name states, converted where the header is read**, so
  `acceleration_voltage_kv` is kV even though both dialects record volts. A breakdown grouping a
  *measured* value rounds it, a header recording what was measured rather than what was set: a
  voltage to 0.1 kV, a pixel size and beam current to four significant figures — the latter two
  derived from the magnification and so differing in the last decimal between two images of one
  field of view. Significant figures rather than decimal places, pixel sizes spanning decades. The
  conversion rounds to six decimals in the stored unit, binary floating point turning 6e-05 m into
  59.99999999999999 um.
- **A declared path counts as matched only if its value converts.** A key can be present and still
  state nothing: a blank, a `"None"` marker, or uninitialised memory, which is what the FEI dialect
  leaves in the stage keys it never fills — `Stage.StageX` reads 6.95e-310 while the real position
  sits in `Stage.CompustageX`. So a subnormal or non-finite number reads as absent and the reader
  moves on, rather than stopping at a key that merely exists.
- **A stage position is comparable only within its coordinate system, so the system is a column of
  its own.** One archive holds specimen positions with z around −59 µm beside RAW ones around
  +32 487 µm, and spells the system `Specimen`, `SPECIMEN` and `RAW` — carried as the header
  spells it rather than normalised into a vocabulary this package would then maintain. A FEI cryo
  stage states a tilt about x and nothing that is a rotation about z, so `stage_rotation_deg` is
  empty there rather than filled from the tilt.
- **Three programs take part in an acquisition, and the header names their versions misleadingly.**
  openfibsem fills `system.info.hardware_version` from the *microscope server's* version and
  `system.info.software_version` from the *scripting client's*, so both keys say the opposite of
  what they hold; `experiment.application_version` is the application's own. They are on unrelated
  release series — xT at 28.x, AutoScript at 4.x, autolamella at 0.x — so one "software version"
  column would answer a different question per image. The microscope's own version is the only one
  the FEI dialect states, under `System.Software`, and so the only one covering the pre-Autolamella
  datasets; that it is the same quantity is measured, not assumed.
- **A label carries no header.** Its scale comes from the image it was drawn from, rescaled by the
  ratio of their widths, a label converted from a prediction not necessarily sharing its image's
  dimensions. `label_measure` returns no areas rather than assuming 1:1.
- **Most TIFFs in a collection have no acquisition header, and should not.** Predictions, labels
  and autofocus frames are not acquisitions; only indexed *images* carry one. A count of
  "unreadable headers" over `rglob("*.tif")` measures the wrong population.

## Counting and export

- **A record can be in more than one group, and the table says so.** A property an image has
  several of — two organisations, a sample both infected and differentiated — counts the image under
  each. The alternative, a bucket per combination, answers "how many images are of infected samples"
  with nothing, which is the question holding them apart exists for. The rows then add to more than
  the images counted, so the percentages stay a share of the *images* and a line says why. Which
  kind a key is comes from `Breakdown.multi`, off the field table, never from what it returned;
  `groups_of` is the one place either kind is read.
- **A filter and a breakdown are one declaration.** The same key function used two ways — group by
  it, or keep the records whose group was chosen — so the filter lists come from `BREAKDOWNS`
  rather than a second table, and cannot disagree with it about what a value *is*. Hence
  "(not recorded)" is selectable, the only way to find images whose header would not read, and a
  measured property is not offered, being unable to answer before Measure is pressed. Modality,
  workflow state and the label filter keep dedicated controls, being fixed questions not data.
- **A command that reports covers every series; a command that acts stays on Adaptive Polishing.**
  The loose series more than doubled what a scan holds, and none of that may change what an
  existing command prints or writes. So `inspect-layout` and `stats` cover everything and say which
  is which, while everything that writes, exports or measures defaults to `--series adaptive`
  (`cli.DEFAULT_SERIES` and `REPORTING_SERIES`). `by_series` returns the slug rather than the
  display label, a value counted having to be one `--series` accepts; and unlike a variant a series
  is a closed vocabulary, so `--series refrence` raises — a typo, not a request for nothing.
- **A milling task is matched by `tasks.py`, and there is no separate stage axis**, `stage-name`
  already answering "which directory". The directory is tried before the filename and is the only
  thing resolving a name Autolamella spells two ways (`Milling/Fiducial/` for `Mill Fiducial`); the
  *lamella* directory is never offered, or a petname sharing a word with a task claims every image
  below it, and a name two tasks claim is dropped rather than guessed. **None is a real answer** for
  over half these images. **Timestamps were ruled out**, the windows in `history` being sparse
  rather than tiling the session: `ref_MillRough_start_eb.tif` falls inside the `SetupLamella` one.
- **Counting is split between the tab and the commands, not moved.** The Export tab counts the set
  its filters describe, choosing a training set meaning looking at what survives a filter, and that
  loop cannot run through a command that prints and exits. `stats` keeps the whole-collection sweep
  and `quantify-labels` the per-lamella table, which nobody acts on one row at a time.
- **Opening the Export tab reads the headers and nothing else.** Under a millisecond an image,
  and it is what every acquisition filter and breakdown is made of. Offering them greyed out until
  a button was found and pressed was a control that did nothing with no way to tell why.
- **Opening a label is asked for, never assumed.** An order of magnitude dearer than a header, and
  it answers a different question: what is *in* the label. So the summary shows what a record
  knows about itself and says plainly that the rest needs Measure labels. **Measured over what the
  filters match**, not the collection, so widening a filter asks again for what it brought in. One
  pass answers the exportable count, the label filter and the label-state breakdown together;
  `classify_label` is the shared rule and `_label_empty` the shared cache, keeping two of either
  having meant reading every label twice.
- **What a measured number means is set beside the breakdown, not among the filters.** The crack
  threshold and the area bin resolution change how a measurement is *read*, not which images are in
  play. Changing the threshold re-measures at once rather than spending the Measure button again:
  `read_label_pixels` caches component pixel counts *without* the threshold, so a new one costs
  arithmetic rather than a second pass.
- **A filter is added, not permanently shown.** The breakdowns cover far more properties than any
  one export is chosen by, so one list per property leaves too little of each to read and buries
  the three that matter among a dozen nobody touches. A row carries the property it is on, and a
  property already in one row is not offered in another, which stops two rows disagreeing. **A few
  properties are counted by but never filtered on** — those running to hundreds of values, and
  those that barely vary — named by `UNFILTERABLE`; they stay breakdowns, and the command line
  still filters on them.
- **A stage is filtered on by its name, and counted on by its path.** `stage` names a folder
  (`project / lamella / adaptive_polish_<timestamp>`), which finds one run again but is useless to
  choose from; `stage-name` is that with the timestamp removed by `layout.stage_base_name`. **What
  names no milling stage counts under `LAMELLA_ROOT`**, and **a run counts under the task it aligns
  for**, the directory its `Alignment/` folder sits in — Autolamella naming a run after the lamella,
  task and time, which is otherwise a row per run. The timestamp pattern lives in `layout.py`, the
  scanner reading it to order stages and this to remove one.
- **Series is the one filter that does not open on everything.** Every other control starts wide
  and narrows towards an export; this one starts on Adaptive Polishing, a tab silently counting
  the rest disagreeing with `create-training-csv`. A dedicated control rather than a derived
  filter row, and checkboxes rather than a dropdown, an export covering two series being a real
  request. The metadata CSV names the series of every row, and `task` beside it: a row not saying
  which it came from cannot be read.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than *corrected*: the tab is narrowed towards an
  export rather than opening on one and hiding the archive. The cost is that an image somebody
  decided against can reach a training set — only if it was labelled and deselected afterwards —
  so the summary line counts the deselected images that would actually be written. Visible rather
  than impossible, "show me everything I rejected" being a real question.
- **Both modalities means side by side, not mixed together.** A training CSV holding SEM and FIB
  rows under one `image` column is not something anyone trains on, the two beams needing different
  models. So a selection holding both writes `sem_image, sem_label, fib_image, fib_label`, one row
  per acquisition; one naming a single beam writes `image, label`. A cell is blank where that beam
  has no image or no label worth pairing with one, and a row survives only if at least one beam has
  a label — the empty-label rule drops a one-beam row and empties a cell where there are two.
- **What makes an SEM and a FIB image the same acquisition is stated once**, in
  `frames.group_by_acquisition`. They do not reliably share a stem — older projects name the file
  inside `fib/` with the same `_SEM` suffix — and **what pairs them depends on the era of the
  filename**: Adaptive Polishing images carry an acquisition number, matched within one stage
  directory, while reference and alignment filenames carry none and differ only by an `_eb`/`_ib`
  suffix, so those match on the stem with the suffix removed (`layout.acquisition_stem`). Conditions
  multiply out into genuine pairs nothing here can rank, and "one image per label" collapses it. The
  train/validation split groups by the *acquisition*: split a paired row and the model is scored on
  a scene it has seen.
- **An export can also copy the files out**, state files and project metadata included, which is
  what makes the copy a collection rather than a pile — `copy_export.py` states the shape and how
  it refuses to overwrite. Two checks are not its: free space is estimated first, and a
  destination inside the folder being copied from is refused, it being about to walk into its own
  output.
- **`check-training-csv` reads the header rather than assuming it.** Any column called `image` or
  `label`, or ending in `_image` or `_label`, holds a path, so one command checks either shape
  above and files written before both. A blank cell is its own answer rather than an absent
  column: it records nothing, so it cannot have gone missing.

## Environment

- **A save refuses a label whose file changed since it was opened.** Proofreading holds a
  lamella's edits in memory until the reviewer saves, leaving minutes to hours in which something
  else can rewrite a label file — not hypothetical, this collection living on OneDrive, which
  writes files back under a running program. So the size and modification time are recorded on
  open and compared before writing, and a mismatch stops the entire batch rather than the one
  file: which version is wanted is not a question this package can answer.
- **Run the suite from the repository root, in `am_segmentation_manager` or
  `am_segmentation_manager_ml`.** Two ways to get this wrong, neither of which announces itself.
  Run `pytest` from *outside* the repository and there is no `pyproject.toml` above the working
  directory, so rootdir becomes that directory, `testpaths` never applies, and pytest recurses into
  everything below it — from `$HOME` that means importing every `.py` under the data archives,
  which looks exactly like a hang. Run it in conda's `base` and the imports resolve to a different
  stack, giving several hundred failures that are all one wrong interpreter.
- **The reference collection is the live archive, and there is no disposable copy of it.**
  `<collection>/` — a couple of hundred gigabytes covering
  every layout era from pre-Autolamella to 2026-05, and still being added to, so any count of it
  written down here is wrong by the time it is read. Point the smoke test at it with
  `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.
  **Never write to it.** Not a test, not a script, not an exploratory one-liner: nothing may create,
  move, overwrite or delete anything inside it without permission asked for and given for that
  specific change. It holds thousands of proofread labels that cannot be reconstructed, and the copy
  that used to absorb mistakes is gone — which is why this entry names real data at all. Read it,
  and copy *out* of it; `<collection>/` holds the original
  labels those were placed from, read-only on the same terms. `test_reference_collection.py`
  snapshots every file before and after; nothing enforces this for anything else you run.
- **`matplotlib` is in the `gui` and `all` extras rather than only `plots`**: against the OpenCV
  and Qt those already install, its size is not what makes an environment large. Why figures are
  never built through `pyplot` is `plots.py`'s own docstring.
- **Every `cv2` hazard is a provider conflict**, which is why `environment.yml` pins `nodefaults` —
  a mixed conda-forge/Anaconda environment paired Anaconda's OpenCV with conda-forge's protobuf so
  that `cv2` would not load at all. **The second environment is pip-first for the same reason**:
  albumentations, which the vendored preprocessing imports, requires `opencv-python-headless`, a
  third provider no ordering of install commands avoids, so `environment-ml.yml` takes only python
  and pip from conda and the whole stack from the `[ml]` extra. That also lets the torch build be
  chosen per machine (`AM_SEG_TORCH_INDEX_URL`). **It is a superset, not a peer**: the curation
  environment is the *lighter* of the two, and splitting them the other way forces an environment
  switch mid-workflow.
- **Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
  `ml/environment.py` answers "is inference possible here" without importing it, as `io/image.py`
  does for the image stack. `ml/preprocessing.py` is a verbatim copy of AM_training's module,
  excluded from ruff: reformatting it destroys the only cheap answer to "is this still the
  preprocessing the model was trained with" — `diff`.
## Inference

- **A fact recoverable from the weights is never declared**, the division `ml/model.py` sets out.
  One default in the description file is load-bearing: **`decoder_attention_type="scse"` is not
  part of these models** — the training call passes it, smp's `FPN` swallows it into `**kwargs`,
  the checkpoints hold no attention tensors, and a plain FPN strict-loads every one.
- **Which modality a model is for is not recorded anywhere, deliberately.** Nothing in a checkpoint
  says, and a filename is whatever whoever trained it chose to type; reading that as evidence makes
  the software's answer depend on someone else's naming habit, and declaring it in the sidecar only
  moves the guess. So the selection states it, and one spanning both modalities is refused.
  **Predicting with the wrong model is cheap to undo**, which is what makes that guard sufficient:
  resumption keys on the model and run name, so re-predicting with the right model replaces the
  wrong ones without `--overwrite`. **The window remembers a model per series and beam** — a
  convenience and not a claim, but it stops the box offering the SEM model on a FIB frame.
- **One prediction per label group, not per image.** The label is the unit of workflow state
  everywhere else, and a prediction exists to become a label, so a group gets one — from
  `label_groups.primary_record`, the same image the curation window shows and a training export
  keeps. Predicting the other conditions would spend inference on noisier pseudo-labels and removes
  a hazard rather than managing one: conditions carry identical filenames into a shared predictions
  directory, so predicting all of them has each write replace the last.
- **Inference runs in a process of its own**, for the reasons `predict_process.py` gives, and **the
  window imports no torch at all**. Stopping is `SIGINT`, caught between whole images, a signal it
  could not handle stranding a half-written temporary. The child's log is forwarded so one file
  holds the whole run.
- **Whether to offer the predict button is decided by locating torch, not importing it.** An
  import costs seconds at every window start, including for someone who never predicts, so
  `ml_stack_present` uses `find_spec`; a missing or broken stack is reported by the child, which has
  to import it anyway. **The device box does not import torch either**, opening on `auto` and `cpu`,
  true everywhere, and gaining the machine's own the moment a child process loads torch. Editable,
  since `resolve_device` passes a named device through unexamined.
- **Predictions are written at model resolution**, which is what the segmentation pipeline's own
  `raw/` already holds; `convert-predictions` resizes to the image when it makes the label, so
  upsampling earlier would only be undone.
