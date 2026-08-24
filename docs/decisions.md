# Decisions

Why the software is built this way, and facts that are not safely derivable from the
code: the code shows what happens, not what breaks if it changes. Work that is *not*
built yet belongs in [`04_future_work.md`](04_future_work.md). If something below
becomes wrong, correct it in place rather than appending a correction.

---

## Architecture

- **No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python
  controllers, which is what makes navigation, cache eviction and state expansion
  testable without a display server. Adding a tab is not a reason to bend it.
- **No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few
  lines with no dependency, over thousands of records rather than millions.

## Filesystem and identity

- **A label directory is `<image-directory>_labels`, a sibling of the image directory.**
  This makes image-dir → label-dir a pure string function both ways with no lookup table,
  which is what lets one module hold the convention. Conditions share a label by default;
  giving one its own means creating the directory, not configuring anything. `sem/` is no
  special case: its private and shared names coincide, so one rule covers the plain layout
  and the five-condition one alike.
- **Image directories keep their Autolamella names.** We do not own `sem` and `fib`;
  renaming them would break reopening the project in Autolamella. Label directories are
  ours, so they are the only ones anything here renames.
- **Legacy label-directory spellings stay readable indefinitely.** `labels`, `labels_fib`
  and `labels_FIB` are all indexed, and a new label lands in whichever directory an image
  set already has, so no project ends up holding two namings. That leaves cosmetic
  tidiness as the only reason to normalise, which is why it is
  `scripts/rename_legacy_label_dirs.py` rather than a command.
- **The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`,
  `sem_lowmag` all work with no code change, because authoritative imaging parameters come
  from the image headers. A variant may not be named `labels`, `deselected`, `corrected`
  or `predictions`: those collide with the label naming rule, so `layout.py` validates it
  and `inspect-layout` reports a violation rather than mis-indexing.
- **A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`.**
  `init-project` writes the latter rather than fabricating the former, because
  `experiment.yaml` is Autolamella's file: writing one misrepresents provenance and could
  collide if Autolamella ever opens that directory. `protocol.yaml` looks like a third
  marker and is not usable as one — some genuine Autolamella projects have
  `experiment.yaml` without it.
- **`init-project` refuses a directory that holds projects of its own.** That is a
  collection, and a marker there makes every unmarked dataset beneath it a *lamella* of one
  project: several samples under one metadata record, invisible in the output.
  `--recursive` is one flag away and does the right thing.
- **Identity is the path relative to the scan root, never the name** — for a project, and
  for a lamella within its project. The reference collection holds two different projects
  whose name, `created_at` and recorded path in `experiment.yaml` are all identical, so
  keying by name silently drops one; and two projects each hold an `05-pet-mutt`, which an
  unqualified row reported as one lamella of 140 images. A lamella is therefore counted as
  `project / lamella` and a stage one level deeper. Stage names are worse still, being
  either shared by every lamella (`adaptive_polish`) or a bare timestamp.
- **The lamella and stage levels are not always directories.** Older datasets collapse
  them; the missing level is synthesised in memory with the parent standing in, and the
  model records that it is synthetic. Data on disk is never reorganised to fit the newest
  layout.
- **The series is a property of an image set, not of a directory.** Autolamella writes far
  more than the Adaptive Polishing images: a lamella directory holds reference images of
  both beams and alignment images at once. So a set is a *group of filenames* within a
  directory rather than the directory itself, which is what `ImageSet.image_names` is, and
  `image_files_of` is the one place either shape is read back. Adaptive sets keep the old
  shape — whole directory, no names — so nothing that indexes them changed.
- **A loose-image stage does not end the stage search; a directory-shaped one does.** The
  invariant the 2026 layout rests on. A lamella holds reference images loose and its
  milling tasks below; each task holds reference images loose and its Adaptive Polishing
  stages below. Were a lamella full of reference images to end the search the way `sem/`
  does, every Adaptive Polishing stage beneath it would vanish from the index — 127 stages
  in the reference collection, silently. Nothing sits below an image directory, so
  stopping there stays right. A directory holding both shapes is returned once and not
  descended into, which loses nothing: the scanner builds both kinds of set for every
  stage, so which branch found it does not change what is indexed.
- **A loose set's label directory is resolved exactly, with no fallback chain.** Unlike
  `resolve_labels_directory`, which falls back to the shared `<modality>_labels`. A dataset
  that collapses the stage level keeps its Adaptive Polishing images and its `labels/` in
  the lamella directory, so a reference set falling back would pair every reference image
  with a label of an entirely different scene: right dimensions, plausible shapes, wrong
  data. The name is `<series>_<modality>_labels`, built from `loose_image_set_name` through
  the same `labels_directory_name` every other label directory goes through.
- **A series nobody has started labelling is not reported as unlabelled.** `no-labels-yet`
  fires once per image set, and the reference and alignment images add 460 sets holding
  5,237 images that nothing has annotated — enough to bury every finding worth acting on.
  `layout.LABELS_EXPECTED` names the series that get the report, and a set whose label
  directory *does* exist is reported on normally whatever its series, so starting to label
  one needs no change there. `unexpected-stage-name` is narrowed the same way: it checks the
  rule "a directory of image directories is a stage", so it only applies to a stage found
  that way — a lamella or alignment run was never going to be named `adaptive_polish*`.
- **The survey counts what it could not reach at all.** Every other finding describes
  something the conventions recognised. `unindexed-images` counts image files no image set
  claims, which is the failure that hides best: before this work a project holding 430
  images surveyed as holding 100, and nothing said so. It costs a walk per project — 4.6s of
  the reference collection's 40s, where reading labels is 38s of it — and it is what makes
  the project-root overview images visible without indexing them as lamella data. Labels,
  predictions and deliberately ignored names are excluded, or annotating a project would
  report thousands.
- **A loose image is ordered by its header's acquisition time; an Adaptive Polishing image
  by the number in its filename.** Nothing in a loose name orders it: 44% of them carry no
  timestamp at all, and the epochs that do appear are when the milling *stage* started, up
  to eleven minutes before the image. The file's own mtime is worse than useless here — not
  one of 4,000 checked is within two minutes of the header, the median being 134 days out,
  because the collection has been copied. The header is complete (5,237 of 5,237) and
  authoritative, so the scanner reads one per loose image and stores it on the record.
  Adaptive Polishing images are never opened, so ordering them cannot change.
- **A series is ordered across its stages, not stage by stage.** A lamella directory holds
  reference images spanning the whole session while the milling tasks below it hold theirs
  in the middle, so the stages interleave in time and no position for that stage in a
  stage-by-stage list is right. `_in_acquisition_order` sorts a timeline by time whenever
  every frame has one — which is exactly a loose series — and leaves an Adaptive Polishing
  timeline in the stage-and-number order it was built in.
- **An alignment run's fixed filenames break a tie the clock cannot.** A header records
  whole seconds and a run takes several images inside one, so time alone left 11 of 121
  runs reading `final_image` before the last `new_image`. `alignment_run_order` ranks the
  names Autolamella writes — reference, then each iteration, then final — and is used only
  to break a tie, never to order images the clock can separate.
- **Discovery asks filenames, never headers.** 668 of the 13,221 images in the reference
  collection state no beam in their name — the three fixed names in an alignment run, and
  `ref_PositionReady.tif`; all of them read as ION. Discovery still opens nothing: it only
  needs to know whether a directory holds anything, and asks `holds_loose_images`, which
  reads filenames. The scanner opens one header per *loose* image — for the beam where the
  name does not say, and for the acquisition time always — memoised so each file is read
  once, taking the collection scan from 1.30s to 2.84s. The survey opens only what it must,
  because it reports what is on disk rather than ordering it.
- **`predictions/clean/` is indexed but can never become a label.** Adaptive Polishing's
  post-processing rewrites the intensity values, so they no longer index the segmentation
  classes. A label converted from one looks entirely valid — right dimensions, plausible
  shapes — and teaches a model the wrong classes. This will not change upstream, so it is a
  property of `layout.py` (`is_convertible_prediction`) rather than a choice offered to the
  caller. It is still indexed deliberately: otherwise a project holding only post-processed
  predictions would report "no predictions", indistinguishable from one the pipeline has
  never reached, when those need opposite actions.

## Workflow state

- **State is declared in a file, not implied by location, and the key names the label.**
  Every directory holding image and label directories carries an `AM_state.yaml` naming the
  groups that are not selected, keyed by the group's label path
  (`<label-directory>/<filename>`, relative to the state file) whether or not that label
  exists. Keying on the label rather than the image makes an image disagreeing with its
  label *unrepresentable* rather than merely detectable: the several conditions of one scene
  resolve to one key, so a change reaches all of them or none. The key is resolved rather
  than located, so it survives annotation beginning.
- **Anything not listed is selected.** An absent entry and an absent file give the same
  answer, so images dropped into a stage tomorrow arrive selected with no reconciliation.
  The cost is that "reviewed and kept" cannot be told from "not yet looked at".
- **The three states are mutually exclusive, and both representations enforce it.** A
  mapping of key to state cannot express two states for one group, where a file grouping
  keys under `corrected:` / `deselected:` headings could. In memory, `ImageRecord` carries
  one `state` field rather than a `selected` and a `corrected` flag for the same reason: two
  booleans have a fourth combination that means nothing, and every reader then has to spell
  out `selected and not corrected` and be trusted to get it right. Duplicate keys are
  rejected on read rather than resolved last-wins, since a silent winner is the failure the
  mapping was chosen to prevent.
- **Whether a group is corrected is asked of the record, never of a path.** Every guard
  protecting proofread work — conversion, compression, label saving — reads the declared
  state the scanner put on the record. A guard that quietly stops firing is worse than none.
- **A stage is transcribed whole or not at all, and one that has not been is refused.**
  `migrate-state` leaves a stage alone if anything in it is ambiguous — images disagreeing
  with their label, or a file sitting in two state folders — because writing the groups that
  were fine would leave the rest unlisted, and unlisted reads as *selected*. For the same
  reason a state change refuses a stage that still keeps its state in folders. Promoting a
  deselected image into a training set is the one failure here with no visible symptom, so
  both paths are shut rather than guarded.
- **`migrate-state` is the only thing that reads a folder name as a state**, and it walks
  the filesystem itself to do it. The scanner does not look inside `deselected/` or
  `corrected/` at all, so a nested file is *missing and reported* rather than present and
  wrong. The state folder names therefore stay reserved: reading a deselected image as
  selected is the one migration failure that would reach a training set unnoticed.
- **A stale entry is reported, never repaired.** Declared state can disagree with the
  filesystem in ways location could not, and each needs a different action, so
  `inspect-layout` separates them: unmigrated folder state; nested files after migration,
  wanting `--move-files`; an empty state folder; and an entry naming a group that is not
  there. That last is kept rather than dropped — a group that vanished because a drive was
  not mounted must not have its curation deleted for it.
- **A recorded path cannot go stale, so `check-training-csv` only checks.** Nothing moves
  files, so a path that does not resolve means the file is gone. What survives is the check
  and a `--prune` that drops the dead rows, because a CSV holds a particular
  train/validation split and losing a few images is not a reason to regenerate it.
- **A write is read-modify-write, and refuses a file that changed underneath it.** Two
  windows on one collection, or a window and a command line, would otherwise discard each
  other's curation with no error. Every file the batch will touch is compared against what
  the plan read *before the first one is written*, so the answer is "nothing happened" rather
  than "some of it happened"; each write then checks again for itself. That is what lets
  `plan_transition` and `apply_plan` be separate — the decisions are the ones that were
  previewed, however long ago, while the safety check is taken fresh. It is not a
  transaction, so each stage's records are brought into step as that stage lands.
- **Every label write goes through `io/atomic.py`, including compression.** The helper
  exists for two things a hand-rolled `mkstemp` + `os.replace` silently loses. It restores
  the destination's own permissions before the rename, because `mkstemp` creates owner-only
  and OpenCV truncates rather than recreates — so a recompressed label kept `0o600` and
  became unreadable to everyone but its author, which on a shared archive is the same as
  losing it. And it takes a fingerprint, so a label something else rewrote mid-run is left
  as it now is. Compression has to decide *after* encoding, the saving being unknowable any
  other way, so `write_atomically` takes a `commit_if` predicate and reports whether it
  committed — keeping that decision inside the window the rename closes.
- **A compression run that skips a file says which of the three reasons applied.** No
  saving, a dry run and "somebody else rewrote it" are one `applied=False` otherwise, and
  only the third is worth acting on, so `stale` is its own field.
- **Overwriting a label is tiered by what replacing it would cost, not by one flag.**
  Conversion sorts destinations three ways: an empty placeholder costs nothing to replace; a
  pseudo-label holding annotation is regenerable machine output, replaced once the user
  confirms; a label whose group is corrected is refused, and no argument to
  `plan_conversions` reaches it. That last tier is stricter than `compress-labels`, which
  *can* be confirmed for corrected labels, because compression preserves the pixels exactly
  where conversion replaces proofread work with a machine guess. Re-converting a corrected
  label means setting that group back to selected first.
- **That is also why compressing is not scoped to a selection.** There is no label it should
  be withheld from, and a per-selection button leaves uncompressed exactly the labels nobody
  happened to tick. In the window it acts on the whole scanned folder, from Add data; the
  command line keeps `--project` and `--lamella` for the rare case where one project is the
  whole question.
- **Reading a label gives three answers, not two.** `is_label_file_empty()` is tri-state,
  and **None means "cannot tell"** — deliberately distinct from True, because reporting "not
  annotated" for a label nobody could read would be a guess presented as a fact.
- **Corrected is a claim, and it is checked where it is made.** A group marked corrected with
  no label or with a placeholder breaks the claim, and a training export is *right* to drop
  it — what was wrong was doing so in silence, leaving a training set short of exactly the
  images you believe you annotated, discoverable only by counting CSV rows months later. So
  the window asks on either side of the gap, both times through `classify_labels` so the two
  dialogs cannot disagree. The export side asks only about *corrected* groups: an unlabelled
  selected image is the ordinary condition of an archive being worked through, and a warning
  firing on every export is one nobody reads by the third time.
- **The command line warns and proceeds where the window asks.** `correct` lists the groups
  whose label holds nothing and marks them anyway. The window is the place for a guard; a
  flag to override a refusal would be a flag that lives in everybody's shell history.
- **What a state change applies to is a set the user builds, at one level at a time.**
  Ticking a project clears the ticked lamellae, ticking a lamella clears the ticked projects,
  and picking frames out clears both — so what is ticked on screen is the whole of what a
  change will touch. Not a precedence rule between levels: that leaves ticks visible which
  decide nothing and, once a tick outlives the project it was made in, ticks *invisible*
  which decide everything, which is how an operation reaches a project nobody was looking at.
  Ticked lamellae go when the project changes; ticked projects stay, because the project list
  still shows them. Everything reaches `transition_records` through `scoped_records()`, so
  group expansion and the corrected-data guard apply unchanged.
- **A scan the user did not ask for keeps the selection; one they did asks again.** Every
  operation that writes files is followed by a rescan, and preparing labels is two of them
  against the same set, so a rescan that emptied the selection would have *convert* ask for
  the choice *predict* was just given. Pressing *Scan* starts clean. What comes back is
  filtered against what the scan found, and **frames come back by which acquisition they are,
  not by where they sat** — `LamellaFrame.key` rather than an index — because an image added
  ahead of them shifts every index after it, and a state change quietly reaching frames
  nobody ticked is the one way this could go wrong without showing it. The *view* is still
  restored by index: landing on a neighbouring frame is a nuisance, acting on one is a fault.

## Metadata

- **Nothing is inherited from parent directories.** A collection directory groups projects
  by session or storage convenience, so lamellae in one *directory* may be different
  samples. Lamellae in one *project* do share a sample, which is why the project is the
  right level for this metadata.
- **The Add data tab suggests, and a suggestion is never a value.** It offers the values
  other projects already use for a field, computed in the controller and reaching a file
  only when the user saves, so `build_experiment_metadata` still resolves from the two files
  and nothing else. `project` is on that list precisely because it names something several
  datasets share, which is how a collection ends up holding two spellings of one project.
- **`project` names the wider project; the directory is the `autolamella_project`.** They
  are different facts, and holding both under one word showed as a `project_project` column
  in one export and a `metadata_project` in another. The directory is what the scanner
  indexes and Curate lists, so it is qualified wherever it is written; the metadata field
  keeps the plain name. Neither is derived from the other, which is why the Add data tab
  offers no button to copy the folder name into the field.
- **Every image header carries settings for both beams, and some values exist for only one
  of them.** `microscope_state.*` describes the beam that took the image; `system.*` holds
  both beams' configured defaults, so reading `system.ion.voltage` from an SEM image yields
  the FIB's 30 kV — plausible and wrong. `image_metadata.FIELDS` therefore confines
  beam-specific fields, including the fallback search, to the beam that fired, and declares
  which beams a value exists for at all. A plasma gas belongs to an ion column and an emitter
  type to an electron one, so they are two fields rather than one "beam source": one field
  for the two reports the FIB's Argon on SEM images, and the SEM's `FEG` emitter as a plasma
  gas. Each stays blank on the other beam, and on an image whose beam the header does not
  state — so a blank `plasma_gas` on a FIB image is a real answer, a gallium column having no
  plasma to record.
- **A value is stored in the unit its field name states, converted where the header is
  read.** So `acceleration_voltage_kv` is kV even though both dialects record volts, and
  nothing downstream converts anything. A breakdown that groups a *measured* value rounds it,
  because a header records what was measured rather than what was set: a voltage to 0.1 kV,
  and a pixel size and beam current to four significant figures, those two being derived from
  the magnification and so differing in the last decimal between two images of one field of
  view. Significant figures rather than decimal places, since pixel sizes span decades and
  one decimal place would lose either the 0.5 nm end or the 500 nm one. The exported value is
  never rounded.
- **A missing measurement is never a default.** The one exception is the integration counts,
  where absence means "not enabled" — that is decoding the field, not defaulting it.
- **Headers are never read during scanning.** Metadata is a mapping keyed by image path,
  built by the commands that need it, so a scan stays a filesystem walk. Reading one header
  is cheap; reading one per image on every GUI launch is not.
- **Where each header value lives is declared once, in `image_metadata.FIELDS`.** That module
  is to header keys what `layout.py` is to directory names: the alternative is a hand-written
  extraction block per field, so adding a field means writing another block and adding a
  dialect means editing every one. Supporting a third dialect should be a column in a table.
- **A label carries no header.** Its scale comes from the image it was drawn from, rescaled
  by the ratio of their widths, because a label converted from a prediction need not share
  its image's pixel dimensions. `label_measure` returns no areas at all rather than assuming
  1:1.
- **Most TIFFs in the collection have no acquisition header, and should not.** Predictions,
  labels and autofocus frames are not acquisitions; only indexed *images* are expected to
  carry one. A count of "unreadable headers" taken over `rglob("*.tif")` measures the wrong
  population.

## Counting and export

- **A filter and a breakdown are one declaration.** They are the same key function used two
  ways — group by it, or keep the records whose group was chosen — so the filter lists come
  from `BREAKDOWNS` rather than a second table beside it, and cannot disagree with the table
  about what a value *is*. Two consequences: "(not recorded)" is selectable, the only way to
  find images whose header would not read; and a measured property is not offered, since it
  cannot answer before Measure is pressed. Modality, workflow state and the label filter keep
  dedicated controls, being fixed questions rather than data.
- **A command that reports covers every series; a command that acts stays on Adaptive
  Polishing.** Indexing reference and alignment images more than doubled what a scan holds,
  and none of that may change what an existing command prints or writes. So `inspect-layout`
  and `stats` cover everything and say which is which — narrowing a report silently is the
  bug this work exists to fix — while `predict`, `convert-predictions`, the state commands,
  `quantify-labels`, `compress-labels`, `find-specks` and both exports default to
  `--series adaptive`. Widening is one flag. `cli.DEFAULT_SERIES` and `REPORTING_SERIES`
  name the two, and each command chooses where it is registered. Checked by diffing a
  training CSV against the previous version: identical, 965 and 242 lines.
- **The series is registered once, as a breakdown**, so it gets `--series` and an Export
  filter row from the same declaration as the count — per the rule above it. `by_series`
  returns the slug rather than the display label, because a value counted must be a value
  `--series` accepts; the label lives on the `Breakdown`, which is where a reader sees it.
  Unlike a variant, a series is a closed vocabulary, so an unknown name raises rather than
  matching nothing: `--series refrence` is a typo, not a request for an empty set.
- **Counting is split between the tab and the commands, not moved.** The Export tab counts
  the set its filters describe, because choosing a training set means looking at what
  survives a filter and that loop cannot run through a command that prints and exits. `stats`
  keeps the whole-collection sweep and `quantify-labels` the per-lamella table: numbers per
  class per lamella is not something anyone acts on one row at a time, and a spreadsheet
  reads it better from the CSV.
- **What a measured number means is set beside the breakdown, not among the filters.** The
  crack threshold and the area bin resolution change how a measurement is *read*, not which
  images are in play. Changing the threshold re-measures at once rather than spending the
  Measure button again: `read_label_pixels` caches component pixel counts *without* the
  threshold, so a new one costs arithmetic rather than a second pass. `load_measurements`
  keeps the headers it reads for the same reason.
- **Opening the Export tab reads the headers and nothing else.** Half a millisecond an image,
  two seconds over the reference collection, and it is what every acquisition filter and
  breakdown is made of. Offering them greyed out until a button was found and pressed was a
  control that did nothing with no way to tell why.
- **Opening a label is asked for, never assumed.** It is ten times the cost of a header —
  1,232 files and 14 seconds here — and answers a different question: what is *in* the label.
  So the summary shows what a record knows about itself and says plainly that the rest needs
  Measure labels. **Measured over what the filters match**, not the collection: measuring
  1,232 labels to answer a question about the 134 in one project is work nobody asked for.
  Widening a filter therefore asks again for what it brought in.
- **Measuring answers the exportable count too.** Whether there is anything in a label falls
  out of the class counts the pass already has, so the exportable count, the label filter and
  the label-state breakdown all come out of the one pass the user asked for. `classify_label`
  is the shared rule and `_label_empty` the shared cache; keeping two of either meant every
  label read twice.
- **A filter is added, not permanently shown.** The breakdowns cover far more properties than
  any one export is chosen by, so one list per property left too little of each to read and
  buried the three that matter among a dozen nobody touches. A row carries the property it is
  on, and a property already in one row is not offered in another, which is what stops two
  rows disagreeing about it. Removing the last row empties it instead, because a box with no
  filter in it still says what could be filtered on where an empty column says nothing.
- **A few properties are counted by but never filtered on.** A lamella is one row per lamella
  and a milling stage one per timestamped folder, both running to hundreds, and an electron
  source does not vary across this collection at all — value lists nobody could work with.
  `UNFILTERABLE` names them. They stay breakdowns, and the command line still filters on
  `--electron-source`; what is dropped is the list in the window, the scarce thing being the
  column it would sit in.
- **Series is the one filter that does not open on everything.** Every other control on the
  tab starts wide and is narrowed towards an export. This one starts on Adaptive Polishing,
  because a scan now holds 9,001 records where it held 3,764, and a tab that silently counted
  the rest would disagree with `create-training-csv`, which defaults the same way — breaking
  the rule below it that one selection exported either way gives the same file. It is a
  dedicated control rather than a derived filter row, like modality and workflow state: three
  fixed choices rather than whatever the data holds. Checkboxes rather than a dropdown,
  because the three are independent and an export covering two is a real request.
- **The metadata CSV names the series of every row.** An added column, and a deliberate
  change to what an existing export writes: once more than one series can be exported, a row
  that does not say which it came from cannot be read. Shared rather than per-beam in the
  paired shape, because a series is decided by filename and the beam suffix plays no part in
  that decision — checked across the reference collection, where 0 of 5,594 acquisitions span
  more than one.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than on *corrected*: the tab is narrowed
  towards an export rather than opening on one and hiding the archive. The cost is that an
  image somebody decided against can reach a training set — only if it was labelled and
  deselected afterwards, since the empty-label rule catches every other case — so the summary
  line counts the deselected images that would actually be written. Made visible rather than
  made impossible, because "show me everything I rejected" is a real question.
- **Both modalities means side by side, not mixed together.** A training CSV holding SEM and
  FIB rows under one `image` column is not something anyone trains on; the two beams need
  different models. So a selection holding both writes `sem_image, sem_label, fib_image,
  fib_label`, one row per acquisition; a selection naming one beam writes `image, label`. A
  cell is blank where that beam has no image or no label worth pairing with one, and a row
  survives only if at least one beam has a label. The empty-label rule drops a row where
  there is one beam and empties a cell where there are two, and is reported either way.
- **What makes an SEM and a FIB image the same acquisition is stated once**, in
  `frames.group_by_acquisition`: the number in the filename, within one stage directory. They
  do not reliably share a stem — older projects name the file inside `fib/` with the same
  `_SEM` suffix as the electron image. The timeline and the export call that one function
  rather than two that could drift, and `extract_image_number` lives in `layout.py` because
  an acquisition number is a naming rule. Conditions multiply out — five SEM conditions
  against one FIB image are five rows repeating the FIB path, each a genuine pair that
  nothing here can rank — and "one image per label" is the control that collapses it. The
  train/validation split groups by the *acquisition* rather than the label, since a paired
  row holds two: what its beams and conditions have in common is the scene, and splitting
  that would score the model on what it has effectively seen.
- **What pairs the two beams depends on the era of the filename.** Adaptive Polishing
  images carry an acquisition number and their SEM and FIB files do not share a stem, so the
  number matches them. Reference and alignment filenames carry no number and the two beams
  differ only by an `_eb`/`_ib` suffix, so those match on the stem with the suffix removed —
  `layout.acquisition_stem`, beside `extract_image_number` because both are naming rules.
  `_frame_key` picks whichever applies, and no Adaptive Polishing image reaches the second
  branch, so adding it changed nothing: every one of the 1,614 adaptive frames in the
  reference collection kept its identity and its members, diffed rather than counted.
- **`check-training-csv` reads the header rather than assuming it.** Any column called
  `image` or `label`, or ending in `_image` or `_label`, holds a path — so one command checks
  either shape and files written before both. A blank cell is its own answer rather than an
  absent column: it records nothing, so it cannot have gone missing, and keeping it is what
  lets a pruned copy come out the same shape.

## Display

- **A window that has not been drawn cannot say it is working.** `show()` only asks for a
  paint; nothing reaches the screen until `exec()` starts delivering events. The startup scan
  ran between the two, so opening on a folder gave seconds of blank grey with no way to tell
  it from a hang; it is queued at zero delay instead. Where the interface thread does have to
  block, the line saying so is `repaint`ed rather than merely set: an ordinary `setText`
  would reach the screen only after the work it announces had finished. `repaint` draws one
  widget synchronously and processes no events, so it is not the re-entrancy hazard
  `BusyDialog` exists to avoid.
- **One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a palette
  and a short stylesheet; nothing else in the window picks a chrome colour. Fusion rather
  than the platform's style because the two bindings draw the same widgets differently and
  the development and user environments differ by binding. Where a stylesheet rule and the
  palette disagree the rule wins, and a widget palette edited in Python is simply ignored —
  yet that palette still reads back as the value it was set to, so a test asserting on it
  passes while the screen shows something else. **Anything about how something *renders* is
  asserted on the pixels**, for that reason. The stylesheet also suppresses Qt's own
  emphasis: the `QPushButton` and `QToolButton` rules override the tint on a dialog's default
  button and the mark on a checked one, so anything relying on either has to re-state it. Its
  12px padding is wider than a one-character label, which clipped such a button blank while
  it reported its text correctly, so `COMPACT_BUTTON` names the exception.
- **A locked editor keeps its normal text colour.** The Add data tab opens locked and is
  mostly used for reading, and Qt's disabled grey puts every value at low contrast. The
  stylesheet dulls the *background* of a disabled input instead. Buttons and radio groups
  grey out normally: they are controls you cannot use, not values you are reading.
- **A chosen row is ticked, never highlighted.** Highlighting cannot say what it means: it is
  also how Qt marks the row you are looking at, it is lost whenever the list is rebuilt, and
  in four visible rows over dozens of values there is no seeing what is chosen without
  scrolling for it. `CheckList` is the one implementation, used by the curation lists and the
  export filters, with a caption saying "2 of 6 ticked". It reports nothing while its
  contents are replaced: that is a redraw rather than a gesture, and an owner that rebuilt
  its list in response to the current row changing would drive itself in a circle.
- **A list re-tints the icon of the row you are on.** Qt draws a selected row's icon in
  `QIcon::Selected` mode, and the variant it generates for that mode is blended with the
  highlight colour — so a status chip whose colour *is* the information came out a different
  colour on exactly the row being looked at. `chip_icon()` supplies the pixmap for every mode
  and state, which leaves Qt nothing to generate. Any future icon carrying meaning rather
  than decoration has to go through it.
- **A list of paths elides; it must not scroll sideways.** Rows carrying paths differ in what
  they *end* with, and a `QListWidget` with a horizontal scrollbar scrolls instead of
  eliding, which cut a session's projects off into identical-looking rows. `ElideMiddle` plus
  `ScrollBarAlwaysOff` keeps both ends, and the tooltip carries the whole label.
- **`QScrollArea.setWidget` shows whatever it is handed**, so hiding a widget and then
  putting it in a scroller leaves it on screen. Hide the scroller instead, and give the
  column a widget to hold its spare height: a layout shares that height between whatever is
  still visible, and a stretch cannot be hidden again.
- **`QFontDatabase.systemFont(FixedFont)` is not a fixed-width font.** On macOS under PyQt5
  it returns the proportional `.AppleSystemUIFont` while reporting the family it was asked
  for, so a table padded into columns comes out unaligned with nothing saying why. A style
  hint alone does not help: Qt consults it only for a family that does not exist.
  `common.fixed_width_font` names real families per platform. Asserted on the metrics (`"i"`
  and `"W"` advance equally), never on `QFont.fixedPitch()`, which reads back what was
  requested, not what resolved.
- **A tooltip says what a control does, in under 200 characters**, and a test holds the
  ceiling across the whole window rather than one tab — enforced narrowly, the ones out of
  scope grow. Every control in the Filters box also says what it narrows: that it decides the
  breakdown *and* the exports is not visible from the box's title.
- **`downscale_to_fit` uses `min(width_ratio, height_ratio)`.** The images are landscape in a
  portrait pane, so **width binds**, and window chrome does not change how they render.
- **The image panes have no scrollbars, and their scene is bigger than the image.** The scene
  carries a quarter of the image as slack on every side, so an edge can be dragged into the
  middle of the pane to be worked on. Scrollbars go because a scene deliberately larger than
  its content leaves them permanently visible and permanently not full.
- **How far a pane zooms out is the view's to say, and it lands through `fitInView`.**
  Reviewing is done at fit, so the curation panes stop there; the proofreading canvas allows
  one step past it, because checking that a label runs to the edge of the frame means seeing
  that edge clear of the border. The wheel's anchor holds the point under the pointer fixed,
  which is what makes zooming *in* follow the eye — and what would leave a frame that now
  fits the pane pushed into whichever corner the pointer was over, so at the floor the two
  cannot both be honoured and fitting wins. `fitInView` also keeps a two-pixel border of its
  own that scaling to the computed floor would crop, so "zoomed all the way out" and "fitted"
  are the same picture.
- **A summary plot is held twice: pane-sized to step through, whole to look into.** A plot is
  ~180 ms to read however it is asked for — a PNG decompresses whole, so OpenCV's
  reduced-decode flags buy nothing — which is slower than a held-down arrow key, so stepping
  is smooth only when every plot is already in memory, and that bounds those copies to the
  size of the pane. But these figures are 4500x3000 and carry annotations that exist only at
  that resolution. So a *fitted* plot is the pane-sized copy, drawn at one array pixel per
  physical pixel, the only scale that resamples nothing; the whole frame is read last, and
  takes over once the figure is magnified past what the pane copy holds. That threshold
  carries a hair of margin, because returning to fit lands either side of exactly 1 and the
  wrong side leaves the one view that has to be exact filtered.
- **Where a copy is placed is read off the copy, and remembered across a swap.** The copy is
  reduced to a box measured in *physical* pixels, so drawing it at whatever device ratio the
  widget reports at the time comes out at half or twice the size wherever the two disagree;
  whichever of its edges touches that box gives the ratio back exactly. An identity transform
  says nothing about where the view is looking, and the scene is deliberately larger than the
  picture, so drawing 1:1 has to centre as fitting would. Where the view *is* looking is held
  as a fraction of the picture rather than re-derived: deriving goes through the scroll
  position, which is a whole number, and stepping while zoomed changes resolution twice a
  frame, so the picture crept towards one corner. It is dropped the moment the user pans,
  which `scrollContentsBy` reports.
- **Read-ahead asks for no more than the cache can keep.** Unbounded, the read that finishes
  last evicts the one that finished first, so a step in either direction lands on something
  that was prepared and then thrown away. `plot_radius` is derived from the entry limit, so a
  lamella that fits is prepared whole and one that does not is prepared around where the user
  is — on the longest lamella here, 61 plots against 38 that fit.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**,
  and that mode defaults to nearest neighbour. A pane showing something several times its own
  size therefore drops pixels rather than averaging them: measured against a figure of
  one-pixel rules, nearest loses whole gridlines. The plot view asks for smooth scaling.
  Acquisitions do not, because zoomed *in* on a label, nearest is what shows where one pixel
  ends and the next begins.
- **The busy dialog is not a `QProgressDialog`.** That class calls `processEvents()` from
  inside `setValue()` whenever it is modal — which this one has to be, since its whole job is
  to keep the window out of files that are being written. Processing events there re-enters
  the event loop from inside a queued-signal handler, so the worker's *finished* can be
  delivered in the middle of a progress update: the dialog is closed and its last reference
  dropped while one of its own methods is still on the stack. It showed as the window
  freezing on the final item of a run. `busy_dialog.py` is a plain `QDialog` that processes
  nothing, and the window detaches it before closing it. Escape is refused for the same
  reason the close button is absent: on a stoppable operation it asks it to stop, and on any
  other there is nothing it could safely do.
- **A long operation runs on a thread, and says how far it has got.** Qt draws nothing while
  the thread owning the widgets is busy, so the work goes to `gui/worker.py` behind a modal
  dialog. That dialog opens indeterminate and becomes a count as soon as the operation
  reports, through one `on_progress(done, total)` contract and one `progressed` signal
  connected once for the whole window — so giving an operation a counter is a callback in its
  loop, not a mechanism. Only the applying run counts, never the preview, which would fill a
  bar instantly and describe the wrong run.
- **Whether it can be *stopped* is a different question, and the answer is what a half-done
  run leaves behind.** Each prediction is written atomically and recorded as this model's
  work before the next begins, so a stopped run has produced finished files and an accurate
  account of them, and the next run resumes from there. Recompression is safer still: a
  re-encoded label holds the same pixels as the one it replaced. Converting predictions must
  not stop — it validates the whole set of destinations before writing any of them, and
  stopping it leaves exactly the half-applied collection that rule exists to prevent.
  Cancellation is a `threading.Event` in `progress.py` polled by the work closure, rather
  than a Qt signal: a queued signal reaches the worker when it next returns to its event
  loop, which inside a long call is never.

- **Curate shows one series at a time; Export ticks several.** Different questions. A
  timeline mixing alignment runs with polishing frames is not something anyone proofreads,
  and a 2026 lamella runs to 21 stages across three series, so the Curate control is a single
  choice that rebuilds the frame stack — the position and the frame ticks go with it, because
  a frame index only means anything within one timeline. An export covering two series is a
  real request, so that one is checkboxes. Both open on Adaptive Polishing.
- **The Curate control is offered from what the lamella holds, not from the vocabulary**, so
  it never offers a choice that would show an empty timeline, and a pre-Autolamella lamella
  gets no control at all. It is set before the "no frame" early return in `_render`: a
  lamella can hold nothing of the series being shown while holding plenty of another, and the
  control is the only way out of that.

- **One record per image, where a lamella holds the same file twice.** Autolamella copies
  the image an alignment aligns *against* into every run that uses it, so one acquisition
  appears as up to six files — 121 `reference_image.tif` over 51 acquisitions here. Each
  would be a separate frame to annotate, a separate row in an export, and a separate
  training pair of the same picture with the same label. The scanner keeps the earliest and
  drops the rest, from the image set as well as the records so the two cannot disagree.
- **The filename is what makes recognising a copy safe.** Two images taken inside one second
  are ordinary — an alignment run takes several — so matching on acquisition time and size
  alone merged 38 groups of genuinely different pictures across the reference collection,
  `new_image_00` and `new_image_01` of one run among them. Matching on filename *and* time,
  across directories of one lamella, merged 34 groups of 70 files, every one byte-identical
  to what it merged with. The residual risk is two runs of one lamella sharing an instant,
  which cannot happen while they cannot overlap. No file is hashed: the acquisition time is
  already read, and it partitions these exactly as content does — checked against SHA-256
  over all 121.
- **A plot describes one acquisition or a whole stage, and the window says which.** An
  Adaptive Polishing stage writes one plot per milling cycle into `plots/`, matched to a
  frame by its acquisition number. An alignment run writes one `figure.png` at its root
  summarising the several images it took to converge, so there is nothing to match it to:
  every frame of the run shows it, and carries a badge saying so, because a figure
  describing five frames read as describing one is worse than no figure. `frame.plot_path`
  stays the frame's own and `stage_plot` the stage's; `displayed_plot` picks, preferring the
  frame's. `AP_summary_plots.png` is recognised by the same rule — it summarises a whole
  milling run — though it never displays today, since every stage that has one also has a
  plot for every acquisition. `centring.png` sits beside it and is deliberately excluded:
  it records where the plugin placed its field of view, not what the run achieved.

## Environment

- **A save refuses a label whose file changed since it was opened.** Proofreading holds a
  lamella's edits in memory until the reviewer saves, which leaves a window of minutes to
  hours in which something else can rewrite a label file. That is not hypothetical here: this
  collection lives on OneDrive, which writes files back under a running program, and the
  whole point of the corrected state is that the annotation in it cannot be reconstructed. So
  the size and modification time are recorded when a label is opened and compared before
  writing, and a mismatch stops the entire batch rather than the one file — which of the two
  versions is wanted is not a question this package can answer.
- **Run the suite from the repository root, in `am_segmentation_manager` or
  `am_segmentation_manager_ml`.** Two ways to get this wrong, and neither announces itself.
  Run `pytest` from *outside* the repository and there is no `pyproject.toml` above the
  working directory, so rootdir becomes that directory, `testpaths` never applies, and pytest
  recurses into everything below it — from `$HOME` that means importing every `.py` under
  `Library/`, `Dropbox/` and a 52 GB `Data/`, which looks exactly like a hang. Anywhere
  *inside* the repository is safe; rootdir discovery walks up. Run it in conda's `base` and
  the imports resolve to a different stack entirely — PyQt5 rather than PySide6, and whatever
  `cv2` base happens to have — giving several hundred failures that are all one wrong
  interpreter. `pytest.importorskip("cv2")` does not protect against a half-removed OpenCV,
  because the shell package still imports; only the attribute access fails.
- **The reference collection is at
  `<collection>/`** — 12 projects, ~3500 indexed
  images, 52 GB, spanning Autolamella and pre-Autolamella layouts. Treat it as read-only.
  Point the smoke test at it with `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.
- **Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side
  effect, and `test_gui_*` sorts before most other tests. That is how a locale-dependent
  `strptime('%p')` bug appears only in full runs, and only on a machine with a non-English
  locale. `LC_ALL=… pytest` does **not** test this; `AM_SEG_TEST_LOCALE=… pytest` does.
- **Never disconnect a signal from inside the handler for that signal.** A connection list
  mutated mid-emit works under PyQt5 and hangs under PySide6 — the suite stalled indefinitely
  on the first test to run a real background operation, with no traceback, because the
  failure path opens a modal box nobody is there to dismiss. Connect once and track what is
  pending in an attribute. The same shape hides in any connect-per-call pattern.
- **`QProxyStyle.styleHint` segfaults PySide6.** Overriding it to answer
  `SH_DialogButtonLayout` — to fix the button order in every dialog at once, since Fusion
  reports the Windows order wherever it runs and ends a row with *Cancel* — crashes the
  process while a `QMessageBox` is being built. Returning the enum rather than an `int` is
  separately wrong and only warns. The save dialog lays its own buttons out instead; any
  dialog that cares about the order has to.
- **An offscreen window cannot answer three questions**, and asking anyway produces tests
  that pass on nothing. It is never activated, so `hasFocus()` is False for every widget and
  `focusWidget()` is what records who asked. Nothing is shown, so `isVisible()` is False
  whatever the code did and `isHidden()` is the state to assert. And a modal dialog has
  nobody to dismiss it, so its `exec_` must be intercepted — per dialog class, not just
  `QMessageBox` — or the suite hangs rather than fails.
- **Every file dialog is Qt's own, never the platform's.** The macOS panel under PyQt5 5.15.2
  offered no selectable files at all for a `*.csv` filter, only folders to navigate into, so
  "Check exported CSV" could not be given a CSV — and because directory dialogs were
  unaffected it read as a missing feature rather than a broken one.
  `widgets/common.build_path_dialog` sets `DontUseNativeDialog` for all of them. Revealing a
  folder in the file manager is not a dialog and stays native.
- **A dropdown you can arrow through is `StepCombo`, not Qt's own keys.** Whether Up and Down
  step through a combo box or open its popup is decided by the style hint `SH_ComboBox_Popup`.
  Under PyQt5 with the `macintosh` style they step, so the difference is invisible here — but
  the hint belongs to the style, and the window ships under a different binding.
- **Figures are built as `Figure()`, never through `pyplot`.** pyplot keeps every figure it
  makes in a global list until something closes it, and choosing a backend for that list is
  what makes a chart drawn over ssh look for a display. A bare `Figure` has neither problem:
  `savefig` works on it, the window attaches its own canvas, and a session of charts is
  collected as the dialogs holding them close. `matplotlib` is in the `gui` and `all` extras
  rather than only `plots`, because against the OpenCV and Qt those already install, its size
  is not what makes an environment large.
- **Never mix conda-forge with Anaconda `defaults`.** `environment.yml` pins `nodefaults` for
  this reason: a mixed environment paired Anaconda's OpenCV with conda-forge's protobuf and
  `cv2` would not load at all.
- **Inference lives in a second environment, and that one is pip-first.** The main environment
  is conda-first because conda's OpenCV and pip's cannot coexist. albumentations, which the
  vendored preprocessing imports, requires `opencv-python-headless` — a third provider of
  `cv2`, and one no ordering of install commands avoids. So `environment-ml.yml` takes only
  python and pip from conda and the whole stack from the `[ml]` extra, where there is exactly
  one OpenCV. It also lets the torch build be chosen per machine, which conda-forge cannot
  express: `AM_SEG_TORCH_INDEX_URL` selects a wheel index. `scripts/setup_ml.sh` builds it.
- **The inference environment is a superset, not a peer.** It gets the window too, via the
  `qt` extra — Qt binding and matplotlib, no OpenCV, since it already has the headless one,
  which is also the better partner for PySide6 (`opencv-python` carries Qt plugins of its
  own). The curation environment is the *lighter* of the two, for someone who never predicts;
  splitting them the other way forces an environment switch mid-workflow.
- **Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
  `ml/environment.py` answers "is inference possible here" without importing it, the way
  `io/image.py` does for the image stack. `ml/preprocessing.py` is a verbatim copy of
  AM_training's module, excluded from ruff: reformatting it would destroy the only cheap
  answer to "is this still the preprocessing the model was trained with" — `diff`.

## Inference

- **A fact recoverable from the weights is never declared.** A checkpoint is a bare
  `state_dict`, and its class count, input channels and tensor names are all readable from it.
  Declaring those beside it adds a second statement that can disagree with the first, and the
  disagreement would be invisible: a description naming four classes over a five-class head
  loads whatever fits. So `ml/model.py` reads them, and `load_state_dict(strict=True)` proves
  the description's architecture and the weights are the same model. What the weights cannot
  state — preprocessing settings and the architecture name — goes in a `.yaml` beside them,
  and every one of those shows itself immediately if wrong, which is what makes a default safe.
- **Which modality a model is for is not recorded anywhere, deliberately.** Nothing in a
  checkpoint says, and `AM_SEM_All_V01.pth` is whatever whoever trained it chose to type —
  reading that as evidence would make the software's answer depend on someone else's
  file-naming habit. Declaring it in the sidecar only moves the guess: a declaration nothing
  can check is a claim wearing the clothes of a fact. So the selection states it, and a
  selection spanning both modalities is refused — the guard is about something the window can
  show you rather than a line in a file you would have to open. Every report leads with the
  modality.
- **Predicting with the wrong model is cheap to undo**, which is what makes that guard
  sufficient rather than merely defensible. Resumption keys on the model and run name, so
  predicting the same images with the right model replaces the wrong ones without
  `--overwrite`: an accident costs the inference time and nothing else.
- **`decoder_attention_type="scse"` is not part of these models.** The training call passes
  it, smp's `FPN` has no such parameter, and it is swallowed into `**kwargs` and forwarded to
  the encoder, which ignores it. The checkpoints contain no attention tensors, and a plain FPN
  strict-loads all 1015 of them. Build plain FPN; do not carry the argument over on the
  assumption that it did something.
- **One prediction per label group, not per image.** The label is the unit of workflow state
  everywhere else here, and a prediction exists to become a label, so a group gets one — made
  from `label_groups.primary_record`, the same image the curation window shows and a training
  export keeps. Predicting the other conditions would cost inference to produce worse
  pseudo-labels: `SEM_8_fi_1` is one frame integration and far noisier than the `SEM_8` it
  shares a label with. It also removes a hazard rather than managing one — conditions carry
  identical filenames into a shared predictions directory, so predicting all of them would
  have each write replace the last.
- **Inference runs in a process of its own, and the window imports no torch at all.** The
  window crashed repeatedly after a completed prediction — a different place each time, always
  in an allocator or destructor on the interface thread, and never once in ten command-line
  runs doing the same work on the same machine and device. What the two differ in is the
  thread inference runs on and the Qt event loop around it, neither fixable from inside. So a
  child holds torch and the window none. Nothing is lost when that child dies, however it
  dies — each prediction is written by rename and its provenance flushed before the next.
  Stopping is `SIGINT`, caught between whole images, a signal it could not handle stranding
  the temporary of a half-written file. The child's log is forwarded so one file still holds
  the whole run: the line naming the device is written there, and it is the line that placed
  the last crash. *spawn*, so the child inherits nothing Qt has touched.
- **Whether to offer the predict button is decided by locating torch, not importing it.** An
  import costs seconds at every window start, including for someone who never predicts, so
  `ml_stack_present` uses `find_spec`. Pressing it runs no deeper check: a stack that is
  missing or broken is reported by the child, which is the process that has to import it
  anyway.
- **The device box does not import torch to fill itself.** Which devices exist differs per
  machine — several CUDA devices on a workstation, an MPS one on a Mac — so the list cannot be
  written down here, and asking costs the seconds `ml_stack_present` avoids. It opens on
  `auto` and `cpu`, true everywhere, and gains the machine's own the moment anything loads
  torch, which is a child process's job too, asked once per session when the box is opened.
  Editable, since `resolve_device` passes a named device through unexamined: an unheard-of
  accelerator can be typed rather than waited for.
- **Predictions are written at model resolution.** A 2048x3072 image yields a 1024x1536
  prediction — which is what the segmentation pipeline's own `raw/` already holds.
  `convert-predictions` resizes to the image when it makes the label, so upsampling earlier
  would only be undone.
