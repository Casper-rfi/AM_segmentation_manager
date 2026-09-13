# Decisions

Why the software is built this way, and facts not safely derivable from the code: the code shows
what happens, not what breaks if it changes. The window, the Qt binding and every trap in either
are [`display.md`](display.md). `CLAUDE.md` says where a fact belongs; correct an entry in place
rather than appending to it.

---

## Architecture

- **No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python controllers, which
  is what makes navigation, cache eviction and state expansion testable without a display server.
  Adding a tab is not a reason to bend it, and the modules held to it are **found rather than
  listed** — a list is extended by whoever adds a controller, and three were added without it
  (`test_module_does_not_import_a_gui_toolkit`). Only `loader` and `worker` are excepted.
- **No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few lines with
  no dependency, over thousands of records rather than millions.

## Filesystem and identity

- **A label directory is `<image-directory>_labels`, a sibling of the image directory.** A pure
  string function both ways with no lookup table; `sem/` is no special case, its private and shared
  names coinciding. Conditions share a label by default, and giving one its own means creating the
  directory rather than configuring anything. **Label directories are the only ones anything here
  renames** — we do not own `sem` and `fib`, and renaming those breaks reopening the project in
  Autolamella. **Legacy spellings stay readable indefinitely** (`labels`, `labels_fib`,
  `labels_FIB`), a new label landing in whichever directory an image set already has, so normalising
  is cosmetic and lives in `scripts/` rather than in a command.
- **A copy is paired back to its original by what the acquisition software recorded, never by name
  or order**, both of which survive a copy unchanged. `scripts/place_renamed_labels.py` names the
  header facts that do and reads them itself, `image_metadata.py` having no use for an identity.
- **Adaptive Polishing images can sit loose in the lamella directory.** Usually they fill a
  `<modality>` directory that names them, but the script-based prototype wrote them beside
  Autolamella's own references as `adapt_mill_img_000_electron_000.tif` — spelling the beam out
  rather than suffixing it, which is why `beam_modality` cannot read it and the rule does. So all
  three labelled series are in `LOOSE_SERIES`, and such a set's labels go to `adaptive_sem_labels/` by the
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
  a third marker and is not usable as one. **A marker in a gathering directory makes every dataset
  beneath it a lamella** — the mistake most likely to be made by hand, since one file per collection
  is the obvious shortcut (`test_a_marker_in_a_collection_directory_does_not_swallow_its_projects`,
  `test_metadata_is_not_inherited_from_a_collection_directory`). No structural test separates the
  two shapes in general, so `project_placement_problem` is the single guard both interfaces ask, the
  search and the levelling are separate calls, and a person confirms. What a proposal *would become*
  is the useful column: lamella names reading like dataset names tell a reader the level is wrong.
- **Identity is the path relative to the scan root, never the name** — for a project, and for a
  lamella within its project. Two projects can have identical names, `created_at` and recorded
  paths, so keying by name silently drops one, and two lamellae sharing a petname merge. A lamella
  is counted as `project / lamella`.
- **The lamella and stage levels are not always directories.** Older datasets collapse them; the
  missing level is synthesised in memory with the parent standing in, and the model records that it
  is synthetic. Data on disk is never reorganised to fit the newest layout.
- **The series is a property of an image set, not of a directory.** A lamella directory holds
  reference images of both beams and alignment images at once, so a set is a *group of filenames*
  within a directory rather than the directory itself; `image_files_of` reads back either shape.
- **A loose-image stage does not end the stage search; a directory-shaped one does.** The invariant
  the 2026 layout rests on: a lamella holds reference images loose and its milling tasks below, each
  task holding its own references loose and its stages below
  (`test_reference_images_above_a_stage_do_not_hide_it`). Nothing sits below an image directory.
- **A loose set's label directory is resolved exactly, with no fallback chain**, unlike
  `resolve_labels_directory`, which falls back to the shared `<modality>_labels`. Falling back would
  pair a reference image with a label of a different scene — right dimensions, plausible shapes,
  wrong data (`test_a_loose_set_never_falls_back_to_the_shared_directory`).
- **A series nobody has started labelling is not reported as unlabelled.** `no-labels-yet` fires
  per image set, and the loose series add hundreds of sets nothing has annotated — enough to bury
  every finding worth acting on. `layout.LABELS_EXPECTED` names the series that get the report; a
  set whose label directory *does* exist is reported on normally whatever its series.
  `unexpected-stage-name` is narrowed the same way, to a stage found by holding image *directories*.
- **Every image is indexed, and `other` holds the ones no naming rule reaches** — the residue, not
  a claim about what an image is for, so a picture nobody anticipated can be opened rather than only
  counted. Its `labels_directory` and `predictions_directory` are `None`, which is what makes every
  write path refuse it where a runtime guess would not
  (`test_alignment_images_are_indexed_but_never_become_segmentation_targets`); an image whose beam
  neither filename nor header states joins it too, a labelled series' set being keyed by beam
  (`test_a_resolver_that_cannot_say_leaves_the_image_reported`). Reaching every image means reaching
  every directory holding one, so `holds_loose_images` asks the extension, not a naming rule
  (`test_a_directory_of_images_no_rule_recognises_is_still_a_stage`).
- **Indexing the residue moved the alarm rather than silencing it.** `unrecognised-image` is the
  tripwire that makes an unanticipated layout visible — `_unindexed_findings` says what it once
  caught — and a set absorbing those files quietly would have disconnected it. So a file is indexed
  *and* still reported, the count staying small and non-zero being what keeps it a tripwire
  (`test_images_no_naming_rule_reaches_are_indexed_as_other_and_still_reported`). Behind it,
  `unindexed-images` keeps what the *traversal* never offered to a set — below the depth limit, or a
  directory whose role stops the search.
- **Three checks cover three ways data hides, and no image is in two of them.** `unindexed-dataset`
  reports a directory that *looks like* a dataset but has no marker; `stray-images` reports image
  files inside no project at all, which is neither that nor anything a project-scoped check can see
  — 215 when it was written, 204 under one `log/` directory, with no command that would have said so
  (`test_an_image_below_a_directory_beside_a_project_is_reported`). Its walk stops at every project
  root, at a directory a role claims, and at one already reported, so it costs a few listings rather
  than a second pass and cannot report a file twice
  (`test_a_directory_already_reported_as_a_dataset_is_not_reported_twice`).
- **A loose frame is ordered by its header's acquisition time, across its whole series.** Nothing in
  a loose filename orders it: many carry no timestamp, and the epochs that do appear are when the
  milling *stage* started, minutes before the image. **mtime is worse than useless**, the collection
  having been copied. Nor can the stages be walked in order: a lamella holds reference images
  spanning the whole session while the milling tasks below hold theirs in the middle.
  `_in_acquisition_order` sorts by time whenever every frame has one — exactly a loose series —
  leaving an Adaptive Polishing timeline, ordered by filename, alone. Ties go to
  `alignment_run_order`: a run takes several images inside the whole second a header records.
- **Indexing opens an image only in the scanner, and only a loose one**, for the beam its filename
  may not state and the time it never does — one memoised read. Discovery opens nothing: it asks
  `holds_loose_images`, which reads filenames. **It is the largest single cost of a scan**, most
  images in this collection being loose, so a stage's headers are read together in a pool of
  threads — on `build_loose_image_sets`'s own predicate, so nothing is opened that it would not look
  at. Half of a read waits on the disk and overlaps; the other half parses tags under the GIL and is
  the ceiling, which is why more threads stop helping at eight.
- **One record per acquisition, where a lamella holds the same picture several times.**
  Autolamella duplicates the image an alignment aligns *against* in two ways, and
  `_copied_image_paths` sets out the two rules that see them and which copy is kept.
  **Directory and time must agree independently on every match** — neither signal carries it
  alone (`test_two_runs_are_told_apart_by_when_they_ran`) — and **nothing is hashed**, the time
  being already read. A third pass runs over the whole *project*, keeping the filename-and-time
  key rather than loosening to time alone, which would merge a misfiled reference whose indexing
  keeps the misfiling visible. **It never empties a lamella**, a duplicate that can be seen
  beating a directory dropped out of the index and the survey at once.
- **A fixed filename marker is matched to the shape Autolamella writes, not loosely**, everything
  after `ref_` being a *task* name somebody chose:
  `test_a_task_named_after_alignment_is_still_a_reference_image` says what a prefix match costs, and
  `_centring_` is anchored to the end for the same reason — as a bare substring it silently drops
  every image of a task called `Centring`, ignored images not being reported.
- **Where a record's label and predictions live is carried, never re-derived**, for the reason
  `label_destination` gives — deriving from the image path works only for an Adaptive Polishing
  image and puts a reference image's files outside the lamella, where the next scan never looks.
  The scanner resolves both once per image set and the record carries them, so reading and writing
  cannot disagree; a record that does not say raises rather than guessing.
- **A ticked lamella covers only the series on screen.** What is ticked has to be the whole of
  what an operation touches; filtering by modality alone reached every series, so predicting a
  lamella while looking at its Adaptive Polishing timeline predicted its reference and alignment
  images too. Each lamella is asked what it can show of the chosen series, as navigating to it
  would — `frames.resolve_series` — so one holding none of it contributes what it has.
- **`predictions/clean/` is indexed but can never become a label**, for the reason
  `test_post_processed_predictions_are_never_converted` gives. That will not change upstream, so it
  is a property of `layout.py` (`is_convertible_prediction`) rather than a choice offered to the
  caller — and the rule excludes post-processed output rather than requiring `raw/`, a flat
  predictions directory being common. Still indexed deliberately: otherwise a project holding only
  post-processed predictions reports "no predictions", indistinguishable from one the pipeline never
  reached, and those need opposite actions.

## Workflow state

- **State is declared in a file, not implied by location, and the key names the label.** `state.py`
  states the format. The *label* is the key because that makes an image disagreeing with its label
  unrepresentable rather than merely detectable — the conditions of one scene resolve to one key, so
  a change reaches all or none — and it is resolved rather than located, so it survives annotation
  beginning. The cost of "no entry means selected" is that "reviewed and kept" cannot be told from
  "not yet looked at".
- **The three states are mutually exclusive, and both representations enforce it.** A mapping of key
  to state cannot express two states for one group, where a file grouping keys under `corrected:` /
  `deselected:` headings could. `ImageRecord` carries one `state` field rather than a `selected` and
  a `corrected` flag for the same reason: two booleans have a fourth combination that means nothing,
  and every reader must then be trusted to spell out `selected and not corrected`.
- **Whether a group is corrected is asked of the record, never of a path.** Every guard protecting
  proofread work — conversion, compression, label saving — reads the declared state the scanner put
  on the record; a guard that quietly stops firing is worse than none.
- **Nothing here reads a folder name as a state; `scripts/migrate_state.py` does, once**, and its
  docstring says why it is a script. The scanner does not look inside `deselected/` or `corrected/`,
  so a nested file is *missing and reported* rather than present and wrong, and those names stay
  reserved.
- **A stale entry is reported, never repaired.** Declared state can disagree with the filesystem
  in ways location could not, and each needs a different action, so `inspect-layout` separates
  them: unmigrated folder state; nested files after migration, wanting `--move-files`; an empty
  state folder; and an entry naming a group that is not there. That last is kept rather than
  dropped — a group that vanished because a drive was not mounted must not lose its curation.
- **A write is read-modify-write, and the whole batch is checked before any of it is written** —
  the rule every batch here follows, state changes, conversion and folder-wide metadata alike, and
  what stops two windows on one collection discarding each other's curation
  (`test_a_stage_changed_since_planning_stops_the_whole_batch`). Every file is compared against what
  the plan read *before the first is written*, so the answer is "nothing happened" rather than "some
  of it happened"; each write then checks again for itself. That is what lets `plan_transition` and
  `apply_plan` be separate — previewed decisions, fresh safety check.
- **Every label write goes through `io/atomic.py`, including compression.** It restores the
  destination's own permissions before the rename, because `mkstemp` creates owner-only — a
  recompressed label kept `0o600` becomes unreadable to everyone but its author, which on a shared
  archive is the same as losing it (`test_a_committed_file_keeps_the_destination_permissions`).
  Compression decides *after* encoding, so `write_atomically` takes a `commit_if` predicate.
- **Overwriting a label is tiered by what replacing it would cost, not by one flag.** Conversion
  sorts destinations three ways: an empty placeholder costs nothing to replace; a pseudo-label
  holding annotation is regenerable machine output, replaced once the user confirms; a corrected
  group is refused, and no argument to `plan_conversions` reaches it. Stricter than
  `compress-labels`, which *can* be confirmed for corrected labels, compression preserving the
  pixels where conversion replaces proofread work with a machine guess. That is also why
  **compressing is not scoped to a selection**: a per-selection button leaves uncompressed exactly
  the labels nobody happened to tick. An overwrite that fails part way must still leave the label
  readable (`test_an_interrupted_conversion_leaves_the_original_label_intact`).
- **Corrected is a claim, and it is checked where it is made.** A group marked corrected with no
  label or with a placeholder breaks the claim, and a training export is *right* to drop it — what
  is wrong is doing so in silence, leaving a training set short of exactly the images you believe
  you annotated. So the window asks, through `classify_labels` so two dialogs cannot disagree, and
  only about *corrected* groups: an unlabelled selected image is the ordinary condition of an
  archive being worked through. The command line warns and proceeds where the window asks.
- **What a state change applies to is a set the user builds, at one level at a time.** Ticking a
  project clears the ticked lamellae and vice versa, and picking frames out clears both — so what
  is ticked on screen is the whole of what a change will touch. Not a precedence rule between
  levels: that leaves ticks visible which decide nothing and, once a tick outlives the project it
  was made in, ticks *invisible* which decide everything. Everything reaches `transition_records`
  through `scoped_records()`, so group expansion and the corrected-data guard apply unchanged.
- **A scan the user did not ask for keeps the selection; one they did asks again.** A rescan that
  emptied it would have an operation ask again for the choice the one before it was just given;
  pressing *Scan* starts clean. What comes back is filtered against what the scan found, and
  **frames come back by which acquisition they are, not by where they sat** — `LamellaFrame.key`
  rather than an index. The *view* is still restored by index: landing on a neighbouring frame is a
  nuisance, acting on one is a fault.
- **A write that says what it created does not rescan.** Saving labels, predicting and converting
  change two fields of `ImageRecord` — `labels` and `predictions` — so what was written goes onto
  the records (`_attach_created_labels`, `CurationController.attach_written`) and a walk of the
  collection finds nothing more. A shared label is named on **every** record of its group, a
  condition left without it drawing as unlabelled. **What was written, never what was planned**:
  a stopped run plans more than it reaches (`test_a_stopped_run_reports_only_what_it_reached`).
- **Naming a written file is not enough: three caches stand between a record and the screen**, and
  what is drawn is not recomputed from the records. `record_label_state` keeps the answer it read,
  the frame cache holds each frame as a composed image *and its label* keyed by position, and the
  Export tab holds label states and measurements — all three reset by the scan that used to follow,
  so a record named without clearing them draws as it did before the file existed
  (`test_a_written_label_changes_what_the_timeline_reports`). `attach_written` and
  `ExportController.forget_labels` drop them together; the frame cache goes whole, being keyed by
  position rather than by what made it, and the Export tab's three are **not keyed alike**
  (`test_a_replaced_label_is_read_again_rather_than_remembered`). A scan still finds what changed
  the files from outside, which is what *Scan* is for.

## Metadata

- **The Add data tab suggests, and a suggestion is never a value.** The values other projects use
  are computed in the controller and reach a file only when the user saves, so
  `build_experiment_metadata` still resolves from the two files and nothing else. **A completer that
  matches anywhere must complete into a popup, and the popup must be taught Tab** — two halves of
  one fault, both putting a value nobody typed into exported metadata
  (`test_matching_anywhere_never_completes_in_place`). `common.match_anywhere` is the only place any
  of it is set, and where the key has to be caught is a Qt matter in [`display.md`](display.md)
  § The binding. Matching anywhere is worth both: `Quantifoil` and `UltrAuFoil` differ in their
  middles.
- **`project` names the wider project; the directory is the `autolamella_project`.** Different
  facts, and holding both under one word showed as a `project_project` column in one export and a
  `metadata_project` in another. The directory is what the scanner indexes and Curate lists, so it
  is qualified wherever written; the metadata field keeps the plain name. Neither is derived from
  the other, which is why Add data offers no button to copy one across.
- **A descriptive field is a row in `metadata.FIELDS`**, as a header value is a row in
  `image_metadata.FIELDS`: adding a field is one row rather than an edit in six modules, and adding
  a header *dialect* is a column rather than an extraction block per field. **Nothing names a
  particular field to find out what it is**, pinned from both sides by
  `test_a_field_that_can_be_counted_by_gets_a_breakdown_of_its_own` and
  `test_a_vocabulary_is_rejected_by_name_rather_than_by_a_named_field`. `defaults` are the usual
  answers offered where `choices` would require — two freezing methods cover almost every sample,
  and jet freezing still has to be typable.
- **A value is the type its field says, from the moment it is typed.** One `format_field_value`
  renders every one back, because the editor decides whether you edited a field by comparing what it
  shows against the resolved value: a deposition time shown as `30.0` where `30` was typed makes
  every project merely looked at report an edit. So a number is refused unless it parses, written
  unquoted, and carried as a float — **which makes zero a recorded value and a falsy one**, so
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
  would survive. Like `sample` it cannot be defaulted from a shared file: a grid belongs to the
  specimen, and one collection directory can hold a yeast project beside a virus one.
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
  *measured* value rounds it, a header recording what was measured rather than what was set: pixel
  size and beam current are derived from the magnification and differ in the last decimal between
  two images of one field of view. Significant figures rather than decimal places, pixel sizes
  spanning decades. The conversion rounds to six decimals in the stored unit, binary floating point
  turning 6e-05 m into 59.99999999999999 um.
- **A declared path counts as matched only if its value converts**, a key being able to state
  nothing while present: the FEI dialect leaves uninitialised memory in the stage keys it never
  fills, so `Stage.StageX` is a subnormal float while the real position sits in
  `Stage.CompustageX`. `test_uninitialised_memory_is_not_a_measurement` pins what reads as absent.
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
  release series, so one "software version" column would answer a different question per image
  (`test_the_three_software_versions_are_kept_apart`). The microscope's own version is the only one
  the FEI dialect states, under `System.Software`, and so the only one covering the pre-Autolamella
  datasets.
- **A label carries no header.** Its scale comes from the image it was drawn from, rescaled by the
  ratio of their widths, a label converted from a prediction not necessarily sharing its image's
  dimensions. `label_measure` returns no areas rather than assuming 1:1.
- **Most TIFFs in a collection have no acquisition header, and should not.** Predictions, labels
  and autofocus frames are not acquisitions; only indexed *images* carry one. A count of
  "unreadable headers" over `rglob("*.tif")` measures the wrong population.

## Counting and export

- **A record can be in more than one group, and the table says so.** A property an image has
  several of counts the image under each; a bucket per combination answers "how many images are of
  infected samples" with nothing, which is the question holding them apart exists for. The rows then
  add to more than the images counted, so the percentages stay a share of the *images* and a line
  says why. Which kind a key is comes from `Breakdown.multi`, off the field table, never from what
  it returned; `groups_of` is the one place either kind is read.
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
  (`scope.DEFAULT_SERIES` and `REPORTING_SERIES`). `by_series` returns the slug rather than the
  display label, a value counted having to be one `--series` accepts; and unlike a variant a series
  is a closed vocabulary, so `--series refrence` raises — a typo, not a request for nothing.
- **A command is not offered a series it could only refuse.** `other` has neither directory, so
  anything that writes, measures or exports label data takes `layout.LABELLED_SERIES`; only the
  three that describe what is on disk take `SERIES` — `stats`, `export-metadata-csv` and
  `export-files`, which copies each image and whatever label pairs with it. Refusing at the argument
  answers before a scan rather than after one
  (`test_a_command_that_acts_will_not_take_other_at_all`), and **`all` therefore means "every series
  this command covers"**, expanded per command rather than passed on
  (`test_all_means_every_series_this_command_covers_not_every_series`).
- **The reprojection is fibsemOS's, reimplemented rather than imported.** Its software wrote both
  the overview and the coordinates, so what its own minimap draws with is the definition of where a
  lamella falls; `overview.py` restates it as FIB-766 left it (fibsem 0.5.2rc1), reading a move's
  height off the stage rather than the grid plane, which misplaces by up to 100 µm a lamella whose
  height was corrected after the overview. Importing it costs 1.9s and 1,255 modules, adds a second
  provider of `cv2` — the hazard § Environment warns about most — and would read a header
  `image_metadata.py` already parses. **The copy is held honest by fibsemOS's own answers** rather
  than by review: `tests/data/fibsem_reprojection*.json`, written by
  `scripts/capture_fibsem_reprojection.py`, carry every real position plus the branches no real
  overview reaches (`test_overview_matches_fibsem`); where a person placed the lamellae holds what
  those answers cannot (`test_each_overview_places_its_lamellae_to_within_tens_of_microns`).
- **Where `experiment.yaml` records no position, a lamella is placed where most of its images were
  taken** (`test_a_lamella_is_placed_where_most_of_its_images_were_taken`): of the 512 lamellae with
  both, 510 agree. **An overview is known by its field of view as well as its name**
  (`test_an_image_as_wide_as_a_grid_is_an_overview_whatever_its_name`): no lamella's own image
  reaches 200 µm, and nothing beside them falls between that and 500. An Arctis header the
  microscope saved takes fibsemOS's Arctis geometry and SPECIMEN frame, which no file here confirms
  but placements by eye can; no file says which grid a lamella was on, so every overview marks it.
- **A milling task is matched by `tasks.py`, and there is no separate stage axis**, `stage-name`
  already answering "which directory". The directory is tried before the filename and is the only
  thing resolving a name Autolamella spells two ways (`Milling/Fiducial/` for `Mill Fiducial`); the
  *lamella* directory is never offered, or a petname sharing a word with a task claims every image
  below it, and a name two tasks claim is dropped rather than guessed. **None is a real answer** for
  over half these images. **Timestamps cannot do this**: the windows in `history` are sparse rather
  than tiling the session, so `ref_MillRough_start_eb.tif` falls inside the `SetupLamella` one.
- **Counting is split between the tab and the commands, not moved.** Choosing a training set means
  looking at what survives a filter, and that loop cannot run through a command that prints and
  exits; `stats` keeps the whole-collection sweep and `quantify-labels` the per-lamella table.
- **Opening the Export tab reads the headers and nothing else.** Under a millisecond an image,
  and it is what every acquisition filter and breakdown is made of. Offering them greyed out until
  a button was found and pressed was a control that did nothing with no way to tell why.
- **Opening a label is asked for, never assumed.** An order of magnitude dearer than a header, and
  it answers a different question: what is *in* the label. So the summary shows what a record knows
  about itself and says plainly that the rest needs Measure labels. **Measured over what the
  filters match**, not the collection, so widening a filter asks again for what it brought in. One
  pass answers three questions at once, for the reason `_remember_label_states` gives;
  `classify_label` is the shared rule and `_label_empty` the shared cache.
- **What a measured number means is set beside the breakdown, not among the filters.** The crack
  threshold and the area bin resolution change how a measurement is *read*, not which images are in
  play. Changing the threshold re-measures at once rather than spending the Measure button again:
  `read_label_pixels` caches component pixel counts *without* the threshold, so a new one costs
  arithmetic rather than a second pass.
- **A filter is added, not permanently shown**, for the reason `FilterRow` gives. A row carries
  the property it is on, and a property already in one row is not offered in another, which stops
  two rows disagreeing. **A few properties are counted by but never filtered on** — those running
  to hundreds of values, and those that barely vary — named by `UNFILTERABLE`; they stay
  breakdowns, and the command line still filters on them.
- **A stage is filtered on by its name, and counted on by its path.** `stage` names a folder
  (`project / lamella / adaptive_polish_<timestamp>`), which finds one run again but is useless to
  choose from; `stage-name` is that with the timestamp removed by `layout.stage_base_name`. **What
  names no milling stage counts under `LAMELLA_ROOT`**, and **a run counts under the task it aligns
  for**, the directory its `Alignment/` folder sits in — Autolamella naming a run after the lamella,
  task and time, which is otherwise a row per run. The timestamp pattern lives in `layout.py`, the
  scanner reading it to order stages and this to remove one.
- **Series is the one filter that does not open on everything.** Every other control starts wide
  and narrows towards an export; this one starts on Adaptive Polishing unless the settings say
  otherwise, a tab *silently* counting the rest disagreeing with `create-training-csv`. A dedicated control rather than a derived
  filter row, and checkboxes rather than a dropdown, an export covering two series being a real
  request. The metadata CSV names the series of every row, and `task` beside it: a row not saying
  which it came from cannot be read.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than *corrected*: the tab is narrowed towards an
  export rather than opening on one and hiding the archive. The cost is that an image somebody
  decided against can reach a training set, so the summary line counts those
  (`test_a_deselected_image_that_would_be_exported_is_counted`).
- **Both modalities means side by side, not mixed together**
  (`test_both_modalities_put_the_two_beams_on_one_row`): a selection holding both writes `sem_image,
  sem_label, fib_image, fib_label`, one row per acquisition, where one naming a single beam writes
  `image, label`. A cell is blank where that beam has no image or no label worth pairing with one,
  and a row survives only if at least one beam has a label — the empty-label rule drops a one-beam
  row and empties a cell where there are two (`test_an_empty_label_is_not_paired_with_its_image`).
- **What makes an SEM and a FIB image the same acquisition is stated once**, in
  `frames.group_by_acquisition`, which says why a stem will not do it. **What pairs them depends on
  the era of the filename**: Adaptive Polishing images match on an acquisition number within one
  stage directory, reference and alignment ones on the stem with `_eb`/`_ib` removed
  (`layout.acquisition_stem`). **The train/validation split groups by the acquisition**: split a
  paired row and the model is scored on a scene it has seen.
- **An export can also copy the files out**, state files and project metadata included, which is
  what makes the copy a collection rather than a pile — `copy_export.py` states the shape. Two
  checks are not its: free space is estimated first, and a destination inside the folder being
  copied from is refused, it being about to walk into its own output.
- **`check-training-csv` reads the header rather than assuming it.** Any column called `image` or
  `label`, or ending in `_image` or `_label`, holds a path, so one command checks either shape above
  and files written before both. A blank cell records nothing, so it cannot have gone missing.
- **A training export replaces nothing unasked, and leaves a manifest of what it was made from.**
  Every file it would write is checked before a label is opened
  (`test_an_existing_validation_file_stops_the_export_before_anything_is_written`): a split
  replaced by accident is gone with the seed that made it. The manifest is written last, so it
  names only files written whole (`test_the_manifest_counts_what_the_files_hold`).

## History

- **JSONL rather than YAML** buys three things a record needs and a control file does not — a
  one-write append, a union of two logs, and a damaged line costing only itself.
- **A label is digested over its pixels, never its file bytes**, or `compress-labels` would report
  each run as thousands of edits; `pixel_digest` states what that costs at the edges.
- **An entry's id hashes its whole body, `at` and `batch` included**, so it is an identity rather
  than a fingerprint of the change: two runs of one operation get two ids. That is what a union
  needs — entries travel by file copy, keep their id and combine exactly once, uncoordinated.
  **Idempotence comes from elsewhere**: `plan_baseline` skips targets already recorded.
- **A log tolerates what `state.py` refuses**: a control file must be understood completely before
  anything acts on it, a record must not lose four thousand entries over one bad line. Forward
  compatibility is append-only plus that tolerance.
- **What a kind carries is a row in `history.KINDS`**, as a metadata field is a row in
  `metadata.FIELDS`, and for the reason stated there.
- **No name is recorded unless somebody states one.** There is no fallback to the account running
  the command: an account name is not a chosen identity, and **a log lives inside its project**, so
  depositing a collection deposits its logs — a guess written there is a name nobody agreed to
  publish. An entry with no stated author carries neither field.
- **Attribution is a record, not an authentication**: `by_source` says how a name was arrived at,
  every value left being a deliberate statement. `declared` is the one nothing can verify, so
  `history` never prints it silently.
- **A dataset that arrives annotated is credited to whoever annotated it; the log, to whoever
  started it** — crediting *everything* to the annotator would replace one false record with
  another, since they never touched this machine. `--as` names the annotator, as `declared`; your
  own name comes from Settings, `--user` or `AM_SEG_USER` — stated, never collected.
- **Recording is always on, and the first write starts it.** `writing_recorder` puts
  `ensure_recording` in front of every change and nowhere else, so **looking never modifies** — a
  rule about call sites, not one the function can enforce. A baseline that fails is logged and the
  change goes ahead: a gap in the record costs less than the work. **The whole project is
  baselined**, or a savepoint would cover whatever happened to have been edited; `init-history` is
  then the catch-up that makes a record start complete, not a switch.
- **`init-history` covers every series**: one covering only Adaptive Polishing would leave the rest
  unrecorded, with no second chance at a baseline.
- **`AM_archive` is reserved and skipped at `subdirectories_by_name`**, the one function every
  traversal goes through; the survey's `rglob` excludes it by `directory_role`, one vocabulary
  rather than two lists. Missing it means a superseded version **reaches a training CSV**.
- **What is archived asks the log, not the state**, for the reason `archive.py` opens with. Unknown
  provenance counts as human work, which is every label in the collection today.
- **Restoring and merging are the ordinary save**, held to batch validation, the stale-file refusal
  and the protection of proofread work; only what is *recorded* differs, which is why
  `WriteProvenance` names the entry kind. A merge that copied the file **wrote over proofread work
  unasked**.
- **A savepoint is taken explicitly and only explicitly.** Not offered before a conversion, and not
  written by a training export: an automatic one recreates the gradient of small saves under a name
  that sounds deliberate, which is the whole of what a savepoint is not.
- **A project may only join a savepoint its record reaches back to.** A history beginning after one
  was taken means the project was not in that state, so joining would claim a moment the record
  knows nothing of, softening a savepoint into a label attachable to anything.
- **A savepoint reports what it can reach rather than guaranteeing it.** Guaranteeing means an
  invariant on the save path held for years and unverifiable when the savepoint is taken. Only
  *untouched machine output* is at risk, and `--close-gap` keeps those once. Reachability asks the
  label first and the archive last: asking the archive first had a revert offer to put back what was
  already there.
- **Nothing coalesces versions at write time**: that is deletion an operation performs on its own,
  so it belongs to `prune-archive`, which **leaves the log alone**. **The savepoints are its only
  policy**: a count of recent versions cannot know which states somebody named, so it could leave
  one pointing at pixels that are gone.
- **A round still open is never collapsed**, so nothing is removed until a savepoint has been
  taken. Cut **by position, not by time** — one written in the same millisecond as a save cannot be
  ordered against it.
- **The log always travels with a copy, whole; the archive only when asked.** A merge cannot
  establish an ancestor without the log. The archive is a cache, so `--with-archive` is opt-in —
  and **a merged log names versions this collection does not hold**, which the version list says.
- **The fork point covers only the projects the copy holds**, from `copy_export.projects_of`, the
  same question deciding whose describing files travel. **Refused rather than half-read**, unlike a
  log: the wrong ancestor discards work while reporting success.
- **A merge compares digests re-read from both trees, never the logs**, per `merge.py`.
- **A label and its workflow state are one decision.** Both resolve under the key
  `<project>/<target>`, so taking the copy's pixels takes its state with them. Deliberate: holding
  their label under a `corrected` claim made here would assert that somebody proofread pixels nobody
  here has seen. The resolver has one row per group because of this, and splitting the key would
  have to come first.
- **One resolution function serves labels, states and metadata fields**, so three kinds of change
  cannot answer one shape differently. **Timestamps decide nothing** — clocks differ between
  machines and a later one is routinely a recompression on real work — nor does any rule about which
  tree wins. A conflict is decided by a person or not at all — by name, or by a folder, project or
  lamella that the resolver or `--resolve` expands into names, the narrowest scope deciding
  (`test_the_narrowest_scope_decides_whatever_order_they_come_in`), so the merge still reads one
  decision per conflict — and **one stops the whole merge**. A scope holding no conflict is refused
  rather than ignored, or a mistyped path would decide nothing without saying so.
- **The fork point proves where a copy came from; the ancestor is the history the two logs share**,
  for the reason `merge.py` opens with. A merge verifies before it compares — path, declared id, and
  the tip looked for here — since a copy of a copy carries a plausible file naming an ancestor that
  was never here.
- **A version list is read, not opened.** Each row carries who, when, what produced it and
  `changed_fraction`, so nine rows saying 0.1% and one saying 34% name the one that matters; it is
  computed when a version is kept and cannot be had cheaply after. **And listing them may not be
  linear in the collection**, running at every arrow key — degrading gradually enough to be blamed
  on the images. Hence `versions_of`'s scandir by prefix and `VersionController`'s cached log.
- **Undoing a conversion is a batch operation**, so trying a newer model over proofread labels is a
  decision rather than a commitment: `revert` reads each label's history for the version the run
  replaced, **leaves alone** a label edited since, and refuses the whole batch where a group is
  corrected and nobody said so. It does not keep the model's untouched guess, which is regenerable
  and whose maker is recorded — seeing it again is a re-run, not a reconstruction.
- **A label directory rename is recorded, because an orphaned history cannot be reattached.** A
  group is keyed by its label's path, so `scripts/rename_legacy_label_dirs.py` and
  `scripts/share_variant_labels.py` change the key of everything they move. A state file survives
  this by being re-declarable by hand; a history does not, so both scripts write a `renamed` entry
  per file through `Recorder.record_directory_rename`.
- **Two ways a group's identity is lost quietly, neither of them guarded.** Splitting a shared label
  directory into per-condition ones changes every key exactly as a rename does, and unlike a rename
  no script records it. And a case-only difference is one file on this filesystem and two targets in
  a log. Both orphan the history of everything they touch.

## Environment

- **A save refuses a label whose file changed since it was opened.** Proofreading holds a
  lamella's edits in memory until the reviewer saves, leaving minutes to hours in which a second
  window, the command line or another reviewer on a shared archive can rewrite a label file. So
  the size and modification time are recorded on open and compared before writing, and a mismatch
  stops the entire batch rather than the one file: which version is wanted is not a question this
  package can answer.
- **Run the suite from the repository root, in `am_segmentation_manager` or
  `am_segmentation_manager_ml`.** Two ways to get this wrong, only one of which announces itself.
  Run `pytest` from *outside* the repository and there is no `pyproject.toml` above the working
  directory, so rootdir becomes that directory, `testpaths` never applies, and pytest recurses into
  everything below it — from `$HOME` that means importing every `.py` under the data archives,
  which looks exactly like a hang. Run it in conda's `base` and the imports resolve to a different
  stack; that one is caught before any test runs, and the refusal names an interpreter to run
  rather than an environment to activate (`test_the_refusal_names_a_command_that_can_be_run`),
  `conda activate` being a shell function absent from the shells it is read in. The Qt binding is
  the part of the wrong stack the suite can name — [`display.md`](display.md) § The binding.
- **The reference collection is the live archive, and there is no disposable copy of it.**
  `<collection>/` — a couple of hundred gigabytes covering
  every layout era from pre-Autolamella to 2026-05, and still being added to, so any count of it
  written down here is wrong by the time it is read. Point the smoke test at it with
  `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.
  **Never write to it.** Not a test, not a script, not an exploratory one-liner: nothing may create,
  move, overwrite or delete anything inside it without permission asked for and given for that
  specific change. It holds thousands of proofread labels that cannot be reconstructed, and the copy
  that used to absorb mistakes is gone — which is why this entry names real data at all. Read it,
  and copy *out* of it. `test_reference_collection.py` snapshots every file before and after;
  nothing enforces this for anything else you run.
- **`matplotlib` is in the `gui` and `all` extras rather than only `plots`**: against the OpenCV
  and Qt those already install, its size is not what makes an environment large. Why figures are
  never built through `pyplot` is `plots.py`'s own docstring.
- **Every `cv2` hazard is a provider conflict**, which is why `environment.yml` pins `nodefaults` —
  a mixed conda-forge/Anaconda environment paired Anaconda's OpenCV with conda-forge's protobuf so
  that `cv2` would not load at all. **The second environment is pip-first for the same reason**:
  albumentations, which the vendored preprocessing imports, requires `opencv-python-headless`, a
  third provider no ordering of install commands avoids, so `environment-ml.yml` takes only python,
  pip and the test tools from conda and the whole stack from the `[ml]` extra. That also lets the torch build be
  chosen per machine (`AM_SEG_TORCH_INDEX_URL`). **It is a superset, not a peer**: the curation
  environment is the *lighter* of the two, and splitting them the other way forces an environment
  switch mid-workflow.
- **Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
  `ml/environment.py` answers "is inference possible here" without importing it, as `io/image.py`
  does for the image stack. **`ml/preprocessing.py` is a verbatim copy of AM_training's module**,
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
- **Which prediction becomes a label is asked for, and the preview says which was used.** An image
  can hold am-seg's own and the microscope's; `prediction_precedence` takes the source chosen,
  am-seg unless told otherwise, and the other only where it is missing. That fallback is counted
  before the button is pressed and reported in the preview rather than asked about, the preview
  being the one question a conversion already asks
  (`test_the_preview_says_how_many_images_use_the_other_source`).
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
