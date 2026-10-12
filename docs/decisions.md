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
- **A batch of reads that holds the GIL goes to worker processes, not threads.** Parsing a header
  is mostly Python: eight threads read one in 0.36 ms, six processes in 0.09. `parallel.py` keeps
  one pool for the session, started by *spawn* so a worker inherits nothing the window imported
  (`test_a_worker_imports_nothing_the_window_did`), and logs again what a worker logged, or a
  batch's warnings would never reach `gui.log` (`test_a_worker_warning_reaches_this_process`).

## Filesystem and identity

- **An image directory is `<family>[_<variant>]`, split at the first `_`; a family shares `<family>_labels`,
  and the project's `AM_families.yaml` may move a condition into another.** A first label goes where the family
  looks afterwards (`test_resolution_does_not_change_once_the_first_label_is_written`). Corrections reach
  `layout.py` as a required argument, and a damaged file is never read as none: a scan indexes by name, writes
  refuse (`test_a_label_group_is_refused_while_the_file_is_damaged`). **One private directory stands in for its
  family**, several being a reported conflict (`test_several_directories_for_one_family_are_a_conflict`), and **none
  serves two families**, as a base moved out of its family would (`test_two_families_never_read_one_label_directory`).
  **Only label directories are renamed**, a correction going in that file: merge pairs a copy by relative path, and
  manifests, CSVs and the history name image paths; fibsem 0.5.2 reopens from `experiment.yaml`, naming no image folder.
  **A lamella's section replaces the project's entries there** (`test_a_lamella_entry_replaces_the_projects_there_only`),
  in the one file, not one per lamella: merge, a copy and the history each carry a project's corrections whole.
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
  them the state key, and would read meaning into a variant. `resolve_labels_directory` gives a
  condition its family's directory and nothing rewrites the filename under it; such data is
  renamed by hand.
- **A variant means nothing the package reads.** `sem_5kv`, `sem_defocus`, `sem_lowmag`
  all work with no code change, authoritative imaging parameters coming from the image headers. A
  variant may not be named `labels`, `predictions`, `plots`, `deselected` or `corrected`: each is
  another directory's role, so `layout.py` refuses it (`test_reserved_variants_are_rejected`) and
  `inspect-layout` reports the directory as matching no role.
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
- **A marker that will not read as YAML never stops a scan, and never exports as blank.** The
  project is indexed without that file and keeps the message on `Project.metadata_problems`;
  `inspect-layout` reports an `unreadable-yaml` error, and both metadata exports refuse the project,
  since a blank field reads as one not yet filled in
  (`test_the_metadata_exports_refuse_a_project_with_an_unreadable_file`). Nothing rewrites the file.
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
  `resolve_labels_directory`, which lets a condition's own directory stand in for its family's. Doing so would
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
- **A loose frame is ordered by its header's acquisition time, across its whole series**, for the
  reasons `ImageRecord.acquired_at` and `_in_acquisition_order` give. **mtime is worse than
  useless**, the collection having been copied. Ties go to `alignment_run_order`: a run takes
  several images inside the whole second a header records.
- **Indexing opens an image only in the scanner, and only a loose one**, for the beam its filename
  may not state and the time it never does — one memoised read. Discovery opens nothing: it asks
  `holds_loose_images`, which reads filenames. **It is the largest single cost of a scan**, most
  images in this collection being loose, so their headers go to worker processes in batches
  gathered across projects (`test_a_scan_hands_its_headers_over_together`) — on
  `build_loose_image_sets`'s own predicate, so nothing is opened that it would not look at. Read a
  stage at a time, most batches were too small to be worth sending to a worker.
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
- **A ticked lamella covers only the chosen series, never a fallback.** What is ticked is the whole
  of what an operation touches; falling back per lamella, as navigating does, would predict reference
  images of every lamella without Adaptive Polishing ones, with the Adaptive Polishing model. So
  `CurationController.batch_series` is one for the collection, and a lamella holding none of it is
  counted on the reach line (`test_a_ticked_project_covers_only_the_chosen_series_of_every_lamella`).
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
- **Whether a group is corrected is asked of its state file as a label is written, never of a path.**
  A record keeps what the scan saw, and a scan reads a damaged file as all selected, so conversion,
  compression and every save ask `declared_states` (`test_a_damaged_state_file_stops_a_conversion_over_corrected_labels`).
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
  of it happened"; each write checks again just before its rename (`test_a_save_leaves_a_label_rewritten_while_it_ran`).
  That is what lets `plan_transition` and `apply_plan` be separate — previewed decisions, fresh check.
- **Every label write goes through `io/atomic.py`, including compression, and so does every YAML
  file and `AM_metadata.yaml` edit.** It restores the destination's own permissions before the
  rename, `mkstemp` creating owner-only: a label kept `0o600` on a shared archive is as good as lost
  (`test_a_committed_file_keeps_the_destination_permissions`). Compression decides *after*
  encoding, so `write_atomically` takes a `commit_if` predicate. **The rename trusts the
  temporary**, and OpenCV reports a failed write by returning False: so `save_label_image` refuses
  pixels a uint8/uint16 file cannot hold exactly, and raises unless the file reads back as asked
  (`test_a_failed_write_through_a_temporary_leaves_the_label_as_it_was`).
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
  what is drawn is not recomputed from the records. `KnownLabels` keeps what each label holds, one
  instance shared by the Curate, Label and Export views; the frame cache holds each frame as a
  composed image *and its label* keyed by position; and the Export tab holds measurements. A record
  named without clearing them draws as it did before the file existed
  (`test_a_written_label_changes_what_the_timeline_reports`). `attach_written` and
  `ExportController.forget_labels` drop them together, and a save on the Label tab forgets what it
  wrote for every view (`test_forgetting_a_label_reaches_every_view_sharing_it`), its measurements
  included (`test_a_save_leaves_no_measurement_of_the_label_it_replaced`); the frame cache
  goes whole, being keyed by position rather than by what made it, and a label and its measurement
  are **not keyed alike** (`test_a_replaced_label_is_read_again_rather_than_remembered`). A scan
  still finds what changed the files from outside, which is what *Scan* is for.

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
  `image_metadata.FIELDS`: adding a field is one row, its heading and place on Add data included,
  rather than an edit in six modules, and adding a header *dialect* is a column rather than an
  extraction block per field. **Nothing names a
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
  spanning decades.
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
  lamella falls; `overview.py` restates it as FIB-766 left it (fibsem 0.5.2), reading a move's
  height off the stage rather than the grid plane, which misplaces by up to 100 µm a lamella whose
  height was corrected after the overview. **Importing it would replace only the arithmetic**:
  fibsemOS projects from its own `FibsemImage`, which an overview the microscope saved does not
  carry, so the header parsing, the Arctis defaults and every refusal would stay here, and `gui`
  would gain fibsem's pandas, scipy and scikit-image to reach twenty lines. **A TESCAN overview is
  refused** (`test_a_tescan_overview_places_nothing`): fibsemOS derives that stage separately, and
  no TESCAN header has been available to check a restatement against. **The copy is held honest
  by fibsemOS's own answers** rather than by review: `tests/data/fibsem_reprojection*.json`,
  written by `scripts/capture_fibsem_reprojection.py`, carry every real position plus the branches
  no real overview reaches (`test_overview_matches_fibsem`); where a person placed the lamellae
  holds what those answers cannot (`test_each_overview_places_its_lamellae_to_within_tens_of_microns`).
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
  a button was found and pressed was a control that did nothing with no way to tell why. **It reads
  none the scan read**: every loose image is opened for its beam and time, and the `ProjectIndex`
  keeps the rest (`test_the_export_tab_reads_only_headers_the_scan_did_not`).
- **Opening a label is asked for, never assumed.** An order of magnitude dearer than a header, and
  it answers a different question: what is *in* the label. So the summary shows what a record knows
  about itself and says plainly that the rest needs Measure labels. Choosing the label filter or a
  label-state breakdown asks too, and the labels are read behind the counting dialog
  (`test_choosing_a_label_filter_reads_the_labels_behind_the_dialog`) **and measured while open**,
  the pixels being counted in the read that says whether there are any, so no breakdown opens them
  again (`test_reading_the_labels_for_the_label_filter_measures_them`). **Measured over what the
  filters match**, not the collection, so widening a filter asks again for what it brought in. One
  pass answers three questions at once, for the reason `_remember_label_states` gives;
  `classify_label` is the shared rule and `KnownLabels` the shared cache — of whether a label is
  blank, never of whether it was measured.
- **What a measured number means is set beside the breakdown, not among the filters.** The crack
  threshold and the area bin resolution change how a measurement is *read*, not which images are in
  play. Changing the threshold re-applies it at once to every label measured, those the filters hide
  included, and opens none: a `LabelMeasurement` carries its component pixel counts, so a new one
  costs arithmetic (`test_a_new_threshold_reaches_labels_the_filters_hide`).
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
  otherwise, a tab *silently* counting the rest disagreeing with `create-training-csv`
  (`test_the_series_control_opens_on_adaptive_polishing_alone`). A dedicated control rather than a
  derived filter row, and switches rather than a dropdown, an export covering two series being a real
  request. The metadata CSV names the series of every row, and `task` beside it: a row not saying
  which it came from cannot be read.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than *corrected*: the tab is narrowed towards an
  export rather than opening on one and hiding the archive. The cost is that an image somebody
  decided against can reach a training set, so the summary line counts those
  (`test_a_deselected_image_that_would_be_exported_is_counted`).
- **Both modalities means side by side, not mixed together** (`test_both_modalities_put_the_two_beams_on_one_row`):
  both write `sem_image, sem_label, fib_image, fib_label`, a row per pairing of an acquisition's SEM
  and FIB images — one per acquisition where each beam has one, as *One image per label* leaves a
  beam of one family; one beam writes
  `image, label`. A cell is blank where that beam has no image or no label worth pairing, and a row
  survives only if one beam has a label — the empty-label rule, **which takes a label that will not
  read with it**, blank being unknowable, drops a one-beam row and empties a cell where there are two
  (`test_an_empty_label_is_not_paired_with_its_image`).
- **What makes an SEM and a FIB image the same acquisition is stated once**, in
  `frames.group_by_acquisition`, which says why a stem will not do it. **What pairs them depends on
  the era of the filename**: Adaptive Polishing images match on an acquisition number within one
  stage directory, reference and alignment ones on the stem with `_eb`/`_ib` removed
  (`layout.acquisition_stem`). **No split divides an acquisition, nor by default a lamella**, whose
  frames are near-duplicates (`test_no_group_is_on_both_sides_of_the_split`).
- **Validating on corrected labels alone gives the validation list those alone give**
  (`test_adding_selected_labels_leaves_the_validation_list_as_it_was`); another label trains, or is
  left out where its group validates, never trained beside it. **An export kept to an earlier one's
  split moves no group across it** (`test_a_later_export_keeps_every_lamella_on_the_side_it_was`),
  a validated group gaining what was proofread since: nothing earlier trained on it.
- **An export can also copy the files out**, state files and project metadata included, which is
  what makes the copy a collection rather than a pile — `copy_export.py` states the shape. Two
  checks are not its: free space is estimated first, and a destination inside the folder being
  copied from is refused, it being about to walk into its own output.
- **An export replaces nothing unasked.** Every CSV or chart it would write is checked before a
  header or a label is opened, and one already there is refused unless `--overwrite` or the window's
  save dialog said otherwise (`test_an_existing_validation_file_stops_the_export_before_anything_is_written`,
  `test_cli_stats_refuses_to_replace_a_csv`, `test_cli_refuses_an_existing_chart_before_counting`);
  one appearing mid-write is refused as the file lands, and the command line reports it the same
  way (`test_cli_reports_an_output_that_appears_while_it_is_written`): an export is something a
  person may have kept, and a split replaced by accident is gone with the seed that made it. A training export also leaves a
  manifest, written last so it names only files written whole
  (`test_the_manifest_counts_what_the_files_hold`), and naming no folder of the machine: an export
  is deposited and moved, so the collection is said from the manifest's own folder and a folder
  named when training is taken before it (`test_a_manifest_names_no_folder_of_this_machine`).
  Each image's metadata is written with the lists, frozen at the export: a model is scored on parts
  of a set after the collection has been corrected, and on a machine that does not hold it
  (`test_every_image_is_listed_once_with_the_side_it_is_on`).

## History

- **JSONL rather than YAML** buys three things a record needs and a control file does not — a
  one-write append, a union of two logs, and a damaged line costing only itself.
- **An entry's id hashes its whole body, `at` and `batch` included**, so it is an identity rather
  than a fingerprint of the change: two runs of one operation get two ids. That is what a union
  needs — entries travel by file copy, keep their id and combine exactly once, uncoordinated.
  **Idempotence comes from elsewhere**: `plan_baseline` skips targets already recorded.
- **A log tolerates what `state.py` refuses**: a control file must be understood completely before
  anything acts on it, a record must not lose four thousand entries over one bad line. Forward
  compatibility is append-only plus that tolerance.
- **No name is recorded unless somebody states one.** There is no fallback to the account running
  the command: an account name is not a chosen identity, and **a log lives inside its project**, so
  depositing a collection deposits its logs — a guess written there is a name nobody agreed to
  publish (`test_no_name_is_resolved_when_nobody_stated_one`). An entry with no stated author
  carries neither field.
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
- **A version that cannot be kept is not written over.** A failed archive copy stops the save at
  that label, keeping what was written before it (`test_a_version_that_cannot_be_kept_is_not_written_over`);
  the copy lands whole or not at all. **One that will not decode is kept as its bytes**, nobody able
  to say it is not somebody's work (`test_a_label_that_will_not_decode_is_kept_as_its_bytes`).
  **A path Windows cannot reach refuses the whole batch before anything is written**, archived copies
  included (`test_one_label_windows_cannot_reach_refuses_the_whole_save`): that failure can be
  foreseen. Long paths are found to work by writing one, `LongPathsEnabled` being necessary but not
  sufficient.
- **A write that stops partway hands back what it wrote**, as a `progress.StoppedPartway`
  (`SaveStopped`, `ConversionStopped`). The window names those files on their records and refreshes
  whatever the failure, failing not being the same as changing nothing
  (`test_an_operation_stopped_partway_names_what_it_wrote_on_the_records`). An operation writing
  files owes the window this, or its partial work stays invisible until the next scan.
- **Restoring and merging are the ordinary save**, held to batch validation, the stale-file refusal
  and the protection of proofread work; only what is *recorded* differs, which is why
  `WriteProvenance` names the entry kind. A merge that copied the file **wrote over proofread work
  unasked**. **A merge reads every label it takes, and rehearses every state, before its first write**
  (`test_a_state_that_cannot_be_declared_stops_the_merge_before_any_label`): the copy's history comes
  with its labels, so a label skipped would leave the log naming pixels this collection never got.
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
  taken (`test_work_done_since_the_last_savepoint_is_never_collapsed`). Cut **by position, not by
  time** — one written in the same millisecond as a save cannot be ordered against it.
- **The log always travels with a copy, whole; the archive only when asked.** A merge cannot
  establish an ancestor without the log. The archive is a cache, so `--with-archive` is opt-in —
  and **a merged log names versions this collection does not hold**, which the version list says.
- **The fork point covers only the projects the copy holds**, from `copy_export.projects_of`, with
  each one's `AM_families.yaml` as the ancestor its families merge against, so a hand edit is seen
  (`test_the_fork_point_holds_the_families_as_they_stood`). **Refused rather than half-read**.
- **A label and its workflow state are one decision.** Both resolve under the key
  `<project>/<target>`, so taking the copy's pixels takes its state with them. Deliberate: holding
  their label under a `corrected` claim made here would assert that somebody proofread pixels nobody
  here has seen. The resolver has one row per group because of this, and splitting the key would
  have to come first.
- **One resolution function serves labels, states, fields and families**, so four kinds of change
  cannot answer one shape differently. **Timestamps decide nothing** — clocks differ between
  machines and a later one is routinely a recompression on real work — nor does any rule about which
  tree wins. A conflict is decided by a person or not at all — by name, or by a folder, project or
  lamella that the resolver or `--resolve` expands into names, the narrowest scope deciding
  (`test_the_narrowest_scope_decides_whatever_order_they_come_in`), so the merge still reads one
  decision per conflict — and **one stops the whole merge**. A scope holding no conflict is refused
  rather than ignored, or a mistyped path would decide nothing without saying so.
- **Undoing a conversion is a batch operation**, so trying a newer model over proofread labels is a
  decision rather than a commitment: `revert` reads each label's history for the version the run
  replaced, **leaves alone** a label edited since, and refuses the whole batch where a group is
  corrected and nobody said so. It does not keep the model's untouched guess, which is regenerable
  and whose maker is recorded — seeing it again is a re-run, not a reconstruction.
- **A label directory rename is recorded, because an orphaned history cannot be reattached.** A
  group is keyed by its label's path, so renaming its directory changes the key of everything in
  it. A state file survives this by being re-declarable by hand; a history does not, so whatever
  renames one writes a `renamed` entry per file through `Recorder.record_directory_rename`
  (`test_renaming_a_label_directory_keeps_its_history_reachable`).
- **Two ways a group's identity is lost quietly, neither of them guarded.** Splitting a shared label
  directory into per-condition ones changes every key exactly as a rename does, and unlike a rename
  no script records it. And a case-only difference is one file on this filesystem and two targets in
  a log. Both orphan the history of everything they touch.

## Environment

- **Run the suite from the repository root, with `uv run pytest`.** Two ways to get this wrong,
  only one of which announces itself. Run `pytest` from *outside* the repository and there is no
  `pyproject.toml` above the working directory, so rootdir becomes that directory, `testpaths`
  never applies, and pytest recurses into everything below it — from `$HOME` that means importing
  every `.py` under the data archives, which looks exactly like a hang. Run it with an interpreter
  outside the project environment — conda's `base` is the usual one — and the imports resolve to a
  different stack; that one is caught before any test runs, and the refusal names a command rather
  than an environment to activate (`test_the_refusal_names_a_command_that_can_be_run`). The Qt
  binding is the part of the wrong stack the suite can name — [`display.md`](display.md) § The
  binding.
- **The reference collection is the live archive, and there is no disposable copy of it.** A couple
  of hundred gigabytes covering every layout era from pre-Autolamella to 2026-05, and still being
  added to, so any count of it written down here is wrong by the time it is read. It is named by
  `AM_SEG_REFERENCE_COLLECTION=<path> uv run pytest -q` rather than by a path written down here, so
  the rule travels to whoever holds a collection instead of describing one machine.
  **Never write to it.** Not a test, not a script, not an exploratory one-liner: nothing may create,
  move, overwrite or delete anything inside it without permission asked for and given for that
  specific change. It holds thousands of proofread labels that cannot be reconstructed, and the copy
  that used to absorb mistakes is gone. Read it, and copy *out* of it.
  `test_reference_collection.py` snapshots every file before and after; nothing enforces this for
  anything else you run.
- **One environment, every dependency from PyPI, and `uv.lock` holding the versions tested.** Every
  `cv2` failure this package has met was two providers of one module — conda's `opencv` beside
  pip's, or `opencv-python` beside `opencv-python-headless` — so every
  extra names the headless build and nothing comes from conda. **Inference and training are extras,
  not environments**: nothing outside `ml/` imports torch
  (`test_torch_is_not_loaded_outside_the_module_that_runs_it`), nor AM-model-training, MONAI or
  mlflow at all, which also set process-wide state
  (`test_the_training_stack_is_imported_only_inside_ml_functions`), so an environment without them
  is a subset rather than a peer. `train` takes the torch `ml` or `ml-cuda13` chose and is installed
  beside each (`test_every_inference_build_is_installed_with_the_training_stack_somewhere`); am-seg
  still predicts, AM-model-training's `am-infer` never calling `model.eval()`.
  `required-environments` locks for Windows, Linux and Apple silicon on
  each supported Python — but a dependency lacking a wheel for one of them locks anyway and is
  built from source there, which on Windows means a C++ compiler. Nothing checks this yet.
- **Windows and Linux take torch built for CUDA 12.6, not PyPI's.** PyPI's Windows torch is
  CPU-only, and its Linux torch needs a driver for CUDA 13 (580 or later); 12.6 runs on any driver
  from 525, the shared A100 workstations' 535 included. A lock cannot follow a machine's driver, so
  a GPU needing a newer build is an extra: `ml-cuda13`, that set with torch from the CUDA 13 index,
  in `conflicts` so uv refuses it beside `ml` (`test_the_two_inference_extras_install_the_same_package_set`).
- **A wrong build is said before a run, not found one step into it**, and *wrong* is two things: a
  driver too old leaves `torch.cuda.is_available()` false, a card too new leaves it **true** with
  every launch failing. So `cuda_problem` also compares each card's compute capability with
  `torch.cuda.get_arch_list()` (`test_a_card_newer_than_every_kernel_is_said_so_and_auto_falls_back`),
  reading a build that names no architectures as no problem; either way `check_environment.py`
  fails and predicting on `auto` takes the CPU with a warning. A GPU named is refused, and so is
  training, days long on a CPU nobody named (`test_work_too_long_for_a_cpu_is_refused_where_auto_would_fall_back`).
- **The version is written once, in `pyproject.toml`, and set with `uv version <X.Y.Z> --no-sync`**,
  which relocks as well; without `--no-sync` it re-syncs `.venv` down to no extras
  (`test_the_lock_records_the_version_pyproject_states`). Between releases it is the next one's
  `.dev0`, so a record written from `main` never passes for a release. Each file this package
  writes carries its own `FILE_VERSION`, and changing one is a breaking change the changelog names
  (`test_a_file_format_change_is_a_breaking_change`).

## Inference

- **How a model is run is its generation's, never a file's**, the division `ml/model.py` sets out:
  the description beside a checkpoint names only the generation, undescribed meaning `1.0`, not
  the latest (`test_a_description_naming_anything_but_the_generation_is_refused`). Generation 1 is
  Adaptive Milling's, which the microscope predicts with, frozen
  (`test_generation_one_s_settings_are_the_ones_predictions_were_made_with`) and held to its own
  answers (`test_generation_one_preprocesses_as_adaptive_milling_does`). The class and channel
  counts are read from the weights and must be the generation's. **`decoder_attention_type="scse"`
  is not part of these models** — the training call passes it, smp's `FPN` swallows it into
  `**kwargs`, the checkpoints hold no attention tensors, and a plain FPN strict-loads every one.
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
  `label_groups.primary_record`, the condition a family's `primary:` names, else its base, else its
  first variant by name — the image a training export keeps. Predicting the other conditions would
  spend inference on noisier pseudo-labels and removes a hazard rather than managing one: conditions
  carry identical filenames into a shared predictions directory, so predicting all of them has each
  write replace the last. **No file is planned twice**: a family whose labels are split is several
  groups, met by one prediction from its primary (`test_a_family_in_conflict_is_predicted_once_per_destination`).
- **Which prediction becomes a label is asked for, and the preview says which was used.** An image
  can hold am-seg's own and the microscope's; `prediction_precedence` takes the source chosen,
  am-seg unless told otherwise, and the other only where it is missing. That fallback is counted
  before the button is pressed and reported in the preview rather than asked about, the preview
  being the one question a conversion already asks
  (`test_the_preview_says_how_many_images_use_the_other_source`).
- **Whether to offer the predict button is decided by locating torch, not importing it**, for the
  reason `ml_stack_present` gives; a missing or broken stack is reported by the child, which has to
  import it anyway. **The device box does not import torch either**, opening on `auto` and `cpu`,
  true everywhere, and gaining the machine's own from a child process the first time it opens.
  Editable, since `resolve_device` passes a device it does not recognise through unexamined.
- **A run sees one GPU, numbered as `nvidia-smi` numbers it.** On a shared workstation the index
  a colleague reads in `nvidia-smi` must be the card a run takes, so `number_as_nvidia_smi` sets
  PCI order, and `resolve_device` pins the chosen card by UUID in `CUDA_VISIBLE_DEVICES` before
  CUDA starts — otherwise torch may open a context on GPU 0 whatever it runs on
  (`test_a_gpu_named_by_number_is_pinned_and_is_cuda_0_in_the_run`). **GPUs are read from
  `nvidia-smi`, not torch or `nvidia-ml-py`**: no dependency, no CUDA context on anyone's card, and
  cheap enough to refresh free memory each time the box opens
  (`test_opening_the_device_box_asks_torch_once_and_nvidia_smi_every_time`).

## Training

- **A run is a directory prepared whole, outside the collection, before anything starts.** Its split
  is the export's, never remade (`test_the_split_is_the_export_s_and_not_remade`), and AM-model-training
  is always given two lists, since left to split one it shuffles through lamellae
  (`test_the_arguments_always_name_a_training_and_a_validation_list`).
- **Running is a lock held, not a process id in a file**, which another program can come to own; the
  system gives a lock back however its process dies
  (`test_a_held_lock_is_a_living_process_whatever_the_host_is_called`), and the id is asked only
  where a filesystem offers no locks. A stop writes its request
  before ending the process, which is all that tells stopped from interrupted
  (`test_a_run_killed_without_a_request_was_interrupted`): a detached Windows process cannot be asked.
  **A run directory is started once** (`test_a_run_directory_is_started_once`), and `run_status` is
  the one answer to how a run am-seg started stands; one it did not start has no process here to ask,
  and counts as training only while its files are an hour fresh
  (`test_an_unfinished_run_written_to_lately_is_taken_as_still_training`). **A run says which GPU it
  is on while it trains, as `nvidia-smi` numbers it**, a pinned GPU being `cuda:0` in its own process
  (`test_a_run_says_which_device_it_is_on_while_it_is_still_going`).
- **The epoch a person marks as best is a file beside the run, kept apart from the run's own**,
  which stays computed; one function reads it and nothing deletes its weights
  (`test_the_epoch_marked_is_kept_whatever_else_is`), a mark that cannot be read stopping the
  deletion (`test_nothing_is_deleted_while_the_mark_cannot_be_read`).
- **Runs are compared in a table computed apart from its drawing** (`training_compare.py`), so the
  window, the CSV and `compare-training` hold the same numbers; a cell nothing was validated for
  is empty, a zero being a score (`test_a_run_with_no_epoch_in_the_range_has_empty_cells_not_zeros`).
  Both list runs through `list_runs` (`test_the_command_line_and_the_window_call_a_run_the_same`). **Every part of a set is shown beside how many images it holds, none hidden or marked**, images with no value being a part (`test_images_holding_no_value_are_a_part_of_their_own_and_the_last`).
- **A validation is a directory inside its run**, started once and read as a run is; each epoch's
  scores are written as it is done (`test_a_validation_that_fails_part_way_keeps_the_epochs_it_finished`). A run's own list has one such directory and no name, scored there after the run only for the epochs it did not score as it trained (`test_only_the_epochs_a_run_did_not_score_as_it_trained_are_scored`).
- **A confusion matrix is kept for each validation image**, so any part of a set is scored afterwards in
  AM-model-training's numbers (`test_per_image_scores_give_the_metrics_am_model_training_reports`): a
  run's own list as it trains, counted beside AM-model-training's validation and kept only where it adds up to
  its count (`test_scores_that_do_not_add_up_to_what_was_counted_are_not_kept`), which nothing can stop.
- **A run's log is shown from its end, each progress bar once.** Its first epoch is many minutes
  with no file changed but the log, where a bar is a line for every update
  (`test_a_bar_rewritten_many_times_is_one_line_as_it_last_stood`); the text is AM-model-training's,
  shown and not parsed for a state, which is `run_status`'s to give. A failure is explained only
  where it is recognised (`test_a_failure_not_recognised_is_given_no_advice`).
- **Models are looked at on images none of them trained on, their predictions held in memory.**
  Two exports split differently, so one run's validation image is another's training image
  (`test_an_image_one_of_the_runs_trained_on_is_not_offered`); what each image is of travels with
  the run, which outlives its export (`test_what_the_validation_images_are_of_is_kept_with_the_run`).
  Predicted as `predict` does, not as a run's validation scored it: what the model will do in use.
  No file, provenance or history entry is made (`test_a_sheet_is_written_and_nothing_else_anywhere`). A run's lists name its images in full, as AM-model-training needs; where the folder its record names is not on this machine both are read as under the collection scanned, one check a run and nothing rewritten (`test_a_run_whose_collection_is_gone_is_read_as_under_the_folder_given`), so one image is one path to every run (`test_an_image_trained_on_another_machine_is_still_not_offered`).
  An image drawn whose file is not here is left out before any model is loaded, and said once by the
  folder looked in rather than a path a tile (`test_images_that_are_not_there_are_not_predicted`).
- **Training is a detached process started as a module**: AM-model-training's workers re-import
  `__main__`, and a script there trains again (`test_a_start_runs_this_module_with_this_interpreter`).
