# Decisions

Why the software is built this way, and facts that are not derivable from code: the code
shows what happens, not what breaks if it changes. Work that is *not* built yet belongs in
[`04_future_work.md`](04_future_work.md). If something below becomes wrong, correct it in
place.

---

## Architecture

- **No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python
  controllers, which is what makes navigation, cache eviction and state expansion testable
  without a display server. It is the one structural rule of the UI; adding a tab is not a
  reason to bend it.
- **An operation belongs in the window when its input is the selection on screen; it stays
  on the command line when it is a sweep over the whole collection.** Naming a selection
  twice — once by ticking it, once as command-line filters — is where the two diverge and
  the wrong data is acted on. The Add data commands are in the window despite being
  sweeps: there the sweep *is* the collection and there is nothing to select. Each previews
  and then applies what it previewed, as two runs of the same function rather than one
  replayed — a preview held across a dialog describes a filesystem the user has had time to
  change.
- **A filter and a breakdown are one declaration in the Export tab.** They are the same key
  function used two ways — group by it, or keep the records whose group was chosen — so the
  filter lists come from `BREAKDOWNS` rather than a second table beside it, and cannot
  disagree with the table about what a value *is*: the filter used to offer a raw pixel
  size no row ever showed. Two consequences: "(not recorded)" is selectable, the only way
  to find images whose header would not read; and a measured property is not offered, since
  it cannot answer before Measure is pressed. Modality, workflow state and the label filter
  keep dedicated controls, being fixed questions rather than data.
- **`filtering.METADATA_FILTERS` stays as it was.** It is the command line's contract, where
  a filter is typed rather than picked and has to tolerate `30000` for a stored `30000.0`.
- **Counting split rather than moved.** The Export tab counts and cross-tabulates the set
  its filters describe, measuring labels when asked, because choosing a training set means
  looking at what survives a filter and that loop cannot run through a command that prints
  and exits. `stats` keeps the whole-collection sweep with its own scope selectors, and
  `quantify-labels` keeps the per-lamella table: a table of numbers per class per lamella is
  not something anyone acts on one row at a time, and a spreadsheet reads it better from the
  CSV.
- **What a measured number means is set beside the breakdown, not among the filters.** The
  crack threshold and the area bin resolution change how a measurement is *read*, not which
  images are in play, so a Filters box promising to narrow something is the wrong home.
  Changing the threshold re-measures at once rather than spending the Measure button again:
  `read_label_pixels` caches component pixel counts *without* the threshold, so a new one
  costs arithmetic rather than a second pass. `load_measurements` keeps the headers it reads
  for the same reason.
- **No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few
  lines with no dependency, over thousands of records rather than millions. pandas belongs
  here only once something genuinely needs it.

## Filesystem conventions

- **A label directory is `<image-directory>_labels`, a sibling of the image directory.**
  `layout.py` states the resolution order; what it cannot say is why that shape. It makes
  image-dir → label-dir a pure string function both ways with no lookup table, which is what
  lets one module hold the convention. Conditions share a label by default; giving one its
  own means creating the directory, not configuring anything. `sem/` is no special case: its
  private and shared names coincide, so one rule covers the plain layout and the
  five-condition one alike.
- **Image directories keep their Autolamella names.** We do not own `sem` and `fib`;
  renaming them would break reopening the project in Autolamella. Label directories are
  ours, so they are the only ones anything here renames.
- **Renaming pre-convention label directories is a script, not a command.** The package
  reads `labels`, `labels_fib` and `labels_FIB` indefinitely, and a new label lands in
  whichever directory an image set already has, so no project ends up holding two namings.
  That leaves cosmetic tidiness as the only reason to rename, over a fixed handful of
  datasets — so it lives in `scripts/rename_legacy_label_dirs.py`, where a layout that does
  not fit is handled by editing the script rather than by generalising a command.
- **The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`,
  `sem_lowmag` all work with no code change, because authoritative imaging parameters come
  from the image headers. A variant may not be named `labels`, `deselected`, `corrected` or
  `predictions` — those collide with the label naming rule, so `layout.py` validates it and
  `inspect-layout` reports a violation rather than mis-indexing.
- **A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`.**
  `init-project` writes the latter rather than fabricating the former, because
  `experiment.yaml` is Autolamella's file: writing one misrepresents provenance and could
  collide if Autolamella ever opens that directory. `protocol.yaml` looks like a third
  marker and is not usable as one: some genuine Autolamella projects have `experiment.yaml`
  without it.
- **`init-project` refuses a directory that holds projects of its own.** That is a
  collection, and a marker there makes every unmarked dataset beneath it a *lamella* of one
  project — several samples under one metadata record, invisible in the output.
  `--recursive` is one flag away and does the right thing, so refusing costs nothing.
- **A project's identity is its path relative to the scan root, not its name.** The
  reference collection holds two different projects whose name, `created_at` and recorded
  path in `experiment.yaml` are all identical — keying by name silently drops one.
- **A lamella is counted as `project / lamella`, and a stage one level deeper.** The same
  rule as above, applied to a table rather than an index: almost every project has an `01-`,
  so an unqualified row merges lamellae with nothing to do with each other — in the
  reference collection two projects each hold an `05-pet-mutt`, reported as one lamella of
  140 images. Stage names are worse, being either shared by every lamella
  (`adaptive_polish`) or a bare timestamp. `aggregate.summarise_lamellae` already keyed on
  the pair; the breakdowns now agree with it.
- **The lamella and stage levels are not always directories.** Older datasets collapse them;
  the missing level is synthesised in memory with the parent standing in, and the model
  records that it is synthetic. Data on disk is never reorganised to fit the newest layout.
- **`predictions/clean/` is indexed but can never become a label.** Adaptive Polishing's
  post-processing rewrites the intensity values, so they no longer index the segmentation
  classes. A label converted from one looks entirely valid — right dimensions, plausible
  shapes — and teaches a model the wrong classes, which is worse than having no label at
  all. This will not change upstream, so it is a property of `layout.py`
  (`is_convertible_prediction`) rather than a choice offered to the caller. It is still
  indexed, deliberately: otherwise a project holding only post-processed predictions would
  report "no predictions", indistinguishable from one the pipeline has never reached, when
  those need opposite actions. Conversion counts the two separately.

## Workflow state

- **State is declared in a file, not implied by location.** Every directory holding image
  and label directories carries an `AM_state.yaml` naming the groups that are not selected.
  A file's directory used to *be* its state, because the external annotation tool wrote an
  empty label for every image in its input folder and keeping images out of that folder was
  the only way to say "not this one". Nothing requires it now, and it was expensive: a state
  change was a batch of moves that could half-succeed, a label could exist in two state
  folders at once, and curating an image rewrote a path a training CSV had already recorded.
- **The label is the unit of state, and the key names the label.** An entry is keyed by the
  group's label path — `<label-directory>/<filename>`, relative to the state file — whether
  or not that label exists. Keying on the label rather than the image makes an image
  disagreeing with its label *unrepresentable* rather than merely detectable: the several
  conditions of one scene resolve to one key, so a change reaches all of them or none.
  Resolved rather than located, so the key survives annotation beginning — which also makes
  images sharing a label directory one group before any label exists, where they used to be
  independent until one appeared.
- **Anything not listed is selected.** An absent entry and an absent file give the same
  answer, so images dropped into a stage tomorrow arrive selected with no reconciliation —
  exactly as they did when the base directory meant selected, and most stages then carry a
  short file or none at all. The cost is that "reviewed and kept" cannot be told from "not
  yet looked at", which the folders could not tell apart either.
- **The three states are mutually exclusive, and both representations enforce it.** A mapping
  of key to state cannot express two states for one group, where a file grouping keys under
  state headings could. That is why the file is not the more readable-looking `corrected:` /
  `deselected:` lists. In memory, for the same reason, `ImageRecord` carries one `state` field
  rather than a `selected` and a `corrected` flag: two booleans have a fourth combination that
  means nothing, and every reader then has to spell out `selected and not corrected` and be
  trusted to get it right. One field makes the filter an equality test. Duplicate keys are
  rejected on read rather than resolved last-wins, since a silent winner is the same failure
  the mapping was chosen to prevent. Deselecting a group that was corrected discards the
  corrected claim; the proofreading itself survives in the label's pixels, which is what an
  export checks.
- **A stage is transcribed whole or not at all, and one that has not been is refused.** The
  two rules are the same rule. `migrate-state` leaves a stage alone if anything in it is
  ambiguous — images disagreeing with their label, or a file sitting in two state folders —
  because writing the groups that were fine would leave the rest unlisted, and unlisted reads
  as *selected*. For the same reason a state change refuses a stage that still keeps its
  state in folders. Promoting a deselected image into a training set is the one failure here
  with no visible symptom, so both paths are shut rather than guarded.
- **`migrate-state` is the only thing that reads a folder name as a state**, and it walks the
  filesystem itself to do it. The scanner does not look inside `deselected/` or `corrected/`
  at all, so a nested file is not indexed: it is *missing and reported* rather than present
  and wrong. That is also why nothing repairs a group any more — images cannot disagree with
  their label when one entry covers them all, so `repair-state` and the consistency machinery
  went with the folders.
- **The state folder names stay reserved**, so an unmigrated dataset reads as *missing* those
  images rather than as holding selected ones. Reading a deselected image as selected is the
  one migration failure that would reach a training set unnoticed.
- **A stale entry is reported, never repaired.** Declared state can disagree with the
  filesystem in ways location could not, and each needs a different action, so
  `inspect-layout` separates them: a stage whose folders still hold its state has not been
  migrated; nested files after migration want `--move-files`; an empty state folder is
  clutter; an entry naming a group that is not there was left behind by a rename or a
  deletion. That last one is kept rather than dropped — a group that vanished because a drive
  was not mounted must not have its curation deleted for it.
- **A recorded path cannot go stale, so `check-training-csv` only checks.** It existed to
  rewrite paths that curation had moved; nothing moves, so a path that does not resolve means
  the file is gone. What survives is the check and a `--prune` that drops the dead rows,
  because a CSV holds a particular train/validation split and losing a few images is not a
  reason to regenerate it.
- **A write is read-modify-write, and refuses a file that changed underneath it.** Two
  windows on one collection, or a window and a command line, would otherwise discard each
  other's curation with no error. Under folders that could not happen, two moves of different
  files never colliding; it is the one failure mode this design adds, so it is guarded twice.
  Every file the batch will touch is compared against what the plan read *before the first
  one is written*, so the answer is "nothing happened" rather than "some of it happened" —
  a batch spans as many state files as it does stages, and half a curation applied is worse
  than none. Each write then checks the same thing again for itself, which closes the window
  the first check leaves open. That is what lets `plan_transition` and `apply_plan` be
  separate: the decisions are the ones that were previewed, however long ago, while the
  safety check is taken fresh. It is not a transaction — a full disk can still stop a batch
  partway — so each stage's records are brought into step as that stage lands, and what is in
  memory describes what is on disk either way.
- **Every label write goes through `io/atomic.py`, including compression.** The helper exists
  for two things a hand-rolled `mkstemp` + `os.replace` silently loses. It restores the
  destination's own permissions before the rename, because `mkstemp` creates owner-only and
  OpenCV truncates rather than recreates — so a recompressed label kept `0o600` and became
  unreadable to everyone but its author, which on a shared archive is the same as losing it.
  And it takes a fingerprint, so a label something else rewrote mid-run is left as it now is
  rather than having the old pixels put back over it. Compression needs to decide *after*
  encoding — the saving cannot be known any other way — so `write_atomically` takes a
  `commit_if` predicate and returns whether it committed, which keeps that decision inside the
  window the rename closes instead of measuring first and replacing later.
- **A compression run that skips a file says which of the three reasons applied.** No saving,
  a dry run, and "somebody else rewrote it" are one `applied=False` otherwise, and only the
  third is worth acting on — so `stale` is its own field, reported by both front ends.
- **Whether a group is corrected is asked of the record, never of a path.** Every guard that
  protects proofread work — conversion, compression, label saving — reads the declared state
  the scanner put on the record. A path stopped answering that question the moment the files
  left their folders, and a guard that quietly stops firing is worse than none.
- **Overwriting a label is tiered by what replacing it would cost, not by one flag.** A run
  that writes nothing has to say why, or it cannot be told from a broken one. So conversion
  sorts destinations three ways: an empty placeholder costs nothing to replace (nothing was
  in it); a pseudo-label holding annotation is regenerable machine output, replaced once the
  user confirms; a label whose group is corrected is refused, and no argument to
  `plan_conversions` reaches it. That last tier is stricter than `compress-labels`, which
  *can* be confirmed for corrected labels: compression preserves the pixels exactly, where
  conversion replaces proofread work with a machine guess. A dialog is the right guard for
  the first and too weak for the second, so re-converting a corrected label means setting
  that group back to selected first.
- **That is also why compressing is not scoped to a selection.** Preserving the pixels exactly
  means there is no label it should be withheld from, so which ones to shrink is not a decision
  worth a control — and a per-selection button leaves uncompressed exactly the labels nobody
  happened to tick, which is the opposite of what the operation is for. In the window it acts
  on the whole scanned folder, from Add data; the command line keeps its `--project` and
  `--lamella` filters for the rare case where one project is the whole question.
- **The Export tab opens on the whole collection, so it counts what that admits.** The
  workflow-state filter starts on "all" rather than on *corrected*, matching the rest of the
  tab: it is narrowed towards an export rather than opening on one and hiding the archive.
  The cost is that an image somebody decided against can reach a training set — only if it
  was labelled and deselected afterwards, since the empty-label rule catches every other
  case — so the summary line counts the deselected images that would actually be written.
  Made visible rather than made impossible, because "show me everything I rejected" is a
  real question and the count is what stops it being answered by accident.
- **What a state change applies to is a set the user builds, at one level at a time.**
  Ticking a project clears the ticked lamellae, ticking a lamella clears the ticked projects,
  and picking frames out clears both — so what is ticked on screen is the whole of what a
  change, or an operation run against the selection, will touch. Not a precedence rule between
  the levels: that leaves ticks visible which decide nothing, and — once a tick outlives the
  project it was made in — ticks *invisible* which decide everything, which is how an
  operation run from this window reaches a project nobody was looking at. Ticked lamellae
  belong to the project on screen and go when it changes; ticked projects stay, because the
  project list still shows them. Frame indices clear with the lamella, because that is all an
  index means here. The reach line counts the lamellae and each modality separately, since the
  modality control decides which of the two moves. Everything reaches `transition_records`
  through `scoped_records()`, so group expansion and the corrected-data guard apply to a
  ticked selection unchanged.
- **A scan the user did not ask for keeps the selection; one they did asks again.** Every
  operation that writes files is followed by a rescan, and preparing labels is two of them
  against the same set — predict, then convert. A rescan that emptied the selection would
  have the second step ask for the choice the first was just given. Pressing *Scan* still
  starts clean, because that is the user saying the collection has moved on. What comes back
  is filtered against what the scan found, so a directory that has gone is dropped rather than
  kept as a tick nobody can see. **Frames come back by which acquisition they are, not by
  where they sat** — `LamellaFrame.key` rather than an index — because an image added ahead of
  them shifts every index after it, and a state change quietly reaching frames nobody ticked
  is the one way this could go wrong without showing it. The *view* is still restored by
  index: landing on a neighbouring frame is a nuisance, where acting on one is a fault.
- **Reading a label gives three answers, not two.** `is_label_file_empty()` is tri-state, and
  **None means "cannot tell"** — deliberately distinct from True, because reporting "not
  annotated" for a label nobody could read would be a guess presented as a fact. Every caller
  that groups labels keeps the two apart.
- **Corrected is a claim, and it is checked where it is made.** Marking a group corrected
  asserts somebody proofread its label, and a training export trusts that. A group marked
  corrected with no label or with a placeholder breaks the claim, and the export is *right*
  to drop it — what was wrong was doing so in silence, leaving a training set short of
  exactly the images you believe you annotated, discoverable only by counting CSV rows months
  later. So the window asks on either side of the gap: before marking a set corrected, and
  before writing a training CSV that would drop one. Both read `classify_labels`, so the two
  dialogs cannot disagree about a label. The export side asks only about *corrected* groups:
  an unlabelled selected image is the ordinary condition of an archive being worked through,
  and a warning firing on every export is one nobody reads by the third time.
- **The command line warns and proceeds where the window asks.** `correct` lists the groups
  whose label holds nothing and marks them anyway. The window is the primary interface and
  the place for a guard; a flag to override a refusal would be a flag that lives in
  everybody's shell history.

## Metadata

- **Nothing is inherited from parent directories.** A collection directory groups projects by
  session or storage convenience, so lamellae in one *directory* may be different samples,
  and resolving a field from a parent would attribute the wrong sample to a project
  invisibly. Lamellae in one *project* do share a sample, which is why the project is the
  right level for this metadata.
- **The Add data tab suggests, and a suggestion is never a value.** It offers the values
  other projects already use for a field, computed in the controller and reaching a file only
  when the user saves — so `build_experiment_metadata` still resolves from the two files and
  nothing else, and the rule above is intact. `project` is on that list precisely because it
  names something several datasets share, which is how a collection ends up holding two
  spellings of one project.
- **`project` names the wider project; the directory is the `autolamella_project`.** They are
  different facts and the software held both under one word, which showed as a
  `project_project` column in one export and a `metadata_project` in another — two
  conventions for one collision. The directory is what the scanner indexes and Curate lists,
  so it is qualified wherever it is written; the metadata field keeps the plain name, since
  that is what somebody filling the form means by it. Neither is derived from the other, which
  is why the Add data tab offers no button to copy the folder name into the field.
- **Every image header carries settings for both beams, and some values exist for only one of
  them.** `microscope_state.*` describes the beam that took the image; `system.*` holds both
  beams' configured defaults, so reading `system.ion.voltage` from an SEM image yields the
  FIB's 30 kV — plausible and wrong. `image_metadata.FIELDS` therefore confines beam-specific
  fields, including the fallback search, to the beam that fired, and declares which beams a
  value exists for at all. A plasma gas belongs to an ion column and an emitter type to an
  electron one, so they are two fields rather than one "beam source": one field for the two
  reports the FIB's Argon on SEM images, and the SEM's `FEG` emitter as a plasma gas. Each
  stays blank on the other beam, and on an image whose beam the header does not state — so a
  blank `plasma_gas` on a FIB image is a real answer, a gallium column having no plasma to
  record.
- **A value is stored in the unit its field name states, converted where the header is read.**
  So `acceleration_voltage_kv` is kV even though both dialects record volts, and nothing
  downstream converts anything. A breakdown that groups a *measured* value rounds it, because
  a header records what was measured rather than what was set: a voltage to 0.1 kV, and a
  pixel size and beam current to four significant figures, those two being derived from the
  magnification and so differing in the last decimal between two images of one field of view.
  Significant figures rather than decimal places, since pixel sizes span decades and one
  decimal place would lose either the 0.5 nm end or the 500 nm one. The exported value is
  never rounded.
- **A missing measurement is never a default.** The one exception is the integration counts,
  where absence means "not enabled" — that is decoding the field, not defaulting it, and it
  is the only field with a floor.
- **Headers are never read during scanning.** Metadata is a mapping keyed by image path,
  built by the commands that need it, so a scan stays a filesystem walk. Reading one header
  is cheap; reading one per image on every GUI launch is not.
- **Where each header value lives is declared once, in `image_metadata.FIELDS`.** That module
  is to header keys what `layout.py` is to directory names, and for the same reason: the
  alternative is a hand-written extraction block per field, so adding a field means writing
  another block and adding a dialect means editing every one. Supporting a third dialect
  should be a column in a table.
- **A label carries no header.** Its scale comes from the image it was drawn from, rescaled
  by the ratio of their widths, because a label converted from a prediction need not share
  its image's pixel dimensions. `label_measure` returns no areas at all rather than assuming
  1:1.
- **Most TIFFs in the collection have no acquisition header, and should not.** Predictions,
  labels and autofocus frames are not acquisitions; only indexed *images* are expected to
  carry one. A count of "unreadable headers" taken over `rglob("*.tif")` measures the wrong
  population.

## Display

- **`downscale_to_fit` uses `min(width_ratio, height_ratio)`.** The images are landscape in a
  portrait pane, so **width binds**, and window chrome does not change how they render.
- **One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a palette
  and a short stylesheet; nothing else in the window picks a chrome colour. Fusion rather
  than the platform's style because the two bindings draw the same widgets differently and
  the development and user environments differ by binding — a window that inherits its style
  is a window nobody has both seen. It is built into Qt, so it costs no dependency. The
  precedence matters more than it looks: where a stylesheet rule and the palette disagree the
  rule wins, and a widget palette edited in Python is simply ignored — yet that palette still
  reads back as the value it was set to, so a test asserting on it passes while the screen
  shows something else. **Anything about how something *renders* is asserted on the pixels**,
  for that reason. It also suppresses what Qt draws by itself: the `QPushButton` and
  `QToolButton` rules override the tint on a dialog's default button and the mark on a
  checked one, so both are re-stated in `theme.py`. Anything relying on Qt's own emphasis has
  to be.
- **A locked editor keeps its normal text colour.** The Add data tab opens locked and is
  mostly used for reading, and Qt's disabled grey puts every value at low contrast. The
  stylesheet dulls the *background* of a disabled input instead and leaves the text dark, so
  a field still reads as locked. Buttons and radio groups grey out normally: they are
  controls you cannot use, not values you are reading.
- **A chosen row is ticked, never highlighted.** Highlighting cannot say what it means: it is
  also how Qt marks the row you are looking at, it is lost whenever the list is rebuilt — which
  is every refresh — and in four visible rows over dozens of values there is no seeing what is
  chosen without scrolling for it, which is how a filter stays on by accident. `CheckList` is
  the one implementation, used by the curation lists and the export filters, with a caption
  saying "2 of 6 ticked" because the count has to be legible without scrolling. It reports
  nothing while its contents are replaced: that is a redraw rather than a gesture, and an owner
  that rebuilds its list in response to the current row changing would drive itself in a
  circle.
- **A list of paths elides; it must not scroll sideways.** Rows carrying paths differ in what
  they *end* with, and a `QListWidget` with a horizontal scrollbar scrolls instead of
  eliding — which cut a session's projects off into identical-looking rows. `ElideMiddle`
  plus `ScrollBarAlwaysOff` keeps both ends, and the tooltip carries the whole label. Applies
  to the metadata panel's project list and to the Export tab's project, lamella and stage
  filters.
- **The image panes have no scrollbars, and their scene is bigger than the image.** The
  scene carries a quarter of the image as slack on every side, so an edge can be dragged into
  the middle of the pane to be worked on. Scrollbars go because a scene deliberately larger
  than its content leaves them permanently visible and permanently not full.
- **How far a pane zooms out is the view's to say, and it lands through `fitInView`.**
  Reviewing is done at fit, so the curation panes stop there; the proofreading canvas allows
  one step past it, because checking that a label runs to the edge of the frame means seeing
  that edge clear of the border — and `zoom_by` snaps to the floor rather than overshooting,
  so a floor of 1/1.1 turns that 0.8 step into a band of a twentieth of the pane. Landing goes
  through `fitInView` rather than `scale` for two reasons. The wheel's anchor holds the point
  under the pointer fixed, which is what makes zooming *in* follow the eye — and what leaves a
  frame that now fits the pane pushed into whichever corner the pointer was over. And
  `fitInView` keeps a two-pixel border of its own that scaling to the computed floor would
  crop, so "zoomed all the way out" and "fitted" are the same picture.
- **A summary plot is held twice: pane-sized to step through, whole to look into.** One copy
  cannot do both jobs. A plot is ~180 ms to read however it is asked for — a PNG decompresses
  whole, so OpenCV's reduced-decode flags buy nothing — which is slower than a held-down arrow
  key, so stepping is smooth only when every plot is already in memory, and that bounds those
  copies to the size of the pane. But these figures are 4500x3000 and carry annotations that
  exist only at that resolution, which is what a plot is opened for. So the pane-sized copies
  cover the lamella and are what a *fitted* plot is, drawn at one array pixel per physical
  pixel — the only scale that resamples nothing. The frame being looked at is also read whole,
  asked for last because nothing on screen is waiting on it, and takes over only once the
  figure is magnified past what the pane copy holds: a swap inside a zoom, where a change of
  scale is what the user just asked for. Drawing it on arrival instead cost both things at
  once — every plot on screen was a resampled one, and each step refitted and then shifted
  again as the copy landed. The threshold carries a hair of margin, because returning to fit
  lands either side of exactly 1 and the wrong side leaves the one view that has to be exact
  filtered.
- **Where a copy is placed is read off the copy, and remembered across a swap.** Two faults
  lived here, both of them placement rather than content. The copy is reduced to a box
  measured in *physical* pixels, so drawing it at whatever device ratio the widget reports at
  the time comes out at half or twice the size wherever the two disagree — whichever of its
  edges touches that box gives the ratio back exactly, so a copy prepared for a pane of
  another size still fills the one it is drawn in. And an identity transform says nothing
  about where the view is looking, while the scene is deliberately larger than the picture, so
  drawing 1:1 has to centre as fitting would. Where the view *is* looking is then held as a
  fraction of the picture rather than re-derived on each swap: deriving goes through the
  scroll position, which is a whole number, and stepping while zoomed changes resolution twice
  a frame, so the picture crept towards one corner. It is dropped the moment the user moves
  the view themselves, which `scrollContentsBy` reports — panning by any means counts, and the
  scrollbar movements the widget causes itself do not.
- **Read-ahead asks for no more than the cache can keep.** Unbounded, the read that finishes
  last evicts the one that finished first, so a step in either direction lands on something
  that was prepared and then thrown away — the cache working against itself. `plot_radius`
  is derived from the entry limit, so a lamella that fits is still prepared whole (almost all
  of them are) and one that does not is prepared around where the user is. On the longest
  lamella in the reference collection, 61 plots against 38 that fit, this is the difference
  between stepping end to end with no wait at all and re-reading at both extremes.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**,
  and that mode defaults to nearest neighbour. A pane showing something several times its own
  size therefore drops pixels rather than averaging them: measured against a figure of
  one-pixel rules, nearest loses whole gridlines. The plot view asks for smooth scaling.
  Acquisitions do not, because zoomed *in* on a label, nearest is what shows where one pixel
  ends and the next begins.
- **`QFontDatabase.systemFont(FixedFont)` is not a fixed-width font.** On macOS under PyQt5
  it returns the proportional `.AppleSystemUIFont` while reporting the family it was asked
  for, so a table padded into columns comes out unaligned with nothing saying why. A style
  hint alone does not help: Qt consults it only for a family that does not exist.
  `common.fixed_width_font` names real families per platform. Asserted on the metrics (`"i"`
  and `"W"` advance equally), never on `QFont.fixedPitch()`, which reads back what was
  requested, not what resolved.
- **A list re-tints the icon of the row you are on.** Qt draws a selected row's icon in
  `QIcon::Selected` mode, and the variant it generates for that mode is blended with the
  highlight colour — so a status chip whose colour *is* the information came out a different
  colour on exactly the row being looked at. `chip_icon()` supplies the pixmap for every mode
  and state, which leaves Qt nothing to generate. Any future icon carrying meaning rather
  than decoration has to go through it.
- **`QScrollArea.setWidget` shows whatever it is handed**, so hiding a widget and then putting
  it in a scroller leaves it on screen. Hide the scroller instead, and give the column a
  widget to hold its spare height: a layout shares that height between whatever is still
  visible, and a stretch cannot be hidden again.
- **A tooltip says what a control does, in under 200 characters**, and a test holds the
  ceiling across the whole window rather than one tab — enforced narrowly, the ones out of
  scope grow. Why the software works as it does belongs here, not in a hover nobody finishes.
  Every control in the Filters box also says what it narrows: that it decides the breakdown
  *and* the exports is not visible from the box's title.
- **The busy dialog is not a `QProgressDialog`.** That class calls `processEvents()` from
  inside `setValue()` whenever it is modal — which this one has to be, since its whole job is
  to keep the window out of files that are being written. Processing events there re-enters
  the event loop from inside a queued-signal handler, so the worker's *finished* can be
  delivered in the middle of a progress update: the dialog is closed and its last reference
  dropped while one of its own methods is still on the stack. It showed as the window
  freezing on the final item of a run, which is where the two signals are most likely to
  race. `busy_dialog.py` is a plain `QDialog` that processes nothing, and the window detaches
  it before closing it so that anything running meanwhile finds no dialog rather than one
  being taken apart. Escape is refused there for the same reason the close button is absent:
  on a stoppable operation it asks it to stop, and on any other there is nothing it could
  safely do.
- **A long operation runs on a thread, and says how far it has got.** Qt draws nothing while
  the thread owning the widgets is busy, so the work goes to `gui/worker.py` and the window
  shows a modal dialog naming it. That dialog opens indeterminate and becomes a count as soon
  as the operation reports, through one `on_progress(done, total)` contract and one
  `progressed` signal connected once for the whole window — so giving an operation a counter
  is a callback in its loop, not a mechanism. Which ones have earned it is a question about
  duration: predicting is seconds an image, measuring and recompressing are tens of
  milliseconds a label over a whole collection, and writing a metadata file per project is
  over before the dialog paints. Only the applying run counts, never the preview: a preview
  that answers in milliseconds would fill a bar instantly and describe the wrong run.
- **Whether it can be *stopped* is a different question, and the answer is what a half-done
  run leaves behind.** Two can be. Each prediction is written atomically and recorded as this
  model's work before the next begins, so a stopped run has produced finished files and an
  accurate account of them, and the next run resumes from there. Recompression is safer
  still: a re-encoded label holds the same pixels as the one it replaced, so a collection
  stopped halfway differs from a finished one only in how much disk it takes — there is
  nothing to resume and nothing to undo. Both stop between whole files, and both report what
  they completed rather than raising it away. Converting predictions is the one that must
  not: it validates the whole set of destinations before writing any of them, and stopping it
  leaves exactly the half-applied collection that rule exists to prevent. Cancellation is a
  `threading.Event` in `progress.py` polled by the work closure, rather than a Qt signal: a
  queued signal reaches the worker when it next returns to its event loop, which inside a
  long call is never. It rides in through the progress callback, so an operation that reports
  progress needs no cancellation branch of its own.
- **Whether to offer the predict button is decided by locating torch, not importing it.** An
  import costs seconds at every window start, including for someone who never predicts.
  `ml_stack_present` uses `find_spec`; the press then runs the full check, because a torch
  that is installed and will not load reads as present to the first and fails on the second.

## Environment

- **A save refuses a label whose file changed since it was opened.** Proofreading holds a
  lamella's edits in memory until the reviewer saves, which leaves a window of minutes to
  hours in which something else can rewrite a label file. That is not hypothetical here: this
  collection lives on OneDrive, which writes files back under a running program, and the
  whole point of the corrected state is that the annotation in it cannot be reconstructed. So
  the size and modification time are recorded when a label is opened and compared before
  writing, and a mismatch stops the entire batch rather than the one file — which of the two
  versions is wanted is not a question this package can answer, and overwriting is the answer
  that cannot be undone.
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
- **Every file dialog is Qt's own, never the platform's.** The macOS panel under PyQt5
  5.15.2 offered no selectable files at all for a `*.csv` filter — only folders to navigate
  into — so "Check exported CSV" could not be given a CSV, and because directory dialogs were
  unaffected it read as a missing feature rather than a broken one.
  `widgets/common.build_path_dialog` sets `DontUseNativeDialog` for all of them. Revealing a
  folder in the file manager is not a dialog and stays native.
- **Figures are built as `Figure()`, never through `pyplot`.** pyplot keeps every figure it
  makes in a global list until something closes it, and choosing a backend for that list is
  what makes a chart drawn over ssh look for a display. A bare `Figure` has neither problem:
  `savefig` works on it, the window attaches its own canvas, and a session of charts is
  collected as the dialogs holding them close. `matplotlib` is in the `gui` and `all` extras
  rather than only `plots`, because against the OpenCV and Qt those already install its size
  is not what makes an environment large.
- **A dropdown you can arrow through is `StepCombo`, not Qt's own keys.** Whether Up and Down
  step through a combo box or open its popup is decided by the style hint
  `SH_ComboBox_Popup`. Under PyQt5 with the `macintosh` style they step, so the difference is
  invisible here — but the hint belongs to the style, and the window ships under a different
  binding. Same reasoning as the file dialogs above.
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
- **Never mix conda-forge with Anaconda `defaults`.** `environment.yml` pins `nodefaults` for
  this reason: a mixed environment paired Anaconda's OpenCV with conda-forge's protobuf and
  `cv2` would not load at all.
- **Inference lives in a second environment, and that one is pip-first.** The main
  environment is conda-first because conda's OpenCV and pip's cannot coexist. albumentations,
  which the vendored preprocessing imports, requires `opencv-python-headless` — a third
  provider of `cv2`, and one no ordering of install commands avoids. So `environment-ml.yml`
  takes only python and pip from conda and the whole stack from the `[ml]` extra, where there
  is exactly one OpenCV. It also lets the torch build be chosen per machine, which
  conda-forge cannot express: `AM_SEG_TORCH_INDEX_URL` selects a wheel index.
  `scripts/setup_ml.sh` builds it.
- **The inference environment is a superset, not a peer.** It gets the window too, via the
  `qt` extra — Qt binding and matplotlib, no OpenCV, since it already has the headless one,
  which is also the better partner for PySide6 (`opencv-python` carries Qt plugins of its
  own). The curation environment is the *lighter* of the two, for someone who never predicts;
  splitting them the other way forces an environment switch mid-workflow, between curating a
  set and predicting it.
- **Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
  `ml/environment.py` answers "is inference possible here" without importing it, the way
  `io/image.py` does for the image stack. `ml/preprocessing.py` is a verbatim copy of
  AM_training's module, excluded from ruff: reformatting it would destroy the only cheap
  answer to "is this still the preprocessing the model was trained with" — `diff`.

## Inference

- **A fact recoverable from the weights is never declared.** A checkpoint is a bare
  `state_dict`, and its class count, input channels and tensor names are all readable from
  it. Declaring those in a file beside it adds a second statement that can disagree with the
  first, and the disagreement would be invisible: a description naming four classes over a
  five-class head loads whatever fits. So `ml/model.py` reads them, and
  `load_state_dict(strict=True)` is what proves the description's architecture and the
  weights are the same model. A mismatch stops at the load rather than at the first
  odd-looking label.
- **What the weights cannot state is declared in a `.yaml` beside them.** Preprocessing
  settings and the architecture name are not in the file, and every one of them shows itself
  immediately if wrong, which is what makes a default safe.
- **Which modality a model is for is not recorded anywhere, deliberately.** Nothing in a
  checkpoint says, and `AM_SEM_All_V01.pth` is whatever whoever trained it chose to type —
  reading that as evidence would make the software's answer depend on someone else's
  file-naming habit. Declaring it in the sidecar only moves the guess: a declaration nothing
  can check is a claim wearing the clothes of a fact.
- **So the selection states it, and a selection spanning both modalities is refused.** One
  model is for one modality; a selection naming both names nothing, and the run stops with
  the two it found. The guard is about something the window can show you — the edit target
  and the images under it — rather than a line in a file you would have to open. Every report
  leads with the modality.
- **Predicting with the wrong model is cheap to undo**, which is what makes that guard
  sufficient rather than merely defensible. Resumption keys on the model and run name, so
  predicting the same images with the right model replaces the wrong ones without
  `--overwrite`: an accident costs the inference time and nothing else.
- **`decoder_attention_type="scse"` is not part of these models.** The training call passes
  it, smp's `FPN` has no such parameter, and it is swallowed into `**kwargs` and forwarded to
  the encoder, which ignores it. The checkpoints contain no attention tensors, and a plain
  FPN strict-loads all 1015 of them. Build plain FPN; do not carry the argument over on the
  assumption that it did something.
- **One prediction per label group, not per image.** The label is the unit of workflow state
  everywhere else here, and a prediction exists to become a label, so a group gets one — made
  from `label_groups.primary_record`, the same image the curation window shows and a training
  export keeps. Predicting the other conditions would cost inference to produce worse
  pseudo-labels: `SEM_8_fi_1` is one frame integration and far noisier than the `SEM_8` it
  shares a label with. It also removes a hazard rather than managing one — conditions carry
  identical filenames into a shared predictions directory, so predicting all of them would
  have each write replace the last.
- **Predictions are written at model resolution.** A 2048x3072 image yields a 1024x1536
  prediction — which is what the segmentation pipeline's own `raw/` already holds, at 12 kB
  against the 6 MB an upsampled one would cost. `convert-predictions` resizes to the image
  when it makes the label, so upsampling earlier would only be undone.
