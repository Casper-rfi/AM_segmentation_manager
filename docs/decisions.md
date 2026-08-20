# Decisions

Why the software is built the way it is, and facts that cost time to rediscover.
Every entry is here because it is **not** derivable from reading the code: the code
shows what happens, not what breaks if it changes.

Work that is *not* built yet belongs in [`04_future_work.md`](04_future_work.md), not
here. If something below becomes wrong, correct it in place.

---

## Architecture

**No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python
controllers. This is what makes navigation, cache eviction and state expansion
testable without a display server. It is the one structural rule of the UI; adding a
tab is not a reason to bend it.

**An operation belongs in the window when its input is the selection on screen; it
stays on the command line when it is a sweep over the whole collection.** Naming a
selection twice — once by ticking it, once as command-line filters — is where the two
diverge and the wrong data gets moved. The Add data commands are in the window despite
being sweeps, because there the sweep *is* the collection and there is nothing to select.
Each previews and then applies what it previewed, as two runs of the same function
rather than one run replayed: a preview held across a dialog describes a filesystem the
user has had time to change.

**A filter and a breakdown are one declaration in the Export tab.** They are the same
key function used two ways — group by it, or keep the records whose group was chosen —
so the filter lists come from `BREAKDOWNS` rather than a second table beside it, and
cannot disagree with the table about what a value *is*: the filter used to offer a raw
pixel size no row ever showed. Two consequences: "(not recorded)" is selectable, the
only way to find images whose header would not read; and a measured property is not
offered, since it cannot answer before Measure is pressed. Modality, workflow state and
the label filter keep dedicated controls, being fixed questions rather than data.

`filtering.METADATA_FILTERS` stays as it was — it is the command line's contract,
where a filter is typed rather than picked and has to tolerate `30000` for a stored
`30000.0`.

**Counting is the exception, and it split rather than moved.** The Export tab counts and
cross-tabulates the set its filters describe, measuring labels when asked — because
choosing a training set means looking at what survives a filter, and that loop cannot
run through a command that prints and exits. `stats` keeps the whole-collection sweep
with its own scope selectors, and `quantify-labels` keeps the per-lamella table: a table
of numbers per class per lamella is not something anyone acts on one row at a time, and
a spreadsheet reads it better from the CSV.

**What a measured number means is set beside the breakdown, not among the filters.**
The crack threshold and the area bin resolution change how a measurement is *read*, not
which images are in play, so a Filters box promising to narrow something is the wrong
home. Changing the threshold re-measures at once rather than spending the Measure button
again: `read_label_pixels` caches component pixel counts *without* the threshold, so a
new one costs arithmetic — 11-70 ms against 7.7 s for the first pass. `load_measurements`
keeps the headers it reads for the same reason.

**No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few
lines with no dependency, over thousands of records rather than millions. pandas belongs
here only once something genuinely needs it.

---

## Filesystem conventions

**A label directory is `<image-directory>_labels`, a sibling of the image directory.**
`layout.py` states the resolution order; what it cannot say is why that shape. It makes
image-dir → label-dir a pure string function both ways with no lookup table, which is
what lets one module hold the convention. Conditions share a label by default; giving one its own means creating the directory, not configuring
anything. `sem/` is no special case: its private and shared names coincide, so one rule
covers the plain layout and the five-condition one alike.

**Image directories keep their Autolamella names.** We do not own `sem` and `fib`;
renaming them would break reopening the project in Autolamella. Label directories are
ours, so they are the only ones anything here renames.

**Renaming pre-convention label directories is a script, not a command.** The package
reads `labels`, `labels_fib` and `labels_FIB` indefinitely, and a new label lands in
whichever directory an image set already has, so no project ends up holding two namings.
That leaves cosmetic tidiness as the only reason to rename, over a fixed handful of
datasets — so it lives in `scripts/rename_legacy_label_dirs.py`, where a layout that
does not fit is handled by editing the script rather than by generalising a command.

**The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`,
`sem_lowmag` all work with no code change, because authoritative imaging parameters
come from the image headers. A variant may not be named `labels`, `deselected`,
`corrected` or `predictions` — those collide with the label naming rule, so
`layout.py` validates it and `inspect-layout` reports a violation rather than
mis-indexing.

**A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`.**
`init-project` writes the latter rather than fabricating the former, because
`experiment.yaml` is Autolamella's file: writing one misrepresents provenance and
could collide if Autolamella ever opens that directory. `protocol.yaml` looks like a
third marker and is not usable as one: some genuine Autolamella projects have
`experiment.yaml` without it.

**A project's identity is its path relative to the scan root, not its name.** The
reference collection holds two different projects whose name, `created_at` and
recorded path in `experiment.yaml` are all identical — keying by name silently drops
one.

**A lamella is counted as `project / lamella`, and a stage one level deeper.** The
same rule as above, applied to a table rather than an index: almost every project has
an `01-`, so an unqualified row merges lamellae with nothing to do with each other —
in the reference collection two projects each hold an `05-pet-mutt`, reported as one
lamella of 140 images. Stage names are worse, being either shared by every lamella
(`adaptive_polish`) or a bare timestamp. `aggregate.summarise_lamellae` already keyed
on the pair; the breakdowns now agree with it.

**The lamella and stage levels are not always directories.** Older datasets collapse
them; the missing level is synthesised in memory with the parent standing in, and the
model records that it is synthetic. Data on disk is never reorganised to fit the
newest layout.

**`predictions/clean/` is indexed but can never become a label.** Adaptive Polishing's
post-processing rewrites the intensity values, so they no longer index the
segmentation classes. A label converted from one looks entirely valid — right
dimensions, plausible shapes — and teaches a model the wrong classes, which is worse
than having no label at all. This will not change upstream, so it is a property of
`layout.py` (`is_convertible_prediction`) rather than a choice offered to the caller.

It is still indexed, deliberately: otherwise a project holding only post-processed
predictions would report "no predictions", indistinguishable from one the pipeline has
never reached, when those need opposite actions. Conversion counts the two separately.

---

## Workflow state

**State is declared in a file, not implied by location.** Every directory holding image
and label directories carries an `AM_state.yaml` naming the groups that are not selected.
A file's directory used to *be* its state, because the external annotation tool wrote an
empty label for every image in its input folder and keeping images out of that folder was
the only way to say "not this one". Nothing requires it now, and it was expensive: a state
change was a batch of moves that could half-succeed, a label could exist in two state
folders at once, and curating an image rewrote a path a training CSV had already
recorded.

**The label is the unit of state, and the key names the label.** An entry is keyed by the
group's label path -- `<label-directory>/<filename>`, relative to the state file --
whether or not that label exists yet. Keying on the label rather than the image is what
makes an image disagreeing with its label *unrepresentable* rather than merely
detectable: the several conditions of one scene resolve to one key, so a change reaches
all of them or none. Resolved rather than located, so the key survives annotation
beginning -- which also makes images sharing a label directory one group before any label
exists, where they used to be independent until one appeared.

**Anything not listed is selected.** An absent entry and an absent file give the same
answer, so images dropped into a stage tomorrow arrive selected with no reconciliation --
exactly as they did when the base directory meant selected, and most stages then carry a
short file or none at all. The cost is that "reviewed and kept" cannot be told from "not
yet looked at", which the folders could not tell apart either.

**The three states are mutually exclusive, and the file's shape enforces it.** A mapping
of key to state cannot express two states for one group, where a file grouping keys under
state headings could. That is why the file is not the more readable-looking `corrected:` /
`deselected:` lists. Duplicate keys are rejected on read rather than resolved last-wins,
since a silent winner is the same failure the mapping was chosen to prevent. Deselecting a
group that was corrected discards the corrected claim; the proofreading itself survives in
the label's pixels, which is what an export checks.

**A stage is transcribed whole or not at all, and one that has not been is refused.** The
two rules are the same rule. `migrate-state` leaves a stage alone if anything in it is
ambiguous -- images disagreeing with their label, or a file sitting in two state folders --
because writing the groups that were fine would leave the rest unlisted, and unlisted reads
as *selected*. For the same reason a state change refuses a stage that still keeps its
state in folders. Promoting a deselected image into a training set is the one failure here
with no visible symptom, so both paths are shut rather than guarded.

**`migrate-state` walks the filesystem itself, and is the only thing that reads a folder
name as a state.** The scanner does not look inside `deselected/` or `corrected/` at all,
so a nested file is not indexed: it is *missing and reported* rather than present and
wrong. That is also why nothing repairs a group any more -- images cannot disagree with
their label when one entry covers them all, so `repair-state` and the consistency
machinery went with the folders.

**`init-project` refuses a directory that holds projects of its own.** That is a
collection, and a marker there makes every unmarked dataset beneath it a *lamella* of one
project -- several samples under one metadata record, invisible in the output. `--recursive`
is one flag away and does the right thing, so refusing costs nothing.

**A stale entry is reported, never repaired.** Declared state can disagree with the
filesystem in ways location could not, and each needs a different action, so
`inspect-layout` separates them: a stage whose folders still hold its state has not been
migrated; nested files after migration want `--move-files`; an empty state folder is
clutter; an entry naming a group that is not there was left behind by a rename or a
deletion. That last one is kept rather than dropped -- a group that vanished because a
drive was not mounted must not have its curation deleted for it.

**A recorded path cannot go stale, so `check-training-csv` only checks.** It existed to
rewrite paths that curation had moved; nothing moves, so a path that does not resolve
means the file is gone. What survives is the check and a `--prune` that drops the dead
rows, because a CSV holds a particular train/validation split and losing a few images is
not a reason to regenerate it.

**A write is read-modify-write, and refuses a file that changed underneath it.** Two
windows on one collection, or a window and a command line, would otherwise discard each
other's curation with no error. Under folders that could not happen, two moves of different
files never colliding; it is the one failure mode this design adds, and it is guarded at
the write, which is atomic.

**The state folder names stay reserved**, so an unmigrated dataset reads as *missing*
those images rather than as holding selected ones. Reading a deselected image as selected
is the one migration failure that would reach a training set unnoticed.

**Whether a group is corrected is asked of the record, never of a path.** Every guard that
protects proofread work -- conversion, compression, label saving -- reads the declared
state the scanner put on the record. A path stopped answering that question the moment the
files left their folders, and a guard that quietly stops firing is worse than none.

**Overwriting a label is tiered by what replacing it would cost, not by one flag.** A run
that writes nothing has to say why, or it cannot be told from a broken one. So conversion
sorts destinations three ways: an empty placeholder costs nothing to replace (nothing was
in it); a pseudo-label holding annotation is regenerable machine output, replaced once the
user confirms; a label whose group is corrected is refused, and no argument to
`plan_conversions` reaches it.

That last tier is stricter than `compress-labels`, which *can* be confirmed for corrected
labels. The difference is what each does to the pixels: compression preserves them exactly,
conversion replaces proofread work with a machine guess. A dialog is the right guard for the
first and too weak for the second, so re-converting a corrected label means setting that
group back to selected first.

**What a state change applies to is a set the user builds, and the finest level present
wins.** Picked-out frames beat ticked lamellae, which beat ticked projects, and with none of
them it is the frame on screen. Finest-first is chosen for its failure mode: a selection the
user has forgotten makes the next change *smaller* than intended, where the opposite order
would make it larger. Ticks are keyed by resolved path, since a row means a different
lamella as soon as the project changes; frame indices clear with the lamella, because that
is all an index means here. Everything reaches `transition_records` through
`scoped_records()`, so group expansion and the corrected-data guard apply to a ticked
selection unchanged.

**Reading a label gives three answers, not two.** `is_label_file_empty()` is tri-state, and
**None means "cannot tell"** -- deliberately distinct from True, because reporting "not
annotated" for a label nobody could read would be a guess presented as a fact. Every caller
that groups labels keeps the two apart.

**Corrected is a claim, and it is checked where it is made.** Marking a group corrected
asserts somebody proofread its label, and a training export trusts that. A group marked
corrected with no label or with a placeholder breaks the claim, and the export is *right* to
drop it -- what was wrong was doing so in silence, leaving a training set short of exactly
the images you believe you annotated, discoverable only by counting CSV rows months later.
So the window asks on either side of the gap: before marking a set corrected, and before
writing a training CSV that would drop one. Both read `classify_labels`, so the two dialogs
cannot disagree about a label.

The export side asks only about *corrected* groups: an unlabelled selected image is the
ordinary condition of an archive being worked through, and a warning firing on every export
is one nobody reads by the third time.

**The command line warns and proceeds where the window asks.** `correct` lists the groups
whose label holds nothing and marks them anyway. The window is the primary interface and the
place for a guard; a flag to override a refusal would be a flag that lives in everybody's
shell history.

---

## Metadata

**Nothing is inherited from parent directories.** A collection directory groups
projects by session or storage convenience, so lamellae in one *directory* may be
different samples, and resolving a field from a parent would attribute the wrong
sample to a project invisibly. Lamellae in one *project* do share a sample, which is
why the project is the right level for this metadata.

**The Add data tab suggests, and a suggestion is never a value.** It offers the values
other projects use for a field, and the name of the folder a project's own YAML files
sit in for `project`. Both are computed in the controller and reach a file only when the
user saves, so `build_experiment_metadata` still resolves from the two files and nothing
else — the rule above is intact. The folder read is the project's own, never a parent,
and is preferred over `experiment.yaml`'s `name` because a project copied elsewhere
keeps the recorded name of the first location.

**Every image header carries settings for both beams, and some values exist for only
one of them.** `microscope_state.*` describes the beam that took the image; `system.*`
holds both beams' configured defaults, so reading `system.ion.voltage` from an SEM image
yields the FIB's 30 kV — plausible and wrong. `image_metadata.FIELDS` therefore confines
beam-specific fields, including the fallback search, to the beam that fired, and
declares which beams a value exists for at all. A plasma gas belongs to an ion column
and an emitter type to an electron one, which is why they are two fields rather than one
"beam source": one field for the two reported the FIB's Argon for SEM images, and the
SEM's `FEG` emitter as if it were a plasma gas. Each stays blank on the other beam, and
on an image whose beam the header does not state. A blank `plasma_gas` on a FIB image is
therefore a real answer — a gallium column has no plasma and records no gas.

**A value is stored in the unit its field name states, converted where the header is
read.** So `acceleration_voltage_kv` is kV even though both dialects record volts, and
nothing downstream converts anything. A breakdown that groups a *measured* value
rounds it, because a header records what was measured rather than what was set: a
voltage to 0.1 kV, and a pixel size and beam current to four significant figures,
those two being derived from the magnification and so differing in the last decimal
between two images of one field of view. Significant figures rather than decimal
places, since pixel sizes span decades and one decimal place would lose either the 0.5
nm end or the 500 nm one. The exported value is never rounded.

**A missing measurement is never a default.** The one exception is the integration
counts, where absence means "not enabled" — that is decoding the field, not defaulting
it, and it is the only field with a floor.

**Headers are never read during scanning.** Metadata is a mapping keyed by image path,
built by the commands that need it, so a scan stays a filesystem walk. Reading one
header is cheap; reading one per image on every GUI launch is not.

**Where each header value lives is declared once, in `image_metadata.FIELDS`.** That
module is to header keys what `layout.py` is to directory names, and for the same
reason: the alternative is a hand-written extraction block per field, so adding a
field means writing another block and adding a dialect means editing every one.
Supporting a third dialect should be a column in a table.

**A label carries no header.** Its scale comes from the image it was drawn from,
rescaled by the ratio of their widths, because a label converted from a prediction
need not share its image's pixel dimensions. `label_measure` returns no areas at all
rather than assuming 1:1.

**Most TIFFs in the collection have no acquisition header, and should not.**
Predictions, labels and autofocus frames are not acquisitions; only indexed *images*
are expected to carry one. A count of "unreadable headers" taken over `rglob("*.tif")`
measures the wrong population.

---

## Display

**`downscale_to_fit` uses `min(width_ratio, height_ratio)`.** The images are landscape
in a portrait pane, so **width binds**, and window chrome does not change how they
render.

**One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a
palette and a short stylesheet; nothing else in the window picks a chrome colour.
Fusion rather than the platform's style because the two bindings draw the same widgets
differently and the development and user environments differ by binding — a window
that inherits its style is a window nobody has both seen. It is built into Qt, so it
costs no dependency.

The precedence matters more than it looks: where a stylesheet rule and the palette
disagree the rule wins, and a widget palette edited in Python is simply ignored — yet
that palette still reads back as the value it was set to, so a test asserting on it
passes while the screen shows something else. **Anything about how something
*renders* is asserted on the pixels**, for that reason.

It also suppresses what Qt draws by itself. Qt tints a dialog's default button and
marks a checked one; the `QPushButton` and `QToolButton` rules overrode both, so the
save dialog offered two very different choices as two identical buttons and the class
being painted was marked by a fill nobody could see across a desk. Both are re-stated
in `theme.py` — anything that relied on Qt's own emphasis has to be.

**A locked editor keeps its normal text colour.** The Add data tab opens locked and is
mostly used for reading, and Qt's disabled grey puts every value at low contrast. The
stylesheet dulls the *background* of a disabled input instead and leaves the text dark,
so a field still reads as locked. Buttons and radio groups grey out normally: they are
controls you cannot use, not values you are reading.

**A list of paths elides; it must not scroll sideways.** Rows carrying paths differ in
what they *end* with, and a `QListWidget` with a horizontal scrollbar scrolls instead
of eliding — which cut a session's projects off into identical-looking rows.
`ElideMiddle` plus `ScrollBarAlwaysOff` keeps both ends, and the tooltip carries the
whole label. Applies to the metadata panel's project list and to the Export tab's
project, lamella and stage filters.

**The image panes have no scrollbars, and their scene is bigger than the image.**
Annotation is easiest to leave a pixel short at the edges, and both limits stopped
exactly there: the zoom floor was the fit scale, pressing the frame against the border,
and the scene was the image, so an edge could never be dragged away from it. The scene
now carries a quarter of the image as slack on every side and the zoom goes two steps
below fit. Scrollbars go because a scene deliberately larger than its content leaves
them permanently visible and permanently not full.

**`QFontDatabase.systemFont(FixedFont)` is not a fixed-width font.** On macOS under
PyQt5 it returns the proportional `.AppleSystemUIFont` while reporting the family it
was asked for, so a table padded into columns comes out unaligned with nothing saying
why. A style hint alone does not help: Qt consults it only for a family that does not
exist. `common.fixed_width_font` names real families per platform. Asserted on the
metrics (`"i"` and `"W"` advance equally), never on `QFont.fixedPitch()`, which reads
back what was requested, not what resolved.

**A list re-tints the icon of the row you are on.** Qt draws a selected row's icon
in `QIcon::Selected` mode, and the variant it generates for that mode is blended with
the highlight colour — so a status chip whose colour *is* the information came out a
different colour on exactly the row being looked at. `chip_icon()` supplies the pixmap
for every mode and state, which leaves Qt nothing to generate. Any future icon carrying
meaning rather than decoration has to go through it.

**`QScrollArea.setWidget` shows whatever it is handed**, so hiding a widget and then
putting it in a scroller leaves it on screen — which is how the Export tab's filters
were visible before there was anything to put in them. Hide the scroller, and give the
column a widget to hold its spare height: a layout shares that height between whatever
is still visible, and a stretch cannot be hidden again.

**A tooltip says what a control does, in under 200 characters**, and a test holds the
ceiling across the whole window. They had grown to five lines apiece explaining *why*
the software works as it does, which belongs here rather than in a hover nobody
finishes. The ceiling was enforced on one tab for a while, and in the gap the curation
timeline grew a 488-character legend. Every control in the Filters box also says what
it narrows: that it decides the breakdown *and* the exports is not visible from the
box's title.

**A long operation runs on a thread, and most of them cannot be cancelled.**
Recompressing a project's labels is minutes, and Qt draws nothing while the thread
owning the widgets is busy — so the work goes to `gui/worker.py` and the window shows
a busy dialog. No cancel button on those: they validate the complete set of changes
before performing any of them, and stopping one halfway leaves exactly the
half-applied collection that rule exists to prevent.

**Predicting is the exception, because stopping it leaves nothing half-applied.**
Each prediction is written atomically and recorded as this model's work before the
next begins, so a stopped run has produced finished files and an accurate account of
them, and the next run resumes from there. That is what earns it a Stop button and a
progress bar where an hour of work would otherwise be a frozen window. Cancellation
is a `threading.Event` in `progress.py`, polled by the work closure, rather than a Qt
signal: a queued signal reaches the worker when it next returns to its event loop,
which inside a long call is never. It is offered only on the applying run — a preview
opens no file and answers in milliseconds.

**Whether to offer the predict button is decided by locating torch, not importing
it.** An import costs seconds at every window start, including for someone who never
predicts. `ml_stack_present` uses `find_spec`; the press then runs the full check,
because a torch that is installed and will not load reads as present to the first and
fails on the second.

---

## Environment

**A save refuses a label whose file changed since it was opened.** Proofreading
holds a lamella's edits in memory until the reviewer saves, which leaves a window
of minutes to hours in which something else can rewrite a label file. That is not
hypothetical here: this collection lives on OneDrive, which writes files back
under a running program, and the whole point of the corrected state is that the
annotation in it cannot be reconstructed. So the size and modification time are
recorded when a label is opened and compared before writing, and a mismatch stops
the entire batch rather than the one file -- which of the two versions is wanted
is not a question this package can answer, and overwriting is the answer that
cannot be undone.

**The reference collection is at
`<collection>/`** — 21 projects, 2670
indexed images, 40 GB, three microscopes. Treat it as read-only. Point the smoke test
at it with `AM_SEG_REFERENCE_COLLECTION=<path> pytest -q`.

**Every file dialog is Qt's own, never the platform's.** The macOS panel under PyQt5
5.15.2 offered no selectable files at all for a `*.csv` filter — only folders to
navigate into — so "Check exported CSV" could not be given a CSV, and because directory
dialogs were unaffected it read as a missing feature rather than a broken one.
`widgets/common.build_path_dialog` sets `DontUseNativeDialog` for all of them. Revealing
a folder in the file manager is not a dialog and stays native.

**Figures are built as `Figure()`, never through `pyplot`.** pyplot keeps every figure
it makes in a global list until something closes it, and choosing a backend for that
list is what makes a chart drawn over ssh look for a display. A bare `Figure` has
neither problem: `savefig` works on it, the window attaches its own canvas, and a
session of charts is collected as the dialogs holding them close. `matplotlib` is in
the `gui` and `all` extras rather than only `plots`, because against the OpenCV and Qt
those already install its size is not what makes an environment large.

**A dropdown you can arrow through is `StepCombo`, not Qt's own keys.** Whether Up
and Down step through a combo box or open its popup is decided by the style hint
`SH_ComboBox_Popup`. Under PyQt5 with the `macintosh` style they step, so the
difference is invisible here — but the hint belongs to the style, and the window
ships under a different binding. Same reasoning as the file dialogs above.

**Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side
effect, and `test_gui_*` sorts before most other tests. That is how a locale-dependent
`strptime('%p')` bug appears only in full runs, and only on a machine with a
non-English locale. `LC_ALL=… pytest` does **not** test this; `AM_SEG_TEST_LOCALE=…
pytest` does.

**Never disconnect a signal from inside the handler for that signal.** A connection
list mutated mid-emit works under PyQt5 and hangs under PySide6 — the suite stalled
indefinitely on the first test to run a real background operation, with no traceback,
because the failure path opens a modal box nobody is there to dismiss. Connect once
and track what is pending in an attribute. The same shape hides in any
connect-per-call pattern.

**`QProxyStyle.styleHint` segfaults PySide6.** Overriding it to answer
`SH_DialogButtonLayout` — to fix the button order in every dialog at once, since Fusion
reports the Windows order wherever it runs and ends a row with *Cancel* — crashes the
process while a `QMessageBox` is being built. Returning the enum rather than an `int`
is separately wrong and only warns. The save dialog lays its own buttons out instead;
any dialog that cares about the order has to.

**An offscreen window cannot answer three questions**, and asking anyway produces tests
that pass on nothing. It is never activated, so `hasFocus()` is False for every widget
and `focusWidget()` is what records who asked. Nothing is shown, so `isVisible()` is
False whatever the code did and `isHidden()` is the state to assert. And a modal dialog
has nobody to dismiss it, so its `exec_` must be intercepted — per dialog class, not
just `QMessageBox` — or the suite hangs rather than fails.

**Never mix conda-forge with Anaconda `defaults`.** `environment.yml` pins
`nodefaults` for this reason: a mixed environment paired Anaconda's OpenCV with
conda-forge's protobuf and `cv2` would not load at all.

**Inference lives in a second environment, and that one is pip-first.** The main
environment is conda-first because conda's OpenCV and pip's cannot coexist.
albumentations, which the vendored preprocessing imports, requires
`opencv-python-headless` — a third provider of `cv2`, and one no ordering of install
commands avoids. So `environment-ml.yml` takes only python and pip from conda and the
whole stack from the `[ml]` extra, where there is exactly one OpenCV. It also lets the
torch build be chosen per machine, which conda-forge cannot express:
`AM_SEG_TORCH_INDEX_URL` selects a wheel index. `scripts/setup_ml.sh` builds it.

**The inference environment is a superset, not a peer.** It gets the window too,
via the `qt` extra — Qt binding and matplotlib, no OpenCV, since it already has the
headless one. Built first without it, which was wrong: it forced a switch of
environment in the middle of the workflow, between curating a set and predicting
it. The curation environment is the *lighter* one, for someone who never predicts;
it is not the more capable one. Headless OpenCV is also the better partner for
PySide6, `opencv-python` carrying Qt plugins of its own.

**Nothing outside `ml/` may import torch.** The subpackage is the boundary, and
`ml/environment.py` answers "is inference possible here" without importing it, the way
`io/image.py` does for the image stack. `ml/preprocessing.py` is a verbatim copy of
AM_training's module, excluded from ruff: reformatting it would destroy the only cheap
answer to "is this still the preprocessing the model was trained with" — `diff`.

---

## Inference

**A fact recoverable from the weights is never declared.** A checkpoint is a bare
`state_dict`, and its class count, input channels and tensor names are all readable
from it. Declaring those in a file beside it adds a second statement that can
disagree with the first, and the disagreement would be invisible: a description
naming four classes over a five-class head loads whatever fits. So `ml/model.py`
reads them, and `load_state_dict(strict=True)` is what proves the description's
architecture and the weights are the same model. A mismatch stops at the load
rather than at the first odd-looking label.

**What the weights cannot state is declared in a `.yaml` beside them.**
Preprocessing settings and the architecture name are not in the file, and every one
of them shows itself immediately if wrong, which is what makes a default safe.

**Which modality a model is for is not recorded anywhere, deliberately.** Nothing in
a checkpoint says, and `AM_SEM_All_V01.pth` is whatever whoever trained it chose to
type — it is not evidence, and reading it as evidence would make the software's
answer depend on someone else's file-naming habit. Recording it in the sidecar
instead was tried and dropped: it moves the guess rather than removing it, and a
declaration nothing can check is a claim wearing the clothes of a fact.

**So the selection states it, and a selection spanning both modalities is refused.**
One model is for one modality; a selection naming both names nothing, and the run
stops with the two it found. This is a better guard than the one it replaced, because
it is about something the window can show you — the edit target and the images under
it — rather than a line in a file you would have to open. Every report leads with the
modality, so what is about to be predicted is never left to be inferred from a list
of paths.

**Predicting with the wrong model is cheap to undo**, which is what makes that guard
sufficient rather than merely defensible. Resumption keys on the model and run name,
so predicting the same images with the right model replaces the wrong ones without
`--overwrite`: an accident costs the inference time and nothing else.

**`decoder_attention_type="scse"` is not part of these models.** The training call
passes it, smp's `FPN` has no such parameter, and it is swallowed into `**kwargs` and
forwarded to the encoder, which ignores it. The checkpoints contain no attention
tensors, and a plain FPN strict-loads all 1015 of them. Build plain FPN; do not carry
the argument over on the assumption that it did something.

**One prediction per label group, not per image.** The label is the unit of
workflow state everywhere else here, and a prediction exists to become a label, so
a group gets one — made from `label_groups.primary_record`, the same image the
curation window shows and a training export keeps. Predicting the other conditions
would cost inference to produce worse pseudo-labels: `SEM_8_fi_1` is one frame
integration and far noisier than the `SEM_8` it shares a label with. It also removes
a hazard rather than managing one — conditions carry identical filenames into a
shared predictions directory, so predicting all of them would have each write
replace the last. On one lamella of the reference collection this is 6 predictions
instead of 15.

**Predictions are written at model resolution.** A 2048x3072 image yields a 1024x1536
prediction — which is what the segmentation pipeline's own `raw/` already holds, at
12 kB against the 6 MB an upsampled one would cost. `convert-predictions` resizes to
the image when it makes the label, so upsampling earlier would only be undone.
