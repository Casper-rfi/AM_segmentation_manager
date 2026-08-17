# Decisions

Why the software is built the way it is, and facts that cost time to rediscover.
Every entry is here because it is **not** derivable from reading the code: the code
shows what happens, not what breaks if it changes.

Work that is *not* built yet belongs in [`04_future_work.md`](04_future_work.md), not
here. If something below becomes wrong, correct it in place.

---

## Architecture

**Qt via `qtpy`, not napari.** napari is the obvious choice in this domain and is
wrong for the main viewer: an image layer wants a uniform `(N, Y, X)` stack, but stage
image dimensions are not equal across a lamella, and the plot PNGs have a different
aspect ratio again. `qtpy` rather than a binding directly, because loading two Qt
bindings into one process crashes in hard-to-debug ways.

**No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python
controllers. This is what makes navigation, cache eviction and state expansion
testable without a display server. It is the one structural rule of the UI; adding a
tab is not a reason to bend it.

**An operation belongs in the window when its input is the selection on screen; it
stays on the command line when it is a sweep over the whole collection.** Naming a
selection twice — once by ticking it, once as command-line filters — is where the two
diverge and the wrong data gets moved. The Add data commands are in the window despite
being sweeps, because there the sweep *is* the collection and there is nothing to select.
`stats` and `quantify-labels` stay out: their output is a table nobody is going to act on
one row at a time. [`00_roadmap.md`](00_roadmap.md) says where each command ended up.

Each of these previews and then applies what it previewed, as two runs of the same
function rather than one run replayed. A preview held across a dialog describes a
filesystem the user has had time to change.

**Label editing is out of scope.** Proofreading stays in the external segmentation
software; this package decides *which* images get proofread and records when they are
done.

**No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few
lines with no dependency, over thousands of records rather than millions. pandas belongs
here only once something genuinely needs it.

---

## Filesystem conventions

**A label directory is `<image-directory>_labels`, a sibling of the image directory.**
`layout.py` states the resolution order; what it cannot say is why that shape. It
makes image-dir → label-dir a pure string function in both directions with no lookup
table, which is what lets one module hold the convention at all. A `labels/<image-dir>/`
subtree would push the `deselected/` and `corrected/` state folders three deep and gain
nothing. Conditions share a label by default; giving one its own means creating the
directory, not configuring anything. `sem/` is not a special case in this: its private
and shared names coincide, which is why one rule covers the plain layout and the
five-condition one alike.

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

**The label is the unit of state, and its location is authoritative.** A state change
expands to the whole label group before anything moves. Where a group has no label
yet, the image's own location decides. A group whose images disagree with its label is
reported as inconsistent, never silently resolved; `repair-state` moves the images to
follow the label, never the reverse, because the label holds the proofread work.

Group expansion sits in front of the existing `transition_records` logic, so every
safety property applies unchanged — only the set they apply to grows. Callers get the
expanded set back, because with expansion the blast radius of a state change is no
longer obvious from the command line.

**Overwriting a label is tiered by what replacing it would cost, not by one flag.** A
run that writes nothing has to say why, or it cannot be told from a broken one. So
conversion sorts destinations three ways: an empty placeholder costs nothing to
replace (see *A label file existing is not evidence of annotation* below); a
pseudo-label holding annotation is regenerable machine output, replaced once the user
confirms; a label in `corrected/` is refused, and no argument to `plan_conversions`
reaches it.

That last tier is stricter than `compress-labels`, which *can* be confirmed for
corrected labels. The difference is what each does to the pixels: compression
preserves them exactly, conversion replaces proofread work with a machine guess. A
dialog is the right guard for the first and too weak for the second, so re-converting
a corrected label means moving the image out of `corrected/` by hand.

Conversion writes to a temporary file and renames it into place, as
`label_compression` already did. Now that a write can land on a file that exists,
failing part way through would truncate a readable label.

**What a state change applies to is a set the user builds, and the finest level
present wins.** Picked-out frames beat ticked lamellae, which beat ticked projects,
and with none of them it is the frame on screen. Finest-first is chosen for its
failure mode, not its ergonomics: a selection the user has forgotten about makes the
next change *smaller* than they intended, where the opposite order would make it
larger. Ticks are keyed by resolved path, since a row means a different lamella as
soon as the project changes; frame indices are cleared with the lamella, because that
is all an index means here. Everything reaches `transition_records` through
`scoped_records()`, so group expansion and the corrected-data guard apply to a ticked
selection unchanged.

**A tick is Qt's own checkbox, not the status chip.** Curation progress and "I am about
to change this" are orthogonal — a finished lamella can be ticked and an untouched one
need not be — so one mark cannot carry both without becoming ambiguous in exactly the
situation that matters. Two marks side by side also gets keyboard toggling for free.

**A label file existing is not evidence of annotation.** The external software writes
a label for every image in its input folder, so a 2048×3072 file of class 0 is
indistinguishable from finished work by any check that only looks at the filesystem. A
training export that includes one pairs an image with a blank mask, which teaches a
model that the structure is background.

Hence the tri-state `is_label_file_empty()`, where **None means "cannot tell"** and is
deliberately distinct from True: reporting "not annotated" for a label nobody could
read would be a guess presented as a fact.

---

## Metadata

**Nothing is inherited from parent directories.** A collection directory groups
projects by session or storage convenience, so lamellae in one *directory* may be
different samples, and resolving a field from a parent would attribute the wrong
sample to a project invisibly. Lamellae in one *project* do share a sample, which is
why the project is the right level for this metadata.

**The Add data tab suggests, and a suggestion is never a value.** It offers the
values other projects already use for a field, and the name of the folder a project's
own YAML files sit in for `project`. Both are computed in the GUI controller and only
reach a file when the user saves, so `build_experiment_metadata` still resolves from
the two files and nothing else — the rule above is intact. The folder read for
`project` is the project's own, never a parent, and the directory is preferred over
`experiment.yaml`'s `name` because a project copied to a second location keeps the
recorded name of the first.

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
nothing downstream converts anything. A by-voltage breakdown rounds to 0.1 kV, because a
header sometimes records a measured voltage rather than the set point; the exported
value does not.

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

**A locked editor keeps its normal text colour.** The Add data tab opens locked and is
mostly used for reading, and Qt's disabled grey puts every value at low contrast. The
stylesheet dulls the *background* of a disabled input instead and leaves the text dark,
so a field still reads as locked. Buttons and radio groups grey out normally: they are
controls you cannot use, not values you are reading.

**The metadata list elides; it must not scroll sideways.** Its rows carry relative
paths that differ in the timestamp they *end* with, and a `QListWidget` with a
horizontal scrollbar scrolls instead of eliding — which cut a session's projects off
at the edge into identical-looking rows. `ElideMiddle` plus `ScrollBarAlwaysOff` keeps
both ends; the tooltip carries the whole label.

**A list re-tints the icon of the row you are on.** Qt draws a selected row's icon
in `QIcon::Selected` mode, and the variant it generates for that mode is blended with
the highlight colour — so a status chip whose colour *is* the information came out a
different colour on exactly the row being looked at. `chip_icon()` supplies the pixmap
for every mode and state, which leaves Qt nothing to generate. Any future icon carrying
meaning rather than decoration has to go through it.

**`QScrollArea.setWidget` shows whatever it is handed**, so hiding a widget and then
putting it in a scroller leaves it on screen — which is how the Export tab's acquisition
filters, meant to appear only once the headers are read, were visible from the start.
Hide the scroller, and give the column a widget to hold its spare height: a layout shares
that height between whatever is still visible, and a stretch cannot be hidden again.

**A long operation runs on a thread, and there is nothing to cancel.** Recompressing
a project's labels is minutes, and Qt draws nothing while the thread owning the
widgets is busy — so the work goes to `gui/worker.py` and the window shows a busy
dialog. No cancel button: these operations validate the complete set of changes before
performing any of them, and stopping one halfway leaves exactly the half-applied
collection that rule exists to prevent.

**Timeline label lines are violet (`#a78bfa`); empty placeholders are grey
(`#8a8a8a`).** Each modality's label line sits outside its state row so the two state
rows stay adjacent for comparison. The placeholder must be *lighter* than the
background, not a dim version of the annotation colour: against a near-black
background a dark colour is indistinguishable from nothing drawn, which collapses
"placeholder" into "no label" — the exact distinction the line exists to make. A test
asserts the lightness ordering (background < empty < annotated).

---

## Environment

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

**The development and user environments differ** — PyQt5 here, PySide6 there, and a
different macOS. Bugs keep reaching the user through that gap, which is why the entries
above prefer one code path over a platform's own. Run the suite under both bindings and
a non-English locale before releasing.

**Never mix conda-forge with Anaconda `defaults`.** `environment.yml` pins
`nodefaults` for this reason: a mixed environment paired Anaconda's OpenCV with
conda-forge's protobuf and `cv2` would not load at all.
