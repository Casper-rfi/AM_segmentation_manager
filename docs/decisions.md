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

**The window forces the Fusion style rather than taking the platform's.** The two
bindings draw the same widgets differently, and the development and user environments
differ by binding — so a window that inherits its style is a window nobody has both
seen. Fusion is built into Qt, so this costs no dependency.

**No Qt below `gui/widgets/`.** Everything the GUI knows lives in plain Python
controllers. This is what makes navigation, cache eviction and state expansion
testable without a display server. It is the one structural rule of the UI; adding a
tab is not a reason to bend it.

**An operation belongs in the window when its input is the selection on screen; it
stays on the command line when it is a sweep over the whole collection.** Naming a
selection twice — once by ticking it, once as command-line filters — is where the two
diverge and the wrong data gets moved. By this test `convert-predictions`,
`compress-labels` and `repair-state` moved into the Curate tab once ticking made "the
selection" a real thing, and `inspect-layout` and `init-project` into Add data, where
the sweep *is* the collection and there is nothing to select. `stats` and
`quantify-labels` did not: both are whole-collection reports, and their output is a
table nobody is going to act on one row at a time.

Each of these previews and then applies what it previewed, as two runs of the same
function rather than one run replayed. A preview held across a dialog describes a
filesystem the user has had time to change.

**Label editing is out of scope.** Proofreading stays in the external segmentation
software; this package decides *which* images get proofread and records when they are
done.

**No dataframe.** `collections.Counter` does everything `aggregate.py` needs in a few
lines with no dependency, over thousands of records rather than millions. pandas was
removed from the requirements precisely because nothing used it, and it should return
only if something genuinely needs it.

---

## Filesystem conventions

**A label directory is `<image-directory>_labels`, a sibling of the image directory.**
`layout.py` states the resolution order; what it cannot say is why that shape. It
makes image-dir → label-dir a pure string function in both directions with no lookup
table, which is what lets one module hold the convention instead of the 200-odd
hardcoded strings it replaced. A `labels/<image-dir>/` subtree would push the
`deselected/` and `corrected/` state folders three deep and gain nothing. Conditions
share a label by default; giving one its own means creating the directory, not
configuring anything.

**Image directories keep their Autolamella names.** We do not own `sem` and `fib`;
renaming them would break reopening the project in Autolamella. Label directories are
ours, so they are the only ones anything here renames.

**Renaming pre-convention label directories is a script, not a command.** The
package reads `labels`, `labels_fib` and `labels_FIB` indefinitely, and
`resolve_labels_directory` returns whichever directory an image set already has —
so a new label lands beside the old ones and a project never ends up holding two
namings. That leaves cosmetic tidiness as the only reason to rename, over a fixed
handful of datasets belonging to one person. It lives in
`scripts/rename_legacy_label_dirs.py`, where a layout that does not fit can be
handled by editing the script rather than by generalising a command.

**The variant is an opaque string and is never parsed.** `sem_5kv`, `sem_defocus`,
`sem_lowmag` all work with no code change, because authoritative imaging parameters
come from the image headers. A variant may not be named `labels`, `deselected`,
`corrected` or `predictions` — those collide with the label naming rule, so
`layout.py` validates it and `inspect-layout` reports a violation rather than
mis-indexing.

**A project is a directory containing `experiment.yaml` or `AM_metadata.yaml`.**
`init-project` writes the latter rather than fabricating the former, because
`experiment.yaml` is Autolamella's file: writing one misrepresents provenance and
could collide if Autolamella ever opens that directory.

**A project's identity is its path relative to the scan root, not its name.** The
reference collection holds two different projects whose name, `created_at` and
recorded path in `experiment.yaml` are all identical — keying by name silently drops
one.

**The lamella and stage levels are not always directories.** Older datasets collapse
them; the missing level is synthesised in memory with the parent standing in, and the
model records that it is synthetic. Data on disk is never reorganised to fit the
newest layout.

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

**A tick is Qt's own checkbox, not the status chip.** Merging them was considered:
Qt draws the check indicator in its own column ahead of the icon, so it would need a
custom item delegate. The cost is not the reason to refuse. Curation progress and "I
am about to change this" are orthogonal — a finished lamella can be ticked and an
untouched one need not be — and one mark cannot carry both without becoming
ambiguous in exactly the situation that matters. Two marks side by side also gets
keyboard toggling for free.

**A label file existing is not evidence of annotation.** The external software writes
a label for every image in its input folder, so a 2048×3072 file of class 0 is
indistinguishable from finished work by any check that only looks at the filesystem. A
training export that includes one pairs an image with a blank mask, which teaches a
model that the structure is background.

`is_label_file_empty()` returns True, False, or **None when it cannot tell** (no image
stack, unreadable file). None is deliberately distinct from True: reporting "not
annotated" for a label nobody could read would be a guess presented as a fact.

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

**Every image header carries settings for both beams.** `microscope_state.*` describes
the beam that took the image; `system.*` holds both beams' configured defaults.
Reading `system.ion.voltage` from an SEM image yields the FIB's 30 kV — plausible and
wrong. `image_metadata.FIELDS` confines beam-specific fields, including the fallback
search, to the beam that fired.

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
palette and a short stylesheet; nothing else in the window picks a chrome colour. The
order matters more than it looks: where a stylesheet rule and the palette disagree, the
rule wins, and a widget palette edited in Python is simply ignored. A per-widget
palette adjustment that kept locked fields readable was therefore silently overridden
the moment the stylesheet arrived, and the palette it set still asserted true in the
test. Anything about how something *renders* is asserted on the pixels for that reason.

**A locked editor keeps its normal text colour.** The Add data tab opens locked and is
mostly used for reading, so Qt's disabled grey put every value at low contrast —
measured at `#bebebe` on white, and at `lightness 170` when the stylesheet reintroduced
it. The stylesheet dulls the *background* of a disabled input instead and leaves the
text dark, so a field still reads as locked. Buttons and radio groups grey out
normally: they are controls you cannot use, not values you are reading.

**The metadata list elides; it must not scroll sideways.** Its rows carry relative
paths that differ in the timestamp they *end* with, and a `QListWidget` with a
horizontal scrollbar scrolls instead of eliding — which cut a session's projects off
at the edge into identical-looking rows. `ElideMiddle` plus `ScrollBarAlwaysOff` keeps
both ends; the tooltip carries the whole label.

**A list re-tints the icon of the row you are on.** Qt draws a selected row's icon
in `QIcon::Selected` mode, and the variant it generates for that mode is blended with
the highlight colour — so a status chip whose colour *is* the information came out as
a different colour on exactly the row being looked at (the corrected teal `#1a7f6e`
rendered as `#207a94`). `chip_icon()` supplies the pixmap for every mode and state,
which leaves Qt nothing to generate. Any future icon carrying meaning rather than
decoration has to go through it.

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

**The development and user environments differ** — PyQt5 here, PySide6 there. Four
bugs have reached the user through that gap. Run the suite under both bindings and a
non-English locale before releasing.

**Never mix conda-forge with Anaconda `defaults`.** `environment.yml` pins
`nodefaults` for this reason: a mixed environment paired Anaconda's OpenCV with
conda-forge's protobuf and `cv2` would not load at all.

**The sample layouts in `tests/conftest.py` are presets, not a taxonomy.**
`build_dataset` takes layout *axes* — lamella level, stage nesting, naming,
modalities, variants. A suite passing across all five named samples does not show that
every layout works; vary the axis the behaviour actually depends on.
