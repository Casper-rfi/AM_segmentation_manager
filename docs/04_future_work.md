# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-18.

A holding area for things deliberately not being designed yet. Each entry
records enough to stop the idea being lost, and the questions that would have to
be answered before it could be built — nothing more. Detail added here before it
is needed becomes detail that has to be maintained and re-argued.

---

## 1. Labelling in the GUI

**What you want:** to proofread labels in this package's own window and stop
using the external fibsemOS labelling UI altogether. It is large, externally
maintained, cannot show every stage of one lamella together, writes
uncompressed labels, and creates a blank label for every image in its input
folder -- the behaviour the `deselected/` move exists to defeat. Model-assisted
labelling is the one feature not being reproduced, and is not wanted.

**What is built:** everything except saving. `operations/label_edit.py` holds
brush, flood fill, class locking and undo on numpy arrays, with no toolkit and
no filesystem; OpenCV supplies every drawing primitive, so none of it needs a
new dependency. The Label tab reads one lamella's selected and corrected frames
at full resolution and draws each as two stacked graphics items -- the image,
and the label as an indexed-8 image whose colour table carries the alpha -- so
the opacity control costs nothing and the label layer redraws without touching
the image. `compose_overlay` stays as it is for Curate, which shows a reduced
frame and never changes one. Brush and fill both work on the left button, with
space or the middle button panning instead; edits are held in a `LabelSession`
scoped to the lamella, PNG-encoded once the reviewer moves off a frame.

**What remains: the save.** Until it exists the tab says so, in its status line
and under F1: an edit lives in the window and nowhere else.

**Decisions already taken.** Save is explicit and covers every label edited in
the lamella, because the work is done one area of each image at a time rather
than one image at a time; pending edits are held PNG-encoded in memory (14 kB
for a 6 MB label), so the whole lamella fits. Saving asks whether to also mark
the saved set corrected, and a label approved unedited can be marked reviewed
without a token brush stroke. Editing is refused where a label's dimensions
disagree with its image, since saving the resampled array would rewrite the
annotation at the wrong scale.

**The consequence most easily missed:** with fibsemOS gone, nothing else
creates a blank label, so the editor must write one where none exists --
resolving the destination through `resolve_labels_directory` exactly as
`predictions.py` does. `compress-labels` becomes legacy-only, since anything
saved here is already LZW.

---

## 2. ML inference to generate predictions

**What you want:** create predictions using the existing `AM_training` package,
from the CLI and then the GUI, closing the loop — predict, convert to labels,
proofread, export, retrain. Raw predictions are what matters here: this package
converts only those, for the reason in [`decisions.md`](decisions.md) §
Filesystem conventions.

The output side is done: `operations/predictions.py` converts predictions into
labels, resizing where dimensions differ. What is missing is producing them.

**Questions to answer before designing:**

1. **Is `AM_training` a dependency or a peer?** Importing it pulls a training
   stack (torch and friends) into a file-management tool — an optional extra at
   minimum. A subprocess keeps them independent, and is the way this is leaning,
   for the same reason as §1: a stable interface beats a shared schema in code.
2. **Where do model weights live, and how is the model recorded?** A prediction
   is only interpretable if you know which model made it. This provenance
   question matters more than the mechanics — a `predictions/` directory with no
   record of what filled it is a problem that worsens with time — and may deserve
   a sidecar file, in the spirit of `AM_metadata.yaml`.
3. **GPU availability and runtime.** Inference over thousands of images is
   minutes to hours, so it needs progress, interruption and resumption, none of
   which any existing operation here needs. It will not fit the current
   validate-then-act shape without thought.
4. **Does it overwrite existing predictions?** By `CLAUDE.md`'s rules, not
   without an explicit flag, and label conversion set the precedent: tier the
   answer by what replacing the file would cost. [`decisions.md`](decisions.md)
   § Workflow state.

**Why it is last:** everything else improves work you do today; this adds a
capability you already have elsewhere. It is the largest piece and the one most
likely to pull a training stack into a near-stdlib package. Its one prerequisite
— being able to *select* the training sets it would feed — is now met.

---

## 4. Importing existing labels into the project hierarchy

**What you want:** the labels that live in folder structures you created by hand,
brought back into the project hierarchy where `layout.py` expects them. These will
be helper scripts, rather than a core feature of the package, and not GUI supported.

Two things make this small *given* `layout.py`: the destination is unambiguous,
since `layout.resolve_labels_directory(...)` gives the right target for any image, so
no script reimplements the convention; and `inspect-layout` already reports images
with no label and labels matching no image, so an import can be checked rather
than trusted.

**The shape it would take:** a dry-run-first command matching on filename within a
selected scope, refusing to overwrite, never deleting from the source, and
reporting unmatched and ambiguous cases before writing anything. Much better
specified once there is a real input to look at.

---

## 5. Jumping from the Export tab to the images it names

**What you want:** the Export tab finds the empty and unlabelled images and names
the lamella holding them — by filter and breakdown, and outright for the
corrected ones when marking a set corrected or exporting. Fixing them happens in
the Curate tab, and the only thing missing is being carried there: switching tabs
and navigating by hand is the whole of it.

**Why it is not designed yet:** the two tabs hold different kinds of selection.
Curate's is a ticked *set* of projects, lamellae or frames, and is the target of
an operation that writes to disk; Export's is a *predicate* over images. Handing
a predicate to a ticked set means choosing what it becomes on arrival — tick
every project containing a match, every lamella, or jump to the first frame and
tick nothing? Each is defensible, and the wrong one silently widens what a later
state change applies to, which is what the ticked set exists to prevent.

---

## 6. Label quantification in the window

**What you want:** `quantify-labels` where the labels are on screen, so a surface
area can be looked at against the image it was measured from.

**Why it is not built:** the Export tab already has the *counting* half —
**Measure labels** opens every label and the crack and area breakdowns group
images by what was found. What is left is the per-lamella table: numbers per
class per lamella, which needs a widget, a sort order and somewhere to put it,
and which a spreadsheet reads better from the CSV than any pane here would.

**The question to answer first:** is the wanted thing that table, or a per-frame
number beside the image being looked at? The second is much cheaper, and may be
the actual need.
