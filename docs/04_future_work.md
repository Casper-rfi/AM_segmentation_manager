# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-18.

A holding area for things deliberately not being designed yet. Each entry
records enough to stop the idea being lost, and the questions that would have to
be answered before it could be built — nothing more. Detail added here before it
is needed becomes detail that has to be maintained and re-argued.

---

## 2. ML inference to generate predictions

**What you want:** create predictions using the existing `AM_training` package,
from the CLI and then the GUI, closing the loop — predict, convert to labels,
proofread, export, retrain. Raw predictions are what matters here: this package
converts only those, for the reason in [`decisions.md`](decisions.md) §
Filesystem conventions. --> Strongly considering to build inference function
based off standalone scripts now, which are functionally equivalent to inference
in AM training package.

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

## 5. Label quantification in the window

**What you want:** `quantify-labels` where the labels are on screen, so a surface
area can be looked at against the image it was measured from.

**Why it is not built:** the Export tab already has the *counting* half —
**Measure labels** opens every label and the crack and area breakdowns group
images by what was found. What is left is the per-lamella table: numbers per
class per lamella, which needs a widget, a sort order and somewhere to put it,
and which a spreadsheet reads better from the CSV than any pane here would.

**The question to answer first:** is the wanted thing that table, or a per-frame
number beside the image being looked at? The second is much cheaper, and may be
the actual need. --> per-frame number beside the image
