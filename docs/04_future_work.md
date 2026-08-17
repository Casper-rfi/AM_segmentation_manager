# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-17.

A holding area for things deliberately not being designed yet. Each entry
records enough to stop the idea being lost, and the questions that would have to
be answered before it could be built — nothing more. Detail added here before it
is needed becomes detail that has to be maintained and re-argued.

---

## 1. Launch the segmentation software from the GUI

**What you want:** a button that opens the external labelling UI on the current
adaptive-milling stage and imaging modality, so proofreading starts without
copying paths by hand.

**Why it is not designed yet:** you noted it *"may require changes in the
fibsemOS labeling UI"*. That is the crux. Until it is known whether that UI can
be launched with arguments — and if so, which — there is nothing here to design.
Everything else is easy.

### What is already in place

* Image and label directory resolution: `layout.py`, via the stage's
  `image_sets`, each of which already carries the label directory it resolves to.
  No new path logic.
* **Copy-path is already in the window's context menu, and is the useful 80 % of
  this feature.** Copy the path, paste it into the other tool. Find out how much
  of the pain that removes before building process launching.

### Questions to answer before designing

1. Can the fibsemOS labelling UI be launched with a working directory and
   a config, or does it only start interactively?
2. If it needs changes, is that repository yours to change, and on what
   timescale?
3. Does it exit cleanly, and should this package wait for it? Waiting means
   knowing when to rescan; not waiting means the curation window shows stale
   state until the user presses Scan.

### On passing the config

You asked whether `segmentation_config_AM.yaml` should be passed from package
code rather than as a file. **Pass the file.** The package already resolves the
config through `config.py` — scan folder first, packaged copy as fallback — so
it can hand over a concrete path, including a temporary one written from
in-memory values if the other tool needs a file. Constructing the settings
programmatically would require the two tools to share a schema in code, which
couples their release cycles. A file is a stable, inspectable interface, and it
is the one that already exists.

### Constraint

Launching an external process that writes into label directories sits outside
this package's safety guarantees — nothing it does can be validated in advance
or dry-run. At minimum it must never be pointed at a `corrected/` directory
without an explicit confirmation, on the same footing as every other operation
that could touch corrected work.

---

## 2. ML inference to generate predictions

**What you want:** create predictions using the existing `AM_training` package,
from the CLI and then the GUI, closing the loop — predict, convert to labels,
proofread, export, retrain. Raw predictions are what matters here: this package
converts only those, for the reason in [`decisions.md`](decisions.md) §
Filesystem conventions.

### What is already in place

`operations/predictions.py` already converts predictions into labels, including
resizing when prediction and image dimensions differ. The output side is done.
What is missing is producing the predictions in the first place.

### Questions to answer before designing

1. **Is `AM_training` a dependency or a peer?** Importing it makes this package
   depend on a training stack (torch and friends) — a large dependency for a
   file-management tool, and one that would have to be an optional extra at
   minimum. Invoking it as a subprocess keeps them independent. Leaning towards
   subprocess, for the same reason as §1: a stable interface beats a shared
   schema in code.
2. **Where do model weights live, and how is the model used recorded?** A
   prediction is only interpretable if you know which model made it. This is a
   provenance question, and it is more important than the mechanics — a
   `predictions/` directory with no record of the model that filled it is a
   problem that gets worse with time. It may deserve a small sidecar file
   alongside the predictions, in the spirit of `AM_metadata.yaml`.
3. **GPU availability and runtime.** Inference over thousands of images is
   minutes to hours, not seconds. That means progress reporting, interruption,
   and resumption — none of which any existing operation in this package needs.
   It is the first genuinely long-running operation, and it will not fit the
   current validate-then-act shape without thought.
4. **Does it overwrite existing predictions?** By `CLAUDE.md`'s rules, no — not
   without an explicit flag. Label conversion has since set the precedent to
   follow: tier the answer by what replacing the file would cost, rather than
   by one flag covering every case. See [`decisions.md`](decisions.md) §
   Workflow state.

### Why it is last

Everything else in the roadmap improves work you are doing today. This adds a
capability you currently have elsewhere. It is also the largest single piece,
and the one most likely to pull heavy dependencies into a package that is
currently near-stdlib. Worth doing eventually; not worth doing before the
metadata work, which is what makes the training sets it would feed actually
selectable.

---

## 3. Deposition to scientific data repositories

**Confirmed as a goal on 2026-08-14, and it is the package's end state** — not
a nice-to-have at the edge of it. The whole chain leads here: curate the data,
proofread the labels, export training lists, quantify, and then deposit the
labelled curated dataset publicly with three CSVs beside it (metadata, label
quantifications, model training). All three of those exports are built.

**The data is deposited in the layout it already has.** Target: Zenodo. There
is no plan to re-format anything after labelling, and no repository-mandated
directory structure or metadata schema to conform to — Zenodo takes files plus
a record of title, authors, licence and description. Consumers will either use
this package to interpret the data or take the training CSVs and ignore the
rest.

So the remaining work is small, and is about making the deposit
*interpretable* rather than about moving files:

* **`segmentation_config_AM.yaml` must travel with the data.** This is the one
  real dependency. A label is a TIFF of small integers; without the file saying
  3 means crack/hole it is uninterpretable, and the class definitions live in
  the package rather than in the data. Everything else needed is already inside
  the tree — `experiment.yaml` and `AM_metadata.yaml` sit in each project.
* **A completeness check before depositing** — no `TODO` metadata placeholders,
  no empty labels among the selected images, no images disagreeing with their
  shared label. `inspect-layout` already reports all three; what is missing is
  a single "is this ready to deposit" verdict over them.
* **A manifest with checksums**, so a 40 GB transfer can be verified.
* **The three CSVs**, included as they are.

Practical constraint worth knowing before the first deposit: Zenodo's default
ceiling is 50 GB per record. The reference collection is 40 GB, which fits, but
not with much room — a larger collection would need either a request to Zenodo
or splitting across records.

---

## 4. Importing existing labels into the project hierarchy

**What you want:** the labels that live in folder structures you created by hand,
brought back into the project hierarchy where `layout.py` expects them.

**Why it is not designed yet:** designing a matching strategy without having seen
the source structures would be speculation, and a bulk filesystem-writing command
built on a guessed input shape is the worst thing to have in a package whose point
is not losing data.

### What is already in place

* **The destination is unambiguous.** `layout.labels_directory_for(...)` gives the
  right target for any image, so a helper script — in this package or outside it —
  never has to reimplement the convention.
* **Verification is available now.** `inspect-layout` reports images with no label
  and labels matching no image, so the result of any import can be checked rather
  than trusted.

### The shape it would take

A dry-run-first command matching on filename within a selected scope, refusing to
overwrite, never deleting from the source, and reporting unmatched and ambiguous
cases before writing anything. That is a small amount of work *given* `layout.py`,
and much better specified once there is a real input to look at.

---

## 5. Label quantification in the window

**What you want:** `quantify-labels` where the labels are on screen, so a surface
area can be looked at against the image it was measured from.

**Why it is not built:** it opens every label in the selection, which puts it in a
different class from the operations that moved into the window — those report a
list of files, this reports a table of numbers per class per lamella, and a table
needs a widget, a sort order and somewhere to put it. The Export tab's breakdown
already answers the cheap version of the question ("how much of what do I have"),
and the command writes a CSV that a spreadsheet reads better than any pane here
would.

`stats` is out for the same reason and one more: it is a report over the whole
collection, and the rule in [`decisions.md`](decisions.md) § Architecture puts
whole-collection sweeps on the command line.

### Questions to answer before designing

* Is the wanted thing a table, or a per-frame number beside the image? The second
  is much cheaper and might be the actual need.
* Measuring is per label file and slow enough to need the worker thread and a
  cache. Where would the results live between refreshes — recomputed, or held on
  the controller like the frame cache?
