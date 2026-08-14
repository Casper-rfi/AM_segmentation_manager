# Future work

Status: **sketches, not designs.** Written 2026-08-14.

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
* `02_gui_improvements_plan.md` §2 adds copy-path to the context menu. **That is
  the useful 80 % of this feature and it costs almost nothing.** Copy the path,
  paste it into the other tool. Do that first and find out how much of the pain
  it removes before building process launching.

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

(Note `01_refactor_plan.md` §1.2 deletes the duplicate config at the repo root.
Resolution goes through `config.py` either way.)

### Constraint

Launching an external process that writes into label directories sits outside
this package's safety guarantees — nothing it does can be validated in advance
or dry-run. At minimum it must never be pointed at a `corrected/` directory
without an explicit confirmation, on the same footing as every other operation
that could touch corrected work.

---

## 2. ML inference to generate predictions

**What you want:** create raw and cleaned predictions using the existing
`AM_training` package, from the CLI and then the GUI, closing the loop —
predict, convert to labels, proofread, export, retrain.

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
   without an explicit flag.

### Why it is last

Everything else in the roadmap improves work you are doing today. This adds a
capability you currently have elsewhere. It is also the largest single piece,
and the one most likely to pull heavy dependencies into a package that is
currently near-stdlib. Worth doing eventually; not worth doing before the
metadata work, which is what makes the training sets it would feed actually
selectable.

---

## 3. Deposition to scientific data repositories

Listed in `CLAUDE.md` as a main function of the package. Nothing implements it,
and it appears nowhere in your notes.

Flagged in `01_refactor_plan.md` §3.1(d) as a decision to make: either it is
still a goal, in which case it needs a plan of its own and would substantially
reuse the metadata export in `03_metadata_plan.md` §6, or it is not, and the
claim should come out of `CLAUDE.md`.

If it is still a goal, the first question is which repository (EMPIAR is the
obvious candidate for FIB/SEM), because the required directory structure and
metadata schema come from the target and nothing can be designed without them.
