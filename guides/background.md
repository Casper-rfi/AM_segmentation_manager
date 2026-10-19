# Background

This page explains how AM-seg reads your data, decides what shares a label,
keeps your work safe and records its history. None of it is needed to start:
the [user guide](user_guide.md) is. Read a section when you want to know why
AM-seg behaves the way it does.

## Projects and layouts

A **project** is any folder holding one of two files:

| File | Written by | Means |
|---|---|---|
| `experiment.yaml` | Autolamella | an Autolamella project, found on its own |
| `AM_metadata.yaml` | AM-seg | a project you added, or the description of one |

Inside a project, images sit under a **lamella** and a **stage**. Autolamella
has laid these out differently over the years, and each layout is read as it
is:

| Layout | Written |
|---|---|
| `project/lamella/Milling/Polishing/adaptive_polish_<timestamp>/sem/` | 2026 |
| `project/lamella/adaptive_polish_<timestamp>/sem/` | 2025 |
| `project/lamella/adaptive_polish/sem/` | 2024 |
| `project/lamella/SEM_8/` | no stage folder |
| `project/SEM/` | no lamella or stage folder |

Where a layout leaves a level out, AM-seg reports it as missing.

## Series and beams

Autolamella writes more than the Adaptive Polishing images, and AM-seg files
every image under one of four **series**:

| Series | Holds |
|---|---|
| Adaptive Polishing | the `sem/` and `fib/` folders of a polishing run |
| reference | the images taken before and after each milling task |
| alignment | the images of each beam alignment |
| other | everything else, such as grid overviews. Never labelled. |

Each image was taken with the electron beam (**SEM**) or the ion beam (**FIB**).

## Labels and predictions

A **label** is an image the size of the one it describes, in which each pixel's
value names a class: background, lamella, and so on. The classes and their
colours come from `segmentation_config_AM.yaml`: the project's own, or the
one AM-seg comes with.

Labels sit beside their images, in a folder named after the image folder:
`sem_labels/` beside `sem/`. Reference and alignment images, which sit loose in
a lamella folder, have theirs in a folder named for their series and beam, such
as `reference_fib_labels/`. **Predictions**, what a model makes of an image, sit
in `sem_predictions/`, and become labels only when you convert them.

**A label file existing does not mean anything was labelled.** The
segmentation software writes an empty label for every image it is given, so
AM-seg looks inside each one. An empty label is never exported as a training
pair.

## Families of imaging conditions

One acquisition can be imaged under several **conditions**, each in a folder
of its own, such as `sem` and `sem_fi4`. They show one scene, so they share one
label. Conditions that share a label form a **family**.

The folder name says which family a condition is in: everything before the
first `_`.

- `sem_fi4` is in the family `sem`, and shares the labels in `sem_labels/`.
- `sem-5kV` and `sem-5kV_fi4` form a family of their own, and share
  `sem-5kV_labels/`.

When naming folders at acquisition, use letters, digits, `-` and `.` in a
family's name, and `_` only to start the condition. Write `sem-5kV_fi4`, not
`sem_5kV_fi4`, which AM-seg reads as condition `5kV_fi4` of `sem`.

To correct a condition in the wrong family:

1. Open the **Families** tab, beside *State* on Curate.
2. Select the condition to see it with the label on top.
3. Switch on *Enable editing*, choose its **Family** and press **Save**.

No folder is renamed. The correction is kept in the project's
`AM_families.yaml`, for the whole project or for one lamella. **Inspect
layout**, on Add data, reports two conditions of one family whose images differ
in size, which one label cannot fit.

Each family has a **primary condition**, marked with a star. It is the one
predicted from, measured and shown first. It is the condition without a suffix,
such as `sem`, or else the first by name. Choose another with **Set primary**.

## States

Conditions that share a label form one **label group**. A group is in exactly
one of three states: selected, deselected or corrected.

- A state belongs to the group, never to one image in it, so changing one
  condition changes all of them.
- Deselecting a corrected group withdraws the claim that it is corrected. The
  proofreading itself stays in the label.
- States are kept in an `AM_state.yaml` in each stage folder, which lists the
  groups that are not selected. Images added later arrive selected.

## What AM-seg keeps safe

- **Nothing on Curate changes a file until you switch on *Enable editing*.**
  The same goes for the fields on Add data, which have a switch of their own.
- **The viewer cannot change your data.** `am-seg gui --viewer` opens Curate
  alone, without the switch, and nothing in it writes into the collection. To
  be certain, give that computer the collection read-only. The viewer reads it
  all the same.
- **A batch is checked whole before any of it is written.** If one file in a
  hundred has changed since you looked, none is written, and you are told which.
- **You see what will happen first.** Operations that change files show what
  they would do before doing it, and larger state changes ask first.
- **A corrected label is never overwritten by an automated operation.** A label
  any operation replaces is kept first under `AM_archive/`. Only a model's
  untouched output is not kept, as the model can make it again.

## History

### What is recorded

The first time anything changes a project, AM-seg starts a record in an
`AM_history.jsonl` in the project folder. From then on every change is added:
a state changed, a label saved, converted, restored or merged, a metadata field
edited, or a condition moved between families. Looking never writes.

Each change is recorded under the name set in *Settings* (the gear, or F2).
With none set, it is recorded under no name. The record lives inside each
project, so **depositing a collection deposits its history**, with the names
in it.

*Compare versions*, on the Label tab, lists the versions of the label on
screen, with who made each and when.

### Savepoints

A **savepoint** names the state of every label at a moment you choose, such as
the end of a round of proofreading before training a model on it.

To take one, press **Savepoint...** in the *Maintenance* card on Add data, and
give it a name you will recognise months later.

Restoring a savepoint restores each label and the state of its images.

- A state is restored only together with its label. Where a label cannot be
  restored, its images keep their state.
- Tick **Also restore project metadata** to restore the metadata fields edited
  in AM-seg since.
- Family corrections are not restored. The preview lists those changed since,
  to change back with *Families* on Curate.

A savepoint can return only what was kept. A label only a model had written,
and nobody touched, is not kept unless you tick **Also keep labels the model
made** when taking the savepoint. This costs disk space.

On a collection that already held work before AM-seg recorded anything, run
`am-seg init-history` once, so a savepoint covers every project rather than
only those changed since. See
[the command line](command_line.md#record-and-undo).

### Undoing work

| To restore | Use |
|---|---|
| what the last conversion or merge replaced, for a selection | *Revert...* on Curate |
| a selection, to a savepoint | *Revert...* on Curate |
| every label under the folder, to a savepoint | *Restore savepoint...* on Add data |
| one label, to any kept version | *Compare versions*, then *Restore...*, on Label |

A label edited since the version it would go back to is reported and left
alone, and a corrected label is not replaced unless you name it. What is there
now is kept first, so restoring a version can itself be undone.

Kept versions add up. **Reclaim space...**, on Add data, deletes the versions
no savepoint needs. Nothing since the most recent savepoint is deleted, and the
record still says what happened.

### Working on a copy

To proofread on another computer, copy a selection out with **Export images and
labels...** on Export. The copy is a collection of its own. It carries the
history of every project in it and an `AM_export.yaml` saying where it came
from.

To bring the work back:

1. Open the original collection.
2. Switch on *Enable editing* on Curate.
3. Press **Merge a copy back...** on Export.

The merge compares both sides against the moment the copy was taken, so a
change made on only one side is simply taken. A **conflict** is something both
sides changed differently. Decide each one, or a whole folder, project or
lamella at once. Nothing is merged until every conflict is decided, and a label
corrected here is never overwritten unless you choose it.

## Training sets

**Training CSV...** on Export writes `<name>_train.csv` and `<name>_val.csv`,
split by **Training**, 80 % by default. With **Folds** at 2 or more it writes
that many pairs instead, for cross-validation, so every image is validated
exactly once.

### How the split is made

- **Keep together** sets what is never divided between training and
  validation: each lamella by default, each project, or only each acquisition.
- With **Seed** on, the same selection always splits the same way.
- To see what a split is made of before writing it, set **Breakdown** to
  *training split*, alone or against another property such as the lamella
  type, microscope model or accelerating voltage. The table counts images:
  *train*, *validation*, *left out*, and *not exported* for an image with no
  label worth pairing.

### Train on proofread and predicted labels together

1. Set the state filter to **corrected + selected**.
2. Leave **Validate on corrected only** switched on (on by default).

The validation list is then the one the corrected labels alone give with the
same seed, so a model trained with the predicted labels and one trained without
are scored on the same images. A predicted label is left out, and counted,
where what is kept together with it is validated on. A lamella's images are
near-duplicates, and training on some while scoring on others flatters the
model.

### Keep to an earlier split

**Validation from...** keeps to the split of an earlier export, chosen by its
manifest. Training again after more proofreading, or with more data, then
scores the new model on nothing an earlier one trained on.

- A lamella the earlier export validated on validates again, with whatever of
  it has been proofread since.
- A lamella it trained on trains again.
- Only lamellae it did not hold are split by **Training** and **Seed**.

To compare an older model on the grown validation list, validate it on the new
export in Train.

### Balance the split

**Balance by** gives validation its share of each value of a property, such as
each kind of lamella. The lamellae of each kind are split among themselves, so
a rare kind is not left out of validation by chance.

- A kind that two or more lamellae hold always validates on one.
- A kind that only one lamella holds trains.

We recommend balancing before training for the first time, as **Validation
from...** keeps whatever the first split decided. With both set, a lamella the
earlier export did not hold goes to whichever side its kind is short on, so a
rare kind arriving one lamella at a time still reaches validation.

### What is written

| File | Holds |
|---|---|
| `<name>_train.csv`, `<name>_val.csv` | image and label paths of each side |
| `<name>_images.csv` | every image once, with its side (in folds, the fold it validates in) and its metadata |
| `<name>_manifest.yaml` | what the export was made from |

`<name>_images.csv` lets a model be scored later on a part of the set, such as
by instrument, voltage or kind of lamella. No file names a folder of your
computer. Every path is written from the collection, and the manifest finds the
collection from its own folder, so the files can be deposited or moved with the
collection as they are.

With one beam chosen, each row is an image and its label. With both, each row
is one acquisition, with its SEM and FIB images and labels side by side. A
label that is empty, or will not open, is left out and counted in the manifest.

An export never replaces a file already there unless you say so, because a
training split written over by accident is gone, with the seed that made it.

## What AM-seg writes

Everything AM-seg knows about your data is in files beside it. There is no
database, and you can read every one of these files yourself.

| File or folder | Where | Holds |
|---|---|---|
| `AM_metadata.yaml` | project folder | the project's description |
| `AM_state.yaml` | stage folder | the state of every group that is not selected |
| `AM_families.yaml` | project folder | corrections to which conditions share a label |
| `<family>_labels/` | beside the images | labels |
| `<family>_predictions/` | beside the images | predictions, with `am_seg_predictions.yaml` naming the model |
| `AM_history.jsonl` | project folder | who changed what, and when |
| `AM_archive/` | project folder | every label version an operation replaced |
| `AM_export.yaml` | an exported copy | where the copy came from, so it can be merged back |

Outside the collection, `~/.am_segmentation_manager/` holds the window's log,
`gui.log`, and the name your changes are recorded under.
