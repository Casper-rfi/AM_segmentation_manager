# Concepts

What am-seg works on, and the words the window and the commands use for it.

## The collection and its projects

A **collection** is any folder you point am-seg at. It holds **projects**, at
any depth, and a project is any folder holding one of two files:

| File | Written by | Means |
|---|---|---|
| `experiment.yaml` | Autolamella | an Autolamella project, found without being asked |
| `AM_metadata.yaml` | am-seg | a project you have named: a dataset from before Autolamella, or one whose `experiment.yaml` has gone |

*Add projects* on the Add data tab proposes the folders holding images that no
project covers, and writes the file for the ones you confirm; `am-seg
init-project` writes it from the command line.

Inside a project, images sit under a **lamella** and a **stage**. Autolamella
has laid these out differently over the years, and every layout back to 2023
is read as it is, without moving anything:

| Layout | Written |
|---|---|
| `project/lamella/Milling/Polishing/adaptive_polish_<timestamp>/sem/` | 2026 |
| `project/lamella/adaptive_polish_<timestamp>/sem/` | 2025 |
| `project/lamella/adaptive_polish/sem/` | 2024 |
| `project/lamella/SEM_8/` | no stage folder |
| `project/SEM/` | no lamella or stage folder |

Where a layout leaves a level out, am-seg reports that level as missing
rather than inventing one. `am-seg inspect-layout <folder>` shows how a folder
is read, changes nothing, and is the place to start on data you have not
opened before.

## Images: series and beams

Autolamella writes far more than the Adaptive Polishing images, and every image
is indexed under one of four **series**:

| Series | What it holds |
|---|---|
| Adaptive Polishing | the `sem/` and `fib/` folders of a polishing run |
| reference | reference images taken before and after each milling task |
| alignment | the images of each beam alignment |
| other | everything no naming rule reaches: grid overviews, diagnostics |

Each image was taken with one of two beams: **SEM**, the electron beam, or
**FIB**, the ion beam. One acquisition can be imaged under several
**conditions**, each in a folder of its own.

The Curate and Label tabs show one series at a time. Labels are made for
Adaptive Polishing, reference and alignment images; *other* is shown but
never labelled.

## Labels and predictions

A **label** is an image the size of the one it describes, where each pixel's
value names a class: background, lamella, and so on. The classes and their
colours come from `segmentation_config_AM.yaml`, a project's own where it has
one and the packaged one otherwise.

Labels live beside their images, one file per image under the same name. A
folder of images has its labels in a folder named after it: `sem_labels/`
beside `sem/`. Images sitting loose in a lamella folder, as reference and
alignment images do, have theirs in a folder named for their series and beam:
`adaptive_sem_labels/`, `reference_fib_labels/`. *Other* has none.
**Predictions**, what a trained model makes of an image, live in
`sem_predictions/`.

An acquisition imaged under several conditions shares one label between
them, since they show one scene. A folder's name says which it shares:
everything before the first `_` names its **family**, and what follows names
the condition. `sem_fi4` beside `sem` shares `sem_labels/`; a separate series,
such as `sem-5kV` and `sem-5kV_fi4`, shares `sem-5kV_labels/` of its own. A
family's name starts with `sem` or `fib` and then a digit, `-`, `.` or `_`,
and may use anything but `_` after that. Folders named before families, such
as `SEM_8` and `SEM_8_fi_1`, are one family, `SEM`. A prediction becomes a label only when you convert it, and
that label is a first draft for a person to proofread.

**A label file existing does not mean anything was labelled.** The
segmentation software writes an empty label for every image it is given, so
am-seg looks inside each one. An empty label is never exported as a training
pair: it would teach a model that nothing is there.

## Label groups and their state

When several conditions of one acquisition share one label, they are one
**label group**, and every group is in exactly one of three **states**:

| State | Means |
|---|---|
| selected | a label is wanted, and not yet finished. Every image starts here. |
| deselected | no label is wanted here. |
| corrected | the label has been proofread. No automated operation may overwrite it. |

A state belongs to the group, never to one image in it, so changing the state
of one condition changes all of them. Deselecting a corrected group withdraws
the claim that it is corrected; the proofreading itself stays in the label.

States are kept in an `AM_state.yaml` in each stage folder, which lists the
groups that are not selected. A group it does not list is selected, so images
added later arrive selected. You can read the file, and edit it by hand.

## Safe by default

- **Nothing on the Curate tab writes until you ask.** Changing states,
  predicting and converting stay locked until *Enable editing* is switched on.
  On the Label tab, saving your proofreading asks first, and marking what you
  saved as corrected needs *Enable editing* too.
- **A batch is checked whole before any of it is written.** If one file in a
  hundred has changed since you looked, or may not be replaced, none of them is
  written and you are told which. A run that fails partway, on a full disk say,
  tells you what it had written.
- **You can see what will happen first.** Most commands that write take
  `--dry-run`; `merge`, `revert`, `savepoint` and `prune-archive` only report
  until given `--apply`.
- **A replaced label is kept.** A label an operation replaces is kept under
  `AM_archive/`, and `revert` puts it back. Only a model's untouched output is not kept, as the model can make it
  again.

## What am-seg writes

Everything am-seg knows about your data is in files beside it. There is no
database: what these files say is what am-seg knows, and you can read every
one of them yourself.

| File or folder | Where | Holds |
|---|---|---|
| `AM_metadata.yaml` | project folder | the project's description: sample, microscope, and so on |
| `AM_state.yaml` | stage folder | the state of every group that is not selected |
| `<family>_labels/` | beside the images | labels: from the segmentation software, converted from predictions, or saved after proofreading |
| `<family>_predictions/` | beside the images | predictions, with `am_seg_predictions.yaml` saying which model made them |
| `AM_history.jsonl` | project folder | who changed what, and when, from the first change on |
| `AM_archive/` | project folder | every label version an operation replaced |
| `AM_export.yaml` | an exported copy | where the copy came from, so `merge` can bring it home |

Outside the collection, `~/.am_segmentation_manager/` holds the window's log,
`gui.log`, and the name your changes are recorded under.
