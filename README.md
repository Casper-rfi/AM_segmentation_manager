am_segmentation_manager
=======================

Manage large collections of FIB/SEM images, segmentation labels, predictions
and metadata produced by Autolamella adaptive milling.

It answers the questions that get hard once a collection outgrows a spreadsheet:
which images are worth labelling, which labels have been proofread, what the
labels actually contain, and which subset should go into a training run or a
public data deposit.

**Why it might be useful to you**

* **It reads the data you already have.** Real collections accumulate layouts:
  the directory shape changed across Autolamella versions, and pre-Autolamella
  datasets have no project structure at all. All of them are indexed as they
  are — nothing on disk is reorganised to fit a newer convention.
* **The filesystem stays the source of truth.** There is no database and no
  hidden state. Workflow state *is* the folder a file sits in, so the same
  answer comes back whether you ask this package, `ls`, or the segmentation
  software.
* **It is careful with proofread work.** Manual annotation is never deleted or
  overwritten by an automated operation. Anything that moves files validates
  the whole batch first, shows you what it would do, and takes `--dry-run`.

Supported layouts
-----------------

A **project** is any directory holding `experiment.yaml` (Autolamella's) or
`AM_metadata.yaml` (this package's). Layouts back to 2023 work, including ones
that collapse the hierarchy — the missing level is reported as synthetic rather
than invented:

    project/lamella/Milling/Polishing/adaptive_polish_<timestamp>/sem/   2026
    project/lamella/adaptive_polish_<timestamp>/sem/                     2025
    project/lamella/adaptive_polish/sem/                                 2024
    project/lamella/SEM_8/                     no stage directory
    project/SEM/                               no lamella or stage directory

Run `am-seg inspect-layout` on a folder to see how it is read and what does not
fit. It is read-only and the right place to start on unfamiliar data.
Real examples are in [`docs/reference_layouts.md`](docs/reference_layouts.md).

Installation
------------

With conda, from a source checkout:

    ./scripts/setup_dev.sh

This creates the environment from `environment.yml`, installs the package, and
checks the optional dependencies import. With pip instead:

    pip install -e '.[all,gui]'

**Two installation hazards, both of which break imports:**

* **Do not install a second Qt binding.** `qtpy` is only an adapter and needs
  PySide6 (or PyQt5) underneath, but loading two into one process crashes at
  startup. If the environment already has PyQt5 (via napari, say), install
  `qtpy` alone.
* **Do not mix conda and pip for the image stack.** conda's `opencv` and pip's
  `opencv-python` both provide `cv2`, and having both leaves an import that
  fails. In a conda environment, do not also run `pip install -e '.[all]'`.

The `[all]` and `[gui]` extras both include matplotlib, so charting works out of
the box. `[plots]` installs it alone, for a command-line-only environment that
wants `stats --plot` and nothing else. Without it everything still runs and the
charts decline to draw.

**Inference is a separate environment.** `am-seg predict` needs torch and
several gigabytes of dependencies, so it is not installed alongside the curation
environment:

    ./scripts/setup_ml.sh

This builds `am_segmentation_manager_ml` from `environment-ml.yml`, installs the
package with the `[ml]` and `[qt]` extras, and reports which devices torch can
see. It is a second environment rather than an addition to the first because
albumentations requires `opencv-python-headless`, which cannot share an
environment with conda's `opencv` — the two-cv2 hazard above, in a form you
cannot avoid by being careful.

**This environment does the whole workflow**, window included, so predicting
does not mean switching environments mid-task. The curation environment is the
lighter one for anyone who never predicts; it is not the more capable one. Note
that predicting is a command — `am-seg predict` — and not yet an action in the
window.

pip's default torch build is CUDA-enabled on Linux and Windows and CPU/MPS on
macOS. To choose another, set an index URL first:

    AM_SEG_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu ./scripts/setup_ml.sh

To check an environment at any time:

    python -c "from am_segmentation_manager.io.image import image_stack_problems; print(image_stack_problems() or 'image stack OK')"

Getting started
---------------

    am-seg inspect-layout <folder>    # read-only: how is this data read?
    am-seg gui <folder>               # the main interface

Installing provides the `am-seg` command; `python -m am_segmentation_manager.cli`
is equivalent and needs no installation.

**The window is the primary interface.** Four tabs in workflow order — *Add
data* to describe projects so they can be indexed and grouped, *Curate* to
review images against their labels, set their state, and predict them, *Label* to proofread one
lamella's selected images at full resolution, and *Export* to choose a subset and
write it out. Proofreading holds a lamella's edits until you save them, saving
offers to mark the reviewed set corrected, and the Label tab will ring the stray
spots too small to be real. Reviewing is safe by default: everything that writes
stays locked until *Enable editing* is ticked. Press **`F1`** for the keys and
mouse gestures live on the tab you are on.

The command line covers the same ground plus the whole-collection reports, and
is the place for anything scripted:

| To do this | Run |
| --- | --- |
| See how a folder is interpreted | `inspect-layout` |
| Make a pre-Autolamella dataset indexable | `init-project` |
| Read or edit project metadata | `show-metadata` / `edit-metadata` |
| Predict labels with a trained model | `predict` |
| Turn predictions into labels for proofreading | `convert-predictions` |
| Choose what gets proofread | `deselect` / `select` / `correct` |
| Fix images that disagree with their shared label | `repair-state` |
| Find stray spots and holes too small to be real | `find-specks` |
| Shrink label files on disk | `compress-labels` |
| Count images by one or two properties, and chart them | `stats` |
| Measure what the labels contain, in um^2 | `quantify-labels` |
| Export metadata and training lists | `export-*-csv` / `create-training-csv` |

Getting help
------------

* `am-seg --help` lists every command in workflow order; `am-seg <command>
  --help` explains one, including how to read its output.
* **`F1`** in the window lists the keys and gestures available on that tab.
  It stays open beside the window while you try them, and `F1` puts it away.
* [`docs/00_roadmap.md`](docs/00_roadmap.md) — what is built and what is next.
* [`docs/decisions.md`](docs/decisions.md) — why it works this way, and the
  environment traps worth knowing.

Contributing
------------

Run the tests from the project root:

    pytest -q

The core is four layers, and higher layers depend on lower ones, never the
reverse:

    layout.py      what things are called
    discovery.py   where they are
    scanner.py     what they mean (the model)
    operations/    what you can do to them

`layout.py` is the single source of truth for filesystem conventions — nothing
else should embed a directory name, so supporting a new layout is a change
there rather than one spread across every module that touches files. Qt appears
only under `gui/widgets/`; everything the window knows lives in plain Python
controllers that test without a display.

Read [`CLAUDE.md`](CLAUDE.md) for the working agreement and the package
philosophy, and [`docs/decisions.md`](docs/decisions.md) before changing
anything in `operations/`.

Maintainer and status
---------------------

Written and maintained by Casper Berger (casper.berger@rfi.ac.uk) at the
Rosalind Franklin Institute. Questions, bugs, and layouts that are not read
correctly are welcome by email — the project is not yet hosted, so there is no
issue tracker.

**Status: in development, and not yet licensed for redistribution.** The
package carries no `LICENSE`, which means default copyright — all rights
reserved — while the licensing question is open. It is intended to be open
source in future.
