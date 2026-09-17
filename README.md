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
  hidden state. Workflow state is declared in an `AM_state.yaml` beside the
  image directories, so the same answer comes back whether you ask this
  package or `cat` the file.
* **It is careful with proofread work.** Manual annotation is never deleted or
  overwritten by an automated operation. Anything that changes a batch
  validates all of it first, shows you what it would do, and takes `--dry-run`.

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

**Every image is indexed, under one of four series**, because Autolamella
writes far more than the Adaptive Polishing images. Reference and alignment
images sit *loose* in a directory rather than filling one, so a lamella
directory holds several at once:

    Adaptive Polishing   the `sem/` and `fib/` directories above
    reference            ref_<task>_*_eb.tif in a lamella directory, and
                         <task>_pre-task_* / <task>-<stage>_finished_* in a
                         milling task directory
    alignment            beam_shift_alignment_*, ref_*_initial_alignment_*,
                         ref_alignment_ib.tif, and the reference_image /
                         new_image_<n> / final_image trio in an Alignment run
    other                every remaining .tif — grid overviews, diagnostics,
                         and anything a naming rule does not reach

A loose image's beam comes from its `_eb`/`_ib` suffix, and from the image
header for the few names that carry neither. Labels for one go in
`<series>_<modality>_labels/` beside the images; `other` has none, and no
command that writes will act on it.

Run `am-seg inspect-layout` on a folder to see how it is read and what does not
fit. It is read-only and the right place to start on unfamiliar data, and it
reports every image that landed in `other` and why — which is how a naming this
package has not seen makes itself visible — and every image file sitting under
the folder but inside no project, which nothing indexes at all.

**`--series` chooses which of them a command covers.** `inspect-layout` and
`stats` report on all of them; every other command acts on Adaptive Polishing
unless told otherwise, so indexing the rest changed nothing that was already
being written. A command that writes, measures or exports label data does not
offer `other` at all, and `--series all` means every series *that command*
covers:

    am-seg stats ... --by series                 # how many of each
    am-seg create-training-csv ... --series reference,adaptive

**`--task` narrows to one milling task**, rolling that task's milling stages
into it, which `--stage` and `--by stage-name` cannot: one task spans several
stage directories (`Rough 1`, `Rough 2`) and a lamella directory holds every
task's reference images at once. Task names come from the project's own
`experiment.yaml`, so they are whatever the protocol's author typed;
separators and case are ignored when matching, and `--task none` selects the
images naming no task at all:

    am-seg stats ... --by task                   # which tasks are here
    am-seg create-training-csv ... --series reference --task polishing

Installation
------------

From a source checkout, with [uv](https://docs.astral.sh/uv/) — one command on
Windows, macOS and Linux, and it installs Python itself:

    uv sync --extra gui               # the window, the command line and charts
    uv sync --extra gui --extra ml    # ... and predicting

This builds `.venv` inside the checkout from `uv.lock`, so every machine gets
the versions the suite was run against, on the Python `.python-version` names.
Run anything in it through `uv run`, without activating it:

    uv run am-seg gui
    uv run pytest

**`uv sync` installs exactly the extras it is given** and removes the rest, so
name every extra you want each time. `uv run` leaves the environment as it is.

The extras combine freely:

| Extra | Adds |
|---|---|
| `gui` | the window, the image stack and charts |
| `ml` | predicting: torch and the model libraries, several gigabytes |
| `all` | the image stack and charts, for the command line without the window |
| `plots` | charts alone, for the command line's plots |

Without an extra everything that needs it declines rather than failing. To see
what this environment can do:

    uv run python scripts/check_environment.py

**On a shared machine.** A scan reads headers in worker processes: every CPU
this process may use but two, and never more than eight, past which reading gets
no faster. Name another number for a session with `AM_SEG_WORKERS`, e.g.
`AM_SEG_WORKERS=4 uv run am-seg gui` (in PowerShell, set `$env:AM_SEG_WORKERS=4`
first). A job scheduler's or `taskset`'s limit is honoured; a container's CPU
quota is not visible, so set the number there. Predicting holds torch to the same
number of CPU threads.

**Choosing a GPU.** The device box, and `am-seg predict --device`, name GPUs as
`nvidia-smi` numbers them, and the box shows each one's model and free memory,
read afresh whenever it opens. `auto` takes the GPU with the most memory free, and
warns when even that has under 4 GB. A run sees only the GPU it was given, so it
takes no memory on the others. Where the machine has NVIDIA GPUs that torch cannot
use -- a CPU-only torch, or a driver too old for its CUDA -- the window, the log
and `scripts/check_environment.py` say so rather than predicting slowly on the CPU.

**On Windows, turn on long paths.** Windows reaches a path of 260 characters or
more only with its long-path setting on, and the versions this package archives
under `AM_archive/` run long -- a collection under `C:\Users\<you>\` or
OneDrive can pass the limit. Without it, the Add data report warns, and saving,
converting and exporting refuse the whole batch rather than failing partway.
Turn it on once, in PowerShell run as administrator, then restart am-seg (and
Windows, if am-seg still cannot reach them):

    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

On a managed machine, ask IT for the Group Policy *Enable Win32 long paths*.
`scripts/check_environment.py` says whether long paths work.

**Which torch build you get.** Windows and Linux receive torch built for CUDA
12.6, which runs on any NVIDIA driver from version 525; macOS on Apple silicon
receives the build that uses its GPU (`mps`). Intel Macs get the window and the
command line but cannot predict: torch no longer publishes builds for them.

With pip instead, into an environment of its own:

    pip install -e '.[gui,ml]'

pip ignores `uv.lock` and takes whichever versions are newest, and on Linux its
torch needs a driver recent enough for CUDA 13.

**Two installation hazards, both of which break imports:**

* **Do not install a second Qt binding.** `qtpy` is only an adapter and needs
  PySide6 underneath, but loading two bindings into one process crashes at
  startup. **PySide6 is the binding this package is developed and tested
  against.** `qtpy` will load PyQt5 if that is what an environment has, and the
  window does still work there, but nothing checks it and it has broken before
  without anyone noticing. Give `am-seg` an environment of its own rather than
  sharing one with a PyQt5 application.
* **Do not add a second OpenCV.** Every extra installs `opencv-python-headless`.
  `opencv-python`, or conda's `opencv`, beside it is a second provider of `cv2`,
  and the import fails. In a conda environment, install nothing but Python with
  conda and the rest with pip.

Getting started
---------------

    am-seg inspect-layout <folder>    # read-only: how is this data read?
    am-seg gui <folder>               # the main interface

Installing provides the `am-seg` command — `uv run am-seg` from a uv checkout;
`python -m am_segmentation_manager.cli` is equivalent and needs no installation.

**The window is the primary interface.** Four tabs in workflow order — *Add data* to
name which directories are projects, describe them so they can be indexed and grouped,
and inspect or recompress the whole folder; *Curate* to review images against their labels,
set their state and predict them; *Label* to proofread one lamella's selected images at full resolution; and *Export*
to choose a subset, write out its lists and metadata, or copy the images and labels
themselves into a folder of their own. Curate and Label each show one series at a time,
chosen beside the beam buttons, with each alignment run's summary figure on Curate's Plots
view and, where a project has one, its grid overview with every lamella marked on Curate's
Grid view, where a menu picks between several. The **Series shown** switches decide which series the lists and their chips
cover at all; *other* — grid overviews, their tiles, diagnostics — starts switched off,
and the lamella list says how many it is leaving out. Proofreading holds a lamella's edits until you save them, saving offers to mark the
reviewed set corrected, and the Label tab rings the spots likely to be too small to be real.
Autolamella projects are found by their `experiment.yaml`; datasets predating it, or with
theirs stripped out, are named on *Add data* through **Add projects**, which proposes the
directories holding images that nothing indexes and lets you confirm them, look inside one
that gathers several datasets, or name a folder the scan cannot recognise.
Reviewing is safe by default: everything that writes stays locked until *Enable editing*
is ticked, and stays locked anyway on a series whose images have no labels to change. Press **`F1`** for the keys and mouse gestures live on the tab you are on.

The command line covers the same ground plus the whole-collection reports, and
is the place for anything scripted:

| To do this | Run |
| --- | --- |
| See how a folder is interpreted | `inspect-layout` |
| Make a pre-Autolamella dataset indexable | `init-project` |
| Read or edit project metadata | `show-metadata` / `edit-metadata` |
| Scan a directory and list the projects found | `list-projects` |
| Predict labels with a trained model | `predict` |
| Turn predictions into labels for proofreading | `convert-predictions` |
| Choose what gets proofread | `deselect` / `select` / `correct` |
| Find stray spots and holes likely to be too small to be real | `find-specks` |
| Shrink label files on disk | `compress-labels` |
| Count images by one or two properties, and chart them | `stats` |
| Measure what the labels contain, in um^2 | `quantify-labels` |
| Export metadata and training lists | `export-*-csv` / `create-training-csv` |
| Export a selection of images and labels into a folder of its own, ready to merge back | `export-files` |
| Check an exported CSV still resolves | `check-training-csv` |
| Start recording a collection's history | `init-history` |
| Name the state of the data, so a round of work can be undone | `savepoint` |
| See the savepoints here, what each covers and what it still reaches | `list-savepoints` |
| Read what has been recorded about a project or a label | `history` |
| Merge a worked-on copy back into the collection | `merge` |
| Put labels back to what they held before a conversion or a merge | `revert` |
| See the label versions a project has kept | `archive` |
| Remove surplus kept versions | `prune-archive` |

Getting help
------------

* `am-seg --help` lists every command under its step of the workflow; `am-seg <command>
  --help` explains one, including how to read its output.
* **If something goes wrong in the window**, it says so in a dialog and writes
  the traceback to `~/.am_segmentation_manager/gui.log`. That file also receives
  a native stack trace if the process is killed by a signal, so it is the first
  place to look when the window disappears without an error. It appends, and is
  moved to `gui.log.1` once it passes 5 MB. To keep one session separate, name
  a file of your own: `am-seg --log-file ~/am-seg.log gui <folder>` — the option
  goes before the command, and a file you name is never rolled aside. On macOS
  the system writes its own report to `~/Library/Logs/DiagnosticReports/` at
  the same moment, which names the library that faulted. If the stack points
  at Python's own allocator rather than at any of this code, run it again with
  `PYTHONDEVMODE=1` set: that turns on the debugging allocator, which stops at
  the point memory is actually damaged instead of at the unrelated place it is
  next touched.
* **`F1`** in the window lists the keys and gestures available on that tab,
  under a heading per job — what changes the view, what chooses which images a
  change covers, what changes them. It stays open beside the window while you
  try them, and `F1` puts it away.
* [`docs/00_roadmap.md`](docs/00_roadmap.md) — what is built and what is next.
* [`docs/decisions.md`](docs/decisions.md) — why it works this way, and the
  environment traps worth knowing.
* [`docs/display.md`](docs/display.md) — the same for the window itself, and
  the Qt traps behind it.

Contributing
------------

Run the tests and check the type hints from the project root; both should
report nothing wrong:

    pytest -q
    mypy

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
philosophy, [`docs/decisions.md`](docs/decisions.md) before changing anything in
`operations/`, and [`docs/display.md`](docs/display.md) before changing anything
under `gui/`.

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
