AM segmentation manager
=======================

Curate, proofread and export the segmentation labels of FIB/SEM images from
Autolamella's adaptive milling.

![The Curate tab: a lamella's SEM image beside its label](https://raw.githubusercontent.com/Casper-rfi/AM_segmentation_manager/main/guides/images/curate.png)

A collection of Autolamella data soon holds thousands of images, their labels,
the predictions of several models, and metadata spread over image headers and
project files. AM segmentation manager answers the questions that get hard once
it outgrows a spreadsheet: which images are worth labelling, which labels have
been proofread, what the labels contain, and which subset should go into a
training run or a public data deposit.

The work runs in five steps, a tab of the window for each, and the model
trained in the last makes the next round of labels quicker to prepare:

![Add data, Curate, Label, Export and Train in a row, and a better model from Train back to Curate](https://raw.githubusercontent.com/Casper-rfi/AM_segmentation_manager/main/guides/images/workflow.svg)

What it does
------------

* **Reads the data you already have.** Every Autolamella layout back to 2023,
  and datasets from before Autolamella, are read as they are: nothing on disk
  is moved to fit a newer convention.
* **Curates, a lamella at a time.** Look through each lamella's frames beside
  their labels, on the grid it was milled from, and mark each label group
  selected, deselected or corrected.
* **Predicts and proofreads.** Run a trained model, turn its predictions into
  labels, and correct them at full resolution with a brush, a fill and a
  finder for specks too small to be real.
* **Exports what you choose.** Filter and count a subset, measure what its
  labels hold, and write training lists, metadata for a deposit, or a copy of
  the files that merges back later.
* **Trains and compares models.** Train a model on an exported training set
  with AM-model-training, follow it as it trains, and compare runs by class,
  microscope or lamella type.
* **Keeps the files as the truth.** There is no database. Everything am-seg
  knows is in files beside your data, which you can read yourself.
* **Is careful with proofread work.** No automated operation overwrites a
  corrected label, every batch is checked whole before any of it is written,
  and every change is recorded, with the label it replaced kept.

Installation
------------

With [uv](https://docs.astral.sh/uv/), which installs Python itself. As an
application, in an environment of its own and with `am-seg` on the PATH:

    uv tool install "adaptive-milling-seg[gui,ml] @ git+https://github.com/Casper-rfi/AM_segmentation_manager"

**On Windows and Linux, the extra names the torch build.** `ml` takes torch
built for CUDA 12.6, which runs on any NVIDIA driver from version 525. For a
card that build has no kernels for, such as a 50-series, write `ml-cuda13` in
its place, which needs a driver from 580:

    uv tool install "adaptive-milling-seg[gui,ml-cuda13] @ git+https://github.com/Casper-rfi/AM_segmentation_manager"

macOS takes the same torch from either. To train models as well, add `train`
beside one of the two: `[gui,ml,train]`. Without `ml` the window installs
several gigabytes smaller, and has no torch. If the shell cannot find `am-seg`
afterwards, run `uv tool update-shell` once.

**To upgrade, run the same command again with `--upgrade`**, naming every
extra you want: the ones in the brackets replace the ones installed.

An install like this takes the newest release of every dependency. For the
versions the tests were run against, or to work on am-seg, use a source
checkout instead:

    uv sync --extra gui               # the window, the command line and charts
    uv sync --extra gui --extra ml    # ... and predicting

This builds `.venv` inside the checkout from `uv.lock`, so every machine gets
the versions the tests were run against. Run anything in it with `uv run`:

    uv run am-seg gui

**`uv sync` installs exactly the extras it is given** and removes the rest, so
name every extra you want each time. The extras combine freely, `ml` and
`ml-cuda13` apart, and `train` goes beside one of those two:

| Extra | Adds |
|---|---|
| `gui` | the window, the image stack and charts |
| `ml` | predicting: torch and the model libraries, several gigabytes |
| `ml-cuda13` | predicting, with torch for CUDA 13, for a card the default build has no kernels for, such as a 50-series |
| `train` | training a model with `am-seg train`: AM-model-training and the libraries it trains with, about another gigabyte; it takes the torch `ml` or `ml-cuda13` chose, and supports Python up to 3.13 |
| `all` | the image stack and charts, for the command line without the window |
| `plots` | charts alone, for the command line's plots |

Windows and Linux get torch built for CUDA 12.6, which runs on any NVIDIA
driver from 525; Apple silicon predicts on its own GPU. To see what this
environment can do, with `uv run` in front in a checkout:

    am-seg check-environment

With pip instead, into an environment of its own:

    pip install -e '.[gui,ml]'

pip ignores `uv.lock` and takes whichever versions are newest.

**On Windows, keep collections in a short folder**, such as `D:\data\`. Windows
cannot reach a path of 260 characters or more by default, and the label versions
AM-seg keeps under `AM_archive/` have long paths, so a collection under
`C:\Users\<you>\` or in OneDrive can pass the limit. Where a path is too long,
saving, converting and exporting refuse the whole batch rather than failing
partway; [Troubleshooting](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/troubleshooting.md#on-windows-a-batch-refused-for-a-path-too-long)
says what to do.

**On Linux, the window needs the system library `libxcb-cursor0`.** Ubuntu's
desktop install has it; a server or a minimal install may not, and adding it
takes an administrator: `sudo apt install libxcb-cursor0`.
`am-seg check-environment` says whether the window can open, and
[Troubleshooting](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/troubleshooting.md) has a way round it without one.

If something will not install or start, see
[Troubleshooting](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/troubleshooting.md).

Uninstalling
------------

Three steps remove everything am-seg put on this computer outside your data:

1. Take the launchers away, from the install they were made with (from a
   checkout, put `uv run` in front):

       am-seg create-shortcut --remove
       am-seg create-shortcut --viewer --remove

   On Windows, a taskbar pin is Windows' own copy: unpin it by hand.
2. Remove the install: `uv tool uninstall adaptive-milling-seg`; or delete the
   checkout, which holds the environment in `.venv`; or, after a pip install,
   `pip uninstall adaptive-milling-seg`.
3. Delete `~/.am_segmentation_manager` (`%USERPROFILE%\.am_segmentation_manager`
   on Windows): the window's log, and the name your changes are recorded under.
   Every install on this computer shares it, so keep it while another remains.

uv keeps the packages it downloaded, torch among them, for any project to reuse;
`uv cache clean` frees that space. Your collections keep the files am-seg wrote
beside the data, [listed in Background](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/background.md#what-am-seg-writes):
they are part of the data, not of the install.

Getting started
---------------

    am-seg create-shortcut            # once: start the window like any other application
    am-seg gui                        # or start it from the command line
    am-seg create-shortcut --viewer   # the viewer's own launcher, beside the window's
    am-seg gui --viewer               # only look: nothing in it changes your data

From a checkout, put `uv run` in front of each. `create-shortcut` adds the
window to the Applications folder on macOS, the applications menu on Linux, and
the Start menu and Desktop on Windows; the window's File menu offers the same,
for the window or the viewer.

Then follow [the user guide](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/user_guide.md), which goes through the
window a step at a time. In the window, hovering over a control says what it
does, and **F1** lists the keys of the tab you are on.

Documentation
-------------

| Document | For |
|---|---|
| [User guide](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/user_guide.md) | getting started: the window, a step at a time |
| [Background](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/background.md) | how am-seg reads your data, what it keeps safe, and its history |
| [The command line](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/command_line.md) | every command, for scripting and for reports on a whole collection |
| [Troubleshooting](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/troubleshooting.md) | when something will not install, start or read as expected |
| [`CHANGELOG.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/CHANGELOG.md) | what each release changed |

Models and data
---------------

The models Adaptive Milling was trained with, and the data they were trained
on, are published on Zenodo:
[doi:10.5281/zenodo.21804785](https://doi.org/10.5281/zenodo.21804785). To try
am-seg before pointing it at your own data, download one of its projects and
scan the folder it unpacks to. The models were trained
with [AM-model-training](https://github.com/rosalindfranklininstitute/AM-model-training), which `am-seg train` runs on the training CSVs am-seg writes.

Contributing
------------

[`CONTRIBUTING.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/CONTRIBUTING.md) says how the package is laid out, how to
run the tests, and where the reasons behind it are written down.

Maintainer and status
---------------------

Written and maintained by Casper Berger (casper.berger@rfi.ac.uk) at the
Rosalind Franklin Institute. Questions, bugs, and layouts that are not read
correctly are welcome as
[GitHub issues](https://github.com/Casper-rfi/AM_segmentation_manager/issues),
or by email. Report a security problem privately instead, as
[`SECURITY.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/SECURITY.md) says.

**Status: in development.** Licensed under the
[Apache License 2.0](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/LICENSE). Code derived from other projects is listed, with
its licence, in [`THIRD_PARTY_NOTICES.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/THIRD_PARTY_NOTICES.md).
