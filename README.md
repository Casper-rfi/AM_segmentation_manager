AM segmentation manager
=======================

AM-seg is software to view and curate large collections of [fibsemOS](https://www.fibsemos.org) Autolamella projects and to create labels to train segmentation models for [Adaptive Milling](https://github.com/rosalindfranklininstitute/adaptive_milling).

![The Curate tab: a lamella's SEM image beside its label](https://raw.githubusercontent.com/Casper-rfi/AM_segmentation_manager/main/guides/images/curate.png)

AM-seg workflow
---------------

AM-seg supports a full labelling and model training workflow:

* **Read FIB/SEM datasets** - View all images in Autolamella (2023 and newer) and manually collected datasets. Add and edit project metadata.
* **Curate** images used for model training and **predict** labels using machine-learning segmentation models.
* **Proofread labels** using the label editor, with a built-in labelling guide.
* **Export** file lists for model training, data subsets and quantify your image collection & labels. Merge exported subset back in to combine changes.
* **Train** new models or fine-tune existing models, track training progress and compare models between each other and on data subsets.
* **Version control** for labels, changes are tracked, create savepoints and reload them. AM-seg is careful not to overwrite proofread labels.

This supports a training feedback loop for improving segmentation model performance:

![Add data, Curate, Label, Export and Train in a row, and a better model from Train back to Curate](https://raw.githubusercontent.com/Casper-rfi/AM_segmentation_manager/main/guides/images/workflow.svg)

Installation
------------

We recommend using [uv](https://docs.astral.sh/uv/) to install AM-seg. To install uv, run:

Linux and macOS:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Windows (PowerShell):

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

* *Close and reopen the terminal after installing uv.*

AM-seg has modular dependencies, so pick an install command below depending on how you plan to use it.

**To view and curate data, edit metadata and proofread labels:**

    uv tool install --python 3.13 "adaptive-milling-seg[gui] @ git+https://github.com/Casper-rfi/AM_segmentation_manager"


**To also predict labels with a pre-trained model:**

    uv tool install --python 3.13 "adaptive-milling-seg[gui,ml] @ git+https://github.com/Casper-rfi/AM_segmentation_manager"

* *For Windows and Linux machines with a GPU from a very recent series (e.g. NVIDIA 50-series), use `ml-cuda13` instead of `ml`. This will install with the correct version of Torch to support GPUs that require driver version 580 or newer. Use the `nvidia-smi` command to check your GPU model and drivers before installing.*
* *Label prediction can be comfortably run with a CPU*

**The full workflow including model training:**

    uv tool install --python 3.13 "adaptive-milling-seg[gui,ml,train] @ git+https://github.com/Casper-rfi/AM_segmentation_manager"

* *Model training realistically requires an NVIDIA GPU*

After installing:

* If `am-seg` is not found, run `uv tool update-shell` and then open a new terminal.
* Check that the environment is set up correctly with the command:

      am-seg check-environment

**To update or add dependencies, run the same command again with `--upgrade`** with all the extras, including the ones already installed. Example: `[gui,ml,train]`.

**For developing AM-seg install from a source checkout:**

    git clone https://github.com/Casper-rfi/AM_segmentation_manager
    cd AM_segmentation_manager
    uv sync --extra gui --extra ml

This builds `.venv` inside the checkout from `uv.lock`. Run anything from this directory
with `uv run` in front, such as `uv run am-seg gui`. **`uv sync` installs
exactly the extras it is given** and removes the rest, so name every extra you
want each time.

If something will not install or start, see
[Troubleshooting](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/troubleshooting.md).

Getting started
---------------

The command line is needed to start it for the first time:

* Start the software with the command:

      am-seg gui

* Add AM-seg as a runnable application and create a shortcut in the software:
  * in the top left menu: File --> Add to Start menu 
    * *(MacOS: Add to Applications, Linux: Add to applications menu)*
  * Click add. Use the toggle to decide to also create a desktop shortcut
  * AM-seg is now findable and runnable like any other program on your machine
* alternatively, use a command to do this directly:

      am-seg create-shortcut

AM-seg also has a viewer-only mode, which can be added in the software by selecting The viewer instead of The window, or through the command line:

    am-seg create-shortcut --viewer

Then follow [the user guide](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/user_guide.md), which covers the full workflow. In the window, hovering provides tooltips and **F1** lists the keys of the tab you are on.

Documentation
-------------

| Document | For |
|---|---|
| [User guide](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/user_guide.md) | getting started: the window, a step at a time |
| [Background](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/background.md) | how AM-seg reads your data, what it keeps safe, and its history |
| [The command line](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/command_line.md) | every command, for scripting and for reports on a whole collection |
| [Troubleshooting](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/troubleshooting.md) | when something will not install, start or read as expected |
| [`CHANGELOG.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/CHANGELOG.md) | what each release changed |
| [`CONTRIBUTING.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/CONTRIBUTING.md) | Describes package layout and how to run the tests |

Models and data
---------------

Adaptive Milling segmentation models and training data are available on Zenodo:
[doi:10.5281/zenodo.21804785](https://doi.org/10.5281/zenodo.21804785). For prediction on SEM images, we recommend to use the model trained on all data ([download link](https://zenodo.org/records/21804785/files/AM_SEM_All_V01.pth?download=1)).

Uninstalling
------------

Three steps remove everything AM-seg put on this computer outside your data:

1. Take the launchers away, from the install they were made with (from a
   checkout, put `uv run` in front):

       am-seg create-shortcut --remove
       am-seg create-shortcut --viewer --remove

   On Windows, a taskbar pin is Windows' own copy: unpin it by hand.
2. Remove the install: `uv tool uninstall adaptive-milling-seg`; or delete the
   checkout, which holds the environment in `.venv`.
3. Delete `~/.am_segmentation_manager` (`%USERPROFILE%\.am_segmentation_manager`
   on Windows): the window's log, and the name your changes are recorded under.
   Every install on this computer shares it, so keep it while another remains.

uv keeps the packages it downloaded, torch among them, for any project to reuse;
`uv cache clean` frees that space. Your collections keep the files AM-seg wrote
beside the data, [listed in Background](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/guides/background.md#what-am-seg-writes):
they are part of the data, not of the install.

Maintainer and status
---------------------

Written and maintained by Casper Berger at the Rosalind Franklin Institute. All code was written by LLMs. For questions, bugs or suggestions please contact me via [GitHub issues](https://github.com/Casper-rfi/AM_segmentation_manager/issues), or by email (casper.berger@rfi.ac.uk).

**Status: in development.** Licensed under the
[Apache License 2.0](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/LICENSE). Code derived from other projects is listed, with
its licence, in [`THIRD_PARTY_NOTICES.md`](https://github.com/Casper-rfi/AM_segmentation_manager/blob/main/THIRD_PARTY_NOTICES.md).
