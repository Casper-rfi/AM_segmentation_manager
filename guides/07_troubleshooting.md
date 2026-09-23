# Troubleshooting

## First, check the environment

After installing, and whenever something will not start, run:

```bash
uv run python scripts/check_environment.py
```

It reports whether each part of am-seg can run here: reading images, the window,
charts, how many worker processes a scan starts, predicting, the devices torch
can see and, on Windows, long paths:

```text
  image stack   OK (numpy, tifffile, cv2)
  Qt binding    OK
  charts        OK (matplotlib)
  workers       6 of 8 CPU(s) (AM_SEG_WORKERS changes it)
  inference     OK (torch 2.14.0)
  devices       mps, cpu
```

A part you did not install is reported and is not a failure; a part installed
and not working is, and names what to do about it.

## The window

### It will not start, or images will not open

The message says which part is missing:

- **No Qt at all.** Install the window's extra: `uv sync --extra gui`.
- **A Qt binding that will not load.** Two bindings in one environment, PySide6
  and PyQt5 say, crash as soon as the second loads. am-seg is developed and
  tested against PySide6; give it an environment of its own rather than sharing
  one with another Qt application.
- **Images will not open.** A second copy of OpenCV breaks the first: am-seg
  installs `opencv-python-headless`, and `opencv-python`, or conda's `opencv`,
  beside it makes `cv2` fail to import. In a conda environment, install only
  Python with conda and everything else with pip.

### It says *Something went wrong*, or disappears

The window writes everything that goes wrong to
`~/.am_segmentation_manager/gui.log`, including a crash that ends the process
without a word. It is the first place to look, and the file to send with a
report. It is moved to `gui.log.1` once it passes 5 MB.

To keep one session's log apart, name a file of your own. The option goes
before the command:

```bash
am-seg --log-file ~/am-seg.log gui /path/to/collection
```

On macOS, the system writes its own report of a crash to
`~/Library/Logs/DiagnosticReports/`, naming the library that failed. If a crash
points at Python's own memory allocator rather than at am-seg, run it again
with `PYTHONDEVMODE=1` set: the debugging allocator stops where memory is
actually damaged, rather than where it is next touched.

### A button stays greyed out

- On **Curate**, everything that writes waits for *Enable editing*, and a
  series nobody labels stays locked whatever the switch says.
- On **Add data**, the fields, *Add projects* and *Maintenance* wait for that
  tab's own *Enable editing*.
- *Predict labels* needs the `ml` extra; its tooltip says so when it is missing.

### *Somebody else changed this collection*

am-seg checks every file it is about to write against what it read. If
another window, a script, or a colleague changed one in between, nothing is
written, and the window reads the folder again. Make the change again on what
is now there.

## The data

### A folder is not read as expected

Run **Inspect layout** on the Add data tab, or `am-seg inspect-layout <folder>`.
It says how every project was read, which images went under *other* and why,
and which images sit in the folder but inside no project. A dataset that is not
a project yet is named with *Add projects*; see [Add data](02_add_data.md).

### A scan is slow, or takes over a shared machine

A scan reads image headers in worker processes: every CPU am-seg may use but
two, and no more than eight. Name another number for a session:

```bash
AM_SEG_WORKERS=4 am-seg gui
```

In PowerShell, run `$env:AM_SEG_WORKERS=4` first. A job scheduler's or
`taskset`'s limit is honoured; a container's CPU quota is not visible, so set
the number there. Predicting keeps torch to the same number of threads.

### On Windows: a batch refused for a path too long

Windows reaches a path of 260 characters or more only with its long-path
setting on, and the label versions kept under `AM_archive/` have long paths.
Without the setting, *Inspect layout* warns, and saving, converting and
exporting refuse the whole batch rather than failing halfway. Turn it on as
the [README's Installation section](../README.md#installation) describes.

## Predicting

### Which GPU is used

The *Device* box on Curate, and `am-seg predict --device`, name GPUs as
`nvidia-smi` numbers them, with each one's model and free memory, read afresh
whenever the box opens. `auto` takes the GPU with the most memory free, and
warns when even that has under 4 GB. A run sees only the GPU it was given, and
takes no memory on the others.

### It runs on the CPU although there is a GPU

The window, the log and `check_environment.py` say why:

- **The torch installed cannot use NVIDIA GPUs**, or the driver is too old for
  it. Windows and Linux get torch built for CUDA 12.6, which runs on any NVIDIA
  driver from version 525.
- **The card is newer than that build.** A 50-series, for one, has no kernels in
  the CUDA 12.6 build however new its driver is, and a run falls back to the CPU
  rather than failing halfway. Install the CUDA 13 build instead, which needs a
  driver from version 580:

  ```bash
  uv sync --extra gui --extra ml-cuda13
  ```

  It replaces `ml` rather than joining it; `uv sync` refuses the two together.

On macOS, Apple silicon predicts on its own GPU (`mps`). An Intel Mac runs the
window and the command line but cannot predict: torch no longer publishes
builds for it.

## The launcher

- **It stopped starting after the environment moved.** A launcher starts the
  Python it was made with. Run `am-seg create-shortcut --replace` from the
  environment you use now, or *Add to...* in the window's File menu. On Linux
  an entry whose Python has gone disappears from the menu.
- **On the GNOME Desktop** a launcher must be allowed once: right-click it and
  choose *Allow Launching*.
- **On a managed Windows machine** the shortcut may be refused, PowerShell being
  locked down. `create-shortcut` says so and changes nothing.

## Reporting a problem

Say what you did and what happened, and send:

- the version: `am-seg --version`;
- the operating system;
- `~/.am_segmentation_manager/gui.log`, if it was the window.

The [README](../README.md#maintainer-and-status) says whom to send it to.
