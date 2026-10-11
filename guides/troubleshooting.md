# Troubleshooting

## First, check the environment

After installing, and whenever something will not start, run from a source
checkout:

```bash
uv run python scripts/check_environment.py
```

It reports whether each part of am-seg can run here: reading images, the window,
charts, how many worker processes a scan starts, predicting, the devices torch
can see and, on Windows, long paths:

```text
  image stack   OK (numpy, tifffile, cv2)
  Qt binding    OK
  Qt window     OK
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

- **No Qt at all.** Install the window's extra: `uv sync --extra gui` in a
  checkout, or `[gui]` in the `uv tool install` command.
- **A Qt binding that will not load.** Two bindings in one environment, PySide6
  and PyQt5 say, crash as soon as the second loads. am-seg is developed and
  tested against PySide6; give it an environment of its own rather than sharing
  one with another Qt application.
- **On Linux, Qt imports but cannot open a window.** Qt says
  `xcb-cursor0 or libxcb-cursor0 is needed`, and `check_environment.py` reports
  `Qt window CANNOT OPEN`. A desktop install of Ubuntu usually has this system
  library; a server or a minimal install often does not. An administrator adds
  it with `sudo apt install libxcb-cursor0`. Without one, download and unpack
  it yourself, and put it beside Qt's own libraries. For an install made with
  `uv tool install`:

  ```bash
  mkdir -p ~/xcb-cursor
  (cd ~/xcb-cursor && apt download libxcb-cursor0)
  dpkg -x ~/xcb-cursor/libxcb-cursor0_*.deb ~/xcb-cursor
  cp ~/xcb-cursor/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0* ~/.local/share/uv/tools/adaptive-milling-seg/lib/python3.*/site-packages/PySide6/Qt/lib/
  ```

  `uv tool dir` says where the tools are if that folder is not there. For a
  source install, run the last line from the checkout with
  `.venv/lib/python3.*/site-packages/PySide6/Qt/lib/` as the destination.

  Upgrading the tool or rebuilding `.venv` loses the copy, so make it again
  afterwards; `~/xcb-cursor` keeps the file to copy. Started from a
  launcher, a window that cannot open shows nothing at all; `gui.log` says why.
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

Press **Inspect layout** on the Add data tab, or run `am-seg inspect-layout
<folder>`. It changes nothing. For each project it gives the lamellae, stages,
images and labels found, and how many images fall in each series; then what
does not fit, from errors down to notes:

```text
AutoLamella-2024-07-05-12-53  [experiment.yaml, AM_metadata.yaml]
  18 lamella(e), 18 stage(s), 1012 image(s), 241 label(s)
  series: 544 Adaptive Polishing, 342 reference, 126 alignment
  metadata: 4 field(s) still marked TODO

Findings
  warn  images-without-labels: AutoLamella-2024-07-05-12-53/07-brief-oryx
         16 of 31 images have no label
```

It also names every image it filed under *other*, and why, and every image
sitting in the folder but inside no project. A dataset that is not a project
yet is added with *Add projects*; see the
[user guide](user_guide.md#add-projects-that-are-not-from-autolamella). Where
the conditions of one family differ in size, see
[Background](background.md#families-of-imaging-conditions).

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
  it. On Windows and Linux the `ml` extra takes torch built for CUDA 12.6, which
  runs on any NVIDIA driver from version 525. An install with `train` and
  neither `ml` nor `ml-cuda13` takes another torch, on Windows one for the CPU
  alone: install again with `ml` among the extras.
- **The card is newer than that build.** A 50-series, for one, has no kernels in
  the CUDA 12.6 build however new its driver is, and predicting on `auto` falls
  back to the CPU rather than failing halfway. Install the CUDA 13 build
  instead, which needs a driver from version 580. From a checkout:

  ```bash
  uv sync --extra gui --extra ml-cuda13
  ```

  It replaces `ml` rather than joining it; `uv sync` refuses the two together.
  After `uv tool install`, run the install command again with `ml-cuda13` in
  place of `ml`, and `--upgrade`.

Either way, a GPU named as the device is refused with the reason, and so is
training on `auto`: on a CPU it takes days. Name `cpu` as the device to train
there regardless.

On macOS, Apple silicon predicts on its own GPU (`mps`). An Intel Mac runs the
window and the command line but cannot predict: torch no longer publishes
builds for it.

## Training

### *The training list holds images of 2 proportions*

A run trains on batches, and a batch is one array: every image in it must be
the same shape once it has been scaled. AM-model-training scales the longer
side to 1536 pixels and keeps the proportions, so a square image among 3:2
ones cannot share a batch with them, and am-seg refuses the run before it
starts, naming the fewer. On Export, set **Breakdown** to *image size
(pixels)* to count them, add a filter on it to leave them out, and export
again; the validation list of a set validated on corrected labels stays as it
was while those images are not corrected. A training batch of 1 also runs,
slowly, and is not comparable with a run trained in larger batches.

### On Windows: *expected scalar type Long but found Int*

A run fails at its first step with this error under AM-model-training 0.1.0,
which reads labels as a number type that is 64-bit on Linux and macOS and
32-bit on Windows. Until a release of it corrects that, change the one line in
the installed copy, here for an install made with `uv tool install`, from Git
Bash:

```bash
sed -i 's/dtype=np\.long,/dtype=np.int64,/' ~/AppData/Roaming/uv/tools/adaptive-milling-seg/Lib/site-packages/am_model_training/augmentations.py
```

In a checkout the file is under `.venv/Lib/site-packages/`. Upgrading or
rebuilding the environment puts the line back, so make the change again
afterwards.

### **Split by** is greyed out

Point at it: it says why. A run trained before am-seg kept what each
validation image scored has nothing to split until **Score own list** has
scored it, which needs its checkpoints still on disk. With *every set* chosen
in **Validated on** there is no one list to split; choose *each run's own
list* or one validation. A validation of an export that wrote no images file
has nothing recorded to split by.

### `am_training.lock` is still there after a run has ended

That is meant. A run, or a validation, is running exactly while a process
holds a lock on that file, which the system gives back however the process
ends; the file being there says nothing. It can be deleted once nothing is
running, and is made again when needed.

## The launcher

- **It stopped starting after the environment moved.** A launcher starts the
  Python it was made with. Run `am-seg create-shortcut --replace` from the
  environment you use now, adding `--viewer` for the viewer's, or *Add to...* in
  the window's File menu. On Linux an entry whose Python has gone disappears from
  the menu.
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
