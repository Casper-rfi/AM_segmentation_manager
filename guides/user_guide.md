# User guide

This guide gives a short overview of how to use AM-seg throughout the workflow.
Additional help is available in the window:

- **Hover over a control** to see what it does.
- **Press F1** to list the keys of the tab you are on, and what its coloured
  marks mean.
- **The scroll wheel never changes a setting.** Over a number, a dropdown or a
  slider it scrolls the page behind; click the control, then type or use the
  arrow keys.

Installing AM-seg is described in the [README](../README.md#installation).
[Background](background.md) explains how AM-seg reads your data and keeps its
history, and [the command line](command_line.md) and
[troubleshooting](troubleshooting.md) have guides of their own.

## 1. Organising the data

AM-seg works on a **collection**: one folder holding your Autolamella projects.
Place the projects in this folder directly, or inside a session folder:

```text
Autolamella data/
├── AutoLamella-2026-10-14-11-56/
├── 20240705_adaptive_milling/
│   ├── AutoLamella-2024-07-05-10-55/
│   └── AutoLamella-2024-07-05-12-53/
└── 20250414_adaptive_milling/
    └── AutoLamella-2025-04-14-20-54/
```

Leave each project as Autolamella wrote it. AM-seg reads every Autolamella
layout from 2023 onwards without changes. Datasets from before Autolamella can
go in the same folder and are added by hand, as described under *Add data*
below.

## 2. Scan the folder

When AM-seg starts, it asks for a folder:

![Image showing the start page, asking for a folder](images/start.png)

Choose your collection. AM-seg finds every project in it and lists them under
the folders they sit in. The first scan of a large collection may take a while,
as it reads the header of every image.

To choose another folder later, use **Browse...** (1). Press **Scan** (2) to
read the folder again after something outside AM-seg has changed it:

![Image showing the folder, Browse and Scan, at the top right of the window](images/scan-bar.png)

**Set your name** with the gear button at the top right (or press F2). AM-seg
records who made each change, and until you set a name it records none.

## 3. Add data

The **Add data** tab is where you add projects and project-level metadata.

![Image showing the Add data tab, describing a project](images/add-data.png)

### Add projects that are not from Autolamella

Projects with an Autolamella `experiment.yaml` appear on their own. A project that is not from Autolamella has to be added by
hand. The *Data* card (4) says how many folders hold images that no project
covers.

1. Switch on **Enable editing** (1).
2. Press **Add projects...** under *Data* (4). The dialog proposes the
   folders that could become projects.
3. Tick the ones you want and press **Write metadata files**. Where a folder
   holds several datasets, select it and press **Look inside** to choose the
   folders below it instead.

Each project you add gets an `AM_metadata.yaml` file, which is what makes it a
project. If a folder is not read the way you expect, see
[Troubleshooting](troubleshooting.md#a-folder-is-not-read-as-expected).

### Describe your projects

The metadata is stored with each project and extends the fields Autolamella
records. To correct it or enter more:

1. Click a project in the list. Click a folder to edit all the projects inside
   it: fields the projects disagree on show as *mixed*. Some fields, such as
   project(s) and user(s), hold several values.
2. Switch on **Enable editing** (1) and fill in the fields (2). Pick a value
   from the dropdown, or use autocomplete, to keep spelling consistent.
3. Press **Preview changes** to see what will be written, then **Save** (3).

The bar beside each project in the list shows how complete its description is.
Hover over a field for what to enter; the same guidance is below.

<!-- generated: fields -->

**General**

| Field | What to enter | Takes |
|---|---|---|
| Description | Goals and context of the experiment | text |
| Organisation(s) | Organisation that acquired the data | text, one or several |
| Project(s) | Name of the wider project this belongs to | text, one or several |
| User(s) | Who acquired the data | text, one or several |

**Sample preparation**

| Field | What to enter | Takes |
|---|---|---|
| Sample(s) | Cell type or organism | text, one or several |
| Sample state(s) | What was done to the sample, e.g. infected, differentiated | text, one or several |
| Sample media | What the sample was in, e.g. culture medium, cryoprotectant, fiducials | text, one or several |
| Grid type | Grid the sample was prepared on, e.g. Quantifoil R2/2 Cu 200 mesh | text |
| Grid ID | Identifier of the physical grid this was prepared on | text |
| Freezing method | How the sample was vitrified | text; usually Plunge freezing or High pressure freezing |

**Lamella preparation**

| Field | What to enter | Takes |
|---|---|---|
| GIS deposition time (seconds) | Seconds of GIS deposition, as a number | a number |
| Lamella type | One of: cells-on-grid, waffle, lift-out, other | one of: cells-on-grid, waffle, lift-out, other |

<!-- end generated -->

AM-seg writes metadata only to `AM_metadata.yaml`. Autolamella's
`experiment.yaml` is never changed, and an entry in `AM_metadata.yaml` takes
precedence over the same entry in it.

## 4. Curate

The **Curate** tab is used to view the data, decide which images are worth
labelling and to predict labels for them.

![Image showing the Curate tab, numbered](images/curate-map.png)

### Viewing your data

1. **Series shown** chooses which image series the lists cover:
   - Adaptive Polishing images
   - Reference images from Autolamella
   - Alignment images from Autolamella
   - All other images
2. **Projects**: click one to list its lamellae, or click a folder to list
   every lamella in it.
3. **Lamellae**: click one to show its images.
4. **SEM** and **FIB** switch between the two beams. **Plots** shows the plots
   saved by Adaptive Polishing and the overview figure Autolamella creates for
   alignment. **Grid** shows the grid overview with the approximate lamella
   positions:

   ![Image showing the Grid view, with every lamella ringed and the one in view named](images/curate-grid.png)

5. **Series** chooses which image series the timeline steps through. Where the
   same image was recorded under several conditions, **Variant** chooses which
   one to show.
6. **The timeline** holds the lamella's frames in order. Click or drag along it
   to move between them (or use the arrow keys).
7. **Enable editing** unlocks everything on this tab that changes a file.
8. **Operations** predict and convert labels, as described below.

### Use the viewer

To look through your data without any risk of changing it, such as on the
microscope's computer, start the viewer with `am-seg gui --viewer`. It is the
Curate tab with all editing removed. To add it to your applications, choose
*The viewer* under *Add to...* in the File menu.

### Decide what gets labelled

Every image has a state that says how it is used:

- **Selected** images are the default. They are predicted on to create labels,
  and can be used as uncorrected labels in training.
- **Corrected** images have been proofread by you, and are suitable for
  training and as validation data.
- **Deselected** images are not labelled or used for training.

The state of each frame is shown by its colour on the timeline:

![Image showing the timeline, with corrected frames in green, selected in blue and deselected in grey](images/timeline.png)

| State | Colour | Means |
|---|---|---|
| selected | blue | a label is wanted, and not finished yet. Every image starts here. |
| corrected | green | the label has been proofread. No automated operation will overwrite it. |
| deselected | grey | no label is wanted. |

To change a state, use the **State** tab on the right:

![Image showing the State tab, with editing switched on](images/curate-states.png)

1. Switch on **Enable editing**. No file changes can be made without this.
2. Choose the images to change. By default it is the frame on screen.
   **Ctrl-click** frames on the timeline to add more (⌘-click on a Mac), or
   **Shift-click** to select a range. To select whole projects or lamellae,
   tick them in the lists. A ticked project or lamella covers only the series
   chosen under **Series**, and a lamella holding none of it is left out.
3. Choose the beam: **SEM**, **FIB** or **Both**.
4. Choose the state. Larger changes ask for confirmation before they are made.

Imaging conditions that share a label change state together, as they are one
segmentation. [Background](background.md#families-of-imaging-conditions) explains
how AM-seg decides which conditions share a label, and how to correct it on
the **Families** tab.

You rarely need to mark images as corrected here: **Save and mark corrected**
on the Label tab does it as you save.

### Predict and convert labels

*Operations on this frame* (8) act on the same selection as the state buttons,
and also require *Enable editing*. Each action shows what it would do before
doing it.

![Image showing the Operations card](images/operations.png)

- **Predict labels...** runs a trained model over the chosen images. Choose the
  model's weights file with **Browse...**, and leave **Device** on *auto* to use
  the GPU with the most memory available, or else the CPU. Predictions are kept
  apart from labels, so they never replace one. It takes about 200 ms per image
  on a GPU and about 1 s on a CPU. Predicting needs the `ml` extra installed.
- **Convert predictions...** turns predictions into labels for you to proofread.
  **From** chooses between AM-seg's predictions and those Adaptive Polishing made
  while milling. A corrected label is never replaced. Replacing a label in any
  other state asks for confirmation first. Replaced labels are kept under
  `AM_archive/` and can be restored with **Revert...**; a label that only held
  an earlier conversion is not kept, as converting again makes it.
- **Revert...** puts labels back to what they held before the last conversion
  or merge, or at a savepoint.

The Adaptive Milling models and their training data are available on Zenodo
([doi:10.5281/zenodo.21804785](https://doi.org/10.5281/zenodo.21804785)). We
recommend `AM_SEM_All_V01.pth`, which was trained on all the data.

## 5. Proofread on the Label tab

The **Label** tab is for proofreading predicted labels, or creating new labels.
Images are shown at their full resolution, and deselected images are not shown.

![Image showing the Label tab, numbered](images/label.png)

### Label editing tools

1. **Choose a class** to paint with, or **Shift-click** the label where it
   shows the class you want to select it. The padlock beside a class locks it,
   so that it cannot be painted over.
2. **Brush** paints where you drag; **Fill** floods the area you click in.
   Scroll to zoom, and hold Space to pan. Resize the brush with the slider, or
   with **Shift + scroll wheel**.
3. **Find specks** places a circle around every small object, as these are
   likely to be incorrectly labelled.

   ![Image showing Find specks ringing three specks](images/specks.png)

4. **Compare versions** lists the earlier versions of this label and the
   predictions for this image. Hold **B** to flick between them, or restore one.
   With unsaved edits, **B** shows the label as saved, so you see what your
   edits changed. A prediction is shown as the label converting it would write.
5. **Save** writes your edits. Until then they are held in the window. With
   *Enable editing* on, on Curate, **Save and mark corrected** also marks the
   frames you reviewed as corrected.

### Follow the labelling guide

AM-seg has a labelling guide that takes you through a series of steps to
correct a predicted label. To enable it, switch on **Checklist**, or press
**C**.

- Press **C** to tick the step you are on and move to the next, and **Shift+C**
  to go back.
- Click **How to do this step...** for an explanation of what to do, with a
  before and after example image.
- Each label keeps its own progress while the window is open, so several
  images can be corrected in parallel.

![Image showing the proofreading checklist, with three steps ticked](images/checklist.png)

## 6. Export

The **Export** tab is used to export file lists, to quantify what is in your
data and to export subsets of it. The file lists are used for model training, or
provide accessible project and image metadata for depositing your data in
public repositories.

![Image showing the Export tab, numbered](images/export.png)

1. **Filters** choose the images: by series, project, beam, state and label.
   **Add filter** filters on anything else, such as the microscope or the sample.
2. **Breakdown** counts them by one property, or by two against each other.
   Switch to **Chart** to draw the counts:

   ![Image showing a breakdown drawn as a chart](images/export-chart.png)

3. **Measure labels** reads every label so that different metrics can be
   quantified, such as the area of each class in µm² and the number of cracks.
4. **Training split** sets how the images are divided into training and
   validation data:

   - *Keep together* keeps each lamella on one side of the split, as either
     training or validation data.
   - *Validate on corrected only* uses only corrected labels for validation.
     This is useful when you also train on predictions that have not been
     proofread (*corrected + selected* in the state filter).
   - *Validation from...* uses an earlier export as a starting point. New
     images and labels are added to the new lists. This keeps the validation
     list completely separate through successive rounds of training.
   - *Balance by* distributes images more equally between training and
     validation for a property, such as lamella type, microscope model or
     acceleration voltage.

   Set **Breakdown** to *training split* to count what each side holds, by
   microscope or kind of lamella.
5. **Write CSVs**: **Training CSV...** writes the image and label paths of a
   training and validation set, for model training. Empty labels are not
   exported in training CSVs. **Per-image metadata CSV...** and **Per-project
   metadata CSV...** write the metadata of each image or project, for
   depositing the data.
6. **Copy out and merge back**: **Export images and labels...** copies the files
   into a folder of their own, to work on elsewhere. **Merge a copy back...**
   combines the changes back into the original collection. This is useful for
   sharing or working on subsets of the data.

## 7. Train a model

The **Train** tab is used to train segmentation models, using the file lists
exported on the Export tab. The training itself is done by
[AM-model-training](https://github.com/rosalindfranklininstitute/AM-model-training),
the software the Adaptive Milling models were trained with. It needs the `train`
extra ([README](../README.md#installation)). Training takes a lot of compute, so
realistically you will only want to do this with an NVIDIA GPU.

### Check your GPU

The **Training batch** size may need to be lowered, depending on the memory of
your GPU. The line below it estimates the memory a batch needs:

| GPU memory | Largest batch likely to fit | For example |
|---|---|---|
| 8 GB | none | |
| 12 GB | 1 | RTX 3060 12 GB, RTX 4070 |
| 16 GB | 1 | RTX 5060 Ti 16 GB, RTX 4080 |
| 24 GB | 3 | RTX 3090, RTX 4090 |
| 32 GB | 4 | RTX 5090 |
| 48 GB | 6 | RTX A6000, L40S |
| 80 GB | 6, the published setting | A100, H100 |

These memory requirements are estimates, derived from training on a 16 GB and
an 80 GB GPU. Other considerations for training:

- 8 GB cards have not been tested.
- A larger batch size typically speeds up training.
- The number of training images does not affect the memory requirements, but
  does increase the training time.
- With a weaker GPU, we recommend refining an existing model with hundreds of
  images.
- For training new models with thousands of images, we recommend a
  workstation.

### Set up a run

Set up a training in **New run**:

1. **Training set**: select the last export made with **Training CSV...** with
   **Use the last export**, or use **Browse...** to choose a saved
   `<name>_manifest.yaml`.
2. Choose **New model** or **Refine a model**. Refining starts from the weights
   of an existing model, such as one downloaded from Zenodo. A new model starts
   from general ImageNet weights, and needs more epochs and more data to give
   good results.
3. Give the run a **Name** and choose the **Device** to use.
4. Lower **Training batch** to what your GPU can handle (see the table above).
5. Press **Start...**. It first checks every label and reports what it will do.
   Empty labels, or labels containing an unsupported class, stop the run before
   anything is written.
6. Press **Start** in the **Start training** window, which then closes. The
   training takes place in the background.

Several runs can be started in parallel, for example on a workstation with
multiple GPUs. We recommend choosing the GPU yourself rather than leaving it on
*auto*, as the memory usage of a run takes some time to go up after it starts.

The defaults of **Training batch** and the **Advanced settings** are those of
the Adaptive Milling paper (<https://doi.org/10.64898/2026.09.20.752565>).
Change them for a single run in the form, or for every run in *Settings* (F2).

The image paths of the training set are found from the scanned folder, or else
from the location of the `<name>_manifest.yaml` file.

**Stop** ends a run, which can take up to a minute. The epoch in progress is
lost, and a stopped run is not continued: start another, refining from one of
its checkpoints.

### Follow your runs

Runs that are training and runs that have finished are listed in **Runs**.

1. Choose the *Training directory* with **Browse...** at the top of the
   column. Keep it outside the data collection.
2. The mark before each run shows its status: not started, training, finished,
   stopped by you, interrupted or failed. F1 shows each mark and what the
   colours mean.
3. The columns help you track progress. Hover over a run for more information.

The columns show the current and maximum epochs, the best epoch and its score,
and the epoch you marked as best (see *Plots* below). The score is the mean of
the weighted-average IoU and Dice on the run's validation list, from 0 to 1. A
run can finish before its last epoch, as training ends early once it has
stopped improving.

As in the data collection, runs can be placed in sub-folders of the training
directory. **Expand** and **Collapse** open and close every folder. Right-click
a run to copy its path or reveal it in the file manager.

**Log** shows the live output of the run, refreshed every 2 seconds. Use it to
follow progress, and to find out when and why a run failed. **Open log** opens
the whole file.

### Visualise and compare training runs

AM-seg offers three ways to look at the results of training runs:

- **Plots** of the training metrics over all epochs.
- A colour-coded **Table** to compare training runs.
- A grid of validation images overlaid with the **Predictions** of different
  models.

All three can be done for subsets of the data, such as different microscopes or
lamella types, to assess model performance across these subsets.

#### Plots

**Plots** show the training metrics over all epochs. They are drawn while the
training is still running, starting once the first epoch has finished. Follow
progress within an epoch in the **Log**.

For the selected run, **Plots** shows its loss and a metric per class for each
epoch:

- Use the class toggles to choose which classes to show.
- Hold the mouse over a line to read its values.
- Scroll to zoom, and hold Space and drag to pan.

Once a run has finished, the plots show which epoch gave the best results. A
best epoch is assigned automatically, but you can mark another one as best.

**Epoch**, under the list of runs, lists the epochs whose weights are on disk,
each with its score. The buttons beside it act on the epoch shown:

- **Mark as best** sets the epoch to be used as the best model, instead of the
  one assigned automatically. A hollow star marks the run's own best and a
  filled one yours. Select the marked epoch and press **Unmark** to undo it.
- **Save a copy...** copies the weights of the epoch shown to wherever you keep
  your models. To predict with it, choose that file as the model on Curate.
- **Delete the others...** deletes the run's checkpoints except the epoch
  shown, about 170 MB each. It asks first and cannot be undone. The epoch you
  marked is always kept, and the run's own best too while **Keep the run's
  best** is on. Nothing is deleted while the run is training or being
  validated.

#### Table

Tick runs in the list, then open **Table**. It shows a row for each training
run and a block for each label class. The values of the chosen validation
metric are colour-coded.

- **Each cell** shows the metric for either:
  - the best or the marked epoch, or
  - the mean or median over a range of epochs.
- **Colours** sets the range the colour gradient is scaled between. With
  **Auto** on, it is the lowest and highest value in the table. To set a range
  yourself, switch **Auto** off and move the sliders with the mouse. The arrow
  keys step a handle by 0.01, and a double click resets it.

#### Predictions

**Predictions** shows a grid of images overlaid with the predictions of
different models. It uses the marked epoch, or else the best epoch, of every
run ticked in the **Runs** list. Validation images are used by default. Switch
**Unseen only** off to include images a run trained on.

1. Choose a **Microscope**, **Lamella type** and **Voltage**, or leave each on
   *any*, and set how many **Images**. Images are picked at random from the
   validation images that match.
2. Press **Pick images**. The images are predicted by every model on the
   **Device** set in **New run**.
3. Adjust the display:
   - Scroll to zoom and drag to pan. **Fit** resets the view.
   - **Show** shows one image at a time (or press 1 to 4). **All** shows every
     image again.
   - **Prediction**, or V, hides the predictions. **Opacity** sets their
     transparency. **Show label**, or holding B, shows each image's own label
     in place of the predictions.

Predictions are made and resized as on Curate, but are held in memory only:
nothing is written to the collection or recorded in its history. This needs
the `ml` extra.

The images are found by combining the relative paths in the training and
validation CSVs with the collection folder you scanned. This means
**Predictions** also work on another machine, provided the validation data is
there. Images that are not found are left out, and the line above the images
says how many.

#### See how a model does on a kind of image

The validation metrics in **Plots** and **Table** can be narrowed down to
subsets of the data. To see how a model does on one microscope or one lamella
type:

1. Leave **Validated on** at *each run's own list*.
2. Choose a property in **Split by**, such as Microscope, Lamella type or
   Voltage.
   - In **Plots**, the metrics are narrowed down to the subset selected in the
     dropdown list beside it.
   - In **Table**, the metrics are shown for each subset, for each class.

The number of images of each subset is reported as `n=12`, or `n=12-18` where
the runs compared hold different numbers.

- Images with no value for the property are shown as `(not recorded)`.
- An image of a project with several values, such as two organisations, counts
  in the part of each.

A run keeps what each image of its validation list scored as it trains. If
**Split by** is greyed out, hover over it to see why. The usual reason is a run
trained before AM-seg kept these scores. Select the run and press **Score own
list** to score it now. This scores every checkpoint still on disk, in the
background, and asks only to confirm. If the run did not record what its images
are of and no export beside it lists them, it asks once for the export the run
was prepared from.

#### Score a model on other data

To score a run on data it has never seen, such as another microscope's:

1. Export that data with **Training CSV...** on the Export tab.
2. Select the run and press **Validate on export...**.
3. Choose the exported `<name>_manifest.yaml`.
4. Choose the result in **Validated on**. **Split by** divides it as it does a
   run's own list, and *every set* shows it beside the runs' own.

Runs validated on the same export can be compared directly. If the run trained
on none of the images, all of them are scored. Otherwise only the export's
validation list is scored, and the confirmation says how many of the images the
run trained on.

Scoring runs every checkpoint over each image once, in the background. The
run's status shows its progress, and **Stop validating** ends it, keeping the
epochs it finished. It needs the `train` extra and, in practice, a GPU.

The same can be done with `am-seg train`, `show-training`, `validate`,
`compare-training`, `choose-best` and `stop-training`
([Train](command_line.md#train)).

## Undo and history

AM-seg records every change and keeps a copy of every label it replaces,
inside each project. To undo work:

- **Revert...** on Curate restores the chosen labels to before the last
  conversion or merge, or to a savepoint.
- **Compare versions** on Label restores any earlier version of one label.
- **Savepoint...** in the *Maintenance* card on Add data names the state of
  every label, so you can return to it later.

[Background](background.md#history) explains what is recorded, savepoints and
working on a copy.
