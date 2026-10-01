# Curate

Go through the images a lamella at a time, decide which are worth labelling,
and predict and convert labels for them. Proofreading those labels is the
[Label](04_label.md) tab's job; this one is where you choose what gets
proofread.

![The Curate tab, showing 03-secure-oyster beside its label](images/curate.png)

The tab has three columns: what to look at on the left, the frame in the
middle, and what to do to it on the right.

## Choose what to look at

**Series shown**, at the top left, decides which series the lists cover:
Adaptive Polishing, reference, alignment and *other*. *Other* starts switched
off; the lamella list says how many images it is leaving out.

Below it are the **projects** the scan found and the **lamellae** of the one
chosen. Click a row to look at it; type in the box above a list to filter it.
The chip beside each row shows how far curation has got, with one square for
each series switched on, SEM above and FIB below. A square says one of five
things: there are no images of that series; there are images, and nobody labels
that series; nothing is corrected yet; part is corrected; or everything is
corrected, or deselected as not wanted. **F1** shows what each looks like.

## Look at a frame

A **frame** is one acquisition: its SEM and FIB images, taken together. The
middle column shows the frame on screen twice, the image alone and the image
under its label, so you can judge the label against what it covers.

Along the top:

- **SEM** and **FIB** choose the beam (keys **1** and **2**).
- **Plots** shows the summary figure an alignment run saved, and **Grid** the
  project's grid overview with every lamella marked on it (keys **3** and
  **4**). Each is greyed out where the project has none.
- **Series** chooses which series the timeline is made of, and **Variant**
  which imaging condition, where a frame was imaged under several. Each
  condition is named by its folder, with its family in bold; a name too long
  for the box loses its middle, so its end still tells `fi_1` from `fi_16`.
  Where they form several families, each family is a heading in the list, with
  its conditions beneath. A star marks a family's primary, the image predicted
  from, where the family has more than one condition. A frame opens on its family's primary condition, and
  one not imaged under the condition chosen shows another of the same family.
- **Show label**, **Show image**, **Opacity** and **Min / max** change how the
  frame is drawn, and nothing else.

The **Grid** view is the one that is not of the frame: it shows where on the
grid the lamella is. Every lamella of the project is ringed, and the one you
are looking at is named. Where a project has several overviews, **Overview**
chooses between them, and a note under the picture says when the positions
are approximate.

![The Grid view: the project's grid overview, every lamella ringed and the one being looked at named](images/curate-grid.png)

Under the frame, the class legend names the colours, and a line of text names
the project, lamella and image. A note above the legend says when what you see
is not the whole story: no label, an empty one the segmentation software wrote,
or one resized to fit.

### The timeline

The strip at the bottom is the lamella's frames in order, SEM above and FIB
below. Each column is coloured by its state, and a thin line under it shows its
label: annotated, an empty file, or none. Both are of the family of conditions
on screen, where a beam was imaged under several.

| Colour | State |
|---|---|
| grey | deselected: no label is wanted here |
| blue | selected: a label is wanted and not yet finished |
| green | corrected: the label has been proofread |

Click or drag along it to move; **Left** and **Right** step a frame, **Up** and
**Down** a lamella.

## Change a state

![Curate with editing switched on, the state controls unlocked](images/curate-editing.png)

The right-hand column's **State** tab changes the state of label groups. It
waits for **Enable editing**, under the tabs, which unlocks the **Families**
tab too.

1. **Choose what to change.** With nothing else chosen, a change applies to the
   frame on screen. To change more, either tick projects or lamellae in the
   lists, or pick frames out on the timeline: **Ctrl-click** a frame, or
   **Shift-click** to extend a run; **Esc** drops them. Only one of these is
   used at a time, so what is ticked on screen is the whole of what will change.
   *Applies to* and *Covers*, at the top of the column, say how much that is.
2. **Choose the beam** under *Apply to*: SEM, FIB or both. Where what you
   chose holds several families of conditions of a beam, a second choice
   appears: **This family**, the one **Variant** shows, or **Every family**.
   Each family has its own label, so changing one leaves the others alone;
   *Covers* names the family it reaches.
3. **Choose the state**, or press **D**, **S** or **C**.

The change is written at once to the stage's `AM_state.yaml`, after a
question where more is at stake:

- changing more than the frame on screen asks first, counting every image the
  change reaches, since images sharing a label move with it;
- marking a group corrected that has no label, or an empty one, asks first;
- taking a group out of *corrected* asks first.

Under the state buttons, *SEM:* and *FIB:* show the state of the image on
screen for each beam.

## Correct the families

A folder's name places it in a family, and a family shares one label. Where
the names got it wrong -- `sem2kVb` showing the same scene as `sem-2kV`, or
`sem_mag2` taken at another magnification than the rest of `sem` -- correct it
here or from the command line. No folder is renamed: the correction is kept in
the project's `AM_families.yaml`, which travels with the data.

The **Families** tab, beside *State* at the top of the right-hand column,
shows the families of the project in view: each beam's families and their
conditions, a star on each family's primary, and *(moved)* on those the file
places. It lists the whole project, a correction reaching every lamella of it,
and greys a condition the lamella in view does not hold, *(not here)*; hover a
row for how many stages hold it. Selecting a condition shows it in the view, on
its own beam, so you can step through a family's conditions with the label on
top and see whether one label fits them all; choosing one under **Variant**
selects its row.

With **Enable editing** on, select a condition and choose its **Family**: one
of the beam's, the one its name gives, or *New family...* to name one. **Set
primary** names the family's primary, and **Reset primary** goes back to the
rule below. Changes show as *not saved* until **Save**, which checks them as
`edit-families` does, asks before taking images out of a corrected group, and
first asks you to save or discard label edits waiting in the project;
**Preview** says what saving would do. The folder is read again afterwards.
From the command line:

```bash
am-seg show-families /path/to/collection --project 003
am-seg edit-families /path/to/collection --project 003 --set sem_mag2=sem-mag2
am-seg edit-families /path/to/collection --project 003 --set sem_mag2=sem-mag2 --apply
```

`show-families` lists each family and its conditions, marking with `*` those
the file places. `edit-families` shows which label each condition would read
and every image whose state would change, and writes only with `--apply`. A
condition joining a family shares its labels and its state; one leaving starts
unlabelled, keeping a deselection but not a correction. A change is refused if
it would leave a label unread, swap one label for another, take a label
from a condition you did not name, or have two families read one label folder
-- which moving a family's base condition, such as `sem`, out of it would do,
its folder name being the family's own; move the family's other conditions
out of it first. One taking images out of a corrected group needs
`--confirm-corrected`. `--clear` puts a condition back where its
name places it.

**Inspect layout** tells you when a correction is needed. It compares every
two conditions of a family that hold an image of the same name, which share
that image's label: images of different sizes are an error,
since one label cannot fit both, and a different pixel size or accelerating
voltage is a warning that the conditions may not be one family.
`edit-families` makes the same comparison for a condition joining a family,
refusing a change of size and warning about the others. Two families reading
one label folder, which a hand edit of `AM_families.yaml` can cause, is an
error too.

Each family has a **primary condition**: the one predicted from, measured at,
kept by *One image per label*, and shown first in the window. It is the base condition, or where a stage
has none, the first variant by name. `--primary sem=sem_fi8` names another,
for instance where the base is too noisy to predict on; `--clear-primary sem`
returns to the default. It moves no label, and predictions already made stay
until you predict again. A stage without the named condition uses the default.

A lamella can have entries of its own, under `lamellae:` in the file, which
replace the project's in that lamella alone: for a folder name meaning one
family in one lamella and another elsewhere.

```yaml
version: 2
families:
  sem_x: sem-5kV
lamellae:
  lam_003:
    families:
      sem_x: sem-2kV
```

On the Families tab, **Applies to** chooses whether a change is the whole
project's or the lamella in view's own; it is fixed while changes are unsaved,
which belong to the choice they were made under. Within a lamella, the family
the project gives a condition is its starting point, and choosing it again
removes the lamella's entry. From the command line, `edit-families --lamella
lam_003` does the same, `--clear` returning the lamella to the project's. Each
lamella is checked as a project-wide change is, `show-families` lists each
lamella's entries after the families, and a merge brings them across as it
does the project's.
**Inspect layout** reports a section naming no lamella of the project.

## Predict and convert

*Operations on this frame*, at the bottom right, act on the same choice as the
state buttons: the ticked rows, the frames picked out, or the frame on screen.
Each one that writes shows what it would do before doing it, and waits for
*Enable editing* too. Predicting needs the `ml` extra installed.

**Predict labels...** runs a trained model over the chosen images.

- Name the model's weights file in the box above, or **Browse...** for it. The
  window remembers one model for each series and beam.
- **Device** chooses where it runs: `auto` takes the GPU with most memory free,
  or names one; `cpu` always works, slowly.
- A model predicts one beam, so a choice covering both SEM and FIB images is
  refused. One prediction is made per label group, from its primary condition
  (see [Correct the families](#correct-the-families)), which **Variant** marks
  with a star and the preview names: *Would predict 12 SEM image(s) from sem*.
  An image this model has already predicted is passed over, so a run stopped
  halfway can be started again.
- Predictions go in the image folder's `_predictions` folder, never into a label.

The models Adaptive Milling was trained with are published on Zenodo
([doi:10.5281/zenodo.21804785](https://doi.org/10.5281/zenodo.21804785)).
`AM_SEM_All_V01.pth` is the one trained on all the data.

**Convert predictions...** turns each prediction into a label, ready to
proofread. **From** chooses whose predictions to use where an image has both:
*am-seg*'s, or the *Microscope*'s, which Autolamella made while milling. The line
under the button says what converting would find. Replacing a label that holds
annotation has to be confirmed, and a corrected label is never replaced.

**Revert...** puts the chosen labels back to what they held before the last
conversion or merge, or at a savepoint. A label with no kept version, or edited
since, is reported and left alone.

## Find specks

**Find specks...** reads every label in the choice and reports the ones holding
pieces too small to be likely real: a stray brush stroke, or a pinhole in a
class. **At most** sets how small, in pixels. Some are genuine, so it changes
nothing; open the lamella on the [Label](04_label.md#find-specks) tab to look
at each one.

## From the command line

| To | Run |
|---|---|
| change states | `am-seg select`, `deselect`, `correct` |
| predict | `am-seg predict --model <weights.pth> --modality SEM ...` |
| convert | `am-seg convert-predictions --modality SEM ...` |
| find specks | `am-seg find-specks ...` |

Each takes the options in [The command line](08_command_line.md#pointing-a-command-at-part-of-a-collection)
to choose which images, and each that writes takes `--dry-run` to say what it
would do first.
