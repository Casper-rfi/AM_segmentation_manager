# Label

Proofread a lamella's labels at full resolution: correct what a prediction got
wrong, paint what it missed, and mark the result corrected. Everything you draw
stays in the window until you save it.

![The Label tab, proofreading 15-brave-tetra](images/label.png)

## Which lamella

The Label tab shows the lamella chosen on the [Curate](03_curate.md) tab; choose
another there, or step with **Up** and **Down**. Only frames worth proofreading
are visited: stepping with **Left** and **Right** passes over a deselected image,
one you decided against.

**SEM** and **FIB** choose the beam, and **Series** the series, each with its own
frames. Where an acquisition was imaged under several conditions, **Variant**
chooses which to draw on. The conditions of one family share one label, so an
edit reaches all of them. Choosing a condition of another family proofreads
that family's label instead: the frames visited become the ones its state says
are worth it, and the status line says which conditions the label on screen
covers. Edits waiting in the other family are kept, and saved with the rest.
Where a family has nothing selected in the beam, the tab shows one that has.

## Look closely

Scroll to zoom about the pointer, and hold **Space** or drag with the middle
button to pan. **G** fits the image to the pane again.

- **Show label** (**V**) draws the label over the image, or not; **Opacity**
  sets how strongly.
- **Show image** draws the image under the label, or leaves a flat dark ground
  instead. Against that ground, an annotation a few pixels across is easy to
  see, and so is the edge of the frame.
- **Min / max** sets the image's contrast, and nothing else.

## Paint

Choose a class in the row under the image, or press its key: **Q**, **W**,
**E**, **R** and **T**, in the order of the row. Then:

- **Brush** (**D**) paints with the class where you drag. **[** and **]**, or
  **Shift** and the scroll wheel, make it smaller or larger.
- **Fill** (**F**) floods the area of one class you click in with the class
  chosen.

The padlock beside each class **locks** it, so neither tool paints over it.
**Shift** with a class's key locks or unlocks it, and **A** locks every class but
background. With the lamella locked, a stroke across its edge changes only the
pixels outside it.

**Ctrl+Z** undoes a stroke, and **Ctrl+Shift+Z** does it again.

## Save

Edits are held in the window, not written. The tab says how many labels are
waiting, and the timeline marks each with a white band. Moving
to another lamella with edits unsaved asks whether to discard them.

**Ctrl+S**, or the save button beside undo and redo, saves the lamella's
edits. The dialog says what it would write. While *Enable editing* is switched
on, on the Curate tab, it offers two ways to save:

- **Save only**: write the labels, and leave their state as it is.
- **Save and mark corrected**: write them, and mark the images you reviewed
  corrected in the same step.

With editing off it offers **Save** alone, and says how to mark them corrected
as well.

Saving replaces the label file. The version it replaces is kept first, under
`AM_archive/`, unless it was a prediction nobody had touched, which the model
can make again.

## Find specks

![The speck finder ringing three specks on 16-casual-swan](images/specks.png)

A speck is a piece of a class a few pixels across, or a pinhole inside one:
often a slip of the brush, sometimes real. Switch on **Find specks** (**X**) to
ring every one on the frame, and count them along the timeline. It hides the
image and draws the label at full strength while it is on, and puts both back
after. **N** moves to the next speck. The slider beside it sets the largest
piece still ringed, in pixels.

Nothing is removed: whether a speck is a mistake is for you to decide, with the
brush.

## Compare versions

**Compare versions** opens a strip under the image listing the versions of
this label: every one kept when it was replaced, newest first, then the
predictions beside the image. The percentage says how much a kept version
differs from the one before it.

- **Hold to compare** (**B**) shows the chosen version in place of the label
  while held, so the two can be flicked between.
- **Show differences** marks every pixel where they disagree.
- **Restore...** puts the chosen version back. What is there now is kept first,
  so a restore can itself be undone.

## Keys

Every key and gesture on this tab is in [Keys and gestures](09_keys.md#label),
and **F1** lists them in the window.
