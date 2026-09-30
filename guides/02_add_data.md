# Add data

Say which folders are projects, check they are read the way you expect, and
describe each one so its metadata travels with every export. Nothing on this
tab touches an image, and only its *Maintenance* card touches a label.

![The Add data tab, describing one project](images/add-data.png)

## Open a folder

Choose the folder holding your collection with **Browse...** at the top of the
window, then **Scan**. The same folder serves all four tabs. You can also open
one with *File ▸ Open folder...* (Ctrl+O), or name it when starting the window:

```bash
am-seg gui /path/to/collection
```

A scan reads the header of every image, so a large collection takes a while
the first time; a banner shows how far it has got. **F5** reads the folder
again, keeping your place, after anything outside the window has changed it.

## Check how the folder is read

**Inspect layout** reports how am-seg has read the folder, and changes nothing.
It is worth running on any data you have not opened before. For each project it
gives the lamellae, stages, images and labels found, and how many images fall in
each series; then its findings, from errors down to notes:

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
sitting in the folder but inside no project, which nothing else will show you.
`am-seg inspect-layout /path/to/collection` prints the same report.

## Name the projects

A folder holding an Autolamella `experiment.yaml` is a project without being
asked, and appears in the list on the left. Data from before Autolamella, or a
project whose `experiment.yaml` has gone, has to be named: the line under the
*Data* card says how many folders hold images that no project covers.

**Add projects...** names them. Switch on *Enable editing* on this tab first.
The dialog holds two lists:

- **Projects already indexed** start ticked. Each is given an
  `AM_metadata.yaml` to describe it, where it has none.
- **Folders holding unindexed images** are proposals, each saying what it would
  become. A folder that gathers several datasets is left unticked, with the
  reason; **Look inside** replaces it with the folders below it, so you can
  pick the right level.

**Write metadata files** puts an `AM_metadata.yaml` into every folder ticked,
which makes it a project. A file already there is never overwritten.

From the command line, `init-project` does the same for one folder, or with
`--recursive` for every project under it. `--defaults` names a YAML file of
values the projects share, such as organisation and user, to copy into each;
a field that differs between projects, such as the sample, is never filled in
this way and is left for you to complete.

## Describe a project

Click a project to see its metadata. Click a **folder** instead to describe
every project under it at once: projects from one session usually share their
sample, grid and user, and typing a value once is how a collection avoids
`HeLa` beside `Hela`. A field the projects disagree on is shown as mixed, and is
left alone unless you change it.

Switch on **Enable editing** to change a field. am-seg writes only
`AM_metadata.yaml`; `experiment.yaml` belongs to Autolamella and is never
touched. A value in `AM_metadata.yaml` takes the place of the same field in
`experiment.yaml`, field by field, so one can be corrected without restating
the rest. Change metadata here or with `am-seg edit-metadata` rather than in
the file itself: both record each change in the project's history, and
merging a copy back compares only recorded changes, so it misses an edit made
by hand.

While you type, a field offers the values other projects in the collection
already use, matching anywhere in the word: `foil` finds both `Quantifoil` and
`UltrAuFoil`. A suggestion becomes a value only when you save.

**Preview changes** says what saving would change: every field for one
project, or how many projects and which fields for a folder. **Save** writes
it; **Discard changes** puts every field back to what is saved.

The dot beside each field says where it stands:

| Dot | Means |
|---|---|
| filled | the field has a value; hover over the dot for the file it came from |
| hollow | not filled in yet, or still a placeholder |
| mixed | the projects in this folder do not agree on it |
| edited | changed here, and not saved yet |

The bar beside each project in the list shows how much of it is described.
Hover over a row for what is still missing.

### The fields

<!-- generated: fields -->

**General**

| Field | What to enter | Takes |
|---|---|---|
| Description | What this experiment was for | text |
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

From the command line, `show-metadata` prints each project's metadata and
where every value came from, and `edit-metadata` changes it:

```bash
am-seg show-metadata /path/to/collection
am-seg edit-metadata /path/to/collection --all --set organisation="Rosalind Franklin Institute" --dry-run
```

## Correct the families

A folder's name places it in a family, and a family shares one label. Where
the names got it wrong -- `sem2kVb` showing the same scene as `sem-2kV`, or
`sem_mag2` taken at another magnification than the rest of `sem` -- correct it
from the command line. No folder is renamed: the correction is kept in the
project's `AM_families.yaml`, which travels with the data.

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
it would leave a label unread, swap one label for another, or take a label
from a condition you did not name, and one taking images out of a corrected
group needs `--confirm-corrected`. `--clear` puts a condition back where its
name places it.

## Maintenance

The *Maintenance* card acts on every label under the folder, and waits for
*Enable editing* like everything else on this tab:

| Button | Does |
|---|---|
| **Savepoint...** | Records what every label holds now, under a name you give it, so a round of work can be undone. |
| **Restore savepoint...** | Puts every label back to what it held at a savepoint. A corrected label is never replaced without being named to you first. |
| **Compress labels...** | Re-encodes every label so it takes less disk space. No pixel changes, and a file that would not shrink is left alone. |
| **Reclaim space...** | Removes the kept label versions that no savepoint needs. |

On a collection that already holds work, run `am-seg init-history` once first,
so a savepoint covers every project; [History](06_history.md) says why.
