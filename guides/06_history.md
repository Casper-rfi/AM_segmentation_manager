# History

am-seg keeps a record of what changed in each project, who changed it and
when, and keeps every label version worth keeping. That record is what lets you
name a round of work, undo part of it, and bring home a copy worked on
somewhere else.

## Recording is always on

The first time anything changes a project, am-seg starts its record: an
`AM_history.jsonl` in the project folder, beginning with what the project held
at that moment. From then on every change is added to it: a state changed, a
label saved, converted, restored or merged, a metadata field edited, a
condition moved between families, a duplicate label removed by
`scripts/remove_duplicate_condition_labels.py`.
Looking never writes; only a change does.

On a collection that already holds work, run `init-history` once. It brings
every project in at once, so a savepoint can cover the whole collection rather
than only the projects that happened to have been changed:

```bash
am-seg init-history --start-path /path/to/collection            # see what it would record
am-seg init-history --start-path /path/to/collection --apply    # record it
```

For a dataset that arrived already annotated, `--as NAME` credits its labels
to whoever annotated them.

### Whose name is recorded

A change is recorded under the name set in *Settings* (**F2**), or given with
`--user` or `AM_SEG_USER`. With none set, it is recorded under no name: am-seg
never guesses one from the computer's account. The corner of the window says
which it is.

The record lives inside each project. **Depositing a collection deposits its
history**, and the names in it with it.

### Reading it

`am-seg history` prints what a project's record holds, or one label's, and can
narrow it by kind of change or by who made it:

```bash
am-seg history /path/to/project
am-seg history /path/to/project/lamella/sem_labels/image_001.tif
am-seg history /path/to/project --kind state-changed --author "Ada"
```

In the window, *Compare versions* on the [Label](04_label.md#compare-versions)
tab lists the versions of the label on screen, with who made each and when.

## Kept versions

Before a label is replaced, by a save, a conversion, a restore or a merge, the
version it replaces is copied to `AM_archive/` in the project folder, as a
whole label file that opens on its own. A prediction nobody touched is not kept,
since the model can make it again.

`am-seg archive` reports how many versions a project keeps, and how much space
they take.

## Savepoints

A **savepoint** names the state of every label at a moment you choose: the end
of a round of proofreading, say, before training a model on it. Take one with
*Savepoint...* on the [Add data](02_add_data.md#maintenance) tab, or:

```bash
am-seg savepoint --start-path /path/to/collection --title "proofreading 1" --apply
```

Give it a name you will recognise months later; nothing suggests one. A
savepoint covers labels; it does not cover states or metadata.

`am-seg list-savepoints` lists them, what each covers, and how much of it can
still be put back. A savepoint reports what it can reach rather than promising
it: a label only a model had written, and nobody touched, may not be kept.
**Also keep labels the model made**, in the savepoint dialog, or `--close-gap`,
keeps those too, at the cost of disk space.

## Undoing work

| To put back | In the window | From the command line |
|---|---|---|
| what the last conversion replaced, for a selection | *Revert...* on Curate | `am-seg revert --undo conversion` |
| what the last merge replaced, for a selection | *Revert...* on Curate | `am-seg revert --undo merge` |
| a selection, to a savepoint | *Revert...* on Curate | `am-seg revert --to <savepoint>` |
| every label under the folder, to a savepoint | *Restore savepoint...* on Add data | `am-seg revert --to <savepoint>`, with every project chosen |
| one label, to any kept version | *Compare versions*, then *Restore...*, on Label | |

A label edited since the version it would go back to is reported and left
alone, and a corrected label is not replaced unless you name it. What is there
now is kept first, as with any label replaced, so putting a version back can
itself be undone. From the
command line, `revert` only reports until given `--apply`.

## Reclaiming space

Kept versions add up. *Reclaim space...* on Add data, or `am-seg
prune-archive`, removes the versions no savepoint needs. Nothing since the most
recent savepoint is removed, so a round still open is never touched, and the
record itself is unchanged: it still says what happened, even where the
version it names has gone.

## Working on a copy

To proofread on another computer, copy a selection out with *Export images and
labels...* on the [Export](05_export.md#the-files-themselves) tab, or
`am-seg export-files`. The copy is a collection of its own, carrying the
history of every project in it and an `AM_export.yaml` saying where it came
from. Kept versions travel only if asked, with `--with-archive`.

When the work is done, bring it home with *Merge a copy back...* on Export, or
`am-seg merge`. The merge compares both sides against the moment the copy was
taken, so a change made on only one side is simply taken. A **conflict** is
something both sides changed differently since, and only a person decides it:
the merge shows the metadata fields in conflict first, then the labels, and
merges nothing until every conflict is answered. Decide one at a time, or a
whole folder, project or lamella at once. A label's state travels with its
pixels, and a label corrected here is never written over unless you choose it.
Nothing is deleted, and both versions of a label in conflict stay in the
archive whichever way it goes.

From the command line, `merge` reports what it would do until given
`--apply`, and `--resolve` decides conflicts: one by its path, or every one
under a folder, project or lamella:

```bash
am-seg merge /path/to/copy --into /path/to/collection
am-seg merge /path/to/copy --into /path/to/collection \
    --resolve AutoLamella-2024-07-05-12-53/03-secure-oyster=theirs --apply
```
