# The command line

Everything the window does can also be done from the command line, for
scripting a job or working on a machine without a display. The command line
also holds a few reports on a whole collection that the window does not show.

```bash
am-seg --help                 # every command, under its step of the work
am-seg <command> --help       # one command: its options, and how to read its output
```

From a checkout, put `uv run` in front: `uv run am-seg --help`.

## Choosing what a command acts on

A command that acts on images takes the folder with `--start-path`, and needs
every level named, so that nothing is acted on by accident: `--project`,
`--lamella`, `--stage` and `--image`, each `all` for every one. The examples
below write `<all>` for `--project all --lamella all --stage all --image all`.
The same options narrow it down:

| Option | Narrows to |
|---|---|
| `--project`, `--lamella`, `--stage`, `--image` | one of each, by name or number; `all` for every one |
| `--series` | `adaptive`, `reference`, `alignment`, or several separated by commas |
| `--task` | one milling task, by the name the project's protocol gave it |
| `--state` | `selected`, `deselected`, `corrected`, or `all` |
| `--modality` | `SEM`, `FIB`, or `all` |

`inspect-layout` and `stats` report on every series; every other command acts
on Adaptive Polishing unless told otherwise, and `--series all` means every
series that command covers.

`--task` takes in every stage of one milling task, as one task can span several
stage folders (`Rough 1`, `Rough 2`). Task names come from the project's own
`experiment.yaml`, so they are whatever its protocol called them; case and
separators are ignored, and `--task none` chooses the images naming no task.

Most commands that write take `--dry-run`, to say what they would do, and the
rest only report until given `--apply`.

## Add data

```bash
am-seg inspect-layout /path/to/collection     # how the folder is read; changes nothing
am-seg init-project /path/to/dataset          # make a folder a project
am-seg init-project /path/to/collection --recursive --defaults shared.yaml
am-seg show-metadata /path/to/collection
am-seg edit-metadata /path/to/collection --all --set organisation="Rosalind Franklin Institute" --dry-run
```

`--defaults` names a YAML file of values the projects share, such as the
organisation and user; a field that differs between projects, such as the
sample, is left for you to fill in. Change metadata with `edit-metadata` or in
the window rather than in the file itself: both record each change in the
project's history, and merging a copy back sees only recorded changes.

## Curate and predict

```bash
am-seg deselect --start-path /path/to/collection --project 001 --lamella 03 --stage all --image all --modality SEM --dry-run
am-seg predict --start-path /path/to/collection <all> --model AM_SEM_All_V01.pth --modality SEM
am-seg convert-predictions --start-path /path/to/collection <all> --modality SEM --dry-run
am-seg find-specks --start-path /path/to/collection <all>
```

`select`, `deselect` and `correct` set the state of label groups. `predict`'s
`--device` names a GPU as `nvidia-smi` numbers it; `auto` takes the one with
the most memory free.

### Families

```bash
am-seg show-families /path/to/collection --project 003
am-seg edit-families /path/to/collection --project 003 --set sem_mag2=sem-mag2
am-seg edit-families /path/to/collection --project 003 --set sem_mag2=sem-mag2 --apply
```

`show-families` lists each family and its conditions, marking with `*` those
`AM_families.yaml` places. `edit-families` shows which label each condition
would read and every image whose state would change, and writes only with
`--apply`. `--clear` puts a condition back where its name places it,
`--primary sem=sem_fi8` names a family's primary condition and
`--clear-primary sem` returns it to the default; `--lamella` makes any of these
the lamella's own. A change taking images out of a corrected group needs
`--confirm-corrected`.

A change is refused if it would leave a label unread, swap one label for
another, take a label from a condition you did not name, or have two families
read one label folder. Moving a family's base condition, such as `sem`, out of
it would do the last, its folder name being the family's own: move the
family's other conditions out first.

## Quantify and export

```bash
am-seg stats --start-path /path/to/collection <all> --by series
am-seg stats --start-path /path/to/collection <all> --by lamella --by state
am-seg quantify-labels --start-path /path/to/collection <all> -o areas.csv
am-seg create-training-csv --start-path /path/to/collection <all> --state corrected --series reference,adaptive
am-seg export-files --start-path /path/to/collection <all> -o /path/to/copy --dry-run
am-seg check-training-csv dataset_train.csv
```

`--by` twice cross-tabulates two properties. `quantify-labels` measures
corrected labels unless told otherwise, since a label made from a prediction
measures the model rather than the sample.

## Train

```bash
am-seg train export/set_manifest.yaml --training-dir /path/to/runs --name round02 --dry-run
am-seg train export/set_manifest.yaml --training-dir /path/to/runs --name round02 \
    --refine AM_SEM_All_V01.pth
am-seg show-training /path/to/runs/round02 --plot curves.png --classes gis lamella crack
am-seg stop-training /path/to/runs/round02
```

`train` needs the `train` extra, and takes the manifest a training export
wrote: the split is the export's, so validation is whatever **Keep together**
kept out of training. Without `--refine` it trains a new model, the encoder
frozen for the first 25 of at most 100 epochs and the learning rate from 1e-5
to 1e-3, which downloads ImageNet weights the first time. With it, it refines
the model given for at most 50 epochs from 2e-6. Each has an option. A run
stops early once it stops improving.

Everything is checked before anything is written, and every problem reported at
once. Each label is opened again: one that will not read, holds only
background, or holds a class the model does not predict stops the run. Keep the
training directory outside the collection. The run gets a directory there named
for it, and a run directory is never reused.

Training runs in a process of its own and goes on after the command returns,
and after the terminal is closed. `show-training` says how a run stands:
training, finished, failed and why, stopped, or ended without finishing. It
also reads what AM-model-training wrote, for a run started anywhere, without
the training stack: the epochs so far, the validation epoch the run counts as
best, and the checkpoints kept, `<name>_val_epochNNN.pth`. `--plot` draws one
run's loss and per-class metric by epoch.

`stop-training` ends a run started on this computer. The epoch in progress is
lost and every checkpoint already written is kept. A stopped run is not
continued: start another, refining from one of its checkpoints.

AM-model-training trains on an NVIDIA GPU or the CPU. On the CPU, Apple
silicon included, a step takes minutes per image.

## Record and undo

```bash
am-seg init-history --start-path /path/to/collection            # see what it would record
am-seg init-history --start-path /path/to/collection --apply    # record it
am-seg savepoint --start-path /path/to/collection --title "proofreading 1" --apply
am-seg list-savepoints --start-path /path/to/collection
am-seg history /path/to/project --kind state-changed --author "Ada"
am-seg revert --start-path /path/to/collection <all> --modality SEM --undo conversion
am-seg revert --start-path /path/to/collection <all> --modality SEM --to "proofreading 1"
```

Run `init-history` once on a collection that already holds work, so a
savepoint covers every project; `--as NAME` credits labels that arrived already
annotated to whoever annotated them. `savepoint --close-gap` also keeps the
labels only a model has written. `archive` reports the label versions a project
keeps, and `prune-archive` removes those no savepoint needs.

### Merging a copy back

```bash
am-seg merge /path/to/copy --into /path/to/collection
am-seg merge /path/to/copy --into /path/to/collection \
    --resolve AutoLamella-2024-07-05-12-53/03-secure-oyster=theirs --apply
```

`--resolve` decides conflicts: one by its path, or every one under a folder,
project or lamella. A family correction is named after its project, as
`<project>:family:sem_x`, or `<project>:family:lam_003/sem_x` for a lamella's
own entry. A label conflict waiting on a family decision appears only once that
decision is made, so decide the family, run the merge again, then decide the
labels.

## Every command

<!-- generated: commands -->

### Add data to the collection

| Command | What it does |
|---|---|
| `am-seg inspect-layout` | Report how a directory tree is interpreted, without changing anything |
| `am-seg init-project` | Write an AM_metadata.yaml template into a project |
| `am-seg show-metadata` | Show each project's resolved metadata, and where each value came from |
| `am-seg edit-metadata` | Set or clear fields in a project's AM_metadata.yaml |
| `am-seg list-projects` | Scan a directory and list discovered projects |

### Curate it, and prepare labels

| Command | What it does |
|---|---|
| `am-seg gui` | Open the curation window to review images and set their state |
| `am-seg predict` | Create predictions for selected images with a trained model |
| `am-seg convert-predictions` | Convert predictions into labels for explicitly selected images |
| `am-seg deselect` | Declare label groups to be in the deselected state |
| `am-seg select` | Declare label groups to be in the selected state |
| `am-seg correct` | Declare label groups to be in the corrected state |
| `am-seg show-families` | Show each project's families of imaging conditions |
| `am-seg edit-families` | Move imaging conditions between families in AM_families.yaml |
| `am-seg find-specks` | Find labels holding components likely to be too small to be real |
| `am-seg compress-labels` | Recompress on-disk label files to reduce storage size |

### Quantify it, and export

| Command | What it does |
|---|---|
| `am-seg stats` | Count images by one or two properties |
| `am-seg quantify-labels` | Measure what the labels contain, in physical units |
| `am-seg export-metadata-csv` | Write the acquisition and project metadata of the selected images |
| `am-seg export-projects-csv` | Write one CSV row per project, with its metadata and counts |
| `am-seg export-files` | Copy the selected images, their labels and the project files out |
| `am-seg create-training-csv` | Export image and label paths to CSV files for ML model training |
| `am-seg check-training-csv` | Check that image/label paths recorded in a training CSV still exist |

### Train models

| Command | What it does |
|---|---|
| `am-seg train` | Train a new model, or refine an existing one, on an exported training set |
| `am-seg show-training` | Report how a training run is going or how it ended, and draw its curves |
| `am-seg stop-training` | Stop a training run that am-seg started on this computer |

### Record what changes, and read it back

| Command | What it does |
|---|---|
| `am-seg init-history` | Start recording a collection's history, for the data already on disk |
| `am-seg savepoint` | Name the state of the data, so a round of work can be undone |
| `am-seg list-savepoints` | List the savepoints here, what each covers and what it can reach |
| `am-seg history` | Print what has been recorded about a project or one of its labels |

### Put work back, and reclaim the space

| Command | What it does |
|---|---|
| `am-seg merge` | Bring a worked-on copy back into the collection it came from |
| `am-seg revert` | Put a selection of labels back to what they held before an operation |
| `am-seg archive` | Report the label versions a project has kept, and what they cost |
| `am-seg prune-archive` | Remove the kept versions no savepoint needs |

### Set up this computer

| Command | What it does |
|---|---|
| `am-seg create-shortcut` | Add the window to this computer's applications, to start it like any other |

<!-- end generated -->
