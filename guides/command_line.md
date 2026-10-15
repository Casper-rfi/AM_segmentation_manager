# The command line

Everything the window does can also be done from the command line, which is
useful for scripting a job or working on a machine without a display. It also
holds a few reports on a whole collection that the window does not show.

```bash
am-seg --help                 # every command, under its step of the work
am-seg <command> --help       # one command: its options, and how to read its output
```

From a checkout, put `uv run` in front: `uv run am-seg --help`.

## Choosing what a command acts on

A command that acts on images takes the folder with `--start-path`. It needs
every level named, so that nothing is acted on by accident: `--project`,
`--lamella`, `--stage` and `--image`, each `all` for every one. The examples
below write `<all>` for `--project all --lamella all --stage all --image all`.
These options narrow the choice down:

| Option | Narrows to |
|---|---|
| `--project`, `--lamella`, `--stage`, `--image` | one of each, by name or number; `all` for every one |
| `--series` | `adaptive`, `reference`, `alignment`, or several separated by commas |
| `--task` | one milling task, by the name the project's protocol gave it |
| `--state` | `selected`, `deselected`, `corrected`, or `all` |
| `--modality` | `SEM`, `FIB`, or `all` |

- `inspect-layout` and `stats` report on every series. Every other command acts
  on Adaptive Polishing unless told otherwise, and `--series all` means every
  series that command covers.
- `--task` takes in every stage of one milling task, as a task can span several
  stage folders (`Rough 1`, `Rough 2`). Task names come from the project's own
  `experiment.yaml`. Case and separators are ignored, and `--task none` chooses
  the images naming no task.
- Most commands that write take `--dry-run`, to say what they would do. The
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
organisation and user. Fields that differ between projects, such as the sample,
are left for you to fill in.

Change metadata with `edit-metadata` or in the window, not in the file itself.
Both record each change in the project's history, and merging a copy back sees
only recorded changes.

## Curate and predict

```bash
am-seg deselect --start-path /path/to/collection --project 001 --lamella 03 --stage all --image all --modality SEM --dry-run
am-seg predict --start-path /path/to/collection <all> --model AM_SEM_All_V01.pth --modality SEM
am-seg convert-predictions --start-path /path/to/collection <all> --modality SEM --dry-run
am-seg find-specks --start-path /path/to/collection <all>
```

`select`, `deselect` and `correct` set the state of label groups. The
`--device` of `predict` names a GPU as `nvidia-smi` numbers it, and `auto` takes
the one with the most memory free.

### Families

```bash
am-seg show-families /path/to/collection --project 003
am-seg edit-families /path/to/collection --project 003 --set sem_mag2=sem-mag2
am-seg edit-families /path/to/collection --project 003 --set sem_mag2=sem-mag2 --apply
```

`show-families` lists each family and its conditions, marking with `*` those
`AM_families.yaml` places. `edit-families` shows which label each condition
would read and every image whose state would change, and writes only with
`--apply`.

- `--clear` puts a condition back where its name places it.
- `--primary sem=sem_fi8` names a family's primary condition, and
  `--clear-primary sem` returns it to the default.
- `--lamella` makes any of these the lamella's own.
- A change taking images out of a corrected group needs `--confirm-corrected`.

A change is refused if it would leave a label unread, swap one label for
another, take a label from a condition you did not name, or have two families
read one label folder. Moving a family's base condition, such as `sem`, out of
it would do the last, so move the family's other conditions out first.

## Quantify and export

```bash
am-seg stats --start-path /path/to/collection <all> --by series
am-seg stats --start-path /path/to/collection <all> --by lamella --by state
am-seg quantify-labels --start-path /path/to/collection <all> -o areas.csv
am-seg create-training-csv --start-path /path/to/collection <all> --state corrected --series reference,adaptive
am-seg create-training-csv --start-path /path/to/collection <all> --state corrected selected --validate-on corrected --modality SEM
am-seg export-files --start-path /path/to/collection <all> -o /path/to/copy --dry-run
am-seg check-training-csv dataset_train.csv
```

- `--by` twice cross-tabulates two properties.
- `quantify-labels` measures corrected labels unless told otherwise, since a
  label made from a prediction measures the model rather than the sample.
- `--validate-on corrected` trains on every state `--state` names and validates
  on corrected labels alone, on the list `--state corrected` gives with the same
  seed. A label in another state is left out where its lamella is validated on.
- `--validation-from first_manifest.yaml` keeps every lamella on the side an
  earlier export put it, and splits only the lamellae that export did not hold.
- `--balance lamella-type` gives validation its share of each kind of lamella.
- `--dry-run` says how the rows would divide and writes nothing. With
  `--by lamella-type` it counts each side by that property.

## Train

```bash
am-seg train export/set_manifest.yaml --training-dir /path/to/runs --name round02 --dry-run
am-seg train export/set_manifest.yaml --training-dir /path/to/runs --name round02 \
    --refine AM_SEM_All_V01.pth
am-seg show-training /path/to/runs/round02 --plot curves.png --classes gis lamella crack
am-seg stop-training /path/to/runs/round02
am-seg choose-best /path/to/runs/round02 --epoch 74
am-seg validate /path/to/runs/round02 held_out/hydra_manifest.yaml --rows all
am-seg validate /path/to/runs/round01
am-seg compare-training /path/to/runs --value median --epochs 60 90 --output runs.csv --plot runs.png
am-seg compare-training /path/to/runs --validated-on hydra --split-by microscope_model
am-seg compare-predictions /path/to/runs --output sheet.png --count 4 --microscope Arctis
```

### Train a model

`train` needs the `train` extra, and takes the manifest a training export wrote.
The split is the export's, so validation is whatever **Keep together** kept out
of training.

- The collection is found from where the manifest sits. Where the two have
  been moved apart, name it with `--start-path`.
- Without `--refine` it trains a new model: the encoder is frozen for the first
  25 of at most 100 epochs, and the learning rate goes from 1e-5 to 1e-3. The
  ImageNet weights are downloaded the first time.
- With `--refine` it refines the model given, for at most 50 epochs from 2e-6.
- Each of these has an option. A run stops early once it stops improving.

Everything is checked before anything is written, and every problem is
reported at once.

- `train` warns, without refusing, where the batch is likely to need more
  memory than the GPU has, and names the largest that should fit
  ([what a run needs](user_guide.md#7-train-a-model)).
- Each label is opened again. One that will not read, holds only background, or
  holds a class the model does not predict stops the run.
- Keep the training directory outside the collection. The run gets a folder
  there named for it, and a run folder is never reused.

Training runs in a process of its own and goes on after the command returns,
and after the terminal is closed. AM-model-training trains on an NVIDIA GPU or
the CPU. On the CPU, Apple silicon included, a step takes minutes per image.

### Follow a run

`show-training` says how a run stands: training, finished, failed and why,
stopped, or ended without finishing, and how many training and validation
pairs it was prepared with. It also reads what AM-model-training wrote, for a
run started anywhere and without the training stack: the epochs so far, the
validation epoch the run counts as best, and the checkpoints kept
(`<name>_val_epochNNN.pth`). `--plot` draws one run's loss and per-class metric
by epoch.

- `choose-best` marks the epoch you count as a run's best where it is not the
  one the run chose, or takes that back with `--clear`. `show-training`
  reports both, and `--plot` marks both.
- `stop-training` ends a run started on this computer. The epoch in progress is
  lost and every checkpoint already written is kept. A stopped run is not
  continued: start another, refining from one of its checkpoints.

### Score a run on other data

`validate` scores a run's checkpoints on the rows of another training export:
its validation list, or with `--rows all` every row, for an export made of data
held out whole.

- Every checkpoint with whole weights on disk is scored, unless `--epochs`
  names some. Each is scored over each image once.
- Scoring runs in a process of its own that `show-training` reports.
  `stop-training`, given the validation's folder, ends it.
- What an epoch scored on each image is kept in the run, under
  `am_validations/<name>`, with the metadata the export wrote for it.
- Scoring loads the model, so it needs the `train` extra and, in practice, a
  GPU.
- The validation is called what the export is, `hydra` for
  `hydra_manifest.yaml`, unless `--name` says otherwise. Runs validated on one
  export share a name. The output says how many of the images the run trained
  on.

With no export, `validate` scores the run's own validation list instead, into
`am_validation_scores`, where a run keeps that as it trains. Use this for a run
trained before AM-seg did so, or for the epochs one missed. It needs no name.
Where the run did not keep what its images are of, `validate` looks for the
export's manifest beside the run, and `--own` with a manifest names it.

### Compare runs

`compare-training` sets every run under a folder side by side, with a row per
run and a column per class from each run's validation. Each value is taken at
the best epoch, at the epoch marked (`--value marked`), or as a `mean` or
`median` over `--epochs`. A run with nothing validated there has empty cells,
never zeros.

- `--plot` shades the table from 0 to 1. `--colour-range 0.8,1` runs the
  colours between two values instead, and `--colour-range auto` between the
  table's lowest and highest.
- `--validated-on` gives each class a column per validation set: one validation
  by name, or `all` for the runs' own lists beside every validation.
- `--split-by` divides the validation named by a column of its images file, or
  without `--validated-on` the runs' own lists. It takes a column's name as the
  images file has it, such as `microscope_model`, `project_lamella_type` or
  `acceleration_voltage_kv`. A name the validation cannot be divided by is
  refused, with the ones it can.
- The best and the marked epoch are still the ones chosen on the run's own list.
- Each set whose size is known gets a column of how many images it holds,
  before the values.
- Images with no value in the `--split-by` column are the set `(not recorded)`.
  An image whose project holds several values there is in the set of each.

`compare-predictions` draws what the models of several runs predict on a few
validation images as one picture, with a row per image and a column per run.
Each run uses the epoch `choose-best` marked, else its own best.

- Images are drawn at random among those **none of the runs trained on**.
  `--include-trained` lifts that, and `--seed` makes the same draw again.
- `--microscope`, `--lamella-type` and `--voltage` narrow the draw to what the
  export recorded.
- `--with-label` adds a column for each image's own label.
- Predictions are made as `predict` makes them and resized as
  `convert-predictions` resizes them. They are held in memory only: nothing is
  written to the collection or recorded in its history.
- It needs the `ml` extra, not the `train` one.
- A run prepared before AM-seg kept what its validation images are of can be
  narrowed once its own validation images have been scored, and only drawn
  before.
- A run's lists name its images where they were when it was prepared. For runs
  trained on another machine, give the collection as it is on this one with
  `--start-path`. An image that is not found is left out and said, and with
  none found nothing is predicted.

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

- Run `init-history` once on a collection that already holds work, so a
  savepoint covers every project. `--as NAME` credits labels that arrived
  already annotated to whoever annotated them.
- `savepoint --close-gap` also keeps the labels only a model has written.
- `revert --to` restores each image's state with its label. `--with-metadata`
  also restores the project metadata edited in AM-seg since. Family
  corrections changed since are listed, not restored.
- `list-savepoints` counts what has changed since each savepoint.
- `archive` reports the label versions a project keeps, and `prune-archive`
  removes those no savepoint needs.

### Merging a copy back

```bash
am-seg merge /path/to/copy --into /path/to/collection
am-seg merge /path/to/copy --into /path/to/collection \
    --resolve AutoLamella-2024-07-05-12-53/03-secure-oyster=theirs --apply
```

`--resolve` decides conflicts: one by its path, or every one under a folder,
project or lamella.

- A family correction is named after its project, as `<project>:family:sem_x`,
  or `<project>:family:lam_003/sem_x` for a lamella's own entry.
- A label conflict waiting on a family decision appears only once that decision
  is made. Decide the family, run the merge again, then decide the labels.

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
| `am-seg predict` | Predict labels for the selected images with a trained model |
| `am-seg convert-predictions` | Convert predictions into labels for the selected images |
| `am-seg deselect` | Set label groups to the deselected state |
| `am-seg select` | Set label groups to the selected state |
| `am-seg correct` | Set label groups to the corrected state |
| `am-seg show-families` | Show each project's families of imaging conditions |
| `am-seg edit-families` | Move imaging conditions between families in AM_families.yaml |
| `am-seg find-specks` | Find labels with specks: components likely too small to be real |
| `am-seg compress-labels` | Compress label files to save disk space |

### Quantify it, and export

| Command | What it does |
|---|---|
| `am-seg stats` | Count images by one or two properties |
| `am-seg quantify-labels` | Measure what the labels contain, in physical units |
| `am-seg export-metadata-csv` | Write the acquisition and project metadata of the selected images |
| `am-seg export-projects-csv` | Write one CSV row per project, with its metadata and counts |
| `am-seg export-files` | Copy the selected images, labels and project files to a folder |
| `am-seg create-training-csv` | Export image and label paths to CSV files for ML model training |
| `am-seg check-training-csv` | Check that image/label paths recorded in a training CSV still exist |

### Train models

| Command | What it does |
|---|---|
| `am-seg train` | Train a new model, or refine an existing one, on an exported training set |
| `am-seg show-training` | Report the progress or result of a training run, and plot its curves |
| `am-seg stop-training` | Stop a training run that am-seg started on this computer |
| `am-seg validate` | Score a training run's checkpoints on another export, or on its own validation set per image |
| `am-seg compare-training` | Compare training runs in a table: one value per run and class |
| `am-seg compare-predictions` | Compare predictions of several runs on a few validation images |
| `am-seg choose-best` | Mark an epoch as the best of a training run, instead of the one the run chose |

### Record what changes, and read it back

| Command | What it does |
|---|---|
| `am-seg init-history` | Start recording the history of a collection, for the data already on disk |
| `am-seg savepoint` | Save the current state of the labels as a savepoint, so a round of work can be undone |
| `am-seg list-savepoints` | List the savepoints here, what each covers and what can still be restored |
| `am-seg history` | Print the recorded history of a project or one of its labels |

### Put work back, and reclaim the space

| Command | What it does |
|---|---|
| `am-seg merge` | Merge an exported copy back into the collection it came from |
| `am-seg revert` | Restore a selection of labels to their state before an operation |
| `am-seg archive` | Report the label versions a project has kept, and their disk usage |
| `am-seg prune-archive` | Remove kept versions that no savepoint needs |

### Set up this computer

| Command | What it does |
|---|---|
| `am-seg create-shortcut` | Add the window to this computer's applications |

<!-- end generated -->
