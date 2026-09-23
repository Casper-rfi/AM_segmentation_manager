# The command line

The window does the everyday work. The command line does the same work for
scripting, and holds the reports on a whole collection.

```bash
am-seg --help                 # every command, under its phase of the work
am-seg <command> --help       # one command: its options, and how to read its output
```

From a checkout, put `uv run` in front: `uv run am-seg --help`.

## Pointing a command at part of a collection

Most commands take a folder and the same options to narrow it down:

| Option | Narrows to |
|---|---|
| `--project`, `--lamella`, `--stage`, `--image` | one of each, by name or number; `all` for every one |
| `--series` | `adaptive`, `reference`, `alignment`, or several separated by commas |
| `--task` | one milling task, by the name the project's protocol gave it |
| `--state` | `selected`, `deselected`, `corrected`, or `all` |
| `--modality` | `SEM`, `FIB`, or `all` |

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
