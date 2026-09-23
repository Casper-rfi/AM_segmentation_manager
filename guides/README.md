# User guide

How to use AM segmentation manager: the window, which is where the work is
done, and the `am-seg` command line beside it. Installing it is in the
[README](../README.md#installation).

The work runs in four phases, and the window has a tab for each:

```mermaid
flowchart LR
    A["<b>Add data</b><br/>name the projects,<br/>describe them"] --> B["<b>Curate</b><br/>choose what to label,<br/>predict"]
    B --> C["<b>Label</b><br/>proofread,<br/>mark corrected"]
    C --> D["<b>Export</b><br/>quantify, deposit,<br/>train a model"]
    D -. a better model .-> B
```

## Chapters

1. [Concepts](01_concepts.md): projects, series, label groups and their three
   states, and every file am-seg writes.
2. [Add data](02_add_data.md): opening a folder, naming its projects, and
   describing them.
3. [Curate](03_curate.md): looking through the frames, changing their state,
   predicting and converting labels.
4. [Label](04_label.md): proofreading at full resolution, saving, specks and
   earlier versions.
5. [Export](05_export.md): choosing a subset, counting and measuring it, and
   writing training sets, metadata and copies.
6. [The command line](08_command_line.md): every command, and how to point one
   at part of a collection.
7. [Keys and gestures](09_keys.md): everything F1 lists, for every tab.

The chapter on history is being written. Until they are, the [README](../README.md) describes each tab.
