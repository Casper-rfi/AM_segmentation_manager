# Future work

Status: **sketches, not designs.** Enough to stop an idea being lost, and the questions to answer
before building it — detail added before it is needed is detail that has to be maintained and
re-argued. What is scheduled is in [`00_roadmap.md`](00_roadmap.md); this holds what is not.

- **Labelling grid overviews**, wanted by no one yet. First: which classes, `SegmentationConfig`
  holding one set for every series; where the labels live; the gamma copy sharing its stitch's.
- **A colourblind-safe class palette**, suggested in a review of the window and wanted by no one
  yet. First: whether it is a preference, which reaches no write path, or a second segmentation
  config; and whether a thin class such as a crack needs an outline as well as a fill.
- **Loading follow-ups not yet agreed**: rescan only the projects an operation touched; drop the
  parsed `positions` from `Project.metadata`, most of a scan's peak memory; memoise task
  resolution; let the command line reuse the headers a scan keeps. First: whether each saves enough
  on the reference collection to be worth what it complicates.
- **Filtering the Export tab from its chart** — filter chips, or clicking a heatmap cell —
  suggested in a review of the window. First: how a filter set there shows in the filter controls,
  so the two never disagree about what is selected.
- **Running the suite in parallel** with `pytest-xdist`, a new dependency. First: whether the
  process-wide state [`display.md`](display.md) § Testing the window names — the locale a
  `QApplication` sets, the error reporter's excepthook — stays confined to one worker.
- **Training through AM-model-training** (Apache-2.0, RFI), which trained these models and reads
  the `image,label` CSVs `create-training-csv` writes; its preprocessing is this package's. Not its
  `am-infer`, which predicts in training mode, never calling `model.eval()`. First: MONAI, mlflow
  and its torch index are not ours, so whether it is a dependency or an environment a child
  process runs, as predicting already is.
- **Generation 2**, when Adaptive Milling has one. First: whether it arrives as another entry in
  `GENERATIONS` held to a recapture of `adaptive_milling_gen1.npz`'s kind, or as a dependency on
  `adaptive-milling`, 27 packages beside `ml` and an API that renamed its generations once.
- **Splitting the five classes that outgrew a screen** — `CurationWindow` (89 methods, 1657
  lines), `CurationController` (82), `LabelPanel` (66), `ExportController` (48) and `ExportPanel`
  (31). None is public API — that is six names and the `am-seg` entry point — so this is
  maintainability, not a break a user would see, and it is deliberately not done before a release.
  First: whether a split follows the tabs, the panels or the operations; and whether it removes
  state rather than only moving methods behind another reference, which costs readability and buys
  nothing.
