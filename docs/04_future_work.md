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
- **Generation 2**, when Adaptive Milling has one. First: whether it arrives as another entry in
  `GENERATIONS`, its preprocessing added beside `IMPLEMENTED_PREPROCESSING` and held to a recapture
  of `adaptive_milling_gen1.npz`'s kind, or as a dependency on `adaptive-milling`: 27 packages beside
  `ml`, torch below 2.9 on Windows, Apache-2.0 with the Commons Clause, which is not open source,
  and an API that renamed its generations once.
- **Splitting the five classes that outgrew a screen** — `CurationWindow` (96 methods, 1789
  lines), `CurationController` (95), `LabelPanel` (72), `ExportController` (49) and `ExportPanel`
  (32). None is public API — that is six names and the `am-seg` entry point — so this is
  maintainability, not a break a user would see, and it is deliberately not done before a release.
  First: whether a split follows the tabs, the panels or the operations; and whether it removes
  state rather than only moving methods behind another reference, which costs readability and buys
  nothing.
- **Family corrections for one stage**, keyed by its path. First: whether a lamella's steps ever need them.
- **Several families of a beam in real data**: a model per family (a 5kV model for 5kV images), and a scene
  holding two families, each exported and measured as its own label group today. First: whether such data
  arrives, and whether a model is chosen per family, per series and beam as now, or both.
