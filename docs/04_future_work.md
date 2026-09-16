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
