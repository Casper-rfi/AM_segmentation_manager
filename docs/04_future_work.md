# Future work

Status: **sketches, not designs.** Enough to stop an idea being lost, and the questions to answer
before building it — detail added before it is needed is detail that has to be maintained and
re-argued. What is scheduled is in [`00_roadmap.md`](00_roadmap.md); this holds what is not.

- **Labelling grid overviews**, wanted by no one yet. First: which classes, `SegmentationConfig`
  holding one set for every series; where the labels live; the gamma copy sharing its stitch's.
- **A colourblind-safe class palette**, suggested in a review of the window and wanted by no one
  yet. First: whether it is a preference, which reaches no write path, or a second segmentation
  config; and whether a thin class such as a crack needs an outline as well as a fill.
- **Removable filter chips above the breakdown.** First: how the project filter's "every project
  ticked is no filter" shows as a chip, and whether removing one unticks a row or takes it away.
- **Clicking a heatmap cell to filter by it.** First: what a click means on an axis nothing can be
  filtered on — a measurement, or one of `UNFILTERABLE` — and whether it adds a row or replaces one.
- **A dark theme**, the image panes being black already. First: which colours are data rather than
  chrome — the workflow states, the class colours, `ZERO_COLOUR` — and must hold on both grounds.
