# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-28. Enough to stop an idea being lost,
and the questions to answer before building it — detail added before it is needed is detail that
has to be maintained and re-argued.

---

## 5. Matching summary plots to the frames they describe

**What is wrong:** no summary plot reaches any frame, in any project in the reference collection.
`_plots_by_image_number` keys a stage's plots by `extract_image_number(plot.name)`, and no plot
naming era carries the `img_NNN` that reads: `GLM_Plot_0.png` (2023-08), `Image_000.tif_00.png`
(2023-08 to 2023-11), `Image_000_window_DL_000_00.png` (2024-01 on). Every plot is dropped and
the GUI's plot pane is empty everywhere. The images themselves pair correctly; this is only the
plots.

**The question to answer first:** the mapping assumes *one* plot per acquisition and keeps the
first it sees, but the later eras write several — 356 plots for 23 acquisitions in one lamella.
So this is not a matter of adding patterns. Decide what a frame should show when an acquisition
has fifteen plots, and whether the extra ones are per-acquisition detail or belong to the stage
the way an alignment run's single figure does, before touching `frames.py`.

**Worth knowing:** the plot filenames name the *image* they describe rather than carrying a bare
number, so matching on the image's filename stem is likely to be simpler and to survive a new
era better than another number pattern. `extract_image_number` is the wrong place for it either
way: it answers what acquisition an *image* is, and three of its callers want exactly that.
