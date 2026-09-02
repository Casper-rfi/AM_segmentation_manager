# Future work

Status: **sketches, not designs.** Last reviewed 2026-09-02. Enough to stop an idea being lost, and
the questions to answer before building it — detail added before it is needed is detail that has to
be maintained and re-argued.

---

## 5. Matching summary plots to the frames they describe

**What is wrong:** no summary plot reaches any frame, in any project. `_plots_by_image_number` keys
a stage's plots by `extract_image_number(plot.name)`, and no plot naming era carries the `img_NNN`
that reads: `GLM_Plot_0.png` (2023-08), `Image_000.tif_00.png` (2023-08 to 2023-11),
`Image_000_window_DL_000_00.png` (2024-01 on). Every plot is dropped and the plot pane is empty
everywhere. The images themselves pair correctly; this is only the plots.

**The question to answer first:** the mapping assumes *one* plot per acquisition and keeps the first
it sees, but the later eras write several — hundreds against a couple of dozen acquisitions in one
lamella. So decide what a frame should show when an acquisition has fifteen plots, and whether the
extra ones are per-acquisition detail or belong to the stage the way an alignment run's single
figure does, before touching `frames.py`.

**Worth knowing:** the plot filenames name the *image* they describe rather than carrying a bare
number, so matching on the image's filename stem is likely simpler and more durable than another
number pattern. `extract_image_number` is the wrong place for it either way: it answers what
acquisition an *image* is, and three of its callers want exactly that.

## 6. An "other" series, so what is not indexed can be looked at

**What is wrong:** every image the conventions do not reach is counted by `unindexed-images` and is
then invisible. A count does not let anyone open one and decide whether the rule or the data is
wrong. `survey._unindexed_findings` already computes exactly that set — reuse the predicate rather
than writing a second one, two definitions of "not indexed" drifting apart being the one way this
ends up worse than the count. Most of what it holds is machinery nobody would label: grid
overviews, model snapshots, cross-correlation frames. That only a handful look like data anyone
wants labelled is itself the finding the series exists to surface.

**The questions to answer first.** *Does the series silence the finding?* If these become indexed,
`unindexed-images` counts nothing and the alarm that made them findable is gone; decide whether it
stays, becomes a count of the `other` series, or goes. *Is `other` in `Series` at all?* Adding it to
the `Literal` makes `--series other` valid everywhere, including for the commands that predict,
convert, export and change state — on images with no label directory and often no readable beam. A
record carrying `labels_directory=None` makes every write path raise rather than guess, but that
refuses at runtime where a vocabulary never offering it refuses earlier; it must also stay out of
`LABELS_EXPECTED`, or every one reports `no-labels-yet`. *What is its modality, and what orders it?*
Many have no beam suffix and no readable header — `unresolved-beam` is this population — and Curate
filters by modality, so decide what the beam control does with a beamless frame first. Path order is
the honest fallback where no header timestamp exists, and saying so beats a timeline implying a
sequence.

**Worth knowing:** an image can be missed for two unrelated reasons and this addresses one. These
match no naming rule; separately, a directory that *does* classify as an image set stops the stage
search descending, hiding everything below it.
