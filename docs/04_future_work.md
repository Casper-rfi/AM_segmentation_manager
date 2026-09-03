# Future work

Status: **sketches, not designs.** Enough to stop an idea being lost, and the questions to answer
before building it — detail added before it is needed is detail that has to be maintained and
re-argued.

---

## 5. Summary plots for the eras whose filenames carry no acquisition number

**What is wrong, measured:** Autolamella writes a PNG per acquisition into a stage's `plots/`, and
the image panel has a Plots view to show the one belonging to the frame on screen.
`_plots_by_image_number` keys them by `extract_image_number(plot.name)`, which reads the current
era's `<lamella>_AP_img_000_plot.png` — **5,701 plots across the collection match, and every one of
the 252 adaptive frames in `20250625_AM_further_testing` has its plot**. The 11,483 that do not are
the older `Image_000_window_DL_000_00.png` shape, which carries no number this reads.

**Answer first:** those older names are not one plot per acquisition — `_00`, `_01`, `_02` are
several windows of one image, hundreds against a couple of dozen acquisitions in a lamella. Decide
what a frame shows when its acquisition has fifteen, and whether the extras are per-acquisition
detail or belong to the stage the way an alignment run's single figure does. Until that is
answered there is nothing to implement.

**Worth knowing:** those filenames name the *image* they describe, so matching on its stem is
likely simpler than another number pattern, and `extract_image_number` is the wrong place either
way — it answers what acquisition an *image* is, which three of its callers want.

## 6. An "other" series, so what is not indexed can be looked at

**What is wrong:** every image the conventions do not reach is counted by `unindexed-images` and is
then invisible — a count does not let anyone open one and decide whether the rule or the data is
wrong. `survey._unindexed_findings` already computes that set; reuse the predicate, two definitions
of "not indexed" drifting apart being the one way this ends up worse than the count. Most of it is
machinery nobody would label, and that only a handful look like data anyone wants labelled is itself
the finding the series exists to surface.

**Answer first.** *Does the series silence the finding?* Indexed, `unindexed-images` counts nothing
and the alarm that made them findable is gone; decide whether it stays, becomes a count of the
series, or goes. *Is `other` in `Series` at all?* Adding it to the `Literal` makes `--series other`
valid everywhere, including for the commands that predict, convert, export and change state — on
images with no label directory and often no readable beam. `labels_directory=None` makes every write
path raise rather than guess, but that refuses at runtime where a vocabulary never offering it
refuses earlier; it must also stay out of `LABELS_EXPECTED`, or every one reports `no-labels-yet`.
*What is its modality, and what orders it?* Many have no beam suffix and no readable header
(`unresolved-beam` is this population) and Curate filters by modality, so decide what the beam
control does with a beamless frame. Path order is the honest fallback where no header timestamp
exists, and saying so beats a timeline implying a sequence.

**Worth knowing:** an image can be missed for two unrelated reasons and this addresses one. These
match no naming rule; separately, a directory that *does* classify as an image set stops the stage
search descending, hiding everything below it.
