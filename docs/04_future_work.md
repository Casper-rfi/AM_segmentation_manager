# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-28. Enough to stop an idea being lost,
and the questions to answer before building it — detail added before it is needed is detail that
has to be maintained and re-argued.

---

## 5. Matching summary plots to the frames they describe

**What is wrong:** no summary plot reaches any frame, in any project.
`_plots_by_image_number` keys a stage's plots by `extract_image_number(plot.name)`, and no plot
naming era carries the `img_NNN` that reads: `GLM_Plot_0.png` (2023-08), `Image_000.tif_00.png`
(2023-08 to 2023-11), `Image_000_window_DL_000_00.png` (2024-01 on). Every plot is dropped and
the plot pane is empty everywhere. The images themselves pair correctly; this is only the plots.

**The question to answer first:** the mapping assumes *one* plot per acquisition and keeps the
first it sees, but the later eras write several — hundreds against a couple of dozen acquisitions
in one lamella. So this is not a matter of adding patterns. Decide what a frame should show when
an acquisition has fifteen plots, and whether the extra ones are per-acquisition detail or belong
to the stage the way an alignment run's single figure does, before touching `frames.py`.

**Worth knowing:** the plot filenames name the *image* they describe rather than carrying a bare
number, so matching on the image's filename stem is likely simpler and more durable than another
number pattern. `extract_image_number` is the wrong place for it either way: it answers what
acquisition an *image* is, and three of its callers want exactly that.

---

## 6. An "other" series, so what is not indexed can be looked at

**What is wrong:** every image the conventions do not reach is counted by `unindexed-images` and
is then invisible. A count says a couple of hundred images went unseen across the collection; it
does not let anyone open one and decide whether the rule or the data is wrong. Making them a
series turns a dead end into something the window can show.

**The set already exists.** `survey._unindexed_findings` computes exactly it: every file under a
project with an image extension that no image set claims, less `is_ignored_image_name` and less
anything inside a `labels`, `predictions`, `plots` or `state` directory. Reuse that predicate
rather than writing a second one — two definitions of "not indexed" drifting apart is the one way
this ends up worse than the count. Today it holds grid overviews and their tiles, `ml-<epoch>`
model snapshots, `cc_peaks/` cross-correlation frames, loose per-project overviews, hand-made
folders, and two lamella final images. Only that last group looks like data anyone would want
labelled, which is itself the finding the series exists to surface.

**The questions to answer first:**

*Does the series silence the finding?* If these become indexed, `unindexed-images` counts nothing
and the alarm that made them findable is gone. Decide whether it stays as-is, becomes a count of
the `other` series, or goes. Losing it would be the worst outcome of this work.

*Is `other` in `Series` at all?* Adding it to the `Literal` makes `--series other` valid
everywhere, including for the commands that predict, convert, export and change state — on images
with no label directory and often no readable beam. The safer shape is a record carrying
`labels_directory=None` and `predictions_directory=None`, so every write path already raises rather
than guessing; but that refuses at runtime where a vocabulary never offering it refuses earlier. It
must also stay out of `LABELS_EXPECTED`, or every one reports `no-labels-yet`.

*What is its modality, and what orders it?* Many have no beam suffix and no readable header —
`unresolved-beam` is exactly this population — and Curate filters by modality, so decide what the
beam control does with a beamless frame first. Loose frames sort by header acquisition time, which
these may not carry at all; path order is the honest fallback, and saying so beats a timeline
implying a sequence.

**Worth knowing:** an image can be missed for two unrelated reasons and this addresses one. These
match no naming rule; separately, a directory that *does* classify as an image set stops the stage
search descending, so one hand-made `SEM_fiducial_current/` at a session root hid 71 ordinary
images below it. An `other` series makes the first kind visible and leaves the second as hidden as
it is now.
