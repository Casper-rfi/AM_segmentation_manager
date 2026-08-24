# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-24.

A holding area for things deliberately not being designed yet. Each entry records
enough to stop the idea being lost, and the questions that would have to be answered
before it could be built — nothing more. Detail added here before it is needed becomes
detail that has to be maintained and re-argued.

---

## 4. Importing existing labels into the project hierarchy

**What you want:** the labels that live in folder structures you created by hand,
brought back into the project hierarchy where `layout.py` expects them. These will be
helper scripts rather than a core feature of the package, and not GUI supported.

Two things make this small *given* `layout.py`: the destination is unambiguous, since
`layout.resolve_labels_directory(...)` gives the right target for any image, so no
script reimplements the convention; and `inspect-layout` already reports images with no
label and labels matching no image, so an import can be checked rather than trusted.

**The shape it would take:** a dry-run-first command matching on filename within a
selected scope, refusing to overwrite, never deleting from the source, and reporting
unmatched and ambiguous cases before writing anything. Much better specified once there
is a real input to look at.

---

## 5. Which milling task an image belongs to

**What you want:** to narrow an export to *the reference images of the polishing task*, in a
collection where a lamella directory holds a mixture from every task. `stage-name` answers
"which directory", a different question once one task spans several stages (`Rough 1`, `Rough 2`).

**What the data holds**, over 6,429 loose images: 43.5% carry a task or stage in the filename
(`ref_<Task>_start_`, `ref_<Task>_final_`, `<Task>_pre-task_`, `ref_milling_<Stage>_finished_`,
`ref_<Stage>_initial_alignment_`, `<Task>-<Stage>_finished_`); 37.1% carry nothing (beam-shift
alignment, and the images inside an alignment run); the rest are numbered Adaptive Polishing
frames. 26 distinct subjects, spelt both `Rough 1` and `Rough-1`.

**Where the vocabulary comes from:** each lamella's entry in `experiment.yaml` — `task_history[]`
with each task's `name` and `task_type` in the 2026-07 era, `protocol` (task key to stage names,
so `Fine 1` belongs to `mill_polishing`) and `states` before it. Matching a filename against that
list is what makes arbitrary, user-set names safe, and what splits `Rough-Milling-Rough-2`.

**The shape it would take:** `ImageRecord.task`, resolved in the scanner from the directory where
the 2026-07 layout provides one (`Milling/<Task>/`), else from the filename matched against the
lamella's vocabulary, else nothing; then `task` as a breakdown and filter, like `series`. **Task
only** — stages roll up into their task.

**Already ruled out:** assigning images by the timestamps in `history`. Those windows are short
and sparse rather than tiling the session, and images named after a step fall outside it —
`ref_MillRough_start_eb.tif` was acquired inside the recorded `SetupLamella` window.
