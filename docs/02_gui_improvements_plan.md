# Curation window improvements

Status: **proposed, not started.** Written 2026-08-14.

Four requested changes. Three are small and independent; the fourth
(status indicators) has a real design decision in it and is written up at
length. None depends on `03_metadata_plan.md`, and all of them use data the
controller already holds.

Build them in the order below, one commit each, suite green between.

---

## 1. Bottom margin below the timeline

**Problem.** The FIB label line is the bottom-most 3 px of the frame timeline
and there is almost nothing between it and the window edge, so it is hard to
see.

**Cause.** `gui/widgets/main_window.py:86`:

```python
centre_layout.setContentsMargins(0, 0, 0, 2)
```

Two pixels of bottom margin. The timeline's own height calculation
(`frame_strip.py:111`) ends exactly on the FIB label line — the SEM band is
label-line-then-row and the FIB band is row-then-label-line, so the FIB line
sits flush against the widget's bottom edge by design.

**Change.** Raise the bottom margin to 8 px.

```python
centre_layout.setContentsMargins(0, 0, 0, 8)
```

**Why the margin and not the widget height.** Adding padding inside
`FrameTimeline` would mean its `_band_height()` arithmetic no longer describes
its own geometry, and that arithmetic is what keeps the two rows aligned. The
gap wanted here is between the timeline and the window, which is the layout's
job.

**Test.** None worth writing — a margin constant has no behaviour. Check by eye.

---

## 2. Right-click → Copy path

**Problem.** No way to get a path out of the window and into a terminal, a file
manager, or a note.

**Scope.** Three places:

| Where | Path to copy |
|---|---|
| Projects list (`project_panel.py`) | `project.path` |
| Lamellae list (`project_panel.py`) | `lamella.path` |
| Image panel (`image_panel.py`) | the displayed image's `image_path` |

**Design.** The panels are Qt-only and hold no model objects — they receive
lists of display strings (`show_projects(labels, current)`). That separation is
worth keeping, so the panel must not learn about `Project`.

Structure it as: the panel emits a request, the window answers it from the
controller.

1. `ProjectPanel` gains two signals, `project_path_requested(int)` and
   `lamella_path_requested(int)`, emitted from a `QListWidget` context menu with
   the row index under the cursor. Set `setContextMenuPolicy(Qt.CustomContextMenu)`
   and connect `customContextMenuRequested`.
2. `CurationController` gains `project_path(position)` and
   `lamella_path(position)`, returning `Path | None`.
3. `MainWindow` connects the two, and puts the result on the clipboard with
   `QApplication.clipboard().setText(str(path))`.
4. The image panel does the same for the current image, via
   `controller.current_image()`.

**Suggested extras, cheap once the menu exists.** A second action, *Reveal in
Finder*, using `QDesktopServices.openUrl(QUrl.fromLocalFile(...))` on the
parent directory. This is the action you will actually want most of the time,
and it costs three lines on top of the menu you are already building. Offer it;
do not build a third mechanism for it later.

**Which path.** Copy the **absolute** path. The relative path is the project's
identity within the index, but a path pasted into a terminal needs to be
absolute to be useful, and that is the entire point of the feature.

**Tests.**
- `controller.project_path(n)` / `lamella_path(n)` return the right path, and
  `None` for an out-of-range position or when nothing is loaded. Pure
  controller, no Qt — put these in `test_gui_controller.py`.
- One widget test that the context-menu signal carries the row index.
  Clipboard interaction itself is not worth testing.

---

## 3. The "Scan root" label

**Problem, as you put it:** the text seems dissociated from the function, and
the Scan button already says "Scan".

**Agreed, and the diagnosis is slightly broader.** `project_panel.py:47–75`
currently builds a `QGroupBox` titled "Scan root" containing (a) a path field
with placeholder text "Folder to scan for Autolamella projects", (b) a
*Browse…* button, and (c) a full-width *Scan* button on its own row. That is a
box title, a placeholder and a button all saying the same thing three times,
across two rows.

**Change.** Remove the group box. One row, three widgets:

```
[ /path/to/collection            ] [ Browse… ] [ Scan ]
```

* The `QLineEdit` keeps its placeholder ("Folder to scan for Autolamella
  projects") and its tooltip, which is where the explanation belongs — visible
  when the field is empty and needed, invisible once a path is in it.
* *Scan* moves onto the row and stops being full width. It is not the primary
  action of the window; it currently looks like it is.
* Keep both buttons' tooltips, especially Scan's ("Re-read the folder from
  disk. Use this after changing files outside the window."), which says the
  non-obvious thing that neither the label nor the button text does.

**Why remove rather than move it.** A label that restates the placeholder and
the button carries no information. Moving it closer would make the redundancy
tidier, not smaller.

**Watch for.** The `Projects` and `Lamellae` boxes below are also `QGroupBox`es
and they earn their titles — the counts go in them (`Projects (12)`). Removing
one group box from a column of three may look unbalanced; check it on screen
before committing, and if it does, the fallback is to keep the box but drop the
separate Scan row rather than to restore the label.

**Test.** None. Layout only.

---

## 4. Status indicators for lamellae and projects

The substantial one. You asked whether it should interact with the per-image
indicators, or be purely visual.

### 4.1 Recommendation: derive it, do not store it

**A lamella's status should be a pure function of the states of its images,
computed on demand. Nothing new is written to disk.**

Consequences, all good:

* It cannot go stale, be forgotten, or disagree with the timeline, because it
  *is* the timeline's data summarised.
* It needs no new file format, no migration, and no interaction rules with the
  per-image indicators — the question "how do they interact?" dissolves,
  because one is a summary of the other.
* It follows `CLAUDE.md`'s rule that state is derived from the filesystem
  rather than kept in a separate store.
* The data is already in the controller: `modality_states()` returns exactly
  the per-frame, per-modality states this needs.

### 4.2 The one thing derivation cannot do — and it may be what you want

Derivation answers *"is this lamella finished?"*. It cannot answer *"have I
looked at this lamella?"*, and reading your note again ("used for a person to
keep track of which lamellae and projects they have already gone through"),
that second question may be the real one.

They differ in a case that will occur: you open a lamella, look through every
frame, and conclude that all of them should stay selected and be proofread
later. You have been through it. Derivation will show it as untouched, because
nothing moved.

Storing "visited" means persisting a marker — a file per lamella, since
`CLAUDE.md` forbids a hidden database — which is a real feature with real
questions (what invalidates it when files change underneath?).

**Recommendation: build the derived indicator first.** It is cheap, it is
certainly useful, and it is quite likely sufficient — for most lamellae "gone
through" and "everything is deselected or corrected" coincide. Use it for a
few weeks. If the gap above turns out to bite in practice, add the visited
marker then, with the benefit of knowing exactly what it needs to do. Building
both now means designing the harder one against a guess.

### 4.3 The status categories

Per modality, as you asked — SEM and FIB reported separately, because a
lamella whose SEM work is complete and whose FIB work has not started is a
normal and important situation that one combined mark cannot express.

| Status | Definition | Rationale |
|---|---|---|
| **empty** | no images of this modality | Distinct from "not started". Many older datasets have no FIB at all, and showing them as 0 % complete forever would be wrong. You called this out. |
| **not started** | every image is `selected` | Nothing has been decided yet. |
| **in progress** | some images `deselected`/`corrected`, some `selected` | |
| **done** | every image is `deselected` or `corrected` | Your definition: nothing is left awaiting a decision. |
| **needs repair** | any frame is in `mismatched_frames()` | Overrides the above. A lamella whose images disagree with their shared label cannot be called done, because the status shown for it is not the whole truth until `repair-state` runs. |

The last row matters and is easy to miss: the controller already computes
mismatches for the amber wash, and a "done" badge on a lamella that is silently
inconsistent would be actively misleading. Reuse `mismatched_frames()`.

A project's status is the **weakest** of its lamellae's, per modality — done
only when every lamella is done, needs-repair if any needs repair. Do not
average: a project shown as 90 % done is not information you can act on,
whereas "one lamella left" is.

### 4.4 Where the code goes

`gui/controller.py`, next to `modality_states()` and `mismatched_frames()`,
which it builds on. A frozen dataclass:

```python
@dataclass(frozen=True)
class CurationStatus:
    """How far curation has progressed, per modality."""
    sem: str   # "empty" | "not started" | "in progress" | "done" | "needs repair"
    fib: str
```

with `lamella_status(lamella) -> CurationStatus` and
`project_status(project) -> CurationStatus`.

**Cost.** `project_status` needs every lamella's frames built. Frame building
is already done per lamella on selection; doing it for all lamellae of a
project on scan is more work. Measure it on the real collection before
committing to computing project status eagerly — if it is slow, compute lamella
status eagerly (cheap, and it is the list you look at most) and project status
lazily or on demand. Do not add a cache before measuring; `label_states()`
already caches by path and may absorb most of the cost.

### 4.5 How to draw it

Reuse `frame_strip.STATE_COLOURS` — the same grey/blue/teal progress ramp
already used by the timeline and the state-panel swatches. A third colour
vocabulary for the same concept would be the wrong outcome.

Suggested: a small two-part swatch on each list row (left half SEM, right half
FIB), painted via a `QListWidgetItem` icon, exactly as `state_panel.py:252`
already builds swatch pixmaps from `STATE_COLOURS`. Amber (`frame_strip.MISMATCH`)
for needs-repair. Nothing drawn for empty — an absent modality should read as
absent, not as a state.

Set the item tooltip to the long form ("SEM: done · FIB: in progress"), so the
swatch does not have to carry the whole meaning on its own.

**Do not** add a legend to the panel. The state panel already has one for the
same colours, and a second would be clutter — this is exactly the feature-creep
boundary you flagged.

### 4.6 Tests

The status logic is pure and Qt-free, so it tests well:

* Each of the five categories, per modality, in `test_gui_controller.py`.
* A lamella with no FIB images reports `fib="empty"`, not `"done"` — the
  regression most likely to slip through, since "no images left awaiting a
  decision" is vacuously true.
* Needs-repair overrides done.
* Project status is the weakest of its lamellae's, not an average.
* One widget test that the swatch appears on list rows.

---

## 5. Not in this plan

Deliberately excluded, so the scope does not grow:

* The CSV-export and metadata tabs. Those are `03_metadata_plan.md` §7, and
  they reverse a recorded scope decision that should be handled there.
* A "visited" marker (§4.2) — revisit after using the derived indicator.
* Any label editing. Still out of scope, still the external software's job.
