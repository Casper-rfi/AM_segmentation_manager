# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-20.

A holding area for things deliberately not being designed yet. Each entry
records enough to stop the idea being lost, and the questions that would have to
be answered before it could be built — nothing more. Detail added here before it
is needed becomes detail that has to be maintained and re-argued.

---

## 4. Importing existing labels into the project hierarchy

**What you want:** the labels that live in folder structures you created by hand,
brought back into the project hierarchy where `layout.py` expects them. These will
be helper scripts, rather than a core feature of the package, and not GUI supported.

Two things make this small *given* `layout.py`: the destination is unambiguous,
since `layout.resolve_labels_directory(...)` gives the right target for any image, so
no script reimplements the convention; and `inspect-layout` already reports images
with no label and labels matching no image, so an import can be checked rather
than trusted.

**The shape it would take:** a dry-run-first command matching on filename within a
selected scope, refusing to overwrite, never deleting from the source, and
reporting unmatched and ambiguous cases before writing anything. Much better
specified once there is a real input to look at.


---

## 5. Provenance for the corrected claim

**What is missing:** `corrected` asserts that a person proofread a label. It is the
most load-bearing claim in the collection — a training export trusts it, and a
deposit publishes it — and nothing records who made it or when. `AM_state.yaml`
holds only the current state, `write_state` logs at DEBUG, and the window's
changes log dies with the session. So the claim cannot be audited, and a
disagreement about whether something was really proofread has no evidence to
settle it. This is the FAIR-R gap: the data is reusable, its curation history is
not.

**Where it could go.** Three options, in rising order of disruption:

* *A sibling `AM_state_log.yaml` per stage*, append-only. Keeps `AM_state.yaml`
  diffable and hand-editable — several decisions in `decisions.md` rest on that —
  and keeps the audit trail from being quietly editable alongside the thing it
  audits. Costs a second file per stage.
* *Fields on the state entry*, turning each value into a mapping. One file, but it
  ends the property that a state file is a flat path→state mapping, which is what
  makes a two-state-at-once entry unrepresentable. Would need a `version` bump.
* *Nothing on disk; a `--log-file` a user opts into.* Cheapest, and useless for a
  deposit, because the evidence does not travel with the data.

**Questions to answer first:**

1. **Identity.** Recording *who* is the point, but this package will be open
   source and an identity is personal data. `git config user.name`? An
   `AM_SEG_OPERATOR` variable? A per-collection name in `AM_metadata.yaml`? Or
   timestamps only, which is honest about being an audit of *when* and not *who*?
2. **Growth.** A bulk deselect across 21 projects writes an entry per group. Does
   the log get trimmed, rotated, or kept forever? Forever is defensible for an
   audit trail and indefensible if it outgrows the data.
3. **Scope.** Only state changes, or label saves too? The save is the actual
   proofreading; the state change is the claim about it. Recording only the
   latter records the assertion without the act.
4. **Retrospective entries.** Every label already on disk was proofread before
   any of this existed. Does the log start empty and honest, or does
   `migrate-state` write "unknown, before <date>" entries so the gap is explicit
   rather than silent?

**The cheaper half, worth doing first either way:** a training CSV is `image,label`
and nothing else, so a published model cannot be traced back to the curation that
produced its training set. Adding `state` and `label_state` columns — which the
metadata CSVs already carry — costs one line each and makes the export
self-describing. That is most of the reproducibility benefit for almost none of
the design argument above.
