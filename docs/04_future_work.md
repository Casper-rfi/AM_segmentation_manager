# Future work

Status: **sketches, not designs.** Last reviewed 2026-08-25.

A holding area for things deliberately not being designed yet: enough to stop the idea
being lost, and the questions to answer before building it. Detail added before it is
needed is detail that has to be maintained and re-argued.

---

## 4. Importing existing labels into the project hierarchy

**What you want:** the labels living in folder structures you created by hand, brought
back into the project hierarchy where `layout.py` expects them. Helper scripts rather
than a core feature, and not GUI supported.

Two things make this small *given* `layout.py`: `resolve_labels_directory(...)` gives
the right target for any image, so no script reimplements the convention; and
`inspect-layout` already reports images with no label and labels matching no image, so
an import can be checked rather than trusted.

**The shape it would take:** a dry-run-first command matching on filename within a
selected scope, refusing to overwrite, never deleting from the source, and reporting
unmatched and ambiguous cases before writing anything. Much better specified once there
is a real input to look at.
