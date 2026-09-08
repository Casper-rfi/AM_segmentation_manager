---
name: doc-audit
description: Check every entry in docs/ against the code it describes, correct what is no longer true, and remove what has finished. Use after merging a branch, when tests/test_docs.py refuses a change for exceeding MAX_DOC_LINES, before a release, or whenever the scientist asks for a documentation clean-up. Not the per-feature pass - that is finish-feature.
---

# Auditing the documents

`finish-feature` checks the documents that mention one change. This checks all of
them, against the code as it is now, and is the only pass that removes anything.

**Two passes. Do not mix them.** Correcting what is false and shortening what is
here are different jobs with different evidence, and running them together lets
the length decide the corrections -- which is how a true entry gets deleted to
make room for a new one.

## Pass 1 -- what is false

For each entry ask **"is this still true?"**, never "is this still relevant". The
second has no factual answer, so it is always answered yes, and a document full
of things nobody would defend is the result.

```bash
grep -n "^## " docs/decisions.md      # sections, with line numbers
sed -n '383,491p' docs/decisions.md   # read one
```

Open what the entry points at. An entry naming a test is checked by running it;
one naming a function is checked by reading it; one naming nothing has had
nothing holding it honest, so check it hardest.

**Status lines are the most likely to be wrong.** They describe a moment, and
moments pass: a branch that has merged, work described as next that is done, a
count that has moved. `00_roadmap.md` is mostly status.

Correct what is wrong. **Remove nothing for length in this pass.**

## Pass 2 -- what has finished

Read the reasons on `MAX_DOC_LINES` in `tests/test_docs.py` and apply exactly
those. Nothing goes to make room, and nothing goes because it reads as less
important than its neighbours.

The reliable finds are structural rather than editorial:

- **A merged branch collapses.** A block describing work in progress becomes one
  line under what the software does. This is usually the largest single win.
- **A plan whose work has landed** goes entirely; it is not annotated as done.
- **A fact stated twice** stays where `CLAUDE.md` § Documentation places it and
  goes from the other. Behaviour of one module belongs in its docstring -- move
  it there, then remove it here.

**Do not compress as a way of shortening.** Rewording an entry shorter keeps
every claim it made and makes each harder to check, which is the opposite of the
job. Length comes out of the reasons above or it does not come out.

## Then

```bash
python -m pytest -q
```

In `am_segmentation_manager` or `am_segmentation_manager_ml`; the run names the
interpreter if it refuses. `test_docs.py` will say whether the ratchets moved --
correcting an entry to name the test that guards it improves both, and tightening
them to match is what stops the improvement being given back later.

## Report

Every entry removed and under which reason. Every correction, with what was wrong.
The line count before and after.

If the count is still high and nothing else legitimately goes, **say so and
stop**. That is an answer. Manufacturing the difference by compressing is how the
documents become dense, unverifiable and still too long.
