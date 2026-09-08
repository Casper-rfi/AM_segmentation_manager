---
name: finish-feature
description: The documentation pass that closes a piece of work on this package - find every document and docstring that mentions what changed, check each claim against the code rather than against memory, and correct what is no longer true. Use when a feature, fix or refactor is working and about to be reported done or committed.
---

# Finishing a feature

Documentation goes stale because nobody re-reads it, not because anyone decides to leave
it wrong. This is the re-reading, and it happens *before* the work is reported done.

**Ask "is this still true?", never "is this still needed?"** The first has an answer the
code settles. The second is an opinion, and the answer to it is nearly always "keep it",
which is how a document ends up describing software that no longer exists.

## 1. Find what mentions what you changed

```bash
git status --short && git diff --name-only
```

For each module you touched, search every place that could describe it — its own name,
the classes and functions you changed, and any command whose behaviour moved:

```bash
grep -rn "scanner\|ProjectIndex\|convert-predictions" docs/ README.md CLAUDE.md
```

Search for the *concepts* too, not only the identifiers. A rule about label groups may
never name the function you rewrote.

## 2. Read by section, not whole

```bash
grep -n "^## " docs/decisions.md     # sections with line numbers
sed -n '383,491p' docs/decisions.md  # read the one you need
```

## 3. Check each claim against the code

Open what the entry points at. An entry naming a test is checked by that test still
existing and still passing; an entry naming a function is checked by reading it. An
entry that describes behaviour without naming anything is the one to be most suspicious
of — nothing has been holding it honest.

Four things go stale most often and are easy to miss:

- **Status lines.** `00_roadmap.md` says what is built and what is on a branch. A
  capability that landed adds a line; a branch that merged collapses to one.
- **The README command table**, if you added, renamed or removed a command.
- **The module docstring of every file you changed** — it states what must stay true
  across the module, and you may have just changed that.
- **A plan whose work has landed.** Delete it; do not annotate it as done.

## 4. Writing a new entry

Put the fact in one place, chosen by the table in `CLAUDE.md` § Documentation. Behaviour
of one module goes in its docstring, not in `docs/`.

An entry in `decisions.md` or `display.md` **names the test that guards it**. A symbol
only shows the name was not renamed; a passing test shows the behaviour still holds.
`tests/test_docs.py` counts both and refuses to let either get worse.

## 5. Removing an entry

Only for one of the reasons listed on `MAX_DOC_LINES` in `tests/test_docs.py`, and say
which in the commit message. **Never to make room for something you are adding.** If
`docs/` is over the ceiling and none of those reasons applies, stop and ask the
scientist — that is a decision, not a tidying job.

## 6. Run the suite

```bash
python -m pytest -q
```

In `am_segmentation_manager` or `am_segmentation_manager_ml`. If the run refuses, it
names the interpreter to use instead.

`test_docs.py` and `test_cli_help.py` catch a renamed test, an unresolvable symbol, a
broken cross-reference and an uncatalogued command. They do not catch an entry that
describes behaviour that changed while its names stayed put. That is what steps 1-3 are
for, and no test replaces them.

## 7. Report what you found

List every document you corrected and which claim was wrong. List separately anything
you found wrong and did **not** change, and why — a fact you could not place, or a
removal that needs the scientist's decision. Saying "documentation checked" without
naming what changed is not evidence that it was.
