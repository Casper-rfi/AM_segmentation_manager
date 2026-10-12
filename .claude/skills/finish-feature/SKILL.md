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

**First decide whether `docs/` gets one at all.** A new entry in `decisions.md` or
`display.md` must answer yes to one of these, each a question the code settles:

- **Does it constrain code outside the module it is about?** A rule a change elsewhere
  could break — the state key, the write path, what the layers may import.
- **Is it a trap no test or docstring beside the code can hold?** A Qt, platform or
  environment behaviour that bites without warning.
- **Is it a fact about the world rather than the code?** The reference collection, a
  quirk of the microscope's header, the torch builds.

If none holds — how one widget looks, why one control sits where it does — it goes in
that module's docstring, with the test named there, and `docs/` stays as it was. Before
adding an entry, look for one it extends: correcting an existing entry beats adding a
neighbour to it.

Put a fact in one place, chosen by the table in `CLAUDE.md` § Documentation.

An entry in `decisions.md` or `display.md` **names the test that guards it**. A symbol
only shows the name was not renamed; a passing test shows the behaviour still holds.
`tests/test_docs.py` caps the entries naming no test, and those naming nothing at all,
and refuses to let either rise.

## 5. Removing an entry

Only for one of the reasons listed on `MAX_DOC_LINES` in `tests/test_docs.py`, and say
which in the commit message. **Never to make room for something you are adding.** If
`docs/` is over the ceiling, first ask whether the new entry passes step 4; then run
`doc-audit`. If neither frees the room, stop and ask the scientist — that is a decision,
not a tidying job. **Never raise `MAX_DOC_LINES` in the commit that needs the room**: a
raise is its own commit, after an audit that found nothing to move.

## 6. Run the suite and the preflight

```bash
uv run python scripts/preflight.py --full
```

From the repository root. The script lists what a push is checked for — the lint, the
type check, the suite with no extra installed, and with `--full` the whole suite — and is
the one place that does; `.githooks/pre-push` runs it without `--full` on every push.

Every check must pass. A test module that imports numpy, Qt or torch at module level
fails the suite with no extra installed: skip it there with `pytest.importorskip`.

`mypy` must report nothing, and it checks every module. Never add a
`[[tool.mypy.overrides]]` entry to `pyproject.toml` to make an error go away.

`test_docs.py` and `test_cli_help.py` catch a renamed test, an unresolvable symbol, a
broken cross-reference and an uncatalogued command. They do not catch an entry that
describes behaviour that changed while its names stayed put. That is what steps 1-3 are
for, and no test replaces them.

## 7. Report what you found

List every document you corrected and which claim was wrong. List separately anything
you found wrong and did **not** change, and why — a fact you could not place, or a
removal that needs the scientist's decision. Saying "documentation checked" without
naming what changed is not evidence that it was.
