# Roadmap

Where the work stands and what to pick up next — the entry point for starting fresh. `README.md`
catalogues the commands; [`decisions.md`](decisions.md) says why the software is the way it is and
[`display.md`](display.md) why the window is; `CLAUDE.md` holds the working agreement. Nothing
those say is restated here.

**One line per thing.** A capability that lands adds a line and a branch that merges collapses to
one; prose has to be rewritten whole every time either happens.

---

## What the software does today

Every main function `CLAUDE.md` names is built, across the three series of images Autolamella writes
and every layout era the collection holds. Curate, Label, Quantify and Export are four tabs, each
with a controller holding no Qt, and the command line does the same work for scripting and testing.
Predictions need the second environment `scripts/setup_ml.sh` builds.

**Label history** — on branch `label-history`, not merged. It adds:

- An append-only `AM_history.jsonl` per project: what changed, who changed it, and when.
- **Recording is always on.** The first change to a project starts its log and records what the
  whole project held; looking writes nothing.
- **No name is recorded unless somebody states one.** Set on F2, shown beside the mark.
- `AM_archive/` keeps every replaced label, and `prune-archive` thins it by collapsing what lies
  between savepoints.
- `savepoint` names the state of every label as a round of model improvement; `revert` puts labels
  back to a savepoint, a conversion or a merge.
- An export carries the history and an `AM_export.yaml` naming the fork point; `merge` brings a copy
  back three-way against the history the two logs share, refusing the whole merge while a conflict
  stands.
- All of it is in the window, placed by the rule in [`display.md`](display.md) § Controls, with
  merge in Export beside the copy it brings home and *Compare versions* in the proofreading view.

## What to pick up next

- **Finish the history work**, in order. Run `init-history` once on the real collection, so a
  savepoint taken there covers everything rather than only what has been touched since — around
  950 KB across 3,236 labels, two minutes, adding two files per project and modifying nothing, and
  the one act never yet performed on it. Then close **one whole round** with a real training run in
  the middle: the parts have each been rehearsed, the loop never once end to end. Then merge.
- **A macOS `.icns` icon**, the current one being a full-bleed PNG where macOS wants a squircle
  inset in its canvas. Artwork rather than code, and Casper's to draw.
- **An "other" series** making unindexed images visible instead of merely counted.
  [`04_future_work.md`](04_future_work.md) §6.
- **Test fixture consolidation** — migrate test files onto the shared `conftest.py` builders when
  editing them for other reasons, never as a sweep. `test_selection.py` still carries a hand-built
  tree.
- **Before publishing**, several packaging gaps are open by decision rather than oversight and all
  become necessary at once: no `LICENSE` (`pyproject.toml` deliberately declares none and carries a
  `"Private :: Do Not Upload"` classifier), no CI, no `CHANGELOG.md`, no `[project.urls]`, and the
  version pinned at `0.0.0`. `mypy` is not configured and `ruff` does not check types, so the
  annotations are unverified. The suite has not been run under a non-English locale — do that
  before any release, for the reasons in [`display.md`](display.md) § Testing the window. And
  [`decisions.md`](decisions.md) § Environment names the collection twice by absolute path, which
  is what makes never writing to it a rule about a real place rather than a principle: decide then
  whether a reader outside this machine needs the paths or only the rule.
