---
name: ui-text
description: The style for every piece of text a user reads - tooltips, captions, panel text, dialogs, operation reports, status lines, the F1 shortcut list, the Settings dialog and the command-line --help. Use before writing or changing any string the window shows or a command prints, and when finish-feature finds that a change added or reworded one. Not for docstrings, comments or docs/, which keep the code's own voice.
---

# Writing text a user reads

**The window does not speak in the code's voice.** Docstrings and `docs/` explain why the
software is the way it is, and say it with words like "holds", "says", "put back". Text
on screen does a different job: it tells someone what a control does, or what just
happened, while they are trying to do something else. Write it as the scientist rewrote
every tooltip, message and help text in the window: plain, short, and about the user's
action rather than the software's reasoning.

## Before writing

Read the texts already beside the one you are writing — the same panel, dialog or
command — and match them. A new tooltip in the Operations box reads like the others in
the Operations box.

## The rules

1. **Say what it does, not why.** The reason a control exists, how the code works, and
   edge cases go in a docstring or `docs/`. On screen: what happens when it is used.
2. **One short sentence where one will do.** A tooltip opens with a verb (Select, Show,
   Set, Save, Display, Delete) or names the thing ("Opacity of the label").
3. **Keep the one consequence the user must know**, and only that: it asks for
   confirmation, it cannot be undone or resumed, corrected labels are never overwritten.
   Drop reassurances such as "Changes nothing" or "Nothing is written" in the window,
   unless saying so is the point of the control (a Preview button).
4. **Never weaken a warning about corrected labels.** Shorten it, but it must still say
   that corrected work is involved, what would be lost or replaced, and that it needs
   confirmation. The same holds for refusals to overwrite, and for empty labels: an
   empty label is not an annotation.
5. **Accuracy beats brevity.** A short text that misdescribes behaviour is worse than a
   long one. Check the code before writing what something does.
6. **Command-line help may be longer**, being the only documentation a command has.
   Keep it plain, and keep what a user needs there: defaults as `(default: X)`, which
   options combine ("Not with --folds"), what is never overwritten, "Shows a preview by
   default" for commands applied with `--apply`, "Changes nothing." for read-only ones.

## Words

| Say | Not | Note |
|---|---|---|
| folder | directory | Except the established caption *Training directory* |
| AM-seg | am-seg | The product. `am-seg` is the command, and stays so in code font |
| restore | put back | `tests/test_ui_text.py` refuses "put back" |
| free (space) | reclaim | The button caption *Reclaim space...* stays |
| Enable editing before ... | Tick 'Enable editing' before ... | Refused by the test |
| Show or hide | Show / hide | Refused by the test |
| variant | condition | One member of a family; "imaging condition" for the idea in general |
| label | segmentation, mask | |
| ... was changed elsewhere | Somebody else changed ... | Refused by the test |
| the selection, selected for a state change | picked out | Refused by the test. Do not confuse with the *selected* state |

## Fixed by the code or a test

- **Shortcuts** are the last line of a tooltip, `Shortcut: X`, built from the binding.
  Never type a key into the prose where the code can supply it; a test checks every
  named key is bound.
- **Placeholders** (`{name}`, `{len(labels)}`) stay exactly as they are when rewording a
  message.
- **Lengths**: a tooltip is at most 200 characters
  (`test_gui_window.py::test_no_tooltip_in_the_window_is_longer_than_anyone_reads`); in
  the Train tab's Predictions view, one line of at most 50
  (`test_gui_predictions_panel.py::test_what_a_control_says_when_hovered_is_a_short_line`).
- **Export > Filters**: every control ends with "Narrows the breakdown and every export."
  (`test_gui_export_panel.py::test_every_filter_says_what_it_narrows`).
- **A caption and its control share one tooltip** (`common.caption_for`).
- **Retired phrasings** are refused by `tests/test_ui_text.py`. Add one there when a
  rewording retires a phrase across the window, with what is said instead.
- **Tests that pin wording**: when a rewording breaks one, update its expected text and
  keep what it guards. Run `uv run python scripts/guide_tables.py` after changing any
  `--help`.

## Examples

From the scientist's own rewrites:

| Before | After |
|---|---|
| Report how the scanned folder is interpreted and what does not fit. Reads nothing but the directory structure, and changes nothing. | Report how the scanned folder is interpreted without changing anything. |
| Remove the kept label versions no savepoint needs. Everything since the most recent savepoint stays, and the history is unchanged: what was recorded still says what happened. | Delete intermediate label files to free up disk space. |
| Draw the label over the acquisition, or over a flat dark ground. That ground is how a few-pixel annotation is found, and its edge is the frame's; the left pane still shows the image. | Display the image |
| How strongly the label colour is drawn over the image | Opacity of the label |
| Exclude from proofreading. No label is created or measured for these, and a training export leaves them out. | Exclude from proofreading and model training. |
| Mark proofreading finished, which a training export trusts. Leaving this state again asks for confirmation. | Mark proofreading finished. Leaving this state again asks for confirmation. |
| Write a label from each selected image's prediction. Replacing an existing label has to be confirmed, and a corrected one is never replaced. Post-processed (clean/) predictions are not used. | Convert (resize) predictions and write them as labels. Labels marked as corrected are never overwritten. |
| Which series the lamella list, the chips and the series menu cover. A lamella holding only series switched off leaves the list and is counted in its title. Opens as Settings (F2) say. | Toggle which series are listed. Lamellae without selected images are not shown. Defaults can be set in Settings (F2). |

Messages and help, rewritten in the same style and approved:

| Before | After |
|---|---|
| {root.name} is a copy, and a copy can hold part of a project while carrying that project's whole log. A savepoint here would claim a state for labels this copy never had. Take it in the collection it came from. | {root.name} is an exported copy, which may hold only part of a project. Create the savepoint in the original collection. |
| {len(names)} of these files are in the corrected state, which holds manually proofread work: ... Move them to '{state}'? | {len(names)} of these images are marked corrected, which means they were proofread: ... Set them to '{state}'? |
| The accent changes as soon as it is chosen, and Cancel puts it back. The rest decide what each control opens at: the window applies them when it next opens, so anything already on screen keeps what it is set to. | Saved defaults apply the next time AM-seg opens; what is on screen now keeps its settings. The accent colour changes immediately. |
| (`--log-file`) Also write the log to this file, and write a native stack trace there if the process crashes outright. Appends, so runs accumulate. | Also write the log, and a stack trace if the process crashes, to this file. Appends to an existing file. |

The second table's third row is rule 5 at work: a shorter draft said saved defaults were
"applied after saving", which the code does not do.
