# Display

Why the window looks and behaves as it does, which binding and platform it is written for, and how
it is tested without a display. **Every Qt trap is here**, including the ones that only bite the
test suite — a segfault under PySide6 is looked for in one file, not two.
[`decisions.md`](decisions.md) holds the same for everything below the interface, and the
environment the package is *installed* into.

**An entry states a rule and names the test that guards it** rather than rehearsing an argument
that test's own docstring already makes; what that requires of an entry, and what may be taken out
of one of these files, is held by `test_docs.py`.

---

## The window

- **A window that has not been drawn cannot say it is working**, so the startup scan is queued with
  the activation below (`test_the_startup_work_waits_for_the_window_to_appear`). Where the interface
  thread must block, the line saying so is `repaint`ed rather than merely set: an ordinary `setText`
  reaches the screen only after the work it announces has finished. Anything longer goes to
  `gui/worker.py` behind the busy dialog — one `on_progress(done, total)` contract and one
  `progressed` signal connected once for the whole window, so giving an operation a counter is a
  callback in its loop. Only the applying run counts, never the preview. That dialog is not a
  `QProgressDialog`, for the re-entrancy `busy_dialog.py` sets out; the window owns it and detaches
  it before closing it.
- **A scan still runs on the interface thread, so it carries its own panel.** `ScanBanner` covers
  the window for the length of one, reporting through the same `on_progress(done, total)`, threaded
  from `Scanner.build_index` through `CurationController.scan`. An overlay rather than a dialog, a
  modal dialog needing the event loop to appear at all; parented to the window rather than to a tab,
  since a scan can be asked for from any of them. **No bar until there is a number**
  (`test_a_scan_that_cannot_count_yet_says_so_without_a_bar`). It comes down in a `finally`, whether
  the read returned or raised (`test_a_failed_scan_still_takes_the_banner_down`).
- **A repaint is not a frame on the screen** (`test_each_update_is_put_on_the_screen`), so every
  banner update turns the event loop once. **Which means input has to be stopped, and
  `ExcludeUserInputEvents` does not stop it**: a press already posted is delivered through it
  (`test_no_user_input_is_delivered_while_the_banner_is_up`). The banner filters the application and
  swallows every press, key, wheel and shortcut itself — that guarantee, held by the widget rather
  than asked of the loop, is what makes turning the loop safe where `busy_dialog.py` says it is not.
  The filter comes off in the same `finally`, one left on being a window nobody can type into.
- **Every accent in `theme.ACCENTS` is a class colour, so it stays on the controls.** Nothing on
  the timeline or beside a label takes it, and a chosen class is ringed in ink rather than in the
  accent, which would merge into its swatch (`test_a_chosen_class_is_ringed_in_ink_not_the_accent`).
  `branding.py` holds both paths to it and the one spelling of the product name; a checkout
  without the asset draws nothing rather than failing. **macOS draws no container for a Dock icon**,
  so the icon there carries one: a superellipse over 824/1024 of the canvas, measured off the
  system's own (`test_the_mac_icon_is_the_shape_macos_draws_its_own_icons_in`).
- **The window asks to be the one in front, and asks after the event loop has started** —
  `test_the_window_asks_to_be_the_one_in_front` says what a window without the keyboard loses on
  macOS. It clears the instant anything is clicked, which is what makes it read as occasional.
  `raise_()`/`activateWindow()` go through the same zero-delay timer the startup scan does, ahead of
  it; `changeEvent` logs activation under `-v`, which is not recoverable after the fact. **Anything
  that must work anyway asks the window system rather than waiting for an event** — the brush
  outline and the wheel's anchor both do, under Images and plots.
- **The window never lets CPython print a traceback.** Qt calls Python from C++, so an exception
  escaping a slot reaches the interpreter's own reporter — which opens the source file to quote the
  failing line, and on macOS has segfaulted the process mid-write, taking the traceback with it. So
  `gui/app.py` replaces both excepthooks with handlers built on `traceback.format_exception`. Every
  failure is logged; one at a time is also offered in a dialog, **queued through a zero-delay
  timer**, a hook running part-way through whatever Qt was doing and a modal dialog re-entering the
  event loop there. Off the widget thread only the log gets it — which is why `am-seg gui` logs to a
  file unasked, its failures not being visible in the terminal that started it.
- **The window fits the screen it is on, and nothing in it may force otherwise.** Qt clamps a resize
  to the window's minimum, so a row of controls that cannot wrap is a floor under the whole window
  rather than a blemish in one tab: rows wrap (`FlowArea`, under Controls), a caption elides
  (`test_a_long_caption_does_not_widen_the_curate_tab`), and the floor is held against a named
  screen size (`test_no_tab_needs_more_room_than_the_smallest_screen_supported`). `fit_to_screen` is
  the other half, `DEFAULT_SIZE` and a restored geometry both naming a screen this may not be on
  (`test_a_window_remembered_off_the_edge_is_moved_back_on`).
- **A worker thread that will not stop outlives the window rather than being collected.** A frame
  read is one call into an image library and cannot be interrupted part-way, so a file on a
  cloud-synced volume outlasts any deadline worth waiting for on close. `loader.keep_alive` states
  the trade and both the loader and `BackgroundOperation` take it.
- **Whether a run can be *stopped* is answered by what a half-done run leaves behind.** A
  prediction is written atomically and recorded before the next begins, so a stopped run has
  finished files and an accurate account of them. **Converting predictions must not stop** — it
  validates every destination before writing any, and stopping it leaves exactly the half-applied
  collection that rule exists to prevent. Cancellation is a `threading.Event` in `progress.py`
  polled by the work closure: a queued Qt signal reaches the worker when it next returns to its
  event loop, which inside a long call is never.

## Controls

- **One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a palette and a
  short stylesheet; nothing else picks a chrome colour. Fusion rather than the platform's style,
  which differs by platform and by Qt version. Where a rule and the palette disagree the rule wins,
  and a widget palette edited in Python is ignored — yet reads back as the value it was set to, so a
  test asserting on it passes while the screen shows something else: **anything about how something
  *renders* is asserted on the pixels**. The stylesheet also suppresses Qt's emphasis on a dialog's
  default button and a checked tool button, so anything relying on either must re-state it; and its
  padding clips a one-character button blank, which `COMPACT_BUTTON` exempts.
- **A group of controls is a card, and its title outweighs the captions inside it.** One rule
  reaches every box on every tab, and **the weight goes on `QGroupBox`, never on
  `QGroupBox::title`** — `test_a_group_title_is_heavier_than_the_captions_inside_it` says why. Its
  fill sits between window and field (`test_a_card_sits_between_the_window_and_the_fields_in_it`).
- **One of a small closed set is a run of segments, not a row of dots.** `controls.SegmentedControl`
  is one widget holding checkable `QToolButton`s in an exclusive group, which also means the run
  **cannot be split across two lines of a flow layout** — loose buttons wrap apart, leaving one
  choice reading as two. The chosen segment is *filled* with the accent where a checked tool button
  elsewhere is merely tinted: those answer "is this on", this answers "which of these", and the two
  must not look alike. **So Convert's *From* is segments**, not a switch whose off is the other
  source (`test_choosing_where_predictions_come_from_recounts_the_line`). The joining is object names
  (`segmentFirst`/`segmentMiddle`/`segmentLast`/`segmentOnly`) and rules in `theme.py`, chrome being
  decided there; **every name has to be spelled out in every rule**, Qt's stylesheets having no
  wildcard, and a name left out is a segment silently drawn as a plain button. `buttons` exposes the
  segments so a caller can grey one out, which is how a view the frame has nothing to show for is
  disabled. **The State group is deliberately not one**: it is non-exclusive so a frame whose SEM
  and FIB disagree can show *nothing* chosen, which a run of segments cannot say. **Nor is the
  Export tab's beam**, which is one of three and still a dropdown: among a run of dropdowns a
  fourth kind of widget reads as clutter before it reads as a choice.
- **A caption and the control it names are one flow item, and so are two controls that pair.** The
  rule the segments keep, for their reason: a flow wraps between its items, so anything loose is
  split at whatever width the pane happens to be
  (`test_a_caption_is_never_a_flow_item_of_its_own`). `controls.keep_together` holds a caption to
  its control and `keep_paired` stacks two into an aligned grid, which also halves the width a row
  of them needs (`test_the_view_controls_are_one_block_on_the_smallest_screen_supported`). What
  pairs answers one question of two things — series over variant, the two Show switches — except on
  Label, where undo, redo and save sit under the beam buttons because image software keeps them there.
- **A row is ticked; a control on its own is switched.** `controls.ToggleSwitch` draws the second and
  is a **subclass of `QCheckBox`** (`test_it_is_a_check_box`), so only `paintEvent` differs. It
  paints everything itself, so **no stylesheet rule may name `QCheckBox`** — a rule sends the
  control back through `QStyle` and draws it twice — and it reserves room for its own focus ring
  (`test_the_focus_ring_fits_inside_the_widget`).
- **A chosen row is ticked, never highlighted**, highlighting being also how Qt marks the row you
  are looking at and lost whenever the list is rebuilt. `lists.CheckList` is the one implementation,
  and it reports nothing while its contents are replaced — a redraw rather than a gesture. Two traps,
  each pinned: a status chip must come from `chip_icon()`
  (`test_a_chip_is_the_same_pixmap_in_every_icon_mode`), and a list elides rather than scrolling
  sideways (`test_a_long_filter_row_elides_rather_than_scrolling_sideways`).
- **The projects are grouped under the folder each sits in, on all three tabs that list them.** An
  Autolamella project is named for the minute it started, so a hundred in a flat list are a hundred
  timestamps; the folder somebody filed them under is what a person recognises. `lists.ProjectTree`
  is one widget beside `CheckList` — a tab panel importing from another is a dependency between two
  things with nothing to do with each other. `Project.folder` and `Project.own_name` split the
  `relative_path` the scanner already writes, and a project at the scan root stays at the top level
  rather than gaining a `(root)` row naming a directory that is not there. **A tick still means what
  it meant**: a folder's box is Qt's tri-state, so ticking one ticks its projects and each reports
  itself. A tree needs the list rules of `theme.py` as well as the list *settings*, or Qt paints its
  current row in the full accent.
- **The furniture round the tree comes from `lists.project_box`, and what differs is passed in** —
  the chip, and whether rows are checkable at all (`test_a_tree_can_be_built_without_tick_boxes`).
  Three tabs list the same projects, and which of them a reader is looking at is not something the
  window should have to teach. **The checkable flag is cleared rather than merely not set**:
  `ItemIsUserCheckable` is on by default and Qt just draws no box while no check state exists, so
  every row would report as checkable and the first `setCheckState` would put the boxes back.
- **The left column is read across rather than down, and is sized for its widest row.** It is the
  narrowest on the tab and holds the most, so captions sit beside their boxes and *One image per
  label* sits at the margin. Squeezed, Qt cuts a caption from both ends rather than shortening it,
  so the column opens wider than its widest row
  (`test_nothing_in_the_filters_column_is_clipped_at_the_width_it_opens_at`). The series switches
  head the column above the projects, where Curate asks the same question of its lists
  (`test_the_series_switches_head_the_column_above_the_projects`); the projects are a filter like
  the rest, drawn *above* the box rather than as a second outline inside it.
- **Export filters *by* project rather than listing them, and that control is never taken away.**
  Which projects an export covers is the first question anyone asks of one, and it was a row
  somebody had to know to add (`test_the_project_filter_cannot_be_taken_away`). Being dedicated is
  also what lets every other filter row be a flat list (`test_a_filter_row_is_always_a_flat_list`).
- **Every project ticked is stored as no filter, and none ticked as a filter matching nothing.** The
  values on offer are recomputed from the records the other filters pass, so a stored list of all of
  them would quietly drop a project that appeared afterwards
  (`test_a_project_that_appears_later_joins_an_unfiltered_export`). The other end needs
  `set_choice_filter`'s `empty_keeps_everything`: an empty choice is how a filter *row* says it has
  stopped filtering, and storing one here would widen the export at the press of a button called
  None (`test_unticking_every_project_matches_nothing_rather_than_everything`).
- **Its four buttons wrap rather than clipping**, the furniture being what the three tabs share
  rather than the number of lines it takes (`test_no_button_in_a_project_box_is_ever_clipped`).
- **A folder is closed until it is opened, and what is open survives a rebuild.** Three things
  open one anyway: holding the current project, being rebuilt after a rescan having been open
  before, and matching a *filter* (`test_a_folder_holding_a_match_is_opened_to_show_it`), which
  `ProjectTree.hide_rows_without` says the rest of.
- **Add data describes a folder of projects at once, and a field they disagree on is left alone**,
  for the reason `MetadataController.select_folder` gives. The scope is a list and one project is a
  list of one, so both are one code path. A field they differ on resolves to `MIXED_MARKER`,
  which goes *in the box*: what to write is decided by comparing the box against the resolved value,
  so an untouched field matches and is never written. The dot beside it is authoritative, being
  derived where the marker is text somebody could have typed; a field is outstanding if *any*
  project still holds a placeholder. A folder is a batch, planned whole before any of it is written
  by the rule in [`decisions.md`](decisions.md) § Workflow state. Writing a value needs no asking;
  **clearing** one is confirmed, reading as tidying a placeholder rather than as emptying that field
  everywhere under the folder.
- **The status chip is a cell per series and beam, and nothing combines across the two.** A column
  per series switched on, SEM above FIB; only `LABELS_EXPECTED` ever varies, a reference column that
  changed being a claim that work nobody intends to do is outstanding. Combined, reference images
  sitting at the default state for ever make a finished project report as part-done. Its width is
  `swatch_width` of the columns drawn rather than a number written down: a column painted outside
  the pixmap is clipped silently.
- **Its cells are told apart by brightness, not by hue, and none is outlined.** The timeline's blue
  and teal are the same brightness, so a cell split between them was separated by the one channel
  the eye cannot use at that size. `CELL_COLOURS` is a ramp, lightest
  for a series with no images and darkest for one finished, the greys of a series nobody labels
  sitting off it. Its hues are still the timeline's, but the chip is a *summary* rather than those
  states repeated — a "done" cell counts deselected images as finished, and those are grey on the
  timeline — which is what lets it have a ramp of its own. An outline round a seven-pixel cell takes
  more of it than the colour it frames, so the fills hold against the white row and each accent's
  selected tint instead. **`SWATCH_HEIGHT` must stay even**, for the reason
  `test_the_chip_is_an_even_number_of_pixels_tall` gives.
- **Choosing a folder lists every lamella under it, and `controller.project` is then the project of
  the lamella in view rather than the row last clicked**, `controller.select_folder` saying why one
  session's lamellae are spread over a dozen projects and why Up and Down cross the boundary between
  them. **The rows need no project prefix**: no folder in the collection holds
  two lamellae of the same petname. Every caller of `project` is asking about what is on screen; a
  project holding no lamellae has none to answer for, so the tree's row stands in. `ViewPosition`
  carries the folder, restoring the project alone silently narrowing the list to one of them.
- **The lamella list is what a row was *clicked*, never what was ticked, and the tree's cursor stays
  on the folder.** Ticking is what a change applies to, and `Selection` keeps one level at a time —
  so ticking a lamella clears the project ticks, and a list filled from those would empty itself the
  moment anyone used it. Moving the cursor onto whichever project a lamella belongs to would
  likewise re-enter that list's own signal and snap the scope back to one project, so the summary
  line is what says which project you are in — load-bearing rather than decorative.
  Choosing a folder moves no frame and no state: the lamella in view is kept when the new list holds
  it, which it does whenever the folder chosen is the one already being worked in.
- **A list is never rebuilt from inside its own signal**, which corrupts the heap rather than merely
  glitching — `test_unticking_a_folder_unticks_every_project_under_it` sets out how. Selecting a project
  rebuilds the lamellae and nothing else; `_refresh_lists` is for a scan or a state change, where
  every chip really has moved. **Where the signal comes from Qt rather than from this package the
  answer is a zero-delay single-shot timer** (`ExportPanel._recount`,
  `test_nothing_is_rebuilt_while_a_tick_is_still_being_delivered`), which also collapses a gesture
  moving many controls into one recount. The property dropdown refills itself from inside its own
  `currentIndexChanged` and goes through the same timer.
- **A widget that wraps takes its height from the width it actually has**, for the reason
  `FlowArea` gives: answering `heightForWidth` alone lays the later rows below the bottom edge,
  addressable and invisible. Its layout's minimum width is zero and its hint one unwrapped row,
  which must differ: made equal, the minimum is a floor and the pane could never be narrower than
  the longest value anybody typed. `addSpacing` is the one thing a row moved off a box layout
  needs and `QLayout` does not have.
- **An editor that collects values commits what is in its box before anyone reads it**, or a value
  typed by someone who then reaches straight for Save is discarded without a word: a chip editor
  commits on Return, on a pick, and on losing focus, and `_edits` asks it to anyway. **Adding a
  widget to a layout reparents it, and reparenting hides it**, so it must be shown again before
  anything measures it (`test_a_chip_is_tall_enough_to_be_seen_as_soon_as_it_exists`).
- **F1 is an application shortcut, and the list closes only when it is the window in front** —
  `test_f1_works_from_the_list_it_opened` and `test_f1_fetches_back_a_list_left_behind_the_window`
  say why neither a window-scoped shortcut nor `isVisible()` will do. It is guarded to do nothing
  while a modal dialog is up (`test_f1_is_ignored_while_an_operation_holds_the_window`).
- **The list is arranged by moving a binding, never by keeping a second table in step with one.**
  Every `Binding` and every `TAB_GESTURES` row names its heading; the heading *order* is the order
  the bindings first name them, and within a heading keys come first, then gestures in
  `TAB_GESTURES` order (`test_a_gesture_sits_under_the_same_heading_as_the_keys_it_belongs_with`,
  `test_the_three_ways_to_resize_the_brush_are_listed_together`). Only "This list" is placed, and it
  is placed last. A heading takes the colour and weight a `QGroupBox` title takes over its card
  (`test_a_heading_is_not_drawn_like_the_rows_under_it`).
- **The diagram and the legend are drawn by whatever draws the real thing**, so neither can claim a
  key or a mark the window does not have: `_keys_of` reads the bindings, a gesture names the keys it
  holds (`test_a_key_held_for_a_gesture_is_lit_on_the_diagram`), and each module supplies
  its own legend with the timeline's sample *grabbed from a `FrameTimeline`*
  (`test_the_legend_shows_the_marks_rather_than_describing_them`). Two Qt facts have no test:
  `common.mark_image` must hold its `QByteArray` in a name, **`QBuffer(QByteArray())` segfaulting
  PySide6** as the array dies under the buffer; and `fit_to_screen` sizes from the label's own hint,
  `adjustSize` being useless when a scroll area reports a small default instead of its contents.

- **Every column of the Export tab can be dragged, as the other tabs' can**, and so is the left
  column split between the projects and the filters under them — which of the two wants the room
  depends on what is being looked for. `FILTER_LIST_HEIGHT` is a floor rather than a cap for the
  same reason (`test_every_division_that_moves_is_marked_as_one`).
- **A collection-wide act goes on Add data; anything narrowed goes on Curate.** `operations_panel`
  promises that everything in it follows the ticks, so a savepoint — which is over the whole dataset
  a model was trained on — would break that promise there. Taking one, restoring one over everything
  and reclaiming archive space sit in *Maintenance* with `compress-labels`, and reverting a
  selection stays in Curate. **What the archive costs is measured when that tab is opened**, never
  during a scan: it walks every project's archive and reads every log, and one stat per project
  decides whether that is worth a worker at all.
- **A refusal carries no count and cannot be confirmed past.** `_operate` offers to override anything
  a report is `blocked` on, which is right for proofread work and wrong for a missing fork point:
  there is nothing to include, and the button that says so cannot help. A refusal returns a headline
  and no count, which `_show_report` shows and dismisses.
- **Neither side of a conflict is offered as the answer.** A default is a rule about which tree wins,
  and a conflict exists precisely because there is not one — so nothing is chosen in advance and
  *Merge* stays disabled until every conflict is answered. **Qt makes the first accepting button in
  a `QDialogButtonBox` the default and draws it in the accent**, which recommends a side in the one
  place nothing may: both are cleared and Cancel takes it, as before every other write. Return and
  Escape therefore reach Cancel, which asks before discarding decisions already made.
- **A conflict heading shows decisions and holds none.** Two marks per row, *Here* and *Copy*, each
  full, dotted or empty: a heading decided both ways must show some of each, which one control per
  row could not (`test_a_heading_shows_how_much_of_it_each_side_holds`). **The keys decide the
  conflict on screen, never a heading**, or one keystroke could decide a folder of a thousand
  (`test_the_keys_act_on_the_conflict_on_screen_never_a_heading`). A field conflict is a row of
  *Field | Here | Copy* and a value is chosen by clicking it, so two values are compared side by side
  rather than read out of a sentence (`test_clicking_a_value_chooses_it_and_clicking_it_again_undecides`).
- **Who changes are recorded under is a standing line in the corner, not a warning per action.**
  Ten actions across all four tabs put a name into a log, and two exports write it into what they
  leave behind — a copy's fork point, a training set's manifest: there is no single tab to say it
  on and far too many places to warn at. It is muted, being a fact rather than a message.
- **A default decides what a control opens at, and nothing else.** Nothing `gui/preferences.py`
  holds reaches a write path (`test_a_default_decides_what_a_control_opens_at_and_nothing_else`).
  The window applies them once, so a control somebody is using is never moved under them; the
  accent alone, being chrome, is shown as it is chosen and put back on Cancel, and nothing may keep
  the old one (`test_switching_the_accent_leaves_none_of_the_old_one_on_screen`). The dialog says
  so, and is built from the same table (`test_the_settings_dialog_offers_every_preference`).
- **A word-wrapped label reports a one-line hint, and the squeeze lands somewhere else.** A dialog
  sized from that hint comes up short and everything below is pushed towards its minimum, nowhere
  near the label that caused it. `controls.WrappingLabel` takes its height from the width it has,
  as `FlowArea` does (`test_every_group_of_settings_gets_the_height_its_rows_need`).
- **The folder is chosen in the tab row, not in a tab.** A scan serves all four, and only Curate
  could show which folder was read; the corner holds the path, Browse and Scan, and says on hover
  what the folder held (`test_the_folder_is_chosen_in_the_tab_row_on_every_tab`).
- **A window that has read nothing offers the folder in the middle of it, where the panes would be**: nothing
  in the window works until a folder is read, empty panes look exactly like a lamella holding nothing, and the
  tab row's Scan is one accented control among many
  (`test_a_window_that_has_read_nothing_offers_the_folder_it_remembers`). The button names the folder and the
  path beneath says which one it is; with none remembered, the one button opens the chooser
  (`test_a_window_with_no_folder_remembered_offers_to_choose_one`). It is a page of the view stack, so
  anything switching views leaves it. **Nothing is scanned at startup** — a remembered root may be large, or
  on a drive that is not there — so Rescan reads the box instead of doing nothing
  (`test_rescan_reads_the_remembered_folder_before_anything_has_been_scanned`).
- **The corner is three runs with a rule between each, not one line of text**, read from the folder to what
  this machine is set to and the gear that changes it, and on to the mark
  (`test_the_corner_reads_from_the_folder_to_the_mark`). **The mark carries no words**: the title bar names
  the product, and printed here too the window said its own name twice
  (`test_the_corner_shows_the_mark_without_naming_the_product`). The gear and the **?** are what nothing else
  mentions, F2 and F1 (`test_the_settings_button_opens_what_f2_opens`), and **the path field gives way before
  the tabs**, Qt otherwise shrinking them to slivers (`test_the_path_field_gives_way_before_the_tabs_do`).
- **Everything that records shares one `Attribution`, so adopting a name is one assignment.** Several
  controllers write to a log and one is reached through the Label tab rather than the window, so
  telling them individually was a list kept by hand in two places that a new one fell off silently.
  They hold the object rather than a recorder now, and the test finds what records instead of
  naming it (`test_setting_a_name_reaches_everything_that_records`) — which is why the next
  controller to record cost nothing but its own construction.

## Images and plots

- **An earlier version replaces the label on the canvas rather than sitting beside it**, and so does
  the other side of a merge conflict. Segmentation is compared by flicking between two pictures in
  one place; side by side halves the resolution of both. Held on a key (**B**, V already showing and
  hiding the label) rather than toggled, **badged while it shows** — the two look alike, and a
  reviewer who cannot tell which is on screen is comparing without knowing what — and **not
  paintable**, or a stroke would land on the label underneath, unseen. The difference mask draws
  with a class value no configuration defines, so it appears in the colour reserved for an
  unconfigured class rather than as a segmentation; **flat, one colour**, since with several classes
  on both sides colouring by what each holds has no readable scheme and the mask becomes the thing
  being deciphered. Two labels of different sizes are still both shown — that one is the wrong size
  is the finding — and only the mask goes, there being no pixel to pair.
- **With the acquisition hidden, something still says where the frame ends.** Over the pane's own
  black, the label's transparent background and the emptiness past the frame's edge are one
  surface, so a label a row short along an edge looks like one that runs to it — which is the
  reason to hide the image at all. `display.HIDDEN_IMAGE_GREY` is the ground, dark enough that a
  speck still outshines it (`test_the_ground_under_a_hidden_image_is_not_the_pane_behind_it`), and
  two mechanisms carry it (`test_hiding_the_image_leaves_a_ground_the_size_of_the_frame`).
- **The strip is closed until somebody opens it, and the count sits on the switch**, so a label
  with nothing to compare says so unopened — with predictions listed too it would otherwise show on
  almost every frame (`test_the_switch_opens_the_strip_and_says_how_many_there_are`).
- **Two labels are compared by looking, never by a number.** Every metric over a pair — pixel share,
  IoU, Dice — presupposes that one of them is ground truth, and where a proofread label meets a good
  model's prediction neither is; worse, pixel share ranks lowest exactly the failure that matters
  most, a crack being a fraction of a percent of the frame. So a batch table ranking labels by how
  far something moved them is refused, and so is a percentage beside a conflict; the answer is one
  frame at a time on a subset.
- **Editing never starts from pixels older than the label's file.** A prepared frame holds the pixels
  its label had when read, perhaps long before it is opened, so the cache fingerprints the file just
  before reading and editing opens with that fingerprint; a frame whose file changed since is read
  again first (`test_a_frame_read_ahead_opens_what_was_written_since`). **A save puts what it wrote,
  and its fingerprint, into the prepared frame**, or the canvas draws the label it replaced. Put in
  rather than dropped: dropping leaves nothing to draw until a background read catches up, which is
  a blank canvas exactly where a save was just made.

- **The brush outline is redrawn from state, never from mouse movement alone**, being drawn in
  *scene* coordinates: anything moving the scene under a stationary pointer leaves a ring around a
  pixel the pointer is no longer over, which
  `test_the_brush_follows_the_scene_when_the_view_moves_under_it` pins.
  `_sync_brush` is the single answer to "should there be a ring, and where", and
  `_poll_pointer` asks it every 50 ms while the window is *not* the one in front, that being the
  case where no event will ask it: `QCursor.pos` is answered by the window system rather than by an
  event. It runs exactly while the brush would paint.
- **Nothing may depend on the scene point Qt recorded from the last mouse event a view handled** —
  what `AnchorUnderMouse` anchors on, and `test_the_wheel_holds_the_point_it_is_pointed_at` and
  `test_painting_does_not_move_where_the_wheel_zooms` are the two ordinary ways such an event never
  arrives. So the anchor is `NoAnchor` and `zoom_by` takes the point to hold: the wheel event's own
  position, the one thing that always knows where the pointer was. It is put back by `centerOn`
  rather than by translating, the linked pane copying this one's transform.
- **The intensity limits are a view of the frame, applied after the cache and never in it.** A
  lookup table over the uint8 array `DisplayImage` already holds, so moving a handle costs one pass
  over 256 values rather than re-reading the file behind it, and at the whole range it hands the
  cached array straight back. Everything outside the limits is clipped
  (`test_everything_outside_the_limits_is_clipped`), which is why this is a *view*:
  `to_display_8bit`, what a written file goes through, clips nothing. **It is kept when the frame
  changes**, comparing one frame with the next being what it is for, and `common.intensity_badge`
  says so in the warning colour whenever it is not the whole range. Curate and Label each hold their
  own; on the canvas only the picture is rescaled, the label being painted into as it is or a stroke
  would be recorded against something that is not on disk.
- **One control sets both limits, and they are called Min and Max** — Fiji's words, abbreviated,
  for the reason `image_panel` gives where it names them. `controls.RangeSlider` draws the interval: a
  filled span between two handles, the grey ends being what is thrown away. The limits are held
  apart by the control rather than by the transform, a picture turning into a threshold reading as a
  fault instead of a setting; the handle being *moved* is the one that gives way, being pushed along
  reading as the control having taken the wrong one. Double click puts one handle back without
  touching the other, Qt offering no gesture for neutral. **Painted entirely itself like
  `ToggleSwitch`, so no stylesheet rule may name it.** Every one-handled slider beside it is a
  `controls.ValueSlider`, Qt's slider drawn the same way, Fusion's having read as another kind of
  control (`test_it_is_drawn_as_the_range_slider_is`).
- **The image panes have no scrollbars, their scene is bigger than the image, and zooming out lands
  through `fitInView`.** The scene carries a quarter of the image as slack on every side, so an edge
  can be dragged into the middle of the pane to be worked on — which is also why scrollbars go, a
  scene deliberately larger than its content leaving them permanently visible and never full.
  `downscale_to_fit` uses `min(width_ratio, height_ratio)`, and these images being landscape in a
  portrait pane, **width binds**. Reviewing is done at fit, so the curation panes stop there; the
  proofreading canvas allows one step past it, checking a label runs to the frame's edge meaning
  seeing that edge. At the floor, fitting wins over the wheel's anchor.
- **A summary plot is held twice: pane-sized to step through, whole to look into.** A plot reads no
  faster whatever is asked for (a PNG decompresses whole, so OpenCV's reduced-decode flags buy
  nothing) and slower than a held-down arrow key, so stepping is smooth only with every plot in
  memory — which bounds those copies to the size of the pane. But these figures carry annotations
  existing only at full resolution. So a *fitted* plot is the pane-sized copy drawn at one array
  pixel per physical pixel, the only scale that resamples nothing; the whole frame takes over once
  magnified past what the pane copy holds. `plot_radius` derives from the entry limit.
- **Where a copy is placed is read off the copy, and remembered across a swap.** The copy is reduced
  to a box in *physical* pixels, so drawing it at whatever device ratio the widget reports comes out
  half or twice the size wherever the two disagree; whichever of its edges touches that box gives
  the ratio back exactly. An identity transform says nothing about where the view is looking, so
  drawing 1:1 has to centre as fitting would. Where the view *is* looking is held as a fraction of
  the picture rather than re-derived through the whole-number scroll position
  (`test_swapping_resolution_repeatedly_does_not_drift`); it is dropped the moment the user pans.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**,
  which defaults to nearest neighbour, so a pane showing something several times its own size
  drops pixels rather than averaging them. The plot view therefore asks for smooth scaling and
  acquisitions do not, for the reason `ImageView.set_smooth_scaling` gives.
- **Curate and Label show one series at a time; Export ticks several.** A timeline mixing alignment
  runs with polishing frames is not something anyone proofreads. Each tab has one choice, which
  rebuilds its frame stack — the position and the ticks go with it, a frame index meaning anything
  only within one timeline. **What was chosen and what a lamella can show are two things**:
  `resolve_series` falls back to a series the lamella holds without touching the choice, or a
  lamella showing an empty timeline has no way back. A dropdown of one entry is drawn as text and
  one of several carries its count (`test_a_dropdown_with_one_entry_reads_as_text`). **The beam follows the same rule** in `_settle_mode`, asked of the whole stack rather than
  the frame on screen; both settle in `_rebuild_frames`. **Which series are covered at all is a
  second choice**, the *Series shown* switches: a lamella holding only series switched off leaves
  the list, counted in its title, and reaches no change
  (`test_the_residue_opens_switched_off_and_counted`); `other` opens off.
- **A series with no labels is shown and not acted on.** The timeline shows `other`, opening one to
  judge whether the rule or the data is wrong being why they are indexed, while the state buttons
  and every operation button are withheld, each reading or writing a label the series lacks
  (`test_the_controls_that_change_data_are_withheld_for_the_residue`), and say why on hover. Asked
  of the series *shown*, not the one chosen: `resolve_series` lands there by itself for a lamella
  holding nothing else. **A second reason, not the editing toggle**, held apart in both panels so
  editing cannot unlock it (`test_switching_editing_back_on_does_not_unlock_the_residue`); behind
  them `controller.scoped_records`, which every operation reads, drops those records. **Both tabs
  say *view only*, not *no label***; the Label tab's `key_for` answers None where
  `label_destination` refuses (`test_the_label_tab_opens_on_the_residue`). **The beam control is
  untouched**, already offering what a frame holds; an image whose beam nothing states reaches no
  frame, and `inspect-layout` reports those.
- **The Grid view shows the project, not the frame**: `available_modes` asks it of the project, so
  it survives stepping through a timeline (`test_the_grid_view_survives_stepping_between_lamellae`),
  is absent only where a project holds no overview, and is drawn before the empty-frame check.
  **Several overviews are one picker**, in the slot the Grid view has no imaging condition for,
  opening on the earliest that places anything and drawn from its gamma copy; one placing nothing is
  shown saying why (`test_the_grid_view_offers_every_overview_in_the_variant_slot`). **The badge
  says the marks are approximate every time**, for what `reproject` ignores; a lamella anyone
  curates that neither `experiment.yaml` nor its images position is named as such
  (`test_a_lamella_the_experiment_never_positioned_is_told_so`). A mark keeps its size as the
  picture is zoomed (`ItemIgnoresTransformations`) and its name sits on a filled chip, `grid_view`
  saying why of both. **Both take the accent**: an overview carries no label for the accent to be
  read as, and it shows against light and dark grey alike.
- **The chart carries its own bar, never matplotlib's toolbar.** That toolbar's *Subplots* and
  *Customize* change a figure bound for a paper with nothing recording it, so the bar offers reset,
  pan, zoom, save, copy and the table as CSV, pan and zoom still driven by a hidden toolbar
  (`test_the_chart_offers_no_way_to_edit_the_figure`). **A cell nothing was counted in is grey and
  unnumbered** (`test_a_combination_nothing_was_acquired_for_is_grey_and_unnumbered`).
- **A plot describes one acquisition or a whole stage, and the window says which.** An Adaptive
  Polishing stage writes one plot per milling cycle into `plots/`, matched to a frame by its
  acquisition number. An alignment run writes one `figure.png` for the several images it took to
  converge, so there is nothing to match it to: every frame of the run shows it and carries a badge
  saying so, a figure describing five frames read as describing one being worse than none.
  `displayed_plot` prefers `frame.plot_path` over `stage_plot`; `centring.png` is excluded as
  recording where the plugin looked rather than what the run achieved.

## The binding

- **PySide6 is the binding, and PyQt5 is no longer one.** The `qtpy` compatibility stays, costing
  nothing, but it had already broken under PyQt5 unnoticed on one `setBrush(Qt.NoBrush)`, and the
  suite now refuses to start there rather than failing a dozen pixel comparisons whose message names
  a `sip.voidptr` and not the binding
  (`test_an_unsupported_binding_stops_the_run_before_any_test`). The window may still open; nothing
  promises it.
- **The window is written for a binding and a platform other than the one it is developed on.**
  Where Qt's behaviour is a style's or a platform's to decide, this package decides instead: every
  file dialog is Qt's own (`paths.build_path_dialog` sets `DontUseNativeDialog`; revealing a folder
  in the file manager stays native), Up and Down step through a combo box via `StepCombo`, and
  `common.fixed_width_font` names real monospace families per platform. That last is asserted on the
  metrics, `"i"` and `"W"` advancing equally (`test_a_report_lines_up_in_a_fixed_width_font`), and
  never on `QFont.fixedPitch()` or `QFontDatabase.systemFont(FixedFont)`, both of which report the
  family requested rather than the one that resolved.
- **A key pressed while a completer's popup is open does not reach the field's event filters.** The
  popup grabs the keyboard and gets the key; `QCompleter` then passes it on by calling
  `widget->event()`, which reaches the field's own handlers and skips every filter installed on it.
  So a filter watching only the field watches the one route the key does not take — and **the
  offscreen platform hides this**, being unable to grab a keyboard and so delivering to the field
  directly. A test written against it passed while the window did nothing. `common.match_anywhere`
  installs on both, and `test_gui_completion.py` sends the key both ways.
- **A PySide6 wrapper lives only while the wrapper that vended it does.** `bar.actions()[0].menu()`
  hands back a `QMenu` that is *already deleted* by the time it is used, the `QAction` wrapper
  having been collected on the way out of the expression — and a `QAction` read from that menu and
  returned past it goes the same way. Nothing warns; the object raises "Internal C++ object already
  deleted" when asked anything. So the window keeps every menu and item it builds in an attribute
  (`test_the_window_has_the_menus_a_desktop_application_has`), and a reader walking the bar holds
  each link of the chain in a local and takes plain values out rather than live objects.
- **Never disconnect a signal from inside the handler for that signal.** A connection list mutated
  mid-emit works under PyQt5 and hangs under PySide6, with no traceback. Connect once and track what
  is pending in an attribute; the same shape hides in any connect-per-call pattern.
- **`QProxyStyle.styleHint` segfaults PySide6.** Overriding it to answer `SH_DialogButtonLayout` —
  to fix the button order in every dialog at once, Fusion reporting the Windows order wherever it
  runs — crashes the process while a `QMessageBox` is being built. The save dialog lays its own
  buttons out instead; any dialog that cares about the order has to.

## Testing the window

- **An offscreen window cannot answer three questions**, and asking anyway produces tests that pass
  on nothing. It is never activated, so `hasFocus()` is False for every widget and `focusWidget()`
  records who asked. Nothing is shown, so `isVisible()` is False whatever the code did and
  `isHidden()` is the state to assert. And a modal dialog has nobody to dismiss it, so its `exec_`
  must be intercepted per dialog class, not just `QMessageBox`, or the suite hangs.
- **The window's error reporter, once installed, turns a later test's exception into a hang.**
  `app.install_error_reporting` replaces `sys.excepthook` for the whole session, and a test through
  `app.run` that does not patch it leaves it on. An exception in a Qt callback afterwards — a
  zero-delay timer firing on a window already destroyed is the usual one — queues the failure
  dialog, and whichever test next turns the event loop opens it and waits for a click nobody gives.
  So patch it, and let a queued call run before `destroy`
  (`test_the_window_opens_in_the_accent_its_settings_hold`); `-o faulthandler_timeout=120` names
  the test that stalls.
- **Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side effect,
  and `test_gui_*` sorts before most other tests. That is how a locale-dependent `strptime('%p')`
  bug appears only in full runs on a non-English locale. `LC_ALL=… pytest` does **not** test this;
  `AM_SEG_TEST_LOCALE=… pytest` does.
- **A test that builds a window destroys it** (`conftest.destroy`), `close()` only hiding one, and
  the suite fails any test that leaves a top-level widget alive
  (`test_a_test_that_leaves_a_widget_alive_is_named_and_one_that_frees_its_own_is_not`).
  `setStyleSheet` and `setPalette` walk every live widget at a cost that grows faster than linearly,
  so a window left behind slows the *later* tests that apply the theme rather than the one that
  leaked it. `deleteLater()` does not free one: the window is held by the bound methods in its own
  key bindings, and `processEvents` never carries out a deferred deletion anyway, which is why
  `destroy` flushes them. The guard only counts: a hover card is a top-level window its strip still
  owns, and deleting what is left crashes a later test.
- **A wait names what it waits for** (`conftest.settle`), which requires a condition and fails on
  timeout. One with nothing to wait for cannot tell the work finishing from the time running out, so
  it sleeps through its whole timeout and then asserts nothing.
