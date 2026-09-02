# Display

Why the window looks and behaves as it does, which binding and platform it is written for, and how
it is tested without a display. **Every Qt trap is here**, including the ones that only bite the
test suite — a segfault under PySide6 is looked for in one file, not two.
[`decisions.md`](decisions.md) holds the same for everything below the interface, and the
environment the package is *installed* into.

---

## The window

- **A window that has not been drawn cannot say it is working.** A startup scan run before the
  event loop gives seconds of blank grey indistinguishable from a hang, so it is queued with the
  activation below. Where the interface thread must block, the line saying so is `repaint`ed rather
  than merely set: an ordinary `setText` reaches the screen only after the work it announces has
  finished. Anything longer goes to `gui/worker.py` behind the busy dialog — one
  `on_progress(done, total)` contract and one `progressed` signal connected once for the whole
  window, so giving an operation a counter is a callback in its loop. Only the applying run counts,
  never the preview. That dialog is not a `QProgressDialog`, for the re-entrancy `busy_dialog.py`
  sets out; the window owns it and detaches it before closing it.
- **A scan still runs on the interface thread, so it carries its own panel.** `ScanBanner` covers
  the window for the length of one, reporting through the same `on_progress(done, total)`, threaded
  from `Scanner.build_index` through `CurationController.scan`. An overlay rather than a dialog, a
  modal dialog needing the event loop to appear at all; parented to the window rather than to a
  tab, since a scan can be asked for from any of them. **No bar until there is a number** — finding
  the roots comes first and cannot say how many there will be, and an indeterminate bar has to
  animate to mean anything. It comes down in a `finally`, whether the read returned or raised.
- **A repaint is not a frame on the screen.** `repaint` reaches a backing store; the frame is
  composited when the main thread next returns to the run loop, which a blocking scan never does —
  so the banner was painted throughout and never once seen. Every update therefore turns the loop
  once. **Which means input has to be stopped, and `ExcludeUserInputEvents` does not stop it**: a
  press already posted is delivered through it. The banner filters the application and swallows
  every press, key, wheel and shortcut itself — that guarantee, held by the widget rather than
  asked of the loop, is what makes turning the loop safe where `busy_dialog.py` says it is not. The
  filter comes off in the same `finally`, one left on being a window nobody can type into.
- **The mark is the only place the orange and green appear.** Both are segmentation class colours
  and the timeline draws its states beside that legend, so a third meaning for either reads as
  data — which is why the banner's bar is the ordinary accent and nothing else is tinted.
  `branding.py` holds the one path to the asset and the one spelling of the product name; a
  checkout without it draws nothing rather than failing.
- **The window asks to be the one in front, and asks after the event loop has started.** Launched
  from a terminal, macOS leaves the terminal in front, and a window that is not in front receives no
  key presses and no pointer motion at all — so F1 does nothing and no tooltip appears. It clears
  the instant anything is clicked, which is what makes it read as occasional.
  `raise_()`/`activateWindow()` go through the same zero-delay timer the startup scan does, ahead of
  it. `changeEvent` logs activation under `-v`, which is not recoverable after the fact. **Anything
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
  reaches every box on every tab. **The weight goes on `QGroupBox`, never on `QGroupBox::title`**:
  Qt ignores font properties on a subcontrol, so the obvious rule is silently no rule, and this one
  reaches the title without reaching the controls inside. No background: the inputs are white, and a
  white card behind them takes away the one thing separating "you can type here" from the panel.
- **One of a small closed set is a run of segments, not a row of dots.** `common.SegmentedControl`
  is one widget holding checkable `QToolButton`s in an exclusive group, which also means the run
  **cannot be split across two lines of a flow layout** — loose buttons wrap apart, leaving one
  choice reading as two. The chosen segment is *filled* with the accent where a checked tool button
  elsewhere is merely tinted: those answer "is this on", this answers "which of these", and the two
  must not look alike. The joining is object names
  (`segmentFirst`/`segmentMiddle`/`segmentLast`/`segmentOnly`) and rules in `theme.py`, chrome being
  decided there; **every name has to be spelled out in every rule**, Qt's stylesheets having no
  wildcard, and a name left out is a segment silently drawn as a plain button. `buttons` exposes the
  segments so a caller can grey one out, which is how a view the frame has nothing to show for is
  disabled. **The State group is deliberately not one**: it is non-exclusive so a frame whose SEM
  and FIB disagree can show *nothing* chosen, which a run of segments cannot say.
- **A row is ticked; a control on its own is switched.** `common.ToggleSwitch` draws the second and
  is a **subclass of `QCheckBox`**, which is the point rather than a detail: only `paintEvent`
  differs, so `isChecked`, `toggled`, Space, focus and the disabled state stay Qt's. It paints
  everything itself, so **no stylesheet rule may name `QCheckBox`** — a rule sends the control back
  through `QStyle` and draws it twice — and it reserves room for its own focus ring, one painted
  past the widget's edge coming out clipped flat.
- **A chosen row is ticked, never highlighted**, highlighting being also how Qt marks the row you
  are looking at and lost whenever the list is rebuilt. `CheckList` is the one implementation, and
  it reports nothing while its contents are replaced — a redraw rather than a gesture. Two traps: Qt
  blends a selected row's icon with the highlight colour, so a status chip whose colour *is* the
  information comes out wrong on exactly the row being looked at (`chip_icon()` supplies every such
  pixmap); and a `QListWidget` with a horizontal scrollbar scrolls instead of eliding, so
  `ElideMiddle` plus `ScrollBarAlwaysOff` keeps both ends.
- **The projects are grouped under the folder each sits in, on all three tabs that list them.** An
  Autolamella project is named for the minute it started, so a hundred in a flat list are a hundred
  timestamps; the folder somebody filed them under is what a person recognises. `common.ProjectTree`
  is one widget beside `CheckList` — a tab panel importing from another is a dependency between two
  things with nothing to do with each other. `Project.folder` and `Project.own_name` split the
  `relative_path` the scanner already writes, and a project at the scan root stays at the top level
  rather than gaining a `(root)` row naming a directory that is not there. **A tick still means what
  it meant**: a folder's box is Qt's tri-state, so ticking one ticks its projects and each reports
  itself. A tree needs the list rules of `theme.py` as well as the list *settings*, or Qt paints its
  current row in the full accent.
- **What differs between the three is passed in.** The chip is a curation status on Curate and a
  completeness bar on Add data; `checkable=False` takes the tick boxes off Add data, which acts on
  the row being *looked at*. **That flag is cleared rather than merely not set** —
  `ItemIsUserCheckable` is on by default and Qt just draws no box while no check state exists, so
  every row would report as checkable and the first `setCheckState` would put the boxes back.
  Export has no project list but a project *filter*, whose values are relative paths: that one draws
  as a tree, named in `NESTED_FILTERS`, a display decision belonging in the interface rather than as
  a flag on `Breakdown` below it. Every other filter stays a list, and the row holds both widgets
  and shows one rather than having either pretend to be the other.
- **A folder is closed until it is opened, and what is open survives a rebuild.** Three things open
  one anyway: holding the current project, being rebuilt after a rescan having been open before, and
  matching a *filter* — hiding a folder's siblings while leaving its own matches folded away reads
  as having found nothing. The folds before a filter are put back when it is cleared, opening a
  folder to search being the search's doing rather than the reader's.
- **Add data describes a folder of projects at once, and a field they disagree on is left alone.** A
  session's projects share their sample, their grid and their operator, and typing that a dozen
  times is how a collection ends up with `HeLa` beside `Hela`. The scope is a list and one project
  is a list of one, so both are one code path. A field they differ on resolves to `MIXED_MARKER`,
  which goes *in the box*: what to write is decided by comparing the box against the resolved value,
  so an untouched field matches and is never written. The badge beside it is authoritative, being
  derived where the marker is text somebody could have typed; a field is outstanding if *any*
  project still holds a placeholder. A folder is a batch, planned whole before any of it is written
  by the rule in [`decisions.md`](decisions.md) § Workflow state. Writing a value needs no asking;
  **clearing** one is confirmed, reading as tidying a placeholder rather than as emptying that field
  everywhere under the folder.
- **The status chip is a cell per series and beam, and nothing combines across the two.** Three
  columns in `SERIES` order, SEM above FIB; only `LABELS_EXPECTED` ever varies, a reference column
  that changed being a claim that work nobody intends to do is outstanding. Combined, reference
  images sitting at the default state for ever make a finished project report as part-done.
- **Its cells are told apart by brightness, not by hue, and none is outlined.** The timeline's blue
  and teal are the same brightness, so a cell split between them was separated by the one channel
  the eye cannot use at that size. `CELL_COLOURS` is a ramp, lightest
  for a series with no images and darkest for one finished, the greys of a series nobody labels
  sitting off it. Its hues are still the timeline's, but the chip is a *summary* rather than those
  states repeated — a "done" cell counts deselected images as finished, and those are grey on the
  timeline — which is what lets it have a ramp of its own. An outline round a seven-pixel cell takes
  more of it than the colour it frames, so the fills hold against the white row and the `SELECTED`
  tint instead. **`SWATCH_HEIGHT` must stay even**: Qt centres a row's check indicator on the row —
  Fusion's is eighteen pixels — but rounds the decoration differently, so an odd-height icon lands
  half a pixel below the tick beside it.
- **Choosing a folder lists every lamella under it, and `controller.project` is then the project of
  the lamella in view rather than the row last clicked.** An Autolamella run restarts for reasons
  that have nothing to do with the specimen, so one session's lamellae are spread over a dozen
  projects mostly holding one each; Up and Down cross a project boundary without stopping, the
  boundary being an artifact. **The rows need no project prefix**: no folder in the collection holds
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
- **A list is never rebuilt from inside its own signal, and breaking the rule corrupts the heap
  rather than merely glitching.** Selecting a project rebuilds the lamellae and nothing else:
  rebuilding the project list there runs while Qt is part-way through the mouse press that caused
  it, emptying the rows Qt holds a position in. `_refresh_lists` is for a scan or a state change,
  where every chip really has moved. The Export tab broke it and segfaulted: refilling a filter row
  clears the tree it is drawn in, so doing that from inside `itemChanged` destroys the item whose
  signal is being emitted while Qt is still propagating the tick to that folder's other children —
  and the process dies later, with nothing to connect it to. **Where the signal comes from Qt rather
  than from this package the answer is a zero-delay single-shot timer** (`ExportPanel._recount`),
  which also collapses a gesture moving many controls into one recount. The property dropdown
  refills itself from inside its own `currentIndexChanged` and goes through the same timer.
- **A widget that wraps takes its height from the width it actually has.** A form asks a field how
  tall it would be at one width and then hands it another, so a flow layout answering only
  `heightForWidth` lays its later rows below the bottom edge — addressable, reported and invisible.
  `FlowArea` fixes its own height from its real width on every resize. Its layout's minimum width is
  zero and its hint one unwrapped row, which must differ: made equal, the minimum is a floor and the
  pane could never be narrower than the longest value anybody typed. It carries a field holding
  however many values it is given *and* the image panel's view controls, which have wanted more
  width than the centre pane opens at since before the intensity control joined them. `addSpacing`
  is the one thing a row moved off a box layout needs and `QLayout` does not have.
- **An editor that collects values commits what is in its box before anyone reads it**, or a value
  typed by someone who then reaches straight for Save is discarded without a word: a chip editor
  commits on Return, on a pick, and on losing focus, and `_edits` asks it to anyway. **Adding a
  widget to a layout reparents it, and reparenting hides it**, so it must be shown again before
  anything measures it: a layout of hidden widgets measures zero, which reads as a field that
  refuses a value *and* loses the one already in it.
- **F1 is an application shortcut, and the list closes only when it is the window in front.** A
  window-scoped `QShortcut` is dead while a second top-level window holds the keyboard, and a
  parented `QDialog` is one — so the key that opened the list could not close it again; and closing
  on `isVisible()` dismissed a list buried behind the window, unseen. Guarded to do nothing while a
  modal dialog is up.
- **The list is grouped by what a row is for, mouse gestures beside the keys they belong with.**
  Picking frames out is one idea whether done with `Esc` or Ctrl-click. Every `Binding` and every
  `TAB_GESTURES` row names its heading, and the heading *order* is the order the bindings first name
  them — so the list is rearranged by moving a binding rather than by keeping a second table in
  step, and *within* a heading by moving a row up its own table: keys first, then gestures in
  `TAB_GESTURES` order. Only "This list" is placed, and it is placed last. **A heading is not bold
  alone**, every left-hand cell being bold as the column a reader scans; it takes the colour and
  weight a `QGroupBox` title takes over its card.
- **It opens with a keyboard, the keys the tab binds picked out on it.** Drawn with `QPainter` like
  the tool marks, so it has no asset to keep in step with the bindings — `_keys_of` reads them, so a
  key added to the window appears on the diagram unasked, and a chord lights its modifiers too. A
  tab with no keys of its own gets no diagram.
- **It also says what the marks mean, and shows them rather than describing them.** Each module
  supplies the legend for what it paints, and the timeline's sample is *grabbed from a
  `FrameTimeline`* — the only way a legend cannot claim a mark the widget never draws.
  `common.mark_image` puts each sample into the HTML as a `data:` URI, there being no file to point
  at; **`QBuffer(QByteArray())` segfaults PySide6**, the array dying under the buffer, so it is held
  in a name. The white band is listed per tab, one mark with two meanings listed on both meaning
  neither. The list scrolls and is capped to the screen; `fit_to_screen` sizes from the label's own
  hint, `adjustSize` being useless when a scroll area reports a small default instead of its
  contents.

## Images and plots

- **The brush outline is redrawn from state, never from mouse movement alone.** It is drawn in
  *scene* coordinates, so anything moving the scene under a stationary pointer — a pan, a fit,
  centring on the next speck, a frame arriving on the loader thread — leaves a ring around a pixel
  the pointer is no longer over, and every state that hides it left it hidden until the pointer was
  nudged. `_sync_brush` is the single answer to "should there be a ring, and where", and
  `_poll_pointer` asks it every 50 ms while the window is *not* the one in front, that being the
  case where no event will ask it: `QCursor.pos` is answered by the window system rather than by an
  event, so it is right when nothing else is. It runs exactly while the brush would paint, started
  and stopped beside the cursor it moves.
- **Nothing may depend on the scene point Qt recorded from the last mouse event a view handled.**
  That is what `AnchorUnderMouse` anchors on, and two ordinary things stop such events arriving:
  the window not being in front, which on macOS means no pointer motion at all; and the
  proofreading canvas accepting a mouse move without passing it on for the whole of a stroke, which
  freezes the recorded point wherever the stroke began until some later move happens to fall
  through. Either way the next wheel zoom pulls the image towards a pixel the pointer left some
  time ago, which is why it looked occasional and why restarting sometimes appeared to fix it. So
  the anchor is `NoAnchor` and `zoom_by` takes the point to hold: the wheel event's own position,
  the one thing that always knows where the pointer was. It is put back by `centerOn` rather than
  by translating, the linked pane copying this one's transform.
- **The intensity limits are a view of the frame, applied after the cache and never in it.** A
  lookup table over the uint8 array `DisplayImage` already holds, so moving a handle costs one pass
  over 256 values rather than invalidating the frame and re-reading the file behind it. It opens on
  the whole range, and at that setting hands the cached array straight back. Everything outside the
  limits is clipped, which is the point — an acquisition whose data sits in a narrow band is
  reviewed by throwing away the empty ends — and is why this is a *view*: `to_display_8bit`, what a
  written file goes through, clips nothing. **It is kept when the frame changes**, comparing one
  frame with the next being what it is for, and `common.intensity_badge` says so in the warning
  colour whenever it is not the whole range: this is the one control that makes the picture differ
  from the file. Curate and Label each hold their own, reviewing and proofreading looking at a frame
  for different reasons; on the canvas only the picture is rescaled, the label being painted into as
  it is or a stroke would be recorded against something that is not on disk.
- **One control sets both limits, and they are called Minimum and Maximum.** Fiji's words, which is
  the vocabulary the people using this already have; "black / white" named the *result* and left the
  reader working out which end each of two sliders was. `common.RangeSlider` draws the interval — a
  filled span between two handles, the grey ends being what is thrown away. The limits are held
  apart by the control rather than by the transform, a picture turning into a threshold reading as a
  fault instead of a setting; the handle being *moved* is the one that gives way, being pushed along
  reading as the control having taken the wrong one. Double click puts one handle back without
  touching the other, Qt offering no gesture for neutral. **Painted entirely itself like
  `ToggleSwitch`, so no stylesheet rule may name it.**
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
  the ratio back exactly. An identity transform says nothing about where the view is looking and the
  scene is larger than the picture, so drawing 1:1 has to centre as fitting would. Where the view
  *is* looking is held as a fraction of the picture rather than re-derived through the whole-number
  scroll position, which crept it towards a corner while stepping; it is dropped the moment the user
  pans.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**, which
  defaults to nearest neighbour. A pane showing something several times its own size drops pixels
  rather than averaging them: against a figure of one-pixel rules, nearest loses whole gridlines, so
  the plot view asks for smooth scaling. Acquisitions do not — zoomed *in* on a label, nearest is
  what shows where one pixel ends and the next begins.
- **Curate and Label show one series at a time; Export ticks several.** A timeline mixing alignment
  runs with polishing frames is not something anyone proofreads. Each tab has one choice, which
  rebuilds its frame stack — the position and the ticks go with it, a frame index meaning anything
  only within one timeline. **What was chosen and what a lamella can show are two things**:
  `resolve_series` falls back to a series the lamella holds without touching the choice, or a
  lamella showing an empty timeline has, the control being hidden for holding one entry, no way
  back. **The beam follows the same rule** in `_settle_mode`, asked of the whole stack rather than
  the frame on screen; both settle in `_rebuild_frames`.
- **A plot describes one acquisition or a whole stage, and the window says which.** An Adaptive
  Polishing stage writes one plot per milling cycle into `plots/`, matched to a frame by its
  acquisition number. An alignment run writes one `figure.png` for the several images it took to
  converge, so there is nothing to match it to: every frame of the run shows it and carries a badge
  saying so, a figure describing five frames read as describing one being worse than none.
  `displayed_plot` prefers `frame.plot_path` over `stage_plot`; `centring.png` is excluded as
  recording where the plugin looked rather than what the run achieved.

## The binding

- **PySide6 is the binding, and PyQt5 is no longer one.** The `qtpy` compatibility stays, costing
  nothing, but nothing runs the suite under PyQt5 and it had already broken there unnoticed on one
  `setBrush(Qt.NoBrush)`. It may still open; nothing promises it.
- **The window is written for a binding and a platform other than the one it is developed on.**
  Where Qt's behaviour is a style's or a platform's to decide, this package decides instead: every
  file dialog is Qt's own (`common.build_path_dialog` sets `DontUseNativeDialog`; revealing a folder
  in the file manager stays native), Up and Down step through a combo box via `StepCombo`, and
  `common.fixed_width_font` names real monospace families per platform. That last is asserted on the
  metrics (`"i"` and `"W"` advance equally) and never on `QFont.fixedPitch()` or
  `QFontDatabase.systemFont(FixedFont)`, both of which report the family requested rather than the
  one that resolved.
- **A key pressed while a completer's popup is open does not reach the field's event filters.** The
  popup grabs the keyboard and gets the key; `QCompleter` then passes it on by calling
  `widget->event()`, which reaches the field's own handlers and skips every filter installed on it.
  So a filter watching only the field watches the one route the key does not take — and **the
  offscreen platform hides this**, being unable to grab a keyboard and so delivering to the field
  directly. A test written against it passed while the window did nothing. `common.match_anywhere`
  installs on both, and `test_gui_completion.py` sends the key both ways.
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
- **Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side effect,
  and `test_gui_*` sorts before most other tests. That is how a locale-dependent `strptime('%p')`
  bug appears only in full runs on a non-English locale. `LC_ALL=… pytest` does **not** test this;
  `AM_SEG_TEST_LOCALE=… pytest` does.
