# Display

Why the window looks and behaves as it does, which binding and platform it is written for, and how
it is tested without a display: the code shows what is drawn, not what breaks when a widget, a
thread or a paint path changes. **Every Qt trap is here**, including the ones that only bite the
test suite — a segfault under PySide6 is looked for in one file, not two.
[`decisions.md`](decisions.md) holds the same kind of reasoning for everything below the interface,
and the environment the package is *installed* into. Anything one module does belongs in that
module's docstring; work not built yet belongs in [`04_future_work.md`](04_future_work.md). Correct
an entry in place rather than appending to it.

---

## The window

- **A window that has not been drawn cannot say it is working.** A startup scan run before the
  event loop gives seconds of blank grey indistinguishable from a hang, which is why it is queued
  with the activation below. Where the interface thread must block, the line saying so is
  `repaint`ed rather than merely set: an ordinary `setText` reaches the screen only after the work
  it announces has finished. Anything longer goes to `gui/worker.py` behind the busy dialog — one
  `on_progress(done, total)` contract and one `progressed` signal connected once for the whole
  window, so giving an operation a counter is a callback in its loop rather than a mechanism. Only
  the applying run counts, never the preview. That dialog is not a `QProgressDialog`, for the
  re-entrancy `busy_dialog.py` sets out; the window owns it and detaches it before closing it.
- **The window asks to be the one in front, and asks after the event loop has started.** Launched
  from a terminal, macOS leaves the terminal in front, and a window that is not in front receives
  no key presses and no pointer motion at all — so F1 does nothing, no tooltip appears, and the
  brush outline sits where it was last drawn. Three faults that look unrelated, one cause, and it
  clears the instant anything is clicked, which is what makes it read as occasional.
  `raise_()`/`activateWindow()` go through the same zero-delay timer the startup scan does, ahead
  of it. `changeEvent` logs activation under `-v`, which is not recoverable after the fact.
- **The window never lets CPython print a traceback.** Qt calls Python from C++, so an exception
  escaping a slot propagates nowhere this package can catch it: the interpreter reports it and Qt
  carries on. That reporter opens the source file to quote the failing line, and on macOS it has
  segfaulted the process mid-write, taking the traceback with it. So `gui/app.py` replaces
  `sys.excepthook` and `threading.excepthook` with handlers built on `traceback.format_exception`.
  Every failure is logged; one at a time is also offered in a dialog, **queued through a zero-delay
  timer** because a hook runs part-way through whatever Qt was doing and a modal dialog re-enters
  the event loop at exactly that point. Off the thread that owns the widgets only the log gets it —
  which is why `am-seg gui` logs to a file without being asked, being the one command whose failures
  are not visible in the terminal that started it.
- **A worker thread that will not stop outlives the window rather than being collected.** A frame
  read is one call into an image library and cannot be interrupted part-way, so a file on a
  cloud-synced volume outlasts any deadline worth waiting for on close. A `QThread` still running
  keeps calling into its Python worker, so letting the interpreter collect that worker is a use
  after free whose crash lands on the worker thread with nothing to say why. `loader.keep_alive`
  holds one object per abandoned thread for the life of the process; `BackgroundOperation` uses it
  too, so the reason is written down once.
- **Whether a run can be *stopped* is answered by what a half-done run leaves behind.** Each
  prediction is written atomically and recorded before the next begins, so a stopped run has
  finished files and an accurate account of them, and the next resumes from there; recompression is
  safer still. Converting predictions must not stop — it validates the whole set of destinations
  before writing any, and stopping it leaves exactly the half-applied collection that rule exists
  to prevent. Cancellation is a `threading.Event` in `progress.py` polled by the work closure: a
  queued Qt signal reaches the worker when it next returns to its event loop, which inside a long
  call is never.

## Controls

- **One stylesheet, and it beats the palette.** `theme.py` sets the Fusion style, a palette and a
  short stylesheet; nothing else picks a chrome colour. Fusion rather than the platform's style,
  which differs by platform and by Qt version. Where a rule and the palette disagree the rule
  wins, and a widget palette edited in Python is ignored — yet reads back as the value it was set
  to, so a test asserting on it passes while the screen shows something else: **anything about how
  something *renders* is asserted on the pixels**. The stylesheet also suppresses Qt's emphasis on
  a dialog's default button and a checked tool button, so anything relying on either must re-state
  it; and its padding clips a one-character button blank, which `COMPACT_BUTTON` exempts.
- **A row is ticked; a control on its own is switched.** Two different questions wore the same
  widget: a tick says which of many rows a change covers, where a standalone control is simply on
  or off. `common.ToggleSwitch` draws the second as a switch and is a **subclass of `QCheckBox`**,
  which is the point rather than a detail: only `paintEvent` differs, so `isChecked`, `toggled`,
  Space, focus and the disabled state stay Qt's. It paints everything itself, so **no stylesheet
  rule may name `QCheckBox`** — a rule sends the control back through `QStyle` and draws it twice.
  It also reserves room around the track for its own focus ring, a ring painted past the widget's
  edge being clipped flat on the side it runs off.
- **A chosen row is ticked, never highlighted.** Highlighting cannot say what it means: it is also
  how Qt marks the row you are looking at, it is lost whenever the list is rebuilt, and over dozens
  of values there is no seeing what is chosen without scrolling. `CheckList` is the one
  implementation, and it reports nothing while its contents are replaced — a redraw rather than a
  gesture. Two further traps: Qt blends a selected row's icon with the highlight colour, so a status chip
  whose colour *is* the information comes out wrong on exactly the row being looked at
  (`chip_icon()` supplies every such pixmap); and a `QListWidget` with a horizontal scrollbar
  scrolls instead of eliding, so `ElideMiddle` plus `ScrollBarAlwaysOff` keeps both ends.
- **A list is never rebuilt from inside its own signal.** Selecting a project rebuilds the lamellae
  and nothing else: nothing about a project row changes when one is picked. Rebuilding it there ran
  while Qt was still part-way through the mouse press that caused it, emptying the rows Qt held a
  position in — and the rebuild's own `setCurrentRow` re-framed the list around the clicked row,
  losing the reviewer's place on every click. `_refresh_lists` is for a scan or a state change,
  where every chip really has moved.
- **A widget that wraps takes its height from the width it actually has.** A form asks a field how
  tall it would be at one width and then hands it another, so a flow layout answering only
  `heightForWidth` lays its later rows below the bottom edge — addressable, reported and invisible.
  `_FlowArea` fixes its own height from its real width on every resize. Its layout's minimum width
  is zero and its hint one unwrapped row, which must differ: made equal, the minimum is a floor and
  the pane could never be narrower than the longest value anybody typed.
- **An editor that collects values commits what is in its box before anyone reads it.** A chip
  editor turns typed text into a value on Return, on a pick, and on losing focus — and `_edits`
  asks it to anyway. Held until confirmed, a value typed by someone who then reaches straight for
  Save is discarded without a word. **Adding a widget to a layout reparents it, and reparenting
  hides it**, so it must be shown again before anything measures it: a layout of hidden widgets
  measures zero, which reads as a field that refuses a value *and* loses the one already in it.
- **F1 is an application shortcut, and the list closes only when it is the window in front.** A
  window-scoped `QShortcut` is dead while a second top-level window holds the keyboard, and a
  parented `QDialog` is one — so the key that opens the shortcut list could not close it again;
  and closing on `isVisible()` dismissed a list buried behind the window, unseen. The key now
  arrives from any window, so it is guarded to do nothing while a modal dialog is up, a list
  opened over one being neither readable nor closable until the operation ends. **The list is
  grouped by what a row is for, and a mouse gesture is filed beside the keys it belongs with** —
  picking frames out is one idea whether it is done with `Esc` or with Ctrl-click, and a keys
  table above a gestures table made the reader do the grouping the list exists to do. Every
  `Binding` and every `TAB_GESTURES` row names its heading; the *order* of the headings is the
  order the bindings first name them, so the list is rearranged by moving a binding rather than by
  keeping a second table in step with it. Only "This list" is placed, and it is placed last.

## Images and plots

- **The brush outline is redrawn from state, never from mouse movement alone.** It is drawn in
  *scene* coordinates, so anything moving the scene under a stationary pointer — a pan, a fit,
  centring on the next speck, a frame arriving on the loader thread — leaves a ring around a pixel
  the pointer is no longer over, and every state that hides it left it hidden until the pointer was
  nudged. `_sync_brush` is the single answer to "should there be a ring, and where"; wheel zoom
  needs nothing, `AnchorUnderMouse` holding the scene point under the pointer.
- **The image panes have no scrollbars, their scene is bigger than the image, and zooming out
  lands through `fitInView`.** The scene carries a quarter of the image as slack on every side, so
  an edge can be dragged into the middle of the pane to be worked on — which is also why
  scrollbars go, a scene deliberately larger than its content leaving them permanently visible and
  never full. `downscale_to_fit` uses `min(width_ratio, height_ratio)`, and these images being
  landscape in a portrait pane, **width binds**. Reviewing is done at fit, so the curation panes
  stop there; the proofreading canvas allows one step past it, checking a label runs to the frame's
  edge meaning seeing that edge. At the floor, fitting wins over the wheel's anchor.
- **A summary plot is held twice: pane-sized to step through, whole to look into.** A plot reads no
  faster whatever is asked for (a PNG decompresses whole, so OpenCV's reduced-decode flags buy
  nothing), and slower than a held-down arrow key, so stepping is smooth only with every plot
  already in memory — which bounds those copies to the size of the pane. But these figures carry
  annotations existing only at full resolution. So a *fitted* plot is the pane-sized copy drawn at
  one array pixel per physical pixel, the only scale that resamples nothing; the whole frame is read
  last and takes over once magnified past what the pane copy holds. Read-ahead asks for no more than
  the cache can keep, so `plot_radius` derives from the entry limit.
- **Where a copy is placed is read off the copy, and remembered across a swap.** The copy is
  reduced to a box in *physical* pixels, so drawing it at whatever device ratio the widget reports
  comes out half or twice the size wherever the two disagree; whichever of its edges touches that
  box gives the ratio back exactly. An identity transform says nothing about where the view is
  looking and the scene is deliberately larger than the picture, so drawing 1:1 has to centre as
  fitting would. Where the view *is* looking is held as a fraction of the picture rather than
  re-derived through the whole-number scroll position, which crept the picture towards a corner
  while stepping; it is dropped the moment the user pans.
- **`QGraphicsPixmapItem` overrides the view's render hint with its own transformation mode**,
  which defaults to nearest neighbour. A pane showing something several times its own size drops
  pixels rather than averaging them: against a figure of one-pixel rules, nearest loses whole
  gridlines, so the plot view asks for smooth scaling. Acquisitions do not — zoomed *in* on a
  label, nearest is what shows where one pixel ends and the next begins.
- **Curate and Label show one series at a time; Export ticks several.** A timeline mixing alignment
  runs with polishing frames is not something anyone proofreads, and a 2026 lamella runs to 21
  stages across three series. Each tab has one choice, which rebuilds its frame stack — the position
  and the ticks go with it, a frame index meaning anything only within one timeline. **What was
  chosen and what a lamella can show are two things**: `resolve_series` falls back to a series the
  lamella holds without touching the choice, so the next lamella that has it opens on it; otherwise
  a lamella showing an empty timeline has, the control being hidden for having one entry, no way
  back. **The beam follows the same rule** in `_settle_mode`, asked of the whole stack rather than
  the frame on screen. Both settle in `_rebuild_frames`.
- **A plot describes one acquisition or a whole stage, and the window says which.** An Adaptive
  Polishing stage writes one plot per milling cycle into `plots/`, matched to a frame by its
  acquisition number. An alignment run writes one `figure.png` summarising the several images it
  took to converge, so there is nothing to match it to: every frame of the run shows it and carries
  a badge saying so, a figure describing five frames read as describing one being worse than no
  figure. `displayed_plot` prefers `frame.plot_path` over `stage_plot`; `centring.png` is excluded
  as recording where the plugin looked rather than what the run achieved.

## The binding

- **PySide6 is the binding, and PyQt5 is no longer one.** The `qtpy` compatibility stays, costing
  nothing, but nothing runs the suite under PyQt5 and it had already broken there unnoticed on one
  `setBrush(Qt.NoBrush)`. It may still open; nothing promises it.
- **The window is written for a binding and a platform other than the one it is developed on.**
  Where Qt's behaviour is a style's or a platform's to decide, this package decides instead: every
  file dialog is Qt's own rather than the platform's (`common.build_path_dialog` sets
  `DontUseNativeDialog`; revealing a folder in the file manager stays native), Up and Down step
  through a combo box via `StepCombo` rather than Qt's own keys, and `common.fixed_width_font` names
  real monospace families per platform. That last is asserted on the metrics (`"i"` and `"W"` advance
  equally) and never on `QFont.fixedPitch()` or `QFontDatabase.systemFont(FixedFont)`, both of which
  report the family requested rather than the one that resolved.
- **Never disconnect a signal from inside the handler for that signal.** A connection list mutated
  mid-emit works under PyQt5 and hangs under PySide6, with no traceback. Connect once and track
  what is pending in an attribute; the same shape hides in any connect-per-call pattern.
- **`QProxyStyle.styleHint` segfaults PySide6.** Overriding it to answer `SH_DialogButtonLayout` —
  to fix the button order in every dialog at once, Fusion reporting the Windows order wherever it
  runs — crashes the process while a `QMessageBox` is being built. The save dialog lays its own
  buttons out instead; any dialog that cares about the order has to.

## Testing the window

- **An offscreen window cannot answer three questions**, and asking anyway produces tests that
  pass on nothing. It is never activated, so `hasFocus()` is False for every widget and
  `focusWidget()` records who asked. Nothing is shown, so `isVisible()` is False whatever the code
  did and `isHidden()` is the state to assert. And a modal dialog has nobody to dismiss it, so its
  `exec_` must be intercepted per dialog class, not just `QMessageBox`, or the suite hangs.
- **Constructing a `QApplication` calls `setlocale(LC_ALL, "")`** — a process-global side effect,
  and `test_gui_*` sorts before most other tests. That is how a locale-dependent `strptime('%p')`
  bug appears only in full runs on a non-English locale. `LC_ALL=… pytest` does **not** test this;
  `AM_SEG_TEST_LOCALE=… pytest` does.
