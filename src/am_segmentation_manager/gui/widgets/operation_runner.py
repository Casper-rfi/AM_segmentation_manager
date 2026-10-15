"""Running one batch operation: preview it, confirm it, then apply it.

Every operation the window offers goes through here: one busy dialog, one
progress contract, one place deciding what a blocked report asks. The work runs on
``gui/worker.py``'s thread and touches nothing of the window's. **Preview and
apply are two runs of the same function**, never one replayed, a preview held
across a dialog describing a filesystem the user has had time to change. What an
operation changed across the tabs is the window's to bring in step, through the
two callbacks it hands over.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from qtpy.QtCore import QObject
from qtpy.QtWidgets import QDialog, QMessageBox, QWidget

from ...progress import Cancellation
from ..report import OperationReport
from ..worker import BackgroundOperation
from .busy_dialog import BusyDialog
from .report_dialog import ReportDialog

logger = logging.getLogger(__name__)

#: What the confirm button says for each thing an operation can be blocked on,
#: where the report does not name its own. Replacing labels does, its button
#: carrying the counts: "them" read as reaching the corrected labels too.
BLOCKED_ACTIONS: dict[str, str] = {
    "corrected": "Include them",
    "conflicts": "Decide each...",
}


class OperationRunner(QObject):
    """Runs one operation at a time off the interface thread, behind a busy dialog.

    A ``QObject`` living on the interface thread, so that the worker's signals
    are delivered there rather than on the thread that did the work.
    """

    def __init__(
        self,
        window: QWidget,
        set_busy: Callable[[bool], None],
        applied: Callable[[str, OperationReport, bool], None],
        after_failure: Callable[[BaseException], None],
    ) -> None:
        """Start the operations thread.

        Args:
            window: What every dialog is parented to.
            set_busy: Locks the buttons that start an operation, or unlocks them.
            applied: Called as ``applied(title, report, rescan)`` once an
                operation previewed through :meth:`operate` has applied.
            after_failure: Called with what an operation raised, before it is
                reported.
        """
        super().__init__(window)
        self._window = window
        self.set_busy = set_busy
        self.applied = applied
        self.after_failure = after_failure
        # Connected once and left alone. Disconnecting a slot from inside the
        # emission of the signal it is handling is a connection list mutated
        # mid-emit, which the two bindings do not treat alike; what is pending
        # is tracked in a plain attribute instead.
        self.operations = BackgroundOperation()
        self.operations.worker.finished.connect(self._finished)
        self.operations.worker.failed.connect(self._failed)
        self.operations.worker.progressed.connect(self._progressed)
        #: The dialog of the operation running now, and what waits on it finishing.
        self._busy: BusyDialog | None = None
        self._pending: tuple | None = None

    @property
    def progressed(self) -> Any:
        """The signal an operation's work emits ``(done, total)`` on as it advances."""
        return self.operations.worker.progressed

    def run(
        self, description: str, work, on_done, *, cancellation: Cancellation | None = None
    ) -> None:
        """Run *work* off the interface thread, then hand its result to *on_done*.

        The one seam through which every operation reaches its thread, so a test
        can substitute an inline call and still exercise the rest of the flow.

        Args:
            cancellation: Pass one to give the dialog a Stop button. The bar is
                a separate question: the dialog opens indeterminate and turns
                into a count as soon as the work reports progress, so an
                operation can say how far it has got without being stoppable.
        """
        self._busy = BusyDialog(self._window, description, stoppable=cancellation is not None)
        if cancellation is not None:
            self._busy.stop_requested.connect(cancellation.request)
        self._busy.show()
        self.set_busy(True)

        # *work* is held here until its outcome is taken, so that it is freed
        # on this thread; see gui/worker.py.
        self._pending = (description, on_done, work)
        self.operations.submit(work)

    def run_stoppable(self, work, on_done, description: str) -> None:
        """Run *work* with a Stop button and a count, then hand its result to *on_done*.

        Args:
            work: Called with a progress callback and the cancellation the button sets.
        """
        cancellation = Cancellation()
        progressed = self.progressed
        self.run(
            description, lambda: work(progressed.emit, cancellation), on_done,
            cancellation=cancellation,
        )

    def running(self) -> bool:
        """Return whether an operation has been started and not yet handed back."""
        return self._pending is not None

    def _progressed(self, done: int, total: int) -> None:
        """Move the busy dialog's bar, on the thread that owns it.

        A report can arrive after the work has returned -- both are queued
        signals from the same thread -- so a closed dialog is a normal outcome
        rather than a fault.
        """
        if self._busy is None:
            return
        self._busy.set_progress(done, total)

    def _finished(self) -> None:
        """Hand a finished operation's result to whatever asked for it."""
        result = self.operations.worker.take_outcome()
        pending = self._take_pending()
        if pending is None:  # pragma: no cover - nothing was submitted
            return
        _description, on_done, _work = pending
        on_done(result)

    def _failed(self) -> None:
        """Report an operation that raised, once the window is back in step.

        Brought into step whatever the failure, because failing is not the same
        as having changed nothing: a run stopped partway names what it wrote.
        """
        error = self.operations.worker.take_outcome()
        logger.warning("Operation failed: %s", error)
        pending = self._take_pending()
        description = pending[0] if pending else "Operation failed"
        self.after_failure(error)
        QMessageBox.critical(self._window, description, str(error))

    def _take_pending(self) -> tuple | None:
        """Close the busy dialog and return what was waiting on it, once.

        Cleared before the callback runs, so a callback that starts the next
        operation -- which is how preview leads to apply -- is not overwritten
        by the one that just finished.
        """
        pending, self._pending = self._pending, None
        # Detached before it is closed, so anything that runs while it closes
        # finds no dialog rather than one that is being taken apart.
        busy, self._busy = self._busy, None
        if busy is not None:
            busy.close()
            busy.deleteLater()
        self.set_busy(False)
        return pending

    def operate(
        self,
        title: str,
        run,
        *,
        cancellation: Cancellation | None = None,
        rescan: bool = False,
        confirm: Callable[[str, OperationReport, str], bool] | None = None,
        notes: tuple[str, ...] = (),
    ) -> None:
        """Preview an operation, confirm it, then apply what was previewed.

        Args:
            title: Names the operation in every dialog it opens.
            run: Called as ``run(allow, dry_run)``, returning an
                :class:`OperationReport`. ``allow`` answers whatever the report
                was blocked on. It runs on the worker thread, so it must touch
                nothing belonging to the window.
            cancellation: Makes the applying run stoppable. The previews are
                not: they open no file and answer in milliseconds, so a Stop
                button on one is a button with nothing to stop.
            rescan: Handed to *applied*: read the collection again afterwards,
                for an operation whose effect is not the list of files it wrote.
            confirm: Asks about a blocked report, in place of the ordinary
                one. For an operation whose answer is more than yes -- a merge
                collects a decision per conflict -- and it runs here, on the
                thread that may open a dialog, rather than inside *run*.
            notes: Lines every preview opens with, saying what the selection
                leaves out.
        """

        ask = confirm if confirm is not None else self.confirm

        def previewed(report: OperationReport, allowed: bool = False) -> None:
            report = replace(report, lines=(*notes, *report.lines))
            if report.blocked and not allowed:
                action = report.action or BLOCKED_ACTIONS.get(report.blocked, "Include them")
                if not ask(title, report, action):
                    return
                self.run(title, lambda: run(True, True), lambda r: previewed(r, True))
                return
            if not report.count:
                self.show_report(title, report)
                return
            if not self.confirm(title, report, "Apply"):
                return
            self.run(
                title,
                lambda: run(allowed, False),
                lambda done: self.applied(title, done, rescan),
                cancellation=cancellation,
            )

        self.run(title, lambda: run(False, True), previewed)

    def confirm(self, title: str, report: OperationReport, action: str) -> bool:
        """Show a report and ask whether to go ahead."""
        return (
            ReportDialog(self._window, title, report.text(), action).exec()
            == QDialog.DialogCode.Accepted
        )

    def show_report(self, title: str, report: OperationReport) -> None:
        """Show a report with nothing to accept."""
        ReportDialog(self._window, title, report.text()).exec()
