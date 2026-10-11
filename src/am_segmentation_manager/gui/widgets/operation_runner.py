"""Running one batch operation: preview it, confirm it, then apply it.

Every operation the window offers goes through here: one busy dialog, one
progress contract, one place deciding what a blocked report asks. The work runs on
``gui/worker.py``'s thread and touches nothing of the window's. **Preview and
apply are two runs of the same function**, never one replayed, a preview held
across a dialog describing a filesystem the user has had time to change. A mixin,
because the run is the window's own; see ``docs/display.md``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QDialog, QMessageBox

from ...progress import Cancellation, StoppedPartway
from ..report import OperationReport
from .busy_dialog import BusyDialog
from .report_dialog import ReportDialog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .window_protocol import MixinBase as _Window
else:
    _Window = object

logger = logging.getLogger(__name__)

#: What the confirm button says for each thing an operation can be blocked on,
#: where the report does not name its own. Replacing labels does, its button
#: carrying the counts: "them" read as reaching the corrected labels too.
BLOCKED_ACTIONS: dict[str, str] = {
    "corrected": "Include them",
    "conflicts": "Decide each...",
}


class OperationRunner(_Window):
    """The window's half of running an operation off the interface thread.

    What it uses of the window is declared in :mod:`.window_protocol`.
    """

    #: The dialog of the operation running now, and what waits on it finishing.
    _busy: BusyDialog | None
    _pending_operation: tuple | None

    def _run_operation(
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
        self._busy = BusyDialog(self, description, stoppable=cancellation is not None)
        if cancellation is not None:
            self._busy.stop_requested.connect(cancellation.request)
        self._busy.show()
        self.operations_panel.set_busy(True)

        # *work* is held here until its outcome is taken, so that it is freed
        # on this thread; see gui/worker.py.
        self._pending_operation = (description, on_done, work)
        self.operations.submit(work)

    def _run_stoppable(self, work, on_done, description: str) -> None:
        """Run *work* with a Stop button and a count, then hand its result to *on_done*.

        Args:
            work: Called with a progress callback and the cancellation the button sets.
        """
        cancellation = Cancellation()
        progressed = self.operations.worker.progressed
        self._run_operation(
            description, lambda: work(progressed.emit, cancellation), on_done,
            cancellation=cancellation,
        )

    def _operation_progressed(self, done: int, total: int) -> None:
        """Move the busy dialog's bar, on the thread that owns it.

        A report can arrive after the work has returned -- both are queued
        signals from the same thread -- so a closed dialog is a normal outcome
        rather than a fault.
        """
        if self._busy is None:
            return
        self._busy.set_progress(done, total)

    def _operation_finished(self) -> None:
        """Hand a finished operation's result to whatever asked for it."""
        result = self.operations.worker.take_outcome()
        pending = self._take_pending_operation()
        if pending is None:  # pragma: no cover - nothing was submitted
            return
        _description, on_done, _work = pending
        on_done(result)

    def _operation_failed(self) -> None:
        """Report an operation that raised, and bring the window back in step.

        Brought into step whatever the failure, because failing is not the same
        as having changed nothing: a run stopped partway names what it wrote,
        and those files go on their records as after a finished run.
        """
        error = self.operations.worker.take_outcome()
        logger.warning("Operation failed: %s", error)
        pending = self._take_pending_operation()
        description = pending[0] if pending else "Operation failed"
        if isinstance(error, StoppedPartway) and error.written:
            self.controller.attach_written(error.written)
            self.export_controller.forget_labels(error.written)
            self.label_panel.refresh()
        self.refresh(reload_lists=True)
        QMessageBox.critical(self, description, str(error))

    def _operation_running(self) -> bool:
        """Return whether an operation has been started and not yet handed back."""
        return self._pending_operation is not None

    def _take_pending_operation(self) -> tuple | None:
        """Close the busy dialog and return what was waiting on it, once.

        Cleared before the callback runs, so a callback that starts the next
        operation -- which is how preview leads to apply -- is not overwritten
        by the one that just finished.
        """
        pending, self._pending_operation = self._pending_operation, None
        # Detached before it is closed, so anything that runs while it closes
        # finds no dialog rather than one that is being taken apart.
        busy, self._busy = self._busy, None
        if busy is not None:
            busy.close()
            busy.deleteLater()
        self.operations_panel.set_busy(False)
        return pending

    def _operate(
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
            rescan: Read the collection again afterwards, for an operation
                whose effect is not the list of files it wrote. See
                :meth:`_operation_applied`.
            confirm: Asks about a blocked report, in place of the ordinary
                one. For an operation whose answer is more than yes -- a merge
                collects a decision per conflict -- and it runs here, on the
                thread that may open a dialog, rather than inside *run*.
            notes: Lines every preview opens with, saying what the selection
                leaves out -- see :meth:`_scope_notes`.

        Preview and apply are two runs of the same function rather than one run
        whose result is replayed: a preview held across a dialog describes a
        filesystem the user has had time to change.
        """

        ask = confirm if confirm is not None else self._confirm_report

        def previewed(report: OperationReport, allowed: bool = False) -> None:
            report = replace(report, lines=(*notes, *report.lines))
            if report.blocked and not allowed:
                action = report.action or BLOCKED_ACTIONS.get(report.blocked, "Include them")
                if not ask(title, report, action):
                    return
                self._run_operation(
                    title, lambda: run(True, True), lambda r: previewed(r, True)
                )
                return
            if not report.count:
                self._show_report(title, report)
                return
            if not self._confirm_report(title, report, "Apply"):
                return
            self._run_operation(
                title,
                lambda: run(allowed, False),
                lambda applied: self._operation_applied(title, applied, rescan),
                cancellation=cancellation,
            )

        self._run_operation(title, lambda: run(False, True), previewed)

    def _operation_applied(
        self, title: str, report: OperationReport, rescan: bool = False
    ) -> None:
        """Record what an operation did and bring the window back in step.

        Args:
            rescan: For an operation whose effect is not a list of files it
                wrote -- a new *project*, which changes what the index holds
                rather than what a record carries. Everything that only creates
                labels or predictions names them through ``written`` instead.
        """
        self.state_panel.show_last_change(f"{title}: {report.headline}")
        logger.info("%s: %s", title, report.headline)
        self._show_report(title, report)
        # An operation that says what it wrote needs no scan to find it: the
        # paths go on the records, and both views drop what they had read from
        # the files those paths name -- the Export tab keys its label states
        # and measurements by path, so a replaced label would keep reporting
        # the one it replaced.
        self.controller.attach_written(report.written)
        self.export_controller.forget_labels(report.written)
        # Marked on both sides, as the steps of a prediction run are: the last
        # thing an operation does is read the whole collection again, and a
        # window that disappears once the work is finished has to be placed
        # either side of that read.
        logger.info("%s: %s", title, "rescanning" if rescan else "refreshing")
        if rescan:
            self.rescan()
        else:
            self.refresh(reload_lists=True)
        logger.info("%s: the window is back in step", title)

    def _confirm_report(self, title: str, report: OperationReport, action: str) -> bool:
        """Show a report and ask whether to go ahead."""
        return ReportDialog(self, title, report.text(), action).exec() == QDialog.DialogCode.Accepted

    def _show_report(self, title: str, report: OperationReport) -> None:
        """Show a report with nothing to accept."""
        ReportDialog(self, title, report.text()).exec()

    def _editable_records(self) -> list | None:
        """Return what an operation would act on, or None with a reason shown.

        The same guard and the same selection the state buttons use: an
        operation that writes files is no less of a change for being called
        something else.
        """
        if not self.controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Tick 'Enable editing' before running an operation.",
            )
            return None
        records = self.controller.scoped_records()
        if not records:
            self._flash(self.controller.left_out_note() or "Nothing selected")
            return None
        return records

    def _scope_notes(self) -> tuple[str, ...]:
        """Return what a preview of the selection says it leaves out, if anything."""
        note = self.controller.left_out_note()
        return (f"{note}.",) if note else ()

