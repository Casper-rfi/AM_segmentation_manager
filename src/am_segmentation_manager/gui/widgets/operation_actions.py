"""What each menu and button actually asks an operation to do.

One method per action the window offers -- predict, convert, recompress, copy
out, merge back, revert, savepoint, reclaim, name projects, export. Each builds
the call, hands it to :class:`~.operation_runner.OperationRunner` and says what
to do with the result; the deciding belongs to
:mod:`am_segmentation_manager.gui.operations_controller`, which holds no Qt.

A mixin on the window, so an action reaches the panels and the controllers the
way the rest of the window does. Nothing here writes a file itself.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QDialog, QMessageBox

from ...operations.predict_process import DeviceProbe, probe_devices
from ...operations.training_export import LeftoverFolds
from ...operations.training_manifest import manifest_path
from ...progress import Cancellation
from .. import preferences
from ..report import OperationReport
from .common import MAX_LISTED_FILES
from .conflict_dialog import ConflictDialog
from .project_dialog import AddProjectsDialog
from .revert_dialog import RevertDialog
from .savepoint_dialog import SavepointDialog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .window_protocol import MixinBase as _Window
else:
    _Window = object

logger = logging.getLogger(__name__)



class OperationActions(_Window):
    """Every batch action the window offers, one method each.

    Mixed into the window beside :class:`~.operation_runner.OperationRunner`,
    whose ``_operate`` and ``_run_operation`` each of these calls.
    """

    #: The devices the machine offers, once asked; whether the archive was measured.
    _devices: DeviceProbe | None
    _archive_measured: bool

    def _predict(self, model: str) -> None:
        """Predict the selection with the model named in the panel.

        Guarded in the order the answers are needed: whether this window may
        write at all, then whether it can predict, then whether it was told
        what with. Asking for a model first would ask for a file before saying
        the operation was never going to run.
        """
        records = self._editable_records()
        if records is None:
            return

        # No check that imports torch. The window predicts in a child process
        # precisely so that torch is never loaded here, and asking whether it
        # would load is asking by loading it. A stack that is missing or broken
        # is reported by the child, which is the process that has to import it.
        if not model:
            self._flash("Choose a model to predict with")
            return
        weights = Path(model)
        # Remembered against the view it was chosen for, so the next session
        # opens with the same model in the box -- one model, many runs.
        self._remember_model()
        device = self._remember_device()

        # Created here rather than in _operate so the work closure and the
        # dialog share one: the closure polls it on the worker thread, the
        # dialog's Stop button sets it on this one.
        cancellation = Cancellation()
        progressed = self.operations.worker.progressed

        # Not rescanned: a prediction that did not exist when the scan ran is
        # not in the index and nothing would offer to convert it, but the run
        # says where it put every one, which is the whole of what a scan would
        # have found here.
        self._operate(
            "Predict labels",
            lambda allow, dry: self.operations_controller.predict(
                records, weights, device=device, dry_run=dry,
                cancellation=None if dry else cancellation,
                on_progress=None if dry else progressed.emit,
            ),
            cancellation=cancellation,
            notes=self._scope_notes(),
        )

    def _offer_devices(self) -> None:
        """Fill the device box with what the machine actually has, and how full its GPUs are.

        Which devices exist is asked of a separate process, because this one
        never imports torch -- that is what predicting in a child is for. It
        costs an interpreter start and a torch import, so it is done once per
        session, when the box is first opened. The GPUs' free memory changes
        while others work, and costs ``nvidia-smi`` alone, so it is read afresh
        every time the box opens.
        """
        from ...ml.gpus import gpu_inventory

        if self._devices is None:
            self.project_panel.announce("Checking which devices are available...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._devices = probe_devices()
            finally:
                QApplication.restoreOverrideCursor()
            logger.info(
                "Devices available for prediction: %s", self._devices.devices or "none"
            )
            if self._devices.problem:
                logger.warning("%s", self._devices.problem)
                self.project_panel.announce(self._devices.problem)
        descriptions = {gpu.device: gpu.describe() for gpu in gpu_inventory().gpus}
        self.operations_panel.show_devices(
            self._devices.devices, descriptions, self._devices.problem
        )
        self.refresh()

    def _convert_predictions(self) -> None:
        records = self._editable_records()
        if records is None:
            return
        origin = self.operations_panel.prediction_origin()
        # Not rescanned, for the reason under Predict labels: the new labels
        # would otherwise keep drawing as unlabelled, and naming them on their
        # records says the same thing without walking the collection.
        self._operate(
            "Convert predictions",
            lambda allow, dry: self.operations_controller.convert_predictions(
                records, allow_overwrite=allow, dry_run=dry, prefer=origin
            ),
            notes=self._scope_notes(),
        )

    def _find_specks(self, size: int) -> None:
        """Report which of the ticked projects hold labels with specks.

        The size comes from the box beside the button rather than from the
        Label tab's slider, which is where it used to come from: a control on
        another tab quietly deciding what this reports is one nobody can see
        when they press it.

        No editing guard, unlike the two buttons above it: this reads and
        writes nothing, and asking someone to switch editing on to look at
        their data would be a lock on the wrong door.

        Stoppable, and freely so -- what a stopped sweep leaves is a smaller
        answer rather than a half-finished anything.
        """
        records = self.controller.scoped_records()
        if not records:
            self._flash("Tick a project, a lamella or a frame first")
            return
        cancellation = Cancellation()
        progressed = self.operations.worker.progressed
        self._run_operation(
            f"Reading the labels of {len(records)} image(s)",
            lambda: self.operations_controller.find_specks(
                records, at_most=size,
                on_progress=progressed.emit, cancellation=cancellation,
            ),
            lambda report: self._show_report("Find specks", report),
            cancellation=cancellation,
        )

    def _revert(self) -> None:
        """Put the selection's labels back to what they held before something.

        The same guard and the same selection every other operation in this
        panel uses: what a revert acts on is what is ticked, which is also what
        was converted. The three ways of naming what to go back to are one
        dialog, for the reason :class:`RevertDialog` gives.
        """
        records = self._editable_records()
        if records is None:
            return
        projects = self.controller.projects
        dialog = RevertDialog(
            self,
            f"{len(records)} image(s) in {self.controller.selection.describe()}",
            self.history_controller.savepoints_here(projects),
        )
        if dialog.exec_() != QDialog.DialogCode.Accepted:
            return
        undoing, savepoint, metadata = dialog.undoing, dialog.savepoint, dialog.metadata
        self._operate(
            "Revert labels",
            lambda allow, dry: self.history_controller.revert_labels(
                records,
                undoing=undoing or "conversion",
                savepoint=savepoint,
                projects=projects,
                metadata=metadata,
                allow_corrected=allow,
                dry_run=dry,
            ),
            rescan=metadata,
            notes=self._scope_notes(),
        )

    def _measure_archive(self) -> None:
        """Say what the kept label versions cost, once per scan.

        Here rather than in the scan: measuring walks every project's archive
        and reads every log, which is a cost worth paying on the tab that can
        act on the number and not worth paying to open a collection. Somebody
        who never opens this tab never pays it.

        Off the interface thread, because on a collection of a hundred
        projects it is seconds; the line stays as it was until the answer
        arrives, rather than flickering through a placeholder.
        """
        if self._archive_measured or not self.controller.projects:
            return
        self._archive_measured = True
        projects = self.controller.projects
        if not self.history_controller.archive_exists(projects):
            # One stat per project rather than a walk of each: a collection
            # that has never replaced a label has nothing to measure, and a
            # dialog flashing up to say so is worse than saying nothing.
            self.metadata_panel.set_archive("")
            return
        self._run_operation(
            "Archive",
            lambda: self.history_controller.measure_archive(projects),
            lambda size: self.metadata_panel.set_archive(size.describe()),
        )

    def _restore_savepoint(self) -> None:
        """Put every label under the folder back to a savepoint.

        A savepoint is taken over the whole collection from this tab, so this
        is where putting the whole collection back belongs -- reaching Curate
        and ticking everything first was the asymmetry this closes. Curate's
        Revert keeps the same choice for a selection.
        """
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing before restoring a savepoint.",
            )
            return
        projects = self.controller.projects
        if not projects:
            self._flash("Scan a folder first")
            return
        savepoints = self.history_controller.savepoints_here(projects)
        if not savepoints:
            QMessageBox.information(
                self, "Restore savepoint",
                "No savepoint has been created yet. Use Savepoint... to save the state of all "
                "labels first.",
            )
            return

        records = [
            record for project in projects for record in project.image_records
        ]
        dialog = RevertDialog(
            self,
            f"{len(records)} image(s) in all {len(projects)} project(s)",
            savepoints,
            operations=False,
        )
        if dialog.exec_() != QDialog.DialogCode.Accepted:
            return
        savepoint, metadata = dialog.savepoint, dialog.metadata
        self._operate(
            "Restore savepoint",
            lambda allow, dry: self.history_controller.revert_labels(
                records,
                savepoint=savepoint,
                projects=projects,
                metadata=metadata,
                allow_corrected=allow,
                dry_run=dry,
            ),
            rescan=metadata,
        )

    def _reclaim_archive(self) -> None:
        """Remove the kept label versions no savepoint needs.

        The whole collection, the command line being per project: what pruning
        keeps is decided by savepoints, which are collection-wide, so asking
        one project at a time asks one question repeatedly.
        """
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing before freeing archive space.",
            )
            return
        projects = self.controller.projects
        if not projects:
            self._flash("Scan a folder first")
            return
        self._operate(
            "Reclaim space",
            lambda allow, dry: self.history_controller.reclaim_archive(
                projects, dry_run=dry
            ),
        )

    def _take_savepoint(self) -> None:
        """Name the state of the whole collection.

        The scan rather than the selection, for the reason
        :meth:`OperationsController.take_savepoint` gives. Behind the same
        editing guard as recompressing: both write, and both act on everything
        under the folder.
        """
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing before creating a savepoint.",
            )
            return
        projects = self.controller.projects
        if not projects:
            self._flash("Scan a folder first")
            return

        dialog = SavepointDialog(self, len(projects))
        if dialog.exec_() != QDialog.DialogCode.Accepted:
            return
        title, note, gap = dialog.title, dialog.description, dialog.closing_gap
        self._operate(
            "Savepoint",
            lambda allow, dry: self.history_controller.take_savepoint(
                projects, title, description=note, close_the_gap=gap, dry_run=dry
            ),
        )

    def _compress_labels(self) -> None:
        """Recompress every label under the scanned folder.

        The whole collection rather than a selection: recompression preserves
        the pixels exactly, so there is no label it should be withheld from,
        and choosing which ones to shrink is not a decision worth a control.
        The corrected-label confirmation still applies -- rewriting the file is
        what needs the yes, not changing what is in it.
        """
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing before compressing labels.",
            )
            return
        records = [
            record
            for project in self.controller.projects
            for record in project.image_records
        ]
        if not records:
            self._flash("Scan a folder first")
            return
        # Only the applying run counts: the preview reads the file sizes and
        # answers at once, where re-encoding a whole collection is minutes with
        # nothing else on screen to say it is working.
        cancellation = Cancellation()
        progressed = self.operations.worker.progressed
        self._operate(
            "Compress labels",
            lambda allow, dry: self.operations_controller.compress_labels(
                records, allow_corrected=allow, dry_run=dry,
                on_progress=None if dry else progressed.emit,
                cancellation=None if dry else cancellation,
            ),
            cancellation=cancellation,
        )

    def _inspect_layout(self) -> None:
        """Report how the scanned directory is interpreted. Read-only."""
        root = self.controller.start_path
        if root is None:
            self._flash("Scan a folder first")
            return
        self._run_stoppable(
            lambda progress, cancellation: self.operations_controller.survey(
                root, on_progress=progress, cancellation=cancellation
            ),
            lambda report: self._show_report("Inspect layout", report),
            "Inspect layout",
        )

    def _add_projects(self) -> None:
        """Ask which directories become projects, then write their files.

        The dialog *is* the preview: every row already says what it would
        become, so this goes straight to writing rather than through the
        confirm-a-report pair the other operations use. Showing the same
        information a second time would not be read.
        """
        root = self.controller.start_path
        if root is None:
            self._flash("Scan a folder first")
            return
        if not self.metadata_controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing before writing metadata files.",
            )
            return
        dialog = AddProjectsDialog(self, self.operations_controller, root)
        if not dialog.exec_():
            return
        chosen = dialog.chosen()
        if not chosen:
            return
        # Rescanned afterwards: making an unindexed dataset into a project is
        # the point of this, and only a scan discovers the projects it created.
        self._run_operation(
            "Add projects",
            lambda: self.operations_controller.create_projects(chosen, dry_run=False),
            lambda report: self._operation_applied("Add projects", report, True),
        )

    def _check_csv(self, path: str) -> None:
        """Check that every path an exported CSV records still exists."""
        root = self.controller.start_path
        if root is None:
            self._flash("Scan a folder first")
            return
        self._run_operation(
            "Check exported CSV",
            lambda: self.operations_controller.check_csv(Path(path), root),
            lambda report: self._show_report("Check exported CSV", report),
        )

    def _measure_labels(self) -> None:
        """Measure every label off the interface thread, then redraw.

        The first thing in the Export tab slow enough to need the worker: it
        opens each label and labels its components, tens of milliseconds each
        against under a millisecond for a header. Long enough that the dialog
        has to say it is advancing rather than only that it is busy.
        """
        labels = self.export_controller.measurable_labels()
        progressed = self.operations.worker.progressed
        self._run_operation(
            f"Measuring {labels} label(s)",
            lambda: self.export_controller.load_measurements(progressed.emit),
            lambda measured: self._labels_measured(measured),
        )

    def _labels_measured(self, measured: int) -> None:
        """Redraw with the measurements the worker read."""
        logger.info("Measured labels for %d image(s)", measured)
        self.export_panel.refresh()

    def _with_labels(self, then: Callable[[], None]) -> None:
        """Read the labels this selection needs, then do the thing needing them.

        An export has to know what is inside each label -- a blank mask is
        never paired with an image -- and that is the one slow read in this
        tab. Doing it here rather than inside the write keeps it off the
        interface thread and behind a dialog that counts, and doing it only
        when it has not already happened means measuring first costs the
        export nothing.
        """
        if self.export_controller.labels_known():
            then()
            return
        labels = self.export_controller.unread_labels()
        progressed = self.operations.worker.progressed

        def read(_labels: int) -> None:
            self.export_panel.refresh()
            then()

        self._run_operation(
            f"Reading {labels} label(s)",
            lambda: self.export_controller.warm_labels(progressed.emit),
            read,
        )

    def _read_labels_for_view(self) -> None:
        """Read the labels the Export tab's filters or breakdown ask about.

        Ignored while an operation is running: that one's callback refreshes
        the tab, which asks again if anything is still unread.
        """
        if self._operation_running():
            return
        self._with_labels(lambda: None)

    def _export_training(self, directory: str) -> None:
        if not self.export_controller.selected_records():
            self._flash("Nothing matches these filters")
            return
        self._with_labels(lambda: self._write_training(directory))

    def _write_training(self, directory: str) -> None:
        if not self._confirm_dropping_corrected():
            return
        output = Path(directory)
        try:
            try:
                files = self.export_controller.export_training(output)
            except LeftoverFolds:
                # Replacing is no answer to files this export would not write.
                raise
            except FileExistsError as existing:
                if not self._confirm_overwriting(existing):
                    return
                files = self.export_controller.export_training(output, overwrite=True)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        if len(files) > 1:
            written = f"{len(files)} folds"
        else:
            written = ", ".join(path.name for path in files[0] if path)
        if len(files) == 1:
            # One split is what a run trains on; the Train tab offers it.
            self.training_controller.exported(
                manifest_path(output, self.export_controller.training.prefix)
            )
        self._flash(f"Wrote {written}, the images file and the manifest", 4000)
        logger.info("Exported training CSV to %s", directory)

    def _confirm_overwriting(self, existing: FileExistsError) -> bool:
        """Ask before replacing a training set written earlier under the same name.

        Cancel is the default: a split replaced by accident cannot be recovered,
        and nothing else records the seed that made it.
        """
        answer = QMessageBox.warning(
            self,
            "Replace the earlier training files?",
            f"{existing}.\n\nReplace them with this export?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_dropping_corrected(self) -> bool:
        """Ask before writing a training set that silently omits proofread work.

        The export skips an image whose label is missing or blank, which is
        right -- pairing an image with a blank mask teaches a model that the
        structure is background. What is wrong is doing it quietly: a corrected
        image being dropped means the proofreading or the label has gone
        missing, and the CSV that results is short with nothing saying so.
        """
        readiness = self.export_controller.unexportable_corrected()
        if not readiness:
            return True

        rows = [
            f"{record.project or '?'} / {record.lamella or '?'} / {record.image_path.name}"
            for record in readiness.records
        ]
        shown = "\n".join(rows[:MAX_LISTED_FILES])
        if len(rows) > MAX_LISTED_FILES:
            shown += f"\n... and {len(rows) - MAX_LISTED_FILES} more"

        answer = QMessageBox.warning(
            self,
            "Export without corrected images?",
            "These images are marked corrected but have an empty label, so the export leaves them "
            f"out ({readiness.summary()}):\n"
            "\n"
            f"{shown}\n"
            "\n"
            "Write the files anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _copy_files(self, directory: str) -> None:
        """Copy the matching images, labels and project files into *directory*.

        On the same preview-then-apply path as the operations that write into
        the collection, and stoppable for the same reason: this one writes
        gigabytes, and a copy that lands by rename leaves whole files wherever
        it is stopped.

        No rescan afterwards. Nothing in the scanned collection has changed --
        the copy is somewhere else, and refuses to be anywhere else.
        """
        if not self.export_controller.selected_records():
            self._flash("Nothing matches these filters")
            return

        cancellation = Cancellation()
        progressed = self.operations.worker.progressed
        self._operate(
            "Export images and labels",
            lambda allow, dry: self.operations_controller.copy_files(
                self.export_controller.selected_records(),
                Path(directory),
                projects=self.export_controller.projects,
                start_path=self.export_controller.start_path,
                scope=self.export_controller.describe_filters(),
                dry_run=dry,
                cancellation=None if dry else cancellation,
                on_progress=None if dry else progressed.emit,
            ),
            cancellation=cancellation,
        )

    def _merge_copy(self, directory: str) -> None:
        """Bring the work done in a copy back into this collection.

        The other half of the round trip the button above it starts. Behind the
        curation tab's editing guard rather than Add data's: what this writes
        is labels and their state, which is what that guard protects.

        A rescan follows, this being the one operation here that changes the
        collection the window is showing rather than writing somewhere else.
        """
        if not self.controller.editing_enabled:
            QMessageBox.information(
                self, "Editing is off",
                "Enable editing on the Curate tab before merging a copy back.",
            )
            return
        source = Path(directory)
        projects = self.controller.projects
        #: What has been decided so far, filled in by the chooser below and
        #: read by every later run. A decision changes what the whole
        #: comparison comes to, so it goes into the plan rather than onto one
        #: already made.
        decided: dict[str, str] = {}

        def ask(title: str, report: OperationReport, action: str) -> bool:
            if report.blocked != "conflicts":
                return self._confirm_report(title, report, action)
            if not self._confirm_report(title, report, action):
                return False
            dialog = ConflictDialog(
                self, report.conflicts, self.controller.config,
                advance=bool(
                    preferences.read(self.settings)["advance_after_deciding"]
                ),
            )
            if dialog.exec_() != QDialog.DialogCode.Accepted:
                return False
            decided.update(dialog.resolutions)
            return True

        self._operate(
            "Merge a copy back",
            lambda allow, dry: self.history_controller.merge_copy(
                source, projects, resolutions=dict(decided),
                allow_corrected=allow, dry_run=dry,
            ),
            rescan=True,
            confirm=ask,
        )

    def _export_image_metadata(self, path: str) -> None:
        if not self.export_controller.selected_records():
            self._flash("Nothing matches these filters")
            return
        try:
            # The save dialog has already asked before naming a file that exists.
            self.export_controller.export_image_metadata(Path(path), overwrite=True)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self._flash(f"Wrote {Path(path).name}", 4000)

    def _export_breakdown(self, path: str) -> None:
        """Write the counts behind the chart, as am-seg stats --output writes them."""
        panel = self.export_panel
        try:
            # The save dialog has already asked before naming a file that exists.
            self.export_controller.export_breakdown_csv(
                Path(path), panel.breakdown.currentData(), panel.cross.currentData() or None,
                overwrite=True,
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self._flash(f"Wrote {Path(path).name}", 4000)

    def _export_project_metadata(self, path: str) -> None:
        """Write one row per project, as am-seg export-projects-csv writes them."""
        try:
            # The save dialog has already asked before naming a file that exists.
            self.export_controller.export_project_metadata(Path(path), overwrite=True)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self._flash(f"Wrote {Path(path).name}", 4000)

