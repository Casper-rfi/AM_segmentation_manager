"""The window's side of the Label tab: which lamella, and saving what was drawn.

Proofreading follows the lamella being curated rather than keeping a list of its
own, so moving between lamellae is the window's question; so is saving, the one
action that writes a person's own work and so asks before discarding anything.
The drawing is :mod:`.label_panel` and what a save would do is
:mod:`am_segmentation_manager.gui.label_controller`; this is a mixin on the
window because each of these reaches both tabs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QDialog, QMessageBox

from ...state import StateFileError
from .save_dialog import SAVE_TITLE, SaveLabelsDialog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..label_controller import SavedLabels
    from .window_protocol import MixinBase as _Window
else:
    _Window = object

logger = logging.getLogger(__name__)

#: Names the restore in every dialog it opens.
RESTORE_TITLE = "Restore an earlier version"


class LabelTab(_Window):
    """Choosing what the Label tab looks at, and writing what it changed."""

    def _point_label_tab(self) -> None:
        """Show the lamella the curation view is on, and draw it.

        The Label tab has no lamella list of its own: proofreading follows the
        lamella being curated, so there is one cursor over the collection rather
        than two that can disagree. Where that would throw away unsaved edits,
        the tab stays where it is instead.
        """
        wanted = self.controller.lamella
        current = self.label_controller.lamella
        changing = wanted is not None and (
            current is None or wanted.path != current.path
        )
        if changing and not self._may_discard_label_edits():
            self.label_panel.refresh()
            return
        self.label_controller.set_lamella(wanted)
        self.label_panel.refresh()

    def _step_label_lamella(self, delta: int) -> None:
        """Move both views to another lamella, from the Label tab."""
        if not self._may_discard_label_edits():
            return
        self._step_lamella(delta)
        self._point_label_tab()

    def _may_discard_label_edits(self) -> bool:
        """Ask before throwing away unsaved label edits; return whether to go on.

        Deliberately not offered as "save these first": saving runs off the
        interface thread and asks its own questions, and threading a navigation
        through the far side of that is how a half-finished save ends up moving
        files under a lamella nobody is looking at any more.
        """
        pending = self.label_controller.pending_count()
        if not pending:
            return True
        lamella = self.label_controller.lamella
        name = lamella.display_name if lamella is not None else "this lamella"
        answer = QMessageBox.warning(
            self,
            "Unsaved label edits",
            f"{pending} label{'s' if pending != 1 else ''} edited in {name} "
            f"{'have' if pending != 1 else 'has'} not been saved.\n\n"
            "Save them with Ctrl+S first, or discard them to continue.",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Discard:
            return False
        self.label_controller.discard_edits()
        return True


    def _may_change_families(self, project) -> bool:
        """Ask before a families change where label edits wait in *project*.

        A move can change which label a condition reads, and an edit waiting
        under the old one would be saved where that condition no longer looks.
        The question is the one a change of lamella asks.
        """
        lamella = self.label_controller.lamella
        if lamella is None or not self.label_controller.pending_count():
            return True
        root = project.path.resolve()
        if root != lamella.path.resolve() and root not in lamella.path.resolve().parents:
            return True
        return self._may_discard_label_edits()

    def restore_label_version(self) -> None:
        """Put the chosen earlier version of this label back, having asked.

        A write, so the window asks rather than the panel acting: what is on
        disk now is kept first, which makes this reversible, and the reviewer
        is told that before deciding rather than after.
        """
        versions = self.label_panel.versions_controller
        chosen = versions.selected
        if chosen is None:
            self._flash("No earlier version is chosen")
            return
        if self.label_controller.pending_count():
            QMessageBox.warning(
                self,
                RESTORE_TITLE,
                "Save or discard the edits waiting first: restoring writes over "
                "this label, and unsaved strokes would be lost with it.",
            )
            return

        corrected = self.label_controller.is_reviewed()
        if QMessageBox.question(
            self,
            RESTORE_TITLE,
            f"Put back the version from {chosen.when or 'an unrecorded time'}"
            f", saved by {chosen.who}?\n\n"
            "What is on disk now is kept first, so this can itself be undone."
            + (
                "\n\nThis label is marked corrected. Restoring writes over "
                "the corrected version."
                if corrected else ""
            ),
        ) != QMessageBox.StandardButton.Yes:
            return

        try:
            versions.restore_selected(corrected=corrected, confirmed=True)
        except (ValueError, PermissionError, RuntimeError) as error:
            QMessageBox.critical(self, RESTORE_TITLE, str(error))
            return
        self.label_controller.discard_edits()
        self.label_panel.refresh()
        self._flash("Earlier version restored")

    def save_labels(self) -> None:
        """Write this lamella's edited labels, after asking what to do with them."""
        if not self.label_controller.pending_count():
            if not self.label_controller.checked_keys():
                self._flash("No label edits to save")
                return
            if not self.controller.editing_enabled:
                # Checked and unchanged, so marking it is all a save could do.
                self._flash(
                    "Nothing to write. To mark the checked labels corrected, "
                    "switch on Enable editing in the Curate tab", 6000,
                )
                return

        try:
            plan = self.label_controller.save_plan()
        except StateFileError as error:
            QMessageBox.critical(
                self,
                SAVE_TITLE,
                f"Nothing has been saved: whether these labels are corrected "
                f"cannot be told, because a state file will not read.\n\n{error}",
            )
            return
        if plan.stale:
            QMessageBox.critical(
                self,
                SAVE_TITLE,
                f"{len(plan.stale)} label(s) changed on disk since you opened "
                f"them, so nothing has been saved:\n\n"
                + "\n".join(item.destination.name for item in plan.stale)
                + "\n\nRe-scan to see what is there now. Re-opening the "
                "lamella will discard the edits made here.",
            )
            return

        choice = self._ask_how_to_save(plan)
        if choice is None:
            return
        mark_corrected = choice
        allow_corrected = bool(plan.corrected)
        self._run_operation(
            SAVE_TITLE,
            lambda: self.label_controller.save(
                mark_corrected=mark_corrected, allow_corrected=allow_corrected
            ),
            self._labels_saved,
        )

    def _ask_how_to_save(self, plan) -> bool | None:
        """Ask whether to mark the saved labels corrected; None cancels.

        Marking corrected is a state change, so it is offered only while editing
        is switched on -- the same guard every other state change is behind.
        With it switched off the dialog says nothing about what marking would
        move: describing an option that is not on offer is what made this ask
        three questions at once.
        """
        editing = self.controller.editing_enabled
        lines = [
            f"Save {plan.summary()} in this lamella?" if plan.count
            else "No label in this lamella was changed."
        ]
        spread = self._modality_note()
        if spread:
            lines.append(spread)
        if plan.corrected:
            lines.append(
                f"{len(plan.corrected)} of them replace a label already marked "
                "corrected."
            )
        approved = len(self.label_controller.approved_records()) if editing else 0
        if approved:
            lines.append(
                f"Marking corrected would cover {approved} image(s), and the "
                "labels they share."
            )
        unannotated = self.label_controller.unannotated_checked() if editing else 0
        if unannotated:
            lines.append(
                f"{unannotated} of the labels checked without a change hold no "
                "annotation. Marking them corrected says they need none."
            )
        unfinished = self.label_controller.unfinished_checklists()
        if unfinished:
            lines.append(
                f"{unfinished} of the labels have checklist steps not yet ticked."
            )

        note = "" if editing else (
            "To set their state to corrected as well, switch on Enable editing "
            "in the Curate tab."
        )
        dialog = SaveLabelsDialog(
            self, "\n\n".join(lines), note, offer_corrected=editing,
            writes=bool(plan.count),
        )
        if dialog.exec_() != QDialog.DialogCode.Accepted:
            return None
        return dialog.mark_corrected

    def _modality_note(self) -> str:
        """Return the line naming what is waiting per modality, or nothing.

        Only when more than one is involved. A save writes the whole lamella's
        edits, so the reviewer working through the SEM images has to be told
        that the FIB labels they touched an hour ago are going out too.
        """
        counts = self.label_controller.pending_modalities()
        if len(counts) < 2:
            return ""
        return "They are " + " and ".join(
            f"{count} {modality}" for modality, count in sorted(counts.items())
        ) + "."

    def _labels_saved(self, saved: SavedLabels) -> None:
        """Record what a save did and bring the window back in step.

        Refreshed rather than rescanned. A save is the one write whose whole
        effect is already in memory when it finishes: the labels it created are
        named on their records, and declaring a group corrected writes a file
        and updates those records without moving anything. Reading the
        collection again to learn what is already known cost twenty seconds of
        the action taken most often, and threw away the prepared frames of the
        lamella being proofread on the way.

        The Export tab's measurements are keyed by image, which the label
        states the save already forgot do not reach, so it is handed what was
        written as it is after any other operation.
        """
        headline = saved.headline
        self.state_panel.show_last_change(f"{SAVE_TITLE}: {headline}")
        logger.info("%s: %s", SAVE_TITLE, headline)
        self._flash(headline, 4000)
        self.export_controller.forget_labels(saved.written)
        self.refresh(reload_lists=True)
        self.label_panel.refresh()
        if saved.problem:
            QMessageBox.warning(
                self,
                SAVE_TITLE,
                f"The labels were saved, but not marked corrected:\n\n{saved.problem}",
            )

    def _schedule_label_loads(self) -> None:
        """Ask the Label tab's loader to prepare the frames around this one."""
        token = self.label_loader.next_token()
        requests = self.label_controller.pending_loads()
        if requests:
            self.label_loader.load_batch(requests, token)
        self._sweep_specks(token)

    def _sweep_specks(self, token: int | None = None) -> None:
        """Count the specks of frames nothing has counted yet.

        On the loader's thread and behind whatever frame preparation is queued
        there, which is the order that matters: the frame on screen is what the
        reviewer is waiting for and the marks are what they will look at next.

        Sharing the loader's token means moving to another lamella abandons a
        sweep in flight. That costs the label it was reading and nothing else,
        the sweep asking each time only for what is still uncounted.
        """
        labels = self.label_controller.uncounted_labels()
        if not labels:
            return
        if token is None:
            token = self.label_loader.next_token()
        self.label_loader.count_specks(
            labels, token, self.label_controller.speck_size
        )

    def _on_speck_counted(self, token: int, key: str, count: int) -> None:
        """Take one label's count from the sweep and redraw the marks."""
        if token != self.label_loader.token:
            return
        self.label_controller.record_speck_count(Path(key), count)
        self.label_panel.refresh_timeline()

    def _on_label_loaded(self, token: int, frame_index: int) -> None:
        """Redraw the canvas when the frame on screen finishes preparing.

        A frame whose label was rewritten while it was being read is discarded
        by the redraw, and asked for again here rather than left loading.
        """
        if token != self.label_loader.token:
            return
        if frame_index == self.label_controller.frame_index:
            self.label_panel.draw_current()
            if self.label_controller.cached_image() is None:
                self._schedule_label_loads()

