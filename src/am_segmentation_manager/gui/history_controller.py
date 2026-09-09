"""Savepoints, merges, reverts and the archive, as the window offers them.

The four things the record makes possible: naming a moment the collection can
be put back to, bringing home a copy worked on elsewhere, putting back the label
a conversion replaced, and reporting or reclaiming the kept versions the other
three draw on. Each previews, is confirmed, then applies -- the same shape
:mod:`.operations_controller` uses, and the same
:class:`~am_segmentation_manager.gui.report.OperationReport` back.

Split from that module because these read and write the log while the others do
not, and because a merge collects a decision per conflict, which nothing else
does. Qt-free: paths are reported against the scan root, an absolute path eight
directories deep being unreadable in a dialog.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..history import Recorder
from ..models import ImageRecord
from ..operations.init_history import writing_recorder
from ..reporting import format_bytes
from .attribution import Attribution
from .report import MAX_REPORTED_FILES, ConflictRow, OperationReport, shown_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveSize:
    """What the kept label versions cost, across a whole collection.

    Attributes:
        bytes_held: What the archive occupies now.
        bytes_reclaimable: What pruning would return. Zero while no savepoint
            has closed a round, which is the policy rather than a shortfall.
        versions: How many kept versions there are.
        unaccounted: Versions no log entry explains, which pruning can never
            remove because what they are cannot be established.
        savepoints: How many savepoints are here, which is what decides
            whether anything is reclaimable at all.
    """

    bytes_held: int = 0
    bytes_reclaimable: int = 0
    versions: int = 0
    unaccounted: int = 0
    savepoints: int = 0

    def describe(self) -> str:
        """Return the one line the Add data tab shows."""
        if not self.versions:
            return "Archive: nothing kept yet."
        held = (
            f"Archive: {format_bytes(self.bytes_held)} across "
            f"{self.versions} kept version(s)"
        )
        if self.bytes_reclaimable:
            line = f"{held}; {format_bytes(self.bytes_reclaimable)} could be reclaimed."
        elif not self.savepoints:
            line = f"{held}; nothing can be reclaimed until a savepoint closes a round."
        else:
            line = f"{held}; nothing to reclaim."
        if self.unaccounted:
            # A finding rather than a total: what these are cannot be
            # established, so nothing ever removes them.
            line += (
                f" {self.unaccounted} of them are not accounted for by any log "
                f"entry and are never removed."
            )
        return line


def _conflict_rows(plan) -> tuple[ConflictRow, ...]:
    """Return one row per undecided thing in *plan*, grouped as it is keyed.

    A label and the workflow state of the same group resolve under one key, so
    they become one row with a line each. Metadata is keyed per field and is a
    row of its own.
    """
    from ..operations.merge import CONFLICT

    rows: dict[str, ConflictRow] = {}
    for project in plan.projects:
        for item in project.labels:
            if item.outcome != CONFLICT:
                continue
            key = f"{project.key}/{item.target}"
            rows[key] = ConflictRow(
                key=key,
                title=key,
                details=("the label: both sides painted it differently",),
                corrected=item.corrected,
                ours=item.our_path,
                theirs=item.their_path,
                image=item.our_image,
            )
        for item in project.states:
            if item.outcome != CONFLICT:
                continue
            key = f"{project.key}/{item.target}"
            line = f"the state: {item.ours} here, {item.theirs} in the copy"
            existing = rows.get(key)
            rows[key] = (
                replace(existing, details=(*existing.details, line))
                if existing is not None
                else ConflictRow(key=key, title=key, details=(line,))
            )
        for item in project.metadata:
            if item.outcome != CONFLICT:
                continue
            key = f"{project.key}:{item.field}"
            rows[key] = ConflictRow(
                key=key,
                title=f"{project.key} — {item.field}",
                details=(f"{item.ours!r} here, {item.theirs!r} in the copy",),
            )
    return tuple(rows[key] for key in sorted(rows))


class HistoryController:
    """What the window asks about the record, and what it asks it to do.

    Holds only where the scan root is, so reports can name paths against it,
    and who changes are recorded as. The projects to act on arrive with each
    call: unlike a scan, a savepoint or a merge names its own scope.
    """

    def __init__(self, attribution: Attribution | None = None) -> None:
        """Create a controller recording through *attribution*."""
        self.start_path: Path | None = None
        #: Who changes are recorded as, shared with every other view.
        self.attribution = attribution if attribution is not None else Attribution()

    @property
    def recorder(self) -> Recorder:
        """Where this view's changes are recorded.

        Shared with every other view, so a name adopted in Settings reaches
        this one without anybody remembering to pass it on.
        """
        return self.attribution.recorder

    def set_start_path(self, start_path: Path | None) -> None:
        """Record the directory reports are written relative to."""
        self.start_path = Path(start_path) if start_path is not None else None

    def _relative(self, path: Path | None) -> str:
        """Return *path* as a report should show it."""
        return shown_path(path, self.start_path)

    def take_savepoint(
        self,
        projects: Sequence,
        title: str,
        *,
        description: str = "",
        close_the_gap: bool = False,
        dry_run: bool = False,
    ) -> OperationReport:
        """Name the state of the whole collection, previewing before it writes.

        A savepoint marks a round of model improvement, which is over the whole
        dataset a model was trained on -- so this takes the scan rather than
        what is ticked in Curate, and there is nothing to narrow it to.

        **It is a write, so it comes through the same door as every other**: a
        project keeping no history joins the record here and is then named,
        rather than being silently left outside the savepoint it looks like it
        is part of.

        **Refused inside a copy.** An export holds part of a project while
        carrying that project's log whole, so a savepoint written there would
        name a state for labels the copy never had.

        The preview carries the two things only a person can decide about, and
        that the command line prints as warnings: the projects whose record
        does not reach back far enough to join, and the labels only the next
        conversion would make unreachable.

        Args:
            projects: Every project in the scan.
            title: What the savepoint is called. Required.
            description: Why it was taken.
            close_the_gap: Keep now the versions that only the next conversion
                would destroy.
            dry_run: Report what it would write without writing it.
        """
        from ..operations.export_record import read_export_record
        from ..operations.init_history import plan_baseline
        from ..operations.savepoint import (
            SavepointRefused,
            apply_savepoint,
            plan_savepoint,
        )

        projects = list(projects)
        if not projects:
            return OperationReport("Nothing has been scanned yet.")
        root = self.start_path
        if root is not None and read_export_record(root) is not None:
            return OperationReport(
                f"{root.name} is a copy, and a copy can hold part of a project "
                f"while carrying that project's whole log. A savepoint here "
                f"would claim a state for labels this copy never had. Take it "
                f"in the collection it came from.",
            )

        if dry_run:
            # Reading the logs and no labels, so a preview stays a preview.
            joining = plan_baseline(projects)
            context = []
            if joining.log_count:
                context.append(
                    f"{joining.log_count} project(s) keep no history yet. They "
                    f"join the record first, recording what "
                    f"{joining.entry_count} label(s) hold now."
                )
            return OperationReport(
                f"Would name the state of {len(projects)} project(s) as "
                f"{title!r}.",
                tuple([*context, *self._savepoint_preview(projects, title)]),
                len(projects),
            )

        batch = writing_recorder(projects, self.recorder)
        try:
            plan = plan_savepoint(projects, title, description=description)
        except SavepointRefused as error:
            return OperationReport(str(error))

        report = apply_savepoint(plan, recorder=batch, close_the_gap=close_the_gap)
        labels = sum(len(states) for states in plan.states.values())
        headline = (
            f"Named {report.entries_written} project(s) as {title!r}, covering "
            f"{labels} label(s)."
        )
        if report.versions_kept:
            headline += (
                f" Kept {len(report.versions_kept)} version(s) the next "
                f"conversion would have destroyed."
            )
        return OperationReport(
            headline,
            tuple(self._savepoint_notes(plan)),
            applied=True,
            count=report.entries_written,
        )

    @staticmethod
    def _savepoint_preview(projects: Sequence, title: str) -> tuple[str, ...]:
        """Return what a savepoint would cover, where enough is recorded to say.

        A collection that records nothing yet cannot be planned against, and
        saying so beats a count of zero that reads as "this would do nothing".
        """
        from ..operations.savepoint import SavepointRefused, plan_savepoint

        try:
            plan = plan_savepoint(projects, title)
        except SavepointRefused:
            return ("What each project holds is counted once it is in the record.",)
        return tuple([*HistoryController._savepoint_notes(plan), *plan.describe()])

    @staticmethod
    def _savepoint_notes(plan) -> list[str]:
        """Return the decisions a savepoint plan raises, as lines."""
        notes = []
        if plan.too_late:
            notes.append(
                f"{len(plan.too_late)} project(s) cannot join: their record "
                f"begins after this savepoint was taken."
            )
        if plan.at_risk:
            notes.append(
                f"{len(plan.at_risk)} label(s) hold untouched model output, "
                f"which the archive does not keep. Tick 'Keep what a conversion "
                f"would destroy' to copy them in now."
            )
        return notes

    def merge_copy(
        self,
        source: Path,
        projects: Sequence = (),
        *,
        resolutions: Mapping[str, str] | None = None,
        allow_corrected: bool = False,
        dry_run: bool = False,
    ) -> OperationReport:
        """Bring a worked-on copy back into the collection it came from.

        The other half of the round trip *Export images and labels* starts, and
        the reason the two sit together: a copy goes out, work happens in it,
        and this brings that work home three-way against the history the two
        logs share.

        **A conflict stops the whole merge**, so this reports how many stand
        and writes nothing. Deciding them is by name, which the command line
        does with ``--resolve``; applying part of a merge would make what
        happened depend on what somebody had not got round to yet.

        Every refusal here carries no count and cannot be confirmed past, as
        an export's does: what they refuse is not the user's to overrule.

        Both trees are read on every run, and the preview and the apply are two
        runs -- which is what makes the apply describe the collection as it is
        now rather than as it was when the dialog opened.

        Args:
            source: The copy, holding the ``AM_export.yaml`` an export wrote.
            projects: The scan, so the destination joins the record before
                anything is written into it.
            resolutions: Conflicts already decided, keyed as
                :func:`~..operations.merge.plan_merge` keys them. Handed to the
                plan rather than applied to one already made: a decision
                changes what the whole comparison comes to, and re-planning is
                also what makes the merge describe the collection as it is now.
            allow_corrected: Confirm taking work over labels already proofread.
            dry_run: Report what it would do without doing it.
        """
        from ..operations.label_save import StaleLabels
        from ..operations.merge import (
            CONVERGED,
            KEEP_OURS,
            TAKE_THEIRS,
            MergeRefused,
            apply_merge,
            plan_merge,
        )

        destination = self.start_path
        if destination is None:
            return OperationReport("Nothing has been scanned yet.")
        try:
            plan = plan_merge(Path(source), destination, resolutions=resolutions)
        except MergeRefused as error:
            return OperationReport(str(error))

        taking = plan.counted(TAKE_THEIRS)
        incoming = sum(len(project.incoming) for project in plan.projects)
        context = [
            f"{taking} to take from the copy, {plan.counted(KEEP_OURS)} already "
            f"here, {plan.counted(CONVERGED)} arrived at the same thing.",
        ]
        if plan.conflicts:
            rows = _conflict_rows(plan)
            return OperationReport(
                f"{len(rows)} thing(s) were changed on both sides since the "
                f"copy was taken, so nothing can be merged until each is "
                f"decided. Both versions stay in the archive either way.",
                tuple(
                    f"{row.title}: {'; '.join(row.details)}" for row in rows
                )[:MAX_REPORTED_FILES],
                blocked="conflicts",
                conflicts=rows,
            )
        if not taking and not incoming:
            return OperationReport(
                f"Nothing to merge: {Path(source).name} holds no work this "
                f"collection does not already have."
            )
        if plan.corrected and not allow_corrected:
            return OperationReport(
                f"{len(plan.corrected)} piece(s) of proofread work here would "
                f"be written over or unmarked. The version here is kept either "
                f"way, so this can be undone -- but not unasked.",
                tuple(
                    getattr(item, "target", getattr(item, "field", ""))
                    for item in plan.corrected
                )[:MAX_REPORTED_FILES],
                taking,
                blocked="corrected",
            )
        if dry_run:
            return OperationReport(
                f"Would merge {taking} change(s) from {Path(source).name}, and "
                f"add {incoming} history entr(ies).",
                tuple([*context, *plan.describe()])[:MAX_REPORTED_FILES],
                taking,
            )

        batch = writing_recorder(projects, self.recorder)
        try:
            report = apply_merge(plan, recorder=batch, allow_corrected=allow_corrected)
        except (MergeRefused, PermissionError, StaleLabels) as error:
            return OperationReport(f"Nothing was merged: {error}")

        return OperationReport(
            f"Merged {report.labels_taken} label(s), {report.states_taken} "
            f"state(s) and {report.fields_taken} metadata field(s); kept "
            f"{len(report.archived)} replaced version(s) and added "
            f"{report.entries_added} history entr(ies).",
            applied=True,
            count=report.labels_taken + report.states_taken + report.fields_taken,
        )

    def revert_labels(
        self,
        records: Iterable[ImageRecord],
        *,
        undoing: str = "conversion",
        savepoint: str = "",
        projects: Sequence = (),
        allow_corrected: bool = False,
        dry_run: bool = False,
    ) -> OperationReport:
        """Put the selection's labels back to what they held before something.

        Three ways of naming what to go back to, and one operation behind all
        of them: undo a conversion, undo a merge, or return to a savepoint.
        They differ only in which recorded version each label is put back to,
        so splitting them across separate controls would make the window harder
        to learn than the command it replaces.

        **A label edited since is left alone and reported.** Putting one back
        anyway discards that editing, which is not a thing to offer beside a
        checkbox; the command line keeps `--allow-superseded` for it.

        Args:
            records: What the selection reaches.
            undoing: A key of :data:`~..operations.revert.UNDOABLE`, used when
                *savepoint* is empty.
            savepoint: A savepoint's id or title to return to instead.
            projects: The scan, needed only to find *savepoint* by name.
            allow_corrected: Confirm putting proofread work back.
            dry_run: Report what it would do without doing it.
        """
        from ..operations.revert import apply_revert, plan_revert, plan_revert_to
        from ..operations.savepoint import (
            SavepointRefused,
            find_savepoint,
        )

        records = list(records)
        if not records:
            return OperationReport("Nothing selected.")
        try:
            if savepoint:
                point = find_savepoint(list(projects), savepoint)
                plan = plan_revert_to(records, point)
                naming = f"the state named {point.title or point.id!r}"
            else:
                plan = plan_revert(records, undoing)
                naming = f"before the last {undoing}"
        except (SavepointRefused, ValueError) as error:
            return OperationReport(str(error))

        if not plan.revertible:
            return OperationReport(
                f"Nothing to put back: none of these {len(plan.items)} label(s) "
                f"can reach {naming}.",
                tuple(f"{item.target}: {item.blocked}" for item in plan.blocked)[
                    :MAX_REPORTED_FILES
                ],
            )

        left = (
            f", leaving {len(plan.blocked)} alone" if plan.blocked else ""
        )
        if plan.corrected and not allow_corrected:
            # The refusal itself lives in apply_revert and still fires; asking
            # here is what lets the window say how much is at stake first.
            return OperationReport(
                f"{len(plan.corrected)} of the {len(plan.revertible)} label(s) "
                f"this would put back to {naming} are proofread work.",
                tuple(item.target for item in plan.corrected)[:MAX_REPORTED_FILES],
                len(plan.revertible),
                blocked="corrected",
            )
        if dry_run:
            return OperationReport(
                f"Would put {len(plan.revertible)} label(s) back to {naming}"
                f"{left}. What is there now is kept, so this can be undone.",
                tuple(plan.describe())[:MAX_REPORTED_FILES],
                len(plan.revertible),
            )

        report = apply_revert(
            plan,
            recorder=writing_recorder(projects, self.recorder),
            allow_corrected=allow_corrected,
        )
        lines = tuple(
            f"{self._relative(path)}: {why}" for path, why in report.failed
        )[:MAX_REPORTED_FILES]
        headline = f"Put {len(report.restored)} label(s) back to {naming}{left}."
        if report.failed:
            headline += f" {len(report.failed)} could not be written."
        return OperationReport(
            headline, lines, applied=True, count=len(report.restored)
        )

    @staticmethod
    def savepoints_here(projects: Sequence) -> Sequence:
        """Return the savepoints over *projects*, newest first.

        Newest first because that is the order somebody reaches for a round in;
        the listing command orders them oldest first, which reads better as a
        history and worse as a chooser.
        """
        from ..operations.savepoint import list_savepoints

        return list(reversed(list_savepoints(list(projects))))

    @staticmethod
    def archive_exists(projects: Sequence) -> bool:
        """Return whether anything here keeps replaced label versions.

        One stat per project, so the window can decide whether measuring is
        worth a worker and a dialog. A collection that has never replaced a
        label has nothing to walk, and that is the ordinary case.
        """
        from ..operations.archive import archive_root

        return any(archive_root(project.path).is_dir() for project in projects)

    def measure_archive(self, projects: Sequence) -> ArchiveSize:
        """Return what the archive holds and what pruning would return.

        **Never called from a scan.** It walks every project's archive
        directory and reads every log, which on a collection of a hundred
        projects is the sort of cost that makes opening one feel slow -- for a
        number wanted a few times a year. The window asks for it once the tab
        that can act on it is open.

        Both numbers, because they answer different questions: what this costs,
        and what could be had back. The second is zero until a savepoint closes
        a round, which is the rule rather than a disappointment.
        """
        from ..operations.archive import archived_versions, plan_prune, total_size
        from ..operations.savepoint import list_savepoints, open_round, reached_versions

        projects = list(projects)
        savepoints = list_savepoints(projects)
        needed = reached_versions(savepoints)

        held = freed = versions = unaccounted = 0
        for project in projects:
            kept = archived_versions(project.path)
            if not kept:
                continue
            versions += len(kept)
            held += total_size(kept)
            unaccounted += sum(1 for version in kept if version.event is None)
            plan = plan_prune(
                project.path,
                reached=needed.get(project.path.resolve(), set()),
                open_round=open_round(project.path),
            )
            freed += plan.bytes_freed
        return ArchiveSize(
            bytes_held=held,
            bytes_reclaimable=freed,
            versions=versions,
            unaccounted=unaccounted,
            savepoints=len(savepoints),
        )

    def reclaim_archive(
        self, projects: Sequence, *, dry_run: bool = False
    ) -> OperationReport:
        """Remove the kept label versions no savepoint needs.

        Over the whole collection, the command line being per project: the
        savepoints that decide what is kept are collection-wide, so asking one
        project at a time asks the same question repeatedly.

        **The history is left alone.** What was recorded still says what
        happened; only the pixels behind it go.
        """
        from ..operations.archive import apply_prune, plan_prune
        from ..operations.savepoint import list_savepoints, open_round, reached_versions

        projects = list(projects)
        if not projects:
            return OperationReport("Nothing has been scanned yet.")

        savepoints = list_savepoints(projects)
        needed = reached_versions(savepoints)
        plans = [
            (
                project,
                plan_prune(
                    project.path,
                    reached=needed.get(project.path.resolve(), set()),
                    open_round=open_round(project.path),
                ),
            )
            for project in projects
        ]
        removable = [(project, plan) for project, plan in plans if plan.remove]
        freed = sum(plan.bytes_freed for _, plan in plans)
        kept = sum(plan.kept for _, plan in plans)

        if not removable:
            return OperationReport(
                f"Nothing to reclaim: all {kept} kept version(s) are needed."
                if savepoints else
                f"Nothing to reclaim yet. With no savepoint taken, the whole "
                f"archive is one round still open and all {kept} kept "
                f"version(s) stay. Naming a state is what closes a round."
            )

        lines = tuple(
            f"{project.key}/{version.target}: {version.path.name} "
            f"({format_bytes(version.size)})"
            for project, plan in removable
            for version in plan.remove
        )[:MAX_REPORTED_FILES]
        count = sum(len(plan.remove) for _, plan in removable)
        if dry_run:
            return OperationReport(
                f"Would remove {count} kept version(s), freeing "
                f"{format_bytes(freed)} and keeping {kept}. The history is "
                f"unchanged: what was recorded still says what happened.",
                lines,
                count,
            )

        removed = sum(apply_prune(plan) for _, plan in removable)
        return OperationReport(
            f"Removed {removed} kept version(s), freeing {format_bytes(freed)}.",
            applied=True,
            count=removed,
        )
