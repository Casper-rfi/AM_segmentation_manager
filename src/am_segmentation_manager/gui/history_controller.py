"""Savepoints, merges, reverts and the archive, as the window offers them.

What the record makes possible: naming a moment to put the collection back to,
bringing home a copy worked on elsewhere, putting back what a conversion
replaced, and reporting or reclaiming kept versions. Each previews, is confirmed,
then applies, returning an :class:`~am_segmentation_manager.gui.report.OperationReport`
as :mod:`.operations_controller` does; kept apart because only these read and
write the log, and a merge collects a decision per conflict. Holds no Qt; paths
are reported against the scan root.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..history import Recorder
from ..io.long_paths import PathTooLong
from ..models import ImageRecord
from ..operations.archive import (
    apply_prune,
    archive_root,
    archived_versions,
    plan_prune,
    total_size,
)
from ..operations.export_record import read_export_record
from ..operations.init_history import plan_baseline, writing_recorder
from ..operations.label_save import SaveStopped, StaleLabels
from ..operations.merge import (
    CONFLICT,
    CONVERGED,
    KEEP_OURS,
    TAKE_THEIRS,
    MergeRefused,
    apply_merge,
    field_key,
    label_key,
    plan_merge,
)
from ..operations.revert import (
    FAMILY_KEPT,
    RevertPlan,
    RevertReport,
    apply_revert,
    plan_revert,
    plan_revert_to,
)
from ..operations.savepoint import (
    SavepointRefused,
    apply_savepoint,
    find_savepoint,
    list_savepoints,
    open_round,
    plan_savepoint,
    reached_versions,
)
from ..reporting import format_bytes
from ..state import StateFileError
from .attribution import Attribution
from .report import ConflictRow, OperationReport, capped, shown_path

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
            return "Archive: empty."
        held = (
            f"Archive: {format_bytes(self.bytes_held)} in {self.versions} kept version(s)"
        )
        if self.bytes_reclaimable:
            line = f"{held}; {format_bytes(self.bytes_reclaimable)} can be freed."
        elif not self.savepoints:
            line = f"{held}; nothing can be freed until a savepoint is created."
        else:
            line = f"{held}; nothing to free."
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
    row of its own, as is each family correction.
    """

    rows: dict[str, ConflictRow] = {}
    for project in plan.projects:
        for entry in project.families:
            if entry.outcome != CONFLICT:
                continue
            key = field_key(project.key, entry.field)
            rows[key] = ConflictRow(
                key=key,
                title=f"{project.key} — {entry.field}",
                details=(entry.describe(),),
                project=project.key,
                metadata_field=entry.field,
                our_value=entry.ours,
                their_value=entry.theirs,
            )
        for item in project.labels:
            if item.outcome != CONFLICT:
                continue
            key = label_key(project.key, item.target)
            rows[key] = ConflictRow(
                key=key,
                title=key,
                details=("the label: both sides painted it differently",),
                corrected=item.corrected,
                ours=item.our_path,
                theirs=item.their_path,
                image=item.our_image,
                project=project.key,
                lamella=_lamella_of(item.record),
                record=item.record,
            )
        for item in project.states:
            if item.outcome != CONFLICT:
                continue
            key = label_key(project.key, item.target)
            line = f"the state: {item.ours} here, {item.theirs} in the copy"
            existing = rows.get(key)
            rows[key] = (
                replace(existing, details=(*existing.details, line))
                if existing is not None
                else ConflictRow(
                    key=key, title=key, details=(line,), project=project.key,
                    lamella=_lamella_of(item.record), record=item.record,
                )
            )
        for item in project.metadata:
            if item.outcome != CONFLICT:
                continue
            key = field_key(project.key, item.field)
            rows[key] = ConflictRow(
                key=key,
                title=f"{project.key} — {item.field}",
                details=(f"{item.ours!r} here, {item.theirs!r} in the copy",),
                project=project.key,
                metadata_field=item.field,
                our_value=item.ours,
                their_value=item.theirs,
            )
    return tuple(rows[key] for key in sorted(rows))


def _lamella_of(record) -> str:
    """Return the lamella *record* belongs to, or empty where there is none."""
    return (record.lamella or "") if record is not None else ""


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

        projects = list(projects)
        if not projects:
            return OperationReport("Nothing has been scanned yet.")
        root = self.start_path
        if root is not None and read_export_record(root) is not None:
            return OperationReport(
                f"{root.name} is an exported copy, which may hold only part of a project. Create "
                "the savepoint in the original collection.",
            )

        if dry_run:
            # Reading the logs and no labels, so a preview stays a preview.
            joining = plan_baseline(projects)
            context = []
            if joining.log_count:
                context.append(
                    f"{joining.log_count} project(s) have no history yet. They are added first, "
                    f"recording the current state of {joining.entry_count} label(s)."
                )
            return OperationReport(
                f"Would create savepoint {title!r} for {len(projects)} project(s).",
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
            f"Created savepoint {title!r} for {report.entries_written} project(s), covering "
            f"{labels} label(s)."
        )
        if report.versions_kept:
            headline += (
                f" Kept {len(report.versions_kept)} version(s) the next "
                f"conversion would have destroyed."
            )
        if report.versions_not_kept:
            headline += (
                f" {len(report.versions_not_kept)} label(s) were changed elsewhere "
                f"or could not be read, so were not kept."
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

        try:
            plan = plan_savepoint(projects, title)
        except SavepointRefused:
            return ("Label counts per project are shown once the project has a history.",)
        return tuple([*HistoryController._savepoint_notes(plan), *plan.describe()])

    @staticmethod
    def _savepoint_notes(plan) -> list[str]:
        """Return the decisions a savepoint plan raises, as lines."""
        notes = []
        if plan.too_late:
            notes.append(
                f"{len(plan.too_late)} project(s) cannot be included: their history starts after "
                "this savepoint was created."
            )
        if plan.at_risk:
            notes.append(
                f"{len(plan.at_risk)} label(s) are unedited model output, which the archive does "
                "not keep. Tick 'Also keep labels the model made' to keep them now."
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
            f"{taking} to take from the copy, {plan.counted(KEEP_OURS)} already here, "
            f"{plan.counted(CONVERGED)} identical on both sides.",
        ]
        if plan.conflicts:
            rows = _conflict_rows(plan)
            return OperationReport(
                f"{len(rows)} item(s) were changed on both sides since the copy was made. Decide "
                "each one before merging; both versions are kept in the archive.",
                capped(f"{row.title}: {'; '.join(row.details)}" for row in rows),
                blocked="conflicts",
                conflicts=rows,
            )
        if not taking and not incoming:
            return OperationReport(
                f"Nothing to merge: {Path(source).name} has no changes this collection does not "
                "already have."
            )
        if plan.corrected and not allow_corrected:
            return OperationReport(
                f"{len(plan.corrected)} corrected item(s) here would be overwritten or unmarked. "
                "The current versions are kept, so this can be undone, but it needs your "
                "confirmation.",
                capped(
                    getattr(item, "target", getattr(item, "field", ""))
                    for item in plan.corrected
                ),
                taking,
                blocked="corrected",
            )
        if dry_run:
            return OperationReport(
                f"Would merge {taking} change(s) from {Path(source).name} and add {incoming} "
                "history entr(ies).",
                capped([*context, *plan.describe()]),
                taking,
            )

        batch = writing_recorder(projects, self.recorder)
        try:
            report = apply_merge(plan, recorder=batch, allow_corrected=allow_corrected)
        except (MergeRefused, PermissionError, StaleLabels, StateFileError, PathTooLong) as error:
            return OperationReport(f"Nothing was merged: {error}")
        except SaveStopped as error:
            return OperationReport(
                f"The merge stopped partway; what it had not reached is unchanged: {error}"
            )

        return OperationReport(
            f"Merged {report.labels_taken} label(s), {report.states_taken} state(s) and "
            f"{report.fields_taken} metadata field(s). Kept {len(report.archived)} replaced "
            f"version(s) and added {report.entries_added} history entr(ies).",
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
        metadata: bool = False,
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
            savepoint: A savepoint's id or title to return to instead, the
                states of the groups restored with their labels.
            projects: The scan, needed only to find *savepoint* by name.
            metadata: With *savepoint*, also restore the project metadata.
            allow_corrected: Confirm putting proofread work back.
            dry_run: Report what it would do without doing it.
        """

        records = list(records)
        if not records:
            return OperationReport("Nothing selected.")
        try:
            if savepoint:
                point = find_savepoint(list(projects), savepoint)
                plan = plan_revert_to(records, point, metadata=metadata)
                naming = f"the state named {point.title or point.id!r}"
            else:
                plan = plan_revert(records, undoing)
                naming = f"before the last {undoing}"
        except (SavepointRefused, ValueError) as error:
            return OperationReport(str(error))

        if not plan.anything:
            return OperationReport(
                f"Nothing to restore: none of these {len(plan.items)} label(s) has a version from "
                f"{naming}.",
                capped(
                    [f"{item.target}: {item.blocked}" for item in plan.blocked]
                    + [f"{state.target}: {state.blocked}" for state in plan.states]
                    + [f"{named}: {FAMILY_KEPT}" for named in plan.families]
                ),
            )

        left = (
            f", leaving {len(plan.blocked)} alone" if plan.blocked else ""
        )
        corrected = plan.corrected_targets
        if corrected and not allow_corrected:
            # The refusal itself lives in apply_revert and still fires; asking
            # here is what lets the window say how much is at stake first.
            return OperationReport(
                f"{len(corrected)} of the label(s) to restore to {naming} are marked "
                "corrected.",
                capped(corrected),
                len(plan.revertible),
                blocked="corrected",
            )
        if dry_run:
            families = (
                f" {len(plan.families)} family correction(s) changed since are not restored."
                if plan.families else ""
            )
            return OperationReport(
                f"Would restore {_restoring(plan)} to {naming}{left}. The current label "
                f"versions are kept, so this can be undone.{families}",
                capped(plan.describe()),
                len(plan.revertible),
            )

        report = apply_revert(
            plan,
            recorder=writing_recorder(projects, self.recorder),
            allow_corrected=allow_corrected,
        )
        lines = capped(
            f"{self._relative(path)}: {why}" for path, why in report.failed
        )
        headline = f"Restored {_restored(report)} to {naming}{left}."
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

        return list(reversed(list_savepoints(list(projects))))

    @staticmethod
    def archive_exists(projects: Sequence) -> bool:
        """Return whether anything here keeps replaced label versions.

        One stat per project, so the window can decide whether measuring is
        worth a worker and a dialog. A collection that has never replaced a
        label has nothing to walk, and that is the ordinary case.
        """

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
                f"Nothing to free: all {kept} kept version(s) are still needed."
                if savepoints else
                f"Nothing to free yet. All {kept} kept version(s) are kept until a savepoint is "
                "created."
            )

        lines = capped(
            f"{project.key}/{version.target}: {version.path.name} "
            f"({format_bytes(version.size)})"
            for project, plan in removable
            for version in plan.remove
        )
        count = sum(len(plan.remove) for _, plan in removable)
        if dry_run:
            return OperationReport(
                f"Would remove {count} kept version(s), freeing {format_bytes(freed)} and keeping "
                f"{kept}. The history is not changed.",
                lines,
                count,
            )

        removed = sum(apply_prune(plan) for _, plan in removable)
        return OperationReport(
            f"Removed {removed} kept version(s), freeing {format_bytes(freed)}.",
            applied=True,
            count=removed,
        )


def _restoring(plan: RevertPlan) -> str:
    """Return what *plan* would restore, as a phrase: labels, then states and metadata."""
    return _phrase(len(plan.revertible), len(plan.restorable_states), len(plan.fields))


def _restored(report: RevertReport) -> str:
    """Return what *report* says was restored, as :func:`_restoring` phrases it."""
    return _phrase(len(report.restored), report.states, report.fields)


def _phrase(labels: int, states: int, projects: int) -> str:
    parts = [f"{labels} label(s)"]
    if states:
        parts.append(f"{states} state(s)")
    if projects:
        parts.append(f"the metadata of {projects} project(s)")
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
