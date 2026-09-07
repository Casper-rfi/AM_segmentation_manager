"""The batch operations the window offers, in one shape.

Predicting, converting predictions into labels, recompressing labels, copying a
selection out and naming new projects each return a different result type. Each
is offered the same way -- preview what it would do, confirm, then apply -- so
this turns them all into one :class:`OperationReport` and the window gets one
dialog rather than one per operation.

Qt-free, like the other controllers here. Paths are reported relative to the
scan root: an absolute path eight directories deep is unreadable in a dialog.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..discovery import DatasetCandidate
from ..history import Recorder
from ..io.path import make_path_relative
from ..models import ImageRecord
from ..operations.csv import find_matching_label
from ..operations.export_record import ExportRecord
from ..operations.init_history import writing_recorder
from ..operations.label_compression import compress_labels, existing_labels
from ..operations.label_measure import records_with_specks
from ..operations.predict import PredictionSource, plan_predictions
from ..operations.predict_process import ProcessOutcome, predict_out_of_process
from ..operations.predictions import (
    SKIP_DESCRIPTIONS,
    ConversionPlan,
    convert_record_predictions,
    count_convertible,
    plan_conversions,
)
from ..operations.selection import current_state
from ..progress import Cancellation, timed
from ..reporting import format_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperationReport:
    """What an operation did, or would do if it ran.

    Attributes:
        headline: The single line summarising the run.
        lines: One line per affected file or group, in the operation's order.
        count: How many things were affected, or would be. Zero means there is
            nothing to apply, which is what the window checks before asking.
        applied: False for a preview.
        blocked: Why the operation cannot run, or empty when it can. A blocked
            report carries no work; it is shown and dismissed.
        written: The files the run actually created, for an operation whose
            whole effect on the index is the paths it wrote. The window names
            them on the records that share them instead of reading the
            collection again; empty for a preview, and for a run whose effect
            is not expressible this way. **Only what was written**, never what
            was planned: a stopped run plans more than it creates, and a record
            claiming a prediction that is not there offers to convert nothing.
    """

    headline: str
    lines: tuple[str, ...] = ()
    count: int = 0
    applied: bool = False
    blocked: str = ""
    written: tuple[Path, ...] = ()

    def text(self) -> list[str]:
        """Return the report as displayable lines, headline first."""
        return [self.headline, *self.lines]


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


#: How many filenames a report lists before leaving the rest to the count. A
#: copy of a whole collection runs to thousands of files, and a dialog listing
#: them is one nobody reads to the end.
MAX_REPORTED_FILES = 20


#: What each kind of copied file is called in a report, in the order a reader
#: wants them: the data first, then what describes it.
COPIED_NOUNS: dict[str, str] = {
    "image": "image",
    "label": "label",
    "state": "state file",
    "metadata": "project file",
}


def _copy_context(plan) -> tuple[str, ...]:
    """Return what a copy would write, counted by what each file is."""
    kinds = Counter(item.kind for item in plan.to_copy)
    counted = ", ".join(
        f"{kinds[kind]} {noun}{'s' if kinds[kind] != 1 else ''}"
        for kind, noun in COPIED_NOUNS.items()
        if kinds.get(kind)
    )
    context = [counted] if counted else []
    if plan.already_there:
        context.append(f"{len(plan.already_there)} file(s) are already there")
    context.append(f"{format_bytes(plan.free_space)} free where it is going")
    return tuple(context)


#: The column headings of the speck report, and the least room each takes.
#: The report dialog is monospace, so columns line up if they are padded to a
#: width -- which is the whole reason to pad rather than to separate with
#: commas, a list of three numbers per row being unreadable at a glance.
SPECK_COLUMNS = (("Specks", 6), ("Labels", 6), ("Per label", 9))


def _speck_table(rows: Sequence[tuple[str, int, int]]) -> tuple[str, ...]:
    """Return the per-project speck counts as aligned columns under a heading.

    The heading carries the units so the rows do not have to: "49 in 3
    label(s)" repeated down a page is the same three words over and over, and
    the numbers are what is being compared.
    """
    if not rows:
        return ()
    name_width = max(len("Project"), max(len(name) for name, _, _ in rows))
    heading = f"{'Project':<{name_width}}  " + "  ".join(
        f"{title:>{width}}" for title, width in SPECK_COLUMNS
    )
    lines = [heading, "-" * len(heading)]
    for name, specks, labels in rows:
        widths = [width for _title, width in SPECK_COLUMNS]
        lines.append(
            f"{name:<{name_width}}  "
            f"{specks:>{widths[0]}}  {labels:>{widths[1]}}  "
            f"{specks / labels:>{widths[2]}.1f}"
        )
    return tuple(lines)


def _skip_summary(plan: ConversionPlan) -> tuple[str, ...]:
    """Return one line per reason records were passed over, largest first.

    A run that converts nothing is the case most in need of an explanation, and
    the three reasons need different actions: wait for the pipeline, re-run it
    without post-processing, or accept that the label is already there.
    """
    counted = [
        (plan.count_skipped(reason), description)
        for reason, description in SKIP_DESCRIPTIONS.items()
    ]
    return tuple(
        f"{count} image(s) skipped: {description}"
        for count, description in sorted(counted, reverse=True)
        if count
    )


class OperationsController:
    """Previews and runs the operations that change files on disk.

    Holds the scan root so reports can name files the way the rest of the window
    does, and nothing else: every method takes the records it acts on, so what
    an operation applies to is decided by the selection and never remembered
    here.
    """

    def __init__(self) -> None:
        self.start_path: Path | None = None
        #: Where this view's changes are recorded. A project with no log is
        #: left without one, so nothing is recorded until a collection opts in.
        self.recorder = Recorder.for_run()


    def set_start_path(self, start_path: Path | None) -> None:
        """Record the directory reports are written relative to."""
        self.start_path = Path(start_path) if start_path is not None else None

    def relative(self, path: Path) -> str:
        """Return *path* as it should be shown, against the scan root."""
        return self._relative(path)

    def _relative(self, path: Path | None) -> str:
        if path is None:
            return "(no label)"
        if self.start_path is None:
            return str(path)
        return make_path_relative(path, self.start_path)

    # -- converting predictions -------------------------------------------

    def describe_predictions(self, records: Iterable[ImageRecord]) -> str:
        """Return the line saying what converting this selection would find.

        Shown beside the button rather than reached by pressing it: "is there
        anything here to convert" is a question about the data, and answering it
        with a dialog makes the user open one to find out there was nothing.

        The reasons matter more than the number, because they need different
        actions -- run the prediction pipeline, re-run it without
        post-processing, or accept the label that is already there.
        """
        counts = count_convertible(records)
        if not counts.total:
            return ""
        if counts.ready:
            leading = f"{counts.ready} of {counts.total} image(s) ready to convert"
        elif counts.labelled:
            # Distinguished from "nothing to convert": there is work available
            # here, it just replaces something and has to be confirmed.
            leading = "Nothing new to convert"
        else:
            leading = "Nothing to convert"

        detail = [
            (counts.labelled, "with a replaceable label"),
            (counts.corrected, "proofread, never replaced"),
            (counts.post_processed, "with post-processed predictions only"),
            (counts.no_prediction, "with no prediction yet"),
        ]
        described = "; ".join(f"{count} {text}" for count, text in detail if count)
        return f"{leading}. {described}." if described else f"{leading}."

    def _refusal_summary(self, plan: ConversionPlan) -> tuple[str, ...]:
        """Return the lines naming proofread labels that stopped a conversion.

        Named individually rather than counted: this is the one outcome the user
        cannot unblock from the dialog, so it has to say which groups to set
        back to selected if they really are to be re-converted.
        """
        if not plan.refused:
            return ()
        return (
            f"{len(plan.refused)} image(s) refused: a proofread label is already "
            f"there. Set the group back to selected to re-convert it.",
        ) + tuple(f"  {self._relative(item.destination)}" for item in plan.refused)

    def convert_predictions(
        self,
        records: Iterable[ImageRecord],
        *,
        allow_overwrite: bool = False,
        dry_run: bool = False,
    ) -> OperationReport:
        """Turn each record's prediction into a label.

        An image that already has a label is held back until *allow_overwrite*
        is set, and one whose label is proofread is refused however it is set:
        conversion replaces annotation with a machine guess. Applying
        re-resolves the plan rather than reusing the preview's, so a file that
        appeared in between is still seen.
        """
        records = list(records)
        plan = plan_conversions(records, overwrite=allow_overwrite)
        context = self._refusal_summary(plan) + _skip_summary(plan)

        if plan.replaceable and not allow_overwrite:
            # Asked before applying, so the window can say how much is at stake
            # and which files. The refusal itself lives in plan_conversions.
            return OperationReport(
                f"{len(plan.replaceable)} of these image(s) already have a label. "
                f"Converting replaces the file.",
                tuple(
                    f"{self._relative(item.destination)}: {item.describe()}"
                    for item in plan.replaceable
                ) + context,
                len(plan.replaceable) + len(plan.conversions),
                blocked="existing",
            )

        lines = tuple(
            f"{self._relative(item.destination)}: {item.describe()}"
            for item in plan.conversions
        )
        if not plan.conversions:
            return OperationReport("Nothing to convert.", context)
        if dry_run:
            return OperationReport(
                f"Would write {len(plan.conversions)} label(s) from predictions.",
                lines + context,
                len(plan.conversions),
            )
        created = convert_record_predictions(
            records, overwrite=allow_overwrite, recorder=self.recorder.for_batch()
        )
        return OperationReport(
            f"Wrote {len(created)} label(s) from predictions.",
            lines + context,
            len(created),
            applied=True,
            written=created,
        )

    def find_specks(
        self,
        records: Iterable[ImageRecord],
        *,
        at_most: int,
        on_progress: Callable[[int, int], None] | None = None,
        cancellation: Cancellation | None = None,
    ) -> OperationReport:
        """Report which projects hold labels with components likely to be too small.

        **Likely**, not certainly: a small hole can be real, and this reports
        rather than judges. What it finds is worth a second look, not a defect.

        **Counted per project, not per label**, which is the question this is
        asked to answer: the per-label list is what `find-specks` prints, and a
        window listing three hundred files is one nobody reads to the end. What
        a reader wants here is which datasets are worth going back to, so the
        projects come back worst first and the proofreading view is where the
        individual labels are then looked at.

        **Ordered by specks per label rather than by their total**, that being
        what says how bad the labels are: forty across forty labels is a
        collection with a stray pixel here and there, and forty across three is
        three labels somebody should open.

        No preview. A preview would have to read every label to say what it
        would find, which is the whole cost -- so this runs once, reports
        progress, and can be stopped. Stopping is free: nothing is written, and
        what was read is simply a smaller sweep.
        """
        records = list(records)
        if not records:
            return OperationReport("Nothing selected to look through.")

        def report(done: int, total: int) -> None:
            if on_progress is not None:
                on_progress(done, total)
            if cancellation is not None:
                cancellation.check()

        found = records_with_specks(records, at_most=at_most, on_progress=report)
        projects = self._specks_by_project(records, found)
        # Counted from the records rather than from what was found: "nothing in
        # 248 labels" is a different statement from "nothing".
        read = len({
            label for label in (find_matching_label(record) for record in records)
            if label is not None
        })
        if not found:
            return OperationReport(
                f"No component of {at_most} pixel(s) or less in {read} label(s).",
                applied=True,
            )
        total = sum(len(specks) for specks in found.values())
        return OperationReport(
            f"{total} component(s) of {at_most} pixel(s) or less, in "
            f"{len(found)} of {read} label(s).",
            _speck_table(projects),
            len(found),
            applied=True,
        )

    @staticmethod
    def _specks_by_project(
        records: Sequence[ImageRecord], found: Mapping[Path, Sequence]
    ) -> list[tuple[str, int, int]]:
        """Return ``(project, specks, labels)`` worst first.

        A label is counted once however many imaging conditions share it, the
        file being what somebody has to open; the first record naming it says
        which project it belongs to, and conditions of one scene cannot
        disagree about that.
        """
        owners: dict[Path, str] = {}
        for record in records:
            label = find_matching_label(record)
            if label is not None and label not in owners:
                owners[label] = record.project or "(no project)"
        counted: dict[str, list[int]] = {}
        for label, specks in found.items():
            entry = counted.setdefault(owners.get(label, "(no project)"), [0, 0])
            entry[0] += len(specks)
            entry[1] += 1
        # Worst first by specks per label, then by how many there are, then by
        # name so the order does not move between runs on the same data.
        return sorted(
            ((name, specks, labels) for name, (specks, labels) in counted.items()),
            key=lambda item: (-item[1] / item[2], -item[1], item[0]),
        )

    # -- predicting ---------------------------------------------------------

    def predict(
        self,
        records: Iterable[ImageRecord],
        weights: Path,
        *,
        device: str = "auto",
        dry_run: bool = False,
        cancellation: Cancellation | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        run: Callable[..., ProcessOutcome] | None = None,
    ) -> OperationReport:
        """Predict the selection with the model at *weights*.

        A preview neither loads the model nor imports torch: what a run would
        write is a question about the filesystem, and answering it in
        milliseconds is what lets the count be shown before anything heavy
        starts.

        **The applying run happens in a child process.** Inference in this one
        crashed the window repeatedly and the command line never once, so the
        window holds no torch at all and the child holds all of it. Nothing is
        lost when that child dies: each prediction is written by rename and its
        provenance flushed before the next begins.

        Stopping is reported rather than raised: a run cancelled after ten of a
        hundred images did write ten, and they are kept.

        Args:
            run: How the child is run. Resolved at the call rather than bound
                as a default, so a test can substitute it -- there being no
                model to load and no reason to start an interpreter.
        """
        records = list(records)
        try:
            _description, source = self._describe_model(weights)
        except (ValueError, FileNotFoundError) as error:
            # Not `blocked`: that is for what the user can confirm past, and
            # a model that is not there or not described is not that. A report
            # with no count is shown and dismissed.
            return OperationReport(str(error))

        if dry_run:
            plan = plan_predictions(records, source, overwrite=False)
            if plan.is_refused:
                return OperationReport(self._refused_modalities(plan.modalities))
            context = self._prediction_context(plan, source)
            if not plan.predictions:
                return OperationReport("Nothing to predict.", context)
            return OperationReport(
                f"Would predict {len(plan.predictions)} {plan.modality} image(s) "
                f"with {weights.name}.",
                tuple(
                    f"{self._relative(item.destination)}: {item.describe()}"
                    for item in plan.predictions
                ) + context,
                len(plan.predictions),
            )

        runner = run or predict_out_of_process
        outcome = runner(
            records, weights, device,
            on_progress=on_progress, cancellation=cancellation,
        )
        return self._report_outcome(outcome, weights)

    def _report_outcome(
        self, outcome: ProcessOutcome, weights: Path
    ) -> OperationReport:
        """Turn what the child reported into the report the window shows."""
        if outcome.error:
            return OperationReport(
                f"The prediction did not run: {outcome.error}"
            )
        if outcome.refused:
            return OperationReport(self._refused_modalities(outcome.modalities))

        context = self._outcome_context(outcome)
        if not outcome.total:
            return OperationReport("Nothing to predict.", context)

        lines = tuple(
            f"{self._relative(Path(destination))}: write"
            for destination in outcome.destinations
        )
        # Sliced by what the child reported writing, not by the length of the
        # plan it was given: `destinations` is filled before the run, so a
        # stopped one names files it never reached.
        created = tuple(Path(d) for d in outcome.destinations[:outcome.written])
        if outcome.stopped:
            return OperationReport(
                f"Stopped after {outcome.written} of {outcome.total} "
                f"{outcome.modality} image(s). What was predicted is kept; run "
                f"it again to continue.",
                lines[:outcome.written] + context,
                outcome.written,
                applied=True,
                written=created,
            )
        return OperationReport(
            f"Predicted {outcome.written} {outcome.modality} image(s) with "
            f"{weights.name}.",
            lines + context,
            outcome.written,
            applied=True,
            written=created,
        )

    @staticmethod
    def _refused_modalities(modalities: Sequence[str]) -> str:
        """Return why a selection spanning both beams cannot be predicted."""
        return (
            f"This selection holds both {' and '.join(modalities)} images. A "
            f"model is trained on one of them, and nothing here can tell which, "
            f"so predict one at a time: set the edit target to SEM or to FIB."
        )

    def _outcome_context(self, outcome: ProcessOutcome) -> tuple[str, ...]:
        """Return the lines explaining what a finished run left out, and why."""
        lines: list[str] = []
        if outcome.represented:
            lines.append(
                f"{outcome.represented} image(s) share a label with one being "
                f"predicted and are covered by it."
            )
        if outcome.up_to_date:
            lines.append(
                f"{outcome.up_to_date} image(s) were already predicted by "
                f"this model and are left alone."
            )
        return tuple(lines)

    def _describe_model(self, weights: Path):
        """Return the model's description and the source recording it.

        Reads the description file and the checkpoint's archive header only --
        no torch, and none of the 170 MB of weights -- so a preview costs
        nothing and a typo in the description is reported before anything loads.
        """
        from ..ml.model import read_model_description, run_name_of

        if not weights.exists():
            raise FileNotFoundError(f"No model at {weights}")
        description = read_model_description(weights)
        source = PredictionSource(name=weights.name, run_name=run_name_of(weights))
        return description, source

    def _prediction_context(self, plan, source: PredictionSource) -> tuple[str, ...]:
        """Return the lines explaining what a run leaves out, and why.

        A run that predicts nothing is the case most in need of an explanation,
        and the three reasons need different actions: nothing, wait, or pass
        --overwrite.
        """
        lines: list[str] = []
        if plan.represented:
            lines.append(
                f"{plan.represented} image(s) share a label with one being "
                f"predicted and are covered by it."
            )
        if plan.up_to_date:
            lines.append(
                f"{len(plan.up_to_date)} image(s) were already predicted by "
                f"this model and are left alone."
            )
        return tuple(lines)

    # -- recompressing labels ----------------------------------------------

    def compress_labels(
        self,
        records: Iterable[ImageRecord],
        *,
        allow_corrected: bool = False,
        dry_run: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        cancellation: Cancellation | None = None,
    ) -> OperationReport:
        """Re-encode the labels of *records* so they take less disk space.

        The preview counts the files and their current size rather than
        measuring the saving: measuring it means re-encoding every label, which
        is the whole cost of the operation, and doing that twice to show a
        number is not worth minutes of waiting. Only the applying run is slow,
        so only it reports progress and only it can be stopped.

        Stopping is safe here for the same reason predicting's is, and for a
        reason the other batch operations do not have: a recompressed label
        holds the same pixels as the one it replaced, so a run stopped halfway
        leaves a collection that differs from a finished one only in how much
        disk it takes. There is nothing to resume and nothing to undo -- the
        rest were never opened.
        """
        records = list(records)
        labels = existing_labels(records)
        if not labels:
            return OperationReport("Nothing to compress: the selection has no labels.")

        corrected_labels = {
            label
            for record in records
            if current_state(record) == "corrected"
            for label in record.labels
        }
        corrected = [label for label in labels if label in corrected_labels]
        if corrected and not allow_corrected:
            # The refusal itself lives in compress_labels and still fires; asking
            # here is what lets the window say how much is at stake in advance.
            return OperationReport(
                f"{len(corrected)} of these {len(labels)} label(s) hold proofread "
                f"work. Recompressing rewrites the file.",
                tuple(self._relative(label) for label in corrected),
                len(labels),
                blocked="corrected",
            )

        total = sum(label.stat().st_size for label in labels)
        if dry_run:
            return OperationReport(
                f"Would re-encode {len(labels)} label(s), {format_bytes(total)} on "
                f"disk now. Pixel values and dimensions are unchanged; a file that "
                f"would not shrink is left alone.",
                tuple(self._relative(label) for label in labels),
                len(labels),
            )

        def report(done: int, total: int) -> None:
            if on_progress is not None:
                on_progress(done, total)
            if cancellation is not None:
                cancellation.check()

        results = compress_labels(
            records, allow_corrected=allow_corrected, on_progress=report,
            recorder=self.recorder.for_batch(),
        )
        applied = [result for result in results if result.applied]
        saved = sum(result.original_size - result.new_size for result in applied)
        lines = tuple(
            f"{self._relative(result.label_path)}: "
            f"{format_bytes(result.original_size)} -> {format_bytes(result.new_size)}"
            for result in applied
        )
        # Named rather than counted: a label something else rewrote mid-run is
        # the one outcome here that is not simply "nothing to gain".
        stale = tuple(
            f"{self._relative(result.label_path)}: changed on disk, left alone"
            for result in results if result.stale
        )
        if len(results) < len(labels):
            headline = (
                f"Stopped after {len(results)} of {len(labels)} label(s), having "
                f"compressed {len(applied)} of them and saved {format_bytes(saved)}. "
                f"The rest are untouched; run it again to finish."
            )
        else:
            headline = (
                f"Compressed {len(applied)} of {len(results)} label(s), saving "
                f"{format_bytes(saved)}."
            )
        if stale:
            headline += f" {len(stale)} changed on disk and were left alone."
        return OperationReport(
            headline,
            lines + stale,
            len(applied),
            applied=True,
        )


    # -- describing and indexing a collection ------------------------------

    # -- copying a selection out -------------------------------------------

    def copy_files(
        self,
        records: Iterable[ImageRecord],
        destination: Path,
        *,
        projects: Sequence = (),
        start_path: Path | None = None,
        scope: str = "",
        dry_run: bool = False,
        cancellation: Cancellation | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> OperationReport:
        """Copy the selection, its labels and its project files into *destination*.

        The preview is the plan: what would be written, how much of it, and
        what is in the way. It reads directory entries only, so a selection of
        thousands answers while the user is still looking at the folder they
        picked rather than long after.

        **A refusal carries no count and cannot be confirmed past.** Everything
        else in this window that stops can be overridden, because what it
        protects is the user's own data and the user may know better; here the
        answer is to choose another folder.

        A completed copy is given the fork point that lets it be merged back,
        by the same call the command line makes.

        Args:
            projects: The scanned projects, which say where each project's own
                files are. Without them the copy is images and labels alone.
            start_path: The scan root the copied paths are relative to.
                Defaults to the one reports are written against.
            scope: What the export was narrowed to, recorded in the fork point
                as context for whoever merges the copy back.
        """
        from ..operations.copy_export import (
            DestinationInsideSource,
            copy_export,
            plan_copy,
            record_fork_point,
        )

        root = start_path or self.start_path
        if root is None:
            return OperationReport("Nothing has been scanned yet.")
        records = list(records)
        try:
            plan = plan_copy(records, destination, root, projects=projects)
        except DestinationInsideSource as error:
            return OperationReport(f"Choose a folder outside the collection: {error}")

        if plan.conflicts:
            return OperationReport(
                f"{len(plan.conflicts)} file(s) already in {destination} differ "
                f"from what this would copy there. Nothing has been copied; "
                f"choose an empty folder, or the one an earlier run of this "
                f"same export wrote.",
                tuple(
                    str(item.destination.name)
                    for item in plan.conflicts[:MAX_REPORTED_FILES]
                ),
            )
        if plan.short_by:
            return OperationReport(
                f"Not enough room in {destination}: "
                f"{format_bytes(plan.bytes_to_copy)} to write, "
                f"{format_bytes(plan.free_space)} free."
            )
        if not plan.to_copy:
            return OperationReport(
                f"Nothing to copy: {destination} already holds all "
                f"{len(plan.already_there)} file(s)."
            )

        if dry_run:
            return OperationReport(
                f"Would copy {len(plan.to_copy)} file(s), "
                f"{format_bytes(plan.bytes_to_copy)}, into {destination}.",
                _copy_context(plan),
                len(plan.to_copy),
            )

        report = copy_export(plan, on_progress=on_progress, cancellation=cancellation)
        headline = (
            f"Copied {len(report.copied)} file(s), "
            f"{format_bytes(report.bytes_copied)}, into {destination}."
        )
        fork = record_fork_point(
            destination, records, projects, root,
            stopped=report.stopped, scope=scope,
        )
        if report.stopped:
            headline += (
                " Stopped early; what was copied is kept, and running it again "
                "carries on from there."
            )
        return OperationReport(
            headline, self._fork_context(fork),
            applied=True, count=len(report.copied),
        )

    @staticmethod
    def _fork_context(fork: ExportRecord | None) -> tuple[str, ...]:
        """Return what the report says about merging this copy back.

        Said out loud rather than left to the file, because whether a copy can
        come home is the one thing about it that cannot be seen by looking in
        the folder.
        """
        if fork is None:
            return (
                "No fork point was written, because the copy was stopped: "
                "this cannot be merged back until the export is run again.",
            )
        lines = ["Written with a fork point, so it can be merged back."]
        missing = fork.without_history
        if missing:
            lines.append(
                f"{len(missing)} project(s) keep no history yet, so a merge "
                f"back has no common ancestor for them: "
                f"{', '.join(missing[:3])}."
            )
        return tuple(lines)

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
        return tuple([*OperationsController._savepoint_notes(plan), *plan.describe()])

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
            plan = plan_merge(Path(source), destination)
        except MergeRefused as error:
            return OperationReport(str(error))

        taking = plan.counted(TAKE_THEIRS)
        incoming = sum(len(project.incoming) for project in plan.projects)
        context = [
            f"{taking} to take from the copy, {plan.counted(KEEP_OURS)} already "
            f"here, {plan.counted(CONVERGED)} arrived at the same thing.",
        ]
        if plan.conflicts:
            return OperationReport(
                f"{len(plan.conflicts)} conflict(s) stand, so nothing can be "
                f"merged. Both sides changed these since the copy was taken, "
                f"and which to keep is decided by name with "
                f"'am-seg merge --resolve'. Both versions stay in the archive "
                f"either way.",
                tuple(plan.describe())[:MAX_REPORTED_FILES],
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
            plan, recorder=self.recorder.for_batch(), allow_corrected=allow_corrected
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

    def survey(self, root: Path, *, detail: bool = False) -> OperationReport:
        """Report how a directory is interpreted, and what does not fit.

        Read-only. This is the first thing to run on unfamiliar data, which is
        why it is on the tab that adds data rather than behind a scan.
        """
        from ..reporting import format_survey
        from ..survey import survey_collection

        found = survey_collection(Path(root))
        counts = found.finding_counts()
        return OperationReport(
            f"{len(found.projects)} project(s); "
            f"{counts['error']} error(s), {counts['warning']} warning(s).",
            tuple(format_survey(found, detail=detail)),
            len(found.projects),
        )

    def undescribed_projects(self, root: Path) -> list[Path]:
        """Return projects under *root* that have no ``AM_metadata.yaml`` yet.

        Autolamella writes ``experiment.yaml``, so these are already indexed
        and already projects. Giving them the descriptive file is bulk work
        with no judgement in it, unlike deciding what counts as a project.
        """
        from ..discovery import find_project_roots
        from ..metadata import AM_METADATA_FILENAME

        return [
            path
            for path in find_project_roots(Path(root))
            if not (path / AM_METADATA_FILENAME).exists()
        ]

    def unindexed_summary(self, root: Path) -> tuple[int, int]:
        """Return how many directories under *root* are not projects, and their
        image count.

        Cheap next to the scan it follows -- one walk of the same tree -- and
        it answers the question the project list cannot: what is sitting here
        that nothing has indexed.
        """
        with timed(logger, "Counting what is not indexed"):
            candidates = self.propose_projects(root)
        return len(candidates), sum(item.images for item in candidates)

    def propose_projects(self, root: Path) -> list[DatasetCandidate]:
        """Return the directories under *root* holding images but not indexed.

        Proposals for a person to confirm. Which level is the project is a fact
        about the acquisition rather than the tree, so this offers and does not
        decide; :meth:`propose_projects_below` is how a proposal made too high
        is corrected.
        """
        from ..discovery import describe_dataset_candidates

        return describe_dataset_candidates(Path(root))

    def propose_projects_below(self, directory: Path) -> list[DatasetCandidate]:
        """Return the proposals one level inside *directory*."""
        from ..discovery import dataset_candidates_below

        return dataset_candidates_below(Path(directory))

    def describe_proposal(self, candidate: DatasetCandidate) -> tuple[str, str]:
        """Return a proposal as a heading and the detail line under it.

        What it would *become* rather than what it is called: lamella names
        that read like dataset names are what tells a reader the level is
        wrong, and no structural test says that.
        """
        heading = (
            f"{self._relative(candidate.path)}  "
            f"-- {len(candidate.lamellae)} lamella(e), {candidate.images} image(s)"
        )
        shown = ", ".join(candidate.lamellae[:4])
        more = "" if len(candidate.lamellae) <= 4 else ", ..."
        if candidate.holds_several_datasets:
            detail = (
                f"laid out {len(candidate.stage_depths)} different ways, so this "
                f"probably gathers several datasets -- would become lamellae: {shown}{more}"
            )
        else:
            detail = f"would become lamellae: {shown}{more}"
        return heading, detail

    def create_projects(
        self, paths: Sequence[Path], *, dry_run: bool = True
    ) -> OperationReport:
        """Write an ``AM_metadata.yaml`` into each of *paths*.

        Every path is checked before any of them is written, so a set holding
        one directory that must not be marked writes nothing rather than
        stopping partway. An existing file is kept and reported; nothing here
        overwrites one.
        """
        from ..operations.init_project import init_projects, project_placement_problem

        chosen = [Path(path) for path in paths]
        if not chosen:
            return OperationReport("Nothing chosen.")

        problems = [
            f"{self._relative(path)}: {problem}"
            for path in chosen
            if (problem := project_placement_problem(path)) is not None
        ]
        if problems:
            return OperationReport(
                f"{len(problems)} of {len(chosen)} chosen directory(ies) cannot be "
                f"made a project; nothing was written.",
                tuple(problems),
                blocked="placement",
            )

        results = init_projects(chosen, dry_run=dry_run)
        lines = []
        for result in results:
            location = self._relative(result.path)
            if result.skipped_existing:
                lines.append(f"kept   {location}")
            else:
                lines.append(f"write  {location} ({len(result.todo)} field(s) to fill in)")

        # A project written outside the scanned folder is a real project that
        # this window will not show: the scan that follows reads the folder it
        # already has. Said here rather than left to be noticed.
        outside = [path for path in chosen if not self._within_scan(path)]
        if outside:
            lines.append(
                f"       {len(outside)} of these are outside the scanned folder and "
                f"will not appear until you scan a folder holding them"
            )

        writing = [result for result in results if not result.skipped_existing]
        skipped = len(results) - len(writing)
        verb = "Would write" if dry_run else "Wrote"
        return OperationReport(
            f"{verb} {len(writing)} metadata file(s); {skipped} already had one.",
            tuple(lines),
            len(writing),
            applied=not dry_run,
        )

    def _within_scan(self, path: Path) -> bool:
        """Return whether *path* is inside the folder this window scanned."""
        if self.start_path is None:
            return True
        try:
            path.resolve().relative_to(self.start_path.resolve())
        except ValueError:
            return False
        return True

    def check_csv(self, csv_path: Path, start_path: Path) -> OperationReport:
        """Check that every path an exported CSV records still exists.

        Read-only. Paths are resolved against *start_path*, the scan root the
        CSV was written against, which is not necessarily this session's.
        """
        from ..operations.csv_check import check_csv_paths, summarize_results

        results = check_csv_paths(Path(csv_path), Path(start_path))
        lines, problems = summarize_results(results, Path(start_path))
        headline, *detail = lines
        if problems:
            headline = f"{headline} -- {problems} row(s) need attention."
        return OperationReport(headline, tuple(detail), problems)
