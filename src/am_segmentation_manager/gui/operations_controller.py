"""The batch operations the window offers, in one shape.

Converting predictions into labels, recompressing labels and repairing groups
whose images disagree with their label each return a different result type. Each
is offered the same way -- preview what it would do, confirm, then apply -- so
this turns all three into one :class:`OperationReport` and the window gets one
dialog rather than three.

Qt-free, like the other controllers here. Paths are reported relative to the
scan root: an absolute path eight directories deep is unreadable in a dialog.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io.path import make_path_relative
from ..layout import state_of_path
from ..models import ImageRecord
from ..operations.label_compression import compress_labels, existing_labels
from ..operations.label_groups import describe_group
from ..operations.predictions import convert_record_predictions, plan_conversions
from ..operations.selection import repair_group_states

logger = logging.getLogger(__name__)

#: Which subdirectory of ``predictions/`` to convert from. The empty string
#: takes whichever prediction a record has, whatever it is filed under.
PREDICTION_KINDS = ("clean", "raw", "")


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
    """

    headline: str
    lines: tuple[str, ...] = ()
    count: int = 0
    applied: bool = False
    blocked: str = ""

    def text(self) -> list[str]:
        """Return the report as displayable lines, headline first."""
        return [self.headline, *self.lines]


def format_bytes(size: int) -> str:
    """Return a byte count in the largest unit that leaves it above one."""
    value = float(size)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover - unreachable, the loop ends at GB


class OperationsController:
    """Previews and runs the operations that change files on disk.

    Holds the scan root so reports can name files the way the rest of the window
    does, and nothing else: every method takes the records it acts on, so what
    an operation applies to is decided by the selection and never remembered
    here.
    """

    def __init__(self) -> None:
        self.start_path: Path | None = None

    def set_start_path(self, start_path: Path | None) -> None:
        """Record the directory reports are written relative to."""
        self.start_path = Path(start_path) if start_path is not None else None

    def _relative(self, path: Path | None) -> str:
        if path is None:
            return "(no label)"
        if self.start_path is None:
            return str(path)
        return make_path_relative(path, self.start_path)

    # -- converting predictions -------------------------------------------

    def convert_predictions(
        self,
        records: Iterable[ImageRecord],
        prediction_kind: str = "clean",
        *,
        dry_run: bool = False,
    ) -> OperationReport:
        """Turn each record's prediction into a label, where it has no label yet.

        Records already carrying a label are left out, so this can never
        overwrite proofread work. Applying re-resolves the plan rather than
        reusing the preview's, so a file that appeared in between is still seen.
        """
        records = list(records)
        planned = plan_conversions(records, prediction_kind)
        lines = tuple(self._relative(item.destination) for item in planned)
        if not planned:
            return OperationReport(
                "Nothing to convert: every selected image either has a label "
                "already or has no matching prediction."
            )
        if dry_run:
            return OperationReport(
                f"Would write {len(planned)} label(s) from predictions.", lines, len(planned)
            )
        written = convert_record_predictions(records, prediction_kind)
        return OperationReport(
            f"Wrote {written} label(s) from predictions.", lines, written, applied=True
        )

    # -- recompressing labels ----------------------------------------------

    def compress_labels(
        self,
        records: Iterable[ImageRecord],
        *,
        allow_corrected: bool = False,
        dry_run: bool = False,
    ) -> OperationReport:
        """Re-encode the labels of *records* so they take less disk space.

        The preview counts the files and their current size rather than
        measuring the saving: measuring it means re-encoding every label, which
        is the whole cost of the operation, and doing that twice to show a
        number is not worth minutes of waiting.
        """
        records = list(records)
        labels = existing_labels(records)
        if not labels:
            return OperationReport("Nothing to compress: the selection has no labels.")

        corrected = [label for label in labels if state_of_path(label) == "corrected"]
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

        results = compress_labels(records, allow_corrected=allow_corrected)
        applied = [result for result in results if result.applied]
        saved = sum(result.original_size - result.new_size for result in applied)
        lines = tuple(
            f"{self._relative(result.label_path)}: "
            f"{format_bytes(result.original_size)} -> {format_bytes(result.new_size)}"
            for result in applied
        )
        return OperationReport(
            f"Compressed {len(applied)} of {len(results)} label(s), saving "
            f"{format_bytes(saved)}.",
            lines,
            len(applied),
            applied=True,
        )

    # -- repairing group state ---------------------------------------------

    def repair_state(
        self,
        records: Iterable[ImageRecord],
        *,
        allow_corrected: bool = False,
        dry_run: bool = False,
    ) -> OperationReport:
        """Move images to agree with the state of the label they share.

        The label is authoritative, so this moves images and never labels. A
        group needing an image moved *out* of ``corrected/`` is refused until
        that is confirmed, and the refusal is reported rather than raised.
        """
        records = list(records)
        try:
            repaired = repair_group_states(
                records, allow_move_from_corrected=allow_corrected, dry_run=dry_run
            )
        except PermissionError as error:
            return OperationReport(
                "Repairing this would move images out of a corrected folder.",
                (str(error),),
                blocked="corrected",
            )
        if not repaired:
            return OperationReport(
                "Nothing to repair: every image agrees with the label it shares."
            )

        lines = tuple(
            f"{self._relative(group.label_path)} -> {group.state()} "
            f"({moved} image(s)): {describe_group(group)}"
            for group, moved in repaired
        )
        verb = "Would repair" if dry_run else "Repaired"
        return OperationReport(
            f"{verb} {len(repaired)} group(s).",
            lines,
            len(repaired),
            applied=not dry_run,
        )

    # -- describing and indexing a collection ------------------------------

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

    def add_metadata_files(self, root: Path, *, dry_run: bool = True) -> OperationReport:
        """Write an ``AM_metadata.yaml`` into every project under *root*.

        Directories holding images but no marker file are included: a dataset
        predating Autolamella is invisible to project discovery, which is
        exactly the data that most needs a marker. Projects that already have a
        metadata file keep it -- nothing here overwrites one.
        """
        from ..discovery import find_dataset_candidates, find_project_roots
        from ..operations.init_project import init_projects

        root = Path(root)
        existing = find_project_roots(root)
        candidates = find_dataset_candidates(root)
        roots = [*existing, *candidates]
        if not roots:
            return OperationReport(f"No projects or unindexed datasets found under {root}.")

        results = init_projects(roots, dry_run=dry_run)
        lines = []
        for result in results:
            location = self._relative(result.path)
            if result.skipped_existing:
                lines.append(f"kept   {location}")
            else:
                lines.append(f"write  {location} ({len(result.todo)} field(s) to fill in)")
        for candidate in candidates:
            lines.append(f"       {self._relative(candidate)} holds images but is not a project yet")

        writing = [result for result in results if not result.skipped_existing]
        skipped = len(results) - len(writing)
        verb = "Would write" if dry_run else "Wrote"
        return OperationReport(
            f"{verb} {len(writing)} metadata file(s); {skipped} project(s) already had one.",
            tuple(lines),
            len(writing),
            applied=not dry_run,
        )

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
