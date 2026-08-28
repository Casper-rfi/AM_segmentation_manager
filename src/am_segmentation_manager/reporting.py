"""Project summary and display utilities.

Functions for producing human-readable summaries of scanned projects.
Extracted from the CLI so they can be reused by future features such as
metadata display and filtered overviews.

Formatting functions return lists of lines rather than printing, so they can be
asserted on in tests; the ``print_*`` wrappers emit them through the module
logger.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .io.path import make_path_relative
from .layout import MODALITIES, SERIES_LABELS
from .operations.selection import current_state
from .survey import SEVERITY_ORDER, CollectionSurvey, Finding, ProjectSurvey

logger = logging.getLogger(__name__)


def format_bytes(size: int) -> str:
    """Return a byte count in the largest unit that leaves it above one."""
    value = float(size)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover - unreachable, the loop ends at GB

# The three mutually exclusive workflow states plus an "all" total.
_STATUS_ORDER = ("deselected", "selected", "corrected", "all")


def _new_counts() -> dict[str, int]:
    """Return a fresh counter dict for workflow states."""
    return {status: 0 for status in _STATUS_ORDER}


def _summary_categories() -> tuple[str, ...]:
    """Return the per-modality image and label categories, in display order."""
    return (
        *(f"{modality} images" for modality in MODALITIES),
        *(f"{modality} labels" for modality in MODALITIES),
    )


def record_state(record) -> str:
    """Return a record's declared workflow state.

    A label takes its group's state rather than having one of its own, which
    is what makes the image and label counts agree.
    """
    return current_state(record)


def summarize_lamella(lamella) -> dict[str, dict[str, int]]:
    """Count images and labels in each mutually exclusive workflow state.

    Returns a dict keyed by category name (``"SEM images"``, ``"FIB images"``,
    ``"SEM labels"``, ``"FIB labels"``) where each value is a counter dict
    with keys from :data:`_STATUS_ORDER`.
    """
    summary = {name: _new_counts() for name in _summary_categories()}
    for stage in lamella.stages:
        for record in stage.images:
            if record.modality not in MODALITIES:
                continue
            image_counts = summary[f"{record.modality} images"]
            image_counts[record_state(record)] += 1
            image_counts["all"] += 1
            label_counts = summary[f"{record.modality} labels"]
            for label in record.labels:
                label_counts[record_state(record)] += 1
                label_counts["all"] += 1
    return summary


def _add_counts(destination: dict[str, int], source: dict[str, int]) -> None:
    """Add *source* counters into *destination* in place."""
    for status in _STATUS_ORDER:
        destination[status] += source[status]


def _count_text(counts: dict[str, int]) -> str:
    """Format a counter dict as a ``/``-separated status string."""
    return "/".join(str(counts[status]) for status in _STATUS_ORDER)


def print_project_summary(project) -> None:
    """Print a compact, aligned scan summary for a single project.

    Output is emitted through the module logger at INFO level so it can
    be captured in tests and redirected by the caller.
    """
    categories = _summary_categories()
    total = {category: _new_counts() for category in categories}
    lamella_summaries = []
    for lamella in project.lamellae:
        summary = summarize_lamella(lamella)
        lamella_summaries.append((lamella, summary))
        for category in categories:
            _add_counts(total[category], summary[category])

    logger.info("\n%s", project.display_name)
    logger.info("%s", "=" * 72)
    logger.info("Path: %s", project.path)
    logger.info("Counts shown as: deselected / selected / corrected / all")
    logger.info("\nAll lamellae")
    logger.info("%-16s %12s %12s %12s %12s", "", "Deselected", "Selected", "Corrected", "All")
    for category in categories:
        logger.info(
            "%-16s %12d %12d %12d %12d",
            category,
            *(total[category][status] for status in _STATUS_ORDER),
        )
    logger.info("\nLamellae")
    logger.info("Counts: deselected / selected / corrected / all")
    logger.info("%s", "-" * 72)
    for lamella, summary in lamella_summaries:
        logger.info("%s", lamella.display_name)
        for category in categories:
            logger.info("  %-14s %s", category, _count_text(summary[category]))
        for stage in lamella.stages:
            logger.info("  %-14s %d image(s)", stage.display_name, len(stage.images))


# ---------------------------------------------------------------------------
# Layout survey
# ---------------------------------------------------------------------------

#: Marker shown against each finding severity.
_SEVERITY_MARKERS = {"error": "ERROR", "warning": "warn ", "info": "info "}


def format_survey(
    survey: CollectionSurvey, detail: bool = False, min_severity: str = "info"
) -> list[str]:
    """Return the survey as displayable lines.

    Args:
        survey: The result of :func:`~am_segmentation_manager.survey.survey_collection`.
        detail: Include the full lamella/stage/image-set tree. Off by default
            because a collection of twenty projects runs to hundreds of stages,
            and the per-project totals plus the findings are what the survey is
            usually consulted for.
        min_severity: Lowest severity of finding to list. The counts line always
            reflects every finding, so filtering the list cannot hide the fact
            that findings exist.
    """
    lines = [
        f"Layout survey: {survey.root}",
        "=" * 78,
    ]
    counts = survey.finding_counts()
    summary = ", ".join(f"{counts[severity]} {severity}" for severity in SEVERITY_ORDER)
    lines.append(f"{len(survey.projects)} project(s) found; findings: {summary}")
    if survey.unindexed:
        # On the headline rather than only among the findings: a folder whose
        # data is mostly not indexed yet should say so where the project count
        # is read, not several screens below it.
        images = sum(candidate.images for candidate in survey.unindexed)
        lines.append(
            f"{len(survey.unindexed)} directory(ies) hold {images} image(s) but are "
            f"not projects yet; see the findings below"
        )

    for project in survey.projects:
        lines.append("")
        lines.extend(_format_project(project, survey.root, detail=detail))

    cutoff = SEVERITY_ORDER.index(min_severity) if min_severity in SEVERITY_ORDER else len(SEVERITY_ORDER)
    findings = [
        finding for finding in survey.all_findings
        if SEVERITY_ORDER.index(finding.severity) <= cutoff
    ]
    if findings:
        lines.append("")
        lines.append("Findings")
        lines.append("-" * 78)
        lines.extend(_format_findings(findings, survey.root))
    return lines


def _format_project(project: ProjectSurvey, root: Path, detail: bool) -> list[str]:
    """Return the lines describing one project."""
    markers = ", ".join(project.markers) if project.markers else "no marker"
    lines = [
        f"{project.relative_path}  [{markers}]",
        f"  {len(project.lamellae)} lamella(e), {project.stage_count} stage(s), "
        f"{project.image_count} image(s), {project.label_count} label(s)",
    ]
    series = project.images_by_series
    if len(series) > 1:
        counts = ", ".join(f"{count} {SERIES_LABELS[name]}" for name, count in series.items())
        lines.append(f"  series: {counts}")

    if project.method:
        lines.append(f"  method: {project.method}")

    metadata_note = _metadata_note(project)
    if metadata_note:
        lines.append(f"  metadata: {metadata_note}")

    synthetic = _synthetic_note(project)
    if synthetic:
        lines.append(f"  hierarchy: {synthetic}")

    if not detail:
        return lines

    for lamella in project.lamellae:
        suffix = " (synthetic)" if lamella.synthetic else ""
        lines.append(f"  {lamella.name}{suffix}")
        for stage in lamella.stages:
            stage_suffix = " (synthetic)" if stage.synthetic else ""
            lines.append(f"    {stage.name}{stage_suffix}")
            for entry in stage.image_sets:
                lines.append(f"      {_format_image_set(entry)}")
    return lines


def _format_image_set(entry) -> str:
    """Return a one-line description of an image set and its labels."""
    image_set = entry.image_set
    labels = image_set.labels_directory.name
    if not image_set.labels_exist:
        labels_note = f"{labels}/ (not created)"
    elif image_set.labels_are_shared:
        labels_note = f"{labels}/ (shared, {entry.label_count})"
    else:
        labels_note = f"{labels}/ ({entry.label_count})"
    return (
        f"{image_set.name:<20} {image_set.display_name:<14} "
        f"{entry.image_count:>4} image(s)  labels: {labels_note}"
    )


def _metadata_note(project: ProjectSurvey) -> str:
    """Summarise how complete a project's descriptive metadata is."""
    metadata = project.metadata
    if metadata.todo_fields:
        return f"{len(metadata.todo_fields)} field(s) still marked TODO"
    missing = metadata.missing_fields()
    if not missing:
        return "complete"
    if metadata.is_empty:
        return "none recorded"
    return f"missing {', '.join(missing)}"


def _synthetic_note(project: ProjectSurvey) -> str:
    """Describe any hierarchy levels that do not exist as directories."""
    lamella_synthetic = any(lamella.synthetic for lamella in project.lamellae)
    stage_synthetic = any(
        stage.synthetic for lamella in project.lamellae for stage in lamella.stages
    )
    missing = [
        name
        for name, is_synthetic in (("lamella", lamella_synthetic), ("stage", stage_synthetic))
        if is_synthetic
    ]
    if not missing:
        return ""
    return f"no {' or '.join(missing)} directories; the parent stands in"


def _format_findings(findings: list[Finding], root: Path) -> list[str]:
    """Return findings grouped by severity, most severe first."""
    lines: list[str] = []
    for severity in SEVERITY_ORDER:
        matching = [finding for finding in findings if finding.severity == severity]
        for finding in matching:
            marker = _SEVERITY_MARKERS[finding.severity]
            location = make_path_relative(finding.path, root)
            lines.append(f"  {marker} {finding.category}: {location}")
            lines.append(f"         {finding.message}")
    return lines


def print_survey(
    survey: CollectionSurvey, detail: bool = False, min_severity: str = "info"
) -> None:
    """Print a layout survey through the module logger."""
    for line in format_survey(survey, detail=detail, min_severity=min_severity):
        logger.info("%s", line)


# ---------------------------------------------------------------------------
# Statistics tables
#
# aggregate.py counts; this turns counts into something readable. Kept here
# rather than there so the counting stays free of presentation.
# ---------------------------------------------------------------------------

def format_counts(
    counts: dict, title: str, *, total: int | None = None, multi: bool = False
) -> list[str]:
    """Return a one-dimensional breakdown as aligned lines.

    A percentage column is included because the question these tables answer is
    usually about balance, and "2386" only answers it next to the total.

    Args:
        total: What the percentages are a share of, where the rows do not add
            up to it. Required for a *multi* breakdown, whose rows add up to
            more than the images counted.
        multi: Whether an image can appear in more than one row. Said out loud,
            because a column of percentages adding to more than 100 reads as an
            arithmetic error otherwise.
    """
    if not counts:
        return [f"{title}: nothing to count"]

    total = sum(counts.values()) if total is None else total
    width = max(len(str(key)) for key in counts)
    lines = [f"{title}  ({total} image(s))"]
    if multi:
        lines.append("  An image holding several values is counted under each.")
    for key, count in counts.items():
        share = f"{100 * count / total:5.1f}%" if total else "    -"
        lines.append(f"  {str(key):<{width}}  {count:>7}  {share}")
    return lines


def format_cross_tabulation(
    table: dict,
    row_title: str,
    column_title: str,
    *,
    title: str | None = None,
    row_order: Sequence = (),
    column_order: Sequence = (),
    multi: bool = False,
) -> list[str]:
    """Return a two-dimensional breakdown as an aligned grid, with margins.

    Args:
        title: Replaces the heading line, for a caller that has to say which
            set was counted. The two titles keep naming the axes either way.
        row_order: Row labels to keep in a declared order, for an axis that is
            a scale rather than a set of names. The column order likewise.
        multi: Whether either axis holds several values per image. The margins
            then count an image once per cell it lands in, and a total larger
            than the collection needs saying rather than working out.
    """
    from .aggregate import totals

    heading = title or f"{row_title} x {column_title}"
    if not table:
        return [f"{heading}: nothing to count"]

    rows, columns = totals(table, row_order, column_order)
    row_width = max([len(str(row)) for row in rows] + [len(row_title)])
    column_widths = {
        column: max(len(str(column)), 7) for column in columns
    }

    header = f"{row_title:<{row_width}} " + " ".join(
        f"{str(column):>{column_widths[column]}}" for column in columns
    )
    lines = [heading]
    if multi:
        lines.append("An image holding several values is counted in each cell.")
    lines += [header, "-" * len(header)]

    for row in rows:
        cells = " ".join(
            f"{table.get((row, column), 0):>{column_widths[column]}}" for column in columns
        )
        row_total = sum(count for (r, _), count in table.items() if r == row)
        lines.append(f"{str(row):<{row_width}} {cells}   = {row_total}")

    footer = " ".join(
        f"{sum(count for (_, c), count in table.items() if c == column):>{column_widths[column]}}"
        for column in columns
    )
    lines.append("-" * len(header))
    lines.append(f"{'total':<{row_width}} {footer}   = {sum(table.values())}")
    return lines


def format_speck_report(
    found: dict, root: Path, config, *, at_most: int, labels_read: int
) -> list[str]:
    """Return the small-component sweep as lines, one per label that has any.

    Grouped by label rather than by component: the label is the file somebody
    has to open and fix, and the sizes are short enough to sit on its line.
    The class is named for each component because it says what you are looking
    at, but it is not part of the test -- a stray pixel is a stray pixel
    whichever class it claims to be.

    Args:
        found: Label path to its specks, from
            :func:`~am_segmentation_manager.operations.label_measure.records_with_specks`.
        root: Scan root, so the paths print short.
        config: The segmentation config, for class names.
        at_most: The threshold that was applied, in pixels.
        labels_read: How many labels were looked at, so a clean sweep says what
            it covered rather than only that it found nothing.
    """
    if not found:
        return [
            f"No component of {at_most} pixel(s) or less in "
            f"{labels_read} label(s)"
        ]

    components = sum(len(specks) for specks in found.values())
    lines = [
        f"{components} component(s) of {at_most} pixel(s) or less, "
        f"in {len(found)} of {labels_read} label(s)",
        "",
    ]
    for label in sorted(found):
        specks = found[label]
        sizes = ", ".join(
            f"{config.name(speck.class_index)} {speck.pixels}px "
            f"at ({speck.x}, {speck.y})"
            for speck in specks
        )
        lines.append(f"  {make_path_relative(label, root)}")
        lines.append(f"      {sizes}")
    return lines


def format_label_quantification(
    summaries: list, config, *, detail: bool = True
) -> list[str]:
    """Return the label-quantification report as lines.

    The headline is per lamella, because that is the level the question is
    asked at: one crack seen in three frames of a milling sequence is one
    crack, not three. Per-label counts follow, and the per-image rows go to the
    CSV rather than here.

    Args:
        summaries: :class:`~am_segmentation_manager.aggregate.LamellaQuantification`
            entries, one per lamella.
        config: The segmentation config, for class names.
        detail: Include the per-lamella table as well as the totals.
    """
    if not summaries:
        return ["Nothing to measure"]

    measurable = [summary for summary in summaries if summary.measured]
    cracked = [summary for summary in measurable if summary.has_crack]
    labels = sum(summary.labels for summary in summaries)
    measured = sum(summary.measured for summary in summaries)
    empty = sum(summary.empty for summary in summaries)
    unmeasurable = sum(summary.unmeasurable for summary in summaries)
    cracked_labels = sum(summary.cracked for summary in summaries)
    cracks = sum(summary.cracks for summary in summaries)

    lines: list[str] = ["Label quantification"]
    if measurable:
        share = 100 * len(cracked) / len(measurable)
        lines.append(
            f"  {len(cracked)} of {len(measurable)} lamellae have a crack  ({share:.1f}%)"
        )
    else:
        lines.append("  No lamella could be measured")

    lines.append(
        f"  {cracked_labels} of {measured} measured label(s) hold {cracks} crack(s)"
    )

    # Said plainly rather than folded into the totals: an empty label is a
    # placeholder the segmentation software wrote, not a proofread label with
    # nothing in it, and counting the two together would overstate the archive.
    skipped = []
    if empty:
        skipped.append(f"{empty} empty")
    if unmeasurable:
        skipped.append(f"{unmeasurable} with no pixel size")
    if skipped:
        lines.append(f"  Excluded from the above: {', '.join(skipped)} of {labels} label(s)")

    areas = _total_areas(summaries)
    if areas:
        lines.append("  Total area by class:")
        width = max(len(config.name(index)) for index in areas)
        for index in sorted(areas):
            lines.append(f"    {config.name(index):<{width}}  {areas[index]:12.2f} um^2")

    if detail and summaries:
        lines.append("")
        lines.extend(_quantification_table(summaries))
    return lines


def _total_areas(summaries: list) -> dict[int, float]:
    """Sum each class's area across every lamella."""
    totals: dict[int, float] = {}
    for summary in summaries:
        for index, area in summary.areas_um2.items():
            totals[index] = totals.get(index, 0.0) + area
    return totals


def _quantification_table(summaries: list) -> list[str]:
    """Return one aligned row per lamella."""
    name_width = max(
        [len(f"{summary.project}/{summary.lamella}") for summary in summaries] + [8]
    )
    header = (
        f"{'lamella':<{name_width}}  {'labels':>6}  {'measured':>8}  "
        f"{'cracked':>7}  {'cracks':>6}  {'crack um^2':>10}  {'largest':>9}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        name = f"{summary.project}/{summary.lamella}"
        lines.append(
            f"{name:<{name_width}}  {summary.labels:>6}  {summary.measured:>8}  "
            f"{summary.cracked:>7}  {summary.cracks:>6}  "
            f"{summary.crack_area_um2:>10.2f}  {summary.largest_crack_um2:>9.2f}"
        )
    return lines
