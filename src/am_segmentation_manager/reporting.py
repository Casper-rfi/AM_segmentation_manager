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
from pathlib import Path

from .io.path import make_path_relative
from .layout import MODALITIES, state_of_path
from .survey import SEVERITY_ORDER, CollectionSurvey, Finding, ProjectSurvey

logger = logging.getLogger(__name__)

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


def filesystem_state(path: Path) -> str:
    """Return an artifact's workflow state from its containing directory.

    Files under a ``deselected/`` or ``corrected/`` parent are reported
    as such; everything else is ``"selected"``.
    """
    return state_of_path(path)


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
            image_counts[filesystem_state(record.image_path)] += 1
            image_counts["all"] += 1
            label_counts = summary[f"{record.modality} labels"]
            for label in record.labels:
                label_counts[filesystem_state(label)] += 1
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
    counts: dict, title: str, *, total: int | None = None
) -> list[str]:
    """Return a one-dimensional breakdown as aligned lines.

    A percentage column is included because the question these tables answer is
    usually about balance, and "2386" only answers it next to the total.
    """
    if not counts:
        return [f"{title}: nothing to count"]

    total = sum(counts.values()) if total is None else total
    width = max(len(str(key)) for key in counts)
    lines = [f"{title}  ({total} image(s))"]
    for key, count in counts.items():
        share = f"{100 * count / total:5.1f}%" if total else "    -"
        lines.append(f"  {str(key):<{width}}  {count:>7}  {share}")
    return lines


def format_cross_tabulation(
    table: dict, row_title: str, column_title: str
) -> list[str]:
    """Return a two-dimensional breakdown as an aligned grid, with margins."""
    from .aggregate import totals

    if not table:
        return [f"{row_title} x {column_title}: nothing to count"]

    rows, columns = totals(table)
    row_width = max([len(str(row)) for row in rows] + [len(row_title)])
    column_widths = {
        column: max(len(str(column)), 7) for column in columns
    }

    header = f"{row_title:<{row_width}} " + " ".join(
        f"{str(column):>{column_widths[column]}}" for column in columns
    )
    lines = [f"{row_title} x {column_title}", header, "-" * len(header)]

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
