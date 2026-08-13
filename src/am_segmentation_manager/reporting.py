"""Project summary and display utilities.

Functions for producing human-readable summaries of scanned projects.
Extracted from the CLI so they can be reused by future features such as
metadata display and filtered overviews.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The three mutually exclusive workflow states plus an "all" total.
_STATUS_ORDER = ("deselected", "selected", "corrected", "all")


def _new_counts() -> dict[str, int]:
    """Return a fresh counter dict for workflow states."""
    return {status: 0 for status in _STATUS_ORDER}


def filesystem_state(path: Path) -> str:
    """Return an artifact's workflow state from its containing directory.

    Files under a ``deselected/`` or ``corrected/`` parent are reported
    as such; everything else is ``"selected"``.
    """
    if path.parent.name == "deselected":
        return "deselected"
    if path.parent.name == "corrected":
        return "corrected"
    return "selected"


def summarize_lamella(lamella) -> dict[str, dict[str, int]]:
    """Count images and labels in each mutually exclusive workflow state.

    Returns a dict keyed by category name (``"SEM images"``, ``"FIB images"``,
    ``"SEM labels"``, ``"FIB labels"``) where each value is a counter dict
    with keys from :data:`_STATUS_ORDER`.
    """
    summary = {name: _new_counts() for name in ("SEM images", "FIB images", "SEM labels", "FIB labels")}
    for stage in lamella.stages:
        for record in stage.images:
            if record.modality not in {"SEM", "FIB"}:
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
    categories = ("SEM images", "FIB images", "SEM labels", "FIB labels")
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
