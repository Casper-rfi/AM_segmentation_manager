"""Exporting metadata to CSV, per image and per project.

Two exports, for two questions. The per-image file answers "what is in this
archive, and under what conditions was it taken"; the per-project file answers
"what are these experiments, and which of them still need metadata filling in".

Both replace the standalone script this package absorbed, and improve on it in
one way that matters: that script needed a CSV of paths as input and a
hand-maintained ``sessions_metadata_manual.csv`` to join session metadata onto.
Both come from the index here, so the manual join disappears.

Everything is written with the standard library's :mod:`csv`. The scale is
thousands of rows.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping
from dataclasses import fields as dataclass_fields
from pathlib import Path

from ..image_metadata import ImageMetadata, read_records_metadata
from ..io.path import make_path_relative
from ..metadata import DESCRIPTIVE_FIELDS
from ..models import ImageRecord, Project
from .csv import find_matching_label
from .label_io import is_label_file_empty
from .selection import current_state

logger = logging.getLogger(__name__)

#: Columns identifying where an image sits, before any measured values.
IDENTITY_COLUMNS = (
    "project", "lamella", "stage", "modality", "variant",
    "state", "label_state", "label_shared", "image", "label",
)

#: The acquisition columns, taken from the dataclass so the two cannot drift.
METADATA_COLUMNS = tuple(
    field.name for field in dataclass_fields(ImageMetadata) if field.name != "source"
)

#: Project-level columns repeated on every image row, so one file answers a
#: question without a join.
PROJECT_COLUMNS = tuple(f"project_{name}" for name in DESCRIPTIVE_FIELDS)


#: One descriptive field is itself called ``project`` -- "the wider project
#: this belongs to" -- which collides with the identity column of the same
#: name. The identity keeps the plain name, since that is what ``project``
#: means when reading a row; the descriptive field is the one qualified.
#: Only that field is renamed, so the common columns stay short.
DESCRIPTIVE_COLUMN_OVERRIDES = {"project": "metadata_project"}


def _descriptive_column(field: str) -> str:
    """Return the CSV column a descriptive metadata field is written to."""
    return DESCRIPTIVE_COLUMN_OVERRIDES.get(field, field)


def _label_state(record: ImageRecord) -> str:
    """Return whether the record's label holds an annotation.

    A label existing is not evidence of annotation: the segmentation software
    writes an empty one for every image it is pointed at. An export that did
    not distinguish these would overstate how much of the archive is done.
    """
    label = find_matching_label(record)
    if label is None:
        return "missing"
    empty = is_label_file_empty(label)
    if empty is None:
        return "unreadable"
    return "empty" if empty else "annotated"


def image_rows(
    records: Iterable[ImageRecord],
    start_path: Path,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    projects: Mapping[Path, Project] | None = None,
) -> list[dict[str, object]]:
    """Return one row per image, ready to write."""
    metadata = metadata or {}
    projects = projects or {}
    rows: list[dict[str, object]] = []

    for record in records:
        label = find_matching_label(record)
        row: dict[str, object] = {
            "project": record.project,
            "lamella": record.lamella,
            "stage": record.stage,
            "modality": record.modality,
            "variant": record.variant or "base",
            "state": current_state(record),
            "label_state": _label_state(record),
            "label_shared": record.label_shared,
            "image": make_path_relative(record.image_path, start_path),
            "label": make_path_relative(label, start_path) if label else "",
        }

        entry = metadata.get(record.image_path)
        for column in METADATA_COLUMNS:
            value = getattr(entry, column, None) if entry is not None else None
            row[column] = "" if value is None else value

        project = projects.get(record.image_path)
        resolved = project.experiment_metadata if project is not None else None
        for name in DESCRIPTIVE_FIELDS:
            value = getattr(resolved, name, None) if resolved is not None else None
            row[f"project_{name}"] = "" if value is None else value
        if project is not None:
            row["project"] = project.key

        rows.append(row)
    return rows


def export_image_metadata_csv(
    records: Iterable[ImageRecord],
    start_path: Path,
    output: Path,
    *,
    include_headers: bool = True,
    projects: Mapping[Path, Project] | None = None,
) -> Path:
    """Write one row per image, with its acquisition and project metadata.

    Args:
        records: The images to export, already filtered.
        start_path: Base directory paths are written relative to.
        output: File to write.
        include_headers: Read each image's TIFF header. Costs about half a
            millisecond per image; turn it off for a purely structural listing.
        projects: Image path to project, from
            :func:`~am_segmentation_manager.aggregate.project_lookup`.

    Returns:
        The path written.
    """
    records = list(records)
    metadata = read_records_metadata(records) if include_headers else {}
    rows = image_rows(records, start_path, metadata, projects)
    columns = [*IDENTITY_COLUMNS, *METADATA_COLUMNS, *PROJECT_COLUMNS]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d image row(s) to %s", len(rows), output)
    if include_headers and len(metadata) < len(records):
        logger.warning(
            "%d image(s) had no readable header; their acquisition columns are "
            "empty", len(records) - len(metadata),
        )
    return output


def project_rows(projects: Iterable[Project], start_path: Path) -> list[dict[str, object]]:
    """Return one row per project, with counts and metadata provenance."""
    rows: list[dict[str, object]] = []
    for project in projects:
        resolved = project.experiment_metadata
        records = [
            record
            for lamella in project.lamellae
            for stage in lamella.stages
            for record in stage.images
        ]
        row: dict[str, object] = {
            "project": project.key,
            "name": project.name,
            "path": make_path_relative(project.path, start_path),
            "lamellae": len(project.lamellae),
            "stages": sum(len(lamella.stages) for lamella in project.lamellae),
            "sem_images": sum(1 for record in records if record.modality == "SEM"),
            "fib_images": sum(1 for record in records if record.modality == "FIB"),
            "corrected": sum(
                1 for record in records if current_state(record) == "corrected"
            ),
            "deselected": sum(
                1 for record in records if current_state(record) == "deselected"
            ),
        }
        for name in DESCRIPTIVE_FIELDS:
            value = getattr(resolved, name, None) if resolved is not None else None
            row[_descriptive_column(name)] = "" if value is None else value
            # Which file supplied it. Bookkeeping-looking, but it is how you
            # answer "why does this project say the sample is X" without
            # opening two YAML files.
            source = resolved.sources.get(name) if resolved is not None else None
            row[f"{_descriptive_column(name)}_source"] = source.name if source else ""
        row["todo_fields"] = " ".join(resolved.todo_fields) if resolved else ""
        rows.append(row)
    return rows


def export_project_metadata_csv(
    projects: Iterable[Project], start_path: Path, output: Path
) -> Path:
    """Write one row per project: its metadata, where each value came from, and
    what it holds."""
    rows = project_rows(projects, start_path)
    columns = [
        "project", "name", "path", "lamellae", "stages",
        "sem_images", "fib_images", "corrected", "deselected",
    ]
    for name in DESCRIPTIVE_FIELDS:
        column = _descriptive_column(name)
        columns.extend([column, f"{column}_source"])
    columns.append("todo_fields")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d project row(s) to %s", len(rows), output)
    return output
