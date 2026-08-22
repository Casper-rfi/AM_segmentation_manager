"""Exporting metadata to CSV, per image, per project and per label.

Three exports for three questions: what is in this archive and under what
conditions was it taken; what these experiments are and which still need
metadata filling in; and what the labels actually contain.

Written with the standard library's :mod:`csv` -- the scale is thousands of rows.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping
from dataclasses import fields as dataclass_fields
from pathlib import Path

from ..config import SegmentationConfig
from ..frames import group_by_acquisition
from ..image_metadata import ImageMetadata, read_records_metadata
from ..io.path import make_path_relative
from ..metadata import DESCRIPTIVE_FIELDS
from ..models import ImageRecord, Project
from .csv import find_matching_label
from .label_io import label_state
from .label_measure import LabelMeasurement, measurement_status
from .selection import current_state

logger = logging.getLogger(__name__)

#: The column naming the project *directory* an image sits in -- the thing the
#: Curate tab lists and the scanner indexes. Qualified, because the project
#: metadata has a field called ``project`` meaning the wider project the work
#: belongs to, and one collection's directories usually all belong to one of
#: those. Named once here because four files carry it.
PROJECT_IDENTITY_COLUMN = "autolamella_project"

#: Columns identifying where an image sits, before any measured values.
IDENTITY_COLUMNS = (
    PROJECT_IDENTITY_COLUMN, "lamella", "stage", "modality", "variant",
    "state", "label_state", "label_shared", "image", "label",
)

#: What a paired row says once, because both beams share it. "modality" is not
#: among them: a paired row has no single modality, which is the point of it.
SHARED_COLUMNS = (PROJECT_IDENTITY_COLUMN, "lamella", "stage", "acquisition")

#: What a paired row says twice, once per beam, under the prefixes below.
PER_BEAM_COLUMNS = (
    "variant", "state", "label_state", "label_shared", "image", "label",
)

#: How a beam's own columns are marked. Every value under one of these was read
#: from that beam's image or its label; anything unprefixed describes the
#: acquisition or the project, which the two share.
BEAM_PREFIXES = {"SEM": "sem", "FIB": "fib"}


def beam_column(modality: str, field: str) -> str:
    """Return the paired CSV's column for one beam's *field*.

    Raises:
        KeyError: If *modality* is not a beam this package knows.
    """
    return f"{BEAM_PREFIXES[modality]}_{field}"

#: The acquisition columns, taken from the dataclass so the two cannot drift.
METADATA_COLUMNS = tuple(
    field.name for field in dataclass_fields(ImageMetadata) if field.name != "source"
)

def image_metadata_column(field: str) -> str:
    """Return the per-image CSV's column for a project metadata *field*.

    Prefixed, because these repeat a project's values on every image row and a
    reader has to be able to tell them from the image's own header. The field
    already called ``project`` is left unprefixed: ``project_project`` says
    nothing the shorter name does not, and there is nothing left to
    disambiguate it from now the directory is
    :data:`PROJECT_IDENTITY_COLUMN`.
    """
    return field if field == "project" else f"project_{field}"


#: Project-level columns repeated on every image row, so one file answers a
#: question without a join.
PROJECT_COLUMNS = tuple(image_metadata_column(name) for name in DESCRIPTIVE_FIELDS)


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
            PROJECT_IDENTITY_COLUMN: record.project,
            "lamella": record.lamella,
            "stage": record.stage,
            "modality": record.modality,
            "variant": record.variant or "base",
            "state": current_state(record),
            "label_state": label_state(record),
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
            row[image_metadata_column(name)] = "" if value is None else value
        if project is not None:
            row[PROJECT_IDENTITY_COLUMN] = project.key

        rows.append(row)
    return rows


def paired_image_rows(
    records: Iterable[ImageRecord],
    start_path: Path,
    metadata: Mapping[Path, ImageMetadata] | None = None,
    projects: Mapping[Path, Project] | None = None,
) -> list[dict[str, object]]:
    """Return one row per acquisition, both beams side by side.

    The same shape as a paired training CSV and grouped by the same rule, so a
    row here describes a row there. What the two beams share -- the project, the
    lamella, the stage, the acquisition number -- is said once; everything read
    from one beam's image or label is prefixed with which beam, because
    ``dwell_time_ns`` alone would not say.

    One row per pair of imaging conditions, as the training export writes: five
    SEM conditions against one FIB image are five rows repeating the FIB
    columns.
    """
    metadata = metadata or {}
    projects = projects or {}
    rows: list[dict[str, object]] = []

    for acquisition in group_by_acquisition(records):
        for sem in acquisition.sem or (None,):
            for fib in acquisition.fib or (None,):
                present = next(
                    record for record in (sem, fib) if record is not None
                )
                row: dict[str, object] = {
                    PROJECT_IDENTITY_COLUMN: present.project,
                    "lamella": present.lamella,
                    "stage": present.stage,
                    "acquisition": (
                        "" if acquisition.image_number is None
                        else acquisition.image_number
                    ),
                }
                for modality, record in (("SEM", sem), ("FIB", fib)):
                    _fill_beam(row, modality, record, start_path, metadata)
                _fill_project(row, projects.get(present.image_path))
                rows.append(row)
    return rows


def _fill_beam(
    row: dict[str, object],
    modality: str,
    record: ImageRecord | None,
    start_path: Path,
    metadata: Mapping[Path, ImageMetadata],
) -> None:
    """Write one beam's columns into a paired row, blank where it has none."""
    label = find_matching_label(record) if record is not None else None
    values: dict[str, object] = {
        "variant": (record.variant or "base") if record is not None else "",
        "state": current_state(record) if record is not None else "",
        "label_state": label_state(record) if record is not None else "",
        "label_shared": record.label_shared if record is not None else "",
        "image": (
            make_path_relative(record.image_path, start_path)
            if record is not None else ""
        ),
        "label": make_path_relative(label, start_path) if label else "",
    }
    for field in PER_BEAM_COLUMNS:
        row[beam_column(modality, field)] = values[field]

    entry = metadata.get(record.image_path) if record is not None else None
    for column in METADATA_COLUMNS:
        value = getattr(entry, column, None) if entry is not None else None
        row[beam_column(modality, column)] = "" if value is None else value


def _fill_project(row: dict[str, object], project: Project | None) -> None:
    """Write the project's own columns, which both beams share."""
    resolved = project.experiment_metadata if project is not None else None
    for name in DESCRIPTIVE_FIELDS:
        value = getattr(resolved, name, None) if resolved is not None else None
        row[image_metadata_column(name)] = "" if value is None else value
    if project is not None:
        row[PROJECT_IDENTITY_COLUMN] = project.key


def paired_columns() -> list[str]:
    """Return the header of a paired per-image metadata CSV."""
    columns = list(SHARED_COLUMNS)
    for modality in BEAM_PREFIXES:
        columns += [beam_column(modality, field) for field in PER_BEAM_COLUMNS]
        columns += [beam_column(modality, field) for field in METADATA_COLUMNS]
    return columns + list(PROJECT_COLUMNS)


def export_image_metadata_csv(
    records: Iterable[ImageRecord],
    start_path: Path,
    output: Path,
    *,
    include_headers: bool = True,
    projects: Mapping[Path, Project] | None = None,
    paired: bool = False,
) -> Path:
    """Write the acquisition and project metadata of the images given.

    Args:
        records: The images to export, already filtered.
        start_path: Base directory paths are written relative to.
        output: File to write.
        include_headers: Read each image's TIFF header. Costs about half a
            millisecond per image; turn it off for a purely structural listing.
        projects: Image path to project, from
            :func:`~am_segmentation_manager.aggregate.project_lookup`.
        paired: Write one row per acquisition with both beams side by side,
            rather than one row per image. What a selection holding both
            modalities produces, matching the training export's shape. A file
            with one row per image file -- which a repository deposition
            usually wants -- is what choosing a single modality gives.

    Returns:
        The path written.
    """
    records = list(records)
    metadata = read_records_metadata(records) if include_headers else {}
    if paired:
        rows = paired_image_rows(records, start_path, metadata, projects)
        columns = paired_columns()
    else:
        rows = image_rows(records, start_path, metadata, projects)
        columns = [*IDENTITY_COLUMNS, *METADATA_COLUMNS, *PROJECT_COLUMNS]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d row(s) to %s", len(rows), output)
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
            PROJECT_IDENTITY_COLUMN: project.key,
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
            row[name] = "" if value is None else value
            # Which file supplied it. Bookkeeping-looking, but it is how you
            # answer "why does this project say the sample is X" without
            # opening two YAML files.
            source = resolved.sources.get(name) if resolved is not None else None
            row[f"{name}_source"] = source.name if source else ""
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
        PROJECT_IDENTITY_COLUMN, "name", "path", "lamellae", "stages",
        "sem_images", "fib_images", "corrected", "deselected",
    ]
    # Unprefixed here: every column in this file is about one project already,
    # so "project_sample" would be saying it twice.
    for name in DESCRIPTIVE_FIELDS:
        columns.extend([name, f"{name}_source"])
    columns.append("todo_fields")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d project row(s) to %s", len(rows), output)
    return output


# ---------------------------------------------------------------------------
# Label measurements
#
# One row per image, not per label. A shared label therefore appears once per
# condition, with identical numbers -- which is what lets a row be traced back
# to the frame it came from. Anything totalling this file must deduplicate on
# the `label` column first, or it will count one crack several times.
# ---------------------------------------------------------------------------

#: Columns describing the measurement itself, after the identity columns and
#: the per-class areas.
MEASUREMENT_COLUMNS = (
    "label_status",
    "label_width",
    "label_height",
    "label_pixel_size_nm",
    "crack_count",
    "crack_area_um2_total",
    "crack_area_um2_largest",
    "crack_areas_um2",
    "min_crack_area_um2",
)


def area_column(config: SegmentationConfig, class_index: int) -> str:
    """Return the CSV column an individual class's area is written to.

    Named after the class rather than its index, since a reader knows what a
    lamella is and not what class 2 is. Characters that would make the column
    awkward to use from a script are replaced, so ``crack/hole`` becomes
    ``area_um2_crack_hole``.
    """
    name = config.name(class_index).strip().lower()
    safe = "".join(character if character.isalnum() else "_" for character in name)
    return f"area_um2_{safe.strip('_')}"


def measurement_rows(
    records: Iterable[ImageRecord],
    measurements: Mapping[Path, LabelMeasurement],
    config: SegmentationConfig,
    start_path: Path,
    projects: Mapping[Path, Project] | None = None,
) -> list[dict[str, object]]:
    """Return one row per image, carrying its label's measurements."""
    projects = projects or {}
    classes = config.classes_to_measure
    rows: list[dict[str, object]] = []

    for record in records:
        measurement = measurements.get(record.image_path)
        label = measurement.path if measurement is not None else find_matching_label(record)
        project = projects.get(record.image_path)
        row: dict[str, object] = {
            PROJECT_IDENTITY_COLUMN: (
                project.key if project is not None else record.project
            ),
            "lamella": record.lamella,
            "stage": record.stage,
            "modality": record.modality,
            "variant": record.variant or "base",
            "state": current_state(record),
            "label_shared": record.label_shared,
            "image": make_path_relative(record.image_path, start_path),
            "label": make_path_relative(label, start_path) if label else "",
            "label_status": measurement_status(measurement),
        }

        for class_index in classes:
            area = measurement.areas_um2.get(class_index) if measurement else None
            row[area_column(config, class_index)] = "" if area is None else f"{area:.6g}"

        if measurement is None:
            for column in MEASUREMENT_COLUMNS:
                row.setdefault(column, "")
            rows.append(row)
            continue

        row["label_width"] = measurement.pixels.width
        row["label_height"] = measurement.pixels.height
        row["label_pixel_size_nm"] = (
            "" if measurement.pixel_size_nm is None else f"{measurement.pixel_size_nm:.6g}"
        )
        row["crack_count"] = "" if measurement.crack_count is None else measurement.crack_count
        row["crack_area_um2_total"] = f"{sum(measurement.crack_areas_um2):.6g}"
        row["crack_area_um2_largest"] = (
            f"{measurement.crack_areas_um2[0]:.6g}" if measurement.crack_areas_um2 else "0"
        )
        # Every qualifying component, space separated, so the size distribution
        # survives into the file a script will read. One column rather than one
        # row per crack keeps this file joinable to the per-image export.
        row["crack_areas_um2"] = " ".join(
            f"{area:.6g}" for area in measurement.crack_areas_um2
        )
        row["min_crack_area_um2"] = f"{measurement.min_crack_area_um2:.6g}"
        rows.append(row)

    return rows


def export_label_measurements_csv(
    records: Iterable[ImageRecord],
    measurements: Mapping[Path, LabelMeasurement],
    config: SegmentationConfig,
    start_path: Path,
    output: Path,
    projects: Mapping[Path, Project] | None = None,
) -> Path:
    """Write one row per measured image, with what its label contains.

    Images with no label are left out -- this is a file of measurements. A
    shared label still appears once per imaging condition, so anything
    totalling the file must deduplicate on the ``label`` column first.

    Args:
        records: The images measured, already filtered.
        measurements: Image path to measurement, from
            :func:`~am_segmentation_manager.operations.label_measure.measure_records`.
        config: Supplies which classes get an area column, and their names.
        start_path: Base directory paths are written relative to.
        output: File to write.
        projects: Image path to project, from
            :func:`~am_segmentation_manager.aggregate.project_lookup`.

    Returns:
        The path written.
    """
    records = list(records)
    # Only images that have a label. On the reference collection a full-archive
    # run selects 2670 images and 238 labels, so writing a row per image would
    # make this file nine parts empty. Which images lack a label is a question
    # export_image_metadata_csv already answers, in its label_state column.
    measured = [record for record in records if record.image_path in measurements]
    rows = measurement_rows(measured, measurements, config, start_path, projects)
    columns = [
        PROJECT_IDENTITY_COLUMN, "lamella", "stage", "modality", "variant",
        "state", "label_shared", "image", "label",
        *(area_column(config, index) for index in config.classes_to_measure),
        *MEASUREMENT_COLUMNS,
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d measurement row(s) to %s", len(rows), output)
    if len(measured) < len(records):
        logger.info(
            "%d of %d image(s) have no label and are not in the file",
            len(records) - len(measured), len(records),
        )
    return output


def export_lamella_quantification_csv(
    summaries: Iterable, config: SegmentationConfig, output: Path
) -> Path:
    """Write one row per lamella, at the level the question is asked at."""
    summaries = list(summaries)
    classes = config.classes_to_measure
    columns = [
        PROJECT_IDENTITY_COLUMN, "lamella", "labels", "measured", "empty",
        "unmeasurable",
        "has_crack", "cracked_labels", "cracks",
        "crack_area_um2", "largest_crack_um2",
        *(area_column(config, index) for index in classes),
    ]

    rows = []
    for summary in summaries:
        row: dict[str, object] = {
            PROJECT_IDENTITY_COLUMN: summary.project,
            "lamella": summary.lamella,
            "labels": summary.labels,
            "measured": summary.measured,
            "empty": summary.empty,
            "unmeasurable": summary.unmeasurable,
            "has_crack": summary.has_crack,
            "cracked_labels": summary.cracked,
            "cracks": summary.cracks,
            "crack_area_um2": f"{summary.crack_area_um2:.6g}",
            "largest_crack_um2": f"{summary.largest_crack_um2:.6g}",
        }
        for class_index in classes:
            area = summary.areas_um2.get(class_index)
            row[area_column(config, class_index)] = "" if area is None else f"{area:.6g}"
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d lamella row(s) to %s", len(rows), output)
    return output
