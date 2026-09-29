"""Writing the image and label lists a training run reads.

One row per acquisition rather than per file: both beams of one scene go side by
side (``sem_image, sem_label, fib_image, fib_label``), a single-beam selection
writes ``image, label``, and a cell is blank where that beam has no image or no
label worth pairing with one.

Two rules nothing may get past: an empty label, or one that could not be read to
tell, is never written as a training pair; and a split never cuts a group -- an
acquisition at the finest, a lamella or a project where asked.
``docs/decisions.md`` § Counting and export says why.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..frames import group_by_acquisition
from ..io.path import exported_relative_path
from ..io.tables import write_csv
from ..models import ImageRecord
from .label_groups import condition_sort_key
from .label_io import has_image_stack_support, is_label_file_empty
from .training_manifest import manifest_path, write_training_manifest
from .training_split import ACQUISITION, DEFAULT_TRAIN_RATIO, Grouping, split_grouped

logger = logging.getLogger(__name__)

#: Header of a training CSV covering one beam.
SINGLE_COLUMNS = ("image", "label")

#: Header of a training CSV covering both. One row is one acquisition under one
#: pair of imaging conditions, so a model can be trained on either beam from
#: one file, or on the two together. A cell is blank where that beam has no
#: image, or no label worth pairing with one.
PAIRED_COLUMNS = ("sem_image", "sem_label", "fib_image", "fib_label")


@dataclass(frozen=True)
class TrainingRow:
    """One row of a training CSV, and what it must not be split away from.

    Attributes:
        values: The cells, in the order of the header they are written under.
        group: What keeps related rows on one side of the train/validation
            split. Splitting near-identical images across the two would have
            the validation score measure how well the model recognises images
            it has effectively already seen.
    """

    values: tuple[str, ...]
    group: str


def find_matching_label(record: ImageRecord) -> Path | None:
    """Return the record's label, or None if it has none on disk.

    A group has one label in one place, so there is nothing to choose between.
    """
    for label in record.labels:
        if label.exists():
            return label
    return None


def keep_one_condition_per_scene(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    """Return one record per scene, dropping the other imaging conditions.

    Records sharing a label are the same scene captured under different imaging
    conditions, so exporting all of them multiplies a dataset by near-identical
    copies. This keeps each scene's primary condition, chosen by
    :func:`~am_segmentation_manager.operations.label_groups.primary_record`.

    Records with no label are kept as they are, since nothing identifies them
    as sharing a scene with anything else.
    """
    candidates = list(records)
    primary: dict[str, ImageRecord] = {}
    for record in candidates:
        if not record.labels:
            continue
        key = str(record.labels[0])
        current = primary.get(key)
        if current is None or condition_sort_key(record) < condition_sort_key(current):
            primary[key] = record

    chosen = {id(record) for record in primary.values()}
    kept = [
        record for record in candidates
        if not record.labels or id(record) in chosen
    ]
    if len(kept) < len(candidates):
        logger.info(
            "Exporting one condition per scene: %d of %d image(s); pass "
            "--conditions all to include every condition",
            len(kept), len(candidates),
        )
    return kept


def _image_cell(record: ImageRecord | None, start_path: Path) -> str:
    """Return an image's recorded path, or blank where that beam has none."""
    return "" if record is None else exported_relative_path(record.image_path, start_path)


def _label_cell(
    record: ImageRecord | None,
    start_path: Path,
    include_empty_labels: bool,
    is_empty: Callable[[Path], bool | None],
) -> str:
    """Return the path of a label worth pairing with an image, or blank.

    Blank covers "no label", "a label holding nothing but background" and "a
    label whose contents could not be read", which are different facts about
    the archive but the same instruction to a training run: nothing here is
    known to be worth learning from. An unreadable label is left out even
    with *include_empty_labels*, which asks for blank masks known to be blank.
    Which it was is the preview's to report, not the file's.
    """
    if record is None:
        return ""
    label = find_matching_label(record)
    if label is None:
        return ""
    empty = is_empty(label)
    if empty is None or (empty and not include_empty_labels):
        return ""
    return exported_relative_path(label, start_path)


def unreadable_labels(
    records: Iterable[ImageRecord],
    is_empty: Callable[[Path], bool | None] = is_label_file_empty,
) -> list[Path]:
    """Return the labels of *records* whose contents could not be read, once each.

    Unreadable here includes a label that could not be opened because the image
    stack is not installed: either way nobody can say whether it is blank.
    """
    found: dict[Path, None] = {}
    for record in records:
        label = find_matching_label(record)
        if label is not None and label not in found and is_empty(label) is None:
            found[label] = None
    return list(found)


def build_training_rows(
    records: Iterable[ImageRecord],
    start_path: Path,
    *,
    paired: bool,
    include_empty_labels: bool = False,
    is_empty: Callable[[Path], bool | None] = is_label_file_empty,
    grouping: Grouping = Grouping(),
) -> list[TrainingRow]:
    """Return the rows a training export would write.

    One function for the preview and for the file, because a count on screen
    that disagreed with what was written would make the preview worse than
    useless.

    Paired, a row is one acquisition under one pair of imaging conditions --
    the cross product, so five SEM conditions against one FIB image are five
    rows repeating the FIB path. That is what "one image per label" is for
    where it is not wanted. A row survives only if at least one beam has a
    label worth pairing with: a row of two images and no annotation teaches
    nothing.

    Unpaired, a row is one image and its label, exactly as before.

    Args:
        records: The images to write, already filtered.
        start_path: Base directory paths are written relative to.
        paired: Write both beams per row, under :data:`PAIRED_COLUMNS`.
        include_empty_labels: Pair an image with a label holding only
            background. Off by default; see :func:`export_training_csv`.
        is_empty: Answers whether a label holds nothing but background. Passed
            in so a caller holding the answers already -- the export view, which
            reads every label once -- does not open them all again.
        grouping: What each row's group is. At the acquisition, the label
            of a single-beam row and the acquisition of a paired one.
    """
    records = list(records)

    def group_of(record: ImageRecord, finest: str) -> str:
        return finest if grouping.key is None else str(grouping.key(record))

    if not paired:
        rows = []
        for record in records:
            label = _label_cell(record, start_path, include_empty_labels, is_empty)
            if not label:
                continue
            rows.append(
                TrainingRow(
                    (_image_cell(record, start_path), label),
                    group=group_of(record, label),
                )
            )
        return rows

    rows = []
    for acquisition in group_by_acquisition(records):
        # Every record of one acquisition is in one stage, so any of them
        # names its lamella and its project.
        group = group_of(acquisition.records[0], acquisition.identity)
        # A beam with no image still yields one blank column rather than no
        # row, so an acquisition captured on one beam is exported rather than
        # dropped for what it does not have.
        for sem in acquisition.sem or (None,):
            for fib in acquisition.fib or (None,):
                values = (
                    _image_cell(sem, start_path),
                    _label_cell(sem, start_path, include_empty_labels, is_empty),
                    _image_cell(fib, start_path),
                    _label_cell(fib, start_path, include_empty_labels, is_empty),
                )
                if values[1] or values[3]:
                    rows.append(TrainingRow(values, group=group))
    return rows


def split_rows(
    rows: Sequence[TrainingRow],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Split built rows, keeping each row's group whole."""
    return split_grouped(rows, lambda row: row.group, train_ratio, seed)


def export_training_csv(
    records: Iterable[ImageRecord],
    start_path: Path,
    output_dir: Path,
    prefix: str = "dataset",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
    include_empty_labels: bool = False,
    conditions: str = "primary",
    paired: bool = False,
    is_empty: Callable[[Path], bool | None] = is_label_file_empty,
    *,
    grouping: Grouping = Grouping(),
    overwrite: bool = False,
    scope: str = "",
    user: str | None = None,
) -> tuple[Path, Path | None]:
    """Export image and label paths to CSV files for ML model training.

    Writes ``<output_dir>/<prefix>_train.csv``, ``<output_dir>/<prefix>_val.csv``
    unless *train_ratio* is 1.0, and last ``<prefix>_manifest.yaml`` saying what
    they were made from.

    Two shapes, decided by *paired*::

        image,label
        rel/path/to/image.tif,rel/path/to/label.tif

        sem_image,sem_label,fib_image,fib_label
        rel/sem.tif,rel/sem_label.tif,rel/fib.tif,rel/fib_label.tif

    Args:
        records: ImageRecords to process, already filtered.
        start_path: Base directory used to resolve relative paths.
        output_dir: Target directory where the CSV files are written.
        prefix: Filename prefix for the output files.
        train_ratio: Ratio of rows assigned to training (0.0 to 1.0).
        seed: Optional seed for a reproducible split.
        include_empty_labels: Pair an image with a label holding nothing but
            background. A label that could not be read is left out either way,
            and counted. Off by default: the segmentation software writes an
            empty label for every image it is pointed at, so an existing label
            file is not evidence of annotation, and exporting one actively
            teaches a model that the structure is background. Turn it on only
            to build negative examples deliberately.
        conditions: ``"primary"`` (default) exports one image per scene, or
            ``"all"`` every imaging condition of it. See
            :func:`keep_one_condition_per_scene`.
        paired: Write both beams on one row. What a selection holding both
            modalities should produce -- a file mixing SEM and FIB rows under
            one ``image`` column is not something anyone trains on.
        is_empty: Answers whether a label holds only background; see
            :func:`build_training_rows`.
        grouping: What the split never divides, recorded in the manifest.
        overwrite: Replace files an earlier export wrote under *prefix*.
        scope: What the records were narrowed to, recorded in the manifest.
        user: The name to record as writing it; see :func:`write_training_manifest`.

    Returns:
        ``(train_csv_path, val_csv_path)``. The second is None when
        *train_ratio* is 1.0.

    Raises:
        FileExistsError: If a file this would write exists and *overwrite* is
            False. Nothing has been read or written by then.
        NoRelativePath: If a file has no path relative to *start_path*, before
            anything is written.
        ValueError: If *conditions* is not a known policy.
    """
    if conditions not in ("primary", "all"):
        raise ValueError(f"Unknown conditions policy: {conditions}")

    train_path = output_dir / f"{prefix}_train.csv"
    val_path = output_dir / f"{prefix}_val.csv" if train_ratio < 1.0 else None
    manifest = manifest_path(output_dir, prefix)
    existing = [
        path for path in (train_path, val_path, manifest)
        if path is not None and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"{', '.join(path.name for path in existing)} already in {output_dir}; "
            f"nothing was written"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(records)
    if conditions == "primary":
        selected = keep_one_condition_per_scene(selected)

    rows = build_training_rows(
        selected,
        start_path,
        paired=paired,
        include_empty_labels=include_empty_labels,
        is_empty=is_empty,
        grouping=grouping,
    )
    if not rows:
        logger.warning("No image/label pairs found to export")
    unreadable = unreadable_labels(selected, is_empty)
    if unreadable:
        logger.warning(
            "Left out %d label(s) whose contents could not be read, so could not "
            "be told blank or annotated%s", len(unreadable),
            "" if has_image_stack_support() else
            ": the image stack is not installed; install the 'all' extra",
        )
    if paired:
        left_out = "label_cells_left_blank", sum(
            1 for row in rows for cell in (row.values[1], row.values[3]) if not cell
        )
    else:
        left_out = "images_left_out", len(selected) - len(rows)
    if not include_empty_labels:
        # Reported either way, because it is the same rule with two shapes: a
        # blank mask is never paired with an image, which drops the row where
        # there is one beam and empties a cell where there are two. Silence
        # would be a training set quietly short of what somebody labelled.
        if paired:
            if blank := left_out[1]:
                logger.info(
                    "Left %d label cell(s) blank: no label, or one holding only "
                    "background; pass --include-empty-labels to pair the latter "
                    "anyway", blank,
                )
        elif (dropped := left_out[1]) > 0:
            logger.info(
                "Excluded %d image(s) with no label, or whose label holds only "
                "background; pass --include-empty-labels to export the latter",
                dropped,
            )

    train_rows, val_rows = split_rows(rows, train_ratio=train_ratio, seed=seed)
    groups = {row.group for row in rows}
    if len(groups) < len(rows) or grouping.level != ACQUISITION:
        logger.info(
            "Split %d row(s) as %d group(s), keeping each %s on one side: "
            "%d training, %d validation",
            len(rows), len(groups), grouping.level, len(train_rows), len(val_rows),
        )
    if len(groups) == 1 and 0.0 < train_ratio < 1.0:
        logger.warning(
            "Every row is in one %s, so all of them went to %s; keep a finer "
            "level together to fill both sides", grouping.level,
            "training" if train_rows else "validation",
        )

    columns = PAIRED_COLUMNS if paired else SINGLE_COLUMNS

    def _write_csv(path: Path, written: list[TrainingRow]) -> Path:
        write_csv(path, columns, (row.values for row in written), overwrite=overwrite)
        logger.info("Wrote %d row(s) to %s", len(written), path)
        return path

    _write_csv(train_path, train_rows)
    if val_path is not None:
        _write_csv(val_path, val_rows)
    write_training_manifest(
        manifest,
        source=start_path,
        scope=scope,
        files={"train": train_path, "validation": val_path},
        columns=columns,
        train_ratio=train_ratio,
        seed=seed,
        keep_together=grouping.level,
        conditions=conditions,
        include_empty_labels=include_empty_labels,
        counts={
            "images": len(selected),
            "rows": len(rows),
            "groups": len(groups),
            "train": len(train_rows),
            "validation": len(val_rows),
            left_out[0]: left_out[1],
            "unreadable_labels_left_out": len(unreadable),
        },
        user=user,
    )
    return train_path, val_path
