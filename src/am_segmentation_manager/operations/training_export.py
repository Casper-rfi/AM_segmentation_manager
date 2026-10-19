"""Writing the image and label lists a training run reads: one train/validation
pair, or one per fold for cross-validation, and with them each image's side and metadata.

A row is one acquisition: both beams side by side (``sem_image, sem_label,
fib_image, fib_label``), or ``image, label`` for one beam, a cell blank where that
beam has no image or no label worth pairing. Two rules nothing may get past: an
empty or unreadable label is never written as a training pair, and a split never
cuts a group -- an acquisition, or a lamella or project where asked.
``docs/decisions.md`` § Counting and export says why.
"""
from __future__ import annotations

import csv
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..families import FamiliesLookup
from ..frames import group_by_acquisition
from ..io.path import exported_relative_path
from ..io.tables import write_csv
from ..models import ImageRecord
from .label_groups import find_matching_label, primary_record
from .label_io import has_image_stack_support, is_label_file_empty
from .selection import current_state
from .training_manifest import manifest_path, read_training_manifest, write_training_manifest
from .training_split import (
    ACQUISITION,
    DEFAULT_TRAIN_RATIO,
    TRAINING,
    VALIDATION,
    Balance,
    Grouping,
    assign_folds,
    fold_training,
    split_grouped,
    split_pinned,
    split_validating,
)

logger = logging.getLogger(__name__)

#: Header of a training CSV covering one beam.
SINGLE_COLUMNS = ("image", "label")

#: Header of a training CSV covering both. One row is one acquisition under one
#: pair of imaging conditions, so a model can be trained on either beam from
#: one file, or on the two together. A cell is blank where that beam has no
#: image, or no label worth pairing with one.
PAIRED_COLUMNS = ("sem_image", "sem_label", "fib_image", "fib_label")


#: The column of an images file saying which side of the split an image is on.
SPLIT_COLUMN = "split"

#: Describes the images of an export for its images file: given them in order,
#: returns the columns and a row for each. Passed in, the metadata export that
#: knows how being built on this module.
DescribeImages = Callable[
    [Sequence[ImageRecord]], tuple[Sequence[str], Sequence[Mapping[str, object]]]
]


def images_path(output_dir: Path, prefix: str) -> Path:
    """Return where the images file for *prefix* lives, existing or not."""
    return output_dir / f"{prefix}_images.csv"


@dataclass(frozen=True)
class TrainingRow:
    """One row of a training CSV, and what it must not be split away from.

    Attributes:
        values: The cells, in the order of the header they are written under.
        records: The images the row names, one per beam it has.
        labelled: Those of *records* whose label the row names.
        group: What keeps related rows on one side of the train/validation
            split. Splitting near-identical images across the two would have
            the validation score measure how well the model recognises images
            it has effectively already seen.
    """

    values: tuple[str, ...]
    group: str
    records: tuple[ImageRecord, ...] = ()
    labelled: tuple[ImageRecord, ...] = ()


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
    scenes: dict[str, list[ImageRecord]] = {}
    for record in candidates:
        if record.labels:
            scenes.setdefault(str(record.labels[0]), []).append(record)

    families = FamiliesLookup()
    chosen = {id(primary_record(members, families)) for members in scenes.values()}
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
        grouping: What each row's group is. At the acquisition, the
            acquisition the row's images are of; a single-beam image of no
            known beam is a group of its label.
    """
    records = list(records)

    def group_of(record: ImageRecord, finest: str) -> str:
        return finest if grouping.key is None else str(grouping.key(record))

    if not paired:
        # Two families of one acquisition have a label each, and are still
        # one scene.
        acquisition_of = {
            id(record): acquisition.identity
            for acquisition in group_by_acquisition(records)
            for record in acquisition.records
        }
        rows = []
        for record in records:
            label = _label_cell(record, start_path, include_empty_labels, is_empty)
            if not label:
                continue
            rows.append(
                TrainingRow(
                    (_image_cell(record, start_path), label),
                    group=group_of(record, acquisition_of.get(id(record), label)),
                    records=(record,),
                    labelled=(record,),
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
                    rows.append(TrainingRow(
                        values, group=group,
                        records=tuple(beam for beam in (sem, fib) if beam is not None),
                        labelled=tuple(
                            beam for beam, label in ((sem, values[1]), (fib, values[3]))
                            if beam is not None and label
                        ),
                    ))
    return rows


def split_rows(
    rows: Sequence[TrainingRow],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Split built rows, keeping each row's group whole."""
    return split_grouped(rows, lambda row: row.group, train_ratio, seed)


def image_sides(
    sides: Sequence[tuple[str, Sequence[TrainingRow]]],
) -> dict[Path, str]:
    """Return the side each image is on, from each side's name and rows.

    An image takes the side of the first row naming it, every row naming it
    being of one acquisition, which no split divides.
    """
    found: dict[Path, str] = {}
    for side, rows in sides:
        for row in rows:
            for record in row.records:
                found.setdefault(record.image_path, side)
    return found


def earlier_sides(manifest_file: Path) -> dict[str, str]:
    """Return the side each image of an earlier export was on, by the path its lists name it by.

    Read from the lists rather than the images file, which every export has.

    Raises:
        ValueError: If *manifest_file* is not a manifest, is of folds, or has
            no validation list to hold on to.
    """
    manifest = read_training_manifest(manifest_file)
    if manifest.folds:
        raise ValueError(f"{manifest_file.name} is a cross-validation export, which has no one validation list")
    if manifest.train is None or manifest.validation is None:
        raise ValueError(f"{manifest_file.name} has no validation list to hold on to")
    sides: dict[str, str] = {}
    for side, path in ((TRAINING, manifest.train), (VALIDATION, manifest.validation)):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for column, cell in row.items():
                        if column and column.endswith("image") and cell:
                            sides[cell] = side
        except OSError as error:
            raise ValueError(f"Could not read {path}: {error}") from error
    return sides


def divide_rows(
    rows: Sequence[TrainingRow],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int | None = 42,
    validate_on: str | None = None,
    earlier: Mapping[str, str] | None = None,
    balance: Balance | None = None,
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    """Divide built rows as an export writes them, for the preview and the file alike.

    Args:
        validate_on: The workflow state a row's labels must all be in to be
            validated on, or None for any. A row in another state trains, or
            is left out where its group validates.
        earlier: The side each image of an earlier export was on, from
            :func:`earlier_sides`; a group is then kept on the side it was.
        balance: What the validation side is given a share of, each row
            counting as its first image does; unbalanced where omitted.

    Returns:
        ``(training, validation, left_out)``; nothing is left out without *validate_on*.

    Raises:
        DividedGroup: If *earlier* put one group on both sides.
        ValueError: If *earlier* names none of these rows' images.
    """
    def validates(row: TrainingRow) -> bool:
        return validate_on is None or all(
            current_state(record) == validate_on for record in row.labelled
        )

    def group(row: TrainingRow) -> str:
        return row.group

    stratum = None if balance is None else (lambda row: balance.key(row.records[0]))

    if earlier is not None:
        def pinned(row: TrainingRow) -> str | None:
            # A paired row names an image in every other cell.
            cells = row.values[::2] if len(row.values) == len(PAIRED_COLUMNS) else row.values[:1]
            return next((earlier[cell] for cell in cells if cell in earlier), None)

        if rows and not any(pinned(row) for row in rows):
            raise ValueError(
                "The earlier export names none of these images; it was made from "
                "another folder, or of other images"
            )
        return split_pinned(rows, group, pinned, train_ratio, seed, validates, stratum)
    if validate_on is None and stratum is None:
        return (*split_rows(rows, train_ratio, seed), [])
    return split_validating(rows, group, validates, train_ratio, seed, stratum)


class LeftoverFiles(FileExistsError):
    """Files of an earlier export are beside this one's, and would outlive it.

    Its own type because replacing is no answer: the files are ones this export
    would not write, so ``overwrite`` leaves them where they are.
    """


def fold_paths(output_dir: Path, prefix: str, folds: int) -> list[tuple[Path, Path]]:
    """Return the ``(train, validation)`` files of each fold, numbered from 1."""
    return [
        (output_dir / f"{prefix}_fold{fold}_train.csv", output_dir / f"{prefix}_fold{fold}_val.csv")
        for fold in range(1, folds + 1)
    ]


def _refuse_what_is_in_the_way(
    output_dir: Path, prefix: str, writing: Iterable[Path | None], overwrite: bool
) -> None:
    """Raise before anything is read if an earlier export's files are in the way.

    A file this export would write is replaced only with *overwrite*. A file of
    the prefix it would *not* write -- a validation list, when this one trains
    on everything, or fold 5 when it writes 3 -- is refused even then: it would
    be left beside a manifest that no longer names it, and a run would take it
    for this export's.
    """
    written = [path for path in writing if path is not None]
    existing = [path for path in written if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{', '.join(path.name for path in existing)} already in {output_dir}; "
            f"nothing was written"
        )
    names = {
        path.name for path in (
            output_dir / f"{prefix}_train.csv", output_dir / f"{prefix}_val.csv",
            images_path(output_dir, prefix), manifest_path(output_dir, prefix),
        )
    }
    folds = re.compile(rf"{re.escape(prefix)}_fold\d+_(?:train|val)\.csv")
    leftover = sorted(
        path for path in output_dir.iterdir()
        if (path.name in names or folds.fullmatch(path.name)) and path not in written
    ) if output_dir.is_dir() else []
    if leftover:
        raise LeftoverFiles(
            f"{', '.join(path.name for path in leftover)} in {output_dir} belong to an "
            f"earlier export this one would not replace; move them or choose another "
            f"prefix. Nothing was written"
        )


def _rows_to_write(
    records: Iterable[ImageRecord], request: TrainingExportRequest
) -> tuple[list[TrainingRow], dict[str, int]]:
    """Return the rows an export writes, and the counts its manifest records.

    Logs what was left out and why, which is the same for every way of
    dividing the rows.
    """
    paired, include_empty_labels = request.paired, request.include_empty_labels
    selected = list(records)
    if request.conditions == "primary":
        selected = keep_one_condition_per_scene(selected)

    rows = build_training_rows(
        selected,
        request.start_path,
        paired=paired,
        include_empty_labels=include_empty_labels,
        is_empty=request.is_empty,
        grouping=request.grouping,
    )
    if not rows:
        logger.warning("No image/label pairs found to export")
    unreadable = unreadable_labels(selected, request.is_empty)
    if unreadable:
        logger.warning(
            "Left out %d label(s) whose contents could not be read, so could not "
            "be told blank or annotated%s", len(unreadable),
            "" if has_image_stack_support() else
            ": the image stack is not installed; install the 'gui' extra",
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
    return rows, {
        "images": len(selected),
        "rows": len(rows),
        "groups": len({row.group for row in rows}),
        left_out[0]: left_out[1],
        "unreadable_labels_left_out": len(unreadable),
    }


def _described(
    describe: DescribeImages | None, sides: Sequence[tuple[str, Sequence[TrainingRow]]]
) -> tuple[list[str], list[dict[str, object]]] | None:
    """Return the images file's columns and rows, or None where no file is asked for.

    An image is listed once, under the side its first row is on: every row
    naming it is of one acquisition, which no split divides.

    Args:
        sides: Each side of the split by what it is called, with its rows.
    """
    if describe is None:
        return None
    seen: dict[Path, tuple[str, ImageRecord]] = {}
    for side, rows in sides:
        for row in rows:
            for record in row.records:
                seen.setdefault(record.image_path, (side, record))
    columns, described = describe([record for _, record in seen.values()])
    return (
        [SPLIT_COLUMN, *columns],
        [{SPLIT_COLUMN: side, **row} for (side, _), row in zip(seen.values(), described)],
    )


def _write_images(
    path: Path, described: tuple[list[str], list[dict[str, object]]] | None, overwrite: bool
) -> Path | None:
    """Write the images file where one was described, and return it."""
    if described is None:
        return None
    columns, rows = described
    write_csv(path, columns, rows, overwrite=overwrite)
    logger.info("Wrote %d image(s) to %s", len(rows), path)
    return path


def _write_rows(
    path: Path, columns: Sequence[str], rows: Sequence[TrainingRow], overwrite: bool
) -> Path:
    write_csv(path, columns, (row.values for row in rows), overwrite=overwrite)
    logger.info("Wrote %d row(s) to %s", len(rows), path)
    return path


@dataclass(frozen=True)
class TrainingExportRequest:
    """What a training export is made of and how it is split, wherever it is written.

    The command line and the Export tab each build one, and a preview of a split
    takes the same one its export does. Where the files go, and whether they may
    replace an earlier export's, are asked when writing.

    Attributes:
        start_path: Base directory used to resolve relative paths.
        prefix: Filename prefix for the output files.
        conditions: ``"primary"`` (default) exports one image per scene, or
            ``"all"`` every imaging condition of it. See
            :func:`keep_one_condition_per_scene`.
        paired: Write both beams on one row. What a selection holding both
            modalities should produce -- a file mixing SEM and FIB rows under
            one ``image`` column is not something anyone trains on.
        include_empty_labels: Pair an image with a label holding nothing but
            background. A label that could not be read is left out either way,
            and counted. Off by default: the segmentation software writes an
            empty label for every image it is pointed at, so an existing label
            file is not evidence of annotation, and exporting one actively
            teaches a model that the structure is background. Turn it on only
            to build negative examples deliberately.
        is_empty: Answers whether a label holds only background; see
            :func:`build_training_rows`.
        grouping: What the split never divides, recorded in the manifest.
        train_ratio: Ratio of rows assigned to training (0.0 to 1.0). Not
            used by a fold export.
        seed: Optional seed for a reproducible split.
        validate_on: The workflow state validated on, or None for any. Rows in
            another state train, except those whose group validates, which
            are left out and counted; see :func:`divide_rows`. Not used by a
            fold export.
        validation_from: The manifest of an earlier export of this collection
            whose split to keep: what it validated on validates again, what
            it trained on trains again, and only the rest is split anew. Not
            used by a fold export.
        balance: What the validation side is given a share of, recorded in
            the manifest; see :func:`~.training_split.split_balanced`. Not used
            by a fold export.
        scope: What the records were narrowed to, recorded in the manifest.
        user: The name to record as writing it; see :func:`write_training_manifest`.
        describe_images: Describes each image for the images file, which says
            which side every image is on and what it is of, so a model can be
            scored on a part of a set long after the collection has changed.
            Called before anything is written; no file where omitted.
    """

    start_path: Path
    prefix: str = "dataset"
    conditions: str = "primary"
    paired: bool = False
    include_empty_labels: bool = False
    is_empty: Callable[[Path], bool | None] = is_label_file_empty
    grouping: Grouping = Grouping()
    train_ratio: float = DEFAULT_TRAIN_RATIO
    seed: int | None = 42
    validate_on: str | None = None
    validation_from: Path | None = None
    balance: Balance | None = None
    scope: str = ""
    user: str | None = None
    describe_images: DescribeImages | None = None


def divide_for_export(
    records: Iterable[ImageRecord], request: TrainingExportRequest
) -> tuple[list[TrainingRow], dict[str, int], list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    """Return what :func:`export_training_csv` would write, writing nothing.

    The export's own first half, so a preview of a split is the split.

    Returns:
        ``(rows, counts, training, validation, left_out)``, *counts* being
        what the manifest records.

    Raises:
        ValueError: As :func:`export_training_csv` raises before it writes.
    """
    grouping, balance = request.grouping, request.balance
    train_ratio, validate_on = request.train_ratio, request.validate_on
    rows, counts = _rows_to_write(records, request)
    earlier = (
        earlier_sides(request.validation_from) if request.validation_from is not None else None
    )
    train_rows, val_rows, left_out = divide_rows(
        rows, train_ratio, request.seed, validate_on, earlier, balance
    )
    if balance is not None:
        validated = {balance.key(row.records[0]) for row in val_rows}
        without = sorted(
            {str(value) for row in train_rows if (value := balance.key(row.records[0])) not in validated}
        )
        logger.info("Balanced the validation side by %s", ", ".join(balance.names))
        if without and train_ratio < 1.0:
            logger.warning(
                "Nothing validates for %s: each is in one %s that can be validated on, "
                "or in none", "; ".join(without), grouping.level,
            )
    if earlier is not None:
        logger.info(
            "Kept to the split of %s: %d training, %d validation",
            request.validation_from, len(train_rows), len(val_rows),
        )
    if validate_on is not None:
        if rows and not val_rows and train_ratio < 1.0:
            raise ValueError(
                f"No row is {validate_on}, or too few to fill a validation side at "
                f"this share, so there is nothing to validate on. Nothing was written"
            )
        validatable = sum(
            1 for row in train_rows
            if all(current_state(record) == validate_on for record in row.labelled)
        )
        counts = {
            **counts,
            f"train_{validate_on}": validatable,
            f"train_not_{validate_on}": len(train_rows) - validatable,
            "left_out_beside_validation": len(left_out),
        }
        logger.info(
            "Validating on %s rows only: %d of them and %d other row(s) train, %d validate",
            validate_on, validatable, len(train_rows) - validatable, len(val_rows),
        )
        if left_out:
            logger.info(
                "Left out %d row(s) that are not %s and share a %s with the "
                "validation side, which training on would let the model see",
                len(left_out), validate_on, grouping.level,
            )
    return rows, counts, train_rows, val_rows, left_out


def export_training_csv(
    records: Iterable[ImageRecord],
    request: TrainingExportRequest,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Export image and label paths to CSV files for ML model training.

    Writes ``<output_dir>/<prefix>_train.csv``, ``<output_dir>/<prefix>_val.csv``
    unless the training share is 1.0, ``<prefix>_images.csv`` where the request
    describes images, and last ``<prefix>_manifest.yaml`` saying what they were
    made from.

    Two shapes, decided by the request's *paired*::

        image,label
        rel/path/to/image.tif,rel/path/to/label.tif

        sem_image,sem_label,fib_image,fib_label
        rel/sem.tif,rel/sem_label.tif,rel/fib.tif,rel/fib_label.tif

    Args:
        records: ImageRecords to process, already filtered.
        output_dir: Target directory where the CSV files are written.
        overwrite: Replace files an earlier export wrote under the prefix.

    Returns:
        ``(train_csv_path, val_csv_path)``. The second is None when the
        training share is 1.0.

    Raises:
        FileExistsError: If a file this would write exists and *overwrite* is
            False. Nothing has been read or written by then.
        LeftoverFiles: If files of an earlier export this one would not write
            are beside them, whatever *overwrite* says.
        NoRelativePath: If a file has no path relative to the start path,
            before anything is written.
        ValueError: If *conditions* is not a known policy, no row is in the
            state *validate_on* names, or *validation_from* cannot be kept to
            -- before anything is written.
    """
    if request.conditions not in ("primary", "all"):
        raise ValueError(f"Unknown conditions policy: {request.conditions}")
    prefix, grouping, train_ratio = request.prefix, request.grouping, request.train_ratio

    train_path = output_dir / f"{prefix}_train.csv"
    val_path = output_dir / f"{prefix}_val.csv" if train_ratio < 1.0 else None
    manifest = manifest_path(output_dir, prefix)
    images = images_path(output_dir, prefix) if request.describe_images is not None else None
    _refuse_what_is_in_the_way(
        output_dir, prefix, (train_path, val_path, images, manifest), overwrite=overwrite
    )
    rows, counts, train_rows, val_rows, _ = divide_for_export(records, request)
    output_dir.mkdir(parents=True, exist_ok=True)
    if counts["groups"] < len(rows) or grouping.level != ACQUISITION:
        logger.info(
            "Split %d row(s) as %d group(s), keeping each %s on one side: "
            "%d training, %d validation",
            len(rows), counts["groups"], grouping.level, len(train_rows), len(val_rows),
        )
    if counts["groups"] == 1 and 0.0 < train_ratio < 1.0:
        logger.warning(
            "Every row is in one %s, so all of them went to %s; keep a finer "
            "level together to fill both sides", grouping.level,
            "training" if train_rows else "validation",
        )

    described = _described(
        request.describe_images, ((TRAINING, train_rows), (VALIDATION, val_rows))
    )
    columns = PAIRED_COLUMNS if request.paired else SINGLE_COLUMNS
    _write_rows(train_path, columns, train_rows, overwrite)
    if val_path is not None:
        _write_rows(val_path, columns, val_rows, overwrite)
    if images is not None:
        _write_images(images, described, overwrite)
    write_training_manifest(
        manifest,
        source=request.start_path,
        scope=request.scope,
        files={"train": train_path, "validation": val_path, "images": images},
        columns=columns,
        train_ratio=train_ratio,
        seed=request.seed,
        keep_together=grouping.level,
        validate_on=request.validate_on,
        validation_from=request.validation_from,
        balance=request.balance.names if request.balance is not None else (),
        conditions=request.conditions,
        include_empty_labels=request.include_empty_labels,
        counts={
            "images": counts["images"],
            "rows": counts["rows"],
            "groups": counts["groups"],
            "train": len(train_rows),
            "validation": len(val_rows),
            **{key: value for key, value in counts.items()
               if key not in ("images", "rows", "groups")},
        },
        user=request.user,
    )
    return train_path, val_path


def export_kfold_csv(
    records: Iterable[ImageRecord],
    request: TrainingExportRequest,
    output_dir: Path,
    folds: int,
    *,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """Export image and label paths as *folds* train/validation pairs for cross-validation.

    Writes ``<prefix>_fold<k>_train.csv`` and ``<prefix>_fold<k>_val.csv`` for
    each fold, in the columns :func:`export_training_csv` writes, the images
    file where asked for, its ``split`` the fold an image validates in, and last
    the manifest. Every row validates in exactly one fold, and trains in the rest;
    no group is in two folds. Arguments not listed are as for
    :func:`export_training_csv`; the request's training share, *validate_on*,
    *validation_from* and *balance* do not apply.

    Args:
        folds: How many folds, at least 2.

    Returns:
        ``(train, validation)`` per fold, as :func:`fold_paths` names them.

    Raises:
        FileExistsError: As for :func:`export_training_csv`.
        LeftoverFiles: As for :func:`export_training_csv`.
        ValueError: If *folds* is below 2, or above the groups there are to
            divide -- before anything is written.
    """
    if request.conditions not in ("primary", "all"):
        raise ValueError(f"Unknown conditions policy: {request.conditions}")
    if folds < 2:
        raise ValueError(f"Cross-validation needs at least 2 folds, not {folds}")

    prefix, grouping, describe_images = request.prefix, request.grouping, request.describe_images
    paths = fold_paths(output_dir, prefix, folds)
    manifest = manifest_path(output_dir, prefix)
    images = images_path(output_dir, prefix) if describe_images is not None else None
    _refuse_what_is_in_the_way(
        output_dir, prefix, (*(path for pair in paths for path in pair), images, manifest),
        overwrite=overwrite,
    )

    rows, counts = _rows_to_write(records, request)
    if counts["groups"] < folds:
        raise ValueError(
            f"{counts['groups']} {grouping.level} group(s) cannot fill {folds} folds; "
            f"choose fewer folds or keep a finer level together. Nothing was written"
        )
    validation_sets = assign_folds(rows, lambda row: row.group, folds, request.seed)
    logger.info(
        "Divided %d row(s) as %d group(s) into %d folds, keeping each %s in one: "
        "%s validation row(s)",
        len(rows), counts["groups"], folds, grouping.level,
        ", ".join(str(len(validation)) for validation in validation_sets),
    )

    images_described = _described(
        describe_images,
        [(f"fold{index}", validation) for index, validation in enumerate(validation_sets, start=1)],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = PAIRED_COLUMNS if request.paired else SINGLE_COLUMNS
    described = []
    for index, ((train_path, val_path), validation) in enumerate(zip(paths, validation_sets)):
        training = fold_training(validation_sets, index)
        _write_rows(train_path, columns, training, overwrite)
        _write_rows(val_path, columns, validation, overwrite)
        groups = sorted({row.group for row in validation})
        fold: dict[str, object] = {
            "fold": index + 1,
            "train": len(training),
            "validation": len(validation),
            "groups": len(groups),
        }
        # An acquisition's name is a path and a number, which says nothing a
        # methods section would; a lamella's or a project's is what it reports.
        if grouping.level != ACQUISITION:
            fold["validates_on"] = groups
        described.append(fold)

    if images is not None:
        _write_images(images, images_described, overwrite)
    write_training_manifest(
        manifest,
        source=request.start_path,
        scope=request.scope,
        files={
            **{
                f"fold{index}_{role}": path
                for index, pair in enumerate(paths, start=1)
                for role, path in zip(("train", "validation"), pair)
            },
            "images": images,
        },
        columns=columns,
        train_ratio=(folds - 1) / folds,
        seed=request.seed,
        keep_together=grouping.level,
        conditions=request.conditions,
        include_empty_labels=request.include_empty_labels,
        counts=counts,
        folds=described,
        user=request.user,
    )
    return paths
