"""Preparing a validation: a run's checkpoints scored on a list it did not train on.

A validation belongs to its run and lives in it, ``<run>/am_validations/<name>/``: the
list with absolute paths, ``images.csv`` naming each row as the export does with the
metadata the export wrote, a copy of the export's manifest, and ``am_validation.yaml``
saying which epochs are scored. **Everything is checked before anything is written**,
labels looked inside as a training run's are, and a directory is never reused. Nothing
here imports the training stack or scores anything; the scores are
``scores_epochNNN.json`` beside the record, written by the process that is started on it.
"""
from __future__ import annotations

import csv
import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from .. import __version__
from ..history import resolve_author, utc_stamp
from ..io.tables import write_csv
from ..io.yaml import read_yaml, version_problem, write_yaml
from ..training_compare import ScoredValidation
from ..training_runs import Checkpoint, read_training_run, written_in
from ..validation_scores import find_scores, read_scores
from .csv import SINGLE_COLUMNS
from .training_manifest import TrainingManifest, read_training_manifest
from .training_run import (
    NAME_PATTERN,
    TrainingRunRefused,
    check_pairs,
    locate_collection,
    read_pairs,
    resolve_pairs,
)

logger = logging.getLogger(__name__)

#: Where a run's validations are kept, in the run's directory.
VALIDATIONS_DIRECTORY: Final[str] = "am_validations"

#: The record of what a validation scores, at the root of its directory.
VALIDATION_RECORD_FILENAME: Final[str] = "am_validation.yaml"

#: The rows scored, with absolute paths, as AM-model-training reads them.
LIST_FILENAME: Final[str] = "list.csv"

#: The same rows in the same order, each named as the export names it, with the
#: metadata the export wrote for it.
IMAGES_FILENAME: Final[str] = "images.csv"

#: Written into every record, so a reader can tell a layout it does not know.
FILE_VERSION: Final[int] = 1

#: Which rows of an export are scored: its validation list, or every row it names.
Rows = Literal["validation", "all"]
ROWS: Final[tuple[str, ...]] = ("validation", "all")

_HEADER: Final[str] = """\
# What this validation scores, written by am-seg before it started: the epochs of
# the training run it sits in, on the rows of list.csv. images.csv names each row
# as the export it came from does; scores_epochNNN.json holds what an epoch scored
# on each of them.
"""


@dataclass(frozen=True)
class ValidationPlan:
    """A validation that has been checked and can be written.

    Attributes:
        name: What the validation is called.
        directory: Where it will be written; it does not exist yet.
        run_directory: The run whose checkpoints are scored.
        checkpoints: The weights to score, by epoch.
        manifest: The export the rows come from.
        rows: Which of its rows.
        device: The device asked for, resolved when the validation starts.
        pairs: Each row as absolute ``(image, label)`` paths.
        images: Each row's cells for ``images.csv``, in the order of *columns*.
        columns: The header of ``images.csv``, ``image`` first.
    """

    name: str
    directory: Path
    run_directory: Path
    checkpoints: tuple[Checkpoint, ...]
    manifest: TrainingManifest
    rows: Rows
    device: str
    pairs: tuple[tuple[Path, Path], ...]
    images: tuple[tuple[str, ...], ...]
    columns: tuple[str, ...]

    def describe(self) -> list[str]:
        """Return what applying this plan would do, a line at a time."""
        epochs = [checkpoint.epoch for checkpoint in self.checkpoints]
        which = "the validation list" if self.rows == "validation" else "every row"
        described = (
            f"with {len(self.columns) - 1} column(s) of metadata to divide them by"
            if len(self.columns) > 1 else
            "with no metadata to divide them by: the export wrote no images file"
        )
        return [
            f"Validation {self.name} in {self.directory}",
            f"  {len(self.pairs)} image(s), {which} of {self.manifest.path}, {described}",
            f"  {len(epochs)} checkpoint(s) scored: epoch(s) {_ranges(epochs)}",
            f"  device: {self.device}",
        ]


@dataclass(frozen=True)
class ValidationRecord:
    """What ``am_validation.yaml`` says a validation scores.

    Attributes:
        directory: The validation's directory.
        name: What it is called.
        epochs: The epochs it scores.
        device: The device asked for.
        list_file: The rows, with absolute paths.
        images_file: The same rows as the export names them, with their metadata.
        rows: Which rows of the export these are.
        created_at: When it was prepared, in UTC.
        created_by: The name it was prepared under, or "".
    """

    directory: Path
    name: str
    epochs: tuple[int, ...]
    device: str
    list_file: Path
    images_file: Path
    rows: str = "validation"
    created_at: str = ""
    created_by: str = ""

    @property
    def run_directory(self) -> Path:
        """Return the run this validation belongs to: two directories up."""
        return self.directory.parent.parent


def validations_directory(run_directory: Path) -> Path:
    """Return where the validations of the run in *run_directory* are kept."""
    return run_directory / VALIDATIONS_DIRECTORY


def plan_validation(
    run_directory: Path,
    manifest_file: Path,
    name: str,
    *,
    rows: Rows = "validation",
    epochs: Sequence[int] | None = None,
    device: str = "auto",
    start_path: Path | None = None,
    inspect_label: Callable[[Path, int], str | None] | None = None,
) -> ValidationPlan:
    """Check that a run's checkpoints can be scored on an export's rows, and return it ready to write.

    Args:
        run_directory: A run am-seg prepared, or a directory AM-model-training wrote.
        manifest_file: The ``<prefix>_manifest.yaml`` of a training export.
        rows: Its validation list, or every row it names.
        epochs: The epochs to score; every one with whole weights on disk where omitted.
        start_path: The collection, as :func:`~.training_run.plan_training_run` takes it.
        inspect_label: Answers why a label cannot be scored against;
            :func:`~.training_run.label_problem` where omitted.

    Raises:
        TrainingRunRefused: Naming every reason, nothing having been written.
    """
    reasons: list[str] = []
    if not NAME_PATTERN.match(name):
        reasons.append(
            f"{name!r} cannot name a validation: use letters, digits, '.', '_' and '-', "
            f"starting with a letter or digit"
        )
    if rows not in ROWS:
        reasons.append(f"Unknown rows {rows!r}; choose from {', '.join(ROWS)}")
    directory = validations_directory(run_directory) / name
    if directory.exists():
        reasons.append(f"{directory} already exists; choose another name")

    written = written_in(run_directory) if run_directory.is_dir() else None
    kept = read_training_run(written).checkpoints if written is not None else ()
    whole = {checkpoint.epoch: checkpoint for checkpoint in kept if checkpoint.whole()}
    if not kept:
        reasons.append(f"{run_directory} holds no checkpoint to score")
    wanted = sorted(whole) if epochs is None else sorted(set(epochs))
    missing = [epoch for epoch in wanted if epoch not in whole]
    if missing:
        reasons.append(
            f"No whole checkpoint for epoch(s) {_ranges(missing)}: deleted, never saved, "
            f"or cut short by a stop"
        )
    elif kept and not wanted:
        reasons.append(f"Every checkpoint in {run_directory} was cut short by a stop")

    try:
        manifest = read_training_manifest(manifest_file)
    except ValueError as error:
        raise TrainingRunRefused([*reasons, str(error)]) from error
    lists: list[Path] = []
    if manifest.folds:
        reasons.append(
            f"{manifest.path.name} is a cross-validation export of {manifest.folds} folds; "
            f"a validation scores one list"
        )
    elif manifest.columns != SINGLE_COLUMNS:
        reasons.append(
            f"{manifest.path.name} lists {', '.join(manifest.columns)}; a validation scores "
            f"one beam's image,label pairs"
        )
    elif rows == "validation" and manifest.validation is None:
        reasons.append(
            f"{manifest.path.name} has no validation list; score every row of it instead"
        )
    else:
        lists = [
            path for path in (
                manifest.train if rows == "all" else None, manifest.validation,
            ) if path is not None
        ]

    source, misplaced = locate_collection(manifest, start_path, run_directory)
    reasons.extend(misplaced)
    if reasons or source is None:
        raise TrainingRunRefused(reasons)

    cells: list[tuple[str, str]] = []
    for path in lists:
        try:
            cells.extend(read_pairs(path))
        except (OSError, ValueError) as error:
            reasons.append(f"Could not read {path.name}: {error}")
    if not cells and not reasons:
        reasons.append(f"{manifest.path.name} names no rows to score")
    try:
        described = _described(manifest.images)
    except (OSError, ValueError) as error:
        reasons.append(f"Could not read {manifest.images}: {error}")
        described = ((), {})
    if reasons:
        raise TrainingRunRefused(reasons)

    pairs = resolve_pairs(cells, source)
    check_pairs(pairs, source, manifest, purpose="scored against", inspect_label=inspect_label)

    extra, by_image = described
    return ValidationPlan(
        name=name,
        directory=directory,
        run_directory=run_directory,
        checkpoints=tuple(whole[epoch] for epoch in wanted),
        manifest=manifest,
        rows=rows,
        device=device,
        pairs=pairs,
        images=tuple(
            (image, *(by_image.get(image, {}).get(column, "") for column in extra))
            for image, _ in cells
        ),
        columns=("image", *extra),
    )


def apply_validation(plan: ValidationPlan, user: str | None = None) -> Path:
    """Write the validation *plan* describes, and return its directory.

    The record is written last, so a directory holding one was written whole.

    Raises:
        FileExistsError: If the directory has appeared since the plan was made.
    """
    plan.directory.parent.mkdir(parents=True, exist_ok=True)
    plan.directory.mkdir(exist_ok=False)
    write_csv(
        plan.directory / LIST_FILENAME, SINGLE_COLUMNS,
        ((str(image), str(label)) for image, label in plan.pairs),
    )
    write_csv(plan.directory / IMAGES_FILENAME, plan.columns, plan.images)
    manifest_copy = plan.directory / plan.manifest.path.name
    shutil.copyfile(plan.manifest.path, manifest_copy)

    by, _ = resolve_author(user)
    body = {
        "version": FILE_VERSION,
        "am_seg_version": __version__,
        "created_at": utc_stamp(),
        "created_by": by or "",
        "name": plan.name,
        "device": plan.device,
        "epochs": [checkpoint.epoch for checkpoint in plan.checkpoints],
        "training_set": {"manifest": manifest_copy.name, "rows": plan.rows, "images": len(plan.pairs)},
        "files": {"list": LIST_FILENAME, "images": IMAGES_FILENAME},
    }
    write_yaml(plan.directory / VALIDATION_RECORD_FILENAME, body, header=_HEADER)
    logger.info("Prepared the validation %s in %s", plan.name, plan.directory)
    return plan.directory


def read_validation_record(directory: Path) -> ValidationRecord | None:
    """Return what the validation in *directory* scores, or None if it has no record.

    Raises:
        ValueError: If the record cannot be understood, or was written by a later am-seg.
    """
    path = directory / VALIDATION_RECORD_FILENAME
    if not path.is_file():
        return None
    try:
        document = read_yaml(path)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    problem = version_problem(document.get("version"), path, FILE_VERSION, required=True)
    if problem is not None:
        raise ValueError(problem)
    try:
        files = document["files"]
        return ValidationRecord(
            directory=directory,
            name=str(document["name"]),
            epochs=tuple(int(epoch) for epoch in document["epochs"]),
            device=str(document.get("device") or "auto"),
            list_file=directory / files["list"],
            images_file=directory / files["images"],
            rows=str((document.get("training_set") or {}).get("rows") or "validation"),
            created_at=str(document.get("created_at") or ""),
            created_by=str(document.get("created_by") or ""),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a validation's record: {error!r}") from error


def find_validations(run_directory: Path) -> list[Path]:
    """Return the validations prepared for the run in *run_directory*, by name."""
    kept = validations_directory(run_directory)
    try:
        children = sorted(child for child in kept.iterdir() if child.is_dir())
    except OSError:
        return []
    return [child for child in children if (child / VALIDATION_RECORD_FILENAME).is_file()]


def read_images(path: Path) -> list[dict[str, str]]:
    """Return each row of a validation's ``images.csv``, in the order scored."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_scored_validation(directory: Path) -> ScoredValidation:
    """Return the validation in *directory* as it has been scored so far, with its images' metadata.

    A scores file that cannot be read is logged and left out.

    Raises:
        ValueError: If the directory holds no record, or one that cannot be read.
    """
    record = read_validation_record(directory)
    if record is None:
        raise ValueError(f"{directory} holds no {VALIDATION_RECORD_FILENAME}: not a validation")
    scores = []
    for path in find_scores(directory):
        try:
            scores.append(read_scores(path))
        except ValueError as error:
            logger.warning("%s", error)
    try:
        images = read_images(record.images_file)
    except OSError as error:
        logger.warning("Could not read %s: %s", record.images_file, error)
        images = []
    return ScoredValidation(record.name, tuple(scores), images)


def _described(images: Path | None) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    """Return an export's metadata columns and each image's row, or neither where it wrote none."""
    if images is None:
        return (), {}
    with images.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(column for column in reader.fieldnames or () if column != "image")
        if "image" not in (reader.fieldnames or ()):
            raise ValueError("it has no image column")
        return columns, {row["image"]: dict(row) for row in reader}


def _ranges(epochs: Sequence[int]) -> str:
    """Return *epochs* in words, a run of them as its first and last."""
    runs: list[list[int]] = []
    for epoch in sorted(epochs):
        if runs and epoch == runs[-1][1] + 1:
            runs[-1][1] = epoch
        else:
            runs.append([epoch, epoch])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs) or "none"
