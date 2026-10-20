"""Preparing a validation: a run's checkpoints scored on a list it did not train on.

A validation belongs to its run and lives in it, ``<run>/am_validations/<name>/``: the
list with absolute paths, ``images.csv`` naming each row as the export does with the
metadata the export wrote, a copy of the export's manifest, and ``am_validation.yaml``
saying which epochs are scored. **Everything is checked before anything is written**,
labels looked inside as a training run's are, and a directory is never reused. Nothing
here imports the training stack or scores anything; the scores are
``scores_epochNNN.json`` beside the record, written by the process that is started on it.
A run's own list is read, and scored after the run, in ``<run>/am_validation_scores/``.
"""
from __future__ import annotations

import csv
import logging
import shutil
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from .. import __version__
from ..history import resolve_author, utc_stamp
from ..io.directories import directory_cache
from ..io.tables import write_csv
from ..io.yaml import read_yaml, version_problem, write_yaml
from ..operations.training_export import SINGLE_COLUMNS
from ..operations.training_manifest import TrainingManifest, read_training_manifest
from ..progress import Cancellation
from .compare import OWN_VALIDATION, ScoredValidation
from .preparation import (
    NAME_PATTERN,
    OWN_SCORES_DIRECTORY,
    RunRecord,
    TrainingRunRefused,
    check_pairs,
    locate_collection,
    own_scores_directory,
    read_described,
    read_pairs,
    read_run_record,
    resolve_collection,
    resolve_pairs,
)
from .runs import Checkpoint, read_training_run, written_in
from .scores import find_scores, read_scores

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
# What this validation scores, written by AM-seg before it started: the epochs of
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
        own: Whether this scores the run's own validation list, into the
            directory the run keeps those scores in, which may exist.
        trained_on: How many of the images the run trained on, whose scores
            say little of what it would do on an image new to it.
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
    own: bool = False
    trained_on: int = 0

    def describe(self) -> list[str]:
        """Return what applying this plan would do, a line at a time."""
        epochs = [checkpoint.epoch for checkpoint in self.checkpoints]
        which = "the validation list" if self.rows == "validation" else "every row"
        described = (
            f"with {len(self.columns) - 1} column(s) of metadata to divide them by"
            if len(self.columns) > 1 else
            "with no metadata to divide them by: the run kept none, and no export of its "
            "validation list was found to take it from" if self.own else
            "with no metadata to divide them by: the export wrote no images file"
        )
        if self.own:
            return [
                f"The run's own validation list, scored an image at a time into {self.directory}",
                f"  {len(self.pairs)} image(s), {described}",
                f"  {len(epochs)} checkpoint(s) scored: epoch(s) {_ranges(epochs)}",
                f"  device: {self.device}",
            ]
        return [
            f"Validation {self.name} in {self.directory}",
            f"  {len(self.pairs)} image(s), {which} of {self.manifest.path}, {described}",
            *([f"  {self.trained_on} of them the run trained on"] if self.trained_on else []),
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
        """Return the run this validation belongs to: two directories up, or one for the run's own."""
        parent = self.directory.parent
        return parent if self.directory.name == OWN_SCORES_DIRECTORY else parent.parent


def validation_name(manifest_file: Path, taken: Collection[str] = ()) -> str:
    """Return what a validation on the export of *manifest_file* is called: the export's own name.

    So that runs validated on one export hold a validation of one name. A
    number is added only where *taken*, the names a run already holds, has it.
    """
    name = manifest_file.name.removesuffix("_manifest.yaml").removesuffix(".yaml")
    if name not in taken and name != OWN_VALIDATION:
        return name
    return next(f"{name}_{count}" for count in range(2, len(taken) + 3) if f"{name}_{count}" not in taken)


def taken_names(run_directory: Path) -> set[str]:
    """Return what the validations of the run in *run_directory* are called."""
    return {directory.name for directory in find_validations(run_directory)}


def suggested_rows(run_directory: Path, manifest_file: Path, start_path: Path | None = None) -> Rows:
    """Return which rows of an export a run is best scored on: every one where it trained on none.

    Its validation list wherever that cannot be told.
    """
    try:
        manifest = read_training_manifest(manifest_file)
        source = resolve_collection(manifest, start_path)
        if manifest.train is None or source is None:
            return "validation"
        trained = _trained_images(run_directory)
        listed = resolve_pairs(read_pairs(manifest.train), source)
    except (OSError, ValueError):
        return "validation"
    if not trained or any(str(image) in trained for image, _ in listed):
        return "validation"
    return "all"


def _trained_images(run_directory: Path) -> set[str]:
    """Return the images the run in, or holding, *run_directory* trained on, as its list writes them.

    Empty where no record says: a run AM-seg did not prepare.
    """
    for directory in (run_directory, run_directory.parent):
        try:
            record = read_run_record(directory)
            if record is not None:
                return {image for image, _ in read_pairs(record.train_list)}
        except (OSError, ValueError):
            continue
    return set()


def validations_directory(run_directory: Path) -> Path:
    """Return where the validations of the run in *run_directory* are kept."""
    return run_directory / VALIDATIONS_DIRECTORY


@directory_cache()
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
    on_progress: Callable[[int, int], None] | None = None,
    cancellation: Cancellation | None = None,
) -> ValidationPlan:
    """Check that a run's checkpoints can be scored on an export's rows, and return it ready to write.

    Args:
        run_directory: A run AM-seg prepared, or a directory AM-model-training wrote.
        manifest_file: The ``<prefix>_manifest.yaml`` of a training export.
        rows: Its validation list, or every row it names.
        epochs: The epochs to score; every one with whole weights on disk where omitted.
        start_path: The collection, as :func:`~.preparation.plan_training_run` takes it.
        inspect_label: Answers why a label cannot be scored against;
            :func:`~.preparation.label_report` where omitted.
        on_progress: Called as ``on_progress(done, total)`` after each label is opened.
        cancellation: Checked before each label is opened.

    Raises:
        TrainingRunRefused: Naming every reason, nothing having been written.
        OperationCancelled: If *cancellation* is requested while labels are opened.
    """
    reasons: list[str] = []
    if not NAME_PATTERN.match(name):
        reasons.append(
            f"{name!r} cannot name a validation: use letters, digits, '.', '_' and '-', "
            f"starting with a letter or digit"
        )
    if name == OWN_VALIDATION:
        reasons.append(f"{name!r} is what a run's own validation is called; choose another name")
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
        described = read_described(manifest.images)
    except (OSError, ValueError) as error:
        reasons.append(f"Could not read {manifest.images}: {error}")
        described = ((), {})
    if reasons:
        raise TrainingRunRefused(reasons)

    pairs = resolve_pairs(cells, source)
    check_pairs(
        pairs, source, manifest, purpose="scored against", inspect_label=inspect_label,
        on_progress=on_progress, cancellation=cancellation,
    )
    trained = _trained_images(run_directory)

    extra, by_image = described
    return ValidationPlan(
        trained_on=sum(1 for image, _ in pairs if str(image) in trained),
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
    A run's own scores are joined where it kept some as it trained: its
    directory may exist, and an images file already there is left as it is.

    Raises:
        FileExistsError: If the directory, or for a run's own a record in it,
            has appeared since the plan was made.
    """
    plan.directory.parent.mkdir(parents=True, exist_ok=True)
    plan.directory.mkdir(exist_ok=plan.own)
    if (plan.directory / VALIDATION_RECORD_FILENAME).exists():
        raise FileExistsError(f"{plan.directory} has been scored once already")
    write_csv(
        plan.directory / LIST_FILENAME, SINGLE_COLUMNS,
        ((str(image), str(label)) for image, label in plan.pairs),
    )
    if not (plan.own and (plan.directory / IMAGES_FILENAME).exists()):
        write_csv(plan.directory / IMAGES_FILENAME, plan.columns, plan.images)
    manifest_copy = plan.directory / plan.manifest.path.name
    if not manifest_copy.exists():
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
        ValueError: If the record cannot be understood, or was written by a later AM-seg.
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
            name=OWN_VALIDATION if directory.name == OWN_SCORES_DIRECTORY else str(document["name"]),
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
    """Return the validations prepared for the run in *run_directory*: its own list's first, then by name.

    Its own is among them only where it was scored after the run, which is
    what leaves a record; what a run scored as it trained has no process to follow.
    """
    kept = validations_directory(run_directory)
    try:
        children = sorted(child for child in kept.iterdir() if child.is_dir())
    except OSError:
        children = []
    return [
        child for child in (own_scores_directory(run_directory), *children)
        if (child / VALIDATION_RECORD_FILENAME).is_file()
    ]


def plan_own_validation(
    run_directory: Path,
    *,
    exports: Sequence[Path] = (),
    epochs: Sequence[int] | None = None,
    device: str = "auto",
    inspect_label: Callable[[Path, int], str | None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancellation: Cancellation | None = None,
) -> ValidationPlan:
    """Check that a run's checkpoints can be scored on its own validation list, and return it ready to write.

    For a run that did not keep what each image scored as it trained, or not
    for every epoch. The list is the run's, as it trained on it. What each
    image is of is the run's images file; for a run prepared without one, that
    of the first of *exports* whose validation list is the run's.

    Args:
        run_directory: A run AM-seg prepared.
        exports: Manifests of training exports the run may have been prepared from.
        epochs: The epochs to score; every one with whole weights on disk and
            no scores yet where omitted.
        on_progress: Called as ``on_progress(done, total)`` after each label is opened.
        cancellation: Checked before each label is opened.

    Raises:
        TrainingRunRefused: Naming every reason, nothing having been written.
        OperationCancelled: If *cancellation* is requested while labels are opened.
    """
    try:
        record = read_run_record(run_directory)
    except ValueError as error:
        raise TrainingRunRefused([str(error)]) from error
    if record is None:
        raise TrainingRunRefused([
            f"{run_directory} is not a run AM-seg prepared, so nothing says what it validated "
            f"on; validate it on a training export instead"
        ])
    reasons: list[str] = []
    directory = own_scores_directory(run_directory)
    if (directory / VALIDATION_RECORD_FILENAME).exists():
        reasons.append(f"{record.name}'s own list has been scored once already, in {directory}")
    written = written_in(run_directory)
    kept = read_training_run(written).checkpoints if written is not None else ()
    whole = {checkpoint.epoch: checkpoint for checkpoint in kept if checkpoint.whole()}
    scored = {read.epoch for read in read_own_validation(record).scores}
    wanted = sorted(set(whole) - scored) if epochs is None else sorted(set(epochs))
    missing = [epoch for epoch in wanted if epoch not in whole]
    if not kept:
        reasons.append(f"{run_directory} holds no checkpoint to score")
    elif missing:
        reasons.append(
            f"No whole checkpoint for epoch(s) {_ranges(missing)}: deleted, never saved, "
            f"or cut short by a stop"
        )
    elif not wanted:
        reasons.append(f"Every checkpoint of {record.name} on disk has its own list scored already")
    if record.manifest is None or record.source is None:
        reasons.append(f"{record.name}'s record does not name the export it was prepared from")
    if reasons:
        raise TrainingRunRefused(reasons)
    assert record.manifest is not None and record.source is not None
    try:
        manifest = read_training_manifest(record.manifest)
        cells = read_pairs(record.validation_list)
        names = own_image_names(record)
    except (OSError, ValueError) as error:
        raise TrainingRunRefused([str(error)]) from error

    pairs = tuple((Path(image), Path(label)) for image, label in cells)
    check_pairs(
        pairs, record.source, manifest, purpose="scored against", inspect_label=inspect_label,
        on_progress=on_progress, cancellation=cancellation,
    )

    columns: tuple[str, ...] = ("image",)
    images: tuple[tuple[str, ...], ...] = tuple((name,) for name in names)
    if record.validation_images is not None:
        extra, by_image = read_described(record.validation_images)
        columns = ("image", *extra)
        images = tuple((name, *(by_image.get(name, {}).get(c, "") for c in extra)) for name in names)
    elif not scored:
        described = _described_by(exports, [image for image, _ in cells])
        if described is not None:
            columns, images = described
    return ValidationPlan(
        name=OWN_VALIDATION,
        directory=directory,
        run_directory=run_directory,
        checkpoints=tuple(whole[epoch] for epoch in wanted),
        manifest=manifest,
        rows="validation",
        device=device,
        pairs=pairs,
        images=images,
        columns=columns,
        own=True,
    )


def _described_by(
    exports: Sequence[Path], images: Sequence[str]
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    """Return the header and rows of an images file for *images*, from the first of *exports* that lists them.

    An export lists them where its validation list names the same images in
    the same order, each path of *images* ending in the export's.
    """
    def same(listed: str, written: str) -> bool:
        listed, written = listed.replace("\\", "/"), written.replace("\\", "/")
        return written == listed or written.endswith("/" + listed.lstrip("/"))

    for path in exports:
        try:
            manifest = read_training_manifest(path)
            if manifest.validation is None or manifest.images is None:
                continue
            listed = [image for image, _ in read_pairs(manifest.validation)]
            extra, by_image = read_described(manifest.images)
        except (OSError, ValueError):
            continue
        if len(listed) == len(images) and all(map(same, listed, images)):
            return ("image", *extra), tuple(
                (name, *(by_image.get(name, {}).get(column, "") for column in extra))
                for name in listed
            )
    return None


def read_images(path: Path) -> list[dict[str, str]]:
    """Return each row of a validation's ``images.csv``, in the order scored."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def own_image_names(record: RunRecord) -> tuple[str, ...]:
    """Return each row of a run's own validation list in its order, named as the export names it.

    As the list itself writes it where the run kept no images file.

    Raises:
        OSError, ValueError: If neither file can be read.
    """
    if record.validation_images is not None:
        return tuple(row["image"] for row in read_images(record.validation_images))
    return tuple(image for image, _ in read_pairs(record.validation_list))


def read_own_validation(record: RunRecord) -> ScoredValidation:
    """Return what the run of *record* scored on each image of its own validation list, so far.

    Called :data:`~.compare.OWN_VALIDATION`, and empty for a run that
    kept none. A scores file that cannot be read is logged and left out.
    """
    scores = []
    for path in find_scores(own_scores_directory(record.directory)):
        try:
            scores.append(read_scores(path))
        except ValueError as error:
            logger.warning("%s", error)
    return ScoredValidation(OWN_VALIDATION, tuple(scores), own_images(record) if scores else ())


def own_images(record: RunRecord) -> list[dict[str, str]]:
    """Return what each image of a run's own validation list is of.

    As the run's images file says; for a run prepared without one, as an
    :data:`IMAGES_FILENAME` beside its scores does; else each image's name alone.
    """
    beside = own_scores_directory(record.directory) / IMAGES_FILENAME
    try:
        if record.validation_images is not None:
            return read_images(record.validation_images)
        if beside.is_file():
            return read_images(beside)
        return [{"image": name} for name in own_image_names(record)]
    except (OSError, ValueError) as error:
        logger.warning("Could not read what %s validated on: %s", record.name, error)
        return []


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


def _ranges(epochs: Sequence[int]) -> str:
    """Return *epochs* in words, a run of them as its first and last."""
    runs: list[list[int]] = []
    for epoch in sorted(epochs):
        if runs and epoch == runs[-1][1] + 1:
            runs[-1][1] = epoch
        else:
            runs.append([epoch, epoch])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs) or "none"
