"""Preparing the directory a training run is started in, from a training export.

A run directory holds the two lists with absolute paths, since AM-model-training
resolves a relative one against the list's folder and an export writes it against
the collection's; a copy of the export's manifest; and ``am_training_run.yaml``,
saying what the run was asked to do. **Everything is checked before anything is
written**, labels looked inside again, and an existing directory is never reused.
Nothing here imports the training stack or starts a run.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from .. import __version__
from ..display import is_empty_label
from ..history import resolve_author
from ..io.image import to_2d
from ..io.tables import write_csv
from ..io.yaml import read_yaml, version_problem, write_yaml
from .csv import SINGLE_COLUMNS
from .label_io import read_label
from .training_manifest import TrainingManifest, read_training_manifest

logger = logging.getLogger(__name__)

#: The record of what a run was asked to do, at the root of its directory.
RUN_RECORD_FILENAME: Final[str] = "am_training_run.yaml"

#: The validation list's name in every run. With the training list named for the
#: run, AM-model-training calls its checkpoints ``<name>_val_epochNNN.pth``.
VALIDATION_LIST_FILENAME: Final[str] = "val.csv"

#: Written into every record, so a reader can tell a layout it does not know.
FILE_VERSION: Final[int] = 1

#: What a run's name may be: it becomes a directory and the stem of every file
#: the run writes.
NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: How many files a refusal names before saying how many more there are.
EXAMPLES_SHOWN: Final[int] = 5

#: What :func:`label_problem` says of a label nothing is annotated in, which an
#: export told to include such labels is allowed.
ONLY_BACKGROUND: Final[str] = "holds only background"

Mode = Literal["new", "refine"]

_HEADER: Final[str] = """\
# What this training run was asked to do, written by am-seg before it started.
#
# `training_set` is the export the lists were copied from; its manifest is beside
# this file. `base` is the checkpoint a refine started from, null for a new model.
# What the run did is in the training_<date>_<time> directory AM-model-training
# writes here.
"""


class TrainingRunRefused(ValueError):
    """A run cannot be prepared as asked; ``reasons`` holds every reason at once."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons: tuple[str, ...] = tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class TrainingSettings:
    """How a model is trained: what AM-model-training is told.

    Attributes:
        mode: ``"new"`` starts from ImageNet weights, ``"refine"`` from a checkpoint.
        max_epochs: The most epochs to train; a run stops early once it stops improving.
        frozen_epochs: Epochs the encoder is held frozen for before it is trained.
        initial_learning_rate: Where the one-cycle schedule starts.
        max_learning_rate: The schedule's peak.
        training_batch_size: Images per training step.
        validation_batch_size: Images per validation step.
        save_all: Keep a checkpoint after every epoch, not only the improving ones.
        seed: Fixes augmentation and shuffling.
    """

    mode: Mode
    max_epochs: int
    frozen_epochs: int
    initial_learning_rate: float
    max_learning_rate: float
    training_batch_size: int = 6
    validation_batch_size: int = 1
    save_all: bool = False
    seed: int = 42

    def problems(self) -> list[str]:
        """Return why these settings cannot be trained with, empty if they can."""
        found = []
        if self.mode not in ("new", "refine"):
            found.append(f"Unknown mode {self.mode!r}; choose new or refine")
        if self.max_epochs < 1:
            found.append("The maximum number of epochs must be at least 1")
        if not 0 <= self.frozen_epochs < max(self.max_epochs, 1):
            found.append(
                "The encoder must be frozen for fewer epochs than are trained, and not fewer than 0"
            )
        if not 0 < self.initial_learning_rate <= self.max_learning_rate:
            found.append(
                "The initial learning rate must be above 0 and no higher than the maximum"
            )
        if self.training_batch_size < 1 or self.validation_batch_size < 1:
            found.append("A batch must hold at least 1 image")
        return found


#: The published settings a new model is trained with: the encoder frozen for 25
#: epochs, a learning rate from 1e-5 to 1e-3.
NEW_MODEL: Final = TrainingSettings(
    mode="new", max_epochs=100, frozen_epochs=25,
    initial_learning_rate=1e-5, max_learning_rate=1e-3,
)

#: The published settings an existing model is refined with: nothing frozen, 50
#: epochs at most, a learning rate from 2e-6.
REFINE: Final = TrainingSettings(
    mode="refine", max_epochs=50, frozen_epochs=0,
    initial_learning_rate=2e-6, max_learning_rate=3e-4,
)


def default_settings(mode: Mode) -> TrainingSettings:
    """Return the default settings for *mode*, those the models were published with."""
    return REFINE if mode == "refine" else NEW_MODEL


@dataclass(frozen=True)
class TrainingPlan:
    """A run that has been checked and can be written.

    Attributes:
        name: What the run is called.
        run_directory: Where it will be written; it does not exist yet.
        settings: How the model is trained.
        base_weights: The checkpoint a refine starts from, else None.
        device: The device asked for, resolved when the run starts.
        manifest: The export the lists come from.
        source: The folder the export's paths were resolved against.
        train: Each training pair as absolute ``(image, label)`` paths.
        validation: Each validation pair, likewise.
    """

    name: str
    run_directory: Path
    settings: TrainingSettings
    base_weights: Path | None
    device: str
    manifest: TrainingManifest
    source: Path
    train: tuple[tuple[Path, Path], ...]
    validation: tuple[tuple[Path, Path], ...]

    @property
    def train_list(self) -> Path:
        """Return where the training list will be written."""
        return self.run_directory / f"{self.name}.csv"

    @property
    def validation_list(self) -> Path:
        """Return where the validation list will be written."""
        return self.run_directory / VALIDATION_LIST_FILENAME

    def describe(self) -> list[str]:
        """Return what applying this plan would do, a line at a time."""
        settings = self.settings
        start = (
            f"refine {self.base_weights}" if settings.mode == "refine"
            else "a new model, from ImageNet weights, which are downloaded if not cached"
        )
        frozen = (
            f", encoder frozen for the first {settings.frozen_epochs}"
            if settings.frozen_epochs else ""
        )
        return [
            f"Run {self.name} in {self.run_directory}",
            f"  {start}",
            f"  {len(self.train)} training and {len(self.validation)} validation pair(s) "
            f"from {self.manifest.path}",
            f"  at most {settings.max_epochs} epoch(s){frozen}; learning rate "
            f"{settings.initial_learning_rate:g} to {settings.max_learning_rate:g}; "
            f"batches of {settings.training_batch_size} and {settings.validation_batch_size}"
            f"{'; every epoch saved' if settings.save_all else ''}",
            f"  device: {self.device}",
        ]


@dataclass(frozen=True)
class RunRecord:
    """What ``am_training_run.yaml`` says a run was asked to do.

    Attributes:
        directory: The run directory the record was read from.
        name: What the run is called.
        settings: How the model is trained.
        base: The checkpoint a refine started from -- its ``file``, ``run``,
            ``size`` and ``sha256`` -- or None for a new model.
        device: The device asked for.
        train_list: The training list, in the run directory.
        validation_list: The validation list, likewise.
        created_at: When the run was prepared, in UTC.
        created_by: The name it was prepared under, or "".
        am_seg_version: The am-seg that prepared it.
    """

    directory: Path
    name: str
    settings: TrainingSettings
    base: Mapping[str, Any] | None
    device: str
    train_list: Path
    validation_list: Path
    created_at: str = ""
    created_by: str = ""
    am_seg_version: str = ""


def label_problem(path: Path, classes: int) -> str | None:
    """Return why the label at *path* cannot be trained on, or None if it can.

    Args:
        classes: How many classes the model predicts; a label may hold values below it.
    """
    array = read_label(path)
    if array is None or array.size == 0:
        return "could not be read"
    label = to_2d(array)
    if is_empty_label(label):
        return ONLY_BACKGROUND
    highest = int(label.max())
    if highest >= classes:
        return f"holds class {highest}, and the model predicts classes 0 to {classes - 1}"
    return None


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """Return the ``(image, label)`` cells of each row of a training list, as written.

    Raises:
        ValueError: If the list is not a single-beam ``image,label`` one.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SINGLE_COLUMNS:
            raise ValueError(
                f"{path} is not an image,label list: {', '.join(reader.fieldnames or ['(no header)'])}"
            )
        return [(row["image"] or "", row["label"] or "") for row in reader]


def plan_training_run(
    manifest_file: Path,
    training_directory: Path,
    name: str,
    settings: TrainingSettings,
    *,
    base_weights: Path | None = None,
    device: str = "auto",
    start_path: Path | None = None,
    inspect_label: Callable[[Path, int], str | None] | None = None,
) -> TrainingPlan:
    """Check that a run can be prepared from an export, and return it ready to write.

    Args:
        manifest_file: The ``<prefix>_manifest.yaml`` of a training export.
        training_directory: Where runs are kept; the run gets a directory of
            its own in it, named *name*.
        base_weights: The checkpoint to refine; required by, and only by, a refine.
        start_path: The folder the export's paths are written from. Needed where
            the manifest recorded a relative one.
        inspect_label: Answers why a label cannot be trained on; :func:`label_problem`
            where omitted.

    Raises:
        TrainingRunRefused: Naming every reason, nothing having been written.
    """
    # The class count is generation 1's, which is all AM-model-training trains.
    from ..ml.model import GENERATION_1

    inspect = inspect_label or label_problem
    reasons: list[str] = list(settings.problems())

    if not NAME_PATTERN.match(name):
        reasons.append(
            f"{name!r} cannot name a run: use letters, digits, '.', '_' and '-', "
            f"starting with a letter or digit"
        )
    run_directory = training_directory / name
    if run_directory.exists():
        reasons.append(f"{run_directory} already exists; choose another name")

    if settings.mode == "refine":
        if base_weights is None:
            reasons.append("Refining needs the model weights to start from")
        elif not base_weights.is_file():
            reasons.append(f"The model weights do not exist: {base_weights}")
    elif base_weights is not None:
        reasons.append("A new model starts from no weights; choose refine to start from these")

    try:
        manifest = read_training_manifest(manifest_file)
    except ValueError as error:
        raise TrainingRunRefused([*reasons, str(error)]) from error

    if manifest.folds:
        reasons.append(
            f"{manifest.path.name} is a cross-validation export of {manifest.folds} folds; "
            f"a run trains on one split"
        )
    elif manifest.columns != SINGLE_COLUMNS:
        reasons.append(
            f"{manifest.path.name} lists {', '.join(manifest.columns)}; a run trains on "
            f"one beam's image,label pairs"
        )
    elif manifest.train is None or manifest.validation is None:
        reasons.append(
            f"{manifest.path.name} has no validation list; export with a training share "
            f"below 100 %, so the split is am-seg's and keeps what it was told together"
        )

    source = _source(manifest, start_path)
    if source is None:
        reasons.append(
            f"{manifest.path.name} records its paths from {manifest.source!r}, which does "
            f"not say where the collection is; name the folder the export was made from"
        )
    elif _inside(run_directory, source):
        reasons.append(
            f"{run_directory} is inside the collection at {source}; keep training runs "
            f"outside it"
        )
    if reasons or source is None or manifest.train is None or manifest.validation is None:
        raise TrainingRunRefused(reasons)

    lists: dict[str, tuple[tuple[Path, Path], ...]] = {}
    for side, path in (("training", manifest.train), ("validation", manifest.validation)):
        try:
            cells = read_pairs(path)
        except (OSError, ValueError) as error:
            reasons.append(f"Could not read the {side} list: {error}")
            continue
        if not cells:
            reasons.append(f"The {side} list {path.name} holds no rows")
        lists[side] = tuple(
            ((source / image).resolve(), (source / label).resolve()) for image, label in cells
        )
    if reasons:
        raise TrainingRunRefused(reasons)

    pairs = [pair for side in lists.values() for pair in side]
    missing = [path for pair in pairs for path in pair if not path.is_file()]
    if missing:
        raise TrainingRunRefused([_listed(
            f"{len(missing)} file(s) the lists name are not under {source}", missing,
        )])

    unusable = [
        f"{label}: {problem}"
        for _, label in pairs
        if (problem := inspect(label, GENERATION_1.classes)) is not None
        and not (manifest.include_empty_labels and problem == ONLY_BACKGROUND)
    ]
    if unusable:
        raise TrainingRunRefused([_listed(
            f"{len(unusable)} label(s) cannot be trained on", unusable,
        )])

    return TrainingPlan(
        name=name,
        run_directory=run_directory,
        settings=settings,
        base_weights=base_weights,
        device=device,
        manifest=manifest,
        source=source,
        train=lists["training"],
        validation=lists["validation"],
    )


def apply_training_run(plan: TrainingPlan, user: str | None = None) -> Path:
    """Write the run directory *plan* describes, and return it.

    The record is written last, so a directory holding one was written whole.

    Args:
        user: The name to record, resolved as a history entry's author is.

    Raises:
        FileExistsError: If the directory has appeared since the plan was made.
    """
    plan.run_directory.mkdir(parents=True, exist_ok=False)
    write_csv(plan.train_list, SINGLE_COLUMNS, (_cells(pair) for pair in plan.train))
    write_csv(plan.validation_list, SINGLE_COLUMNS, (_cells(pair) for pair in plan.validation))
    manifest_copy = plan.run_directory / plan.manifest.path.name
    shutil.copyfile(plan.manifest.path, manifest_copy)

    by, _ = resolve_author(user)
    body = {
        "version": FILE_VERSION,
        "am_seg_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "created_by": by or "",
        "name": plan.name,
        "mode": plan.settings.mode,
        "base": describe_weights(plan.base_weights) if plan.base_weights else None,
        "settings": {key: value for key, value in asdict(plan.settings).items() if key != "mode"},
        "device": plan.device,
        "training_set": {
            "manifest": manifest_copy.name,
            "source": str(plan.source),
            "train": len(plan.train),
            "validation": len(plan.validation),
        },
        "files": {"train": plan.train_list.name, "validation": plan.validation_list.name},
    }
    write_yaml(plan.run_directory / RUN_RECORD_FILENAME, body, header=_HEADER)
    logger.info("Prepared the training run %s in %s", plan.name, plan.run_directory)
    return plan.run_directory


def read_run_record(run_directory: Path) -> RunRecord | None:
    """Return what the run in *run_directory* was asked to do, or None if it has no record.

    Raises:
        ValueError: If the record cannot be understood, or was written by a later am-seg.
    """
    path = run_directory / RUN_RECORD_FILENAME
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
        settings = replace(
            default_settings(document["mode"]), mode=document["mode"], **document["settings"]
        )
        files = document["files"]
        return RunRecord(
            directory=run_directory,
            name=str(document["name"]),
            settings=settings,
            base=document.get("base"),
            device=str(document.get("device") or "auto"),
            train_list=run_directory / files["train"],
            validation_list=run_directory / files["validation"],
            created_at=str(document.get("created_at") or ""),
            created_by=str(document.get("created_by") or ""),
            am_seg_version=str(document.get("am_seg_version") or ""),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} is not a training run's record: {error!r}") from error


def find_run_directories(path: Path) -> list[Path]:
    """Return the prepared runs at *path*: itself if it is one, else those directly inside it."""
    if (path / RUN_RECORD_FILENAME).is_file():
        return [path]
    try:
        children = sorted(child for child in path.iterdir() if child.is_dir())
    except OSError as error:
        logger.warning("Could not look inside %s: %s", path, error)
        return []
    return [child for child in children if (child / RUN_RECORD_FILENAME).is_file()]


def _source(manifest: TrainingManifest, start_path: Path | None) -> Path | None:
    """Return the folder the manifest's paths resolve against, or None if nothing says."""
    if start_path is not None:
        return start_path.resolve()
    recorded = Path(manifest.source) if manifest.source else None
    return recorded if recorded is not None and recorded.is_absolute() else None


def _inside(path: Path, directory: Path) -> bool:
    """Return whether *path* is *directory* or below it."""
    return path.resolve().is_relative_to(directory.resolve())


def _cells(pair: tuple[Path, Path]) -> tuple[str, str]:
    """Return one pair as the two cells of its row."""
    image, label = pair
    return str(image), str(label)


def _listed(summary: str, items: Sequence[object]) -> str:
    """Return *summary* followed by the first few of *items*, and how many more there are."""
    shown = ", ".join(str(item) for item in items[:EXAMPLES_SHOWN])
    more = len(items) - EXAMPLES_SHOWN
    return f"{summary}: {shown}" + (f", and {more} more" if more > 0 else "")


def describe_weights(weights: Path) -> dict[str, Any]:
    """Return what identifies a checkpoint: its name, the run inside it, its size and digest."""
    # Reads the archive's names only, with zipfile, so it needs no training stack.
    from ..ml.model import run_name_of

    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "file": str(weights.resolve()),
        "run": run_name_of(weights),
        "size": weights.stat().st_size,
        "sha256": digest.hexdigest(),
    }
