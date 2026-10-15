"""Choosing a few validation images to look at several models' predictions on.

Reads runs' directories and writes nothing. **An image is offered only where no run
looked at trained on it**, a model looking good on what it has seen; what each image
is of comes from the file a run keeps beside its validation list, or else beside its
scores, and an image no file describes can be drawn but not chosen by what it is of. Which images, and which
checkpoint of a run, are decided here on paths and strings, apart from predicting.
A run prepared where the collection was somewhere else is read as under the folder given.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..operations.metadata_csv import COMMON_COLUMNS, column_caption
from .choice import chosen_best_or_none
from .listing import ListedRun
from .preparation import RunRecord, read_pairs, read_run_record, relocate_pairs
from .runs import Checkpoint, TrainingRun, read_training_run
from .validation import own_images

logger = logging.getLogger(__name__)

#: What an image can be chosen by, each ``(caption, column of the images file)``.
PREVIEW_FILTERS: Final[tuple[tuple[str, str], ...]] = COMMON_COLUMNS

#: How many images are looked at where nothing says otherwise, and the most.
DEFAULT_PREVIEW_COUNT: Final[int] = 3
MOST_PREVIEW_IMAGES: Final[int] = 4

#: The columns of the images file said of an image where there is room for more than a line.
PREVIEW_DETAILS: Final[tuple[str, ...]] = (
    "autolamella_project", "lamella", "microscope_model", "project_lamella_type",
    "acceleration_voltage_kv", "beam_current_pa", "pixel_size_nm", "hfw_um",
    "image_width", "image_height",
)


@dataclass(frozen=True)
class PreviewImage:
    """A validation image a model can be looked at on.

    Attributes:
        image: The image.
        label: Its label, as the run that validated on it was given it.
        described: What it is of, by column of the images file; empty where
            no run kept that for it.
    """

    image: Path
    label: Path
    described: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)

    def describe(self) -> str:
        """Return what the image is of in a line, by :data:`PREVIEW_FILTERS`, or "" where nothing says."""
        said = []
        for caption, column in PREVIEW_FILTERS:
            value = self.described.get(column, "")
            if value:
                said.append(f"{value} kV" if column == "acceleration_voltage_kv" else value)
        return " · ".join(said)

    def details(self) -> list[str]:
        """Return what the image is of at more length, a line for each of :data:`PREVIEW_DETAILS` it has."""
        return [
            f"{column_caption(column)}: {self.described[column]}"
            for column in PREVIEW_DETAILS if self.described.get(column)
        ]


@dataclass(frozen=True)
class PreviewRun:
    """What choosing images needs of one run.

    Attributes:
        name: What the run is called.
        validation: The images it validated on.
        training: The images it trained on, or None where nothing says: a run
            am-seg did not start.
        looked_in: The collection's folder its images are read as under, or
            None where nothing says.
    """

    name: str
    validation: tuple[PreviewImage, ...] = ()
    training: frozenset[Path] | None = None
    looked_in: Path | None = field(default=None, compare=False)

    def trained_on(self, image: PreviewImage) -> bool | None:
        """Return whether the run trained on *image*, or None where nothing says."""
        return None if self.training is None else image.image in self.training


def read_preview_run(
    name: str, record: RunRecord | None, collection: Path | None = None
) -> PreviewRun:
    """Return what the run *record* describes validated and trained on.

    A run with no record, or whose lists cannot be read, offers no images and
    is not known to have trained on any.

    Args:
        collection: Where the collection is here. Both lists are read as under
            it where the folder the record names is not on this machine, and
            as written where it is.
    """
    if record is None:
        return PreviewRun(name)
    try:
        lists = [read_pairs(record.train_list), read_pairs(record.validation_list)]
    except (OSError, ValueError) as error:
        logger.warning("Could not read the lists of %s: %s", name, error)
        return PreviewRun(name)
    if collection is not None and record.source is not None and not record.source.is_dir():
        logger.info("Reading %s as under %s: %s is not here", name, collection, record.source)
        trained, validation = (relocate_pairs(rows, record.source, collection) for rows in lists)
        looked_in: Path | None = collection
    else:
        looked_in = record.source
        trained, validation = (
            tuple((Path(image), Path(label)) for image, label in rows) for rows in lists
        )
    training = frozenset(image for image, _ in trained)
    described = own_images(record)
    if len(described) != len(validation):
        described = [{} for _ in validation]
    return PreviewRun(
        name,
        tuple(
            PreviewImage(image, label, row)
            for (image, label), row in zip(validation, described)
        ),
        training,
        looked_in,
    )


@dataclass(frozen=True)
class PreviewModel:
    """One run as it is looked at: the checkpoint predicting, and what the run saw.

    Attributes:
        name: What the run is called.
        checkpoint: The checkpoint it is looked at with, or None where there is none to use.
        which: That checkpoint in words, or why there is none.
        run: What the run validated and trained on.
    """

    name: str
    checkpoint: Checkpoint | None
    which: str
    run: PreviewRun

    def caption(self) -> str:
        """Return what heads the run's predictions: its name and the checkpoint used."""
        return f"{self.name}\n{self.which}"


def read_preview_model(listed: ListedRun, collection: Path | None = None) -> PreviewModel:
    """Return the run *listed* as it is looked at, read from its directory now.

    Args:
        collection: Where the collection is here, as :func:`read_preview_run` takes it.
    """
    try:
        record = read_run_record(listed.home) if listed.home != listed.written else None
    except ValueError as error:
        logger.warning("%s", error)
        record = None
    written = read_training_run(listed.written) if listed.written is not None else None
    chosen = chosen_best_or_none(listed.home)
    checkpoint, which = preview_checkpoint(written, chosen.epoch if chosen else None)
    return PreviewModel(
        listed.name, checkpoint, which, read_preview_run(listed.name, record, collection)
    )


def preview_images(runs: Sequence[PreviewRun], unseen_only: bool = True) -> list[PreviewImage]:
    """Return the images *runs* can be looked at on, each once, in path order.

    Args:
        unseen_only: Leave out an image any of *runs* trained on.
    """
    found: dict[Path, PreviewImage] = {}
    for run in runs:
        for image in run.validation:
            # The one that says what it is of, where two runs validated on it.
            if image.image not in found or (image.described and not found[image.image].described):
                found[image.image] = image
    if unseen_only:
        trained = set().union(*(run.training or frozenset() for run in runs))
        found = {path: image for path, image in found.items() if path not in trained}
    return [found[path] for path in sorted(found)]


def offered_values(images: Sequence[PreviewImage], column: str) -> tuple[str, ...]:
    """Return the values *images* hold in *column*, each once, in order."""
    return tuple(sorted({image.described.get(column, "") for image in images} - {""}))


def pick_images(
    images: Sequence[PreviewImage],
    chosen: Mapping[str, str] | None = None,
    count: int = DEFAULT_PREVIEW_COUNT,
    seed: int = 0,
) -> list[PreviewImage]:
    """Return up to *count* of *images* at random, among those that are what *chosen* asks.

    Args:
        chosen: A value by column of the images file; a column left out, or
            given "", is any.
        seed: Fixes the draw, so the same one can be made again.
    """
    wanted = {column: value for column, value in (chosen or {}).items() if value}
    among = sorted(
        (
            image for image in images
            if all(image.described.get(column, "") == value for column, value in wanted.items())
        ),
        key=lambda image: image.image,
    )
    return random.Random(seed).sample(among, min(max(count, 0), len(among)))


def split_found(
    images: Sequence[PreviewImage], exists: Callable[[Path], bool] = Path.is_file
) -> tuple[list[PreviewImage], list[PreviewImage]]:
    """Return *images* as those whose file is there and those whose is not, each in order.

    Args:
        exists: Says whether an image's file is there; the filesystem unless a test says otherwise.
    """
    found = [image for image in images if exists(image.image)]
    return found, [image for image in images if image not in found]


def describe_not_found(
    not_found: Sequence[PreviewImage], drawn: int, runs: Sequence[PreviewRun], advice: str
) -> str:
    """Return what to say of the *not_found* among *drawn* images, or "" where every one was found.

    Names the folders *runs* were looked for in once, rather than a path an image.

    Args:
        advice: How the collection is named where none of the images was found.
    """
    if not not_found:
        return ""
    folders = sorted({str(run.looked_in) for run in runs if run.validation and run.looked_in})
    where = f" They were looked for under {', '.join(folders)}." if folders else ""
    if len(not_found) == drawn:
        return f"None of the {drawn} image(s) drawn is on this machine.{where} {advice}"
    return (
        f"{len(not_found)} of the {drawn} image(s) drawn are not on this machine and "
        f"are left out.{where}"
    )


def preview_checkpoint(
    written: TrainingRun | None, marked: int | None
) -> tuple[Checkpoint | None, str]:
    """Return the checkpoint a run is looked at with, and what it is in words.

    The epoch marked as its best, else the one the run counts as best; None,
    with why, where that epoch's weights are not whole on disk.
    """
    if written is None:
        return None, "no epoch trained"
    best = written.best()
    epoch = marked if marked is not None else best.epoch if best is not None else None
    if epoch is None:
        return None, "no epoch validated"
    which = f"epoch {epoch}, " + ("marked" if marked is not None else "run's best")
    checkpoint = written.checkpoint_for(epoch)
    if checkpoint is None:
        return None, f"{which}: its weights are not on disk"
    if not checkpoint.whole():
        return None, f"{which}: its weights are incomplete"
    return checkpoint, which
