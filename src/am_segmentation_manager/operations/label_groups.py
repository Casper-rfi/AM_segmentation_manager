"""Images that share a label, and the state they share with it.

Several imaging conditions of one scene share one segmentation, so the label, not
the image, is the unit of proofreading: a :class:`LabelGroup` is one label file and
every image referencing it, a private label forming a group of one. A group is
identified by where its label lives *or would live*, so images are one group
before any label exists. Groups are built from the filesystem rather than an index,
so a command given one path expands correctly; the state is declared, not read here.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..families import FamiliesLookup, families_for
from ..frames import primary_among
from ..layout import (
    NO_CORRECTIONS,
    build_image_sets,
    classify_image_directory,
    classify_loose_image,
    condition_of,
    image_files_by_name,
    resolve_labels_directory,
    resolve_loose_labels_directory,
    resolve_loose_predictions_directory,
    resolve_predictions_directory,
    subdirectories_by_name,
)
from ..models import ImageRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelGroup:
    """A label and every image that shares it.

    Attributes:
        label_path: Where the label is, or None when there is none yet.
        identity: Where the label lives or would live. Unlike *label_path* this
            is set whether or not the file exists, and it is what the group is
            keyed by. None only when the image sits in no recognised image
            directory.
        records: The images referencing that label, in a stable order. Always
            at least one.
    """

    label_path: Path | None
    records: tuple[ImageRecord, ...]
    identity: Path | None = None

    @property
    def is_shared(self) -> bool:
        """Return whether more than one image shares this label."""
        return len(self.records) > 1

def labels_directory_for(
    image_path: Path, cache: dict[Path, Path | None] | None = None
) -> Path | None:
    """Return where *image_path*'s label lives or would live.

    Answers for a label directory that does not exist yet, which is what lets a
    group be identified before anything has been annotated. An image filling a
    ``<modality>`` directory takes its answer from that directory; one sitting
    loose takes it from what its filename says the set is. Returns None when
    neither says anything.

    Args:
        cache: Reused across a batch. Resolution lists the stage's
            subdirectories and reads the project's family corrections, which is
            once per image directory rather than per image. Only the
            directory-based answer is cached: a loose set is chosen by filename,
            and one directory holds several of them.

    Raises:
        FamiliesFileError: If the project's families file will not read.
    """
    image_directory = image_path.parent
    if cache is not None and image_directory in cache:
        return cache[image_directory]
    labels_directory = record_from_image_path(image_path, check_label=False).labels_directory
    if cache is not None and classify_image_directory(image_directory.name) is not None:
        cache[image_directory] = labels_directory
    return labels_directory


def record_from_image_path(path: Path, *, check_label: bool = True) -> ImageRecord:
    """Return a record for an image named by its path alone, with its label if there is one.

    **Only the image's own directory is read as an imaging condition**: nothing
    sits below an image directory, so an ancestor named like one -- a collection
    folder called ``sem_runs`` -- says nothing about the image
    (``test_an_ancestor_named_like_a_condition_is_not_one``). An image filling a
    condition's directory resolves under its project's family corrections; one
    sitting loose is read from its filename, as its set is, and carries its
    stage; one whose name says nothing gets a bare record.

    Args:
        check_label: Look for the label file; off where only its directory is
            wanted.

    Raises:
        FamiliesFileError: If the project's families file will not read.
    """
    image_directory = path.parent
    classification = classify_image_directory(image_directory.name)
    if classification is None:
        return _loose_record(path, check_label)

    corrections = families_for(image_directory)
    condition = condition_of(image_directory.name, corrections) or classification
    siblings = subdirectories_by_name(image_directory.parent)
    labels_directory, _, _ = resolve_labels_directory(
        image_directory, condition.modality, siblings, corrections=corrections
    )
    predictions_directory, _, _ = resolve_predictions_directory(
        image_directory, condition.modality, siblings, corrections=corrections
    )
    label_path = labels_directory / path.name
    return ImageRecord(
        image_path=path,
        labels=[label_path] if check_label and label_path.exists() else [],
        modality=condition.modality,
        variant=condition.variant,
        family=condition.family,
        labels_directory=labels_directory,
        predictions_directory=predictions_directory,
    )


def _loose_record(image_path: Path, check_label: bool) -> ImageRecord:
    """Return a record for an image sitting loose in its stage directory.

    The set is named by the image's series and modality, both of which its
    filename states, so two images in one directory can belong to different
    sets -- an Adaptive Polishing pair is ``adaptive_sem_labels`` and
    ``adaptive_fib_labels``, never one directory shared. An image whose name
    says neither gets a bare record: nothing here would be a fact rather than a
    guess.
    """
    classified = classify_loose_image(image_path.name)
    if classified is None or classified.modality is None:
        return ImageRecord(image_path=image_path)

    siblings = subdirectories_by_name(image_path.parent)
    labels_directory, _ = resolve_loose_labels_directory(
        image_path.parent, classified.series, classified.modality, siblings
    )
    predictions_directory, _ = resolve_loose_predictions_directory(
        image_path.parent, classified.series, classified.modality, siblings
    )
    label_path = labels_directory / image_path.name
    return ImageRecord(
        image_path=image_path,
        labels=[label_path] if check_label and label_path.exists() else [],
        modality=classified.modality,
        series=classified.series,
        labels_directory=labels_directory,
        predictions_directory=predictions_directory,
        stage_path=image_path.parent,
    )


def group_identity(
    record: ImageRecord, cache: dict[Path, Path | None] | None = None
) -> Path | None:
    """Return the label path identifying *record*'s group, existing or not.

    A scanned record carries the directory its label belongs in, so nothing is
    re-derived. The fallback covers a record built by hand, answering from the
    image directory where there is one and from the filename where the image
    sits loose.
    """
    if record.labels:
        return record.labels[0]
    if record.labels_directory is not None:
        return record.labels_directory / record.image_path.name
    labels_directory = labels_directory_for(record.image_path, cache)
    if labels_directory is None:
        return None
    return labels_directory / record.image_path.name


def group_key(
    record: ImageRecord, cache: dict[Path, Path | None] | None = None
) -> str:
    """Return the identity of the group *record* belongs to.

    Records resolving to the same label share a key, whether or not that label
    exists yet. An image in no recognised image directory has no label to be
    keyed by, so it is keyed by itself and forms a group of one.
    """
    identity = group_identity(record, cache)
    if identity is None:
        return str(record.image_path)
    return str(identity)


def build_group(
    record: ImageRecord, families: FamiliesLookup | None = None
) -> LabelGroup:
    """Return the label group *record* belongs to, read from the filesystem.

    Sibling images are found by resolving which image sets in the same stage
    share this record's label directory, then looking for the same filename in
    each. Reading from disk rather than from a scanned index means a command
    given one file path still expands to the whole group. A sibling carries its
    stage, label and predictions directories and the group's state, as a
    scanned record would, since any of them may be the one predicted.

    Args:
        families: Reused across a batch, so each project's file is read once.

    Raises:
        FamiliesFileError: If the project's families file will not read.
    """
    identity = group_identity(record)
    if identity is None:
        return LabelGroup(label_path=None, records=(record,), identity=None)

    labels_directory = identity.parent
    stage_directory = labels_directory.parent
    filename = identity.name
    label_path = identity if identity.exists() else None

    siblings: list[ImageRecord] = []
    corrections = (families or FamiliesLookup()).at(stage_directory)
    listing = subdirectories_by_name(stage_directory)
    for image_set in build_image_sets(stage_directory, corrections=corrections):
        if image_set.labels_directory != labels_directory:
            continue
        found = image_files_by_name(image_set.directory).get(filename)
        if found is None:
            continue
        if found == record.image_path:
            siblings.append(record)
            continue
        assert image_set.modality is not None, "a directory set names its modality"
        predictions_directory, _, _ = resolve_predictions_directory(
            image_set.directory, image_set.modality, listing, corrections=corrections
        )
        siblings.append(
            ImageRecord(
                image_path=found,
                labels=[label_path] if label_path is not None else [],
                project=record.project,
                lamella=record.lamella,
                stage=record.stage,
                stage_path=stage_directory,
                modality=image_set.modality,
                variant=image_set.variant,
                family=image_set.family,
                labels_directory=labels_directory,
                predictions_directory=predictions_directory,
                label_shared=image_set.labels_are_shared,
                state=record.state,
                task=record.task,
            )
        )

    if not siblings:  # pragma: no cover - defensive; the record's own image is missing
        siblings = [record]
    return LabelGroup(
        label_path=label_path, records=tuple(siblings), identity=identity
    )


def build_groups(
    records: Iterable[ImageRecord], families: FamiliesLookup | None = None
) -> list[LabelGroup]:
    """Return the distinct label groups the given records belong to.

    Records of the same group collapse into one entry, so a caller that passes
    all five conditions of a scene gets one group back, not five.

    Args:
        families: As :func:`build_group` takes it. Given corrections, it groups
            a stage under corrections other than its project's file holds.
    """
    groups: dict[str, LabelGroup] = {}
    cache: dict[Path, Path | None] = {}
    families = FamiliesLookup() if families is None else families
    for record in records:
        key = group_key(record, cache)
        if key in groups:
            continue
        groups[key] = build_group(record, families)
    return list(groups.values())


def primary_record(
    records: Iterable[ImageRecord], families: FamiliesLookup | None = None
) -> ImageRecord:
    """Return the record standing for a scene: its *primary* imaging condition.

    As :func:`~..frames.primary_among` chooses it, with the project's
    ``AM_families.yaml`` looked up here, so no caller can choose without it.
    What is predicted, measured and kept by a one-per-scene export.

    Args:
        families: Reused across a batch, so each project's file is read once.

    Raises:
        FamiliesFileError: If the project's families file will not read.
    """
    records = list(records)
    if not any(record.family for record in records):
        return primary_among(records, NO_CORRECTIONS)
    return primary_among(records, (families or FamiliesLookup()).at(records[0].image_path))


def expand_to_groups(
    records: Iterable[ImageRecord], known: Iterable[ImageRecord] = ()
) -> list[ImageRecord]:
    """Return the given records plus every image sharing a label with them.

    The records passed in are preserved as the objects they are, so a caller
    holding them -- a long-running UI, for example -- sees its own records
    updated when a transition succeeds. **A sibling found on disk is the
    caller's own record where *known* holds one for its image**, and a new
    record only where it does not; membership is decided by the disk either way.

    Deduplication is by image path, so a file named twice -- once directly and
    once through its group -- is declared once.

    Args:
        known: Records the caller holds and wants kept in step, such as every
            record of the lamellae a change touches.
    """
    supplied = list(records)
    held = {str(record.image_path): record for record in known}
    seen: set[str] = set()
    expanded: list[ImageRecord] = []

    for record in supplied:
        key = str(record.image_path)
        if key not in seen:
            seen.add(key)
            expanded.append(record)

    for group in build_groups(supplied):
        for record in group.records:
            key = str(record.image_path)
            if key not in seen:
                seen.add(key)
                expanded.append(held.get(key, record))

    if len(expanded) > len(supplied):
        logger.info(
            "Expanded %d record(s) to %d through shared labels",
            len(supplied), len(expanded),
        )
    return expanded


def describe_group(group: LabelGroup) -> str:
    """Return a one-line description of a group, for confirmation prompts.

    Images are named by the directory holding them, which is the imaging
    condition -- ``sem``, ``sem_5kv`` -- and so is what distinguishes the
    members of a shared group from each other.
    """
    if group.label_path is None:
        return f"{len(group.records)} image(s), no label yet"
    names = ", ".join(
        sorted(record.image_path.parent.name for record in group.records)
    )
    return f"{group.label_path.name} shared by {len(group.records)} image(s): {names}"
