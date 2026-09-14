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

from ..layout import (
    build_image_sets,
    classify_image_directory,
    classify_loose_image,
    image_files_by_name,
    resolve_labels_directory,
    resolve_loose_labels_directory,
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
            subdirectories, which is one listing per image directory rather
            than per image. Only the directory-based answer is cached: a loose
            set is chosen by filename, and one directory holds several of them.
    """
    image_directory = image_path.parent
    classification = classify_image_directory(image_directory.name)
    if classification is None:
        return _loose_labels_directory_for(image_path)

    if cache is not None and image_directory in cache:
        return cache[image_directory]
    resolved, _, _ = resolve_labels_directory(
        image_directory,
        classification.modality,
        subdirectories_by_name(image_directory.parent),
    )
    if cache is not None:
        cache[image_directory] = resolved
    return resolved


def _loose_labels_directory_for(image_path: Path) -> Path | None:
    """Return where a loose image's label lives or would live.

    The set is named by the image's series and modality, both of which its
    filename states, so two images in one directory can belong to different
    sets -- an Adaptive Polishing pair is ``adaptive_sem_labels`` and
    ``adaptive_fib_labels``, never one directory shared.
    """
    classified = classify_loose_image(image_path.name)
    if classified is None or classified.modality is None:
        return None
    resolved, _ = resolve_loose_labels_directory(
        image_path.parent,
        classified.series,
        classified.modality,
        subdirectories_by_name(image_path.parent),
    )
    return resolved


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


def build_group(record: ImageRecord) -> LabelGroup:
    """Return the label group *record* belongs to, read from the filesystem.

    Sibling images are found by resolving which image sets in the same stage
    share this record's label directory, then looking for the same filename in
    each. Reading from disk rather than from a scanned index means a command
    given one file path still expands to the whole group.
    """
    identity = group_identity(record)
    if identity is None:
        return LabelGroup(label_path=None, records=(record,), identity=None)

    labels_directory = identity.parent
    stage_directory = labels_directory.parent
    filename = identity.name
    label_path = identity if identity.exists() else None

    siblings: list[ImageRecord] = []
    for image_set in build_image_sets(stage_directory):
        if image_set.labels_directory != labels_directory:
            continue
        found = image_files_by_name(image_set.directory).get(filename)
        if found is None:
            continue
        if found == record.image_path:
            siblings.append(record)
            continue
        siblings.append(
            ImageRecord(
                image_path=found,
                labels=[label_path] if label_path is not None else [],
                project=record.project,
                lamella=record.lamella,
                stage=record.stage,
                modality=image_set.modality,
                variant=image_set.variant,
                label_shared=image_set.labels_are_shared,
            )
        )

    if not siblings:  # pragma: no cover - defensive; the record's own image is missing
        siblings = [record]
    return LabelGroup(
        label_path=label_path, records=tuple(siblings), identity=identity
    )


def build_groups(records: Iterable[ImageRecord]) -> list[LabelGroup]:
    """Return the distinct label groups the given records belong to.

    Records of the same group collapse into one entry, so a caller that passes
    all five conditions of a scene gets one group back, not five.
    """
    groups: dict[str, LabelGroup] = {}
    cache: dict[Path, Path | None] = {}
    for record in records:
        key = group_key(record, cache)
        if key in groups:
            continue
        groups[key] = build_group(record)
    return list(groups.values())


def condition_sort_key(record: ImageRecord) -> tuple[bool, str]:
    """Order the unsuffixed image directory first, then variants by name."""
    return (record.variant is not None, (record.variant or "").lower())


def primary_record(records: Iterable[ImageRecord]) -> ImageRecord:
    """Return the record standing for a scene: its *primary* imaging condition.

    The unsuffixed image directory where there is one, otherwise the
    first-sorting variant -- the same image the curation window displays, and
    the one a training export keeps. Choosing it this way rather than filtering
    for the unsuffixed directory matters for the oldest layouts, which have no
    unsuffixed directory at all: their SEM sets are ``SEM_8``, ``SEM_8_fi_1``
    and so on.
    """
    return min(records, key=condition_sort_key)


def expand_to_groups(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    """Return the given records plus every image sharing a label with them.

    The records passed in are preserved as the objects they are, so a caller
    holding them -- a long-running UI, for example -- sees its own records
    updated when a transition succeeds. Siblings discovered on disk are added as
    new records.

    Deduplication is by image path, so a file named twice -- once directly and
    once through its group -- is declared once.
    """
    supplied = list(records)
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
                expanded.append(record)

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
