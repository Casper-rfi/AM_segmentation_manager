"""Images that share a label, and the state they share with it.

Several imaging conditions of one scene share a single segmentation, which makes
the label -- not the image -- the unit of proofreading. A :class:`LabelGroup` is
one label file and every image referencing it; a private label forms a group of
one, keeping this a single code path.

A group is identified by where its label lives *or would live*, so images
sharing a label directory are one group before any label exists -- keying on the
image instead would split them, then merge them the moment a label appeared.
Groups are built from the filesystem rather than a scanned index, so a command
given one file path expands correctly.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..layout import (
    RecordState,
    base_path,
    build_image_sets,
    classify_image_directory,
    image_files_by_name,
    resolve_labels_directory,
    state_of_path,
    subdirectories_by_name,
)
from ..models import ImageRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelGroup:
    """A label and every image that shares it.

    Attributes:
        label_path: The label's location with any state folder stripped, or
            None when the group's images have no label yet.
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

    def state(self) -> RecordState:
        """Return the group's workflow state.

        The label's location is authoritative: it is the thing being
        proofread, and it is what a state change is ultimately about. Without a
        label, the image's own location is all there is to go on.
        """
        located = self.located_label()
        if located is not None:
            return state_of_path(located)
        return state_of_path(self.records[0].image_path)

    def located_label(self) -> Path | None:
        """Return where the label actually is on disk, or None if absent."""
        if self.label_path is None:
            return None
        return _existing_state_variant(self.label_path)

    def inconsistent_records(self) -> list[ImageRecord]:
        """Return images whose own state disagrees with the group's.

        Normally empty. A non-empty result means files were moved by something
        other than this package, or an operation was interrupted partway.
        """
        state = self.state()
        return [
            record for record in self.records
            if state_of_path(record.image_path) != state
        ]

    @property
    def is_consistent(self) -> bool:
        """Return whether every image agrees with the group's state."""
        return not self.inconsistent_records()


def _existing_state_variant(path: Path) -> Path | None:
    """Return where a file is, across its possible state locations.

    Unlike :func:`~am_segmentation_manager.operations.selection.existing_variant`
    this does not reject a file present in several states: reporting a group's
    state must never raise, and duplicate state copies are surfaced separately
    as their own finding.
    """
    from ..layout import state_variants

    for candidate in state_variants(path):
        if candidate.exists():
            return candidate
    return None


def labels_directory_for(
    image_path: Path, cache: dict[Path, Path | None] | None = None
) -> Path | None:
    """Return where *image_path*'s label lives or would live.

    Answers for a label directory that does not exist yet, which is what lets a
    group be identified before anything has been annotated. Returns None when
    the image sits in no recognised image directory.

    Args:
        cache: Reused across a batch. Resolution lists the stage's
            subdirectories, which is one listing per image directory rather
            than per image.
    """
    image_directory = base_path(image_path).parent
    if cache is not None and image_directory in cache:
        return cache[image_directory]

    classification = classify_image_directory(image_directory.name)
    resolved: Path | None = None
    if classification is not None:
        resolved, _, _ = resolve_labels_directory(
            image_directory,
            classification.modality,
            subdirectories_by_name(image_directory.parent),
        )
    if cache is not None:
        cache[image_directory] = resolved
    return resolved


def group_identity(
    record: ImageRecord, cache: dict[Path, Path | None] | None = None
) -> Path | None:
    """Return the label path identifying *record*'s group, existing or not."""
    if record.labels:
        return base_path(record.labels[0])
    labels_directory = labels_directory_for(record.image_path, cache)
    if labels_directory is None:
        return None
    return labels_directory / base_path(record.image_path).name


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
        return str(base_path(record.image_path))
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
    label_path = identity if _existing_state_variant(identity) is not None else None

    siblings: list[ImageRecord] = []
    for image_set in build_image_sets(stage_directory):
        if image_set.labels_directory != labels_directory:
            continue
        found = image_files_by_name(image_set.directory).get(filename)
        if found is None:
            continue
        if base_path(found) == base_path(record.image_path):
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
                selected=state_of_path(found) == "selected",
                corrected=state_of_path(found) == "corrected",
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

    Deduplication is by image path with any state folder stripped, so the same
    file can never appear twice and be moved twice.
    """
    supplied = list(records)
    seen: set[str] = set()
    expanded: list[ImageRecord] = []

    for record in supplied:
        key = str(base_path(record.image_path))
        if key not in seen:
            seen.add(key)
            expanded.append(record)

    for group in build_groups(supplied):
        for record in group.records:
            key = str(base_path(record.image_path))
            if key not in seen:
                seen.add(key)
                expanded.append(record)

    if len(expanded) > len(supplied):
        logger.info(
            "Expanded %d record(s) to %d through shared labels",
            len(supplied), len(expanded),
        )
    return expanded


def find_inconsistent_groups(records: Iterable[ImageRecord]) -> list[LabelGroup]:
    """Return the groups among *records* whose images disagree with their label."""
    return [group for group in build_groups(records) if not group.is_consistent]


def describe_group(group: LabelGroup) -> str:
    """Return a one-line description of a group, for confirmation prompts.

    Images are named by their image directory rather than their immediate
    parent, so one that has moved into a state folder is still reported under
    the directory a reader recognises.
    """
    if group.label_path is None:
        return f"{len(group.records)} image(s), no label yet"
    names = ", ".join(
        sorted(base_path(record.image_path).parent.name for record in group.records)
    )
    return f"{group.label_path.name} shared by {len(group.records)} image(s): {names}"
