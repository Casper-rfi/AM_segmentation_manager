"""The record a training export leaves beside its CSV files.

``<prefix>_manifest.yaml`` says what the lists were made from -- the scope, the
split, its seed and what it kept together, how many rows each side holds, which
am-seg wrote them, when and under whose name -- so a training run can be
described, and repeated, from the files it read. **Written after the CSVs**, so a manifest names only files
that were written whole, **and it names no folder of this machine**: where the
collection is, it says from its own folder (``test_a_manifest_names_no_folder_of_this_machine``). Nothing here chooses rows or writes a CSV;
:func:`read_training_manifest` is how a training run finds the lists again.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .. import __version__
from ..history import resolve_author, utc_stamp
from ..io.path import NoRelativePath, exported_relative_path
from ..io.yaml import read_yaml, version_problem, write_yaml

logger = logging.getLogger(__name__)

#: What follows the prefix in a manifest's filename.
MANIFEST_SUFFIX: Final[str] = "_manifest.yaml"

#: Written into every manifest, so a reader can tell a layout it does not know.
FILE_VERSION: Final[int] = 1

_HEADER: Final[str] = """\
# What this training set was made from, written by am-seg beside its CSV files.
#
# Every path in the lists is written from the collection's folder. `collection`
# is where that folder was, from this file's own folder; where the two have been
# moved apart, name the collection when training: scan it in the window, or
# give `am-seg train --start-path`.
#
# `files.images`, where there is one, lists every image once with the side of
# the split it is on (for folds, the fold it validates in) and what it is of.
#
# `scope` is what the export was narrowed to, in the words of whatever made it:
# the Export tab's filters, or the command line's selectors. `split.keep_together`
# is what the split never divided between training and validation: an
# acquisition, a lamella or a project. A manifest that does not say kept
# acquisitions together. A cross-validation export has `split.folds`, and under
# `folds` what each fold holds; its `train_ratio` is the share each fold trains on.
"""


def manifest_path(output_dir: Path, prefix: str) -> Path:
    """Return where the manifest for *prefix* lives, existing or not."""
    return output_dir / f"{prefix}{MANIFEST_SUFFIX}"


def write_training_manifest(
    path: Path,
    *,
    source: Path,
    scope: str,
    files: Mapping[str, Path | None],
    columns: Sequence[str],
    train_ratio: float,
    seed: int | None,
    keep_together: str,
    conditions: str,
    include_empty_labels: bool,
    counts: Mapping[str, int],
    folds: Sequence[Mapping[str, object]] = (),
    user: str | None = None,
) -> Path:
    """Write the manifest of a finished training export, and return where it went.

    Args:
        files: Each CSV by its role, None for one this export did not write.
        folds: What each fold of a cross-validation export holds, in order;
            empty for a single split.
        user: The name to record, resolved as a history entry's author is.
    """
    by, _ = resolve_author(user)
    split: dict[str, object] = {
        "train_ratio": train_ratio,
        "seed": seed,
        "keep_together": keep_together,
    }
    if folds:
        split["folds"] = len(folds)
    body = {
        "version": FILE_VERSION,
        "am_seg_version": __version__,
        "written_at": utc_stamp(),
        "written_by": by or "",
        "collection": _from(path.parent, source),
        "scope": scope,
        "files": {role: file.name if file else None for role, file in files.items()},
        "columns": list(columns),
        "split": split,
        "conditions": conditions,
        "include_empty_labels": include_empty_labels,
        "counts": dict(counts),
    }
    if folds:
        body["folds"] = [dict(fold) for fold in folds]
    write_yaml(path, body, header=_HEADER)
    logger.info("Wrote the manifest to %s", path)
    return path


@dataclass(frozen=True)
class TrainingManifest:
    """What a manifest says about the lists beside it.

    Attributes:
        path: The manifest itself.
        collection: Where the manifest says the folder every path in the lists
            is written from is: from its own folder, or as an older manifest
            recorded it in full. None where it does not say.
        train: The training list, or None if the export wrote none.
        validation: The validation list, or None if every row went to training.
        images: The file saying which side each image is on and what it is of,
            or None for an export that wrote none.
        columns: The lists' header.
        folds: How many folds a cross-validation export wrote; 0 for one split.
        include_empty_labels: Whether background-only labels were exported on purpose.
    """

    path: Path
    collection: Path | None
    train: Path | None
    validation: Path | None
    images: Path | None
    columns: tuple[str, ...]
    folds: int
    include_empty_labels: bool


def read_training_manifest(path: Path) -> TrainingManifest:
    """Return what the manifest at *path* records.

    Raises:
        ValueError: If it is missing, is not a manifest, or was written by a
            later am-seg.
    """
    try:
        document = read_yaml(path)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    if not document or not isinstance(document.get("files"), dict):
        raise ValueError(f"{path} is not a training export's manifest")
    problem = version_problem(document.get("version"), path, FILE_VERSION, required=True)
    if problem is not None:
        raise ValueError(problem)

    def beside(role: str) -> Path | None:
        name = document["files"].get(role)
        return path.parent / str(name) if name else None

    split = document.get("split")
    return TrainingManifest(
        path=path,
        collection=_collection(path, document),
        train=beside("train"),
        validation=beside("validation"),
        images=beside("images"),
        columns=tuple(str(column) for column in document.get("columns") or ()),
        folds=int(split.get("folds") or 0) if isinstance(split, dict) else 0,
        include_empty_labels=bool(document.get("include_empty_labels")),
    )


def _from(folder: Path, collection: Path) -> str | None:
    """Return *collection* as a path from *folder*, or None where there is none, as across drives."""
    try:
        return exported_relative_path(collection, folder)
    except NoRelativePath as error:
        logger.warning("The manifest will not say where the collection is: %s", error)
        return None


def _collection(path: Path, document: dict) -> Path | None:
    """Return where a manifest says the collection is, or None where it does not."""
    recorded = document.get("collection")
    if recorded:
        return path.parent / str(recorded)
    # Before `collection`, the folder was recorded as the export was given it:
    # in full, or from wherever the command happened to be run.
    older = Path(str(document.get("source") or ""))
    return older if older.is_absolute() else None
