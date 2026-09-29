"""The record a training export leaves beside its CSV files.

``<prefix>_manifest.yaml`` says what the lists were made from -- the scope, the
split, its seed and what it kept together, how many rows each side holds, which
am-seg wrote them, when and under whose name -- so a training run can be
described, and repeated, from the files it read. **Written after the CSVs**, so a manifest names only files
that were written whole. Nothing here chooses rows or writes a CSV.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from .. import __version__
from ..history import resolve_author
from ..io.yaml import write_yaml

logger = logging.getLogger(__name__)

#: What follows the prefix in a manifest's filename.
MANIFEST_SUFFIX: Final[str] = "_manifest.yaml"

#: Written into every manifest, so a reader can tell a layout it does not know.
FILE_VERSION: Final[int] = 1

_HEADER: Final[str] = """\
# What this training set was made from, written by am-seg beside its CSV files.
#
# `scope` is what the export was narrowed to, in the words of whatever made it:
# the Export tab's filters, or the command line's selectors. `split.keep_together`
# is what the split never divided between training and validation: an
# acquisition, a lamella or a project. A manifest that does not say kept
# acquisitions together.
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
    user: str | None = None,
) -> Path:
    """Write the manifest of a finished training export, and return where it went.

    Args:
        files: Each CSV by its role, None for one this export did not write.
        user: The name to record, resolved as a history entry's author is.
    """
    by, _ = resolve_author(user)
    body = {
        "version": FILE_VERSION,
        "am_seg_version": __version__,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "written_by": by or "",
        "source": str(source),
        "scope": scope,
        "files": {role: file.name if file else None for role, file in files.items()},
        "columns": list(columns),
        "split": {
            "train_ratio": train_ratio,
            "seed": seed,
            "keep_together": keep_together,
        },
        "conditions": conditions,
        "include_empty_labels": include_empty_labels,
        "counts": dict(counts),
    }
    write_yaml(path, body, header=_HEADER)
    logger.info("Wrote the manifest to %s", path)
    return path
