#!/usr/bin/env python3
"""Opaque ids for the overviews the reprojection fixtures are built from.

The fixtures in ``tests/data`` name each overview ``overview_NN.tif`` rather
than by the experiment it was captured from, so the repository discloses no
project, grid or collaborator name. Which real image each id stands for is kept
outside the repository, in ``~/.am_segmentation_manager/overview_sources.json``,
so a fixture stays traceable to the header it came from.

**This is a helper for the two standalone scripts that write those fixtures**,
not part of the package. An id, once assigned, is never reused for another
image: both scripts write the same collection and must agree on what a name
means.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Where the ids are resolved back to the images they were captured from. Outside
#: the repository by design -- it is the one file that carries the real names.
SOURCES_PATH = Path.home() / ".am_segmentation_manager" / "overview_sources.json"

NOTE = (
    "Which real overview each opaque id in tests/data/*.json came from. "
    "Kept outside the repository so the fixtures name no experiment."
)


def load_sources(path: Path = SOURCES_PATH) -> dict[str, str]:
    """Return the recorded ids, mapping each opaque name to its real path."""
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["sources"])


def identify(real: str, path: Path = SOURCES_PATH) -> str:
    """Return the opaque id standing for *real*, assigning one if it is new.

    Args:
        real: The overview's path relative to the collection.

    Returns:
        The opaque filename the fixtures record, e.g. ``overview_07.tif``.
    """
    sources = load_sources(path)
    for opaque, recorded in sources.items():
        if recorded == real:
            return opaque
    taken = {int(name.removeprefix("overview_").removesuffix(".tif")) for name in sources}
    opaque = f"overview_{max(taken, default=0) + 1:02d}.tif"
    sources[opaque] = real
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"note": NOTE, "sources": dict(sorted(sources.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return opaque
