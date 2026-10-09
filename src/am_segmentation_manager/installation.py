"""How this copy was installed, and so how it is brought up to date.

A surface outside the four layers, reading only the record the installer left
beside the package. **Nothing here reaches the network**: it says where to look
for a newer version and what to run, and looks for nothing itself.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from .identity import APPLICATION_ID, REPOSITORY_URL

#: A source checkout built with ``uv sync``, the repository as ``uv tool install``
#: fetched it, or anything else.
InstallKind = Literal["checkout", "repository", "other"]

#: The file an installer writes for a package that did not come from an index.
DIRECT_URL_FILENAME: Final[str] = "direct_url.json"


def install_record() -> Mapping[str, Any]:
    """Return what the installer recorded of where this package came from, or ``{}``."""
    try:
        text = distribution(APPLICATION_ID).read_text(DIRECT_URL_FILENAME)
    except PackageNotFoundError:
        return {}
    try:
        record = json.loads(text) if text else {}
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


def install_kind(record: Mapping[str, Any]) -> InstallKind:
    """Return how the package was installed, from its install record."""
    if (record.get("dir_info") or {}).get("editable"):
        return "checkout"
    if record.get("vcs_info"):
        return "repository"
    return "other"


def checkout_directory(record: Mapping[str, Any]) -> Path | None:
    """Return the source checkout *record* names, or None where it names no folder."""
    address = urlparse(str(record.get("url") or ""))
    if install_kind(record) != "checkout" or address.scheme != "file":
        return None
    return Path(url2pathname(address.path))


def update_steps(record: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return how to update this copy, a line per step, the first saying where to look.

    Args:
        record: The install record, read from the installed package where omitted.
    """
    record = install_record() if record is None else record
    look = f"Newer versions, and what changed in each, are listed at {REPOSITORY_URL}"
    kind = install_kind(record)
    if kind == "checkout":
        directory = checkout_directory(record)
        where = f"in {directory}" if directory else "in the checkout"
        return (
            look,
            f"This is a source checkout: run git pull {where}",
            "then uv sync with the extras you use, such as: uv sync --extra gui --extra ml",
        )
    if kind == "repository":
        return (
            look,
            "This was installed with uv tool install: run the command you installed "
            "with again, adding --upgrade",
            "uv tool upgrade forgets --torch-backend, so is not the way",
        )
    return (look, "To update, follow Installing in the README there")
