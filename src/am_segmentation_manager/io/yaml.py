"""Reading and writing the YAML files the package keeps.

A missing file reads as an empty mapping, most of these files being optional.
**A write lands whole or not at all**, through :func:`.atomic.write_atomically`,
so an interrupted run leaves the previous file rather than a truncated one.
PyYAML's C loader is used where it was built against libyaml: Autolamella's
``experiment.yaml`` reaches a megabyte and a scan reads one per project.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .atomic import write_atomically

logger = logging.getLogger(__name__)


def _loader():
    """Return the fastest available safe YAML loader.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("PyYAML is required to read YAML files: install pyyaml") from exc

    try:
        return yaml.CSafeLoader
    except AttributeError:  # pragma: no cover - depends on the local PyYAML build
        logger.debug("libyaml not available; using the pure Python YAML loader")
        return yaml.SafeLoader


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    A missing file reads as an empty mapping, since most of the files this
    package reads are optional.

    Raises RuntimeError with a helpful message if PyYAML is not installed or
    the file cannot be parsed.
    """
    import yaml

    loader = _loader()
    if not path.exists():
        logger.debug("YAML file does not exist: %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=loader) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse YAML file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a mapping at the top level of {path}, found {type(data).__name__}")
    logger.debug("Read YAML from %s: %s keys", path, len(data))
    return data


def write_yaml(
    path: Path, body: Mapping[str, Any], *, header: str = "", sort_keys: bool = False
) -> None:
    """Write *body* to *path* as YAML beneath *header*, replacing the file whole.

    Args:
        header: Comment lines to open the file with, each ending in a newline.
        sort_keys: Sort every mapping's keys; otherwise they keep their order.
    """
    import yaml

    text = header + yaml.safe_dump(
        dict(body), sort_keys=sort_keys, allow_unicode=True, default_flow_style=False
    )
    write_atomically(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
