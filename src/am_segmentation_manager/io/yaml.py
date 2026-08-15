"""Simple YAML helpers.

Wraps PyYAML for the rest of the package so the import and I/O error handling are
localised. Call :func:`read_yaml`.

Autolamella's ``experiment.yaml`` reaches hundreds of kilobytes and the scanner
reads one per project, so parsing dominates scan time. PyYAML's C loader is used
where it was built against libyaml, falling back to the pure Python loader.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
