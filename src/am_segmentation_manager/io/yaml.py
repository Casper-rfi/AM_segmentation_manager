"""Simple YAML helpers.

This module wraps PyYAML usage for the rest of the package so the import and
I/O error handling are localised. Consumers should call read_yaml(path).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    Raises RuntimeError with a helpful message if PyYAML is not installed or
    the file cannot be parsed.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("PyYAML is required to read YAML files: install pyyaml") from exc

    if not path.exists():
        logger.debug("YAML file does not exist: %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            logger.debug("Read YAML from %s: %s keys", path, len(data))
            return data
    except Exception as exc:
        raise RuntimeError(f"Failed to parse YAML file {path}: {exc}") from exc
