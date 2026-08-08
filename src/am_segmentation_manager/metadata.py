"""Metadata utilities for am_segmentation_manager.

This module contains helpers for reading and merging metadata from
experiment.yaml and AM_metadata.yaml. It keeps merging logic in one place so
higher-level code works with dataclasses and merged dictionaries.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def merge_metadata(base: Dict, override: Dict) -> Dict:
    """Merge two metadata dictionaries, with `override` taking precedence.

    This performs a shallow merge: keys present in override replace base
    values. Deeper merging can be added when needed.
    """
    if not base:
        base = {}
    if not override:
        return dict(base)
    merged = dict(base)
    merged.update(override)
    logger.debug("Merged metadata keys: %s", list(merged.keys()))
    return merged


def parse_created_at(value: float | int | str | None) -> datetime | None:
    """Convert an Autolamella ``created_at`` Unix timestamp to a datetime.

    The *experiment.yaml* file stores creation time as a Unix timestamp
    (e.g. ``1785231172.267575``).  This helper converts the value to a
    timezone-aware UTC datetime.

    Args:
        value: Unix timestamp as a float, int, or numeric string.
            Returns ``None`` when *value* is ``None`` or cannot be parsed.
    """
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        logger.warning("Could not parse created_at timestamp: %r", value)
        return None
