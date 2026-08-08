"""Metadata utilities for am_segmentation_manager.

This module contains helpers for reading and merging metadata from
experiment.yaml and AM_metadata.yaml. It keeps merging logic in one place so
higher-level code works with dataclasses and merged dictionaries.
"""
from __future__ import annotations

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
