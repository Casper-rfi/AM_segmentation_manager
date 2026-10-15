"""Tests for the metadata module."""
from __future__ import annotations

from am_segmentation_manager.metadata import merge_metadata


def test_merge_metadata():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    merged = merge_metadata(base, override)
    assert merged == {"a": 1, "b": 3, "c": 4}
