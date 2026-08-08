"""Tests for the metadata module."""
from __future__ import annotations

from datetime import datetime, timezone
from am_segmentation_manager.metadata import merge_metadata, parse_created_at


def test_merge_metadata():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    merged = merge_metadata(base, override)
    assert merged == {"a": 1, "b": 3, "c": 4}


def test_parse_created_at():
    # Valid timestamp float/int/str
    dt = parse_created_at(1785231172.267575)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc

    dt_str = parse_created_at("1785231172.267575")
    assert isinstance(dt_str, datetime)

    # Invalid / None
    assert parse_created_at(None) is None
    assert parse_created_at("invalid_number") is None
