"""Compatibility shim providing ProjectIndex at package level.

This module simply re-exports ProjectIndex for convenience.
"""
from __future__ import annotations

from .scanner import ProjectIndex

__all__ = ["ProjectIndex"]
