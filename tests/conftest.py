"""Shared pytest configuration and fixtures.

Import path setup used to live here, and in a copy at the top of most test
files. It is now declared once as ``pythonpath = ["src"]`` in
``pyproject.toml``, which is pytest's supported mechanism, so the tests run
against the source tree whether or not the package is installed.

This file is the intended home for **shared dataset builders** — fixtures that
construct the canonical directory layouts from ``docs/reference_layouts.md`` so
that a new layout generation is supported by changing one place rather than the
32 test files that currently build trees by hand. See
``docs/01_refactor_plan.md`` section 5.1; the plan is to add builders here and
migrate tests onto them as each is touched, rather than in one sweep.
"""
from __future__ import annotations
