"""Tests for the scanner module.

The test attempts to run against a reference dataset folder named
`20260728_CB_AM_bimap_overnight` if present. If not present the test is
skipped so the test suite can run in CI without the large dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Ensure src/ is on sys.path when running tests from the repository root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.scanner import Scanner


# Look for a reference dataset folder under the repository root. This
# makes the test robust to where pytest is invoked from and to minor naming
# differences (e.g. 'bitmap' vs 'bimap'). If no matching dataset is found the
# module is skipped.
DATASET_GLOB = "20260728_CB_AM*"
ROOT = Path(__file__).resolve().parents[1]


def _find_dataset() -> Path | None:
    for p in ROOT.glob(DATASET_GLOB):
        if p.is_dir():
            return p
    return None


dataset_path = _find_dataset()
if dataset_path is None:
    pytest.skip(f"No dataset matching {DATASET_GLOB} under {ROOT}", allow_module_level=True)


def test_build_index_on_reference_dataset():
    start = dataset_path
    scanner = Scanner(start)
    idx = scanner.build_index()

    # basic assertions: index is a ProjectIndex and has at least one project
    assert idx is not None
    projects = idx.all_projects()
    assert isinstance(projects, list)
    assert len(projects) >= 1

    # each project should have a name and path
    for p in projects:
        assert p.name
        assert p.path.exists()
