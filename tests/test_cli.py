"""Tests for CLI project listing output."""
from __future__ import annotations

from pathlib import Path
import re
import sys

# Ensure src/ is on sys.path when running tests from the repository root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.cli import main


def _write_file(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_list_projects_includes_image_and_label_counts(capfd, tmp_path: Path):
    project_root = tmp_path / "AutoLamella-1"
    project_root.mkdir(parents=True)
    (project_root / "experiment.yaml").write_text("project: test")

    lamella = project_root / "01-test-lamella"
    stage = lamella / "Adaptive_polish_1"
    sem = stage / "sem"
    fib = stage / "fib"
    labels = stage / "labels"
    labels_fib = stage / "labels_fib"

    _write_file(sem / "selected.tif")
    _write_file(sem / "deselected" / "deselected.tif")
    _write_file(sem / "corrected" / "corrected.tif")
    _write_file(fib / "selected.tif")
    _write_file(labels / "selected.tif")
    _write_file(labels / "deselected" / "deselected.tif")
    _write_file(labels / "corrected" / "corrected.tif")
    _write_file(labels_fib / "selected.tif")

    result = main(["list-projects", str(tmp_path)])
    captured = capfd.readouterr()

    assert result == 0
    assert "001: AutoLamella-1" in captured.out
    assert "Counts shown as: deselected / selected / corrected / all" in captured.out
    assert "SEM images" in captured.out
    assert "FIB images" in captured.out
    assert "SEM labels" in captured.out
    assert "FIB labels" in captured.out
    # SEM artifacts span all three mutually exclusive states, while FIB
    # artifacts are shown separately and remain selected.
    assert re.search(r"SEM images\s+1\s+1\s+1\s+3", captured.out)
    assert re.search(r"FIB images\s+0\s+1\s+0\s+1", captured.out)
    assert re.search(r"SEM labels\s+1\s+1\s+1\s+3", captured.out)
    assert re.search(r"FIB labels\s+0\s+1\s+0\s+1", captured.out)
    assert "INFO:" not in captured.out
    assert "01-test-lamella" in captured.out
