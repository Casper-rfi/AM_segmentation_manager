"""Tests for CLI project listing output."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

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
    _write_file(sem / "deselected.tif")
    _write_file(sem / "corrected.tif")
    _write_file(fib / "selected.tif")
    _write_file(labels / "selected.tif")
    _write_file(labels / "deselected.tif")
    _write_file(labels / "corrected.tif")
    _write_file(labels_fib / "selected.tif")

    from am_segmentation_manager.state import read_state, write_state

    write_state(read_state(stage), {
        "labels/deselected.tif": "deselected",
        "labels/corrected.tif": "corrected",
    })

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


# ---------------------------------------------------------------------------
# find-specks
# ---------------------------------------------------------------------------

def _speckled_project(tmp_path: Path):
    """A one-lamella project whose SEM label holds a single stray pixel."""
    import numpy as np
    import tifffile

    from conftest import build_dataset

    dataset = build_dataset(tmp_path, acquisitions=1, modalities=("SEM",))
    clean = np.zeros((32, 40), dtype=np.uint8)
    clean[4:20, 4:20] = 2
    speckled = clean.copy()
    speckled[30, 38] = 1
    tifffile.imwrite(str(dataset.label("SEM", 0)), speckled)
    return dataset


def test_find_specks_names_the_label_and_where_to_look(capfd, tmp_path: Path):
    pytest.importorskip("tifffile")
    _speckled_project(tmp_path)

    result = main([
        "find-specks", "--start-path", str(tmp_path),
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
    ])
    out = capfd.readouterr().out

    assert result == 0
    assert "img_000_SEM.tif" in out
    assert "(38, 30)" in out, "the position is what makes it findable"


def test_find_specks_says_what_it_covered_when_it_finds_nothing(capfd, tmp_path: Path):
    pytest.importorskip("tifffile")
    _speckled_project(tmp_path)

    result = main([
        "find-specks", "--start-path", str(tmp_path), "--at-most", "1",
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
    ])
    out = capfd.readouterr().out

    assert result == 0
    assert "1 pixel(s) or less" in out and "label(s)" in out


def test_find_specks_refuses_a_threshold_below_one_pixel(tmp_path: Path):
    pytest.importorskip("tifffile")
    _speckled_project(tmp_path)

    result = main([
        "find-specks", "--start-path", str(tmp_path), "--at-most", "0",
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
    ])

    assert result == 2


# --- the crash log ----------------------------------------------------------

def test_the_log_file_carries_the_thread_and_appends(tmp_path: Path):
    """One file holds both the log and any crash dump, so what the process was
    doing is beside where it stopped -- and a second run must not erase the
    first one's evidence."""
    import logging

    from am_segmentation_manager.cli import open_log_file

    path = tmp_path / "logs" / "am-seg.log"
    for message in ("first run", "second run"):
        handler = open_log_file(path)
        record = logging.LogRecord(
            "am_segmentation_manager", logging.INFO, __file__, 0, message, (), None
        )
        handler.emit(record)
        handler.flush()

    written = path.read_text(encoding="utf-8")
    assert "first run" in written and "second run" in written
    assert "MainThread" in written


def test_the_log_file_receives_crash_dumps(tmp_path: Path):
    """The point of the file: a window that vanishes must leave a stack in it.

    Run in a subprocess that really takes the signal, because the only way to
    prove the dump reaches this file rather than stderr is to let it happen.
    """
    import os
    import subprocess
    import sys
    import textwrap

    log = tmp_path / "am-seg.log"
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from am_segmentation_manager.cli import open_log_file
        open_log_file(Path({str(log)!r}))
        os.abort()
        """
    )
    # The subprocess needs the src/ that pytest's own pythonpath setting puts
    # on this interpreter's, which a fresh interpreter does not inherit.
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, env=environment
    )

    assert finished.returncode != 0, finished.stderr.decode()
    written = log.read_text(encoding="utf-8")
    assert "Fatal Python error" in written
    assert "Current thread" in written
