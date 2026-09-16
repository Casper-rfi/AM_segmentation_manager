"""The suite's refusal to let a test leave a top-level widget alive.

A leaked widget fails nothing where it leaks: it slows every later test that
applies the theme, so the cost turns up in another file, or only in a full run.
``conftest.no_widget_left_behind`` names the test instead, and this checks it
does.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qtpy") is None, reason="Qt binding not installed"
)

TESTS = Path(__file__).resolve().parent


def test_a_test_that_leaves_a_widget_alive_is_named_and_one_that_frees_its_own_is_not(
    tmp_path: Path,
):
    """Run in a separate pytest, because the failure has to land on a test and
    in this run it would land on this one."""
    (tmp_path / "test_leaky.py").write_text(
        textwrap.dedent(
            """
            import os
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

            from qtpy.QtWidgets import QApplication, QWidget

            from conftest import destroy

            APP = QApplication.instance() or QApplication([])
            KEPT = []

            def test_keeps_a_widget():
                KEPT.append(QWidget())

            def test_frees_its_widget():
                destroy(QWidget())
            """
        ),
        encoding="utf-8",
    )
    # The inner run needs the src/ and tests/ that pytest put on this
    # interpreter's path; a fresh interpreter does not inherit them.
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(TESTS), *sys.path]))

    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider",
         "-rE", "test_leaky.py"],
        cwd=tmp_path, capture_output=True, text=True, env=environment, timeout=120,
    )

    output = finished.stdout + finished.stderr
    assert "2 passed, 1 error" in output, output
    assert "ERROR at teardown of test_keeps_a_widget" in output, output
    assert "left 1 top-level widget(s) alive: QWidget" in output, output
    assert "test_frees_its_widget" not in output.split("short test summary")[-1], output


def test_destroying_a_window_takes_the_tooltip_it_showed_with_it(qt_app):
    """Qt's one tooltip belongs to no widget and, on Windows, outlives the
    window that showed it; the guard then blamed whichever test showed one."""
    from qtpy.QtCore import QPoint, QRect
    from qtpy.QtWidgets import QToolTip, QWidget

    from conftest import destroy, top_level_widget_names

    window = QWidget()
    QToolTip.showText(QPoint(200, 200), "path copied", None, QRect(), 2000)
    assert "QTipLabel" in top_level_widget_names()

    destroy(window)

    assert "QTipLabel" not in top_level_widget_names()
