"""The application names itself to the desktop, however it was started.

Covers :mod:`am_segmentation_manager.identity`, the macOS name in
:mod:`am_segmentation_manager.gui.macos`, and the command line run the way a
launcher runs it: by ``python -m``, and under Windows' ``pythonw``, which gives
the process no console.
"""
from __future__ import annotations

import importlib
import logging
import runpy
import subprocess
import sys

import pytest

from am_segmentation_manager import identity
from am_segmentation_manager.gui import macos

#: The module, not the function the package re-exports under the same name.
cli_main = importlib.import_module("am_segmentation_manager.cli.main")


def test_every_picture_of_the_application_ships():
    """Each platform's launcher takes its own format of the one mark."""
    for path in (identity.ICON_PATH, identity.MAC_ICON_PATH, identity.WINDOWS_ICON_PATH):
        assert path.is_file(), path


@pytest.mark.skipif(sys.platform != "darwin", reason="the menu bar's name is macOS's")
def test_the_menu_bar_takes_the_name_it_is_given():
    """Read back from the bundle AppKit names the process by."""
    before = macos.bundle_name()
    try:
        assert macos.set_bundle_name("Named in a test")
        assert macos.bundle_name() == "Named in a test"
    finally:
        macos.set_bundle_name(before or "python")


def test_off_macos_the_name_is_left_alone(monkeypatch):
    """No Objective-C runtime to call, and nothing to call it for."""
    monkeypatch.setattr(macos.sys, "platform", "linux")

    assert macos.set_bundle_name("anything") is False
    assert macos.bundle_name() is None


def test_python_dash_m_runs_the_command_line():
    """What a launcher runs, needing only the environment's interpreter."""
    result = subprocess.run(
        [sys.executable, "-m", "am_segmentation_manager.cli", "--version"],
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("am-seg ")


def test_a_spawned_worker_does_not_run_the_command_again(monkeypatch):
    """*spawn* imports the main module again as ``__mp_main__`` in every
    worker; the scan's pool and the prediction child would each re-run the
    command that started them."""
    ran = []
    monkeypatch.setattr(cli_main, "main", lambda argv=None: ran.append(argv) or 0)

    runpy.run_module("am_segmentation_manager.cli.__main__", run_name="__mp_main__")

    assert ran == []


def test_the_command_line_runs_with_no_console(monkeypatch, tmp_path):
    """``pythonw`` sets ``sys.stdout`` to None, and a handler writing to it
    would fail on every message."""
    monkeypatch.setattr(sys, "stdout", None)

    cli_main.main(["list-projects", str(tmp_path)])

    assert not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logging.getLogger().handlers
    )
