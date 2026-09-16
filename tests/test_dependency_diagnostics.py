"""Tests for how a missing or broken optional dependency is reported.

These exist because of a real incident: a fresh conda environment reported
"numpy, tifffile and opencv-python are not installed" when the packages were
installed and cv2 was failing to import -- two installations of it, one from
conda's `opencv` and one from pip's `opencv-python`. The message sent the user
to install them again, which is the one action that makes it worse.

The rule these pin: say which dependency failed, and distinguish "not
installed" from "installed but will not import".
"""
from __future__ import annotations

import builtins

import pytest

from am_segmentation_manager.io import image as image_io
from conftest import break_import


def test_no_problems_when_everything_imports():
    pytest.importorskip("cv2")

    assert image_io.image_stack_problems() == []
    assert image_io.has_image_stack_support()


def test_a_missing_package_is_reported_as_not_installed(monkeypatch):
    break_import(monkeypatch, "cv2", ImportError("No module named 'cv2'"))

    problems = dict(image_io.image_stack_problems())

    assert "opencv" in problems
    assert "not installed" in problems["opencv"]


def test_a_broken_package_is_not_reported_as_missing(monkeypatch):
    """The incident case. cv2 was present and raising, and the old code caught
    every exception and called it "not installed"."""
    break_import(monkeypatch, "cv2", RuntimeError("dylib load failed"))

    problems = dict(image_io.image_stack_problems())

    assert "installed but failed to import" in problems["opencv"]
    assert "not installed" not in problems["opencv"]
    assert "dylib load failed" in problems["opencv"]


def test_the_error_names_what_failed_and_why(monkeypatch):
    break_import(monkeypatch, "tifffile", RuntimeError("boom"))

    with pytest.raises(RuntimeError) as raised:
        image_io.require_image_stack("Reading an image")

    message = str(raised.value)
    assert "Reading an image" in message
    assert "tifffile" in message
    assert "boom" in message


def test_the_hint_warns_against_a_second_opencv():
    """The fix for the incident is not "install it again", so the hint must not
    read that way where a second provider of cv2 is the cause."""
    assert "conda's opencv" in image_io.INSTALL_HINT
    assert "opencv-python-headless" in image_io.INSTALL_HINT


def test_every_problem_is_reported_not_just_the_first(monkeypatch):
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name in {"numpy", "cv2", "tifffile"}:
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)

    assert len(image_io.image_stack_problems()) == 3
