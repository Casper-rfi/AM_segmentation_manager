"""Tests for ``scripts/check_environment.py``: what it counts as a failure."""
from __future__ import annotations

import check_environment


def test_an_extra_that_was_not_installed_is_not_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(check_environment, "check_image_stack", lambda: True)
    monkeypatch.setattr(check_environment, "check_window", lambda: None)
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_present", lambda: False
    )

    assert check_environment.main() == 0
    assert "predicting needs the ml extra" in capsys.readouterr().out


def test_a_broken_image_stack_fails_the_check(monkeypatch, capsys):
    monkeypatch.setattr(
        "am_segmentation_manager.io.image.image_stack_problems",
        lambda: [("opencv", "installed but failed to import: boom")],
    )
    monkeypatch.setattr(check_environment, "check_window", lambda: None)
    monkeypatch.setattr(check_environment, "check_inference", lambda: True)

    assert check_environment.main() == 1
    assert "boom" in capsys.readouterr().out


def test_an_installed_inference_stack_that_does_not_load_fails_the_check(monkeypatch):
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_present", lambda: True
    )
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_problems",
        lambda: [("torch", "installed but failed to import: libcudart")],
    )

    assert check_environment.check_inference() is False
