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


def test_gpus_torch_cannot_use_fail_the_check(monkeypatch, capsys):
    """A CPU-only torch on a GPU workstation installs cleanly and then predicts
    at a tenth of the speed; the check is where that should surface."""
    import sys
    import types

    from am_segmentation_manager.ml import environment, gpus

    gpu = gpus.Gpu(0, 0, "GPU-0", "NVIDIA A100 80GB PCIe", 70 * 1024, 80 * 1024, "0:0")
    torch = types.ModuleType("torch")
    torch.__version__ = "2.14.0+cpu"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(environment, "ml_stack_present", lambda: True)
    monkeypatch.setattr(environment, "ml_stack_problems", lambda: [])
    monkeypatch.setattr(environment, "available_devices", lambda: ("cpu",))
    monkeypatch.setattr(environment, "verify_preprocessing", lambda device: (1, 3, 341, 512))
    monkeypatch.setattr(gpus, "gpu_inventory", lambda: gpus.GpuInventory(gpus=(gpu,)))
    monkeypatch.setattr(environment, "cuda_problem", lambda inventory: "built without CUDA")

    assert check_environment.check_inference() is False
    printed = capsys.readouterr().out
    assert "cuda:0 · NVIDIA A100 80GB PCIe" in printed
    assert "NOT USABLE: built without CUDA" in printed
