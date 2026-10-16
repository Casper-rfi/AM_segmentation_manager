"""Tests for ``am-seg check-environment``: what it reports and counts as a failure."""
from __future__ import annotations

from am_segmentation_manager import __version__
from am_segmentation_manager.cli import environment_check as check_environment
from am_segmentation_manager.cli.main import main


def test_an_extra_that_was_not_installed_is_not_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(check_environment, "check_image_stack", lambda: True)
    monkeypatch.setattr(check_environment, "check_window", lambda: None)
    monkeypatch.setattr(
        "am_segmentation_manager.ml.environment.ml_stack_present", lambda: False
    )

    assert check_environment.run_checks() == 0
    assert "predicting needs the ml extra" in capsys.readouterr().out


def test_a_broken_image_stack_fails_the_command(monkeypatch, capsys):
    monkeypatch.setattr(
        "am_segmentation_manager.io.image.image_stack_problems",
        lambda: [("opencv", "installed but failed to import: boom")],
    )
    monkeypatch.setattr(check_environment, "check_window", lambda: None)
    monkeypatch.setattr(check_environment, "check_inference", lambda: True)
    monkeypatch.setattr(check_environment, "check_training", lambda: True)

    assert main(["check-environment"]) == 1
    assert "boom" in capsys.readouterr().out


def test_the_report_opens_with_the_version_and_how_it_was_installed(monkeypatch):
    monkeypatch.setattr(
        check_environment, "version_date", lambda record: "Latest change: 2026-10-16"
    )
    checkout = {"url": "file:///src/am-seg", "dir_info": {"editable": True}}

    line = check_environment.installation_line(checkout)

    assert line == f"am-seg {__version__} · source checkout · Latest change: 2026-10-16"


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


QT_MISSING_CURSOR = """\
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.
"""


def _child(returncode: int, stderr: str = ""):
    import subprocess

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    return run


def test_a_qt_that_opens_a_window_is_reported_ok(capsys):
    check_environment.check_qt_window(_child(0), "linux", {"DISPLAY": ":0"})

    assert "Qt window     OK" in capsys.readouterr().out


def test_a_missing_system_library_is_named_with_the_command_that_installs_it(capsys):
    """Qt imports without it, so only opening a window finds it missing."""
    check_environment.check_qt_window(_child(-6, QT_MISSING_CURSOR), "linux", {"DISPLAY": ":0"})

    printed = capsys.readouterr().out
    assert "CANNOT OPEN: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed" in printed
    assert "sudo apt install libxcb-cursor0" in printed


def test_a_failure_naming_no_library_says_how_to_find_it(capsys):
    check_environment.check_qt_window(_child(1, "no platform plugin\n"), "win32", {})

    printed = capsys.readouterr().out
    assert "CANNOT OPEN: no platform plugin" in printed
    assert "QT_DEBUG_PLUGINS=1" in printed


def test_linux_without_a_display_starts_no_child(capsys):
    def run(command, **kwargs):
        raise AssertionError("no child should start without a display")

    check_environment.check_qt_window(run, "linux", {})

    assert "not checked: no display" in capsys.readouterr().out


def test_a_child_that_hangs_is_reported_rather_than_waited_on(capsys):
    import subprocess

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    check_environment.check_qt_window(run, "linux", {"WAYLAND_DISPLAY": "wayland-0"})

    assert "did not answer within" in capsys.readouterr().out


def test_a_training_stack_not_installed_is_not_a_failure(monkeypatch, capsys):
    from am_segmentation_manager.ml import environment

    monkeypatch.setattr(environment, "training_stack_present", lambda: False)

    assert check_environment.check_training() is True
    assert "training a model needs the train extra" in capsys.readouterr().out


def test_an_installed_training_stack_that_does_not_load_fails_the_check(monkeypatch, capsys):
    from am_segmentation_manager.ml import environment

    monkeypatch.setattr(environment, "training_stack_present", lambda: True)
    monkeypatch.setattr(
        environment, "training_stack_problems",
        lambda: [("am-model-training", "installed but failed to import: no mlflow")],
    )

    assert check_environment.check_training() is False
    assert "no mlflow" in capsys.readouterr().out


def test_a_broken_training_stack_fails_the_whole_check(monkeypatch):
    monkeypatch.setattr(check_environment, "check_image_stack", lambda: True)
    monkeypatch.setattr(check_environment, "check_window", lambda: None)
    monkeypatch.setattr(check_environment, "check_inference", lambda: True)
    monkeypatch.setattr(check_environment, "check_training", lambda: False)

    assert check_environment.run_checks() == 1
