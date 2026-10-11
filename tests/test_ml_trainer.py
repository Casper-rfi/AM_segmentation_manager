"""Tests for what AM-model-training is asked to do, without AM-model-training.

The trainer is replaced by a stand-in module, so these run in every environment.
What is pinned is the translation: am-seg's settings into AM-model-training's
arguments, always two lists, and a device it can train on.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from am_segmentation_manager.ml import trainer as training

SETTINGS = {
    "max_epochs": 50, "frozen_epochs": 0,
    "initial_learning_rate": 2e-6, "max_learning_rate": 3e-4,
    "training_batch_size": 6, "validation_batch_size": 1,
    "save_all": True, "seed": 7,
}


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:0", (False, 0)), ("cuda:3", (False, 3)), ("cuda", (False, 0)),
     ("cpu", (True, 0)), ("mps", (True, 0))],
)
def test_a_torch_device_becomes_what_am_model_training_is_told(device, expected):
    assert training.trainable_device(device) == expected


def test_the_arguments_always_name_a_training_and_a_validation_list():
    """One list would be split by AM-model-training at random, through lamellae."""
    arguments = training.training_arguments(
        Path("run.csv"), Path("val.csv"), Path("run"),
        device="cuda:0", base_weights=None, **SETTINGS,
    )

    assert arguments["csv"] == (Path("run.csv"), Path("val.csv"))
    assert "validation_split" not in arguments


def test_settings_reach_am_model_training_under_its_own_names():
    arguments = training.training_arguments(
        Path("run.csv"), Path("val.csv"), Path("run"),
        device="cuda:0", base_weights=Path("base.pth"), **SETTINGS,
    )

    assert arguments["output_path"] == Path("run")
    assert arguments["weights_path"] == Path("base.pth")
    assert (arguments["max_epochs"], arguments["frozen_epochs"]) == (50, 0)
    assert (arguments["initial_learning_rate"], arguments["max_learning_rate"]) == (2e-6, 3e-4)
    assert arguments["save_all_models"] is True
    assert arguments["seed"] == 7
    assert (arguments["cpu_only"], arguments["gpu_number"]) == (False, 0)
    assert arguments["log_mlflow"] is False


def _stand_in(monkeypatch, device: str) -> dict:
    """Replace AM-model-training and the device choice, returning what training was called with."""
    called: dict = {}
    main = types.ModuleType("am_model_training.main")
    main.train = lambda **arguments: called.update(arguments)
    package = types.ModuleType("am_model_training")
    package.main = main
    monkeypatch.setitem(sys.modules, "am_model_training", package)
    monkeypatch.setitem(sys.modules, "am_model_training.main", main)
    monkeypatch.setattr(training, "require_training_stack", lambda operation: None)
    monkeypatch.setattr(training, "resolve_device", lambda requested, **how: device)
    return called


def test_training_runs_on_the_gpu_the_environment_pinned(monkeypatch):
    called = _stand_in(monkeypatch, "cuda:0")

    used = training.train(Path("run.csv"), Path("val.csv"), Path("run"), device="cuda:2", **SETTINGS)

    assert used == "cuda:0"
    assert (called["cpu_only"], called["gpu_number"]) == (False, 0)


def test_training_does_not_let_auto_fall_back_to_a_cpu_nobody_named(monkeypatch):
    """Days on a CPU, where a GPU the torch build cannot use was expected."""
    _stand_in(monkeypatch, "cuda:0")
    asked: list[dict] = []
    monkeypatch.setattr(
        training, "resolve_device", lambda requested, **how: asked.append(how) or "cuda:0"
    )

    training.train(Path("run.csv"), Path("val.csv"), Path("run"), **SETTINGS)

    assert asked == [{"fall_back": False}]


def test_a_device_am_model_training_cannot_use_trains_on_the_cpu_and_says_so(monkeypatch, caplog):
    """Apple silicon's GPU is offered by torch and unknown to AM-model-training."""
    called = _stand_in(monkeypatch, "mps")

    with caplog.at_level("WARNING"):
        used = training.train(Path("run.csv"), Path("val.csv"), Path("run"), **SETTINGS)

    assert used == "cpu"
    assert called["cpu_only"] is True
    assert "not on mps" in caplog.text


def test_training_without_the_stack_says_what_to_install(monkeypatch):
    from am_segmentation_manager.ml import environment

    monkeypatch.setattr(
        environment, "training_stack_problems",
        lambda: [("am-model-training", "not installed")],
    )

    with pytest.raises(RuntimeError, match="--extra train"):
        training.train(Path("run.csv"), Path("val.csv"), Path("run"), **SETTINGS)


def test_the_device_is_told_as_nvidia_smi_numbers_it_before_training_begins(monkeypatch):
    """A pinned GPU is cuda:0 in its own process whichever card it is."""
    called = _stand_in(monkeypatch, "cuda:0")
    monkeypatch.setattr(training, "numbered_device", lambda device: "cuda:2")
    told = []

    used = training.train(
        Path("run.csv"), Path("val.csv"), Path("run"),
        device="auto", on_device=lambda device: told.append((device, dict(called))), **SETTINGS,
    )

    assert used == "cuda:2" and told == [("cuda:2", {})]
    assert (called["cpu_only"], called["gpu_number"]) == (False, 0)
