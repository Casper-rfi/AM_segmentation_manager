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


# -- what each validation image scored, told as the run validates -------------


def _validation(torch, classes: int = 3, images: int = 4):
    """What AM-model-training's validate is given, for a model that returns its input."""
    from am_model_training.metrics import Metrics
    from am_model_training.utils import MONAI_KEYS

    class Loss(torch.nn.Module):
        def forward(self, outputs, labels):
            return outputs.float().abs().mean()

    generator = torch.Generator().manual_seed(3)
    batches = [
        {
            MONAI_KEYS.IMAGE: torch.rand((1, classes, 8, 8), generator=generator),
            MONAI_KEYS.LABEL: torch.randint(0, classes, (1, 1, 8, 8), generator=generator),
        }
        for _ in range(images)
    ]
    device = torch.device("cpu")
    objects = types.SimpleNamespace(
        model=torch.nn.Identity(), device=device, loss_function=Loss(),
        validation=types.SimpleNamespace(
            metrics=Metrics(device=device, num_classes=classes), dataloader=batches,
            inferer=lambda images, model: model(images),
            post_transform=lambda output: output.argmax(0, keepdim=True),
        ),
    )
    parameters = types.SimpleNamespace(
        total_validation_data=images, validation_batch_size=1, loss_weights=None,
        key_val_metrics=["mean_iou"], num_classes=classes,
        update_metrics=lambda epoch, metrics, stage: False,
    )
    return objects, parameters, batches


def test_each_image_s_matrix_is_told_and_adds_up_to_what_am_model_training_counted():
    """Through AM-model-training's own validate, which is what the run calls each epoch."""
    torch = pytest.importorskip("torch")
    validate = pytest.importorskip("am_model_training.run.validate").validate
    from am_model_training.utils import MONAI_KEYS

    objects, parameters, batches = _validation(torch)
    told: list = []

    metrics, best = training.recording_validate(validate, lambda *said: told.append(said))(
        objects, parameters, epoch=7, log_mlflow=False
    )

    (epoch, matrices, losses, total), = told
    assert epoch == 7 and best is False and len(matrices) == len(losses) == 4
    assert total == metrics.confusion_matrix.tolist()
    assert [[sum(cells) for cells in zip(*rows)] for rows in zip(*matrices)] == total
    for batch, matrix in zip(batches, matrices):
        label, predicted = batch[MONAI_KEYS.LABEL].flatten(), batch[MONAI_KEYS.IMAGE].argmax(1).flatten()
        assert matrix == [
            [int(((label == true) & (predicted == said)).sum()) for said in range(3)]
            for true in range(3)
        ]
    assert losses == pytest.approx([float(batch[MONAI_KEYS.IMAGE].mean()) for batch in batches])
    assert "update" not in vars(objects.validation.metrics)


def test_a_telling_that_fails_leaves_the_validation_as_it_was(caplog):
    torch = pytest.importorskip("torch")
    validate = pytest.importorskip("am_model_training.run.validate").validate
    objects, parameters, _ = _validation(torch)

    def refuse(*said):
        raise OSError("disk full")

    recorded = training.recording_validate(validate, refuse)(objects, parameters, epoch=1)
    plain = validate(objects, parameters, epoch=1)

    assert recorded[0].confusion_matrix.tolist() == plain[0].confusion_matrix.tolist()
    assert "was not kept" in caplog.text


def test_training_validates_through_the_recording_and_puts_back_what_it_stood_in_for(monkeypatch):
    called = _stand_in(monkeypatch, "cpu")
    loop = types.ModuleType("am_model_training.run.train")
    loop.validate = original = lambda *given, **named: "validated"
    run = types.ModuleType("am_model_training.run")
    run.train = loop
    monkeypatch.setitem(sys.modules, "am_model_training.run", run)
    monkeypatch.setitem(sys.modules, "am_model_training.run.train", loop)
    during: list = []
    sys.modules["am_model_training.main"].train = lambda **arguments: (
        called.update(arguments), during.append(loop.validate)
    )

    training.train(Path("t.csv"), Path("v.csv"), Path("out"), on_validated=lambda *said: None, **SETTINGS)

    assert during[0] is not original and loop.validate is original
    assert called["csv"] == (Path("t.csv"), Path("v.csv"))
