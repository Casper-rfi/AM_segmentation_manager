"""Tests for how the optional inference and training stacks are reported when absent.

Each is an extra of a gigabyte or more that many environments lack, so the case
that matters most is the one where it is not installed and everything else must
keep working anyway. These pin that the reporting is importable, honest about
which dependency failed, and that asking for a device, a preprocessing run or
training says what to install rather than raising ImportError from somewhere deep.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from am_segmentation_manager.ml import environment as ml_env
from am_segmentation_manager.ml.gpus import Gpu, GpuInventory
from conftest import break_import


def test_the_module_imports_without_torch():
    """The whole point of the boundary: asking the question costs no torch."""
    assert isinstance(ml_env.ml_stack_problems(), list)


def test_a_missing_package_is_reported_as_not_installed(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    problems = dict(ml_env.ml_stack_problems())

    assert "not installed" in problems["torch"]
    assert not ml_env.has_ml_support()


def test_a_broken_package_is_not_reported_as_missing(monkeypatch):
    """A torch whose native libraries will not load is installed, not absent."""
    break_import(monkeypatch, "torch", OSError("libcudart.so.12: cannot open shared object file"))

    problems = dict(ml_env.ml_stack_problems())

    assert "installed but failed to import" in problems["torch"]
    assert "not installed" not in problems["torch"]
    assert "libcudart" in problems["torch"]


def test_requiring_the_stack_names_the_operation_and_the_fix(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError) as raised:
        ml_env.require_ml_stack("Predicting")

    message = str(raised.value)
    assert message.startswith("Predicting needs")
    assert "torch" in message
    assert "--extra ml" in message


def test_listing_devices_without_torch_explains_rather_than_raising_importerror(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="--extra ml"):
        ml_env.available_devices()


def test_verifying_preprocessing_without_torch_explains_itself(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="--extra ml"):
        ml_env.verify_preprocessing()


def test_an_explicitly_named_device_needs_no_torch_to_resolve(monkeypatch):
    """Asking what is installed is only needed when something has to choose.
    A controller told 'cpu' should not import torch to agree."""
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    assert ml_env.resolve_device("cpu", GpuInventory(), {}) == "cpu"
    assert ml_env.resolve_device("cuda:1", GpuInventory(), {}) == "cuda:1"


def test_choosing_a_device_without_torch_explains_itself(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="--extra ml"):
        ml_env.resolve_device("auto", GpuInventory(), {})


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_auto_takes_the_most_capable_device():
    assert ml_env.resolve_device("auto", GpuInventory(), {}) == ml_env.available_devices()[0]


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_cpu_is_always_offered_and_offered_last():
    """CPU is the fallback, so it must be present whatever else is."""
    devices = ml_env.available_devices()

    assert devices[-1] == "cpu"


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_preprocessing_returns_a_batched_rgb_tensor_preserving_aspect_ratio():
    """A 256x384 image at image_size=512 keeps its 2:3 shape, not padded square."""
    assert ml_env.verify_preprocessing(device="cpu", image_size=512) == (1, 3, 341, 512)


def test_devices_are_not_discovered_at_the_cost_of_an_import(monkeypatch):
    """The window offers a device to predict on and must not spend seconds
    importing torch to build a dropdown -- the same reason ml_stack_present
    locates the modules rather than loading them."""
    import sys

    from am_segmentation_manager.ml.environment import discovered_devices

    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert discovered_devices() == ()


def test_devices_are_reported_once_torch_is_loaded(monkeypatch):
    """Pressing Predict loads it, so the box fills in from then on -- with
    whatever the machine has, which is several CUDA devices on a workstation
    and an MPS one on a Mac."""
    import sys
    import types

    from am_segmentation_manager.ml import environment

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(
        environment, "available_devices", lambda: ("cuda:0", "cuda:1", "cpu")
    )

    assert environment.discovered_devices() == ("cuda:0", "cuda:1", "cpu")


# ---------------------------------------------------------------------------
# Choosing a GPU on a shared workstation
# ---------------------------------------------------------------------------
#
# Every GPU here is described rather than present: no test machine has one.

def _gpu(index: int, free_gb: float, name: str = "NVIDIA A100 80GB PCIe") -> Gpu:
    return Gpu(
        index=index, physical_index=index, uuid=f"GPU-{index:04d}", name=name,
        free_mib=int(free_gb * 1024), total_mib=80 * 1024, pci_bus_id=f"0:{index}",
    )


FOUR = GpuInventory(
    gpus=(_gpu(0, 1), _gpu(1, 71), _gpu(2, 39), _gpu(3, 29)), driver_version="535.288.01"
)


def _torch_that_can_use_cuda(
    monkeypatch,
    available: bool = True,
    built: str | None = "12.6",
    architectures: tuple[str, ...] = ("sm_80", "sm_86", "sm_90"),
    capability: tuple[int, int] = (8, 0),
):
    """Put a stand-in torch in place, answering the questions device choice asks.

    *architectures* are the kernels the build carries and *capability* is the
    card in the machine, which is how a build too old for a card is staged.
    """
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.__version__ = "2.14.0"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        is_initialized=lambda: False,
        get_arch_list=lambda: list(architectures),
        device_count=lambda: 1,
        get_device_capability=lambda index: capability,
        get_device_name=lambda index: "NVIDIA GeForce RTX 5060 Ti",
    )
    torch.version = types.SimpleNamespace(cuda=built)
    threads: list[int] = []
    torch.set_num_threads = threads.append
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(ml_env, "require_ml_stack", lambda operation: None)
    return threads


def test_auto_runs_on_the_gpu_with_most_memory_free_and_no_other(monkeypatch):
    """cuda:0 is the GPU whoever sits nearest the machine uses first."""
    _torch_that_can_use_cuda(monkeypatch)
    environ: dict[str, str] = {}

    assert ml_env.resolve_device("auto", FOUR, environ) == "cuda:0"
    assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-0001"
    assert environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_a_gpu_named_by_number_is_pinned_and_is_cuda_0_in_the_run(monkeypatch):
    environ: dict[str, str] = {}

    assert ml_env.resolve_device("cuda:2", FOUR, environ) == "cuda:0"
    assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-0002"


def test_a_device_that_is_not_a_listed_gpu_is_passed_on_untouched(monkeypatch):
    def not_asked(**kwargs):
        raise AssertionError("nvidia-smi is asked only where a GPU might be meant")

    monkeypatch.setattr(ml_env, "gpu_inventory", not_asked)
    environ: dict[str, str] = {}

    assert ml_env.resolve_device("cpu", environ=environ) == "cpu"
    assert ml_env.resolve_device("cuda:7", FOUR, environ) == "cuda:7"
    assert "CUDA_VISIBLE_DEVICES" not in environ


def test_a_gpu_short_of_memory_is_warned_about_before_the_load(monkeypatch, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger=ml_env.__name__):
        ml_env.resolve_device("cuda:0", FOUR, {})

    assert "1.0 GB free" in caplog.text


def test_gpus_torch_cannot_use_are_said_so_and_auto_falls_back(monkeypatch, caplog):
    """A CPU run on a GPU workstation is an hour where it should be minutes, so
    it is announced rather than merely slow."""
    import logging

    _torch_that_can_use_cuda(monkeypatch, available=False)
    monkeypatch.setattr(ml_env, "available_devices", lambda: ("cpu",))

    with caplog.at_level(logging.WARNING, logger=ml_env.__name__):
        assert ml_env.resolve_device("auto", FOUR, {}) == "cpu"

    assert "built for CUDA 12.6" in caplog.text
    assert "535.288.01" in caplog.text


def test_a_card_newer_than_every_kernel_is_said_so_and_auto_falls_back(monkeypatch, caplog):
    """A driver new enough to start CUDA is not a build that can run on the card:
    the kernels are missing, and the first launch is what would say so."""
    import logging

    _torch_that_can_use_cuda(monkeypatch, capability=(12, 0))
    monkeypatch.setattr(ml_env, "available_devices", lambda: ("cpu",))

    with caplog.at_level(logging.WARNING, logger=ml_env.__name__):
        assert ml_env.resolve_device("auto", FOUR, {}) == "cpu"

    assert "sm_120" in caplog.text
    assert "sm_90" in caplog.text, "and what it does carry, or there is nothing to act on"
    assert "ml-cuda13" in caplog.text, "and the way out of it"


def test_a_card_the_build_carries_kernels_for_is_not_reported(monkeypatch):
    """A minor version below the card's runs on it, or every check would be a warning."""
    _torch_that_can_use_cuda(monkeypatch, architectures=("sm_86",), capability=(8, 9))

    assert ml_env.cuda_problem(FOUR) == ""


def test_a_build_naming_no_architectures_is_not_read_as_a_problem(monkeypatch):
    """Absent evidence is not evidence: refusing a GPU here would take a working
    one away over a question torch did not answer."""
    _torch_that_can_use_cuda(monkeypatch, architectures=(), capability=(12, 0))

    assert ml_env.cuda_problem(FOUR) == ""


def test_a_torch_built_without_cuda_is_named_as_the_reason(monkeypatch):
    _torch_that_can_use_cuda(monkeypatch, available=False, built=None)

    problem = ml_env.cuda_problem(FOUR)

    assert "built without CUDA" in problem
    assert "uv sync" in problem


def test_there_is_no_problem_to_report_without_gpus_or_with_usable_ones(monkeypatch):
    _torch_that_can_use_cuda(monkeypatch)

    assert ml_env.cuda_problem(GpuInventory()) == ""
    assert ml_env.cuda_problem(FOUR) == ""


def test_a_gpu_is_not_pinned_once_cuda_has_started_in_the_process(monkeypatch):
    """CUDA reads the visible GPUs once; changing them afterwards would claim a
    pinning that did not happen."""
    import sys

    _torch_that_can_use_cuda(monkeypatch)
    sys.modules["torch"].cuda.is_initialized = lambda: True
    environ: dict[str, str] = {}

    assert ml_env.resolve_device("cuda:2", FOUR, environ) == "cuda:2"
    assert "CUDA_VISIBLE_DEVICES" not in environ


def test_prediction_uses_as_many_cpu_threads_as_am_seg_may(monkeypatch):
    """Preprocessing runs on the CPU whatever the device, and torch would
    otherwise take every core of a shared machine."""
    threads = _torch_that_can_use_cuda(monkeypatch)
    monkeypatch.setenv("AM_SEG_WORKERS", "3")

    assert ml_env.limit_cpu_threads() == 3
    assert threads == [3]


# -- training -------------------------------------------------------------------


def test_training_is_not_offered_without_the_inference_stack(monkeypatch):
    """AM-model-training brings no torch of its own; the inference extra chooses it."""
    monkeypatch.setattr(ml_env, "ml_stack_present", lambda: False)

    assert not ml_env.training_stack_present()


def test_training_is_offered_only_where_am_model_training_is_found(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(ml_env, "ml_stack_present", lambda: True)
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *args: None if name == "am_model_training" else real(name, *args),
    )

    assert not ml_env.training_stack_present()


def test_a_missing_inference_stack_is_the_training_stack_s_problem(monkeypatch):
    """Reported as torch, the thing to install first, not as a trainer that would not load."""
    monkeypatch.setattr(ml_env, "ml_stack_problems", lambda: [("torch", "not installed")])

    assert ml_env.training_stack_problems() == [("torch", "not installed")]


def test_a_missing_trainer_is_reported_as_not_installed(monkeypatch):
    monkeypatch.setattr(ml_env, "ml_stack_problems", lambda: [])
    break_import(monkeypatch, "am_model_training.main",
                 ImportError("No module named 'am_model_training'"))

    problems = dict(ml_env.training_stack_problems())

    assert "not installed" in problems["am-model-training"]
    assert not ml_env.has_training_support()


def test_a_trainer_whose_dependencies_fail_is_installed_not_missing(monkeypatch):
    """Importing it loads MONAI and mlflow; one of those failing is a broken install."""
    monkeypatch.setattr(ml_env, "ml_stack_problems", lambda: [])
    break_import(monkeypatch, "am_model_training.main", AttributeError("mlflow has no attribute"))

    problems = dict(ml_env.training_stack_problems())

    assert "installed but failed to import" in problems["am-model-training"]


def test_mlflow_is_quietened_before_the_trainer_is_imported(monkeypatch):
    import os

    variable, quiet = ml_env.QUIET_MLFLOW
    seen: list[str | None] = []

    def importing(modules):
        seen.append(os.environ.get(variable))
        return []

    monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(ml_env, "ml_stack_problems", lambda: [])
    monkeypatch.setattr(ml_env, "_import_problems", importing)

    ml_env.training_stack_problems()

    assert seen == [quiet]


def test_requiring_the_training_stack_names_the_operation_and_the_fix(monkeypatch):
    monkeypatch.setattr(ml_env, "ml_stack_problems", lambda: [])
    break_import(monkeypatch, "am_model_training.main",
                 ImportError("No module named 'am_model_training'"))

    with pytest.raises(RuntimeError) as raised:
        ml_env.require_training_stack("Training")

    message = str(raised.value)
    assert message.startswith("Training needs am-model-training")
    assert "--extra train" in message


@pytest.mark.skipif(not ml_env.training_stack_present(), reason="train extra not installed")
def test_the_installed_training_stack_imports():
    """Where the extra is installed, AM-model-training loads beside this torch."""
    assert ml_env.training_stack_problems() == []


def test_a_pinned_gpu_is_named_by_its_number_in_nvidia_smi():
    pinned = GpuInventory(gpus=(replace(FOUR.gpus[2], index=0),))

    assert ml_env.numbered_device("cuda:0", pinned) == "cuda:2"
    assert ml_env.numbered_device("cuda:0", GpuInventory()) == "cuda:0"
    assert ml_env.numbered_device("cpu", pinned) == "cpu"
