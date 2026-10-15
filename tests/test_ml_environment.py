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
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
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
    _torch_that_can_use_cuda(monkeypatch)
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

    _torch_that_can_use_cuda(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=ml_env.__name__):
        ml_env.resolve_device("cuda:0", FOUR, {})

    assert "1.0 GB free" in caplog.text


def test_gpus_torch_cannot_use_are_said_so_and_auto_falls_back(monkeypatch, caplog):
    """A CPU run on a GPU workstation is an hour where it should be minutes, so
    it is announced rather than merely slow."""
    import logging

    _torch_that_can_use_cuda(monkeypatch, available=False)

    with caplog.at_level(logging.WARNING, logger=ml_env.__name__):
        assert ml_env.resolve_device("auto", FOUR, {}) == "cpu"

    assert "built for CUDA 12.6" in caplog.text
    assert "535.288.01" in caplog.text


def test_a_card_newer_than_every_kernel_is_said_so_and_auto_falls_back(monkeypatch, caplog):
    """A driver new enough to start CUDA is not a build that can run on the card:
    the kernels are missing, and the first launch is what would say so."""
    import logging

    _torch_that_can_use_cuda(monkeypatch, capability=(12, 0))
    assert ml_env.available_devices() == ("cuda:0", "cpu"), "CUDA starts on such a card"

    with caplog.at_level(logging.WARNING, logger=ml_env.__name__):
        assert ml_env.resolve_device("auto", FOUR, {}) == "cpu"

    assert "sm_120" in caplog.text
    assert "sm_90" in caplog.text, "and what it does carry, or there is nothing to act on"
    assert "ml-cuda13" in caplog.text, "and the way out of it"


@pytest.mark.parametrize(
    ("torch", "reason"),
    [({"capability": (12, 0)}, "sm_120"), ({"available": False}, "built for CUDA 12.6")],
)
def test_a_named_gpu_torch_cannot_use_is_refused_before_anything_loads(
    monkeypatch, torch: dict, reason: str
):
    """The first launch would say so too, one step into the run and as a traceback."""
    _torch_that_can_use_cuda(monkeypatch, **torch)

    with pytest.raises(RuntimeError, match="cuda:2 cannot be used") as refused:
        ml_env.resolve_device("cuda:2", FOUR, {})

    assert reason in str(refused.value)


def test_work_too_long_for_a_cpu_is_refused_where_auto_would_fall_back(monkeypatch):
    """A training run on the CPU is days, and nobody asked for one."""
    _torch_that_can_use_cuda(monkeypatch, capability=(12, 0))

    with pytest.raises(RuntimeError, match="sm_120") as refused:
        ml_env.resolve_device("auto", FOUR, {}, fall_back=False)

    assert "name cpu" in str(refused.value), "and the way to have it regardless"
    assert ml_env.resolve_device("cpu", FOUR, {}, fall_back=False) == "cpu"


def test_refusing_to_fall_back_changes_nothing_where_the_gpu_can_be_used(monkeypatch):
    _torch_that_can_use_cuda(monkeypatch)

    assert ml_env.resolve_device("auto", FOUR, {}, fall_back=False) == "cuda:0"


@pytest.mark.parametrize(
    ("record", "said", "unsaid"),
    [
        ({"dir_info": {"editable": True}}, "uv sync --extra gui --extra ml-cuda13", "uv tool"),
        ({"vcs_info": {"vcs": "git"}}, "[gui,ml-cuda13] and --upgrade", "uv sync"),
        ({}, "uv sync --extra gui --extra ml-cuda13 in a checkout, or the uv tool", ""),
    ],
)
def test_the_way_to_install_a_build_is_the_one_for_this_copy(record, said, unsaid):
    """`uv sync` does nothing for a copy installed with `uv tool install`."""
    command = ml_env.how_to_install("ml-cuda13", record)

    assert said in command
    assert not unsaid or unsaid not in command


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
