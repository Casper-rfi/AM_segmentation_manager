"""Tests for how the optional inference stack is reported when it is absent.

The stack is several gigabytes and lives in its own environment, so the case
that matters most is the one every developer machine is in: torch is not
installed, and everything that is not inference must keep working anyway. These
pin that the reporting is importable, honest about which dependency failed, and
that asking for a device or a preprocessing run says what to install rather
than raising ImportError from somewhere deep.
"""
from __future__ import annotations

import pytest

from am_segmentation_manager.ml import environment as ml_env
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
    assert "setup_ml.sh" in message


def test_listing_devices_without_torch_explains_rather_than_raising_importerror(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="setup_ml.sh"):
        ml_env.available_devices()


def test_verifying_preprocessing_without_torch_explains_itself(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="setup_ml.sh"):
        ml_env.verify_preprocessing()


def test_an_explicitly_named_device_needs_no_torch_to_resolve(monkeypatch):
    """Asking what is installed is only needed when something has to choose.
    A controller told 'cpu' should not import torch to agree."""
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    assert ml_env.resolve_device("cpu") == "cpu"
    assert ml_env.resolve_device("cuda:1") == "cuda:1"


def test_choosing_a_device_without_torch_explains_itself(monkeypatch):
    break_import(monkeypatch, "torch", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="setup_ml.sh"):
        ml_env.resolve_device("auto")


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_auto_takes_the_most_capable_device():
    assert ml_env.resolve_device("auto") == ml_env.available_devices()[0]


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_cpu_is_always_offered_and_offered_last():
    """CPU is the fallback, so it must be present whatever else is."""
    devices = ml_env.available_devices()

    assert devices[-1] == "cpu"


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_preprocessing_returns_a_batched_rgb_tensor_preserving_aspect_ratio():
    """A 256x384 image at image_size=512 keeps its 2:3 shape, not padded square."""
    assert ml_env.verify_preprocessing(device="cpu", image_size=512) == (1, 3, 341, 512)
