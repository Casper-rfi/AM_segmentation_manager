"""Tests for reading NVIDIA GPUs from nvidia-smi and pinning a run to one.

No test here needs a GPU: nvidia-smi's answer is recorded, in the form a
workstation of four A100s gives it.
"""
from __future__ import annotations

import logging
import subprocess

import pytest

from am_segmentation_manager.ml import gpus
from am_segmentation_manager.ml.gpus import (
    GpuInventory,
    gpu_inventory,
    least_used,
    number_as_nvidia_smi,
    pin_to,
)

FOUR_A100S = """\
0, GPU-aaaa0000-0000-0000-0000-000000000000, NVIDIA A100 80GB PCIe, 1024, 81920, 00000000:17:00.0, 535.288.01
1, GPU-bbbb1111-1111-1111-1111-111111111111, NVIDIA A100 80GB PCIe, 72960, 81920, 00000000:31:00.0, 535.288.01
2, GPU-cccc2222-2222-2222-2222-222222222222, NVIDIA A100-PCIE-40GB, 40000, 40960, 00000000:B1:00.0, 535.288.01
3, GPU-dddd3333-3333-3333-3333-333333333333, NVIDIA A100-PCIE-40GB, 30000, 40960, 00000000:CA:00.0, 535.288.01
"""


def answering(stdout: str = FOUR_A100S, returncode: int = 0):
    """Stand in for subprocess.run, answering as nvidia-smi would."""
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")
    return run


def test_every_gpu_is_read_with_its_model_memory_and_driver():
    inventory = gpu_inventory(answering(), environ={})

    assert [gpu.device for gpu in inventory.gpus] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert inventory.gpus[1].name == "NVIDIA A100 80GB PCIe"
    assert inventory.gpus[1].free_mib == 72960
    assert inventory.driver_version == "535.288.01"


def test_a_gpu_is_described_for_a_person_choosing_one():
    gpu = gpu_inventory(answering(), environ={}).gpus[1]

    assert gpu.describe() == "cuda:1 · NVIDIA A100 80GB PCIe · 71.2 of 80 GB free"


def test_a_machine_without_nvidia_smi_has_no_gpus_to_describe():
    def missing(command, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert gpu_inventory(missing, environ={}) == GpuInventory()


def test_nvidia_smi_failing_is_no_gpus_rather_than_an_error():
    """What a driver that has lost its GPU answers, among other things."""
    assert gpu_inventory(answering("", returncode=9), environ={}) == GpuInventory()


def test_an_answer_not_understood_is_warned_about_and_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger=gpus.__name__):
        inventory = gpu_inventory(answering("[Not Supported], a, b\n"), environ={})

    assert inventory == GpuInventory()
    assert "not understood" in caplog.text


# --- what CUDA_VISIBLE_DEVICES leaves ---------------------------------------

def test_gpus_a_scheduler_hides_are_not_offered_and_the_rest_renumber():
    """CUDA numbers what is visible from zero, so a job granted GPUs 2 and 3
    sees them as cuda:0 and cuda:1."""
    inventory = gpu_inventory(answering(), environ={"CUDA_VISIBLE_DEVICES": "2,3"})

    assert [(gpu.device, gpu.physical_index) for gpu in inventory.gpus] == [
        ("cuda:0", 2), ("cuda:1", 3),
    ]


def test_a_gpu_can_be_made_visible_by_its_uuid_or_a_prefix_of_it():
    inventory = gpu_inventory(answering(), environ={"CUDA_VISIBLE_DEVICES": "GPU-cccc"})

    assert [gpu.physical_index for gpu in inventory.gpus] == [2]


def test_the_visible_list_ends_where_cuda_ends_it():
    """An entry naming no GPU ends the list; what follows is ignored."""
    for restriction, shown in (("1,7,2", [1]), ("", []), ("-1", []), ("3,3", [3])):
        inventory = gpu_inventory(answering(), environ={"CUDA_VISIBLE_DEVICES": restriction})
        assert [gpu.physical_index for gpu in inventory.gpus] == shown, restriction


def test_a_restriction_not_understood_describes_no_gpus():
    """A MIG slice cannot be placed among whole GPUs, and a guess would pin the
    wrong one."""
    environ = {"CUDA_VISIBLE_DEVICES": "MIG-0123abcd"}

    assert gpu_inventory(answering(), environ=environ) == GpuInventory()


# --- choosing and pinning ---------------------------------------------------

def test_the_least_used_gpu_is_the_one_with_most_memory_free():
    assert least_used(gpu_inventory(answering(), environ={})).device == "cuda:1"


def test_equally_free_gpus_are_chosen_by_the_lower_number():
    equal = FOUR_A100S.replace(" 1024,", " 72960,")

    assert least_used(gpu_inventory(answering(equal), environ={})).device == "cuda:0"


def test_no_gpus_means_none_is_chosen():
    assert least_used(GpuInventory()) is None


def test_gpus_are_numbered_as_nvidia_smi_numbers_them_unless_told_otherwise():
    environ: dict[str, str] = {}
    number_as_nvidia_smi(environ)
    assert environ == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    environ = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    number_as_nvidia_smi(environ)
    assert environ == {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}


def test_a_pinned_gpu_is_the_only_one_visible_and_is_cuda_0_there():
    """By UUID, which names the same card whatever order is in force."""
    gpu = gpu_inventory(answering(), environ={}).gpus[2]
    environ: dict[str, str] = {}

    assert pin_to(gpu, environ) == "cuda:0"
    assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-cccc2222-2222-2222-2222-222222222222"


@pytest.mark.parametrize("device, found", [("cuda:3", 3), ("cuda", 0), ("cuda:9", None), ("cpu", None)])
def test_a_device_is_matched_to_the_gpu_it_names(device, found):
    gpu = gpu_inventory(answering(), environ={}).by_device(device)

    assert (gpu.index if gpu else None) == found
