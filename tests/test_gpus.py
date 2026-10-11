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


# -- what training takes of a GPU ------------------------------------------------


def _card(name: str, total_gb: float, free_gb: float | None = None, index: int = 0) -> gpus.Gpu:
    free = total_gb if free_gb is None else free_gb
    return gpus.Gpu(index, index, f"GPU-{index}", name, round(free * 1024), round(total_gb * 1024), "0")


def _said(batch: int, device: str, *cards: gpus.Gpu) -> str:
    return " ".join(gpus.training_memory_cautions(batch, device, gpus.GpuInventory(gpus=cards)))


def test_the_estimate_is_what_was_measured_at_a_batch_of_one_and_of_six():
    assert gpus.estimated_training_mib(1) / 1024 == pytest.approx(8.5, abs=0.1)
    assert gpus.estimated_training_mib(6) / 1024 == pytest.approx(43.0, abs=0.1)
    needed = [gpus.estimated_training_mib(batch) for batch in range(1, 9)]
    assert needed == sorted(needed)


@pytest.mark.parametrize(
    ("memory_gb", "batch"), [(8, 0), (12, 1), (16, 1), (24, 3), (32, 4), (48, 6), (80, 10)]
)
def test_the_largest_batch_for_a_card_is_the_largest_whose_estimate_leaves_headroom(memory_gb, batch):
    assert gpus.largest_training_batch(memory_gb * 1024) == batch
    if batch:
        assert gpus.estimated_training_mib(batch) <= memory_gb * 1024 * gpus.TRAINING_HEADROOM
    assert gpus.estimated_training_mib(batch + 1) > memory_gb * 1024 * gpus.TRAINING_HEADROOM


def test_a_batch_a_card_cannot_hold_is_said_with_the_batch_that_would_fit():
    said = _said(6, "cuda:0", _card("NVIDIA GeForce RTX 5060 Ti", 16))

    assert "Batches of 6 need about 43 GB" in said
    assert "cuda:0 (NVIDIA GeForce RTX 5060 Ti) has 16 GB" in said
    assert "largest batch likely to fit is 1" in said


def test_a_card_too_small_for_one_image_is_not_told_to_go_lower():
    said = _said(1, "cuda:0", _card("NVIDIA GeForce RTX 4060", 8))

    assert "Even a batch of 1 needs about 8.5 GB" in said and "largest batch" not in said


def test_a_batch_that_fits_a_card_nothing_else_is_using_is_given_no_caution():
    assert _said(6, "cuda:0", _card("NVIDIA A100 80GB PCIe", 80)) == ""
    assert _said(1, "cuda:0", _card("NVIDIA GeForce RTX 5060 Ti", 16, free_gb=15)) == ""


def test_a_card_big_enough_but_in_use_says_how_much_is_free_now():
    said = _said(6, "cuda:0", _card("NVIDIA A100 80GB PCIe", 80, free_gb=30))

    assert "30.0 GB free of 80 now" in said and "probably fail" not in said


def test_auto_is_judged_against_the_gpu_it_would_take():
    small_but_free = _card("NVIDIA GeForce RTX 5060 Ti", 16, index=0)
    large_but_full = _card("NVIDIA A100 80GB PCIe", 80, free_gb=4, index=1)

    assert "cuda:0 (NVIDIA GeForce RTX 5060 Ti)" in _said(6, "auto", small_but_free, large_but_full)


def test_a_machine_with_no_gpu_or_asked_for_the_cpu_is_told_how_slow_that_is():
    assert "No NVIDIA GPU was found" in _said(6, "auto")
    assert _said(6, "cpu", _card("NVIDIA A100 80GB PCIe", 80)).startswith("Training on the CPU")


def test_a_gpu_named_that_cannot_be_described_is_taken_on_trust():
    assert _said(6, "cuda:3") == ""
