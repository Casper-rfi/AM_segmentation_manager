"""Which NVIDIA GPUs this process may use, how full they are, and pinning a run to one.

Read from ``nvidia-smi``, which every NVIDIA driver installs: it answers without
torch and without opening a CUDA context on anyone's GPU, so the window may ask
each time its device box opens. **``cuda:N`` means the GPU ``nvidia-smi`` numbers
N**, among those ``CUDA_VISIBLE_DEVICES`` leaves visible, because
:func:`number_as_nvidia_smi` has CUDA count in the same order. Where
``nvidia-smi`` is absent or cannot answer, there are no GPUs to describe and the
caller behaves as it did without this module. Nothing here imports torch.
"""
from __future__ import annotations

import csv
import logging
import os
import subprocess
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "Gpu",
    "GpuInventory",
    "LOW_MEMORY_MIB",
    "gpu_inventory",
    "estimated_training_mib",
    "largest_training_batch",
    "least_used",
    "number_as_nvidia_smi",
    "pin_to",
    "training_memory_cautions",
]

#: Free memory below which a GPU is warned about before a model is loaded onto
#: it, rather than failing out of memory partway through a run.
LOW_MEMORY_MIB = 4 * 1024

#: What training generation 1 takes of a GPU, as ``nvidia-smi`` reported it: a
#: part whatever the batch, and a part for each image in it. Fitted to two runs,
#: a batch of 1 at about 8.5 GB (RTX 5060 Ti, Windows) and of 6 at about 43 GB
#: (A100, Linux). The encoder frozen took 0.5 GB less at 6, which is within what
#: two cards differ by, so one figure serves a run throughout.
TRAINING_BASE_MIB = round(1.6 * 1024)
TRAINING_MIB_PER_IMAGE = round(6.9 * 1024)

#: The share of a GPU's memory a run is taken to have: a desktop and the driver
#: hold some of it, and the figures above are of two runs, not a bound.
TRAINING_HEADROOM = 0.95

#: The fields asked of ``nvidia-smi``, in the order its answer lists them.
QUERY_FIELDS = (
    "index", "uuid", "name", "memory.free", "memory.total", "pci.bus_id", "driver_version",
)

#: How long ``nvidia-smi`` may take; it answers in a fraction of a second.
QUERY_SECONDS = 10.0


@dataclass(frozen=True)
class Gpu:
    """One GPU as this process may use it.

    Attributes:
        index: Its number as torch counts it here, the N of ``cuda:N``.
        physical_index: Its number in ``nvidia-smi``, which differs from
            ``index`` only where ``CUDA_VISIBLE_DEVICES`` hides some.
        uuid: The identity CUDA accepts in ``CUDA_VISIBLE_DEVICES``.
        name: The model, as ``nvidia-smi`` names it.
        free_mib: Memory not in use by anyone, in MiB.
        total_mib: All of its memory, in MiB.
        pci_bus_id: Where it sits, which fixes the order both tools count in.
    """

    index: int
    physical_index: int
    uuid: str
    name: str
    free_mib: int
    total_mib: int
    pci_bus_id: str

    @property
    def device(self) -> str:
        """Return the torch device naming this GPU."""
        return f"cuda:{self.index}"

    def describe(self) -> str:
        """Return the device with its model and free memory, for a person choosing one."""
        return (
            f"{self.device} · {self.name} · "
            f"{self.free_mib / 1024:.1f} of {self.total_mib / 1024:.0f} GB free"
        )


@dataclass(frozen=True)
class GpuInventory:
    """The GPUs this process may use, and the driver serving them."""

    gpus: tuple[Gpu, ...] = ()
    driver_version: str = ""

    def by_device(self, device: str) -> Gpu | None:
        """Return the GPU *device* names, or None if it names none of these."""
        return next((gpu for gpu in self.gpus if gpu.device == _canonical(device)), None)


Runner = Callable[..., subprocess.CompletedProcess]


def gpu_inventory(
    run: Runner | None = None, environ: Mapping[str, str] | None = None
) -> GpuInventory:
    """Return the NVIDIA GPUs this process may use, as ``nvidia-smi`` reports them.

    Empty wherever ``nvidia-smi`` is missing, fails or answers in a form not
    understood, and wherever ``CUDA_VISIBLE_DEVICES`` names a GPU in a way not
    understood here -- a MIG slice, say -- since numbering those would be a guess.
    """
    run = subprocess.run if run is None else run
    environ = os.environ if environ is None else environ
    try:
        answered = run(
            ["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=QUERY_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("No GPUs from nvidia-smi: %s", error)
        return GpuInventory()
    if answered.returncode != 0:
        logger.debug("nvidia-smi failed: %s", answered.stderr.strip())
        return GpuInventory()

    rows = _parse(answered.stdout)
    if rows is None:
        logger.warning("nvidia-smi answered in a form not understood: %r", answered.stdout[:200])
        return GpuInventory()
    visible = _visible(rows, environ.get("CUDA_VISIBLE_DEVICES"))
    if visible is None:
        logger.info(
            "CUDA_VISIBLE_DEVICES=%r names GPUs in a form not understood here; "
            "GPUs are not described or pinned", environ.get("CUDA_VISIBLE_DEVICES"),
        )
        return GpuInventory()
    gpus = tuple(
        Gpu(index=index, physical_index=int(row[0]), uuid=row[1], name=row[2],
            free_mib=int(row[3]), total_mib=int(row[4]), pci_bus_id=row[5])
        for index, row in enumerate(visible)
    )
    return GpuInventory(gpus=gpus, driver_version=rows[0][6] if rows else "")


def least_used(inventory: GpuInventory) -> Gpu | None:
    """Return the GPU with the most free memory, the lowest-numbered of equals."""
    if not inventory.gpus:
        return None
    return max(inventory.gpus, key=lambda gpu: (gpu.free_mib, -gpu.index))


def estimated_training_mib(batch: int) -> int:
    """Return about how much GPU memory training with batches of *batch* takes, in MiB."""
    return TRAINING_BASE_MIB + TRAINING_MIB_PER_IMAGE * batch


def largest_training_batch(memory_mib: int) -> int:
    """Return the largest batch likely to train in *memory_mib* of GPU memory; 0 where none is."""
    room = memory_mib * TRAINING_HEADROOM - TRAINING_BASE_MIB
    return max(0, int(room // TRAINING_MIB_PER_IMAGE))


def training_memory_cautions(batch: int, device: str, inventory: GpuInventory) -> list[str]:
    """Return what a person should know of the memory a run needs before it starts, or nothing.

    An estimate, so never a reason to refuse a run. Where no GPU can be
    described, a GPU named is taken on trust and nothing is said of it.

    Args:
        batch: Images per training step.
        device: ``"auto"``, ``"cpu"`` or a GPU, as the run was asked.
        inventory: The GPUs this machine has, as :func:`gpu_inventory` returns them.
    """
    slow = "a step takes minutes an image there, which is days for a run of any size."
    if device == "cpu":
        return [f"Training on the CPU: {slow}"]
    gpu = least_used(inventory) if device == "auto" else inventory.by_device(device)
    if gpu is None:
        if device == "auto":
            return [f"No NVIDIA GPU was found, so this run would train on the CPU: {slow}"]
        return []
    needed = estimated_training_mib(batch)
    need = f"Batches of {batch} need about {needed / 1024:.0f} GB of GPU memory"
    card = f"{gpu.device} ({gpu.name})"
    if needed > gpu.total_mib * TRAINING_HEADROOM:
        fits = largest_training_batch(gpu.total_mib)
        instead = (
            f"The largest batch likely to fit is {fits}, under Advanced settings; a smaller "
            f"batch trains differently from the published settings."
            if fits else
            f"Even a batch of 1 needs about {estimated_training_mib(1) / 1024:.1f} GB."
        )
        return [
            f"{need}, and {card} has {gpu.total_mib / 1024:.0f} GB: the run would probably "
            f"fail as it starts. {instead}"
        ]
    if needed > gpu.free_mib:
        return [
            f"{need}, and {card} has {gpu.free_mib / 1024:.1f} GB free of "
            f"{gpu.total_mib / 1024:.0f} now: whatever else is using it would have to end first."
        ]
    return []


def number_as_nvidia_smi(environ: MutableMapping[str, str] | None = None) -> None:
    """Have CUDA number GPUs by where they sit, as ``nvidia-smi`` does.

    CUDA's own order puts the fastest first, so on a machine of mixed GPUs
    ``cuda:1`` and ``nvidia-smi``'s GPU 1 could be different cards. Takes effect
    only before CUDA is first used in the process; an order already set is kept.
    """
    environ = os.environ if environ is None else environ
    environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def pin_to(gpu: Gpu, environ: MutableMapping[str, str] | None = None) -> str:
    """Leave *gpu* the only one CUDA can see in this process, and return its device there.

    Without this, torch may open a context on GPU 0 whichever GPU it runs on,
    taking memory from whoever is using that one. Named by UUID, which means
    the same card whatever order is in force. Takes effect only before CUDA is
    first used in the process.
    """
    environ = os.environ if environ is None else environ
    environ["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    return "cuda:0"


def _parse(text: str) -> list[list[str]] | None:
    """Return ``nvidia-smi``'s rows as fields, or None if any row is malformed."""
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(line for line in text.splitlines() if line.strip())
    ]
    for row in rows:
        if len(row) != len(QUERY_FIELDS) or not all(
            part.isdigit() for part in (row[0], row[3], row[4])
        ):
            return None
    return rows


def _visible(rows: Sequence[list[str]], restriction: str | None) -> list[list[str]] | None:
    """Return the rows *restriction* leaves visible, in the order CUDA numbers them.

    None if *restriction* holds an entry this cannot place. As CUDA does, the
    list ends at the first entry naming no GPU.
    """
    if restriction is None:
        return sorted(rows, key=lambda row: int(row[0]))
    visible: list[list[str]] = []
    for entry in (part.strip() for part in restriction.split(",")):
        if entry.isdigit():
            match = next((row for row in rows if int(row[0]) == int(entry)), None)
        elif entry.startswith("GPU-"):
            match = next((row for row in rows if row[1].startswith(entry)), None)
        elif entry == "" or entry.lstrip("-").isdigit():
            break
        else:
            return None
        if match is None or match in visible:
            break
        visible.append(match)
    return visible


def _canonical(device: str) -> str:
    """Return *device* with a bare ``cuda`` read as ``cuda:0``, as torch reads it."""
    device = device.strip()
    return "cuda:0" if device == "cuda" else device
