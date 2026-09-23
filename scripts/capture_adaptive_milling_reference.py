#!/usr/bin/env python3
"""Capture Adaptive Milling's own generation-1 answers, which ``tests/test_ml_generation.py`` holds to.

**This is a standalone script, not an ``am-seg`` command**, and it runs in an
environment with ``adaptive-milling`` installed, which neither of this
package's environments is. Run it when a release of adaptive-milling changes
``adaptive_milling/models/``, or before adding a generation, and commit what it
writes.

It writes ``tests/data/adaptive_milling_gen1.npz``: small synthetic images, the
tensor ``Gen1Model`` feeds its model for each, and the architecture it builds as
a table of parameter names and shapes. The images are stored rather than
seeded, so the suite needs neither adaptive-milling nor a random generator to
agree with. ``Gen1Model`` runs at a small ``max_image_size`` so the fixture stays
small; it loads a randomly initialised model, the weights being irrelevant to
what is captured.

Nothing existing is overwritten unless ``--overwrite`` says so.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from adaptive_milling.models import MODEL_GENERATIONS_DICT

REPOSITORY = Path(__file__).resolve().parent.parent
GENERATION = "1.0"

#: Longest edge the captured tensors are resized to. Real images are larger
#: than 1536 and smaller ones are upsampled, so the cases below go both ways.
IMAGE_SIZE = 48


def images() -> dict[str, np.ndarray]:
    """Return the images captured, by what each exercises."""
    rng = np.random.default_rng(20260923)
    bright = np.zeros((64, 96), dtype=np.uint8)
    bright[20, 30] = 255
    return {
        "uint8 landscape, downsampled": rng.integers(0, 256, (64, 96), dtype=np.uint8),
        "uint16 landscape, downsampled": rng.integers(0, 65536, (64, 96), dtype=np.uint16),
        "uint16 portrait, downsampled": rng.integers(0, 65536, (96, 64), dtype=np.uint16),
        "uint8 square, downsampled": rng.integers(0, 256, (80, 80), dtype=np.uint8),
        "uint16 landscape, upsampled": rng.integers(0, 65536, (24, 36), dtype=np.uint16),
        "uint8 odd sizes, rounded": rng.integers(0, 256, (51, 77), dtype=np.uint8),
        "a single bright pixel": bright,
    }


def gen1(weights: Path):
    """Return adaptive-milling's generation 1 on *weights*, at :data:`IMAGE_SIZE`."""
    # Constructed directly: load_model passes no size, and Gen1Model takes it
    # only at construction.
    return MODEL_GENERATIONS_DICT[GENERATION](weights, device="cpu", max_image_size=IMAGE_SIZE)


def architecture(model: torch.nn.Module) -> dict[str, list[int]]:
    """Return *model*'s parameters and buffers as names and shapes."""
    return {name: list(tensor.shape) for name, tensor in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPOSITORY / "tests" / "data")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing fixture")
    args = parser.parse_args()

    path = args.out / "adaptive_milling_gen1.npz"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")

    with tempfile.TemporaryDirectory() as scratch:
        weights = Path(scratch) / "random.pth"
        # Built as Gen1Model.load builds it, so the file loads strictly.
        torch.save(
            smp.FPN(
                encoder_name="efficientnet-b6", encoder_weights=None, classes=5,
                in_channels=3, activation=None, decoder_attention_type="scse",
            ).state_dict(),
            weights,
        )
        model = gen1(weights)
        cases = images()
        arrays: dict[str, np.ndarray] = {}
        for index, image in enumerate(cases.values()):
            arrays[f"image_{index}"] = image
            arrays[f"input_{index}"] = model._preprocess(image).numpy()

    meta = {
        "source": f"adaptive-milling {version('adaptive-milling')}, generation {GENERATION}",
        "versions": {name: version(name) for name in (
            "adaptive-milling", "torch", "torchvision", "segmentation-models-pytorch", "numpy",
        )},
        "image_size": IMAGE_SIZE,
        "cases": list(cases),
        "architecture": architecture(model.model),
    }
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {path}: {len(cases)} images, {len(meta['architecture'])} tensors, "
          f"{path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
