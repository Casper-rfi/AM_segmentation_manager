"""Tests holding generation 1 to Adaptive Milling's own answers.

``data/adaptive_milling_gen1.npz`` carries synthetic images, the tensor
adaptive-milling's ``Gen1Model`` fed its model for each, and the architecture it
built, captured in its own environment by
``scripts/capture_adaptive_milling_reference.py``; the suite here runs with no
adaptive-milling installed. The forward pass beyond these is
segmentation-models-pytorch's own module on both sides.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from am_segmentation_manager.ml import environment as ml_env
from am_segmentation_manager.ml.model import GENERATION_1, build_architecture, model_input

FIXTURE = Path(__file__).parent / "data" / "adaptive_milling_gen1.npz"

needs_ml = pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")

#: How far a preprocessed value may differ from Adaptive Milling's. Not a
#: tolerance on the preprocessing but on the arithmetic: the same operations in
#: the same precision, bit-identical where captured, and allowed only the last
#: bits another CPU's vector instructions may round differently.
TOLERANCE = 1e-6


def fixture():
    """Return the fixture's metadata and its images and tensors, in case order."""
    np = pytest.importorskip("numpy")
    with np.load(FIXTURE, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        cases = [
            (name, data[f"image_{index}"], data[f"input_{index}"])
            for index, name in enumerate(meta["cases"])
        ]
    return meta, cases


@needs_ml
def test_generation_one_preprocesses_as_adaptive_milling_does():
    import numpy as np

    meta, cases = fixture()
    generation = replace(GENERATION_1, image_size=meta["image_size"])

    for name, image, expected in cases:
        ours = model_input(image, generation, rgb=True).numpy()

        assert ours.shape == expected.shape, name
        np.testing.assert_allclose(ours, expected, rtol=0, atol=TOLERANCE, err_msg=name)


@needs_ml
def test_generation_one_builds_adaptive_milling_s_architecture():
    """Every parameter and buffer by name and shape, which is what a strict load
    of one's weights into the other needs."""
    meta, _ = fixture()

    ours = {name: list(tensor.shape) for name, tensor in build_architecture(GENERATION_1).state_dict().items()}

    assert ours == meta["architecture"]


def test_the_fixture_covers_both_bit_depths_and_both_orientations():
    """A shrinking fixture would weaken the comparison without failing it."""
    meta, cases = fixture()
    size = meta["image_size"]
    shapes = [image.shape for _, image, _ in cases]

    assert {str(image.dtype) for _, image, _ in cases} == {"uint8", "uint16"}
    assert any(height > width for height, width in shapes)
    assert any(height < width for height, width in shapes)
    assert any(height == width for height, width in shapes)
    assert any(max(shape) > size for shape in shapes)
    assert any(max(shape) < size for shape in shapes)
    assert meta["source"].startswith("adaptive-milling ")
